# ESIMD XMX scan for Intel GPUs

The `onednn` backend can run the Pearl scan as one ESIMD kernel per panel
instead of gemmstone's GEMM plus a separate jackpot pass. On Arc A380 it mines
at **18.79 TMAC/s** per complete 131072² attempt (scan 19.17), against 16.66
(17.02) for the best gemmstone configuration
([oneDNN on Arc](intel_a380_onednn.md)). A pool share mined this way was
accepted.

## What the kernel does

One thread computes a 32x32 block of C and owns its four 16x16 hash tiles:

- **Operands in DPAS layout.** The miner's GPU prep writes noisy A in 8x32
  blocks (one DPAS A operand, 256 B) and the job's noisy B^T in 32 x ES VNNI
  blocks (one B operand), so every operand is one block load. The proof uses
  the signal A from the same prep, so the noisy layout is free to choose.
  This needs the work-group noise kernel (`ocl_noisy_matrix_rowmajor_wg`).
- **dpasw on Xe-HPG.** Consecutive threads (a fused EU pair) share rows and
  each loads half of every A block; without `dpasw` the A380 scan was 15.2
  TMAC/s, with it 18.3 (8x4 groups) and 19.8 (16x2 groups).
- **Everything after the GEMM stays in registers.** Every 128 k the running C
  of each tile is XOR-reduced and folded into a 16-word state; after K = 4096
  the state is hashed with keyed BLAKE3 and compared with the target. The
  first hit claims the found flag with `cmpxchg` and writes its coordinates.
  There is no tile_xor buffer and no second pass.
- 16x2-thread work groups (512x64 of C), L1 prefetch two k-steps ahead,
  default linear group order (band swizzles were 17–66% slower).

Execution size 16 with plain `dpas` (Xe2: Arc B-series, Lunar Lake; Xe-HPC)
is compiled in and selected from the device's minimum sub-group size, but has
not been run on such hardware. Devices without XMX keep gemmstone.

## Using it

| `CP_INTEL_GEMM` | Behaviour |
|---|---|
| `auto` (default) | ESIMD when `kernels/libcp_esimd.so` (Linux) or `kernels/cp_esimd.dll` (Windows) loads and the GPU has XMX; otherwise gemmstone |
| `esimd` | Require ESIMD; fail initialization otherwise |
| `gemmstone` | Never load the library |

ESIMD needs layout TN (the default) and the non-fused jackpot (the default).
The backend banner shows `ESIMD XMX scan (dpasw es=8, 512x64 work-group tile,
in-kernel jackpot)` when it is active. If the library loads but its kernel
cannot be built on the installed driver, the backend keeps gemmstone.

### Building on Linux

```sh
source /opt/intel/oneapi/setvars.sh        # icpx from the oneAPI DPC++ compiler
cd src/onednn && ./prepare_onednn_deps.sh && cd ../..
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_OPENCL=ON \
      -DCP_ENABLE_ONEDNN=ON -DCP_ENABLE_ESIMD=ON
cmake --build build -j4                     # also builds build/kernels/libcp_esimd.so
```

`cppminer` itself is still built with the regular compiler; only the library
uses `icpx`. The library records an RPATH into the oneAPI tree (after
`$ORIGIN`) and links `libumf`/`libhwloc` directly (the SYCL runtime's adapters
need them), so the miner runs without `setvars.sh` on the machine where oneAPI
is installed. For machines without oneAPI,
`scripts/package_esimd_runtime.sh build/kernels` copies the SYCL runtime it
needs (libsycl, the Unified Runtime loader and OpenCL adapter, libumf, libhwloc
and the Intel math/runtime libraries, ~70 MB) and Intel's license files next
to it. This was tested in containers with only the Intel compute-runtime
installed (23.43 from Ubuntu 24.04 and 26.31).

### Building on Windows

`build.ps1 -Backend Cpu,OpenCl,OneDnn -EnableEsimd` builds
`kernels/cp_esimd.dll` with oneAPI `icx` (from a oneAPI installation, or from
a conda DPC++ prefix in `CP_ONEAPI_ROOT`, which
`scripts/setup_windows_oneapi.ps1 -Prefix <dir>` creates).
`scripts/package_esimd_runtime.ps1` copies the DLL imports and the
dynamically loaded UR loader and OpenCL adapter next to the library. The
miner registers the kernels directory with `SetDllDirectory` for these later
runtime loads; the initial altered-search-path flag alone is insufficient.

### Standalone prototype

`src/esimd/esimd_gemm_bench.cpp` is the measured prototype (USM, no miner):
`verify` compares every milestone tile word and every tile digest with a CPU
reference, `bench M N` times an MxNx4096 scan. Compile-time knobs: `TM`/`TN`
(thread tile), `WGM`/`WGN` (work group), `PF` (prefetch), `DPASW`, `ES`.

## Measurements (Arc A380)

| Configuration (16384x16384x4096 prototype) | TMAC/s |
|---|---:|
| dpas, 32x32 tile, 8x4 group | 15.30 |
| dpas + prefetch 2 | 15.63 |
| dpasw, 8x4 group | 18.34 |
| **dpasw, 16x2 group, prefetch 2** | **19.84** |
| dpasw, 32x2 / 64x2 / 4x8 groups | 13.75 / 6.76 / 12.29 |
| dpasw, 64x16 or 16x64 thread tiles | 17.15 or less |

A register-only DPAS loop on this card reaches 26.0 TMAC/s at ~1.6 GHz, so
the kernel runs at about 76% of that ceiling.

In the miner at 131072² (Ubuntu 24.04 container, oneAPI DPC++ 2026.1.1,
compute-runtime 26.31, host kernel 6.8 with i915): 18.786 TMAC/s per full
attempt, 19.167 scan, prep 0.075 s. On this newer runtime gemmstone scans at
only 11.99 TMAC/s (17.02 on runtime 23.43), so ESIMD and the new runtime
belong together.

## Validation

- Prototype: 0 differing words over 16 384 milestone tile XORs and 0 over 512
  BLAKE3 digests against the CPU reference, for dpas and dpasw.
- Miner: 30 mock shares built and verified through the proof verifier
  (8 of them without the oneAPI environment); one live pool share accepted.
- Not established: Xe2/Xe-HPC hardware, Intel GPU execution on Windows,
  long-term acceptance rate.
