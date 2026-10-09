# Pearl Hexagon backend (`--backend hexagon`)

Pearl mining on the compute DSP (cDSP) of Snapdragon phones. The GEMM and the
milestone XOR run on the DSP's HVX vector units. The host CPU prepares the
matrices and runs the BLAKE3 jackpot. Tested on a Snapdragon 480 (SM4350,
Hexagon v66, 2 HVX contexts).

| Device | Backend | Hashrate (m = n = 32768) |
|---|---|---|
| Snapdragon 480 cDSP | hexagon | ~395 GMAC/s while scanning, ~380 GMAC/s overall |
| Snapdragon 480 CPU (2x A76 + 6x A55) | cpu (NEON DotProd) | ~130 GMAC/s |

The overall rate includes about 0.44 s of host work per 11.6 s attempt, mostly
hashing the signal A for its noise seed. Measured from `adb shell`, which limits
the process to the six A55 cores.

Where the scan rate goes, against the kernel's ~413 GMAC/s peak:

| Cost | Share |
|---|---|
| Copying a 128 KB slice of B into VTCM per 128-column tile and K block, amortized over the launch's 1024 row tiles | ~1.9% |
| FastRPC call per launch (~2.7 ms of 170 ms) and packing each row panel (`set_a`, ~6.8 ms per 8 launches) | ~2% |
| The rest of the gap between DSP calls, host-side | ~1% |

With `OMP_NUM_THREADS=7` (one more OpenMP thread than the six visible cores)
the scan reaches ~399 GMAC/s on this phone. The cause isn't understood, so it is
not the default.

## Hash tile: 4 x 64

The DSP kernel keeps a 4 x 128 register tile of C in 16 HVX accumulators.
zk-pow accepts hash tiles of at most h*w = 256 elements, so 4 x 128 cannot be a
hash tile. The kernel instead XORs each 4 x 64 half separately (gemm_xor
Case 1.1):

- `mining_config`: rows 0..3, columns 0..63 (`PEARL_CONTIGUOUS_4x64_CONFIG`).
- Proof tile layout: `CP_TILE_LAYOUT_CONTIGUOUS_4x64` = 6, in `cp_proof.h` and
  in `rust/cp-proof-ffi`.
- Jackpot scale factor: 4 * 64 * (k/r) * 128.

The Rust tests build and zk-pow-verify a 4 x 64 proof. A share mined on the
phone was also proven and verified with `cp_proof_verify`. Whether a pool
accepts this new configuration has not been tested.

## How it works

The matrix is scanned in **launches** of 128 x 128 macro blocks, like the GPU
backends' macro panels: by default 32 x 32 macros, so 4096 rows x 4096 columns
(65,536 hash tiles) per DSP launch. A row of launches is a **row panel**.

| Step | Where | When |
|---|---|---|
| Noisy B^T (signal B^T = 0) | host, OpenMP | once per job |
| Pack B into DSP memory (`pearlx set_b`) | cDSP | once per job |
| Sparse random signal A, `a_noise_seed` | host | every attempt |
| Noisy A for a row panel | host, OpenMP | per panel, one slice per launch of the panel before |
| Pack the row panel (`pearlx set_a`) | cDSP | per panel |
| GEMM + milestone XOR of the panel against one block of columns (`pearlx gemm_xor`) | cDSP | per launch |
| BLAKE3 jackpot over the launch's tile XORs | host, OpenMP | per launch |

A dedicated thread issues the launches back to back. The mining thread follows,
checking each finished launch and building the next panel's noisy rows one
launch ahead. The DSP only waits when the record buffer it is about to reuse
hasn't been checked yet, or a new panel's rows aren't built. In practice it is
busy about 97.5% of the scan. Every hit is recomputed on the host from the
noisy rows, with the B columns derived again from the job seed, before it
becomes a share. A wrong result from the DSP can therefore never be submitted.

Proofs use the host signal A with an all-zero B^T, like the CPU backend.

On the DSP, `set_a` packs the panel's rows once into the kernel's chunk layout.
In each launch, for each 128-column tile of B and each K block of 1024, the DSP
copies a 128 KB slice of B into VTCM and streams the panel's row tiles through
the kernel. The kernel is the HVX `vrmpyz`
(Z-buffer) GEMM from `gemm_xor/hexagon`. After every 128 values of k it XORs
each half's 8 accumulators lane-wise, packs both halves into one vector with a
`vdeal`, and reduces the 8 milestones to words between tiles. The XOR costs
about 4.5% over the plain GEMM.

## Files

| Path | Contents |
|---|---|
| `cp_pearl_hexagon_worker.cpp` | Worker: job and attempt prep, the launch pipeline, jackpot, hit recompute |
| `cp_pearlx_client.c` | FastRPC client: loads `libcdsprpc.so` at run time and marshals the `pearlx` calls by hand. No Hexagon SDK headers are needed |
| `dsp/inc/pearlx.idl` | DSP interface: `set_b`, `set_a`, `gemm_xor`, `info` |
| `dsp/src/pearlx_imp.c` | DSP side: packing, VTCM, threads over both HVX contexts |
| `dsp/src/pearlx.S`, `dsp/src/pearlx_xor.inc` | HVX kernel (Case 1.1 fold and reduction) |
| `dsp/src/pearlx_test.c` | Android self-test: `pearlx` against a CPU reference, plus timing |
| `dsp/CMakeLists.txt`, `dsp/build.ps1` | Hexagon SDK build of the DSP library and the self-test |

## Build

There are two parts:

1. **The DSP library** `libpearlx_skel.so`. It needs the Hexagon SDK on an x86
   host (tested with SDK 5.5.7.0 on Windows):

   ```powershell
   cd src\pearl\hexagon\dsp
   .\build.ps1                                   # hexagon_Release_toolv87_v66\ship\libpearlx_skel.so
   .\build.ps1 -NoBuild -Test "-n 32768 -r 4096 -w 4096 -g 2"   # push + self-test on the phone
   ```

   Ship `libpearlx_skel.so` and the SDK's `libworker_pool.so` from
   `hexagon_Release_toolv87_v66\ship`.

2. **The miner.** It needs no SDK.
   - On the phone in Termux: `./build.sh --backend cpu,hexagon`.
   - Cross-compiled with the Android NDK, for example the one in the Hexagon
     SDK. Push `libomp.so` from the NDK alongside the miner.

     ```powershell
     cmake -S . -B build/android-hexagon -G Ninja `
       -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake `
       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static `
       -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_CPU=ON -DCP_ENABLE_HEXAGON=ON
     cmake --build build/android-hexagon
     ```

     A cross-build without an Android Rust target falls back to the proof stub,
     which can't submit shares. Build in Termux, which has Rust, to get proofs.

## Run

The DSP library loads in an unsigned protection domain, so root is not needed.
FastRPC finds the skel through `ADSP_LIBRARY_PATH`:

```sh
export ADSP_LIBRARY_PATH=/path/to/dir/with/libpearlx_skel.so   # and libworker_pool.so
./cppminer --backend hexagon --pool stratum+tcp://HOST:PORT --wallet prl1... --worker phone
./cppminer --backend hexagon --mock --mock-diff 40              # offline: first share + verify
```

| Option | Hexagon meaning |
|---|---|
| `--m N --n N` | Matrix size in units of 1024. The default is 32 (32768) instead of 128: each matrix is 128 MiB on the host, and B is kept again in DSP memory. 131072 needs 512 MiB per matrix |
| `--row-period-batch N` | 128-row macros per DSP launch (default 32 = 4096 rows). Rows set how far each VTCM slice of B is amortized. Per row panel: 2048 rows ~770 GOPS, 4096 ~792 |
| `--batch-size N` / `--col-period-batch N` | 128-column macros per DSP launch (default 32 = 4096 columns). Columns barely matter: a full-width launch is only 0.3% faster than 4096 columns |

Testing was done from `adb shell` (`/data/local/tmp`). Running from Termux has
not been tested yet. If `cp_pearlx_open` fails, check that the app may open
`/dev/fastrpc-cdsp` and that `ADSP_LIBRARY_PATH` is set.

Only Hexagon v66 has been tested. `vrmpyz` (the Z buffer) is a v66 HVX
instruction, so other DSP versions need checking before use.
