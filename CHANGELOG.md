# Changelog

## v0.5 (tentative)
- Experimental OneDNN backend for intel GPU
- Fix OpenCL dot product extension on intel GPU
- Shrink opencl macro size to 64x64 in 4x8 tile mode, prevent to many work item per work group 
- Introduce quantus algorithm
- Quantus wgpu backend via quantus-miner GpuEngine FFI
- Quantus OpenCL Poseidon2 worker under src/qpow/opencl
- Pearl wgpu backend
- Reduce host memory usage (CPU, CUDA, OpenCL, OneDNN, Wgpu)
- MinGW and MSYS2 support (thanks to @danifest751)
- Configurable matrix size: `--m` / `--n` in units of 1024 (default 128x128)
- Pearl CPU: scan macro blocks scheduled one at a time (a chunk of 4096 left an 8k×8k scan on a single thread: 118 -> ~660 GMAC/s on a Zen4 8-core); `--threads N` now also sets the Pearl CPU pool, default one pinned thread per physical core, `--smt` for one per logical CPU; `OMP_PLACES` / `OMP_PROC_BIND` disable the built-in pinning

### Pearl wgpu
- vec4<u32> A/B panel loads in the GEMM shader
- Disable naga loop bounding on the GEMM shader (~36x faster, GTX 1070 86 GMAC/s -> ~4 TMAC/s)
- Rewrite prepack_a (one 256-WI group per 8 rows, noise hashed once, packed u32 stores); per-attempt prep 1.3s -> ~0.13s on GTX 1070
- GEMM accumulator tile as named vec4<i32> locals instead of array<i32, 64>; fixes Intel iGPU (UHD 770 35 -> ~540 GMAC/s)
- Single-buffered LDS GEMM (one 32 KiB k-block panel per barrier pair), `--wgpu-lds on|off`, default on for discrete GPUs (GTX 1070 ~4.0 -> ~5.0 TMAC/s)
- `--wgpu-tile 4x4|4x8|8x8|8x16[/64x64|/128x128]` and `--wgpu-macro`, same tiles/hash tiles as OpenCL; shader tile code generated at engine init (default stays 8x8/128)
- Bind a_pre / b_pre / a_sig in windows under `max_storage_buffer_binding_size` (512 MiB buffers vs 256 MiB on Mali); scan batches split only where a macro column exceeds the limit

### Host memory reduction
- GPU backend: per-hit D2H 512 MiB -> ~0.3-0.8 MiB, no 2x512 MiB host A/B buffers. See proof.md (CUDA, OpenCL, OneDNN, Wgpu)
- Wgpu: readback staging buffer 512 MiB -> 256 KiB (host-visible; was sized to the whole A matrix)
- CPU, OpenCL `--cpu-gen` and oneDNN host fallback: drop the 512 MiB all-zero host B^T buffer; proofs use zero-matrix Merkle sub-roots cached per job
- CPU `--prepack fused` is now the default (~1.5 GiB steady vs ~2.5 GiB for separate)
- Host-matrix proofs hash A/B^T in place: transient per-share peak ~1.5 GiB -> < 1 MiB (no flatten/pad/MerkleTree copies of the 512 MiB matrix)

## v0.4
- ARM CPU + NEON support.
- AVX-VNNI support.
- Termux OpenCL support.
- Dynamic openmp schedule for CPUs with big and small cores.
- Lightweight random matrix generation on CPU, only sparsely fill matrix elements (one random write per column).
- Zero-B on cuda worker (reduce VRAM usage, 2GiB -> 1.5 GiB).


## v0.3
- Switch to cert V3 to align with the salted seed hardfork. Old cert V2 shares are no longer accepted by the network.
- Separate OpenCL kernel compilation process with error handling. Improve compatibility for driver/compiler that crashes on specific compilation failure.
- Skip matrix prep kernel if --cpu-gen is on. This fixes unsupported intrinsics on some drivers like beignet.
- Add OpenCL **4×8** hash tile (`--ocl-tile 4x8`) to further reduce register pressure on small GPUs; make **4×8** the default OpenCL tile (AMD still auto-selects **8×16**).
- OpenCL issue mode `--ocl-issue auto|broadcast|packed` (default **auto** = DP4A detection then broadcast fallback).
- OpenCL int8 promotion selection `--ocl-cpm-type float|int`: broadcast B scalar in float (default) or int32. 

OpenCL mode note:
Broadcast+float issue may improve performance since some GPUs are weak in int but strong in float. For example, on UHD 630, broadcast+float yields 115GH/s, packed+int yields 100GH/s, and broadcast+int yields 90GH/s.

## v0.2.1
- Refactor CUTLASS GEMM main loop to reduce XOR overhead. This buys back the lost performance in the last version due to more frequent XOR boundary. 8.0TH -> 9.1TH on a GTX 1070.

## v0.2
- Use rank=128 to avoid pearl rank penalty. This adds a bit more overhead compared to rank=256, may hurt hashrate slightly.
- SSE fallback for non-AVX CPU.
- Add CPU thread affinity and prioritizing physical cores than SMT threads.
- Use 8x8 tile as default for the opencl worker, lowering register pressure for integrated GPUs. Use 8x16 tile if detects AMD GPU.

## v0.1

First public release of **CPPminer** — a cross-platform Pearl (LuckyPool plain_proof) miner in C++.

### Package contents

| Binary | Backends | Notes |
|--------|----------|--------|
| `cppminer_all.exe` | CUDA + OpenCL + CPU | Pick at runtime with `--backend` |
| `cppminer_opencl.exe` | OpenCL + CPU | AMD / OpenCL GPUs |
| `cppminer_cpu.exe` | CPU only | AVX2 x86_64 |

Also ship next to the CUDA-capable binary:

- `cudart64_12.dll` (CUDA runtime)

CUDA Toolkit is **not** required to run.

### Hardware support

- **Legacy NVIDIA** — Pascal cards via the CUDA CUTLASS fused kernel (e.g. GTX 10-series).
- **AMD GPUs** — OpenCL worker.
- **CPU** — AVX2 OpenMP worker.

### Basic usage

```powershell
# CPU
.\cppminer_cpu.exe --backend cpu --pool stratum+tcp://HOST:PORT --wallet prl1... --worker rig01

# NVIDIA (CUDA) — use cppminer_all.exe
.\cppminer_all.exe --backend cuda --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 --wallet prl1... --worker rig01

# AMD / OpenCL
.\cppminer_opencl.exe --backend opencl --pool stratum+tcp://pearl-eu1.luckypool.io:3360 --wallet prl1... --worker rig01
```
