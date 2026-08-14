# Changelog

## v0.3 (tentative)
- Put OpenCL kernel compilation into a subprocess with error handling. Improve compatibility for ill-behaving drivers.
- Skip matrix prep kernel if --cpu-gen is on (fixes unsupported intrinsics on beignet driver)
- Fold milestone XORs into `msg[16]` during GEMM; drop redundant `ms_xor[32]`. On Beignet (Haswell GT1) reported kernel private memory drops from 1152 B/WI to **384 B/WI** on the default 8×8 tile (8×16 stays 1152 B/WI).
- Add OpenCL **4×8** hash tile (`--ocl-tile 4x8`): half-height register tile for tiny iGPUs; proof FFI `tile_layout=4`. On Beignet Haswell GT1, production scan **~28 GMAC/s** (mock share verified) vs ~9.9 GMAC/s at 8×8.
- OpenCL `--ocl-lds [KWG]`: CLBlast-style `__local` A/B staging (default KWG=16) so work-items reuse packed panels instead of each reloading from global.
- OpenCL scalar GEMM inner loop matches CLBlast GEMMK=0 (`cpm += aval * bscalar`). ~1.1 → ~9.9 GMAC/s on Beignet Haswell GT1 (`--dev` 8×8, no LDS). See `docs/opencl_issue_shape.md`.
- OpenCL `--ocl-cpm-int`: same issue with int8 lanes and int32 acc (default remains float `mad`).
- Lightweight random matrix generation (TODO)

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
