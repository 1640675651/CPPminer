# Changelog

## v0.3 (tentative)
- Align with the cert V3 salted seed hardfork. Old cert v2 shares are not accepted anymore (CPU, OpenCL done, CUDA TODO).
- Local `--verify` and jackpot target scaling use pearl **rank-penalized** bound (at `r=128` identical to the old `h×w×k` scale).
- Separate OpenCL kernel compilation process with error handling. Improve compatibility for driver/compiler that crashes at compilation failure.
- Skip matrix prep kernel if --cpu-gen is on. This fixes unsupported intrinsics on some driver like beignet.
- Add OpenCL **4×8** hash tile (`--ocl-tile 4x8`) to further reduce register pressure on small GPUs.
- OpenCL issue mode `--ocl-issue auto|broadcast|packed` (default **auto** = DP4A detection then broadcast fallback).
- OpenCL int8 promotion selection `--ocl-cpm-type float|int`: broadcast B scalar in float (default) or int32. 

Broadcast+float issue may improve performance since some GPUs are weak in int but strong in float. For example, on UHD 630, broadcast+float yields 115GH/s, packed+int yields 100GH/s, and broadcast+int yields 90GH/s.
- Lightweight random matrix generation, only fill some matrix elements (TODO)
- Zero-B on cuda worker (TODO)
- Smaller matrix low-memory mode (TODO)
- ARM CPU support (TODO)

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
