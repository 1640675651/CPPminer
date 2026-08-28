# Changelog

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
