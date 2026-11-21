# Building CPPminer

How to build the miner from source. Release zips need none of this; see the [README](../README.md) to start mining.

## Build Requirements

- MSVC + **CMake** (Windows) or GCC/Clang + CMake (Linux/macOS)
- **Rust toolchain** (`cargo` / `rustup`, or `conda install -c conda-forge rust`) for in-process proof build and `--verify`
- **CPU build:** portable scalar baseline with runtime ISA dispatch: x86 AVX512-VNNI/AVX-VNNI-INT8/AVX-VNNI/AVX512-BW/AVX2/SSSE3/scalar and AArch64 I8MM/DotProd/NEON/scalar (`--simd`)
- **CUDA build:** NVIDIA GPU + CUDA Toolkit 12.x (+ CUTLASS, fetched by `build.ps1`).
- **OpenCL build:** OpenCL 1.2 runtime ICD from the GPU driver. Windows builds link vendored `third_party/opencl/lib/x64/OpenCL.lib` + Khronos headers (no CUDA/oneAPI/AMD SDK). Optional `cl_khr_integer_dot_product`, `__builtin_amdgcn_sdot4`.
- **OneDNN build:** Intel XeLP/XeHPG GPU + OpenCL + vendored oneDNN gemmstone/ngen (see `src/onednn/README.md`).
- **Hexagon build:** Android/Linux on a Snapdragon with a cDSP (tested: Snapdragon 480, Hexagon v66). The miner needs no SDK; the DSP library `libpearlx_skel.so` is built with the Hexagon SDK (see [`src/pearl/hexagon/README.md`](../src/pearl/hexagon/README.md)).
- **Intel NPU build:** Windows on a Core Ultra (NPU3720 class) with the stock NPU driver; OpenVINO and the patched `npu_compiler` built from source at pinned commits by `src/pearl/intel_npu/prepare_intel_npu_deps.ps1` (Git, Ninja, Python 3; hours on the first run). See [`src/pearl/intel_npu/README.md`](../src/pearl/intel_npu/README.md).
- **wgpu build:** Rust toolchain. Enable with `-DCP_ENABLE_WGPU=ON` / `-Backend Wgpu`. Builds two FFI crates: `rust/cp-pearl-wgpu-ffi` (Pearl) and `rust/cp-quantus-wgpu-ffi` (Quantus; build scripts fetch [`Quantus-Network/quantus-miner`](https://github.com/Quantus-Network/quantus-miner) into `third_party/quantus-miner` for `engine-gpu`).

## Build options (CMake)

```bash
cmake -S . -B build \
  -DCP_ENABLE_CPU=ON \
  -DCP_ENABLE_CUDA=OFF \
  -DCP_ENABLE_OPENCL=OFF \
  -DCP_ENABLE_ONEDNN=OFF
cmake --build build --config Release
```

| Option | Default | Meaning |
|--------|---------|---------|
| `CP_ENABLE_CPU` | ON | CPU worker |
| `CP_ENABLE_CUDA` | OFF | CUDA/CUTLASS worker |
| `CP_ENABLE_OPENCL` | OFF | OpenCL worker |
| `CP_ENABLE_ONEDNN` | OFF | Intel GPU oneDNN/gemmstone worker |
| `CP_ENABLE_WGPU` | OFF | wgpu workers: Pearl (`cp-pearl-wgpu-ffi`) and Quantus GpuEngine (`cp-quantus-wgpu-ffi`; fetches `third_party/quantus-miner`) |
| `CP_ENABLE_HEXAGON` | OFF | Pearl Snapdragon cDSP (HVX) worker; Android/Linux only |
| `CP_ENABLE_INTEL_NPU` | OFF | Pearl Intel NPU worker (OpenVINO + patched NPU compiler; `CP_INTEL_NPU_OPENVINO_DIR`, `CP_INTEL_NPU_OPENVINO_BIN_DIR`) |
| `CP_ENABLE_CUBLAS` | OFF | Link cuBLAS for `--cublas-period` debug path (needs CUDA) |
| `CP_CUDA_ARCH` | native | e.g. `61` for Pascal |

Enable multiple backends in one binary; select at runtime with `--backend`. Both algos are always compiled in; use `--algo pearl|quantus`. Runtime and compile-time **algo×backend matrix**:

| | cpu | cuda | opencl | onednn | wgpu | hexagon | intel_npu |
|--|-----|------|--------|--------|------|---------|-----------|
| pearl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| quantus | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |

## Build (Windows)

Requires MSVC Build Tools and CMake (same CMake pipeline as `build.sh` on *nix). CPU-only default:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
# equivalent: -Backend Cpu
```

CUDA, OpenCL, or combinations (comma-separated list):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,OpenCl,OneDnn
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Wgpu
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Cuda,OpenCl
# Intel NPU: the first build fetches, patches and builds OpenVINO and the NPU compiler (hours)
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,IntelNpu
# Optional debug: link cuBLAS (large DLLs; not needed for production CUTLASS path)
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -EnableCublas -CudaArch 61
```

Produces `cppminer.exe` in the repo root (plus `cp_pearl_wgpu_ffi.dll` and `cp_quantus_wgpu_ffi.dll` when wgpu is enabled, and the OpenVINO runtime, the patched NPU compiler and `kernels/intel_npu` with IntelNpu).

## Build (*nix)
```bash
./build.sh --backend cpu,opencl,onednn,cuda
./build.sh --backend cpu,wgpu
```
This script pulls third-party dependencies and runs CMake.

### Hexagon

`./build.sh --backend cpu,hexagon` builds only the miner (its Hexagon worker and FastRPC client), and `build.ps1` has no Hexagon option. Neither script builds the DSP libraries the miner loads on the cDSP, `libpearlx_skel.so` and `libworker_pool.so`: they need the Hexagon SDK on an x86 machine. See [Build in the Hexagon README](../src/pearl/hexagon/README.md#build) for those, and for cross-compiling the miner for Android.

## Vendored proof stack (`third_party/`)

| Crate | Path | Purpose |
|-------|------|---------|
| pearl-blake3 | `third_party/pearl-blake3` | Merkle proof build |
| zk-pow | `third_party/zk-pow` | Jackpot verify (`--verify`) |
| plonky2 | `third_party/plonky2` | zk-pow compile dependency |

`build.ps1` / `build.sh` both drive CMake, which runs `cargo build --release` in `rust/cp-proof-ffi/` when `cargo` is available and links the static library into the miner (`cp_proof_ffi.lib` on Windows, `libcp_proof_ffi.a` on Linux/macOS). Proofs are built and verified in-process.
