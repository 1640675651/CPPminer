# Architecture

Source layout, backends and modules, and the developer-fee schedule.

## Backends

| Backend | Directory | Status |
|---------|-----------|--------|
| CPU | `src/cpu/` | Pearl: SIMD GEMM+XOR |
| CUDA | `src/cuda/` | Pearl: Pascal dp4a CUTLASS fused GEMM+XOR+jackpot |
| OpenCL | `src/opencl/` | Pearl: fused GEMM+XOR+jackpot (AMD / generic OpenCL) |
| OneDNN | `src/onednn/` | Pearl: Intel GPU gemmstone IGEMM + tile XOR + GPU jackpot |
| wgpu | `src/pearl/wgpu/` + `rust/cp-pearl-wgpu-ffi` | Pearl: WGSL fused GEMM+XOR+jackpot (Vulkan / DX12 / Metal), optional LDS staging |
| Hexagon | `src/pearl/hexagon/` (+ DSP library in `dsp/`) | Pearl: Snapdragon cDSP HVX fused GEMM+XOR+jackpot ([README](../src/pearl/hexagon/README.md)) |
| Intel NPU | `src/pearl/intel_npu/` | Pearl: Core Ultra NPU via OpenVINO + patched NPU compiler, int8 GEMM on the DPU + tile XOR on the SHAVE cores ([README](../src/pearl/intel_npu/README.md)) |
| Quantus CPU | `src/qpow/cpu/` | Poseidon2 midstate search (scalar + AVX2 4-wide) |
| Quantus CUDA | `src/qpow/cuda/` | Poseidon2 kernel on the nonce line, self-tested per device |
| Quantus wgpu | `src/qpow/wgpu/` + `rust/cp-quantus-wgpu-ffi` | GpuEngine FFI |
| Quantus OpenCL | `src/qpow/opencl/` | Poseidon2 kernel on the nonce line (`qpow/nonce_line.hpp`); product/reduction variants probed and self-tested per device ([docs](quantus_opencl.md)) |

## Layout

```
include/          Public headers (pool, mine, worker, fee, …)
src/common/       Pool, job loop, fee scheduler, shared worker dispatch
src/cpu/          CPU worker + fused GEMM+XOR
src/cuda/         CUDA kernels, CUTLASS, CUDA worker adapter
src/opencl/       OpenCL fused path + kernels
src/pearl/hexagon/ Snapdragon cDSP worker; dsp/ = HVX kernel + FastRPC library (Hexagon SDK)
src/pearl/intel_npu/ Intel NPU worker: OpenVINO launch graph, prebuilt SHAVE kernel, npu_compiler patch + prepare script
rust/             cp-proof-ffi (plain_proof Merkle + bincode)
third_party/      blake3, pearl-blake3, zk-pow, plonky2, opencl (+headers), cutlass (CUDA)
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_fee** — 1% tile-debt developer fee on fixed fee pools, your pool as fallback (reconnect + authorize)
- **cp_mine** — Job loop: A/B gen, noise fuse, worker scan, Rust proof build
- **cp_worker** — Backend selection (`cpu` / `cuda` / `opencl` / `onednn`)
- **cp_cpu** — Fused GEMM+XOR + host BLAKE3 jackpot
- **cp_gpu** — CUDA plain_proof path (under `src/cuda/`)
- **cp_opencl** — OpenCL plain_proof path (under `src/opencl/`)
- **cp_onednn** — Intel GPU oneDNN/gemmstone path (under `src/onednn/`)
- **cp_pearl_hexagon** — Snapdragon cDSP HVX GEMM+XOR via FastRPC + host BLAKE3 jackpot (under `src/pearl/hexagon/`)
- **cp_noise** — Matrix generation and pearl noise

## Dev fee schedule

A transparent **1%** developer fee uses a **tile-debt** schedule:

- `T` = hash tiles in one full matrix scan for the active backend/layout/dims; CPU/OpenCL use the selected contiguous tile shape; CUDA periodic/CUTLASS may differ).
- User scans: `debt += tiles` (including cancelled partial scans).
- When `debt >= 100*T`, reconnect and mine under the developer wallet.
- Fee scans: `debt -= 100 * tiles`; leave fee mode when `debt < 100*T`.
- Seed `debt = 50*T` so the first fee cycle is centered in the period.
