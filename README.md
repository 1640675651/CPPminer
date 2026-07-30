# CPPminer

Cross-Platform Pearl miner written in C++.

Pool / job logistics live under `src/common/`. Each compute backend is a separate worker directory:

| Backend | Directory | Status |
|---------|-----------|--------|
| CPU | `src/cpu/` | Fused GEMM+XOR (contiguous 8×16) |
| CUDA | `src/cuda/` | Pascal CUTLASS Fused GEMM+XOR+jackpot |
| OpenCL | `src/opencl/` | Fused GEMM+XOR+jackpot (AMD / generic OpenCL) |

## Requirements

- MSVC (Windows) or GCC/Clang (Linux)
- **Rust toolchain** (`cargo` / `rustup`, or `conda install -c conda-forge rust`) for in-process proof build and `--verify`
- **CPU build:** x86_64 with OpenMP; AVX2 preferred, SSSE3/scalar fallback at runtime (`--simd`)
- **CUDA build:** NVIDIA GPU + CUDA Toolkit 12.x (+ CUTLASS, fetched by `build.ps1`).
- **OpenCL build:** OpenCL 1.2 runtime ICD from the GPU driver. Windows builds link vendored `third_party/opencl/lib/x64/OpenCL.lib` + Khronos headers (no CUDA/oneAPI/AMD SDK). Optional `cl_khr_integer_dot_product`, `__builtin_amdgcn_sdot4`.

## Build options (CMake)

```bash
cmake -S . -B build \
  -DCP_ENABLE_CPU=ON \
  -DCP_ENABLE_CUDA=OFF \
  -DCP_ENABLE_OPENCL=OFF
cmake --build build --config Release
```

| Option | Default | Meaning |
|--------|---------|---------|
| `CP_ENABLE_CPU` | ON | CPU worker |
| `CP_ENABLE_CUDA` | OFF | CUDA/CUTLASS worker |
| `CP_ENABLE_OPENCL` | OFF | OpenCL worker |
| `CP_ENABLE_CUBLAS` | OFF | Link cuBLAS for `--cublas-period` debug path (needs CUDA) |
| `CP_CUDA_ARCH` | native | e.g. `61` for Pascal |

Enable multiple backends in one binary; select at runtime with `--backend cpu|cuda|opencl`.

## Build (Windows)

CPU-only (default — no CUDA Toolkit required):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
# equivalent: -Backend Cpu
```

CUDA, OpenCL, or combinations (comma-separated list):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,OpenCl
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Cuda,OpenCl
# Optional debug: link cuBLAS (large DLLs; not needed for production CUTLASS path)
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -EnableCublas -CudaArch 61
```

Produces `cppminer.exe` in the repo root.

## Run

```powershell
# CPU
.\cppminer.exe --backend cpu --wallet prl1... --worker worker_name

# CUDA (CUTLASS fused GEMM+jackpot)
.\cppminer.exe --backend cuda --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 `
  --wallet prl1... --worker worker_name --devices 0

# CUDA debug: cuBLAS period GEMM + separate XOR/jackpot (requires -EnableCublas build)
.\cppminer.exe --backend cuda --cublas-period --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 `
  --wallet prl1... --worker worker_name --devices 0

# OpenCL (LuckyPool production layout)
.\cppminer.exe --backend opencl --pool stratum+tcp://pearl-eu1.luckypool.io:3360 `
  --wallet prl1... --worker worker_name

# Offline mock: first share + zk-pow verify (no pool)
.\cppminer.exe --backend cuda --mock
.\cppminer.exe --backend opencl --mock
.\cppminer.exe --backend cpu --mock
```

### Options

| Flag | Description |
|------|-------------|
| `--backend` | `cpu` / `cuda` / `opencl` (must be compiled in) |
| `--pool` | `stratum+tcp://host:port` |
| `--wallet` | Wallet address (required unless `--mock`) |
| `--worker` | Worker name (default `rig01`) |
| `--devices` | CUDA device list |
| `--dev` | Use 8192×8192 matrices for testing |
| `--cpu-gen` | Host matrix prep on GPU paths (OpenCL ~1 GiB VRAM; CUDA debug) |
| `--cutlass-fused` | CUDA: fused CUTLASS GEMM + jackpot (**default**) |
| `--cublas-period` | CUDA debug: cuBLAS period GEMM (only if built with `CP_ENABLE_CUBLAS`) |
| `--no-cutlass-fused` | CUDA debug: non-CUTLASS period path |
| `--period-batch N` | Batch size for scan launches (default 1024; see below) |
| `--col-period-batch N` | Alias for `--period-batch` |
| `--row-period-batch N` | CUDA only: row-period batch (default 1, max 1024) |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build proof without submitting |
| `--verify` | In-process zk-pow jackpot verify before submit (needs vendored `zk-pow`) |
| `--mock` / `-mock` | Offline: fixed job id, mine until first share, verify, exit (implies dry-run+verify) |
| `--mock-diff D` | Mock pool difficulty (default 58; higher = longer before first share) |
| `--prepack MODE` | CPU: `separate` (default), `reuse`, or `fused` matrix prepack |
| `--simd ISA` | CPU: `auto` (default), `avx2`, `sse`, `scalar` |

### Scan batching (`--period-batch`)

Host syncs after each batch (cancel / progress / share check). Meaning differs by backend:

**OpenCL — 1D macro slicing**

Macros are a 2D grid (`macro_rows × macro_cols`, each 128×128), walked as a flat index `mb`. `--period-batch N` is how many **macro blocks** each kernel launch covers (`CP_MACRO_BATCH_*` in `include/cp_config.h`).

- Default: `1024` (one full macro-row at production `m=n=131072`)
- Max: `1048576` (full matrix: `1024×1024` macros)
- `--row-period-batch` is ignored on OpenCL

**CUDA — 2D launch window**

Default **CUTLASS fused** path tiles the matrix in **128×128 CTAs** (`CP_CUTLASS_CTA_M/N`). `--row-period-batch` / `--period-batch` count how many of those CTAs to launch per step (clipped to remaining).

`--cublas-period` (debug) uses BzMiner **periods** instead: `PP_ROW_PERIOD=128` × `PP_COL_PERIOD=256` (not the same as the 128×128 CTA).

| Flag | Role | Default | Max |
|------|------|---------|-----|
| `--row-period-batch` | Row CTAs (or row periods) per launch | 1 | 1024 |
| `--period-batch` / `--col-period-batch` | Col CTAs (or col periods) per launch | 1024 | 1024 |

CUTLASS fused needs no C buffer; `--cublas-period` sizes a period GEMM / C window.

## Performance

Hashrate on matrix size `m=n=131072`, `k=4096`. Rates are MAC/s (`docs/hashrate_calculation.md`). Figures are indicative; your results will vary with clocks, drivers, and batch settings.

### AMD GPU (OpenCL)

| Device | Hashrate |
|--------|----------|
| Radeon pro 5500m  (gfx1012) | ~5.0 TH/s |

### NVIDIA GPU (CUDA)

| Device | Hashrate |
|--------|----------|
| GTX 1070 | ~8.5 TH/s |

### CPU

| Device | Hashrate | Bzminer v25.0.1b2 baseline |
|--------|----------| ---------------------------|
| Core i9 9980hk @ 2.4GHz AVX2 | ~440 GH/s | ~300GH/s |
| Core i5 12490F @ 4.0GHz AVX2 | ~640 GH/s | ~400GH/s |


## Vendored proof stack (`third_party/`)

`cp-proof-ffi` is self-contained under this repo — no external `pearl/` checkout:

| Crate | Path | Purpose |
|-------|------|---------|
| pearl-blake3 | `third_party/pearl-blake3` | Merkle proof build |
| zk-pow | `third_party/zk-pow` | Jackpot verify (`--verify`) |
| plonky2 | `third_party/plonky2` | zk-pow compile dependency |

`build.ps1` runs `cargo build --release` in `rust/cp-proof-ffi/`, producing `cp_proof_ffi.lib` (linked into the miner) and `cp_proof_ffi.dll` (for `scripts/plain_proof_host.py`).

If any are missing, copy from the Pearl repo:

```powershell
Copy-Item -Recurse pearl\pearl-blake3 third_party\pearl-blake3
Copy-Item -Recurse pearl\zk-pow       third_party\zk-pow
Copy-Item -Recurse pearl\plonky2      third_party\plonky2
```

Set `CP_PROOF_FFI` to override the shared library path for Python verify.

## Dev Fee

A transparent **1%** developer fee uses a **tile-debt** schedule on the **same pool**:

- `T` = hash tiles in one full matrix scan for the active backend/layout/dims; CPU/OpenCL 8×16, CUDA periodic/CUTLASS may differ).
- User scans: `debt += tiles` (including cancelled partial scans).
- When `debt >= 100*T`, reconnect and mine under the developer wallet.
- Fee scans: `debt -= 100 * tiles`; leave fee mode when `debt < 100*T`.
- Seed `debt = 50*T` so the first fee cycle is centered in the period.

## Layout

```
include/          Public headers (pool, mine, worker, fee, …)
src/common/       Pool, job loop, fee scheduler, shared worker dispatch
src/cpu/          CPU worker + fused GEMM+XOR
src/cuda/         CUDA kernels, CUTLASS, CUDA worker adapter
src/opencl/       OpenCL fused path + kernels
rust/             cp-proof-ffi (plain_proof Merkle + bincode)
third_party/      blake3, pearl-blake3, zk-pow, plonky2, opencl (+headers), cutlass (CUDA)
scripts/          plain_proof_host.py (optional verify)
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_fee** — Same-pool 1% tile-debt developer fee (reconnect + authorize)
- **cp_mine** — Job loop: A/B gen, noise fuse, worker scan, Rust proof build
- **cp_worker** — Backend selection (`cpu` / `cuda` / `opencl`)
- **cp_cpu** — Fused GEMM+XOR + host BLAKE3 jackpot
- **cp_gpu** — CUDA plain_proof path (under `src/cuda/`)
- **cp_opencl** — OpenCL plain_proof path (under `src/opencl/`)
- **cp_noise** — Matrix generation and pearl noise
