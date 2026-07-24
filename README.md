# CPminer

Cross-platform miner for **LuckyPool** plain_proof Pearl mining.

Pool / job logistics live under `src/common/`. Each compute backend is a separate worker directory:

| Backend | Directory | Status |
|---------|-----------|--------|
| CPU | `src/cpu/` | Case 3.3 fused GEMM+XOR (contiguous 8×16) |
| CUDA | `src/cuda/` | CUTLASS / cuBLAS Pascal+ path |
| OpenCL | `src/opencl/` | Case 3.3 fused GEMM+XOR+jackpot (AMD / generic OpenCL) |

## Requirements

- MSVC (Windows) or GCC/Clang (Linux)
- **Rust toolchain** (`cargo` / `rustup`, or `conda install -c conda-forge rust`) for in-process proof build and `--verify`
- **CPU build:** AVX2-capable x86_64, OpenMP
- **CUDA build:** NVIDIA GPU + CUDA Toolkit 12.x (+ CUTLASS, fetched by `build.ps1`)
- Python 3 + `numpy` + `blake3` — **only** if using `--verify` or `scripts/plain_proof_host.py`

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
| `CP_ENABLE_CPU` | ON | Case 3.3 CPU worker |
| `CP_ENABLE_CUDA` | OFF | CUDA/CUTLASS worker |
| `CP_ENABLE_OPENCL` | OFF | OpenCL worker |
| `CP_CUDA_ARCH` | native | e.g. `61` for Pascal |

Enable multiple backends in one binary; select at runtime with `--backend cpu|cuda|opencl`.

## Build (Windows)

CPU-only (default — no CUDA Toolkit required):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
# equivalent: -Backend Cpu
```

CUDA, OpenCL, or combinations:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -CudaArch 61
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend CpuOpenCl
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Both -CudaArch 75
```

Produces `cpminer.exe` in the repo root.

## Run

```powershell
# CPU (contiguous 8×16 tiles; recommended --dev while testing)
.\cpminer.exe --backend cpu --dev --wallet prl1... --worker test --dry-run --max-nonce 1

# CUDA (CUTLASS fused GEMM+jackpot; Case 7.1 tile / milestone-major A,B)
.\cpminer.exe --backend cuda --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 `
  --wallet prl1... --worker test --devices 0

# CUDA debug: cuBLAS period GEMM + separate XOR/jackpot (BzMiner 8x16)
.\cpminer.exe --backend cuda --cublas-period --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 `
  --wallet prl1... --worker test --devices 0

# OpenCL (LuckyPool production layout)
.\cpminer.exe --backend opencl --pool stratum+tcp://pearl-eu1.luckypool.io:3360 `
  --wallet prl1... --worker test

# Offline mock: first share + zk-pow verify (no pool; use --dev for a quick run)
.\cpminer.exe --backend cuda --dev --mock
.\cpminer.exe --backend opencl --dev --mock
.\cpminer.exe --backend cpu --dev --mock
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
| `--cublas-period` | CUDA debug: cuBLAS period GEMM + separate XOR/jackpot |
| `--period-batch N` | Batch size for scan launches (default 1024; see below) |
| `--col-period-batch N` | Alias for `--period-batch` |
| `--row-period-batch N` | CUDA only: row-period batch (default 1, max 1024) |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build proof without submitting |
| `--verify` | In-process zk-pow jackpot verify before submit (needs vendored `zk-pow`) |
| `--mock` / `-mock` | Offline: fixed job id, mine until first share, verify, exit (implies dry-run+verify) |
| `--mock-diff D` | Mock pool difficulty (default 58; higher = longer before first share) |

### Scan batching (`--period-batch`)

Host syncs after each batch (cancel / progress / share check). Meaning differs by backend:

**OpenCL — 1D macro slicing**

Macros are a 2D grid (`macro_rows × macro_cols`, each 128×128), walked as a flat index `mb`. `--period-batch N` is how many **macro blocks** each kernel launch covers (`CP_MACRO_BATCH_*` in `include/cp_config.h`).

- Default: `1024` (one full macro-row at production `m=n=131072`)
- Max: `1048576` (full matrix: `1024×1024` macros)
- `--row-period-batch` is ignored on OpenCL

**CUDA — 2D period window**

Uses period tiles (`PP_ROW_PERIOD=128`, `PP_COL_PERIOD=256`):

| Flag | Role | Default | Max |
|------|------|---------|-----|
| `--row-period-batch` | Row periods per launch | 1 | 1024 |
| `--period-batch` / `--col-period-batch` | Col periods per launch | 1024 | 1024 |

A launch covers `row_batch × col_batch` periods (clipped to remaining periods). CUTLASS fused (default) needs no `C_hist`; `--cublas-period` sizes a period GEMM / `C_hist` window.

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

## Layout

```
include/          Public headers (pool, mine, worker, …)
src/common/       Pool, job loop, shared worker dispatch
src/cpu/          CPU worker + Case 3.2/3.3 GEMM+XOR
src/cuda/         CUDA kernels, CUTLASS, CUDA worker adapter
src/opencl/       OpenCL Case 3.3 fused path + kernels
rust/             cp-proof-ffi (plain_proof Merkle + bincode)
third_party/      blake3, pearl-blake3, zk-pow, plonky2, cutlass (CUDA)
scripts/          plain_proof_host.py (optional verify)
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_mine** — Job loop: A/B gen, noise fuse, worker scan, Rust proof build
- **cp_worker** — Backend selection (`cpu` / `cuda` / `opencl`)
- **cp_cpu** — Case 3.3 fused GEMM+XOR + host BLAKE3 jackpot
- **cp_gpu** — CUDA plain_proof path (under `src/cuda/`)
- **cp_opencl** — OpenCL plain_proof path (under `src/opencl/`)
- **cp_noise** — Matrix generation and pearl noise
