# CPminer

Cross-platform miner for **LuckyPool** plain_proof Pearl mining.

Pool / job logistics live under `src/common/`. Each compute backend is a separate worker directory:

| Backend | Directory | Status |
|---------|-----------|--------|
| CPU | `src/cpu/` | Case 3.3 fused GEMM+XOR (contiguous 8×16) |
| CUDA | `src/cuda/` | CUTLASS / cuBLAS Pascal+ path |
| OpenCL | — | Build flag reserved; not implemented yet |

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
| `CP_ENABLE_OPENCL` | OFF | Reserved (errors if ON) |
| `CP_CUDA_ARCH` | native | e.g. `61` for Pascal |

Enable multiple backends in one binary; select at runtime with `--backend cpu|cuda`.

## Build (Windows)

CPU-only (default — no CUDA Toolkit required):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
# equivalent: -Backend Cpu
```

CUDA or both:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -CudaArch 61
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Both -CudaArch 75
```

Produces `cpminer.exe` in the repo root.

## Run

```powershell
# CPU (contiguous 8×16 tiles; recommended --dev while testing)
.\cpminer.exe --backend cpu --dev --wallet prl1... --worker test --dry-run --max-nonce 1

# CUDA (LuckyPool production layout: BzMiner scattered 8×16)
.\cpminer.exe --backend cuda --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 `
  --wallet prl1... --worker test --devices 0
```

### Options

| Flag | Description |
|------|-------------|
| `--backend` | `cpu` / `cuda` / `opencl` (must be compiled in) |
| `--pool` | `stratum+tcp://host:port` |
| `--wallet` | Wallet address (required) |
| `--worker` | Worker name (default `rig01`) |
| `--devices` | CUDA device list |
| `--dev` | Use 8192×8192 matrices for testing |
| `--cpu-gen` | Host matrix gen on CUDA path (debug) |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build proof without submitting |
| `--verify` | In-process zk-pow jackpot verify before submit (needs vendored `zk-pow`) |

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
rust/             cp-proof-ffi (plain_proof Merkle + bincode)
third_party/      blake3, pearl-blake3, zk-pow, plonky2, cutlass (CUDA)
scripts/          plain_proof_host.py (optional verify)
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_mine** — Job loop: A/B gen, noise fuse, worker scan, Rust proof build
- **cp_worker** — Backend selection (`cpu` / `cuda` / …)
- **cp_cpu** — Case 3.3 fused GEMM+XOR + host BLAKE3 jackpot
- **cp_gpu** — CUDA plain_proof path (under `src/cuda/`)
- **cp_noise** — Matrix generation and pearl noise
