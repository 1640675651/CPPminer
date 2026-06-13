# CPminer

Cross-platform GPU miner for **LuckyPool** plain_proof Pearl mining.

This repo contains only the production plain_proof path: C noise generation, CUDA jackpot scan, and in-process Rust proof building.

## Requirements

- NVIDIA GPU + CUDA Toolkit 12.x
- MSVC (Windows) or GCC/Clang (Linux)
- **Rust toolchain** (`rustup`) for in-process proof building
- Python 3 + `numpy` + `blake3` — **only** if using `--verify` or the standalone `plain_proof_host.py build` command (optional)

## cp-proof-ffi (Rust)

Proof build and optional `--verify` use `rust/cp-proof-ffi` (pearl-blake3 Merkle + zk-pow verify). `build.ps1` runs `cargo build --release`, producing `cp_proof_ffi.lib` (linked into the miner) and `cp_proof_ffi.dll` (used by `scripts/plain_proof_host.py`).

Set `CP_PROOF_FFI` to override the shared library path.

## pearl-blake3 source

Proof assembly uses [`pearl/pearl-blake3`](../pearl/pearl-blake3) via a path dependency in `cp-proof-ffi`. A vendored copy may also exist under `third_party/pearl-blake3` for reference; keep it in sync when updating Merkle wire format:

```powershell
Remove-Item -Recurse -Force third_party\pearl-blake3
Copy-Item -Recurse pearl\pearl-blake3 third_party\pearl-blake3
```

Then rebuild (`build.ps1` runs `cargo build` in `rust/cp-proof-ffi/`).

## Build (Windows)

```powershell
conda activate pearl
powershell -ExecutionPolicy Bypass -File build.ps1
# optional: -CudaArch 61
```

Produces `cpminer.exe` in the repo root. The script clears conda compiler overrides and strips MinGW from PATH during nvcc so MSVC works inside the pearl env.

## Build (CMake)

```bash
mkdir build && cd build
cmake .. -DCP_CUDA_ARCH=61   # or native
cmake --build . --config Release
```

## Run

```powershell
.\cpminer.exe `
  --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 `
  --wallet prl1... `
  --worker test `
  --devices 0
```

### Options

| Flag | Description |
|------|-------------|
| `--pool` | `stratum+tcp://host:port` |
| `--wallet` | Wallet address (required) |
| `--worker` | Worker name (default `rig01`) |
| `--agent` | Agent string (default `cpminer/1.0`) |
| `--devices` | CUDA device list, e.g. `0` or `0,1` |
| `--dev` | Use 8192×8192 matrices for testing |
| `--contiguous-tiles` | Debug tile layout (contiguous 8×16 blocks vs production scattered) |
| `--cpu-gen` | CPU BLAKE3 matrix gen + noise (debug; default is full GPU path) |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build proof without submitting |
| `--verify` | Run Python verify before submit (off by default; needs cp_proof_ffi.dll) |
| `--python` | Python for optional `--verify` only (`CP_PYTHON` env) |
| `--host-bridge` | `plain_proof_host.py` for optional verify only |

## Layout

```
include/     Public headers (config, pool, mine, gpu, util, proof)
src/         Implementation (.cpp, .cu, .cuh)
rust/        cp-proof-ffi (C FFI → pearl-blake3)
third_party/ Vendored pearl-blake3 (from Pearl repo)
scripts/     plain_proof_host.py (optional verify only)
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_mine** — Job loop: A/B gen, noise fuse, Rust proof build, continue-after-share
- **cp_proof** — In-process plain_proof Merkle + bincode (Rust FFI → stock `pearl-blake3`)
- **cp_gpu** — CUDA plain_proof jackpot kernel
- **cp_noise** — Matrix generation and pearl noise (from zk-pow reference)
