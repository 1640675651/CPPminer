# CPminer

Cross-platform GPU miner for **LuckyPool** plain_proof Pearl mining.

This repo contains only the production plain_proof path: C noise generation, CUDA jackpot scan, and Python proof build/verify/submit.

## Requirements

- NVIDIA GPU + CUDA Toolkit 12.x
- MSVC (Windows) or GCC/Clang (Linux)
- Python 3 with dependencies for `scripts/plain_proof_host.py`

## Build (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
# optional: -CudaArch 61
```

Produces `cpminer.exe` in the repo root.

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
| `--contiguous-tiles` | Debug tile layout (pool uses BzMiner scattered) |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build/verify proofs without submitting |
| `--no-verify` | Skip Python verify before submit |
| `--python` | Python executable (`CP_PYTHON` or `PEARL_PYTHON` env) |
| `--host-bridge` | Path to `plain_proof_host.py` |

## Layout

```
include/     Public headers (config, pool, mine, gpu, util)
src/         Implementation (.cpp, .cu, .cuh)
scripts/     plain_proof_host.py
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_mine** — Job loop: A/B gen, noise fuse, proof build, continue-after-share
- **cp_gpu** — CUDA plain_proof jackpot kernel
- **cp_noise** — Matrix generation and pearl noise (from zk-pow reference)
