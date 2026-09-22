# Intel GPU oneDNN / gemmstone backend

Case 3.3 miner path for Intel XeLP (Gen12LP) and XeHPG (Core Ultra / MTL/ARL) using oneDNN's gemmstone JIT catalog, nGEN OpenCL packaging, and milestoned tile XOR — adapted from [`gemm_xor/onednn`](../../gemm_xor/onednn).

## Requirements

- Windows + MSVC (or C++17 + CMake)
- Intel GPU with OpenCL (Intel oneAPI / graphics driver)
- Git (for `prepare_onednn_deps.bat`)

## Vendor deps

```bat
cd src\onednn
prepare_onednn_deps.bat
```

Fetches oneDNN into project `third_party/onednn-src` and vendors nGEN/gemmstone under `src/onednn/third_party/` (+ `case5_patches/` overlay).

**Important:** `third_party/gemmstone` is generated locally (gitignored). The build applies `case5_patches/` on every configure. If you see errors like `case5TileXorWrap is not a member of gemmstone::GEMMProblem`, patches were not applied:

```bat
cd src\onednn
prepare_onednn_deps.bat refresh
```

Then re-run CMake configure (`build.ps1 -Backend OneDnn` or delete `build/win/cmake` and configure again).

## Build

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend OneDnn
# or combined:
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,OneDnn
```

CMake flag: `-DCP_ENABLE_ONEDNN=ON`

## Run

```bat
cppminer.exe --backend onednn --list-devices
cppminer.exe --backend onednn --wallet prl1... --worker rig01 --mock
cppminer.exe --backend onednn --onednn-layout NT --mock --mock-diff 50
```

Default panel batching: `--row-period-batch 256 --period-batch 256` (256×256 hash tiles per GEMM panel). Override with CLI flags.

**Layout names** (C always column-major N): two letters for A then B — `TN` (default), `TT`, `NT`, `NN`. Override with `--onednn-layout` or env `CASE5_GEMM_LAYOUT=TN|TT|NT|NN` (legacy three-letter forms like `TNN` still work). Also `CASE5_A_LAYOUT` / `CASE5_B_LAYOUT`: `row`/`col`.

## Design

| Item | Choice |
|------|--------|
| Kernel | oneDNN `kernel.db` `select()` + pearl 8×16 tile filter + fallbacks |
| XOR | Milestone K=128 → `xor_period = 128/unrollK`; gemmstone `case5_tile_xor` hook |
| Layout | Host A/B (Pearl) → device gemmstone layouts via GPU/CPU prep. Default **TN** (A row, B col). Also `TT`, `NT`, `NN` via `--onednn-layout` or `CASE5_GEMM_LAYOUT`. |
| Jackpot | GPU `cp_onednn_jackpot_scan` after GEMM (`tile_xor` buffer) or fused in-kernel (`--fused-jackpot`) |
| HW | Gen12LP or XeHPG only (`case5_ngen::is_supported_device`) |

## Debug environment variables

All debug hooks read the env **once** at first use (`static` init). Unset or `=0` disables them with negligible runtime cost (one branch per call site).

### tile_xor coverage (mining panels)

| Variable | Default | Effect |
|----------|---------|--------|
| `CASE5_DEBUG_TILE_XOR_ZEROS` | off | `1` — after each GEMM panel, read back `tile_xor` before jackpot and log panels where many tiles are all-zero (incomplete GEMM coverage). Non-fused path only. |
| `CASE5_DEBUG_TILE_XOR_ZERO_WARN_PCT` | `1.0` | Warn when empty-tile fraction ≥ this percent. |
| `CASE5_DEBUG_TILE_XOR_ZERO_MAX_LOG` | `16` | Cap `SUSPICIOUS` log lines per run (`0` = unlimited). |

Example (PowerShell):

```powershell
$env:CASE5_DEBUG_TILE_XOR_ZEROS = "1"
$env:CASE5_DEBUG_TILE_XOR_ZERO_WARN_PCT = "1.0"
```

### tile_xor dump (full readback)

| Variable | Default | Effect |
|----------|---------|--------|
| `CASE5_DUMP_TILE_XOR` | off | `1` — print `tile_xor` grid after readback (used by gemm_xor-style verify tooling; not wired on every CPPminer path). |
| `CASE5_DUMP_TILE_XOR_VERBOSE` | off | `1` — also list every non-zero `(ms, lr, lc, idx)`. |

### Kernel launch / catalog

| Variable | Default | Effect |
|----------|---------|--------|
| `CASE5_DEBUG_LAUNCH` | off | `1` — log walk-order args passed to the kernel (`group_count_m/n`, boustrophedon slice, etc.) on each `bind_case5_kernel_args`. |
| `CASE5_DEBUG_SELECT` | off | `1` — log each catalog candidate tried during JIT kernel build. |

### Kernel JIT (advanced)

| Variable | Default | Effect |
|----------|---------|--------|
| `CASE5_XOR_SPLIT` | `both` | XOR sub-tile split at **kernel build**: `both`/`2d`, `vertical`, `native`/`none`/`off`/`0`. See `CASE5_XOR_TILE.md` in gemm_xor. |
| `CASE5_XOR_SLM_FAST` | on | `0` — disable SLM fast path in tile_xor JIT (only if SLM staging enabled in build). |
| `CASE5_XOR_SLM_DEBUG_SPATIAL` | off | Set to any value — extra stderr from spatial-id store path inside generated kernel. |

### Related (not debug)

Layout: `CASE5_GEMM_LAYOUT`, `CASE5_A_LAYOUT`, `CASE5_B_LAYOUT` (see **Run** above). Fused jackpot test key: `CASE5_BLAKE3_KEY` (hex).


## Layout

```
src/onednn/
  cp_onednn_worker.*          miner worker (zero-B, scan loop)
  case33_gemm_onednn.*        gemmstone OpenCL GEMM + host jackpot
  case5_ngen_gemm.*           JIT build + HW dispatch
  case5_gemm_launch.*         kernel args + walk order
  case5_kernel_select.*       catalog selection + pearl tile filters
  case5_generator.cpp         gemmstone TU (patched pieces)
  case5_catalog.cpp           embedded kernel.db
  case5_patches/              gemmstone/ngen overlays (edit here only; applied by prepare script)
  prepare_onednn_deps.bat
  third_party/                ngen, gemmstone (generated — do not edit)
../../third_party/onednn-src   oneDNN checkout (after prepare)
```
