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

**Layout names** (C always column-major N): two letters for A then B — `TN` (default), `TT`, `NT`, `NN`. Override with `--onednn-layout` or env `CASE5_GEMM_LAYOUT=TN|TT|NT|NN` (legacy three-letter forms like `TNN` still work). Also `CASE5_A_LAYOUT` / `CASE5_B_LAYOUT`: `row`/`col`.

## Design

| Item | Choice |
|------|--------|
| Kernel | oneDNN `kernel.db` `select()` + pearl 8×16 tile filter + fallbacks |
| XOR | Milestone K=128 → `xor_period = 128/unrollK`; gemmstone `case5_tile_xor` hook |
| Layout | Host A/B (Pearl) → device gemmstone layouts via GPU/CPU prep. Default **TN** (A row, B col). Also `TT`, `NT`, `NN` via `--onednn-layout` or `CASE5_GEMM_LAYOUT`. |
| Jackpot | Host `cp_jackpot` scan after GEMM (`tile_xor` readback) or fused in-kernel (`--fused-jackpot`) |
| HW | Gen12LP or XeHPG only (`case5_ngen::is_supported_device`) |

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
