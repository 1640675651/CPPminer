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

On Linux run `sh prepare_onednn_deps.sh` (same behaviour, `refresh` argument too); CMake does not run it automatically there. Arc (XeHPG) results: [docs/intel_a380_onednn.md](../../docs/intel_a380_onednn.md).

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

Default panel batching: `--row-period-batch 256 --batch-size 256` (256×256 hash tiles per GEMM panel). Systolic (XMX, Arc) kernels default to 2048×2048, or 4096×4096 on cards with at least 12 GiB when the kernel writes the folded message (two 1 GiB `tile_xor` buffers). The onednn limit is 4096 per dimension. Override with CLI flags. `CASE5_PROFILE_SCAN=1` prints GPU busy/idle and GEMM/jackpot time per attempt.

**Layout names** (C always column-major N): two letters for A then B — `TN` (default), `TT`, `NT`, `NN`. Override with `--onednn-layout` or env `CASE5_GEMM_LAYOUT=TN|TT|NT|NN` (legacy three-letter forms like `TNN` still work). Also `CASE5_A_LAYOUT` / `CASE5_B_LAYOUT`: `row`/`col`.

## Design

| Item | Choice |
|------|--------|
| Kernel | XeHPG with XMX: built-in measured-best strategy first (`sS32x2 sB32 aB wg 16x2 cb4x2 ks32 k128 xaf ...`, 16×64 per thread; `CASE5_PREFER=0` skips it), then oneDNN `kernel.db` `select()` (A380-tuned 16×32 entry moved to the front) + fallbacks. Operand alignment comes from the leading dimensions before catalog matching. |
| XOR | Milestone K=128: `xor_period = 128/unrollK`, or every 128 k inside the panel when unrollK > 128; gemmstone `case5_tile_xor` hook |
| XMX fold | For one milestone per k-unroll and a 1×4/1×8 sub-tile grid (the preferred strategy): the fold is split at dpas chain boundaries (upper rows after pass 0 of the milestone's last step, lower rows after pass 0 of the next step), SIMD16 column pairs with 3-input `bfn` xor into 16-lane accumulators, one packed reduction, and per-pass dpas chains only in the first/last step of a milestone (`xaf` elsewhere). `CASE5_XOR_INTERLEAVE=0` restores the end-of-step fold. |
| tile_xor | **Tile-major**: the milestone words of spatial tile `sid` are contiguous at `sid * num_ms` (folded 16-word messages likewise at `sid * 16`). On XMX the kernel buffers 16 milestones per sub-tile and folds each block into the 16-word jackpot message (`rotl 13`, xor), writing it once per tile at `sid * 16`; the jackpot kernel then only hashes (`CASE5_XOR_FOLD16=0` writes the raw milestones instead). Elsewhere one dword per milestone. |
| Layout | Host A/B (Pearl) → device gemmstone layouts via GPU/CPU prep. Default **TN** (A row, B col). Also `TT`, `NT`, `NN` via `--onednn-layout` or `CASE5_GEMM_LAYOUT`. |
| Jackpot | GPU `cp_onednn_jackpot_scan` after GEMM (`tile_xor` buffer, tile-major) or fused in-kernel (`--fused-jackpot`). On XeHPG with the built-in strategy the fused path is **cooperative**: the folded 16-word message stays in registers, every thread parks its tiles' messages, spatial ids and coordinates in the k loop's (then dead) SLM after a barrier, and a quarter of the work group's threads hash 16 tiles per SIMD16 instruction (keyed BLAKE3, 7 rounds) and judge them against the bound in the kernel; only a found flag and one winner's coordinates reach memory. The jackpot's key, bound and panel offsets travel in the found buffer (bound to the kernel's `tile_xor` argument) because the 16x64 strategy has no spare registers for them. Other strategies keep the older per-tile SIMD1 fused path. |
| HW | Gen12LP or XeHPG only (`case5_ngen::is_supported_device`) |

Arc A770, mock scan at 131072², sustained, per attempt (GEMM + folded message + device jackpot):
**100.5 TMAC/s** with the built-in strategy and 4096×4096 panels (was 95 with raw milestones and
1024×1024 panels); 75 TMAC/s with `CASE5_PREFER=0` (the A380-tuned catalog-333 kernel, older
measurement); `--fused-jackpot` **102.3 TMAC/s** with the cooperative in-kernel jackpot (0.688 s per attempt vs 0.707 separate; the older per-tile fused path was 66-80). UHD 770 (catalog-30): 1.35 TMAC/s. Mock shares verify on both GPUs in both
jackpot modes. Known issue: `--onednn-layout NT` on XeHPG selects an 8×4 catalog kernel (8×4 hash
tiles) and its mock proof fails verification; the kernel's tile XOR itself verifies in the gemm_xor
harness, so the mismatch is in how an 8×4 hash tile is proven (also seen on the A380 in PR #36).

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
| `CASE5_JACKPOT_COOP` | on | `0` — `--fused-jackpot` uses the older per-tile SIMD1 fused path (wrap-GRF fold + Case 5.6 BLAKE3) instead of the cooperative SIMD16 epilogue. |
| `CASE5_COOP_AUDIT` | off | `1` — the cooperative kernel also writes every tile's digest and message into the found buffer and the host recomputes every digest after each panel (`[coop audit] ... mismatch=0 unwritten=0`); use small panels (`--row-period-batch 512 --batch-size 512`). |
| `CASE5_COOP_SKIP` | `0` | Timing only (no shares): bit 1 skips the whole cooperative epilogue, bit 2 the hashing, bit 4 the stores. |
| `CASE5_SLM_PAD` | unset | Timing only: bytes added to (or, negative, removed from) the kernel's SLM allocation. The A770 loses about a quarter of its GEMM throughput above 16 KB. |
| `CASE5_DEBUG_RA` | off | `1` — print the number of free GRFs at the main generator phases (`2` also a free-register map before the k loop); for "Insufficient registers" diagnosis. |

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
