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
| Kernel | Three layers (`case5_kernel_select.cpp`): (1) per-device overrides keyed by PCI device id (empty placeholder), (2) per-family measured kernels, (3) oneDNN `kernel.db` `select()` in its own order + fallbacks; a kernel that does not build falls through to the next candidate, and `CASE5_PREFER=0` skips (1) and (2). Family kernels: **XeHPG** (A770) `sB32 sB32 aB wg 16x2 cb4x2 ks32 k128 xaf ...` 16×64 with packed A and B (row-major-A variant `sS32x2 ...` above K = 4096); **Gen12LP** (UHD 770) `ab16x2 ab8x2 ab int bm8192` 16×32 (kernel.db entry 24, gemmstone's packed int8 kernel), needs packed A and B (98.0% of the measured dp4a peak in gemm_xor pure GEMM; per 131072² attempt against catalog-30 plain: 51.96 → 47.91 s separate jackpot (register-folded 16-word messages, `CASE5_SEPARATE_WRAP`; 48.50 s with per-milestone stores), 53.20 → 48.33 s fused (SIMD-N judge, below; the generic tile-XOR fold emits its hash tiles' chains round-robin and uses 3-input `bfn` on XeHP and later, `CASE5_XOR_BFN=0` for 2-input `xor`); the hand strategy `sb16 sb16 sb l4 nse k16 cab1 wg 4x4` it replaced: 48.74 / 49.80 s); **Xe-LPG** (Meteor Lake iGPU: `HW::XeHPG` to gemmstone but without XMX, so the family table is keyed by (HW, XMX)) `aB16 aB16 aB wg 4x4 cab4 ks16 nse bo sr bk0 grf256 l4` 32×32 (kernel.db entry 329, label `xelpg-family (catalog-329)`) with A and B packed in 32-row / 32-column panels, crosspack 4: 7.4 TOPS pure GEMM on a 64-EU part at 1.9 GHz in gemm_xor (about 95% of the dp4a peak), ahead of the best plain NT kernel; verified on the A770, which builds the same code for this strategy (miner mock shares in both jackpot modes, prep byte checks), not yet in the miner on an Xe-LPG device. gemmstone's selector does not choose these: the miner asks it for plain TN, where it matches one Gen12LP entry (30); the catalog's packed int8 entries (Gen12LP 24, as fast as the Gen12LP family kernel in gemm_xor; XeHPG 287/289, whose tiled packed A the prep does not write) are only matched for packed problems. Operand alignment comes from the leading dimensions before catalog matching. |
| XOR | Milestone K=128: `xor_period = 128/unrollK`, or every 128 k inside the panel when unrollK > 128; gemmstone `case5_tile_xor` hook |
| XMX fold | For one milestone per k-unroll and a 1×4/1×8 sub-tile grid (the preferred strategy): the fold is split at dpas chain boundaries (upper rows after pass 0 of the milestone's last step, lower rows after pass 0 of the next step), SIMD16 column pairs with 3-input `bfn` xor into 16-lane accumulators, one packed reduction, and per-pass dpas chains only in the first/last step of a milestone (`xaf` elsewhere). `CASE5_XOR_INTERLEAVE=0` restores the end-of-step fold. |
| tile_xor | **Tile-major**: the milestone words of spatial tile `sid` are contiguous at `sid * num_ms` (folded 16-word messages likewise at `sid * 16`). On XMX the kernel buffers 16 milestones per sub-tile and folds each block into the 16-word jackpot message (`rotl 13`, xor), writing it once per tile at `sid * 16`; the jackpot kernel then only hashes (`CASE5_XOR_FOLD16=0` writes the raw milestones instead). Elsewhere one dword per milestone. |
| Layout | Host A/B (Pearl) → device gemmstone layouts via GPU/CPU prep. Default **TN** (A row, B col). Also `TT`, `NT`, `NN` via `--onednn-layout` or `CASE5_GEMM_LAYOUT`. With a family kernel A is written in gemmstone's **packed** layout (panels of the tile height, 16 rows on both families, 4 consecutive k per dword: the dpas/dp4a operand layout) so the kernel block-loads it with fewer address updates (`CASE5_A_PACKED=0` keeps row-major A). B is written once per job in gemmstone's packed-rows layout: panels of the tile width with the family's crosspack (A770 64 columns × 32 k, UHD 770 32 × 4; `CASE5_B_PACKED=0` keeps column-major B). On XeHPG above K = 4096 only B is packed (single-buffered packed-A loads lose there). The prep writes packed B directly (`ocl_noisy_matrix_b_packed`, no column-major scratch) for panel widths up to 64 and crosspacks dividing 64, and the host upload takes any panel width and crosspack. Packed A: 16-row panels use `ocl_noisy_matrix_a_packed16`, other heights up to 64 rows `ocl_noisy_matrix_a_packed` (Xe-LPG: 32); crosspack 4 only. |
| Jackpot | GPU `cp_onednn_jackpot_scan` after GEMM (`tile_xor` buffer, tile-major) or fused in-kernel (`--fused-jackpot`). On XeHPG with the built-in strategy the fused path hashes **in the GEMM kernel**: the folded 16-word messages stay in registers and, after the k loop, every thread hashes its own N sub-tiles (N = 4 for the 16x64 strategy) as N SIMD lanes with keyed BLAKE3, after a register transpose; it judges them against the bound in the kernel and stores only a found flag and one winner's coordinates. No SLM, no barriers. The jackpot's key, bound and panel offsets travel in the found buffer (bound to the kernel's `tile_xor` argument) because the 16x64 strategy has no spare registers for them. Other strategies keep the older per-tile SIMD1 fused path. |
| HW | Gen12LP or XeHPG only (`case5_ngen::is_supported_device`) |

Arc A770, mock scan at 131072², sustained, per attempt (GEMM + folded message + device jackpot):
**100.5 TMAC/s** with the built-in strategy and 4096×4096 panels (was 95 with raw milestones and
1024×1024 panels); 67 TMAC/s with `CASE5_PREFER=0` (gemmstone's own top pick, catalog-354, 32x16; separate
jackpot); `--fused-jackpot` **108.5 TMAC/s** end-to-end with packed A and the in-kernel jackpot (0.648 s per attempt, scan 109.5 TMAC/s; 0.694 s with row-major A, 0.664 s with packed A and the separate jackpot; the older per-tile fused path was 66-80). UHD 770 (catalog-30): 1.35 TMAC/s. Mock shares verify on both GPUs in both
jackpot modes. Proofs exist only for contiguous 4×8, 4×64, 8×8, 8×16 and 16×16 hash tiles
(`cp_contiguous_tile_layout_for_shape`), so kernel selection skips candidates whose hash tile is
anything else (`case5_ngen::set_hash_tile_filter`; `CASE5_DEBUG_SELECT=1` shows "has no proof layout").
Before this guard, `--onednn-layout NT` on XeHPG picked catalog-323 (8×4 tiles), whose shares could not
be proven (the proof side silently assumed 8×16); NT now runs catalog-330 (16×16 tiles, about 1.0 s per
attempt instead of 17.5) and NN catalog-292 (1.5 s instead of 3.1), and both verify. Non-TN layouts
remain diagnostics: only TN gets the family kernels.

## Matrix preparation (per attempt)

See [docs/prep.md](../../docs/prep.md) for the full write-up, including PR #27.

Per nonce the GPU builds the noisy A (131072 x 4096, 512 MB). The random signal A_sig is generated
in full once (`ocl_gen_random_matrix16`) and hashed in full once; every later nonce replaces one random
row (the values the full generator would give that row for the nonce's seed, written from the host) and
updates the BLAKE3 Merkle commitment incrementally: the row's 256-chunk group is rehashed by one work
group of the same kernel and the stored subroots are reduced again (`CP_OCL_PREP_ROWPOKE=0` restores the
full regeneration; a new job key triggers one full rehash since the hash is keyed). Then 128 uniform
noise bytes per row (`ocl_uniform_rows_b3`: one keyed
single-block BLAKE3 per 32-byte block, words in registers) and the noisy matrix
(`ocl_noisy_matrix_rowmajor_sg`: one 32-lane sub-group per row, 4 consecutive k per lane so an
instruction touches 128 contiguous bytes, 8 units in flight, the row's noise bytes in local memory).
A770 per attempt: row update 0.2 ms (was 4.3 for the full generation), commitment 0.7 ms (was 2.8),
uniform rows 0.3 ms (was 8.7 with the byte-array hash), noise 3.0 ms (was 6.1: the 16-byte-per-lane
access pattern alone cost 5.5 ms as a plain copy); prep 4 ms (was 22) of a 0.694 s attempt, so
end-to-end is 101.4 TMAC/s against about 102 scan-only. `CP_OCL_PREP_CHECK=1` also recomputes the full
commitment every nonce and compares roots. Both new kernels are checked byte for byte against
the simple per-row kernels with `CP_OCL_PREP_CHECK=1`; `CP_OCL_PREP_TIMING=1` prints the split;
`CP_OCL_PREP_UROWS_B3=0` / `CP_OCL_PREP_SG=0` restore the previous kernels;
`CP_OCL_PREP_DEFS="-DNSG_KPL=16 -DNSG_UNROLL=1 -DNSG_SLM_TABLE=0 -DNSG_DIAG=1"` passes build options
to the prep kernels for experiments (the `NSG_*` defines select the noise kernel's decomposition;
`NSG_DIAG` variants produce wrong output and are for timing only). The OpenCL backend shares
the signal update, commitment and uniform-rows steps but has its own prepack noise kernel (A770 via
OpenCL: prep 11 -> 4 ms of a 1.06 s attempt at 65536^2). The sub-group noise kernel is compiled only
where `cl_intel_subgroups` exists; other devices (the GTX 1070's OpenCL) keep the work-group kernel.

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
| `CASE5_DEBUG_RANK` | off | `1` — print gemmstone's own catalog ranking for the problem (top 15, model score = predicted time, lower ranks first), before the miner reorders candidates. |

### Host/device transfers (scan stalls)

On the Arc A770 (Windows driver), `clEnqueueWriteBuffer` occasionally sleeps about 70 s (40-100 s seen)
waiting on a GPU fence in `D3DKMTWaitForSynchronizationObjectFromCpu` (stack captured during a stall),
while the CPU and GPU sit idle; then it returns normally with correct data. The scan shows `0 MAC/s` or
a frozen percentage with a falling rate until it returns, which can coincide with the next pool job.
The fused scan makes about 25 small transfers per attempt (per-panel arguments into the found buffer,
the found-flag read, prep seeds and the row update), so it hit this about once per 325 attempts. Every
variant that still called the transfer API stalled sooner or later: read-only or read-write buffers,
staging through an `ALLOC_HOST_PTR` buffer, non-blocking writes followed by `clFinish`.

`OpenClContext::write_buffer` / `read_buffer` therefore move transfers of up to 64 KB on Intel discrete
GPUs without a transfer call: the bytes go through a 64 KB host USM block
(`cl_intel_unified_shared_memory`) and a one-line copy kernel, completed with `clFinish`. Result: 0
stalls in 2000 fused attempts (about 6 expected before), prep 6.0 to 5.3 ms, scan unchanged. A kernel
must not write a `CL_MEM_READ_ONLY` buffer or read a `CL_MEM_WRITE_ONLY` one, so the small buffers the
host writes or reads (`a_key_buf_`, `bound_buf_`, the prep's `d_job_key_` and `d_noise_seed_`, the
OpenCL backend's hit coordinates) are now `CL_MEM_READ_WRITE`. Other read-only or write-only buffers
and larger transfers (such as the witness read after a hit) still use the transfer calls.

| Variable | Default | Effect |
|----------|---------|--------|
| `CP_OCL_XFER_STAGE` | on for Intel discrete GPUs | `0` — transfer calls for everything (the stalling path); `1` — the kernel path on any device with USM. Shared with the OpenCL backend. |
| `CP_OCL_XFER_WATCH` | off | `<ms>` — log every transfer slower than `<ms>` with its path, size, offset, the buffer's size and flags (`[ocl] SLOW write 32 B @0 (buffer 32 B, flags 0x4): host 70.2 s`), and GPU timestamps when `CP_OCL_QUEUE_PROFILING=1`. |
| `CP_ONEDNN_PANEL_WATCH` | off | `<ms>` — log GEMM panels slower than `<ms>`, split into host enqueue time (including the per-panel argument writes) and queue wait, with the kernel's GPU timestamps when `CP_OCL_QUEUE_PROFILING=1`. |

### Kernel JIT (advanced)

| Variable | Default | Effect |
|----------|---------|--------|
| `CASE5_XOR_SPLIT` | `both` | XOR sub-tile split at **kernel build**: `both`/`2d`, `vertical`, `native`/`none`/`off`/`0`. See `CASE5_XOR_TILE.md` in gemm_xor. |
| `CASE5_XOR_SLM_FAST` | on | `0` — disable SLM fast path in tile_xor JIT (only if SLM staging enabled in build). |
| `CASE5_XOR_SLM_DEBUG_SPATIAL` | off | Set to any value — extra stderr from spatial-id store path inside generated kernel. |
| `CASE5_JACKPOT_COOP` | on | `0` — `--fused-jackpot` uses the wrap-GRF fused path (as on Gen12LP) instead of the XMX in-kernel SIMD-N epilogue. |
| `CASE5_SEPARATE_WRAP` | on | Separate jackpot on kernels without the XMX interleaved fold (Gen12LP, Xe-LPG): the milestones are folded into the 16-word jackpot message in registers (rotl 13, xor into word ms % 16, as in the fused path) and written once per tile, so the jackpot kernel only hashes (scan tag `igemm+wrapGRF+msg16+GPUjackpot`). `0` — one global store per hash tile and milestone in the k loop (the old path); the register fold won in gemm_xor once a thread holds two or more hash tiles (UHD 770 entry 24, Xe-LPG). |
| `CASE5_JACKPOT_SIMDN` | on | Wrap-GRF fused path (Gen12LP, e.g. UHD 770): each thread hashes its N logical tiles (any M x N sub-tile grid with 2..8 tiles: UHD 770 1x2, Xe-LPG 2x2) in one N-lane BLAKE3 compression with a branchless lane compare, and only winning lanes write the found buffer. `0` — the older per-tile SIMD1 judge (one compression and one found-buffer atomic per tile). UHD 770 fused attempt 49.45 → 48.44 s; first shares identical to the separate jackpot on fixed seeds. |
| `CASE5_COOP_AUDIT` | off | `1` — the cooperative kernel also writes every tile's digest and message into the found buffer and the host recomputes every digest after each panel (`[coop audit] ... mismatch=0 unwritten=0`); use small panels (`--row-period-batch 512 --batch-size 512`). |
| `CASE5_A_PACKED` | on (family kernels) | `0` — row-major A (A770: gathered A loads, about 7% slower; UHD 770: the family kernel is skipped, catalog entry 30 runs); `1` — force packed A onto every candidate, panel = unroll M, with `CASE5_A_CP=<crosspack>` (default 4) and `CASE5_A_TILE=r,c` (timing experiments: the prep only writes 16-row panels with crosspack 4). |
| `CASE5_B_PACKED` | on (family kernels) | `0` — column-major B (A770 about 4% slower; UHD 770: catalog entry 30 runs); `1` — force packed B onto every candidate (panel = unroll N) with `CASE5_B_CP=<crosspack>` (default 32) and `CASE5_B_TILE=r,c`. |
| `CASE5_COOP_OWN` | on | `0` — SLM variant of the fused epilogue: every thread parks its messages in the k loop's dead SLM, one barrier, and 16 threads hash 8 tiles each as SIMD8 (`CASE5_COOP_LANES=16`: 8 threads, SIMD16). Same speed as the default; kept for comparison. The default (own tiles) cannot be combined with `CASE5_COOP_SKIP=2`. |
| `CASE5_COOP_ADD3` | on | `0` — the cooperative BLAKE3 computes `a + b + m` with two `add`s instead of one `add3` (A/B timing; no measurable difference). |
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
