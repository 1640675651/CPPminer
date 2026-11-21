# All command-line options

Every option, including backend tuning and debugging flags. The common ones are in the [README](../README.md#common-options).

## Options

Options that take a value accept `--opt VALUE` or `--opt=VALUE`. An unknown option (including one for a backend that is not compiled in), a missing value, or a malformed, out-of-range or too long value stops the miner with status 1 before it starts.

| Flag | Description |
|------|-------------|
| `--algo` | `pearl` (default) or `quantus` (`qpow` / `qpow-poseidon2` aliases). Quantus: `cpu` / `cuda` / `wgpu` / `opencl`; Pearl: `cpu` / `cuda` / `opencl` / `onednn` / `wgpu` / `hexagon` / `intel_npu`. `--pool` required for Quantus unless `--mock` |
| `--backend` | `cpu` / `cuda` / `opencl` / `onednn` / `wgpu` / `hexagon` / `intel_npu` (alias `npu`) (must be compiled in; must be valid for `--algo`).
| `--pool` | `stratum+tcp://host:port` (required for `--algo quantus` unless `--mock`) |
| `--wallet` | Wallet address (required unless `--mock`) |
| `--worker` | Worker name (default `rig01`). On a `kryptex` pool host the wallet is sent as `WALLET.worker` as well |
| `--pool-pass STR` | `mining.authorize` password (default `x`). Kryptex reads a custom share difficulty from it: `--pool-pass d=2097152` (their default; `d = hashrate_H/s * target_share_seconds / 4294967296`) |
| `--threads N` | CPU backend OpenMP threads, Pearl and Quantus. Default: all CPUs the process may use (respects `taskset` / cpusets); `OMP_NUM_THREADS` overrides the default. Pearl threads are pinned physical cores first, then SMT siblings, so N up to the core count gives one thread per core. Set `OMP_PLACES` / `OMP_PROC_BIND` to let the OpenMP runtime place threads instead, `CP_CPU_AFFINITY=0` to disable pinning |
| `--devices` | Mine on which device id (`--list-devices`). Multi-GPU pearl mining is not supported yet, you can use one process per GPU. |
| `--list-devices` | List devices for the selected backend and exit |
| `--m N`, `--n N` | Matrix rows / columns in units of 1024 (default 128 = 131072; each ≤ 256, `m*n` ≤ 128×128) |
| `--cpu-gen` | Host matrix prep on GPU paths (OpenCL ~1 GiB VRAM; CUDA debug) |
| `--api-port N` | Serve a read-only HTTP stats API on `127.0.0.1:N`: `/summary` (JSON: hashrate over 10 s / 60 s / 15 min, shares, devices with PCI addresses) and `/hiveos`. See [docs/api.md](api.md) |
| `--api-bind ADDR` | API listen address (default `127.0.0.1`; `0.0.0.0` for all interfaces) |
| `--cutlass-fused` | CUDA: fused CUTLASS GEMM + jackpot (**default**) |
| `--cublas-period` | CUDA debug: cuBLAS period GEMM (only if built with `CP_ENABLE_CUBLAS`) |
| `--no-cutlass-fused` | CUDA debug: non-CUTLASS period path |
| `--batch-size N` | Launch batch. Pearl: col/macro panel size (default 1024; backend may remap). Quantus: **nonces per launch** (wgpu default 1000000; CUDA/OpenCL default automatic, about 100 ms per launch). Aliases: `--period-batch`, `--col-period-batch` |
| `--period-batch N` | Alias for `--batch-size` |
| `--col-period-batch N` | Alias for `--batch-size` |
| `--row-period-batch N` | CUDA: row-period batch (default 32, max 1024); oneDNN, Hexagon, Intel NPU: see below |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build proof without submitting; saves `pp_header.bin` / `pp_proof.b64` next to the executable (a normal run writes nothing to disk) |
| `--verify` | In-process zk-pow jackpot verify before submit (needs vendored `zk-pow`) |
| `--mock` / `-mock` | Offline: fixed job, mine until first share, verify, exit (implies dry-run). Pearl: zk-pow verify; Quantus: Poseidon2 `hash < target` |
| `--mock-diff D` | Mock difficulty (higher = longer). Defaults: Pearl **44** (unscaled target 2^(256-D)); Quantus **1000000** (`U512::MAX / D`). `--mock-diff` overrides for either. |
| `--cert-version N` | Force certificate / noise-seed version: `1`/`2` = legacy, `3` = salted (V3). Default **3**. Without this flag, pool `mining.notify` `cert_version` wins when present (1–3); otherwise default 3 |
| `--prepack MODE` | CPU: `fused` (default), `reuse`, or `separate` matrix prepack |
| `--simd ISA` | CPU: `auto` (default), `hybrid`, `avx512vnni`, `avxvnni-int8` (signed `vpdpbssd`, no u8s8 offset work; 64-bit MSVC/Linux builds, preferred by `auto` over `avxvnni`), `avxvnni`, `avx512` (base AVX512F+BW, `avx512bw` alias), `avx2`, `ssse3` (`sse` alias), `i8mm`, `dotprod`, `neon`, `scalar`. Quantus: `hybrid` runs one scalar and one AVX2 Poseidon2 worker per physical core (SMT siblings); `auto` is the best available mode (currently `hybrid`; may select a wider kernel such as AVX-512 in the future); `avx2` forces AVX2 on every thread; any other value runs scalar. Pearl: `hybrid` is the same as `auto` |
| `--simd-test` | Compare every available CPU SIMD kernel against scalar and exit (Quantus: AVX2 Poseidon2 field ops and hash parity vs scalar) |
| `--prepack-test` | Check CPU fused and reuse prepack against separate at m=n=8192 (prepacked bytes + full tile XOR) and exit |

## OpenCL options

| Flag | Description |
|------|-------------|
| `--ocl-platform P` | Restrict device enumeration to platform index `P` |
| `--ocl-tile MxN[/MmMm]` | Register tile: `4x8` (default), `4x4`, `8x8`, or `8x16` (auto on AMD discrete GPUs). Optional `/64x64` or `/128x128` sets the macro (same as `--ocl-macro`) |
| `--ocl-macro MxN` | Macro block: `64x64` or `128x128` (default `128x128`, independent of tile) |
| `--ocl-issue MODE` | GEMM issue: `auto` (default), `broadcast` (B-scalar `mad`), or `packed` (per-C `dot4`) |
| `--ocl-dot MODE` | Dot backend: `auto` (default; AMD sudot→sdot4→KHR→scalar), `sudot`, `sdot4`, `khr`, `force-khr`, `asm`, or `off` |
| `--ocl-cpm-type T` | Scalar (no int8 dot) MAC: `auto` (default; `mad24` on AMD GCN gfx6-8, else `float`), `float`, `int`, or `mad24` |
| `--ocl-lds on/off` | Stage A/B panels in `__local` (default `off`) |
| `--ocl-double-buff on/off` | Register double-buffered GEMM k-group loop (default `off`; faster on some GPUs, but more registers per work-item can spill on others) |
| `--ocl-throttle L` | OpenCL and oneDNN backends: `cl_khr_throttle_hints` queue hint, `low`, `med`, `high`, or `off`. With `low`, Intel's driver sleeps while waiting for the GPU instead of spinning a CPU core; `med`, `high` and `off` spin. Default `off`: `low` cuts oneDNN's CPU use from 0.5-1 core to about 0 (A770 and UHD 770, hashrate within 0.6%), but on the A770 the driver's sleeping wait occasionally misses its wake-up and a scan stalls for ~70 s, and `--backend opencl` loses 3.6% with it. Ignored by devices without the extension (NVIDIA) |

`--ocl-tile` sets the GEMM register tile. Jackpot XOR, hashrate counting, and proof layout use the corresponding semantic hash tile. `--ocl-macro` sets the WG macro block (work-group covers one macro); it is independent of the register tile. Examples: `--ocl-tile 4x8 --ocl-macro 64x64` or `--ocl-tile 4x8/64x64`. `--ocl-issue` / `--ocl-dot` / `--ocl-cpm-type` select the nest ([`docs/opencl_issue_shape.md`](opencl_issue_shape.md)). Default **auto** picks the best available accelerated dot, then float **B-scalar broadcast** fallback.

The GEMM flags above are for Pearl. Quantus on OpenCL picks its kernel variant per device at start-up, and environment variables override the choice ([`docs/quantus_opencl.md`](quantus_opencl.md)).

## wgpu options (Pearl)

| Flag | Description |
|------|-------------|
| `--wgpu-tile MxN[/MmMm]` | Register tile: `8x8` (default), `4x4`, `4x8`, or `8x16`. Optional `/64x64` or `/128x128` sets the macro (same as `--wgpu-macro`) |
| `--wgpu-macro MxN` | Macro block: `64x64` or `128x128` (default `128x128`) |
| `--wgpu-lds on/off` | Stage A/B k-block panels in workgroup memory (needs 2×macro×128 B: 32 KiB at 128, 16 KiB at 64; default `on` for discrete GPUs, `off` for integrated) |

Tile choices match `--ocl-tile` / `--ocl-macro`, including the hash tile used for the jackpot and proof (`4x4` hashes as `4x8`, one work-item per hash tile computing two 4x4 halves). A workgroup covers one macro with one work-item per hash tile (e.g. 256 for 8x8/128, 512 for 4x8/128, 32 for 8x16/64). Smaller tiles use fewer registers per work-item (useful on GPUs with small register files); on a GTX 1070 the default 8x8/128 with LDS is fastest.

## OneDNN options

Intel GPU backend (XeLP / Gen12LP or XeHPG). Requires `-Backend OneDnn` at build time; see [`src/onednn/README.md`](../src/onednn/README.md) for gemmstone deps.

| Flag | Description |
|------|-------------|
| `--fused-jackpot` | **Fused path (default):** one gemmstone kernel (IGEMM + fold + BLAKE3 + in-kernel jackpot compare). No `tile_xor` global readback. `--no-fused-jackpot` selects the separate jackpot kernel. |
| `--onednn-layout NAME` | Device A/B layout: `TN` (default), `TT`, `NT`, `NN`. `T`/`N` = row/column major; C is always N. Env: `CASE5_GEMM_LAYOUT`. (Deprecated, now debug only) |

**Scan paths**

- **`--no-fused-jackpot`:** Case 5 IGEMM writes each tile's 16-word jackpot message to `tile_xor` → separate device jackpot kernel. On XMX GPUs the message comes from the interleaved fold; elsewhere (Gen12LP, Xe-LPG) the milestones are folded into it in registers and flushed once per tile (`CASE5_SEPARATE_WRAP=0`: one store per milestone instead). Higher VRAM traffic (`tile_xor` panel buffer) but simpler judge (OpenCL reference in `kernels/cp_onednn_jackpot.cl`).
- **Default (`--fused-jackpot`):** fused kernel — fold, BLAKE3, and target compare in-register; host readback is `found_flag` + packed `(t_rows, t_cols)` only. Lower panel I/O.

**Batching:** `--row-period-batch` and `--batch-size` count **hash tiles** on the gemmstone unroll grid (typically 16×16 at production; see startup log `hash tile: MxN logical`). Host syncs after each panel for cancel/progress/share checks. Non-fused panels also size the `tile_xor` GPU buffer (~`row_batch × col_batch × (K/128)` dwords per panel).

## Scan batching (`--batch-size`)

Host syncs after each batch (cancel / progress / share check). Meaning differs by backend:

**OpenCL — 1D macro slicing**

Each macro block defaults to 128×128 (`--ocl-macro 64x64` for the smaller size). Macros are a 2D grid (`macro_rows × macro_cols`), walked as a flat index `mb`. `--batch-size N` is how many **macro blocks** each kernel launch covers (`CP_MACRO_BATCH_*` in `include/cp_config.h`). Tile / issue flags: [OpenCL options](#opencl-options).

- Default: `1024` (one full macro-row at production `m=n=131072` with 128×128 macros)
- Max: `1048576` (full matrix: `1024×1024` macros at 128×128; more macros when using 64×64)
- `--row-period-batch` is ignored on OpenCL

**CUDA — 2D launch window**

Default **CUTLASS fused** path tiles the matrix in **128×128 CTAs** (`CP_CUTLASS_CTA_M/N`). `--row-period-batch` / `--batch-size` count how many of those CTAs to launch per step (clipped to remaining).

`--cublas-period` (debug) uses BzMiner **periods** instead: `PP_ROW_PERIOD=128` × `PP_COL_PERIOD=256` (not the same as the 128×128 CTA).

| Flag | Role | Default | Max |
|------|------|---------|-----|
| `--row-period-batch` | Row CTAs (or row periods) per launch | 32 | 1024 |
| `--batch-size` / `--period-batch` / `--col-period-batch` | Col CTAs (or col periods) per launch | 1024 | 1024 |

CUTLASS fused needs no C buffer; `--cublas-period` sizes a period GEMM / C window.

**OneDNN — 2D hash-tile panels**

Same flags as CUDA row/col batching, but counts **hash tiles** (gemmstone logical unroll grid, e.g. 16×16 → 8192×8192 tiles at `m=n=131072`). Each panel = one gemmstone GEMM launch (+ separate jackpot kernel unless `--fused-jackpot`).

`--fused-jackpot` also applies to `--backend hexagon`: the jackpot runs on the DSP instead of the host CPU, ~1% slower with almost no CPU use (see [the Hexagon README](../src/pearl/hexagon/README.md)). With `--backend intel_npu` it runs on the NPU's SHAVEs: ~9% slower, 0.08 instead of 0.20 cores and 512 B instead of 512 KiB of tile words per launch through DRAM (see [the Intel NPU README](../src/pearl/intel_npu/README.md)). Off by default for both.

| Flag | Role | Default |
|------|------|---------|
| `--row-period-batch` | Hash-tile rows per panel | 256 |
| `--batch-size` / `--period-batch` / `--col-period-batch` | Hash-tile cols per panel | 256 |

At defaults with 16×16 hash tiles and production dims, one panel covers 256×256 = 65536 hash tiles before host sync. `--row-period-batch` is ignored on OpenCL.

**Hexagon — 2D macro launches**

One DSP launch covers `--row-period-batch` × `--batch-size` macro blocks of 128×128 (clipped to the matrix). Each row panel of A is packed on the DSP once and then run against one block of columns per launch.

| Flag | Role | Default |
|------|------|---------|
| `--row-period-batch` | 128-row macros per launch | 32 (4096 rows) |
| `--batch-size` / `--period-batch` / `--col-period-batch` | 128-column macros per launch | 32 (4096 columns) |

At the defaults one launch is 4096×4096 = 65536 hash tiles (4×64), about 170 ms of DSP time. Taller launches amortize the DSP's per-tile copies of B better (2048 rows is ~3% slower). Column width barely matters.

**Intel NPU — 2D macro launches**

One NPU launch covers `--row-period-batch` × `--batch-size` macro blocks of 128×128, one compiled graph per shape (cached in `npu_cache/` after the first compile). Columns stop at 4 macros; rows above 4 macros are whole 512-row blocks; a shape that does not fit or divide the matrix is lowered with a message.

| Flag | Role | Default |
|------|------|---------|
| `--row-period-batch` | 128-row macros per launch | 16 (2048 rows) |
| `--batch-size` / `--period-batch` / `--col-period-batch` | 128-column macros per launch (at most 4) | 4 (512 columns) |

2048×512 is the fastest measured shape: 1.59 TOPS per launch against 1.50 for 512×512 and 1.47 for 16384×512, and 0.80 against 0.75 TMAC/s mining at 131072² ([details](../src/pearl/intel_npu/README.md#measurements-core-ultra-9-275hx-npu-2-tiles)).
