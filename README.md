# CPPminer

Cross-Platform Pearl (now multi-algo) miner written in C++ supporting a wide range of hardware. Select the algorithm at runtime with `--algo` (default `pearl`).

| Algo | Backends | PoW |
|------|----------|-----|
| `pearl` | `cpu` / `cuda` / `opencl` / `onednn` / `wgpu` / `hexagon` | GEMM+XOR jackpot + `plain_proof` |
| `quantus` | `cpu` / `wgpu` / `opencl` | Poseidon2 QPoW (`qpow-poseidon2`) |

Pool / job logistics live under `src/common/`. Pearl compute backends are separate worker directories; Quantus lives under `src/qpow/`:

| Backend | Directory | Status |
|---------|-----------|--------|
| CPU | `src/cpu/` | Pearl: fused GEMM+XOR (contiguous 8×16) |
| CUDA | `src/cuda/` | Pearl: Pascal CUTLASS fused GEMM+XOR+jackpot |
| OpenCL | `src/opencl/` | Pearl: fused GEMM+XOR+jackpot (AMD / generic OpenCL) |
| OneDNN | `src/onednn/` | Pearl: Intel GPU gemmstone IGEMM + tile XOR + GPU jackpot |
| Pearl wgpu | `src/pearl/wgpu/` + `rust/cp-pearl-wgpu-ffi` | Pearl: WGSL fused GEMM+XOR+jackpot (Vulkan / DX12 / Metal), optional LDS staging |
| Pearl Hexagon | `src/pearl/hexagon/` (+ DSP library in `dsp/`) | Pearl: Snapdragon cDSP HVX fused GEMM+XOR (contiguous 4×64) + host jackpot ([README](src/pearl/hexagon/README.md)) |
| Quantus CPU | `src/qpow/cpu/` | Poseidon2 midstate search (scalar + AVX2 4-wide) |
| Quantus wgpu | `src/qpow/wgpu/` + `rust/cp-quantus-wgpu-ffi` | GpuEngine FFI |
| Quantus OpenCL | `src/qpow/opencl/` | Poseidon2 ulong kernel (port of mining_u64.wgsl) |

## Requirements

- MSVC + **CMake** (Windows) or GCC/Clang + CMake (Linux/macOS)
- **Rust toolchain** (`cargo` / `rustup`, or `conda install -c conda-forge rust`) for in-process proof build and `--verify`
- **CPU build:** portable scalar baseline with runtime ISA dispatch: x86 AVX512-VNNI/AVX-VNNI/AVX512-BW/AVX2/SSSE3/scalar and AArch64 DotProd/NEON/scalar (`--simd`)
- **CUDA build:** NVIDIA GPU + CUDA Toolkit 12.x (+ CUTLASS, fetched by `build.ps1`).
- **OpenCL build:** OpenCL 1.2 runtime ICD from the GPU driver. Windows builds link vendored `third_party/opencl/lib/x64/OpenCL.lib` + Khronos headers (no CUDA/oneAPI/AMD SDK). Optional `cl_khr_integer_dot_product`, `__builtin_amdgcn_sdot4`.
- **OneDNN build:** Intel XeLP/XeHPG GPU + OpenCL + vendored oneDNN gemmstone/ngen (see `src/onednn/README.md`).
- **Hexagon build:** Android/Linux on a Snapdragon with a cDSP (tested: Snapdragon 480, Hexagon v66). The miner needs no SDK; the DSP library `libpearlx_skel.so` is built with the Hexagon SDK (see [`src/pearl/hexagon/README.md`](src/pearl/hexagon/README.md)).
- **wgpu build:** Rust toolchain. Enable with `-DCP_ENABLE_WGPU=ON` / `-Backend Wgpu`. Builds two FFI crates: `rust/cp-pearl-wgpu-ffi` (Pearl) and `rust/cp-quantus-wgpu-ffi` (Quantus; build scripts fetch [`Quantus-Network/quantus-miner`](https://github.com/Quantus-Network/quantus-miner) into `third_party/quantus-miner` for `engine-gpu`).

## Build options (CMake)

```bash
cmake -S . -B build \
  -DCP_ENABLE_CPU=ON \
  -DCP_ENABLE_CUDA=OFF \
  -DCP_ENABLE_OPENCL=OFF \
  -DCP_ENABLE_ONEDNN=OFF
cmake --build build --config Release
```

| Option | Default | Meaning |
|--------|---------|---------|
| `CP_ENABLE_CPU` | ON | CPU worker |
| `CP_ENABLE_CUDA` | OFF | CUDA/CUTLASS worker |
| `CP_ENABLE_OPENCL` | OFF | OpenCL worker |
| `CP_ENABLE_ONEDNN` | OFF | Intel GPU oneDNN/gemmstone worker |
| `CP_ENABLE_WGPU` | OFF | wgpu workers: Pearl (`cp-pearl-wgpu-ffi`) and Quantus GpuEngine (`cp-quantus-wgpu-ffi`; fetches `third_party/quantus-miner`) |
| `CP_ENABLE_HEXAGON` | OFF | Pearl Snapdragon cDSP (HVX) worker; Android/Linux only |
| `CP_ENABLE_CUBLAS` | OFF | Link cuBLAS for `--cublas-period` debug path (needs CUDA) |
| `CP_CUDA_ARCH` | native | e.g. `61` for Pascal |

Enable multiple backends in one binary; select at runtime with `--backend`. Both algos are always compiled in; use `--algo pearl|quantus`. Runtime and compile-time **algo×backend matrix**:

| | cpu | cuda | opencl | onednn | wgpu | hexagon |
|--|-----|------|--------|--------|------|---------|
| pearl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| quantus | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ |

## Build (Windows)

Requires MSVC Build Tools and CMake (same CMake pipeline as `build.sh` on *nix). CPU-only default:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
# equivalent: -Backend Cpu
```

CUDA, OpenCL, or combinations (comma-separated list):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,OpenCl,OneDnn
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Wgpu
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Cuda,OpenCl
# Optional debug: link cuBLAS (large DLLs; not needed for production CUTLASS path)
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -EnableCublas -CudaArch 61
```

Produces `cppminer.exe` in the repo root (plus `cp_pearl_wgpu_ffi.dll` and `cp_quantus_wgpu_ffi.dll` when wgpu is enabled).

## Build (*nix)
```bash
./build.sh --backend cpu,opencl,onednn,cuda
./build.sh --backend cpu,wgpu
```
This script pulls third-party dependencies and runs CMake.

### Hexagon

`./build.sh --backend cpu,hexagon` builds only the miner (its Hexagon worker and FastRPC client), and `build.ps1` has no Hexagon option. Neither script builds the DSP libraries the miner loads on the cDSP, `libpearlx_skel.so` and `libworker_pool.so`: they need the Hexagon SDK on an x86 machine. See [Build in the Hexagon README](src/pearl/hexagon/README.md#build) for those, and for cross-compiling the miner for Android.
## Run

```
cppminer [--algo pearl|quantus] --backend BACKEND [--devices IDS]
         [--pool stratum+tcp://HOST:PORT] --wallet ADDRESS [--worker NAME] [backend options]
```

- `--algo` defaults to `pearl`. Pearl runs on `cpu`, `cuda`, `opencl`, `onednn`, `wgpu` and `hexagon`; Quantus on `cpu`, `opencl` and `wgpu`. The backend must be enabled in the build.
- Pearl defaults to the LuckyPool Pearl pool; Quantus needs `--pool`.
- `--list-devices` prints the device indices that `--devices` takes.
- `--mock` mines a fixed job offline until the first share, verifies it and exits; it needs no pool or wallet.

Examples (`.\cppminer.exe` on Windows, `./cppminer` elsewhere):

```sh
# Pearl on GPU 0 (CUDA build)
cppminer --backend cuda --devices 0 --wallet prl1... --worker rig01

# Quantus on the CPU
cppminer --algo quantus --backend cpu --pool stratum+tcp://HOST:PORT --wallet qzpp... --worker rig01

# Find a GPU, then test it offline
cppminer --backend opencl --list-devices
cppminer --backend opencl --devices 0 --mock
```

### Options

| Flag | Description |
|------|-------------|
| `--algo` | `pearl` (default) or `quantus` (`qpow` / `qpow-poseidon2` aliases). Quantus: `cpu` / `wgpu` / `opencl`; Pearl: `cpu` / `cuda` / `opencl` / `onednn` / `wgpu` / `hexagon`. `--pool` required for Quantus unless `--mock` |
| `--backend` | `cpu` / `cuda` / `opencl` / `onednn` / `wgpu` / `hexagon` (must be compiled in; must be valid for `--algo`).
| `--pool` | `stratum+tcp://host:port` (required for `--algo quantus` unless `--mock`) |
| `--wallet` | Wallet address (required unless `--mock`) |
| `--worker` | Worker name (default `rig01`). On a `kryptex` pool host the wallet is sent as `WALLET.worker` as well |
| `--pool-pass STR` | `mining.authorize` password (default `x`). Kryptex reads a custom share difficulty from it: `--pool-pass d=2097152` (their default; `d = hashrate_H/s * target_share_seconds / 4294967296`) |
| `--threads N` | CPU backend OpenMP threads, Pearl and Quantus. Default: all CPUs the process may use (respects `taskset` / cpusets); `OMP_NUM_THREADS` overrides the default. Pearl threads are pinned physical cores first, then SMT siblings, so N up to the core count gives one thread per core. Set `OMP_PLACES` / `OMP_PROC_BIND` to let the OpenMP runtime place threads instead, `CP_CPU_AFFINITY=0` to disable pinning |
| `--devices` | CUDA device ids, OpenCL flat index, or wgpu mining-adapter indices (`--list-devices`) |
| `--list-devices` | List devices for the selected backend and exit |
| `--m N`, `--n N` | Matrix rows / columns in units of 1024 (default 128 = 131072; each ≤ 256, `m*n` ≤ 128×128) |
| `--cpu-gen` | Host matrix prep on GPU paths (OpenCL ~1 GiB VRAM; CUDA debug) |
| `--cutlass-fused` | CUDA: fused CUTLASS GEMM + jackpot (**default**) |
| `--cublas-period` | CUDA debug: cuBLAS period GEMM (only if built with `CP_ENABLE_CUBLAS`) |
| `--no-cutlass-fused` | CUDA debug: non-CUTLASS period path |
| `--batch-size N` | Launch batch. Pearl: col/macro panel size (default 1024; backend may remap). Quantus wgpu/OpenCL: **nonces per launch** (default 1000000). Aliases: `--period-batch`, `--col-period-batch` |
| `--period-batch N` | Alias for `--batch-size` |
| `--col-period-batch N` | Alias for `--batch-size` |
| `--row-period-batch N` | CUDA only: row-period batch (default 32, max 1024) |
| `--max-nonce N` | Stop after N attempts per job |
| `--dry-run` | Build proof without submitting; saves `pp_header.bin` / `pp_proof.b64` next to the executable (a normal run writes nothing to disk) |
| `--verify` | In-process zk-pow jackpot verify before submit (needs vendored `zk-pow`) |
| `--mock` / `-mock` | Offline: fixed job, mine until first share, verify, exit (implies dry-run). Pearl: zk-pow verify; Quantus: Poseidon2 `hash < target` |
| `--mock-diff D` | Mock difficulty (higher = longer). Defaults: Pearl **44** (unscaled target 2^(256-D)); Quantus **1000000** (`U512::MAX / D`). `--mock-diff` overrides for either. |
| `--cert-version N` | Force certificate / noise-seed version: `1`/`2` = legacy, `3` = salted (V3). Default **3**. Without this flag, pool `mining.notify` `cert_version` wins when present (1–3); otherwise default 3 |
| `--prepack MODE` | CPU: `fused` (default), `reuse`, or `separate` matrix prepack |
| `--simd ISA` | CPU: `auto` (default), `hybrid`, `avx512vnni`, `avxvnni`, `avx512` (base AVX512F+BW, `avx512bw` alias), `avx2`, `ssse3` (`sse` alias), `i8mm`, `dotprod`, `neon`, `scalar`. Quantus: `hybrid` runs one scalar and one AVX2 Poseidon2 worker per physical core (SMT siblings); `auto` is the best available mode (currently `hybrid`; may select a wider kernel such as AVX-512 in the future); `avx2` forces AVX2 on every thread; any other value runs scalar. Pearl: `hybrid` is the same as `auto` |
| `--simd-test` | Compare every available CPU SIMD kernel against scalar and exit (Quantus: AVX2 Poseidon2 field ops and hash parity vs scalar) |
| `--prepack-test` | Check CPU fused and reuse prepack against separate at m=n=8192 (prepacked bytes + full tile XOR) and exit |

### OpenCL options

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

`--ocl-tile` sets the GEMM register tile. Jackpot XOR, hashrate counting, and proof layout use the corresponding semantic hash tile. `--ocl-macro` sets the WG macro block (work-group covers one macro); it is independent of the register tile. Examples: `--ocl-tile 4x8 --ocl-macro 64x64` or `--ocl-tile 4x8/64x64`. `--ocl-issue` / `--ocl-dot` / `--ocl-cpm-type` select the nest ([`docs/opencl_issue_shape.md`](docs/opencl_issue_shape.md)). Default **auto** picks the best available accelerated dot, then float **B-scalar broadcast** fallback.

### wgpu options (Pearl)

| Flag | Description |
|------|-------------|
| `--wgpu-tile MxN[/MmMm]` | Register tile: `8x8` (default), `4x4`, `4x8`, or `8x16`. Optional `/64x64` or `/128x128` sets the macro (same as `--wgpu-macro`) |
| `--wgpu-macro MxN` | Macro block: `64x64` or `128x128` (default `128x128`) |
| `--wgpu-lds on/off` | Stage A/B k-block panels in workgroup memory (needs 2×macro×128 B: 32 KiB at 128, 16 KiB at 64; default `on` for discrete GPUs, `off` for integrated) |

Tile choices match `--ocl-tile` / `--ocl-macro`, including the hash tile used for the jackpot and proof (`4x4` hashes as `4x8`, one work-item per hash tile computing two 4x4 halves). A workgroup covers one macro with one work-item per hash tile (e.g. 256 for 8x8/128, 512 for 4x8/128, 32 for 8x16/64). Smaller tiles use fewer registers per work-item (useful on GPUs with small register files); on a GTX 1070 the default 8x8/128 with LDS is fastest.

### OneDNN options

Intel GPU backend (XeLP / Gen12LP or XeHPG). Requires `-Backend OneDnn` at build time; see [`src/onednn/README.md`](src/onednn/README.md) for gemmstone deps.

| Flag | Description |
|------|-------------|
| `--fused-jackpot` | **Fused path (default):** one gemmstone kernel (IGEMM + fold + BLAKE3 + in-kernel jackpot compare). No `tile_xor` global readback. `--no-fused-jackpot` selects the separate jackpot kernel. |
| `--onednn-layout NAME` | Device A/B layout: `TN` (default), `TT`, `NT`, `NN`. `T`/`N` = row/column major; C is always N. Env: `CASE5_GEMM_LAYOUT`. (Deprecated, now debug only) |

**Scan paths**

- **`--no-fused-jackpot`:** Case 5 IGEMM writes each tile's 16-word jackpot message to `tile_xor` → separate device jackpot kernel. On XMX GPUs the message comes from the interleaved fold; elsewhere (Gen12LP, Xe-LPG) the milestones are folded into it in registers and flushed once per tile (`CASE5_SEPARATE_WRAP=0`: one store per milestone instead). Higher VRAM traffic (`tile_xor` panel buffer) but simpler judge (OpenCL reference in `kernels/cp_onednn_jackpot.cl`).
- **Default (`--fused-jackpot`):** fused kernel — fold, BLAKE3, and target compare in-register; host readback is `found_flag` + packed `(t_rows, t_cols)` only. Lower panel I/O.

**Batching:** `--row-period-batch` and `--batch-size` count **hash tiles** on the gemmstone unroll grid (typically 16×16 at production; see startup log `hash tile: MxN logical`). Host syncs after each panel for cancel/progress/share checks. Non-fused panels also size the `tile_xor` GPU buffer (~`row_batch × col_batch × (K/128)` dwords per panel).

### Scan batching (`--batch-size`)

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

`--fused-jackpot` also applies to `--backend hexagon`: the jackpot runs on the DSP instead of the host CPU, ~1% slower with almost no CPU use (see [the Hexagon README](src/pearl/hexagon/README.md)).

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

## Performance

Hashrate on matrix size `m=n=131072`, `k=4096`, `r=128`. Rates are MAC/s (`docs/hashrate_calculation.md`). Figures are indicative; your results will vary with clocks, drivers, and batch settings. GPUs are tested with stock settings, no clock/power/voltage tuning unless explicitly stated. Power is readed from HWinfo64.

### NVIDIA GPU (CUDA)

| Device | ISA | Hashrate | Power (W) |
|--------|-----|----------|-----------|
| GTX 1070 | DP4A | 9.1 TH/s | 150 |

### AMD GPU (OpenCL)

| Device | ISA | Hashrate | Power (W) |
|--------|-----|----------|-----------|
| Radeon Pro 5500M | DP4A | 5.4 TH/s | 50 |

### Intel GPU (OneDNN)

| Device | ISA | Hashrate | Power (W) |
|--------|-----|----------|-----------|
| UHD 770 | DP4A | 1.4 TH/s | 13 |
| Xe-LPG 64EU (Core Ultra 9 275HX) | DP4A | 3.5 TH/s | 14 |

### Other GPU (OpenCL)
| Device | ISA | Tile size | Hashrate |
|--------|-----|-----------|----------|
| Intel HD graphics 10EU (Haswell) | Scalar | 4x8 | 25 GH/s |
| Intel UHD 630 | Scalar | 4x8 | 140 GH/s |
| Mali-G57 MC2 | DP4A | 4x8 | 115 GH/s |

### Hexagon DSP

| Device | Hashrate |
|--------|----------|
| Hexagon 686 (Snapdragon 480, v66, 2x HVX, 940MHz) | 400 GH/s |

### CPU

| Device | ISA | Hashrate | Bzminer v25.0.1b2 baseline |
|--------|-----|----------| ---------------------------|
| Celeron G1840 @ 2.8GHz | SSSE3 | 35 GH/s | 4 GH/s |
| Core i9 9980HK @ 2.4GHz | AVX2 | 440 GH/s | 300 GH/s |
| Core i5 12490F @ 4.0GHz | AVX-VNNI | 1.1 TH/s | 400 GH/s |
| Core i9 12900K 8P @ 4.9GHz + 8E @ 3.7GHz | AVX-VNNI | 2.3 TH/s | 920 GH/s |
| BCM2837 4x A53 @ 1.2GHz (rpi3b m=n=32) | NEON | 9 GH/s | N/A |
| Dimensity 6300 2x A76 @ 2.6GHz + 6x A55 @ 2.0GHz | NEON DotProd | 160 GH/s | N/A |

## Vendored proof stack (`third_party/`)

`cp-proof-ffi` is self-contained under this repo — no external `pearl/` checkout:

| Crate | Path | Purpose |
|-------|------|---------|
| pearl-blake3 | `third_party/pearl-blake3` | Merkle proof build |
| zk-pow | `third_party/zk-pow` | Jackpot verify (`--verify`) |
| plonky2 | `third_party/plonky2` | zk-pow compile dependency |

`build.ps1` / `build.sh` both drive CMake, which runs `cargo build --release` in `rust/cp-proof-ffi/` when `cargo` is available and links the static library into the miner (`cp_proof_ffi.lib` on Windows, `libcp_proof_ffi.a` on Linux/macOS). Proofs are built and verified in-process.

If any are missing, copy from the Pearl repo:

```powershell
Copy-Item -Recurse pearl\pearl-blake3 third_party\pearl-blake3
Copy-Item -Recurse pearl\zk-pow       third_party\zk-pow
Copy-Item -Recurse pearl\plonky2      third_party\plonky2
```

## Kryptex gzip stratum (v2)

Kryptex's "Pearl stratum gzip protocol" ([spec](https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe)) is response-driven:

- `mining.authorize` always carries `"password"` (`--pool-pass`, default `x`) and `"type":"v2"`. On a `kryptex` host the `wallet` is sent as `WALLET.worker`; the separate `worker`/`agent` fields (LuckyPool) are sent too.
- If the pool answers `{"error":null,"id":1,"result":true,"type":"v2"}` the miner logs `[pool] proof encoding: gzip` and every `mining.submit` carries `plain_proof` = base64 of the **gzip** stream (zlib wbits 31) of the bincode proof (`cp_proof_gzip_b64` in `rust/cp-proof-ffi`, pure-Rust miniz_oxide). `[plain] proof gzip: N -> M chars` shows the saving per share.
- A response without `"type":"v2"` (LuckyPool) keeps plain base64 proofs. The flag is re-evaluated on every authorize (dev-fee wallet switch included).
- Custom share difficulty: `--pool-pass d=2097152` (Kryptex default; `d = hashrate_H/s * target_share_seconds / 4294967296`).

gzip helps because the committed matrices are repetitive: the light random A (one random row) and the all-zero B^T compress well, e.g. an OpenCL proof goes from 68160 to 1948 base64 chars (35x).

## Dev Fee

A transparent **1%** developer fee uses a **tile-debt** schedule on the **same pool**:

- `T` = hash tiles in one full matrix scan for the active backend/layout/dims; CPU/OpenCL use the selected contiguous tile shape; CUDA periodic/CUTLASS may differ).
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
src/pearl/hexagon/ Snapdragon cDSP worker; dsp/ = HVX kernel + FastRPC library (Hexagon SDK)
rust/             cp-proof-ffi (plain_proof Merkle + bincode)
third_party/      blake3, pearl-blake3, zk-pow, plonky2, opencl (+headers), cutlass (CUDA)
```

## Modules

- **cp_pool** — LuckyPool stratum TCP, reader thread, plain_proof submit
- **cp_fee** — Same-pool 1% tile-debt developer fee (reconnect + authorize)
- **cp_mine** — Job loop: A/B gen, noise fuse, worker scan, Rust proof build
- **cp_worker** — Backend selection (`cpu` / `cuda` / `opencl` / `onednn`)
- **cp_cpu** — Fused GEMM+XOR + host BLAKE3 jackpot
- **cp_gpu** — CUDA plain_proof path (under `src/cuda/`)
- **cp_opencl** — OpenCL plain_proof path (under `src/opencl/`)
- **cp_onednn** — Intel GPU oneDNN/gemmstone path (under `src/onednn/`)
- **cp_pearl_hexagon** — Snapdragon cDSP HVX GEMM+XOR via FastRPC + host BLAKE3 jackpot (under `src/pearl/hexagon/`)
- **cp_noise** — Matrix generation and pearl noise
