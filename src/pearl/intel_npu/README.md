# Pearl on the Intel NPU

`--backend intel_npu` mines Pearl on the NPU of Intel Core Ultra processors (NPU3720 class: Meteor Lake, Arrow
Lake). The NPU runs the exact int8 GEMM and the per-tile milestone XOR; the host builds the noisy matrices and
runs the BLAKE3 jackpot. The design comes from `gemm_xor_benchmark/npu` (Case 3.4 on the NPU), which has the full
measurements and the reasoning behind each choice.

Arrow Lake Core Ultra 9 275HX: about **0.81 TMAC/s** (810 GH/s) with the default 2048 × 512 launch, mock shares
verify at 8192² and 131072².

## How it works

| Piece | Where | Role |
|-------|-------|------|
| Launch graph | `cp_npu_graph.cpp` | One OpenVINO model per launch shape (rows × cols × 4096, in 128 × 128 macros): per row block of 512 and K panel of 128, an int32 GEMM on the DPU of the A panel with a 64-column block of B, then the `TileXorAcc` op (8 panels per op) keeps the running sum and XORs every 16 × 16 tile on the SHAVEs. The column chains of a row block run in lockstep so each A panel leaves CMX after its last use. |
| Engine | `cp_npu_engine.cpp` | The patched compiler's switches, the compiled-graph cache (`npu_cache/`, keyed by the kernel ELFs and the compiler DLL), infer requests, NPU-visible host tensors (inputs are read in place). |
| Worker | `cp_pearl_npu_worker.cpp` | Zero-B like the CPU and Hexagon workers. Per job: noisy B built on the host and packed per launch column block. Per attempt: the sparse signal A, its incremental hash, the seeds; a prep thread packs each launch row panel of noisy A ahead of its launches; the calling thread keeps 4 launches queued (one per infer request, `start_async`), and as each completes it checks that output's jackpot and queues the next launch on that request. A hit is recomputed on the host before it becomes a share. |
| SHAVE kernels | `kernels/tile_xor_acc.3720xx*.elf`, `kernels/tile_jackpot.3720xx*.elf` | Prebuilt hand-scheduled `TileXorAcc` kernel (it takes the tile height, 8 or 16 rows, from its tensors) and the C `TileJackpot` kernel of `--fused-jackpot`. Sources and generator: `gemm_xor_benchmark/npu/kernels` and `tools/gen_tile_xor_asm.py`, built by `tools/build_tile_xor_kernel.ps1` (needs MoviTools to rebuild). |
| Compiler patch | `patches/npu_compiler_all.diff` | Against `openvinotoolkit/npu_compiler` at `patches/NPU_COMPILER_COMMIT`: int32 DPU output (`NPU_EXPERIMENTAL_I32_DPU_OUTPUT=1`), the `TileXorAcc` and `TileJackpot` ops from the frontend to the SHAVE kernel calls (both split over the 2 tiles by rows and over each tile's 2 ACT SHAVEs), the tile height (`NPU_EXPERIMENTAL_TILE_XOR_ROWS`, 8 by default), loading external SHAVE kernels (`NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR`), the tile-word layout and cycle costs. The miner sets all three variables itself. |

Hash tile 16 × 16 (proof layout `CP_TILE_LAYOUT_CONTIGUOUS_16x16`). The SHAVE work is the same as with 8 × 16 tiles,
but the NPU writes half the tile words (512 KiB per 2048 × 512 launch) and the host hashes half as many tiles: at
131072², 0.81 TMAC/s and 13.4 s of jackpot per attempt against 0.80 TMAC/s and 25.5 s. A and B go to the NPU as
u8 holding value + 128, which the compiler folds into a cast; B is fed as rows of B^T (the form the int32 DPU path
computes exactly). Each row block of a launch reads B through its own graph input, all bound to the same host
tensor: the compiler merges identical slices of one input, so a B block shared by the row blocks would stay in CMX
across all of them (2048 × 512: 1.57 TOPS with one input per row block, 1.09 with one shared input).

### `--fused-jackpot`

With `--fused-jackpot` the BLAKE3 jackpot runs on the NPU too: per row block a `TileJackpot` op takes the
`TileXorAcc` tile words straight from CMX, folds each tile's 32 milestones, runs one keyed BLAKE3 compression per
tile (4 tiles per 128-bit vector) and compares the digest with the bound. Only one word per tile row goes back
(1 + the first hit's tile column, or 0): 512 B per launch instead of 512 KiB, and the host just scans those words.
The attempt's key and bound go to the NPU in a small input tensor. A hit is still recomputed on the host before it
becomes a share.

| 131072², 2048 × 512 | Host jackpot (default) | `--fused-jackpot` |
|---|---|---|
| Scan per attempt | 87.0 s, 0.81 TMAC/s | 95.6 s, 0.74 TMAC/s |
| Host jackpot / hit check per attempt | 12.5 s | 0.09 s |
| Miner CPU (process time / wall) | 0.20 cores | 0.08 cores |
| DRAM traffic per launch, besides 12 MiB of A and B reads | 512 KiB written by the NPU, read by the CPU | 512 B |

It costs about 9%: the SHAVEs already set the pace, and the jackpot adds about 1,500 SHAVE cycles per 4 tiles
(moviCompile's schedule of the C kernel, limited by dependent-op latencies; `cp_npu_launch_test --fused --profile`
shows the per-op times). Default off, like Hexagon's; use it when the CPU has other work (another miner on the
GPU or CPU). `cp_npu_launch_test --fused` checks every tile row's first hit against the host jackpot of the
reference words, with a bound loose enough that about 1 tile in 32 hits.

## Dependencies

The stock OpenVINO NPU stack cannot run this: the compiler exposes no int32 GEMM output, has no `TileXorAcc` op and
loads no external SHAVE kernels. The backend needs OpenVINO built from source (for the developer package that the
in-plugin compiler builds against) and the patched `npu_compiler`, both at pinned commits (`patches/OPENVINO_COMMIT`,
`patches/NPU_COMPILER_COMMIT`). `prepare_intel_npu_deps.ps1` does all of it, like `prepare_onednn_deps.bat` does for
gemmstone:

```powershell
powershell -ExecutionPolicy Bypass -File src\pearl\intel_npu\prepare_intel_npu_deps.ps1
```

1. Clones OpenVINO into `third_party/openvino-src` at the pinned commit with its submodules, configures it with Ninja
   (NPU plugin only, no CPU/GPU plugins, frontends, Python or tests) and builds it.
2. Clones `npu_compiler` into `third_party/npu_compiler-src` at the pinned commit with its submodules (LLVM among
   them), applies `npu_compiler_all.diff` (skipped when the tree already has it), configures it against OpenVINO's
   developer package and builds `openvino_intel_npu_compiler`, which lands in OpenVINO's `bin/intel64/Release`.
3. Writes `third_party/intel_npu/paths.cmake` (OpenVINO build and binary directories) for the CMake build.

Requires Git, MSVC 2022 (C++ tools), CMake 3.24+, Ninja and Python 3. The first run downloads several GB and builds
for hours; later runs only rebuild what changed. `-Refresh` re-runs both CMake configures, `-NoBuild` only fetches
and patches.

| Environment | Effect |
|-------------|--------|
| `OPENVINO_SRC` | Use an existing OpenVINO checkout (its build tree must be `<src>/build-x86_64/Release` or `OPENVINO_BUILD_DIR`, binaries `<src>/bin/intel64/Release` or `OPENVINO_BIN_DIR`). If it is built, it is used as is. |
| `NPU_COMPILER_SRC` | Use an existing `npu_compiler` checkout; its HEAD is never moved, the patch is applied if missing. |
| `OPENVINO_COMMIT`, `NPU_COMPILER_COMMIT` | Other revisions than the pins. The two must match each other, and the patch may need rework on a newer compiler. |

## Build

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,IntelNpu
```

`build.ps1` runs the prepare script, configures with `-DCP_ENABLE_INTEL_NPU=ON` and copies next to `cppminer.exe`:
the OpenVINO runtime (`openvino.dll`, `openvino_intel_npu_plugin.dll`, `tbb*.dll`, `plugins.xml`), the patched
`openvino_intel_npu_compiler.dll` with its loader, and `kernels/intel_npu/*.elf`. With plain CMake pass
`-DCP_INTEL_NPU_OPENVINO_DIR=<build-x86_64/Release> -DCP_INTEL_NPU_OPENVINO_BIN_DIR=<bin/intel64/Release>` if
`paths.cmake` does not exist. The build also produces `cp_npu_launch_test.exe`, which checks one launch against a host
reference on random data and measures the launch rate.

Windows only so far (the patch and the OpenVINO build are tested there).

## Run

```powershell
.\cppminer.exe --backend intel_npu --list-devices
.\cppminer.exe --backend intel_npu --mock --m 8 --n 8
.\cppminer.exe --backend intel_npu --wallet prl1... --worker rig01
```

The launch is `--row-period-batch` × `--col-period-batch` (alias `--batch-size`) macro blocks of 128 × 128, as on
the other backends; the default is 16 × 4 = 2048 × 512, the fastest shape measured (see below). Columns stop at 4
macros (8 chains of 64, all live in lockstep); rows above 4 macros are whole 512-row blocks; both must divide the
matrix (`--m`/`--n`). A shape that does not fit is lowered to the nearest one that does, with a message.

The first start with a shape compiles its launch graph into `npu_cache/` next to the executable (seconds for one row
block, about 9 s more per row block); later starts load it in under a second. A new kernel ELF or compiler DLL
selects a new cache subdirectory.

| Environment | Default | Effect |
|-------------|---------|--------|
| `CP_NPU_REQUESTS` | 4 | Launches queued on the NPU (infer requests), all driven by one host thread. 2, 4 and 8 measured the same in `cp_npu_launch_test` (2048 × 512: 1.58–1.62 TOPS); with 1 the NPU idles while each output is returned and the next launch submitted (1.49 TOPS). In the miner a request also stays out of the queue while its output is hashed (about 0.8 ms per 5.3 ms launch; 1.5 ms with 8 × 16 tiles, where 1 request gave 0.55 TMAC/s at 131072² and 2 already the full rate). |
| `CP_NPU_CACHE_DIR` | `npu_cache` next to the exe | Compiled-graph cache root. |
| `NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR` | `kernels/intel_npu` next to the exe | Where the compiler loads `tile_xor_acc` from (set by the miner unless already set). |

## Measurements (Core Ultra 9 275HX, NPU 2 tiles)

Launch shapes, `cp_npu_launch_test --rows R --cols C` (K = 4096, 4 requests, every shape bit-exact against the host;
measured with 8 × 16 tiles, 16 × 16 tiles give 512 × 512 1.495 and 2048 × 512 1.61 TOPS):

| Launch (macros) | Row blocks | First compile | Per launch | TOPS |
|-----------------|------------|---------------|------------|------|
| 512 × 256 (4 × 2) | 1 | 5 s | 0.78 ms | 1.35–1.39 |
| 512 × 512 (4 × 4) | 1 | 7 s | 1.43 ms | 1.47–1.51 |
| 1024 × 512 (8 × 4) | 2 | 17 s | 2.75 ms | 1.51–1.57 |
| **2048 × 512 (16 × 4)** | 4 | 37 s | 5.40 ms | **1.55–1.59** |
| 2048 × 512, one shared B input | 4 | 28 s | 7.9 ms | 1.08–1.11 |
| 4096 × 512 (32 × 4) | 8 | 78 s | 11.2 ms | 1.52–1.53 |
| 8192 × 512 (64 × 4) | 16 | 171 s | 22.7 ms | 1.50–1.52 |
| 16384 × 512 (128 × 4) | 32 | 413 s | 47 ms | 1.45–1.49 |

Interleaved 8-second runs: 512 rows 1.50, 1024 rows 1.56, 2048 rows 1.59 TOPS. Fewer columns leave the SHAVEs
fewer chains to overlap; more row blocks per launch amortize its turnaround until, past 2048 rows, the larger graph
schedules slightly worse.

| Mining run | Result |
|------------|--------|
| Mock, default 131072², 2048 × 512 (16384 launches per attempt) | 86.7 s per attempt, 0.81 TMAC/s; job setup 0.58 s (B seed 0.52 s, noisy B 0.05 s); share and proof verify |
| Same, 8 × 16 tiles | 87.4–88.0 s per attempt, 0.80 TMAC/s |
| Mock, default 131072², `--row-period-batch 4 --col-period-batch 4` (65536 launches, 8 × 16 tiles) | 93.4 s per attempt, 0.75 TMAC/s |
| Mock, `--m 8 --n 8`, 2048 × 512 (64 launches per attempt) | 0.35 s per attempt, 0.65–0.75 TMAC/s; share and 16 × 16 proof verify |

Per full-size attempt the host spends about 0.3 s building noisy A panels and 13.4 s of jackpot, all on the driving
thread while the NPU runs the queued launches (it is busy 15% of the attempt; with 8 × 16 tiles 25 s, and four
driving threads, one per request, measured the same rate). At 8192² an attempt is only 16 rounds of 4 launches, so
filling and draining the queue (one launch latency, about 21 ms with 2048 rows) costs as much as the larger launch
gains.

## Ceiling: the SHAVEs

The DPU is not what limits this backend. The NPU's int8 peak is 13.1 TOPS (2 tiles × 2048 MAC/cycle × 1.6 GHz),
and the compiler reaches about 11 TOPS on a plain GEMM of these shapes. The milestone XOR runs at 1.6 TOPS,
7× slower than that GEMM. The bottleneck is the 4 ACT SHAVEs (2 per tile, vector units 128 bits wide). For each
int32 partial the DPU writes, one per output element and K panel of 128 (256 ops of GEMM), they must add it into
the running accumulator and XOR it into its tile word. Measurements from `gemm_xor_benchmark/npu/REPORT.md`:

| Limit | Cycles per element-milestone, per SHAVE | Chip | TMAC/s |
|-------|------------------------------------------|------|--------|
| VAU floor: 2 vector ops (add, XOR) per 4 elements, the schedule's one bubble per 17 cycles | 0.53 | 2.3 TOPS | 1.15 |
| The hand-scheduled `TileXorAcc` loop as written | 0.557 | 2.2 TOPS | 1.1 |
| In the graph: the same loop with CMX stalls | 0.735 | 1.66 TOPS | 0.83 |
| Plus per-task fixed costs and scheduler gaps: what the miner reaches | | **1.62 TOPS** | **0.81** |

- **CMX read rate.** The SHAVEs read 8–9 bytes per cycle of on-chip memory while the DPU writes the next
  partials into the same CMX (about 25 GB/s) and the DMA engines move A, B and the tile words. Stalls from that
  arbitration make up about 20% of a task. They scale with the bytes read, and the 4 bytes of partial per
  element-milestone cannot shrink. An XOR-only kernel, with the DPU producing cumulative sums, still runs at
  0.62 cycles per element, so even then the design stays around 2.0 TOPS.
- **Fixed costs per task, about 8%.** The kernel's parameter block is in uncached memory (about 70 cycles per
  load, ~2 K cycles per task), and every task starts with a cold instruction cache. Bigger tasks would amortize
  them, but they need CMX that A, the accumulators and the partials already fill.
- **Scheduler gaps, 4–20% depending on the graph.** Each chain's GEMMs need their own weight DMA and barrier. The
  lockstep chains of a 512-row block hide most of that, and the SHAVEs stay about 96% busy.

The SHAVEs set the pace, so anything else they run comes straight off the hashrate. `--fused-jackpot` adds about
1,500 cycles per 4 tiles (one BLAKE3 compression), which costs 9%. The host side has plenty of slack: one thread
spends about 15% of the scan on the jackpot.

The VAU floor of 2.3 TOPS (1.15 TMAC/s) is the hard ceiling of this design on NPU3720; the benchmark puts a
realistic target at about 1.8 TOPS (0.9 TMAC/s). Getting there is compiler and runtime work rather than kernel
work:
- an in-place accumulator, to free a 64–128 KiB buffer per op so chains can interleave at 1024 × 64 blocks
  (~1.7 TOPS);
- grouped DPU tasks, one weight DMA and barrier per 4 panels;
- kernel parameters in CMX, worth about 8% of task time.

The kernel itself has nothing obvious left.

## Acknowledgements

The SHAVE kernel builds on [`npunlock`](https://github.com/hsfzxjy/npunlock) (used at `88bd584`), the project that
reconstructed how to compile C into runnable NPU3720 SHAVE kernels:

- **The toolchain.** The SHAVE compiler, assembler and linker are the MoviTools DLLs from an older Intel NPU driver
  package; npunlock's `wiki/GET_MOVITOOLS.md` documents where to get them and how to extract them.
- **The kernel image and its invocation.** npunlock's `wiki/REVERSE_ENGINEERING.md` and `wiki/HOW_NPUNLOCK_WORKS.md`
  establish how a SHAVE kernel image is laid out (`.text` at `0x1d000000`, `.arg.data` at `0x1e000000`, entry from
  `e_entry`) and how the firmware invokes it with a parameter block.

Neither the miner nor its build uses npunlock's code: `kernels/tile_xor_acc.3720xx*.elf` are prebuilt, follow the NPU
compiler's own kernel ABI, and are loaded by the patched compiler. npunlock and MoviTools are only needed to rebuild
the kernel (see `gemm_xor_benchmark/npu`).
