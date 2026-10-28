# Pearl on the Intel NPU

`--backend intel_npu` mines Pearl on the NPU of Intel Core Ultra processors (NPU3720 class: Meteor Lake, Arrow
Lake). The NPU runs the exact int8 GEMM and the per-tile milestone XOR; the host builds the noisy matrices and
runs the BLAKE3 jackpot. The design comes from `gemm_xor_benchmark/npu` (Case 3.4 on the NPU), which has the full
measurements and the reasoning behind each choice.

Arrow Lake Core Ultra 9 275HX: about **0.80 TMAC/s** (800 GH/s) with the default 2048 × 512 launch, mock shares
verify at 8192² and 131072².

## How it works

| Piece | Where | Role |
|-------|-------|------|
| Launch graph | `cp_npu_graph.cpp` | One OpenVINO model per launch shape (rows × cols × 4096, in 128 × 128 macros): per row block of 512 and K panel of 128, an int32 GEMM on the DPU of the A panel with a 64-column block of B, then the `TileXorAcc` op (8 panels per op) keeps the running sum and XORs every 8 × 16 tile on the SHAVEs. The column chains of a row block run in lockstep so each A panel leaves CMX after its last use. |
| Engine | `cp_npu_engine.cpp` | The patched compiler's switches, the compiled-graph cache (`npu_cache/`, keyed by the kernel ELFs and the compiler DLL), infer requests, NPU-visible host tensors (inputs are read in place). |
| Worker | `cp_pearl_npu_worker.cpp` | Zero-B like the CPU and Hexagon workers. Per job: noisy B built on the host and packed per launch column block. Per attempt: the sparse signal A, its incremental hash, the seeds; a prep thread packs each launch row panel of noisy A ahead of its launches; the calling thread keeps 4 launches queued (one per infer request, `start_async`), and as each completes it checks that output's jackpot and queues the next launch on that request. A hit is recomputed on the host before it becomes a share. |
| SHAVE kernel | `kernels/tile_xor_acc.3720xx*.elf` | Prebuilt hand-scheduled `TileXorAcc` kernel. Source and generator: `gemm_xor_benchmark/npu/kernels` and `tools/gen_tile_xor_asm.py` (needs MoviTools to rebuild). |
| Compiler patch | `patches/npu_compiler_all.diff` | Against `openvinotoolkit/npu_compiler` at `patches/NPU_COMPILER_COMMIT`: int32 DPU output (`NPU_EXPERIMENTAL_I32_DPU_OUTPUT=1`), the `TileXorAcc` op from the frontend to the SHAVE kernel call, loading external SHAVE kernels (`NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR`), the tile-word layout and cycle costs. The miner sets both variables itself. |

Hash tile 8 × 16 (proof layout `CP_TILE_LAYOUT_CONTIGUOUS`, the same as the CPU worker). A and B go to the NPU as
u8 holding value + 128, which the compiler folds into a cast; B is fed as rows of B^T (the form the int32 DPU path
computes exactly). Each row block of a launch reads B through its own graph input, all bound to the same host
tensor: the compiler merges identical slices of one input, so a B block shared by the row blocks would stay in CMX
across all of them (2048 × 512: 1.57 TOPS with one input per row block, 1.09 with one shared input).

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
| `CP_NPU_REQUESTS` | 4 | Launches queued on the NPU (infer requests), all driven by one host thread. 2, 4 and 8 measured the same in `cp_npu_launch_test` (2048 × 512: 1.58–1.62 TOPS); with 1 the NPU idles while each output is returned and the next launch submitted (1.49 TOPS). In the miner a request also stays out of the queue while its output is hashed (about 1.5 ms per 5.4 ms launch), so use at least 2. |
| `CP_NPU_CACHE_DIR` | `npu_cache` next to the exe | Compiled-graph cache root. |
| `NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR` | `kernels/intel_npu` next to the exe | Where the compiler loads `tile_xor_acc` from (set by the miner unless already set). |

## Measurements (Core Ultra 9 275HX, NPU 2 tiles)

Launch shapes, `cp_npu_launch_test --rows R --cols C` (K = 4096, 4 requests, every shape bit-exact against the host):

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
| Mock, default 131072², 2048 × 512 (16384 launches per attempt) | 87.4–88.0 s per attempt, 0.80 TMAC/s; job setup 0.57 s (B seed 0.51 s, noisy B 0.06 s); share and proof verify |
| Mock, default 131072², `--row-period-batch 4 --col-period-batch 4` (65536 launches) | 93.4 s per attempt, 0.75 TMAC/s |
| Mock, `--m 8 --n 8`, 2048 × 512 (64 launches per attempt) | 0.37 s per attempt, 0.65–0.75 TMAC/s; share and proof verify |

Per full-size attempt the host spends about 0.3 s building noisy A panels and 25 s of jackpot, all on the driving
thread while the NPU runs the queued launches (it is busy 29% of the attempt; four driving threads, one per
request, measured the same 0.80 TMAC/s). At 8192² an attempt is only 16 rounds of 4 launches, so filling and
draining the queue (one launch latency, about 21 ms with 2048 rows) costs as much as the larger launch gains.

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
