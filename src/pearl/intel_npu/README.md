# Pearl on the Intel NPU

`--backend intel_npu` mines Pearl on the NPU of Intel Core Ultra processors (NPU3720 class: Meteor Lake, Arrow
Lake). The NPU runs the exact int8 GEMM and the per-tile milestone XOR; the host builds the noisy matrices and
runs the BLAKE3 jackpot. The design comes from `gemm_xor_benchmark/npu` (Case 3.4 on the NPU), which has the full
measurements and the reasoning behind each choice.

Arrow Lake Core Ultra 9 275HX: about **0.73 TMAC/s** (730 GH/s), mock shares verify at 8192² and 131072².

## How it works

| Piece | Where | Role |
|-------|-------|------|
| Launch graph | `cp_npu_graph.cpp` | One OpenVINO model for a 512 × 512 × 4096 launch: per K panel of 128, an int32 GEMM on the DPU of the A panel with a 64-column block of B, then the `TileXorAcc` op (8 panels per op) keeps the running sum and XORs every 8 × 16 tile on the SHAVEs. The 8 column chains run in lockstep so each A panel leaves CMX after its last use. |
| Engine | `cp_npu_engine.cpp` | The patched compiler's switches, the compiled-graph cache (`npu_cache/`, keyed by the kernel ELFs and the compiler DLL), infer requests, NPU-visible host tensors (inputs are read in place). |
| Worker | `cp_pearl_npu_worker.cpp` | Zero-B like the CPU and Hexagon workers. Per job: noisy B built on the host and packed per 512-column block. Per attempt: the sparse signal A, its incremental hash, the seeds; a prep thread packs each 512-row panel of noisy A ahead of its launches; 4 lanes (one infer request and one host thread each) run launches back to back and check the jackpot of their own output. A hit is recomputed on the host before it becomes a share. |
| SHAVE kernel | `kernels/tile_xor_acc.3720xx*.elf` | Prebuilt hand-scheduled `TileXorAcc` kernel. Source and generator: `gemm_xor_benchmark/npu/kernels` and `tools/gen_tile_xor_asm.py` (needs MoviTools to rebuild). |
| Compiler patch | `patches/npu_compiler_all.diff` | Against `openvinotoolkit/npu_compiler` at `patches/NPU_COMPILER_COMMIT`: int32 DPU output (`NPU_EXPERIMENTAL_I32_DPU_OUTPUT=1`), the `TileXorAcc` op from the frontend to the SHAVE kernel call, loading external SHAVE kernels (`NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR`), the tile-word layout and cycle costs. The miner sets both variables itself. |

Hash tile 8 × 16 (proof layout `CP_TILE_LAYOUT_CONTIGUOUS`, the same as the CPU worker). A and B go to the NPU as
u8 holding value + 128, which the compiler folds into a cast; B is fed as rows of B^T (the form the int32 DPU path
computes exactly). One launch covers one 512-row accumulator block: the compiler merges identical slices of an input,
so a B block shared by several row blocks would stay in CMX across all of them.

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

The first start compiles the launch graph (about 10 s) into `npu_cache/` next to the executable; later starts load
it in under a second. A new kernel ELF or compiler DLL selects a new cache subdirectory.

| Environment | Default | Effect |
|-------------|---------|--------|
| `CP_NPU_LANES` | 4 | Infer requests in flight, one host thread each. 2, 4 and 8 measured the same in `cp_npu_launch_test`; with 1 the NPU idles through each launch's turnaround (about 0.3 ms per 1.46 ms launch in the benchmark). |
| `CP_NPU_CACHE_DIR` | `npu_cache` next to the exe | Compiled-graph cache root. |
| `NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR` | `kernels/intel_npu` next to the exe | Where the compiler loads `tile_xor_acc` from (set by the miner unless already set). |

## Measurements (Core Ultra 9 275HX, NPU 2 tiles)

| Run | Result |
|-----|--------|
| `cp_npu_launch_test`, 512 × 512 × 4096 launches, 4 requests | 1.46 ms per launch, 1.47–1.50 TOPS; bit-exact against the host |
| Mock, `--m 8 --n 8` (256 launches per attempt) | 0.37–0.40 s per attempt, 0.69–0.74 TMAC/s; share found, host recompute matches, proof verifies |
| Mock, default 131072² (65536 launches per attempt) | 0.73 TMAC/s; job setup 0.58 s (B seed 0.51 s, noisy B 0.06 s); share and proof verify |

Per attempt at 8192² the host spends about 0.02 s building noisy A panels and 0.1 s of jackpot (spread over the
lanes, while the NPU runs). The launch shape trades about 6% against the best multi-row-block graph of the benchmark
(1.58 TOPS at 2048 × 512 × 4096) for single-use B blocks and a 10-second compile.
