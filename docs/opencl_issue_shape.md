# OpenCL GEMM issue shape

Scalar / packed paths in `src/opencl/kernels/case33_gemm_xor.cl`. Select with `--ocl-issue`.

The OpenCL register tile can be `4x4`, `4x8`, `8x8`, or `8x16`. Pearl requires
at least 32 cells per hash tile, so the `4x4` path assigns one work-item to a
semantic `4x8` hash tile. It processes the two four-column halves sequentially
with a reused 4x4 accumulator and folds both halves directly into one private
message before BLAKE3. No inter-work-item exchange is required. The `4x4`
path uses four-column packed-B groups; wider paths retain eight-column groups.

All register tiles use column-major work-item order. Within each row-tile run,
adjacent work-items share a column tile and walk consecutive row tiles, making
each B load uniform across those work-items. Packing and the existing `vload4`
loads are unchanged.

| Mode | Flag | Inner loop |
|------|------|------------|
| **auto** (default) | `--ocl-issue auto` | DPI if available, else CLBlast **cpm** (beignet-fix) |
| **broadcast** | `--ocl-issue broadcast` | Force cpm: `cpm += aval * bscalar` (`CASE32_NO_DPI`) |
| **packed** | `--ocl-issue packed` | Per-C `dot4` / DP4A into `acc[j,i]` |

`broadcast` / scalar fallback pass `-DCASE32_NO_DPI=1` so Intel cannot auto-enable KHR DPI and silently switch to packed dots. AMD `sdot4` / `dot_acc_sat` stay on **packed** or **auto** when DPI builds.

### cpm type (`--ocl-cpm-type`)

| Type | Flag | Acc tile |
|------|------|----------|
| **float** (default) | `--ocl-cpm-type float` | `float4 mad`, flush to int32 each KR |
| **int** | `--ocl-cpm-type int` | int8→int32 lanes, `int4` mul+add |

Only applies on the cpm nest (auto scalar fallback or `--ocl-issue broadcast`).

### LDS staging (`--ocl-lds`)

Optional `__local` A/B panel staging with work-group barriers (`CASE32_USE_LDS`). Default **off**.

On most GPUs this **regressed** scan throughput: barrier + global→local copy cost outweighed reuse. Prefer leaving it off unless a device shows a clear win in an A/B test.

## packed

```text
for each of NR columns:
  for each of MR rows:
    acc[j,i] += dot4(A[i][k:k+4], B[j][k:k+4])
```

## broadcast / cpm

```text
cvec += avec * bscalar    // mad(float4, float, float4)
```

| Parameter | Value | Role |
|-----------|------:|------|
| `VWM` | 4 | vector along M (`aval`) |
| `VWN` | 4 | four N columns; each B lane is a scalar broadcast |
| K-step | 4 | packed `char4` lanes |

## Compare

```bash
# default auto (on Intel without DPI → cpm float)
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 --dev --period-batch 32

# force cpm (same nest as beignet-fix scalar)
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 --dev --period-batch 32 \
  --ocl-issue broadcast

# force packed dots
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 --dev --period-batch 32 \
  --ocl-issue packed
```

Wait for `[ocl] attempt timing: … GMAC/s`. Look for `clblast cpm float` in the backend line and `private=… B/WI` from the kernel mem print.

Broadcast+float issue may improve performance since some GPUs are weak in int but strong in float. For example, on UHD 630, broadcast+float yields 115GH/s, packed+int yields 100GH/s, and broadcast+int yields 90GH/s.
