# OpenCL GEMM issue shape

Scalar / packed paths in `src/opencl/kernels/case33_gemm_xor.cl`. Select with `--ocl-issue`.

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

## Regression note (dev vs beignet-fix)

On beignet-fix, **no-DPI always meant cpm** (`!CASE32_PACKED_DOT`). An intermediate `dev` change gated cpm behind `--ocl-issue broadcast` and made the no-DPI `#else` a slow per-C `case32_dot4` loop, also dropping `#pragma unroll` on loads. That is restored: scalar ≡ cpm again (case36 / beignet-fix).

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
