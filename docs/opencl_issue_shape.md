# OpenCL GEMM issue shape

Scalar / packed paths in `src/opencl/kernels/case33_gemm_xor.cl`. Select with `--ocl-issue`.

| Mode | Flag | Inner loop |
|------|------|------------|
| **packed** (default) | `--ocl-issue packed` | Per-C `dot4` / DP4A into `acc[j,i]` |
| **broadcast** | `--ocl-issue broadcast` | `cpm += aval * bscalar` (float4 `mad`, flush each KR) |

`broadcast` forces a scalar kernel build (DPI off) so the nest actually runs. AMD `sdot4` / `dot_acc_sat` stay on **packed**.

## packed (current)

One packed K-reduction per C scalar:

```text
for each of NR columns:
  for each of MR rows:
    acc[j,i] += dot4(A[i][k:k+4], B[j][k:k+4])
```

## broadcast

Keep a vector C tile. Load A rows at one K, broadcast one B scalar (CLBlast GEMMK=0 style):

```text
cvec += avec * bscalar    // mad(float4, float, float4)
```

| Parameter | Value | Role |
|-----------|------:|------|
| `VWM` | 4 | vector along M (`aval`) |
| `VWN` | 4 | four N columns; each B lane is a scalar broadcast |
| K-step | 4 | packed `char4` lanes |

```text
for k in 0..3:
  for mi in 0 .. MR/VWM:
    aval = 4 A rows at this k
    for ni in 0 .. NR/VWN:
      cpm[(j0+*)*(MR/VWM)+mi] += aval * b[j0+*][k]
```

`aval` is reused across N. Float panel is `convert_int4`’d into `acc[]` every KR=128 (exact: |sum|<2²¹).

## Compare

```bash
# default packed
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 --dev --period-batch 32

# B-broadcast issue
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 --dev --period-batch 32 \
  --ocl-issue broadcast
```

Wait for `[ocl] attempt timing: … GMAC/s`. Backend line prints `broadcast` when active.
