# OpenCL GEMM issue shape

Scalar path in `src/opencl/kernels/case33_gemm_xor.cl` (`case32_cpm_kgroup`). Host prints `clblast cpm` when this path is active. AMD packed-dot (`sdot4` / `dot_acc_sat`) is unchanged.

## Before

One packed K-reduction per C scalar. For an MR×NR tile that is 64 independent dots per K-step (8×8):

```text
for each of NR columns:
  for each of MR rows:
    acc[j,i] += dot4(A[i][k:k+4], B[j][k:k+4])
```

Each MAC is a 4-wide K-reduction into one C element. A is not reused across N.

## After (CLBlast GEMMK=0)

Keep a vector C tile. Load a vector of A rows at one K, broadcast one B scalar, update the whole vector:

```text
cvec += avec * bscalar
```

| Parameter | Value | Role |
|-----------|------:|------|
| `VWM` | 4 | vector along M (`aval`) |
| `VWN` | 4 | four N columns per `ni`; each B lane is a scalar broadcast |
| K-step | 4 | packed `char4` lanes, unrolled like CLBlast `KWI` |

```text
for k in 0..3:                          # packed K=4
  for mi in 0 .. MR/VWM:                # VWM=4
    aval = 4 A rows at this k
    for ni in 0 .. NR/VWN:              # VWN=4
      cpm[(j0+0)*(MR/VWM)+mi] += aval * b[j0+0][k]
      cpm[(j0+1)*(MR/VWM)+mi] += aval * b[j0+1][k]
      cpm[(j0+2)*(MR/VWM)+mi] += aval * b[j0+2][k]
      cpm[(j0+3)*(MR/VWM)+mi] += aval * b[j0+3][k]
```

`aval` is reused across N. That is the issue-shape change.
