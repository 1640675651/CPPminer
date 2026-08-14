# OpenCL scalar GEMM notes

Issue shape (K-reduction per C vs `cpm += aval * bscalar`): [`opencl_issue_shape.md`](opencl_issue_shape.md).

Haswell Gen7 has no DP4A. Default scalar `cpm` uses float `mad`; `--ocl-cpm-int` uses int32. Flush `cpm` to int32 `acc[]` every KR=128 (float integers are exact only to 2²⁴; one panel max |sum| ≈ 2.06e6).

LDS (`--ocl-lds`) did not raise GT1 rate (UMA 8 KiB cache). In-kernel BLAKE3 is ~0.15% of MACs/tile.

## Measured (Beignet, Haswell GT1 Desktop)

`--cpu-gen`, **no** `--ocl-lds`. Scan rates are **`attempt timing`** (tiles / scan_sec).

| Tile | Inner loop | Private | Scan |
|------|------------|--------:|-----:|
| 8×8 `--dev` | Scalar `int8` `dot4` | 384 B/WI | ~1.1 GMAC/s |
| 8×8 `--dev` | `float4` `dot` (K-reduction per C element) | 384 B/WI | ~1.9 GMAC/s |
| 8×8 `--dev` | CLBlast `cpm += aval * bscalar` | 128 B/WI | ~9.9 GMAC/s |
| **4×8 production** | **CLBlast `cpm`** | **128 B/WI** | **~28 GMAC/s** |

4×8 is one `float4` along M (`MR/VWM = 1`) and 512 WI/macro — the same 4×8-float C tile CLBlast uses on Haswell. 8×8 keeps two `float4`s along M and 256 WI/macro. Production `m=n=131072`, `--period-batch 128`, `--ocl-tile 4x8`: progress ~26–28 GMAC/s, `attempt timing` **27.84 GMAC/s**, mock share **verify OK**.

CLBlast SGEMM on this chip plateaus around ~20 GMAC/s (float FMA counted as 2 FLOP). 4×8 fused int8 GEMM is in that ballpark (slightly above on MAC/s).

### Reproduce

```bash
# 8×8 baseline (~10 GMAC/s on --dev)
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 8x8 \
  --period-batch 32 --max-nonce 1 --dev

# 4×8 (Haswell GT1: ~28 GMAC/s production)
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 \
  --period-batch 128
```

Wait for `[ocl] attempt timing: … X GMAC/s`. In-scan **progress** lines are `tiles / wall time since scan start` and can decay during a long kernel.

## What not to chase next on this iGPU

- **`--ocl-lds`:** off by default; no win on this UMA GT1.
- **BLAKE3 out of the kernel** for FLOPs (runtime is tiny). Register footprint of the inlined compress may still matter for SIMD width; private is 128 B/WI with the `cpm` layout without splitting the kernel.

On this GT1, **`--ocl-tile 4x8`** is the second lever after issue shape: it matches CLBlast’s 4×8-float WI tile (one `float4` along M) and ~3× the 8×8 `cpm` rate. 8×8 remains the default for iGPUs that do not want 512 WI/macro.

## Integer cpm (`--ocl-cpm-int`)

Same loop nest (`cpm += aval * bscalar`). Inputs stay packed `char`; the C tile is `int4` instead of `float4 mad`. Products do not fit in `char` (`|a*b| ≤ 16129`), so the accumulator is int32, not wrapping int8. Default remains float (Haswell GT1 ~28 GMAC/s). Compare `attempt timing`; backend line prints `clblast cpm int`.

```bash
./cppminer --backend opencl --mock --cpu-gen --ocl-tile 4x8 \
  --period-batch 128 --ocl-cpm-int
```

## Code

| Piece | Where |
|-------|--------|
| Issue shape | [`opencl_issue_shape.md`](opencl_issue_shape.md) |
| Inner MAD | `case32_cpm_kgroup` in `src/opencl/kernels/case33_gemm_xor.cl` |
| Integer cpm | `--ocl-cpm-int` / `CASE32_CPM_INT` |
| Packed-dot fallback | `CASE32_PACKED_DOT` (asm / builtin sdot4 / `dot_acc_sat`) |
| Build flag | default scalar path; DPI probes still tried first |
| Private / local sizes | `docs/memory.md` |
| Hashrate formulas | `docs/hashrate_calculation.md` |
