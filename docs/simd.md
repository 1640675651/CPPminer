# CPU SIMD paths

This document describes the CPU implementation of the Case 3.3 noisyGEMM and
XOR scan. It also explains the separate BLAKE3 dispatch used by mining and
proof generation.

`--simd` controls only the Case 3.3 CPU GEMM path. It does not select a
BLAKE3 implementation.

## Runtime selection

The portable scalar path is always compiled. Architecture-specific source files
are compiled only for their target architecture, and `auto` chooses the best
runtime-supported path.

| Target | `--simd auto` order | Runtime check |
|---|---|---|
| x86/x86-64 | AVX2, then SSSE3, then scalar | CPUID; AVX2 additionally requires OS AVX state enabled by XCR0 |
| AArch64 | NEON, then scalar | Advanced SIMD is required by the AArch64 architecture profile |
| Other targets | scalar | None |

Forcing an unavailable ISA, such as `--simd neon` on x86 or `--simd avx2` on
a CPU without AVX2, fails instead of silently selecting another path. `sse` is
an alias for `ssse3`.

The current NEON noisyGEMM kernel is AArch64-only. The scalar path remains the
baseline for 32-bit ARM and all other unsupported CPU targets.

## Shared algorithm

Every path uses the same matrix dimensions, panel ordering, milestone schedule,
and XOR result. The x86 fast paths alter A's packed byte representation as
described below:

- A microtile is 8 rows by 16 columns.
- The inner dimension is processed in KR panels. One panel is one milestone.
- Each path keeps a cumulative `int32` result tile and XOR-folds its 128 cells
  after each milestone.
- The scalar output is the correctness reference. `--simd-test` compares every
  compiled and runtime-supported SIMD path against it.

The paths differ in their multiply-accumulate instructions and in how many
rows and columns stay in vector registers at one time.

## NoisyGEMM comparison

| Path | Source | Arithmetic | Register tile | Notes |
|---|---|---|---|---|
| Scalar | `case33_gemm_xor.cpp` | Exact signed `int8 * int8 -> int32` | Scalar loops over 8x16 | Portable reference; no intrinsics |
| SSSE3 | `case33_gemm_xor_ssse3.cpp` | Fast unsigned/signed byte multiply plus compensation, or signed emulation for exact mode | Selectable 4x8, 8x8, or 4x16 | Uses `pmaddubsw` and `pmaddwd` |
| AVX2 | `case33_gemm_xor_avx2.cpp` | Fast unsigned/signed byte multiply plus compensation, or signed emulation for exact mode | 8x16 | Uses 256-bit byte multiply-add operations |
| NEON | `case33_gemm_xor_neon.cpp` | Exact signed widening `int8 -> int16`, then `int16 * int16 -> int32` | 8x16 | Uses `vmlal_s16`; baseline AArch64 NEON only |

### x86 byte-dot conversion

The x86 kernels use `_mm_maddubs_epi16` or `_mm256_maddubs_epi16`. Those
instructions require their first byte operand to be unsigned and their second
to be signed. In the default fast mode, the A input is represented as
`uint8(A + 128)`, and the result applies this correction for every output
column and milestone:

```text
sum(A * B) = sum((A + 128) * B) - 128 * sum(B)
```

The prepack stage adds 128 to A, and the B preparation computes the per-column
compensation. This work is only enabled for the AVX2 and SSSE3 paths.

### NEON signed widening arithmetic

NEON does not need the x86 conversion. It sign-extends eight A bytes to
`int16`, broadcasts one signed B byte as `int16`, and accumulates four lanes at
a time:

```cpp
lo = vmlal_s16(lo, vget_low_s16(a), vget_low_s16(b));
hi = vmlal_s16(hi, vget_high_s16(a), vget_high_s16(b));
```

Together, `lo` and `hi` cover all eight output rows for one output column.
This is exact signed arithmetic and supports `INT8_MIN`; no `A + 128`
prepacking or compensation is performed for NEON.

`vmlal_s16` can look narrower than a 128-bit NEON operation. Its accumulator
is a 128-bit `int32x4_t`, but each multiplicand is a 64-bit `int16x4_t`:

```text
acc[0..3] += a[0..3] * b[0..3]
```

Four 16-bit products widen into four 32-bit accumulator lanes, which already
occupy 128 bits. The kernel starts with 128-bit `int16x8_t` A and B vectors,
then uses `vget_low_s16` and `vget_high_s16` to issue two `vmlal_s16`
instructions. Eight 32-bit results require 256 bits, represented by the two
independent 128-bit accumulators, `lo` and `hi`.

It is vectorized arithmetic, but not an ARM byte-dot-product instruction. A
future DotProd implementation could add a separate runtime chain:

```text
DotProd -> NEON widening -> scalar
```

That must remain separate because DotProd is optional, while NEON is baseline
on AArch64.

## Build isolation

ISA-specific noisyGEMM translation units receive ISA options only on their own
source files. The rest of the program remains at the compiler's portable
baseline.

| Source | GCC/Clang option | MSVC option |
|---|---|---|
| SSSE3 kernel | `-mssse3` | Intrinsics are isolated in its source file |
| AVX2 kernel | `-mavx2` | `/arch:AVX2` |
| BLAKE3 SSE2/SSE4.1/AVX2/AVX-512 kernels | Per-source ISA options | Per-source `/arch:` options |

The SSSE3 source also uses a GCC/Clang `target("ssse3")` attribute on its
intrinsic helpers. This prevents an always-inline intrinsic from being inlined
into baseline code without the required target feature.

## BLAKE3 SIMD

The miner's C BLAKE3 library has its own runtime dispatcher and is independent
of the noisyGEMM `--simd` setting.

| Target | BLAKE3 chain |
|---|---|
| x86/x86-64 | AVX-512, AVX2, SSE4.1, SSE2, portable |
| AArch64 | NEON, portable |
| Other targets | portable |

The x86 BLAKE3 kernels are built with per-source ISA flags. BLAKE3 performs
its own CPUID and OS-state checks before calling them.

The proof FFI is a Rust static library that also includes BLAKE3. On AArch64,
CPPminer's C BLAKE3 symbols are prefixed with `cp_blake3_` so the mining C
implementation and Rust proof implementation, including their NEON kernels,
can coexist in one executable without duplicate symbols.

## Validation and benchmarking

Run the parity test after changing a kernel or its packing logic:

```bash
./cppminer --wallet test --simd-test
```

It runs scalar first, then compares each available SIMD implementation against
the scalar XOR output in both configured integer modes.

For performance comparisons, keep the same thread count, matrix dimensions,
CPU frequency policy, and job input. A speedup of several times from scalar to
NEON or x86 SIMD is expected when noisyGEMM dominates the reported MAC/s rate.
The scalar path is a portable correctness baseline, not a hand-tuned scalar
microkernel.
