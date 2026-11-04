# Quantus OpenCL backend

This document describes how `--algo quantus --backend opencl` picks its kernel on each
device, what its start-up log lines mean, and the environment variables that override
its choices. The kernel is `src/qpow/opencl/kernels/qpow_mining.cl` and the host side is
`src/qpow/opencl/cp_qpow_opencl_worker.cpp`.

## What the kernel computes

Quantus hashes a 64-byte nonce with Poseidon2 over the Goldilocks field
(p = 2^64 - 2^32 + 1, width 12). The miner does not hash consecutive nonces. It
counts 64-byte values and hashes, for each one, the nonce that `qpow/nonce_line.hpp`
maps it to. Along one such "nonce line" only two lanes of the state change after the
first linear layer, so the host folds the midstate and the first round into 14
constants per launch, and the kernel starts with two S-boxes instead of twelve. The
CPU backend searches the same nonce lines.

A work-item computes only the first 8 bytes of the hash. A nonce whose first 8 bytes
are at most the target's becomes a candidate, up to 15 per launch. The host
re-hashes every candidate with the reference code before it submits a share, so a
wrong GPU result can lose a share but never produce an invalid one.

## Start-up: build, self-test, probe

At start-up the worker builds several variants of the kernel, checks each one, times
it, and mines with the fastest. A typical log on an Arc B580:

```
[qpow-ocl] probe mul=3 red=1 local=64: 43.09 MH/s
[qpow-ocl] probe mul=3 red=1 local=256: 43.06 MH/s
[qpow-ocl] probe mul=1 red=1 local=64: 37.17 MH/s
[qpow-ocl] probe mul=1 red=1 local=256: 37.10 MH/s
[qpow-ocl] probe mul=3 red=2 local=64: 40.82 MH/s
[qpow-ocl] probe mul=3 red=3 local=64: 42.95 MH/s
[qpow-ocl] device[0]: Intel(R) Arc(TM) B580 Graphics (discrete) CUs=160 kernel mul=3 red=1 local=64 batch=auto self-test ok
```

Each `probe` line is one variant:

- `mul` is how the 64x64 -> 128-bit product is computed (kernel option `QV_MUL`).
- `red` is how that product is reduced back to 64 bits (kernel option `QV_RED`).
- `local` is the work-group size.
- The rate is the kernel alone. A short run sizes the probe to about 50 ms of work,
  and the best of three runs counts. Mining also pays for host work, so it runs a
  little slower than the probe (B580: 43.1 MH/s probe, 40.0 MH/s mining).

| Value | `mul` (product) | `red` (reduction) |
|---|---|---|
| 1 | `ulong` multiply plus `mul_hi` | one carry fold (exact except a ~2^-64 corner case) |
| 2 | four 32x32 products with explicit carries | signed form with shift-derived borrows (exact) |
| 3 | carry-free chain of 32x32+64 multiply-adds (one `mad_u64_u32` each on AMD GCN5+/RDNA) | form 1 in gfx12 inline assembly (AMD RDNA4); on other devices it builds as form 1 |

The search runs in two steps:

1. `mul` 3, then `mul` 1, each with `red=1`, at work-group sizes 64 and 256.
2. `red` 2 and 3 for the best `mul`, at the best work-group size only.

`mul=2` is never probed; only `CP_QPOW_OCL_MUL=2` selects it. A variant must pass the
self-test before it is timed. One that fails to build or to match prints an error
instead of a probe line. The final `device[...]` line names the variant that will mine
and whether the launch size is automatic (`batch=auto`) or fixed by `--batch-size`.

Probe rates vary by a few percent between runs, so variants within about 1% of each
other can swap from one start to the next. On the A770, `red=2` has measured anywhere
from 106 to 126 MH/s.

### Self-test

Each variant hashes 4096 nonces from a fixed header and counter. The host compares
128 of them with the reference code. Any mismatch rejects the variant and prints

```
[qpow-ocl] self-test mismatch (mul=M red=R) at nonce +N: got ..., want ...
```

If no variant passes, the device does not mine:
`no kernel variant passed the self-test, not mining on this device`. The inputs are
fixed, so a correct kernel gives the same answer on every run. A mismatch points to a
compiler problem or an unstable GPU clock.

### Build fallback for the carry form

By default the kernel detects 64-bit carries with `__builtin_add_overflow`
(`QV_OVF=1`), which is 3-4% faster on Intel and NVIDIA. The UHD 770's compiler turns
it into `__spirv_IAddCarry` and then fails with
`undefined reference to _Z17__spirv_IAddCarryll`. When a build fails, the worker
prints the driver's build log once, retries with `-DQV_OVF=0` (carry by compare), and
uses that form for every later variant:

```
[qpow-ocl] retrying the build with the compare-based carry (-DQV_OVF=0)
```

There is no retry if `CP_QPOW_OCL_OPTS` already sets `QV_OVF`.

## Launch size

With no `--batch-size`, each launch is sized to about 100 ms of work, starting from
the winning probe rate and then adjusted from the measured rate. The size is kept
between 65,536 and 2^30 nonces. Two launches are in flight at a time. `--batch-size N`
fixes the size at N nonces. Before automatic sizing, the default was 10^6 nonces,
far too few for a large GPU: the A770 mined at 10.7 MH/s with it and at 105.9 MH/s
with automatic sizing and the current kernel.

The two launches stay queued between search calls: each call collects the older one,
queues the next and returns, and a new job drops whatever is still queued. Every share
of a finished launch is submitted and the search continues after it, so no nonce is
hashed twice.

## Environment variables

| Variable | Effect |
|---|---|
| `CP_QPOW_OCL_MUL=1\|2\|3` | Probe only this product form in step 1 (both work-group sizes are still probed) |
| `CP_QPOW_OCL_RED=1\|2\|3` | Use this reduction form in step 1 and skip step 2 |
| `CP_QPOW_OCL_OPTS="..."` | Extra kernel build options, appended to every variant's build |

Kernel build options for `CP_QPOW_OCL_OPTS` (defaults in brackets; the full list is in
the header of `qpow_mining.cl`):

| Option | Effect |
|---|---|
| `-DQV_OVF=0\|1` | Carry by compare, or by `__builtin_add_overflow` [1 where the compiler has it] |
| `-DQV_WRED_FAST=0\|1` | Exact lazy-sum reduction, or skip its rare carry fold [1]. The fast form is wrong for about 2 hashes in a million; the host re-check filters those, so the only cost is the occasional missed share |
| `-DQV_NPW=N` | Hash N consecutive nonces per work-item [1; 2 on gfx12]. The host launches 1/N of the work-items |
| `-DQV_EXT22=0\|1` | Linear layers in lazy 96-bit sums, or in 22-bit limbs [1; 0 on gfx12] |
| `-DQV_GFMA_CHAIN=0\|1` | With `mul=3`, fold the lazy row sum into the multiply-add chain [1] |
| `-DQV_SG=N` | Intel only: require sub-group size N |

On AMD RDNA4 (device names starting with `gfx12`) the host adds `-DQV_GFX12=1`, and
also `-DQV_NPW=2` unless `CP_QPOW_OCL_OPTS` sets `QV_NPW`.

Examples:

```sh
# Fix the variant to the B580's usual winner (only the work-group size is still probed)
CP_QPOW_OCL_MUL=3 CP_QPOW_OCL_RED=1 ./cppminer --algo quantus --backend opencl ...

# Exact reduction, carry by compare
CP_QPOW_OCL_OPTS="-DQV_WRED_FAST=0 -DQV_OVF=0" ./cppminer --algo quantus --backend opencl ...
```

On Windows, set the variable first, for example `$env:CP_QPOW_OCL_RED = "2"` in
PowerShell.

## Measured rates

Mining rates against a local test pool, CPPminer `dev` with PR #29 merged:

| Device | Chosen variant | Probe | Mining |
|---|---|---|---|
| Intel Arc A770 | `mul=3 red=1 local=64` | 124-127 MH/s | 105.9 MH/s |
| Intel Arc B580 | `mul=3 red=1 local=64` | 43.1 MH/s | 40.0 MH/s |
| Intel UHD 770 | `mul=3 red=1` or `red=2` (within 1%), with `QV_OVF=0` | 4.3 MH/s | 4.37 MH/s |

The PR #29 author measured 211 MH/s on an AMD R9700 (RDNA4) with `red=3`, `QV_NPW=2`
and `QV_EXT22=0`.
