# Gen12LP fused jackpot compare bug

Fused Case 5.6 path: `--backend onednn --fused-jackpot` (IGEMM + wrapGRF + BLAKE3 + in-kernel
`digest_beats_target` on Gen12LP / XeLP, e.g. Intel UHD 770).

## Symptom

False jackpot shares under mock mining (`--mock --mock-diff 50`). GPU sets `found_flag` /
`blake3_beats[spatial]` even when the readback digest does not beat the target on CPU.

Host log (before gate):

```
[onednn] ignoring GPU jackpot claim (cpu_beat=0 on GPU digest)
```

## Correct reference logic

CPU / OpenCL reference (`cp_jackpot.hpp`, `cp_onednn_jackpot.cl`):

```cpp
inline bool digest_beats_target(const uint32_t digest[8], const uint32_t bound[8]) {
    for (int w = 7; w >= 0; --w) {
        if (digest[w] < bound[w]) return true;   // beat (digest below target)
        if (digest[w] > bound[w]) return false;  // lose
    }
    return true;  // all words equal => beat
}
```

## JIT implementation (broken on Gen12LP)

Generated in `third_party/gemmstone/generator/pieces/case5_blake3.cxx`:

```cpp
mov(1, dWord, digKeep.ud(w));
cmp(1 | gt, gtMask, dWord, bWord.ud(0));
cmp(1 | ne | f0[0], gtMask, uint32_t(0));
jmpi(1 | f0[0], labelGt);

cmp(1 | lt, ltMask, dWord, bWord.ud(0));
cmp(1 | ne | f0[0], ltMask, uint32_t(0));
jmpi(1 | f0[0], labelLt);
```

Compare masks go to GRF; `beatVal` / `resolved` updated with bitwise ops (no `jmpi` in loop).
See **Fix** section below.

Bound words arrive via kernel scalars `blake3_b0..blake3_b7` (not a separate global buffer;
Gen12LP scrambled some global bound pointer bindings).

## What is *not* wrong

Diagnostic readback (`blake3_cmp_dump` buffer, host in `case33_gemm_onednn.cpp`) shows:

| Check | Result |
|-------|--------|
| BLAKE3 digest vs CPU full-tile recompute | Match (`digest_match=yes`) |
| Bound operand at compare time vs host bound | Match (`bound_ok=yes`) |
| Digest operand at compare vs readback digest | Match (`dig_ok=yes`) |
| CPU `digest_beats_target` on readback digest | Correct (`cpu_beat=0` on false positives) |

The bug is the **branch outcome**, not digest, bound delivery, or BLAKE3.

## Example trace (word w=7, false positive)

```
  w   gpu_dig     gpu_bound   f1_gt f1_lt !gt_j !lt_j !store gpu cmp gpu act
  7  e4699691  00020000     1     0   YES    --     --   lt   BEAT
  digest(readback)=e4699691  cpu_bound=00020000  cpu cmp=gt  cpu act=lose
  dig_ok=yes  bound_ok=yes  cmp_ok=NO
  reach_store_beat=YES
```

Unsigned compare of `gpu_dig` vs `gpu_bound`: **gt → lose**. Kernel recorded **lt → BEAT**.

## Root cause (narrowed)

`f1[0]` is materialized correctly immediately after each `cmp` (via conditional `add` in
`storeFlagAtSpatial`), but **`jmpi` does not see the same flag state**:

1. **`f1_gt=1`** after gt `cmp`, yet **`!gt_j=YES`** — `jmpi(1|f1[0], labelGt)` fell through
   (jump not taken).
2. **`f1_lt=0`** after lt `cmp`, yet **`!lt_j=--`** and cmp code **2 (lt/BEAT)** — lt `jmpi`
   jumped as if `f1[0]` were still set from the prior gt `cmp`.

This pattern matches a **flag pipeline hazard on Gen12LP**: `cmp` updates `f1[0]`, the
flag-dump `add` observes the new value, but the very next `jmpi` reads a **stale** flag.

Hardcoded imm32 bounds at codegen time produced correct compare branches, which ruled out
logic errors in the compare loop structure itself.

## `blake3_cmp_dump` layout (65 u32 per tile)

| Offset | Content |
|--------|---------|
| 0–7 | Digest operand at compare (`digKeep.ud(w)`) |
| 8–15 | Bound operand at compare (`bWord`) |
| 16–23 | Path code: 1=gt/lose, 2=lt/beat, 3=eq/next, unevaluated=`0xFFFFFFFF` |
| 24–31 | `f1[0]` after gt `cmp` (0/1) |
| 32–39 | `f1[0]` after lt `cmp` (0/1) |
| 40–47 | `!gt_j`: 1 if execution fell through gt `jmpi` |
| 48–55 | `!lt_j`: 1 if execution fell through lt `jmpi` |
| 56–63 | `!store`: 1 if execution fell through unconditional `jmpi(1, labelStoreBeat)` |
| 64 | `reach_store_beat`: 1 if `labelStoreBeat` was reached |

Reproduce:

```powershell
.\cppminer.exe --backend onednn --fused-jackpot --mock --mock-diff 50 --max-nonce 1
```

Inspect `[onednn] gpu digest vs bound per-word GPU vs CPU compare` lines in stderr.

## Mitigation (in place)

`case33_gemm_onednn.cpp` readbacks digest on GPU `found_flag` and only propagates a hit when
`cp_jackpot::digest_beats_target(digest, bound)` is true on the GPU-written digest. False GPU
claims are logged and suppressed.

## Fix (branchless compare, 2026-08-30)

Gen12LP `jmpi` after `cmp` is unreliable even when compare masks are materialized in GRF
(`sync.nop` and compare-to-GRF + flag `jmpi` both failed). **Working fix:** branchless
MSB-first compare using GRF masks only — no `jmpi` in the compare loop.

```cpp
cmp(1 | gt, gtMask, dWord, bWord.ud(0));
cmp(1 | lt, ltMask, dWord, bWord.ud(0));
or_(1, wordDecided, gtMask, ltMask);
not_(1, notResolved, resolved);
and_(1, active, wordDecided, notResolved);
shr(1, beatCand, ltMask, 31);              // 1 if digest < bound
not_(1, notActive, active);
and_(1, beatVal, beatVal, notActive);
and_(1, tmp, beatCand, active);
or_(1, beatVal, beatVal, tmp);
or_(1, resolved, resolved, active);
// after all words: if !resolved, all equal => beat
not_(1, tmp, resolved);
and_(1, tmp, tmp, 1u);
or_(1, beatVal, beatVal, tmp);
```

Claim gating: `atomic add beatVal` (0 or 1) instead of unconditional `atomic add 1`, so
`beatVal==0` tiles never set `found_flag`.

Mock run (`--mock --mock-diff 50 --max-nonce 1`): `cmp_ok=yes`, `gpu_trace_beat` matches
CPU, real share found and verified — no spurious `ignoring GPU jackpot claim`.

## Coord delivery (Gen12LP, 2026-08-30)

**Working:** per-tile `blake3_beats[spatial]` unconditional stores + host scan for first
beat (`find_fused_panel_hit_`). Mock PASS with tile-aligned coords (e.g. `t_rows=32 t_cols=28384`).

**Not working in fused epilogue:**

| Approach | Result |
|----------|--------|
| Branchless blend to `out_t_rows`/`out_t_cols` | Wrong coords (`1,1` then `0,0`); used wrong constant (`case5XorSubGridM` vs `case5XorSubM` for hash multiply) |
| Branchless blend with correct `case5XorSubM/N` | Still wrong (`0,0`) |
| Single winner spatial in `found_flag[+36]` + host derive | Wrong coords (`0,16`); branchless atomic claim unreliable on Gen12LP (see `gen12_cmp_isolate`) |
| `jmpi` after win → unconditional coord store | Coords never written |
| `atomic_cmpxchg` claim | GPU hang |

**Root cause class:** Gen12LP flag/atomic claim hazard — first-hit branchless blend and `jmpi`
coord stores are unreliable inside the fused GEMM epilogue (GRF budget + execution order).
Unconditional per-tile beat stores work; host scan picks minimum spatial beat (not first atomic).

**Path to `(found, row, col)` only:** separate lightweight OpenCL jackpot kernel post-GEMM
(like `cp_onednn_jackpot.cl`), or fix claim in `gen12_cmp_isolate` then port branchless blend.

   (e.g. `cmp` into GRF + `cmp.ne` on mask, or `sel` into GRF then integer branch).
3. Use separate flag registers for gt vs lt (e.g. `f0` vs `f1`) with documented clobber rules
   after BLAKE3 (BLAKE3 may clobber `f0[0]` — compare loop already uses `f1[0]` for that reason).
4. Disable in-kernel judge on Gen12LP and use the two-kernel OpenCL judge path only.

## Related files

| File | Role |
|------|------|
| `case5_patches/gemmstone/generator/pieces/case5_blake3.cxx` | Fused BLAKE3 epilogue + compare loop (patched into vendored gemmstone) |
| `case33_gemm_onednn.cpp` | Host dump, CPU recompute, safety gate |
| `case5_gemm_launch.cpp` | Kernel args (`blake3_cmp_dump`, `blake3_b0..b7`, …) |
| `src/cpu/cp_jackpot.hpp` | Reference `digest_beats_target()` |
| `kernels/cp_onednn_jackpot.cl` | Two-kernel OpenCL reference judge |
