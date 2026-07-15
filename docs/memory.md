# Memory footprint (CPU zero-B path)

Production dimensions (`cp_config.h`): **m = n = 131072**, **k = 4096**, **r = 256**.  
Dev (`--dev`): **m = n = 8192**.

## Per-buffer sizes

| Buffer | Bytes | Production |
|--------|------:|-----------:|
| Signal A (`h_Ap_global`) | m × k | 512 MiB |
| Signal B^T (`h_BpT_global`, zero for zero-B) | n × k | 512 MiB |
| Noisy / scan A (`g_A_noisy`) | m × k | 512 MiB |
| Noisy / scan B (`g_zero_b.B_noisy`) | n × k | 512 MiB |
| Prepack A (`g_gemm.a_pre_`, separate mode only) | m × k | 512 MiB |
| Prepack B (`g_gemm.b_pre_`, separate mode only) | n × k | 512 MiB |
| B u8s8 compensation (`b_comp_ms_`) | 16 × n × 4 | 8 MiB |

Prepack layout occupies the same number of bytes as row-major (`m×k` for A, `n×k` for B^T column-major storage).

General formulas:

```
|A|  = m × k
|B|  = n × k
|b_comp_ms_| = 64 × n   (16 milestones × int32 per column)
```

With `--dev` (8192²), each matrix buffer is **32 MiB** instead of 512 MiB.

## CPU zero-B layout

The default CPU worker caches **noisy B once per job** and rebuilds **noisy A each nonce**. Signal `B^T` stays zero; only `h_Ap_global` is randomized per attempt.

Persistent for the process lifetime:

- `h_Ap_global`, `h_BpT_global` — allocated in `cp_mine_init_host_buffers()`

Per job / per attempt (in `cp_cpu_worker.cpp` + `Case33GemmXor`):

- `g_zero_b.B_noisy` — B scan buffer (or transient row-major before prepack)
- `g_A_noisy` — A scan buffer (or transient row-major before prepack)
- `g_gemm.a_pre_` / `g_gemm.b_pre_` — only in **separate** prepack mode
- `g_gemm.b_comp_ms_` — always (FastU8S8 path)

## Prepack modes (`--prepack MODE`)

| Mode | CLI | Steady matrix RAM | Brief peak |
|------|-----|-------------------|------------|
| **separate** (default) | `--prepack separate` | **~3 GiB** | ~3 GiB |
| **reuse** | `--prepack reuse`, `--inplace-prepack` | **~2 GiB** | **~2.5 GiB** during prepack |
| **fused** | `--prepack fused` | **~2 GiB** | **~2 GiB** (no full-matrix temp) |

Steady-state breakdown (production):

### separate (~3 GiB)

```
h_Ap_global          512 MiB   signal A (per nonce)
h_BpT_global         512 MiB   signal B^T = 0
g_zero_b.B_noisy     512 MiB   row-major noisy B (kept after prepack)
g_A_noisy            512 MiB   row-major noisy A (per nonce)
g_gemm.b_pre_        512 MiB   B scan / prepack layout
g_gemm.a_pre_        512 MiB   A scan / prepack layout
b_comp_ms_             8 MiB
─────────────────────────────
total               ~3072 MiB
```

GEMM reads `a_pre_` and `b_pre_`; row-major copies in `g_*_noisy` are redundant but still allocated.

### reuse (~2 GiB steady, ~2.5 GiB peak)

```
h_Ap_global          512 MiB
h_BpT_global         512 MiB
g_zero_b.B_noisy     512 MiB   B scan (after prepack+swap)
g_A_noisy            512 MiB   A scan (after prepack+swap)
b_comp_ms_             8 MiB
```

Flow: build row-major noisy → prepack into a **temporary** vector the same size as the matrix → `std::swap` with the noisy buffer → temp freed.

While prepack runs, source (row-major) and destination (temp) coexist → **+512 MiB** for that matrix for a few milliseconds (once per job for B, once per nonce for A).

### fused (~2 GiB steady and peak)

Same steady buffers as **reuse**, but noise injection and panel prepack are combined:

1. For each 8-row (A) or 16-column (B) tile, fuse noise into a **thread-local stripe** (~36–68 KiB).
2. Pack panels directly into the scan buffer.
3. Compute `b_comp_ms_` during B column fusion.

No full-matrix temporary. Extra memory is OpenMP thread-local stripes plus a **~128 KiB** permutation-pairs table per fused build.

**Recommended** for production mining when RAM is tight.

## Transient allocations (all modes)

| When | What | Size (production) |
|------|------|-------------------|
| Job start | `pearl_b_noise_seed_from_bt` | negligible |
| Job start (non-fused) | `pearl_build_noisy_b` perm pairs | ~32 KiB heap |
| Each nonce (non-fused A) | `pearl_build_noisy_a` perm pairs | ~32 KiB heap |
| Fused prepack | perm pairs in `Case33GemmXor` | ~128 KiB |
| Share found | proof buffer | up to 512 KiB (`PLAIN_PROOF_B64_MAX`) |

`pearl_commitment_seeds` (full A+B keyed digest) is **not** run on the zero-B CPU fast path; A noise seed comes from `pearl_a_noise_seed_from_a`.

## Non–zero-B / legacy host path

If matrix prep is not handled by the worker (`--cpu-gen` or non-CPU backend with host matrices), `cp_mine.cpp` may also allocate:

- `h_A_scan`, `h_B_scan` — another **m×k + n×k** if host noisy matrices are materialized

CUDA backend uses device buffers (not covered here); host signal buffers `h_Ap_global` / `h_BpT_global` are still allocated.

## Quick reference

```text
# lowest steady RAM (~2 GiB matrices + 1 GiB signal)
cpminer.exe --backend cpu --prepack fused ...

# legacy / debug (simplest, highest RAM)
cpminer.exe --backend cpu --prepack separate ...

# middle ground (2 GiB steady, brief 2.5 GiB spikes)
cpminer.exe --backend cpu --prepack reuse ...
```

At startup, `[mode]` logs print estimated matrix MiB for the active prepack mode.
