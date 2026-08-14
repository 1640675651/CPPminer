# Memory footprint

Production dimensions (`cp_config.h`): **m = n = 131072**, **k = 4096**, **r = 256**.  
Dev (`--dev`): **m = n = 8192**.

Each full matrix (signal or coalesced prepack) is:

| Size | Production | `--dev` |
|------|------------:|--------:|
| m × k or n × k | **512 MiB** | **32 MiB** |

Host signal slots `h_Ap_global` / `h_BpT_global` are always allocated in `cp_mine_init_host_buffers()` (**1 GiB** production) for every backend.

---

# CPU zero-B path

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

## Transient allocations (all CPU modes)

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

## CPU quick reference

```text
# lowest steady RAM (~2 GiB matrices + 1 GiB signal)
cpminer.exe --backend cpu --prepack fused ...

# legacy / debug (simplest, highest RAM)
cpminer.exe --backend cpu --prepack separate ...

# middle ground (2 GiB steady, brief 2.5 GiB spikes)
cpminer.exe --backend cpu --prepack reuse ...
```

At startup, `[mode]` logs print estimated matrix MiB for the active prepack mode.

---

# OpenCL zero-B path

OpenCL always handles matrix prep in the worker (`cp_opencl_worker_handles_matrix_prep`). Two modes:

| Mode | CLI | Prep | Typical footprint |
|------|-----|------|-------------------|
| **GPU prep** (default) | `--backend opencl` | Device random A + hash + fused prepack | **~1 GiB host + ~1.5 GiB VRAM** |
| **Host prep** | `--backend opencl --cpu-gen` | CPU noisy + coalesced prepack → H2D | **~2.5–3 GiB host + ~1 GiB VRAM** |

Scan does **not** allocate a device C matrix: GEMM, tile XOR, and jackpot are fused; host only reads a found-flag (+ coords on hit).

## Per-buffer sizes (OpenCL)

### Host

| Buffer | Bytes | Production | GPU prep | `--cpu-gen` |
|--------|------:|-----------:|:--------:|:-----------:|
| Signal A (`h_Ap_global`) | m × k | 512 MiB | yes (filled on share D2H) | yes (each nonce) |
| Signal B^T (`h_BpT_global`, zero) | n × k | 512 MiB | yes (zeros; not handed off) | yes |
| `g_zero_b.B_noisy` | n × k | 512 MiB | cleared / unused | yes (once/job) |
| `Case33GemmOcl::a_pre_host_` | m × k | 512 MiB | no | yes (after prepack) |
| `Case33GemmOcl::b_pre_host_` | n × k | 512 MiB | no | yes (after prepack) |
| Transient `a_noisy` (attempt) | m × k | 512 MiB | no | peak only |

### Device (`Case33GemmOcl` + `Case33OclPrep`)

| Buffer | Bytes | Production | GPU prep | `--cpu-gen` |
|--------|------:|-----------:|:--------:|:-----------:|
| `a_buf_` (coalesced A) | m × k | 512 MiB | yes | yes |
| `b_buf_` (coalesced B) | n × k | 512 MiB | yes | yes |
| `d_A_sig_` (device signal A) | m × k | 512 MiB | yes | no |
| `d_pairs_` | K × 2 × 4 | 32 KiB | yes | yes (prep init) |
| `d_merkle_roots_` | see below | ~64 KiB | yes | no |
| Jackpot (`a_key`, `bound`, `found`, coords) | tens of bytes | ~0 | yes | yes |
| `dummy_buf_` | 4 B | ~0 | yes | yes |

Merkle workspace:

```
raw_max   = max(m, n) × K
pad_max   = ceil(raw_max / 1024) × 1024
chunks    = pad_max / 1024
merkle    = ceil(chunks / 256) × 32     (~64 KiB at production)
```

Coalesced `a_buf_` / `b_buf_` byte count equals row-major `m×k` / `n×k` (`case32_layout.hpp` macro blocking).

## OpenCL GPU prep (default)

Steady state (production):

```
HOST
  h_Ap_global           512 MiB   reclaimed / D2H only on share
  h_BpT_global          512 MiB   zeros; proof may read this (not copied)
DEVICE
  a_buf_                512 MiB   GEMM A
  b_buf_                512 MiB   GEMM B (job)
  d_A_sig_              512 MiB   random A + hash source + D2H on hit
  d_pairs_ / merkle     ~0.1 MiB
─────────────────────────────────────
host                    ~1024 MiB
VRAM                    ~1536 MiB
combined                ~2.5 GiB
```

Flow:

1. Job: GPU builds noisy B into `b_buf_` (and keeps `d_A_sig_` capacity).
2. Each nonce: GPU random A → keyed hash → fused prepack into `a_buf_` (source stays in `d_A_sig_`).
3. Scan: fused GEMM+XOR+jackpot; no full tile-XOR buffer.
4. Share: reclaim host A slot → D2H `d_A_sig_` → **handoff** pointer to proof thread (no host memcpy of A). B^T stays the shared zero `h_BpT_global`.

`g_zero_b.B_noisy` is cleared on the GPU-prep path. `h_A_scan` / `h_B_scan` are not allocated (OpenCL handles prep).

## OpenCL `--cpu-gen` (host prep)

Steady / peak (production):

```
HOST STEADY
  h_Ap_global           512 MiB
  h_BpT_global          512 MiB
  g_zero_b.B_noisy      512 MiB
  a_pre_host_           512 MiB
  b_pre_host_           512 MiB
                        ─────
                        2560 MiB
HOST PEAK (a_noisy live during attempt)
  + a_noisy             512 MiB  → ~3072 MiB
DEVICE
  a_buf_ + b_buf_      1024 MiB   (no d_A_sig_)
─────────────────────────────────────
steady combined         ~3.5 GiB
peak combined           ~4.0 GiB
```

Matches the startup hint: `--cpu-gen` for host prep (**~1 GiB VRAM** = scan A+B only).

On share, both A and B^T may be handed off (`handoff_bt=1`); reclaim runs every attempt because host A is rewritten each nonce.

## Share handoff and proof (OpenCL)

| Item | Behavior |
|------|----------|
| Queue depth | **1** (single ownership slot) |
| GPU prep | Hand off **A only**; proof uses zero `h_BpT_global` for B |
| `--cpu-gen` | Hand off **A and B^T** |
| Copy | **None** — pointer move; mining waits if the next hit arrives before proof returns the buffer |
| Proof scratch | up to **512 KiB** `PLAIN_PROOF_B64_MAX` on the proof thread |

Peak host matrix RAM does **not** double during proof on the GPU-prep path.

## OpenCL quick reference

```text
# default: ~1 GiB host signal + ~1.5 GiB VRAM
cpminer.exe --backend opencl ...

# host matrix gen: ~2.5–3 GiB host + ~1 GiB VRAM
cpminer.exe --backend opencl --cpu-gen ...

# small matrices for bring-up
cpminer.exe --backend opencl --dev ...
```

## Code map (OpenCL)

| Area | Location |
|------|----------|
| Host signal / handoff flags | `src/common/cp_mine.cpp` |
| Worker GPU vs `--cpu-gen` | `src/opencl/cp_opencl_worker.cpp` |
| Scan buffers `a_buf_` / `b_buf_` | `src/opencl/case33_gemm_ocl.cpp` |
| Prep `d_A_sig_` / merkle / pairs | `src/opencl/case33_ocl_prep.cpp` |
| Layout / block sizes | `src/opencl/case32_layout.hpp` |
| Share ownership | `src/common/cp_share_queue.cpp` |

## Fused GEMM kernel private memory

Per-work-item private use in `case33_macro_gemm_xor` (`case33_gemm_xor.cl`, mining /
`fuse_jackpot=1`). Reported via `CL_KERNEL_PRIVATE_MEM_SIZE` after build.

### Source arrays (tile-dependent)

| Array | 8×8 (`--ocl-tile 8x8`) | 8×16 (`--ocl-tile 8x16`) |
|-------|------------------------:|-------------------------:|
| `acc[NR×MR]` (int32, flushed each KR) | 256 B | 512 B |
| `cpm[NR × MR/VWM]` (`float4`, scalar path) | 256 B | 512 B |
| `msg[16]` (online milestone fold) | 64 B | 64 B |
| `a_pack[MR]` / `b_pack[NR]` | 32+32 B | 32+64 B |
| `digest[8]` | 32 B | 32 B |

Milestone XORs fold into `msg[16]` during the GEMM loop (no separate `ms_xor[32]` buffer).

Inlined `b3_compress64` adds ~192 B of stack (`v[16]`, `m[16]`, `t[16]`). Beignet often
reserves that frame for the whole kernel even though BLAKE3 runs only after GEMM.

Issue shape: [`opencl_issue_shape.md`](opencl_issue_shape.md). Rates and tile notes: [`opencl_gemm.md`](opencl_gemm.md).

### Measured on Beignet (Intel Haswell GT1, scalar build)

| Tile | Work-items / 128×128 macro | Private / WI | Notes |
|------|---------------------------:|-------------:|-------|
| **4×8** | 512 | **128 B** | `--ocl-tile 4x8`; one `float4` along M; ~28 GMAC/s production on GT1 |
| 8×8 | 256 | **128 B** (was 384 with per-element `dot4`; 1152 with `ms_xor[32]`) | CLBlast `cpm` issue |
| 8×16 | 128 | **1152 B** | `acc[128]` spill dominates |

`--ocl-lds` adds work-group local A/B panels (`LDS_KWG` default 16):

```text
lds_a  = MICRO_M × (LDS_KWG/4) × KG_BYTES_A
lds_b  = MICRO_N × (LDS_KWG/4) × KG_SLICE_B
```

For 8×8 / KWG=16 this is 2 KiB + 2 KiB per work-group (not per WI). Measured on
Beignet (Haswell GT1, `--ocl-lds`): `CL_KERNEL_LOCAL_MEM_SIZE` = **4100 B/WG**,
private stays **128 B/WI** with the `cpm` inner loop.
