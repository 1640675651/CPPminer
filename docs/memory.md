# Memory footprint

Production dimensions (`cp_config.h`): **m = n = 131072**, **k = 4096**, **r = 128**.  
Dev (`--dev`): **m = n = 8192**.

Each full matrix (signal or coalesced prepack) is:

| Size | Production | `--dev` |
|------|------------:|--------:|
| m × k or n × k | **512 MiB** | **32 MiB** |

Host signal slots are allocated in `cp_mine_init_host_buffers()`: `h_Ap_global` (512 MiB) unless the backend proves from device witnesses (CUDA / OpenCL / oneDNN GPU prep), and `h_BpT_global` (512 MiB) only when `cp_worker_needs_host_bt()` (wgpu / CUDA `--cpu-gen`). CPU, OpenCL and oneDNN prove their all-zero B^T without a host matrix (`cp_proof_build(bt = NULL)`, or a B^T witness with no sub-roots), which caches zero-matrix sub-roots (64 KiB) per job.

## Share proof memory (all backends)

Proof building (`rust/cp-proof-ffi`) never copies a signal matrix; peak host RAM during a proof equals steady state plus well under 1 MiB.

| Path | Input | Proof-thread scratch (production) |
|------|-------|-----------------------------------|
| Host matrix (`cp_proof_build`) | borrowed `h_Ap_global` / `h_BpT_global`, hashed in place | 2048 sub-roots per matrix (64 KiB each), proven leaf chunks, ≤ 256 KiB padded copy only if a matrix ends mid-block |
| Zero B^T (`cp_proof_build` with `bt = NULL`, or an empty B^T witness) | nothing | 64 KiB zero sub-roots, cached per job (first share hashes 512 MiB of zeros, streamed) |
| Device witness (`cp_proof_build_witness`, CUDA / OpenCL / oneDNN) | A sub-roots + tile blocks from the GPU | ~0.3–0.8 MiB witness per share |
| All | base64 output | up to 512 KiB (`PLAIN_PROOF_B64_MAX`) |

Sub-roots are the CVs of aligned 256-chunk (256 KiB) subtrees, hashed in parallel with BLAKE3 SIMD. The rest of the Merkle path is rebuilt from them plus the blocks holding the proven rows. Production timing (A in place + zero B^T): ~30 ms on the first share of a job, ~16 ms after.

---

# CPU zero-B path

The default CPU worker caches **noisy B once per job** and rebuilds **noisy A each nonce**. Signal `B^T` stays zero and has no host buffer; only `h_Ap_global` is randomized per attempt.

## Per-buffer sizes

| Buffer | Bytes | Production |
|--------|------:|-----------:|
| Signal A (`h_Ap_global`) | m × k | 512 MiB |
| Signal B^T | — | not allocated (zero; proof uses cached sub-roots) |
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

## Prepack modes (`--prepack MODE`)

| Mode | CLI | Steady matrix RAM | Brief peak |
|------|-----|-------------------|------------|
| **fused** (default) | `--prepack fused` | **~1.5 GiB** | **~1.5 GiB** (no full-matrix temp) |
| **reuse** | `--prepack reuse`, `--inplace-prepack` | **~1.5 GiB** | **~2 GiB** during prepack |
| **separate** | `--prepack separate` | **~2.5 GiB** | ~2.5 GiB |

Steady-state breakdown (production):

### separate (~2.5 GiB)

```
h_Ap_global          512 MiB   signal A (per nonce)
g_zero_b.B_noisy     512 MiB   row-major noisy B (kept after prepack)
g_A_noisy            512 MiB   row-major noisy A (per nonce)
g_gemm.b_pre_        512 MiB   B scan / prepack layout
g_gemm.a_pre_        512 MiB   A scan / prepack layout
b_comp_ms_             8 MiB
─────────────────────────────
total               ~2568 MiB
```

GEMM reads `a_pre_` and `b_pre_`; row-major copies in `g_*_noisy` are redundant but still allocated.

### reuse (~1.5 GiB steady, ~2 GiB peak)

```
h_Ap_global          512 MiB
g_zero_b.B_noisy     512 MiB   B scan (after prepack+swap)
g_A_noisy            512 MiB   A scan (after prepack+swap)
b_comp_ms_             8 MiB
```

Flow: build row-major noisy → prepack into a **temporary** vector the same size as the matrix → `std::swap` with the noisy buffer → temp freed.

While prepack runs, source (row-major) and destination (temp) coexist → **+512 MiB** for that matrix for a few milliseconds (once per job for B, once per nonce for A).

### fused (~1.5 GiB steady and peak)

Same steady buffers as **reuse**, but noise injection and panel prepack are combined:

1. For each 8-row (A) or 16-column (B) tile, fuse noise into a **thread-local stripe** (~36–68 KiB).
2. Pack panels directly into the scan buffer.
3. Compute `b_comp_ms_` during B column fusion.

No full-matrix temporary. Extra memory is OpenMP thread-local stripes plus a **~128 KiB** permutation-pairs table per fused build.

**Default** and recommended for production mining.

## Transient allocations (all CPU modes)

| When | What | Size (production) |
|------|------|-------------------|
| Job start | `pearl_b_noise_seed_from_bt` | negligible |
| Job start (non-fused) | `pearl_build_noisy_b` perm pairs | ~32 KiB heap |
| Each nonce (non-fused A) | `pearl_build_noisy_a` perm pairs | ~32 KiB heap |
| Fused prepack | perm pairs in `Case33GemmXor` | ~128 KiB |
| Share found | proof: A sub-roots + leaf chunks (A hashed in place, no copy) + base64 | < 1 MiB |
| First share of a job | zero-B^T sub-roots (hash 512 MiB of zeros, streamed) | 64 KiB, cached for the job |

Peak host RAM therefore equals the steady-state figures above; a share does not add a matrix-sized allocation.

`pearl_commitment_seeds` (full A+B keyed digest) is **not** run on the zero-B CPU fast path; A noise seed comes from `pearl_a_noise_seed_from_a`.

## CPU quick reference

```text
# default: lowest steady RAM (~1 GiB scan matrices + 512 MiB signal A)
cppminer.exe --backend cpu ...

# legacy / debug (simplest, highest RAM)
cppminer.exe --backend cpu --prepack separate ...

# middle ground (1.5 GiB steady, brief 2 GiB spikes)
cppminer.exe --backend cpu --prepack reuse ...
```

At startup, `[mode]` logs print estimated matrix MiB for the active prepack mode.

---

# OpenCL zero-B path

OpenCL always handles matrix prep in the worker (`cp_opencl_worker_handles_matrix_prep`). Two modes:

| Mode | CLI | Prep | Typical footprint |
|------|-----|------|-------------------|
| **GPU prep** (default) | `--backend opencl` | Device random A + hash + fused prepack | **no host matrices + ~1.5 GiB VRAM** |
| **Host prep** | `--backend opencl --cpu-gen` | CPU noisy + coalesced prepack → H2D | **~2–2.5 GiB host + ~1 GiB VRAM** |

Scan does **not** allocate a device C matrix: GEMM, tile XOR, and jackpot are fused; host only reads a found-flag (+ coords on hit).

## Per-buffer sizes (OpenCL)

### Host

| Buffer | Bytes | Production | GPU prep | `--cpu-gen` |
|--------|------:|-----------:|:--------:|:-----------:|
| Signal A (`h_Ap_global`) | m × k | 512 MiB | no (share witness from device) | yes (each nonce) |
| Signal B^T (`h_BpT_global`, zero) | n × k | 512 MiB | no | no (proof uses cached zero sub-roots) |
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
| `d_a_subroots_` (A sub-roots kept for the share witness) | same as merkle | ~64 KiB | yes | no |
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
  (no signal matrices)
DEVICE
  a_buf_                512 MiB   GEMM A
  b_buf_                512 MiB   GEMM B (job)
  d_A_sig_              512 MiB   random A + hash source + witness blocks on hit
  d_pairs_ / merkle     ~0.1 MiB  (merkle scratch + saved A sub-roots)
─────────────────────────────────────
host                    ~0 MiB matrices
VRAM                    ~1536 MiB
combined                ~1.5 GiB
```

Flow:

1. Job: GPU builds noisy B into `b_buf_` (and keeps `d_A_sig_` capacity).
2. Each nonce: GPU random A → keyed hash (A sub-roots copied to `d_a_subroots_` before the root reduction) → fused prepack into `a_buf_` (source stays in `d_A_sig_`).
3. Scan: fused GEMM+XOR+jackpot; no full tile-XOR buffer.
4. Share: D2H only A's sub-roots (64 KiB) and the ≤ 8 blocks of `d_A_sig_` holding the tile rows (≤ 2 MiB) into a `CpShareWitness`. B^T is proven from host-cached zero sub-roots.

`g_zero_b.B_noisy` is cleared on the GPU-prep path. `h_A_scan` / `h_B_scan` are not allocated (OpenCL handles prep).

## OpenCL `--cpu-gen` (host prep)

Steady / peak (production):

```
HOST STEADY
  h_Ap_global           512 MiB
  g_zero_b.B_noisy      512 MiB
  a_pre_host_           512 MiB
  b_pre_host_           512 MiB
                        ─────
                        2048 MiB
HOST PEAK (a_noisy live during attempt)
  + a_noisy             512 MiB  → ~2560 MiB
DEVICE
  a_buf_ + b_buf_      1024 MiB   (no d_A_sig_)
─────────────────────────────────────
steady combined         ~3.0 GiB
peak combined           ~3.5 GiB
```

Matches the startup hint: `--cpu-gen` for host prep (**~1 GiB VRAM** = scan A+B only).

On share, only A is handed off (`handoff_bt=0`; no host B^T); reclaim runs every attempt because host A is rewritten each nonce.

## Share handoff and proof (OpenCL)

| Item | Behavior |
|------|----------|
| Queue depth | **1** (single ownership slot) |
| GPU prep | Enqueue a **share witness** (A sub-roots + ≤ 8 blocks); no host matrix is loaned |
| `--cpu-gen` | Hand off **A only**; proof hashes it in place, B^T from cached zero sub-roots |
| Copy | **None** — pointer move; mining waits if the next hit arrives before proof returns the buffer |
| Proof scratch | < 1 MiB on the proof thread plus the witness (see *Share proof memory*) |

Peak host matrix RAM does **not** double during proof on either path.

# oneDNN zero-B path

oneDNN reuses `Case33OclPrep` for matrix prep, so its GPU-prep path keeps A's sub-roots in `d_a_subroots_` exactly like OpenCL. The mode is chosen at init, not by `--cpu-gen` (which oneDNN ignores): GPU prep whenever the prep kernels build, otherwise a host fallback.

| Mode | Host signal | Share proof |
|------|-------------|-------------|
| **GPU prep** (default) | none | `CpShareWitness` (A sub-roots + ≤ 8 blocks of `d_A_sig_`), B^T from cached zero sub-roots |
| **Host fallback** (prep kernels failed) | `h_Ap_global` (512 MiB) + `a_host_` / `b_host_` upload staging + `g_zero_b.B_noisy` | Host A hashed in place, `bt = NULL` |

No `h_BpT_global` is allocated in either mode.

# CUDA zero-B path (default)

| Buffer | Role | Production |
|--------|------|------------|
| `d_A_sig` | Signal A (random per nonce) + D2H on hit | 512 MiB |
| `d_Ap` | Noisy A (scan) | 512 MiB |
| `d_BpT` | Noisy B (scan; built once/job, signal B imaginary/zero) | 512 MiB |
| `d_Bt_sig` | **Not allocated** on zero-B path | 0 |

**~1.5 GiB VRAM** for the three full matrices (plus small noise/seed/jackpot scratch). `--cpu-gen` still allocates `d_Bt_sig` as H2D staging.

Flow: job → `gpu_prepare_job_b` (noise-only into `d_BpT`, zero-B^T sub-roots kept on host); each nonce → random A + hash + A-side noise into `d_Ap`, A sub-roots saved on device. Share D2H copies only A's sub-roots (64 KiB) and the ≤ 8 blocks holding the tile rows; no host A/B buffers are allocated. `--cpu-gen` uses the host-matrix path instead.

## OpenCL quick reference

```text
# default: no host signal matrices + ~1.5 GiB VRAM
cppminer.exe --backend opencl ...

# host matrix gen: ~2–2.5 GiB host + ~1 GiB VRAM
cppminer.exe --backend opencl --cpu-gen ...

# small matrices for bring-up
cppminer.exe --backend opencl --dev ...
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
