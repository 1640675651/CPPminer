# Plain proof: noise and proof dependency chains

This document describes how CPminer derives **pearl noise** for mining scans and what data is required to build a **plain_proof** share. It mirrors the reference logic in `src/common/cp_noise.c` and `rust/cp-proof-ffi` (zk-pow–compatible).

Production dimensions (unless `--dev`): `m = n = 131072`, `k = 4096`, `rank r = 256`.

---

## 1. Job key and commitments

Everything noise- and proof-related is anchored on the **job key** and **signal** matrix commitments.

### `job_key`

```
job_key = BLAKE3(header || mining_config)
```

| Input | Size | Source |
|--------|------|--------|
| `header` | 76 bytes | Pool `mining.notify` (incomplete block header; `nbits` at bytes 72–75 LE) |
| `mining_config` | 52 bytes | `PEARL_SCATTERED_CONFIG`, `PEARL_CONTIGUOUS_CONFIG`, or `PEARL_CUTLASS_CONFIG` in `cp_noise.c` |

`pearl_job_key()` and `cp-proof-ffi` use the same concatenation. The config must match the worker **tile layout** (CUDA scattered, CPU contiguous, CUTLASS fused).

### Signal matrices (miner-chosen)

Signal matrices are the **unnoised** A and B^T the miner used for the attempt (not the noisy matrices scanned by GEMM). The protocol does **not** derive them from the block header or stratum nonce. The miner searches over valid `(A, B^T)` pairs until a tile hits the jackpot target; the share proves Merkle strips of whatever matrices were committed.

Each entry must be in `[-64, 63]` (same range zk-pow uses for random matrix generation). Zero is valid.

**How CPminer fills them:**

| Backend | Signal A / B^T source | Proof on share |
|---------|------------------------|----------------|
| **CUDA** (default) | GPU random (`cp_gen_random_matrix_kernel`; per-attempt CSPRNG via `cp_random_u64`) | `cudaMemcpy` from `d_A_sig` / `d_Bt_sig` → `h_Ap_global` / `h_BpT_global` |
| **CPU** (default) | Zero signal `B^T`; random A per attempt (`pearl_generate_random_a` from **CSPRNG** bytes, not header/nonce) | `h_BpT_global = 0`, random `h_Ap_global` |
| **OpenCL** (default) | Same zero-B strategy; A seed from **CSPRNG** (`cp_random_bytes`) into GPU `ocl_gen_random_matrix` | D2H `d_A_sig_` on share |

The zk-pow reference miner (`third_party/zk-pow/src/ffi/mine.rs`) also uses independent random A/B per attempt. `pearl_generate_ab()` is a **CPminer CPU convenience**, not a protocol rule.

### Signal matrix hashes

```
hash_a = BLAKE3_keyed(pad_1024(A_row_major),  job_key)
hash_b = BLAKE3_keyed(pad_1024(B^T_row_major), job_key)
```

Padding zero-fills each matrix to the next multiple of 1024 bytes before hashing (`padded_chunk_len` in `cp_noise.c`).

### Noise seeds (standard, non-MoE)

```
b_noise_seed = BLAKE3(job_key || hash_b)
a_noise_seed = BLAKE3(b_noise_seed || hash_a)
```

In zk-pow notation: `(b_noise_seed, a_noise_seed)` is the **commitment hash** pair. MoE jobs replace `hash_a` with `hash_activations` (routing-dependent); CPminer’s plain path uses the standard chain above.

**Implication:** `E_AL`, `E_AR`, `E_BL`, and `E_BR` are **not** functions of `job_key` alone. They depend on `job_key` **and** the committed signal matrices (via `hash_a` / `hash_b`). A new random A (CUDA) or a new `ab_seed` (CPU) → new hashes → new noise.

---

## 2. Noise generation dependency chain

High-level flow:

```
header + mining_config
        → job_key
        → hash_a, hash_b (signal A, B^T)
        → a_noise_seed, b_noise_seed
        → E_AL, E_AR, E_BL, E_BR
        → noise rows/cols
        → noisy = signal + noise   (matrices scanned by GEMM/jackpot)
```

Jackpot verification recomputes the same noise from **proof strips + job_key** (zk-pow `compute_noise`), using `(signal + noise)` in **i32** — equivalent to noisy int8 when sums stay in `[-128, 127]`.

### Fixed labels and constants

| Constant | CPminer (`cp_noise.c`) | Role |
|----------|--------------------------|------|
| `PEARL_SEED_LABEL_A` | `"A_tensor"` (32-byte padded) | Domain separator for A-side noise PRF |
| `PEARL_SEED_LABEL_B` | `"B_tensor"` | Domain separator for B-side noise PRF |
| `RANGE_MASK` | `63` | Low 6 bits of each PRF byte |
| `ZERO_PT` | `16` | Bias subtracted after masking |
| `k` | `4096` | Dot-product length |
| `rank` | `256` | Width of uniform blocks; inner-hash period |

Vendored `third_party/zk-pow` uses `ZERO_POINT_TRANSLATION = 32` (`[-32, 31]` uniform range). CPminer and CUDA use `ZERO_PT = 16`. Pool verify must use the same convention as the miner.

`ZERO_PT` is **not** in `MiningConfiguration`; it is a hardcoded implementation constant.

### PRF: `get_random_hash`

Keyed Blake3 over a 64-byte message:

```
msg[0..31]  = mostly zero; prepend slot holds (1 + block_index) as i32 LE
msg[32..63] = 32-byte domain label ("A_tensor" / "B_tensor")
digest      = BLAKE3_keyed(msg, key = a_noise_seed or b_noise_seed)
```

- Uniform matrices (`E_AL`, `E_BR`): prepend slot **0**
- Permutation matrices (`E_AR`, `E_BL`): prepend slot **1**

---

## 3. The four noise blocks

Conceptually:

```
NOISE_A = E_AL × E_AR    (sparse matvec, not dense GEMM)
NOISE_B = E_BL × E_BR
```

### `E_AL` and `E_BR` (uniform)

One **row** of length `rank` is generated on demand:

```
E_AL[row_idx][j] = (digest_byte & 63) - ZERO_PT
```

With `ZERO_PT = 16`: each entry ∈ **[-16, 47]**.

- **`E_AL`**: rows indexed by global A row (`0 .. m-1` when fusing full matrices; tile anchor rows `t_rows + pattern[]` at verify time).
- **`E_BR`**: rows indexed by global B^T row / B column (`col` index).

The full virtual matrix is never stored; mining calls `generate_uniform_row()` per row/column in `pearl_build_noisy_matrices()`.

### `E_AR` and `E_BL` (permutation)

Not int8 values — **`k` pairs of `u32` indices** into `[0, rank - 1]`:

```
first_idx  = random_u32 & (rank - 1)
second_idx = first_idx ^ (1 + mul_hi_u32(rank - 1, random_u32))
```

Each output coordinate along `k` applies one **+1 / −1** sparse row:

```
noise[l] = E_uniform[first_idx] - E_uniform[second_idx]
```

(`matvec_sparse_perm` in `cp_noise.c` / `pearl_noise.rs`.)

| Block | Blake3 key | Label | Depends on |
|-------|------------|-------|------------|
| `E_AL` row | `a_noise_seed` | `A_tensor` | `row_idx`, `rank`, constants |
| `E_AR` (full) | `a_noise_seed` | `A_tensor` | **`k`**, **`rank`** (shared for entire matrix) |
| `E_BL` (full) | `b_noise_seed` | `B_tensor` | **`k`**, **`rank`** |
| `E_BR` row | `b_noise_seed` | `B_tensor` | `col_idx`, `rank` |

`E_AR` / `E_BL` do **not** depend on `t_rows` / `t_cols`. Only which **`E_AL` / `E_BR` rows** are needed depends on the tile anchor.

### Fusing noise into noisy matrices

```c
noisy[l] = (int8_t)((int32_t)signal[l] + (int32_t)noise[l]);
```

(`pearl_fuse_noise_row_a` / `_b`.)

### Value ranges (CPminer `ZERO_PT = 16`)

| Quantity | Range |
|----------|--------|
| Signal A, B (miner-chosen / random gen) | `[-64, 63]` |
| Uniform `E_AL` / `E_BR` entry | `[-16, 47]` |
| Noise along `k` (after perm matvec) | `[-63, 63]` |
| Noisy element (signal + noise) | `[-127, 126]` (fits in `int8` without wrap) |
| Proof strip check (zk-pow verify) | `[-64, 64]` on **signal** strips in the proof |

Protocol MMA type is `Int7xInt7ToInt32` (7-bit signed operands in the proof); scanning uses **noisy** wider values.

### GEMM operand ranges (FastU8S8) and the zero-B idea

Noise sampling is fixed by the protocol (`ZERO_PT`, `RANGE_MASK`, permutation structure) — miners cannot widen or narrow it via `MiningConfiguration`.

**Noisy operand magnitudes** (CPminer `ZERO_PT = 16`):

| Matrix | Signal | Noise (typical) | Noisy (signal + noise) |
|--------|--------|-----------------|-------------------------|
| A | `[-64, 63]` | `[-63, 63]` | **`[-127, 126]`** (~full int8) |
| B | `[-64, 63]` | `[-63, 63]` | **`[-127, 126]`** |

The CPU worker’s `FastU8S8` path is algebraically exact for full int8×int8 with compensation; it is **especially** comfortable when one operand stays in int7 (`[-64, 63]`) and the other in int8.

**Proposed workaround:** set **signal B = 0**. Then:

```
noisy_B = 0 + noise_B  ∈  [-63, 63]   (int7)
noisy_A = signal_A + noise_A  ∈  [-127, 126]   (still int8)
```

Scan GEMM becomes **s8 × s7** in the sense above — B-side products use int7 magnitudes, which avoids the worst-case int8×int8 paths in approximate kernels.

**Regenerating only A per attempt:** on the CUDA path the miner may fix signal `B^T = 0` and draw a new random A each attempt. Then:

- `hash_b`, `b_noise_seed`, `E_BL`, `E_BR`, and B-side noise fusion are **fixed for the job** (reusable across attempts).
- Each new A still updates `hash_a` → `a_noise_seed` → `E_AL` / `E_AR` and A-side fusion (B fixed does not remove A-side noise work).
- `noisy_B = noise_B` only → B operand stays in int7 for GEMM.

Proofs commit the actual signal strips via Merkle (`cp_proof_build` takes `a` and `bt` from the hit attempt). A share with `B = 0` is valid if the pool accepts those strips: `hash_b = BLAKE3_keyed(padded zeros, job_key)` is the correct commitment for that matrix.

#### Where zero-B applies

| Path | Fix B = 0, regen only A? | Notes |
|------|--------------------------|-------|
| **CUDA** (`gpu_prepare_noisy_matrices`) | **Yes** — miner chooses matrices; set `d_Bt_sig = 0`, skip B RNG, cache B-side noise | Matches production protocol; proof uses copied `h_Bt_sig` |
| **CPU** (`cp_cpu_worker`, default) | **Yes** — `cp_cpu_worker_begin_job` caches noisy B; per attempt `pearl_generate_random_a` + A-noise only | `cp_worker_worker_handles_matrix_prep()` skips host gen in `cp_mine` |
| **CUDA `--cpu-gen`** | **No** — legacy `pearl_generate_ab` host path in `cp_mine` | Full A/B from `ab_seed` |
| **Pool verify** | **Yes** — recomputes noise from proof strips + `job_key`, not from header nonce | Same as any other miner-chosen B |

#### Caveats

1. **A-side noisy range unchanged.** Zero-B only tightens the **B** operand; **A** remains wide int8 after noise. Jackpot correctness still requires zk-pow’s int32 MAC `(s_a + n_a) * (s_b + n_b)`; FastU8S8 on CPU is only an approximation unless verified exactly.
2. **Search space.** Fixing B = 0 changes which `(A, B)` pairs are explored; that is a mining strategy, not a protocol violation.
3. **CPU/OpenCL default uses zero-B.** Signal `B^T = 0`; only A is randomized per attempt from OS CSPRNG (`cp_random_bytes`), not from the stratum header/nonce.

### Compact dependency summary

```
E_AR, E_BL = F_perm(key_side, k, rank)

E_AL[row]  = F_uniform(a_noise_seed, row, rank)
E_BR[col]  = F_uniform(b_noise_seed, col, rank)

a_noise_seed = H(job_key, hash_a(A))
b_noise_seed = H(job_key, hash_b(B^T))
job_key      = H(header, mining_config)
```

---

## 4. Proof generation dependency chain

Proof building (`cp_proof_build` / `build_plain_proof_b64` in `rust/cp-proof-ffi`) produces a **base64 bincode `PlainProof`**. It does **not** embed noisy matrices, pool target, or jackpot hash.

### Inputs required

| Input | Role |
|--------|------|
| `header` (76 B) | With `mining_config`, forms `job_key` |
| `mining_config` (52 B) | Must match `tile_layout`; validated byte-for-byte |
| `a` | Full **signal** A, row-major `int8[m×k]` |
| `bt` | Full **signal** B^T, row-major `int8[n×k]` |
| `m`, `n`, `k`, `rank` | Dimensions (`PlainProof` metadata) |
| `t_rows`, `t_cols` | Tile anchor from jackpot hit |
| `tile_layout` | `0` scattered, `1` contiguous 8×16, `2` CUTLASS, `3` contiguous 8×8, `4` contiguous 4×8 |

**Not required:** noisy A/B, `a_noise_seed`, pool target, stratum nonce (nonce only affects CPU `pearl_generate_ab` seeding; CUDA proof uses GPU-generated signal matrices from the hit attempt).

### Build steps

```
1. Validate mining_config matches tile_layout row/col patterns
2. job_key = BLAKE3(header || mining_config)
3. row_indices_A  = t_rows  + layout_row_pattern[]
4. row_indices_Bt = t_cols  + layout_col_pattern[]
5. Merkle multileaf proofs over pad_1024(A) and pad_1024(B^T), keyed by job_key
6. PlainProof { m, n, k, noise_rank, a: MatrixMerkleProof, bt: MatrixMerkleProof }
7. base64(bincode(PlainProof))
```

### Tile layouts and strip counts

| `tile_layout` | A strip rows | B^T strip rows | Config constant |
|---------------|--------------|----------------|-----------------|
| 0 scattered | 8 | 16 | `PEARL_SCATTERED_CONFIG` |
| 1 contiguous 8×16 | 8 | 16 | `PEARL_CONTIGUOUS_CONFIG` |
| 2 CUTLASS | 8 | 8 | `PEARL_CUTLASS_CONFIG` |
| 3 contiguous 8×8 | 8 | 8 | `PEARL_CONTIGUOUS_8x8_CONFIG` |
| 4 contiguous 4×8 | 4 | 8 | `PEARL_CONTIGUOUS_4x8_CONFIG` |

Row/column patterns are defined in `rust/cp-proof-ffi/src/lib.rs`. OpenCL `--ocl-tile 4x8` uses layout 4.

### `PlainProof` payload

```rust
struct PlainProof {
    m, n, k, noise_rank,
    a:  { merkle_proof, row_indices },   // h rows of A
    bt: { merkle_proof, row_indices },   // w rows of B^T
}
```

### Consistency requirements

All of the following must align or build/verify fails:

- `header` ↔ pool job
- `mining_config` ↔ worker backend / `tile_layout`
- `mining_config` ↔ `job_key` used when matrices were generated
- `t_rows`, `t_cols` ↔ tile where the worker found a share
- `a`, `bt` ↔ **signal** matrices from the hit attempt (same as used for `hash_a` / `hash_b` when noise was built)

### Pool submit

Stratum `mining.submit` sends:

- `job_id`
- `plain_proof` (base64)
- `hs` (hashrate)

The pool already holds `header` from `mining.notify`. It parses the proof, reconstructs public/private params, recomputes noise and jackpot, and checks difficulty (`nbits` from header).

### Local verify (`--verify`)

`cp_proof_verify` checks jackpot against a **pool target** (32-byte BE U256) using zk-pow, with the same header and proof blob. Optional pre-submit check in the miner when `--verify` is enabled.

---

## 5. End-to-end diagram

```
                    mining.notify
                         │
                         ▼
              ┌──────────────────────┐
              │ header + mining_cfg  │
              └──────────┬───────────┘
                         │ job_key
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  miner-chosen A,B   hash_a/B     (pool keeps header)
  (CUDA: GPU random; CPU: pearl_generate_ab)
         │               │
         │               ▼
         │        a_noise_seed, b_noise_seed
         │               │
         ▼               ▼
      signal A,B     E_AL,AR,BL,BR
         │               │
         └───────┬───────┘
                 ▼
           noisy A, B^T  ──► GEMM scan ──► hit (t_rows, t_cols)
                 │
                 │  on share (CUDA: copy d_A_sig/d_Bt_sig)
                 ▼
    cp_proof_build(signal A, B^T, t_rows, t_cols, header, mining_cfg)
                 │
                 ▼
           plain_proof b64 ──► mining.submit
                 │
                 ▼
    pool: strips + job_key → noise → jackpot hash → nbits check
```

---

## 6. Source references

| Topic | Location |
|--------|----------|
| Noise + commitments | `src/common/cp_noise.c` |
| CUDA noise constants | `src/cuda/cp_gpu_gen.cuh`, `cp_noise_phase.cuh` |
| Proof build FFI | `rust/cp-proof-ffi/src/lib.rs`, `mining_config.rs` |
| Mine loop / `cp_proof_build` call | `src/common/cp_mine.cpp` |
| API | `include/cp_proof.h` |
| zk-pow noise | `third_party/zk-pow/src/circuit/pearl_noise.rs` |
| zk-pow jackpot | `third_party/zk-pow/src/circuit/chip/jackpot/helper.rs` |
| zk-pow verify | `third_party/zk-pow/src/api/verify.rs` |
| CLI helper | `scripts/plain_proof_host.py` |
