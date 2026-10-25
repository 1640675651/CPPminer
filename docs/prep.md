# GPU matrix preparation

Every Pearl attempt (nonce) mines on a fresh noisy A. On the GPU backends (oneDNN and OpenCL) the
whole preparation runs on the device, in the shared module `src/opencl/case33_ocl_prep.cpp` with the
kernels in `src/opencl/kernels/cp_ocl_prep.cl` and `cp_ocl_merkle.cl`. Production dimensions are
m = n = 131072, k = 4096, r = 128, so A is 512 MiB.

Once the GEMM got fast (Arc A770 with the oneDNN backend: about 0.69 s per attempt), preparation became
a visible share of the attempt. This page records how it was made cheap and what was learned on the
way. Short version: per-attempt prep on the A770 went from 22 ms to 4 ms, and end-to-end throughput
from about 98 to 101.4 TMAC/s against about 102 scan-only.

## What is prepared

Per job (once): the noisy B. Signal B is all zeros (zero-B mode), so B is pure noise; its seed comes
from the keyed digest of the zero matrix.

Per nonce:

```
signal A_sig  ->  keyed BLAKE3 Merkle root  ->  noise seed = f(root)  ->  uniform rows  ->  noisy A
```

| Step | What it is |
|---|---|
| Signal A_sig | Random int8 matrix, values in [-64, 63], from splitmix64 per element (`ocl_gen_random_matrix16`). |
| Commitment | Keyed BLAKE3 Merkle hash of A_sig: 1 KiB chunks, 256-chunk subtrees (subroots), then a reduction to the root. The share proof commits to this root. |
| Noise seed | Derived on the host from the root (`pearl_a_noise_seed_from_hash`). |
| Uniform rows | 128 bytes per row: four keyed single-block BLAKE3 hashes of `(1 + block, zeros, "A_tensor")` under the noise seed, each byte mapped to `(b & 63) - 16`. |
| Pair table | Two indices in 0..127 per k, from a hash of the noise seed (`ocl_build_perm_pairs`). |
| Noisy A | `A[r][k] = A_sig[r][k] + el[r][p0[k]] - el[r][p1[k]]` (wrapping int8): a rank-128 noise added to the signal. |

The data dependency in the middle is hard: nothing that uses the noise can start before the
commitment root exists and the host has derived the seed. That rules out fusing the two halves.

The noisy A is written in a backend-specific layout:

| Backend | Noise kernel | Output |
|---|---|---|
| oneDNN (gemmstone) | `ocl_noisy_matrix_rowmajor_sg` | plain row-major A (or one of the other gemmstone layouts through the column-major kernels) |
| OpenCL | `ocl_fused_prepack_a_wg` | prepacked macro-block layout of the OpenCL GEMM |

The signal, commitment, uniform rows and pair table are shared by both backends.

## PR #27: faster Pearl GPU matrix prep (merged 2026-10-09)

Commit `ddc134b`, "opencl: faster Pearl GPU matrix prep (work-group noise, unrolled BLAKE3, zero-B
digest on GPU, 256-wide A prepack)", by danifest751. It laid the foundation everything below builds on:

1. **Work-group row-major noise and 16-wide signal generation.** The old noise kernel ran one work
   item per 4 KiB row with byte accesses strided by K across lanes. The uniform rows moved to their own
   row-parallel pass, and one work group per row now builds the noisy row from local memory with
   16-byte vector stores. The signal generator writes 16 elements per item with the same per-index
   values. Arc A380, 131072 x 4096: generation 21 -> 12.5 ms, noise 154 -> 56 ms, prep 200 -> 96 ms per
   attempt. It also added the tools used ever since: `CP_OCL_PREP_CHECK=1` (byte comparison against
   the per-row reference kernel), `CP_OCL_PREP_TIMING=1` (stage times) and `CP_OCL_PREP_WG=0`.
2. **Column-major B noise through the same work-group path.** Column j of the gemmstone TN B is the
   row-major layout without a signal. A380: per-job B prep about 200 -> 80 ms.
3. **25x faster keyed matrix hash.** The chunk hash had gathered each 64-byte block byte by byte with
   a bounds test per byte, and the table-driven rounds indexed the message at run time, which put it in
   scratch memory. Full chunks now load their words with four 16-byte loads and compress fully
   unrolled, with the message permutation done as register renames. R9700: hash of the 512 MiB A
   51 -> 2 ms, attempt prep 61 -> 12 ms (about +6% Pearl throughput, since prep and scan are serial).
   The fused prepack kernels also store one 4-byte word per work item instead of one thread writing a
   stripe byte by byte.
4. **Zero-B job digest on the GPU.** Each new job hashed the all-zero 512 MiB B^T with the scalar CPU
   BLAKE3, 0.78 s per job on an EPYC 7302 with jobs arriving about every 20 s. The chunk kernel got an
   all-zero path with no loads; `CP_OCL_ZERO_B_CHECK=1` compares against the CPU digest.
5. **256-wide fused A prepack** for the OpenCL backend. The old 8-lane kernel regenerated each row's
   uniform noise for every k block. The new one precomputes the noise rows once, stages signal and
   noise in padded local memory and writes coalesced words. R9700: prep 12 -> 7 ms. `CP_OCL_FPA_WG=0`
   restores the old kernel.
6. **Fit into 32 KiB of local memory.** The AMD Windows driver allows 32 KiB per work group (ROCm
   64 KiB), and the prepack staged about 34 KiB. The host reads `CL_DEVICE_LOCAL_MEM_SIZE` and builds
   with `FPA_SPLIT` work groups per block; a failed launch falls back to the 8-lane kernel.

## Arc A770 follow-up (2026-10-19)

With the fused in-kernel jackpot the oneDNN scan runs at about 102 TMAC/s, and the wall time per
attempt was 0.713 s against 0.693 s of scan. `CP_OCL_PREP_TIMING=1` split the 22 ms of prep:

| Step (A770, per attempt) | Before | After |
|---|---:|---:|
| Signal A_sig | 4.3 ms | 0.2 ms |
| Commitment | 2.8 ms | 0.7 ms |
| Uniform rows | 8.7 ms | 0.3 ms |
| Noisy A | 6.1 ms | 2.9 ms |
| **Prep total** | **22 ms** | **4 ms** |
| Wall per attempt | 0.713 s | 0.694 s |

### 1. Uniform rows with the hash in registers

`ocl_uniform_rows` built each keyed BLAKE3 through private byte arrays (`uchar msg[64]`, `uchar
digest[32]` and helpers taking byte pointers). On Intel GPUs the compiler placed those arrays in
scratch memory, and 524,288 hashes per attempt cost 8.7 ms. `ocl_uniform_rows_b3` computes the same
hash with one work item per 32-byte block, the 16 message words built directly as `uint`s, and the
existing word-based `d_b3_compress_words`; it writes the 32 output bytes with two 16-byte stores.

| GPU | Before | After |
|---|---:|---:|
| Arc A770 | 8.7 ms | 0.3 ms |
| UHD 770 (32768 rows) | 13.9 ms | 0.5 ms |
| GTX 1070 | no change | |

NVIDIA's compiler kept the byte arrays in registers, so the old kernel was already fast there.
`CP_OCL_PREP_UROWS_B3=0` restores the old kernel; `CP_OCL_PREP_CHECK=1` compares the two byte for byte.

### 2. Noisy A: the access pattern, not the lookups

The work-group kernel from PR #27 does two local-memory byte gathers per output byte. The assumption
that those gathers were the cost turned out to be wrong. A copy-only diagnostic variant
(`-DNSG_DIAG=1`, wrong output, timing only) measured the memory floor for each work decomposition
(A770, 1 GiB moved per attempt):

| Variant | Time |
|---|---:|
| Copy, 16 bytes per work item per access (the PR #27 pattern) | 5.5 ms |
| Copy, 64 bytes per work item | 6.1 ms |
| Copy, 4 bytes per work item, 32 items per row, 8 units in flight | 2.7 ms |
| Full kernel with that pattern, row table in local memory | **2.9 ms** |
| Same pattern, lookups as Intel sub-group shuffles | 5.5 ms |
| 16 bytes per item, sub-group shuffles (first attempt) | 6.0 ms |
| Several rows per sub-group with pair indices hoisted into registers | 16-35 ms (spills) |

With 4 bytes per work item, 32 neighbouring items cover 128 contiguous bytes per access; 16-byte
accesses per item halve the achieved bandwidth on this hardware. Issuing 8 units' loads before their
stores keeps enough bytes in flight. Against expectations, local-memory byte gathers cost about
0.2 ms, while register shuffles cost 3 ms.

The resulting kernel (`ocl_noisy_matrix_rowmajor_sg`; the name is historical) is plain OpenCL 1.2:
8 rows per 256-item work group, the rows' 128 noise bytes loaded into local memory with one 4-byte
word per item, then streaming the row 4 bytes per item. It uses no vendor extensions.

| GPU | PR #27 kernel | Now |
|---|---:|---:|
| Arc A770 (131072 rows) | 6.1 ms | 2.9 ms |
| UHD 770 (32768 rows) | 58.1 ms | 12.5 ms |

`CP_OCL_PREP_SG=0` keeps the PR #27 kernel. `CP_OCL_PREP_DEFS="-D..."` passes build options to the
prep kernels for experiments: `NSG_KPL` (4 or 16 bytes per item), `NSG_UNROLL` (units in flight) and
`NSG_DIAG` (1 copy only, 2 synthetic pair indices; both timing only).

### 3. One random row per nonce and an incremental commitment

The signal does not have to be regenerated in full per nonce: the share commits to whatever A_sig was
used, and the noise rebuilt from the new root changes every element of A anyway. The CPU worker used to
write one random value per column into a zero matrix (the Hexagon worker still does; the CPU now writes
one row into it). For the GPU a full
random row is the better unit:

- A row is 4 KiB contiguous: four consecutive 1 KiB Merkle chunks, inside one 256-chunk subtree. Per-column
  writes touch about 1,900 of the 2,048 subtrees, which would need a stored hash per chunk to avoid
  rehashing nearly everything.
- With a row, the incremental commitment reuses the existing kernels: `ocl_keyed_chunk_roots` got a
  `group_base` argument and rehashes the one affected subtree into the stored subroots (the same
  buffer the witness path keeps); the stored subroots are copied and reduced to the new root.

A_sig is zeroed (`clEnqueueFillBuffer`) on the first attempt and after a dimension change, and hashed in
full then and on a new job key (the hash is keyed). Until 2026-10-25 it was generated in full instead;
the zero rows are as valid a signal and fill in as nonces write them. Every nonce the host derives the row index from the nonce's seed and the full
generator kernel runs over just that row (global work offset row * K / 16, K / 16 work items), so the
values are the ones the full generation would give. (Until 2026-10-20 the host computed the row and
wrote 4 KiB; see section 6.) Signal 4.3 -> 0.2 ms,
commitment 2.8 -> 0.7 ms (mostly launch and sync latency). `CP_OCL_PREP_CHECK=1` also recomputes the full
commitment every nonce and compares roots; `CP_OCL_PREP_ROWPOKE=0` restores full regeneration.

Other GPUs gain the same way: the OpenCL backend on the A770 at 65536^2 went from 11 to 4 ms of prep per
attempt, and on the GTX 1070 at 32768^2 from 9.1 to 2.4 ms.

The CUDA backend (`gpu_prepare_attempt_a`) uses the same scheme: `cp_gen_random_range_kernel` writes the
row with the full generator's formula, and `cp_keyed_chunk_roots_kernel` got a `block_offset` argument
to rehash the one subtree into `d_a_subroots`. Anything else that writes `d_A_sig` (host upload, the
full A/B prep, the alignment test) marks the signal stale, so the next attempt zeroes it again
(`cudaMemset`) and hashes it in full.
GTX 1070 at 131072^2: random A 22 ms -> under 1 ms, commitment 35 ms -> under 1 ms, prep 73 -> 24 ms
per attempt (the rest is the A noise). `CP_CUDA_PREP_CHECK=1` compares the row with the host formula and
the incremental root with a full rehash every nonce; `CP_CUDA_PREP_ROWPOKE=0` restores full regeneration.
The host round trips of section 6 are gone there too: the job key is written when it changes,
`cp_a_noise_seed_kernel` derives A's noise seed from the root on the device and writes it as the noise
seed, the jackpot key (`d_a_key8`) and the witness root (`d_a_root`, read when a share is proven), and the
found flag is cleared with a device memset. The first derivation in a process is compared with the host
formula (fallback to the host on a mismatch; every nonce with `CP_CUDA_PREP_CHECK=1`). Steady state per
attempt: no host-to-device copies, and one 4-byte found-flag read per scan batch (32 on the GTX 1070).

The wgpu backend (`PearlEngine::prep_a_signal` in `rust/cp-pearl-wgpu-ffi`) does the same with its
existing kernels: `pearl_gen_random_matrix` regenerates the row's 1,024 words (its parameters gained a
`word_end`, so a dispatch never writes past its binding), and `pearl_keyed_chunk_roots` runs one work
group with `bid_begin` = the row's group, writing straight into `a_subroots`; a copy of those is reduced
to the root. `a_sig` is cleared once when the job buffers are created; the first attempt, a new job key
or a change of dimensions commits the whole signal again. GTX 1070 (Vulkan)
at 131072^2: prep 145 -> 50 ms per attempt. `CP_WGPU_PREP_CHECK=1` compares the row with the host
formula and the incremental root with a full rehash every nonce; `CP_WGPU_PREP_ROWPOKE=0` restores full
regeneration. The noise seed is still derived on the host (one 32-byte root read per attempt).

The CPU worker (`zero_b_prepare_attempt` in `src/cpu/cp_cpu_worker.cpp`) rewrites one row per nonce
with `pearl_fill_random_a_row` (same row choice and values as the GPU kernels); like the GPUs, A starts
as the zeroed launch buffer and fills in as rows are written. The commitment uses the
incremental `PearlMatrixHash`, which rehashes the row's 4 chunks and their tree ancestors; the first
nonce, a new job key or another buffer hashes everything. 131072^2 on an i9-12900K: prep 1.3 s (full
regeneration) -> 0.12 s per attempt, against a 46 s scan; the old per-column writes changed chunks all
over A and rehashed the whole matrix every nonce.
`CP_CPU_PREP_CHECK=1` compares the incremental root with a full rehash every nonce;
`CP_CPU_PREP_ROWPOKE=0` restores full regeneration.

### 4. Noisy A written in the systolic (packed) layout

The preferred kernel used to gather A: every load message was 8 lanes reading one dword (4 consecutive
k) from 8 different rows, which assembles the VNNI operand format in flight but costs 16 small
messages per thread and k step, double-buffered (`sS32x2`) to hide their latency. Since the prep writes
all of A every nonce anyway, it can write A directly in gemmstone's packed-columns layout
(`MatrixLayout::Pc`: 16-row panels, 4 consecutive k per dword) at no extra cost, and the kernel then
block-loads A single-buffered (`sB32`). A timing-only switch that declared A packed without changing the
data measured it first (0.638 vs 0.688 s per attempt); without the packing the same block strategy does
not fit the register file, and single-buffered gathers are slower (0.733 s).

`ocl_noisy_matrix_a_packed16` writes the packed layout: one 256-item work group per 16-row panel, the
signal tile staged through local memory with coalesced loads, item w = 16 g + r producing the dword of
row r and k group g so each unit's stores are 1 KB contiguous. Panel p starts at p x 16 x lda bytes
(gemmstone offsets a panel by i0 x lda), element (m, k) at (k / 4) x 64 + (m % 16) x 4 + k % 4 inside it.
The host path that uploads a CPU-built A packs the same way. It takes 4.4 ms against 2.9 ms for the
row-major kernel (more units in flight: 4.0 ms; not worth the local memory).

| A770, 131072^2 x 4096, per attempt | Row-major A | Packed A |
|---|---:|---:|
| Scan, fused jackpot | 0.689 s | 0.642 s |
| Wall, fused jackpot | 0.694 s | 0.648 s |
| Wall, separate jackpot | 0.708 s | 0.664 s |
| End-to-end, fused | 101.4 TMAC/s | 108.5 TMAC/s |

`CASE5_A_PACKED=0` restores row-major A; the UHD 770 (catalog kernel) and the OpenCL backend are not
affected.

### 5. Per-job noisy B in the packed layout

B is built once per job (zero-B: pure noise), so its layout costs nothing per attempt. With A packed, a
B packed as gemmstone packed rows (`MatrixLayout::Pr`: 64-column panels, 32 consecutive k per column)
lets each thread fetch its share of a B slice as one contiguous 128-byte piece instead of four 32-byte
pieces from four columns, and gemmstone uses a linear copy split. The strategy string is unchanged; the
saving is energy per byte, which the 210 W cap turns into clock (about +90 MHz).
`ocl_noisy_matrix_b_packed` writes the noisy B straight into the panels (any panel width up to 64 and
crosspack dividing 64; A770 64 x 32, UHD 770 32 x 4): one work group per panel keeps the panel's uniform
rows in local memory, and within each 64-k unit the panel's output is contiguous, so stores coalesce.
Until 2026-10-20 the prep wrote column-major B into a 512 MiB scratch and repacked it (A770 18-20 ms,
UHD 770 270-400 ms per job, scratch included); the fused kernel takes 8 ms and 263-283 ms and needs no
scratch, so the oneDNN prep holds the same three large buffers as the OpenCL worker (signal A, packed
noisy A, packed noisy B). `CP_OCL_PREP_CHECK=1` builds the column-major reference in scratch and
compares every byte through the index map.

| A770, 131072^2 x 4096, per attempt | Packed A | Packed A and B |
|---|---:|---:|
| Wall, fused jackpot | 0.648 s | 0.627 s |
| Scan, separate jackpot | 0.661 s | 0.631 s |
| Pure GEMM, 98304^2 x 4096 | 224.5 TOPS | 233-235 TOPS |

`CASE5_B_PACKED=0` restores column-major B. Why not zero-A instead: the noise seeds are chained
(`b_noise_seed = H(job_key, root(B))`, `a_noise_seed = H(b_noise_seed, root(A))`), so with a per-nonce B
both noisy matrices would change every nonce.

### 6. No host round trips in the per-nonce prep (2026-10-20)

Per attempt the host used to write the row (4 KiB), the job key and A's noise seed, read the Merkle root
back to derive that seed, and write the seed again as the scan's jackpot key. Now:

| Step | Before | Now |
|---|---|---|
| Changed row | host computes, 4 KiB write | generator kernel over the row |
| Job key | written every attempt | written when it changes (once per job) |
| Subroot copy before the reduction | `clEnqueueCopyBuffer` (copy engine) | copy kernel (`OpenClContext::copy_buffer`) |
| A noise seed | root read, host BLAKE3, seed write | `ocl_a_noise_seed` (cert-v3 binding + BLAKE3) on the device |
| Jackpot key for the scan (oneDNN) | host writes, 48 B per panel in the fused kernel's arguments | device copy once per attempt; 16 B of tile arguments per panel |
| Witness root | kept on the host after every attempt | read from the device when a share is proven |

The oneDNN worker passes no key buffer to the prep and a null key to the scan; kernels read the key on
the device and host-side users (the per-tile fused kernel on Gen12, the coop audit) fetch it on demand.
The OpenCL backend still asks for a host copy of the key (one 32-byte read instead of a root read and a
seed write). The first derivation in a process is compared with the host formula and falls back to the
host on a mismatch; `CP_OCL_PREP_CHECK=1` also checks the GPU row against the host formula every nonce.

A770, fused scan, steady state per attempt: host-to-device writes 23 (4,628 B) -> 19 (308 B), reads 5 -> 4,
prep 5.3 -> 5.0 ms. The bound (pool target) is likewise written only when it changes, which takes the
steady state to 13 writes (196 B): per panel the cleared found flag and hit coordinates and 16 B of
tile arguments, plus one flag clear at scan start.

### Verification

Every change was checked the same way:

- `CP_OCL_PREP_CHECK=1` byte comparison against the reference kernels (noisy A and B, uniform rows,
  packed A through its index map) and full-versus-incremental commitment roots;
- mock shares built and zk-verified (`--mock --verify`) on the A770 (both jackpot modes), the UHD 770
  and, for the OpenCL backend, the A770 and GTX 1070.

## Lessons

- **Measure the split first.** The first guess (byte gathers in the noise kernel) was wrong; the real
  costs were a hash living in scratch memory and the memory access width.
- **Byte arrays in private memory are a trap on Intel GPUs.** The same code ran fine on NVIDIA. Keep
  hash state and messages in `uint` variables.
- **Access width matters more than instruction count.** A copy-only diagnostic gives the floor for a
  decomposition before any arithmetic is added.
- **Discard cold attempts.** The first attempt of a run includes kernel builds and caches; one early
  UHD 770 reading (21 ms) was a cold-start artifact, the steady value is 12.5 ms.
- **The prep kernels are one program for every device.** The OpenCL backend's device 0 on the test
  machine is a GTX 1070 (NVIDIA OpenCL 1.2, no sub-group built-ins). An Intel-only construct in
  `cp_ocl_prep.cl` made the whole program fail to build there. Keep the shared kernels plain OpenCL 1.2,
  and guard anything vendor-specific with its extension macro and a host fallback.

## Considered and not done

| Idea | Why not |
|---|---|
| Fuse signal generation with the commitment | Both compute bound; saves about 1 ms (one 512 MiB re-read and a sync). |
| Fuse uniform rows into the noise kernel | The uniform rows already take 0.3 ms. |
| Apply the noise inside the GEMM | Byte arithmetic beside the `dpas` instructions and 64 registers of row tables; the algebraic split fails because the XOR milestones are taken on partial sums. |
| Incremental noise | The noise seed comes from the new root, so every element changes. |

## What is left

On the A770 the remaining 5-6 ms is mostly the packed noise kernel (4.4 ms, against a 2.7 ms copy floor)
plus small launches.
Overlapping prep with the previous nonce's scan on a second queue with a double-buffered A could hide
part of it (the GEMM is power-capped, so mainly the memory-bound parts).

Prepacking B once per job into the systolic (VNNI) layout was tried and dropped (2026-10-19). The
disassembled k loop of the preferred kernel shows that B is never repacked in registers: each thread's
32-byte transposed block loads from global memory land in four registers that are stored to SLM
unchanged as one 128-byte block, and the systolic arrangement comes from the addressing alone. A
prepacked B could at most replace four 32-byte loads with one 128-byte load per thread and k step, and
gemmstone's packed-B path (`MatrixLayout::Pr`, panel width = unrollN, crosspack 4) runs out of registers
for the 16x64 strategy and for every catalog fallback, even with single-buffered A.
