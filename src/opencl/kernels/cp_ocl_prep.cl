/* OpenCL matrix prep: random A, noise, fused coalesced prepack (signed s8s8). */

#ifndef MR
#define MR 8
#endif
#ifndef NR
#define NR 16
#endif
#ifndef KR
#define KR 128
#endif
#ifndef R_RANK
#define R_RANK 128
#endif

#ifndef MACRO_M
#define MACRO_M 128
#endif
#ifndef MACRO_N
#define MACRO_N 128
#endif

#define CP_RANGE_MASK 63
#define CP_ZERO_PT 16
#define CP_B3_LINES 8
#define K_GROUPS (KR / 4)
#define KG_BYTES_A (MR * 4)
#define KG_SLICE_B (NR * 4)
#define MICRO_M (MACRO_M / MR)
#define MICRO_N (MACRO_N / NR)
#define MACRO_KG_STRIP_A (MICRO_M * KG_BYTES_A)
#define MACRO_KG_STRIP_B (MICRO_N * KG_SLICE_B)
#define MACRO_KB_BLOCK_A (K_GROUPS * MACRO_KG_STRIP_A)
#define MACRO_KB_BLOCK_B (K_GROUPS * MACRO_KG_STRIP_B)

inline void ocl_generate_uniform_row_glob(int row_idx, int num_cols, __global const uchar *seed,
                                          int is_b, uchar *row_out) {
    int start_idx = row_idx * num_cols;
    int block = start_idx / D_B3_OUT;
    int out_i = 0;
    while (block * D_B3_OUT < start_idx + num_cols) {
        uchar digest[32];
        d_get_random_hash_glob(block, seed, is_b, 0, digest);
        for (int k = 0; k < D_B3_OUT; k++) {
            int idx = block * D_B3_OUT + k;
            if (idx >= start_idx && idx < start_idx + num_cols) {
                row_out[out_i++] = (uchar)((digest[k] & CP_RANGE_MASK) - CP_ZERO_PT);
            }
        }
        block++;
    }
}

__kernel void ocl_gen_random_matrix(ulong rng_seed, int matrix_tag, int total_elems,
                                    __global char *out) {
    int idx = get_global_id(0);
    if (idx >= total_elems) {
        return;
    }
    ulong s = rng_seed ^ ((ulong)matrix_tag * 0xD1B54A32D192ED03UL) ^
              (ulong)idx * 0x9E3779B97F4A7C15UL;
    s = cp_splitmix64(s);
    out[idx] = (char)((int)((s >> 32) % 128u) - 64);
}

/* ocl_gen_random_matrix values, 16 consecutive elements per work item (total % 16 == 0). */
__kernel void ocl_gen_random_matrix16(ulong rng_seed, int matrix_tag, int total_elems,
                                      __global char *out) {
    const int base = (int)get_global_id(0) * 16;
    if (base >= total_elems) {
        return;
    }
    const ulong tag = rng_seed ^ ((ulong)matrix_tag * 0xD1B54A32D192ED03UL);
    char v[16];
    for (int j = 0; j < 16; ++j) {
        const ulong s = cp_splitmix64(tag ^ (ulong)(base + j) * 0x9E3779B97F4A7C15UL);
        v[j] = (char)((int)((s >> 32) % 128u) - 64);
    }
    vstore16(vload16(0, v), 0, out + base);
}

__kernel void ocl_build_perm_pairs(int is_b, __global const uchar *noise_seed, int k, int rank,
                                   __global uint *pairs_out) {
    const int block_idx = (int)get_global_id(0);
    const int col0 = block_idx * CP_B3_LINES;
    if (col0 >= k) {
        return;
    }

    uchar digest[32];
    d_get_random_hash_glob(block_idx, noise_seed, is_b, 1, digest);

    const uint rank_mask = (uint)(rank - 1);
    for (int j = 0; j < CP_B3_LINES; j++) {
        int col = col0 + j;
        if (col >= k) {
            break;
        }
        uint w = d_b3_load32_priv(digest + j * 4);
        uint first = w & rank_mask;
        uint second = first ^ (1u + cp_mul_hi_u32((uint)(rank - 1), w));
        pairs_out[(size_t)col * 2] = first;
        pairs_out[(size_t)col * 2 + 1] = second;
    }
}

/* Coalesced B: one WG per (jm, kb, tc); matches prepack_b_coalesced on host. */
__kernel void ocl_fused_prepack_b(__global uchar *b_pre_out, __global const uchar *b_noise_seed,
                                  __global const uint *pairs, int N, int K, int rank,
                                  int blocks_k, int macro_cols, int has_signal,
                                  __global const char *b_signal_colmajor) {
    const int g = get_group_id(0);
    const int tc = g % MICRO_N;
    const int kb = (g / MICRO_N) % blocks_k;
    const int jm = g / (MICRO_N * blocks_k);
    const int col = get_local_id(0);
    if (jm >= macro_cols || col >= NR) {
        return;
    }

    const int k0 = kb * KR;
    const int ncol = (jm * MICRO_N + tc) * NR + col;
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(ncol, rank, b_noise_seed, 1, el);

    __local uchar stripe[NR][KR];
    for (int t = 0; t < KR; ++t) {
        const int l = k0 + t;
        int pos = (int)(char)el[pairs[(size_t)l * 2]];
        int neg = (int)(char)el[pairs[(size_t)l * 2 + 1]];
        int sig = 0;
        if (has_signal) {
            sig = (int)b_signal_colmajor[(size_t)ncol * (size_t)K + (size_t)l];
        }
        stripe[col][t] = (uchar)((char)(sig + (pos - neg)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* every column's work-item stores its own 4-byte k-group words */
    {
        const size_t block_base =
                ((size_t)jm * (size_t)blocks_k + (size_t)kb) * (size_t)MACRO_KB_BLOCK_B;
        for (int kg = 0; kg < K_GROUPS; ++kg) {
            const size_t dst = block_base + (size_t)kg * (size_t)MACRO_KG_STRIP_B +
                               (size_t)tc * (size_t)KG_SLICE_B + (size_t)col * 4;
            const uint w = (uint)stripe[col][kg * 4] | ((uint)stripe[col][kg * 4 + 1] << 8) |
                           ((uint)stripe[col][kg * 4 + 2] << 16) |
                           ((uint)stripe[col][kg * 4 + 3] << 24);
            *(__global uint *)(b_pre_out + dst) = w;
        }
    }
}

/* Coalesced A: one WG per (im, kb, tr); matches prepack_a_coalesced on host. */
__kernel void ocl_fused_prepack_a(__global uchar *a_pre_out, __global const uchar *a_noise_seed,
                                  __global const uint *pairs, __global const char *a_signal,
                                  int M, int K, int rank, int blocks_k, int macro_rows) {
    const int g = get_group_id(0);
    const int tr = g % MICRO_M;
    const int kb = (g / MICRO_M) % blocks_k;
    const int im = g / (MICRO_M * blocks_k);
    const int row = get_local_id(0);
    if (im >= macro_rows || row >= MR) {
        return;
    }

    const int k0 = kb * KR;
    const int nrow = (im * MICRO_M + tr) * MR + row;
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(nrow, rank, a_noise_seed, 0, el);

    __local uchar stripe[MR][KR];
    for (int t = 0; t < KR; ++t) {
        const int l = k0 + t;
        int pos = (int)(char)el[pairs[(size_t)l * 2]];
        int neg = (int)(char)el[pairs[(size_t)l * 2 + 1]];
        int sig = (int)a_signal[(size_t)nrow * (size_t)K + (size_t)l];
        stripe[row][t] = (uchar)((char)(sig + (pos - neg)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Every row's work-item stores its own 4-byte k-group words (thread 0 storing the whole
       stripe byte by byte took 7 ms per 512 MB A on an R9700). */
    {
        const size_t block_base =
                ((size_t)im * (size_t)blocks_k + (size_t)kb) * (size_t)MACRO_KB_BLOCK_A;
        for (int kg = 0; kg < K_GROUPS; ++kg) {
            const size_t dst = block_base + (size_t)kg * (size_t)MACRO_KG_STRIP_A +
                               (size_t)tr * (size_t)KG_BYTES_A + (size_t)row * 4;
            const uint w = (uint)stripe[row][kg * 4] | ((uint)stripe[row][kg * 4 + 1] << 8) |
                           ((uint)stripe[row][kg * 4 + 2] << 16) |
                           ((uint)stripe[row][kg * 4 + 3] << 24);
            *(__global uint *)(a_pre_out + dst) = w;
        }
    }
}

/* Same bytes as ocl_fused_prepack_a, one 256-wide work group per (im, kb) block, or per
 * 1/FPA_SPLIT of its rows. The rows' uniform noise (el_rows, R_RANK bytes per row, from
 * ocl_uniform_rows) and the signal tile are staged in local memory with a 4-byte row pad, so
 * lanes that read the same k of neighbouring rows hit different banks. Output word of block row
 * r and k group kg is kg * MACRO_M + r (byte offset kg * MACRO_KG_STRIP_A + r * 4), so stores
 * are coalesced. FPA_SPLIT (host, from the device's local memory size): 1 needs ~34 KB at
 * MACRO_M 128; AMD's Windows driver gives a work group 32 KB, so it gets 2. */
#ifndef FPA_SPLIT
#define FPA_SPLIT 1
#endif
#define FPA_WG 256
#define FPA_ROWS (MACRO_M / FPA_SPLIT)
#define FPA_SW (KR / 4 + 1)
#define FPA_EW (R_RANK / 4 + 1)
__kernel __attribute__((reqd_work_group_size(FPA_WG, 1, 1))) void
ocl_fused_prepack_a_wg(__global uint *a_pre_out, __global const uint *el_rows,
                       __global const uint *pairs, __global const char *a_signal, int K,
                       int blocks_k) {
    __local uint sig[FPA_ROWS * FPA_SW];
    __local uint el[FPA_ROWS * FPA_EW];
    __local uchar p0[KR];
    __local uchar p1[KR];

    const int g = (int)get_group_id(0);
    const int part = g % FPA_SPLIT;
    const int blk = g / FPA_SPLIT;
    const int kb = blk % blocks_k;
    const int im = blk / blocks_k;
    const int t = (int)get_local_id(0);
    const int rbase = part * FPA_ROWS; /* first block row of this group */
    const int row0 = im * MACRO_M + rbase;
    const int k0 = kb * KR;

    for (int i = t; i < FPA_ROWS * (KR / 16); i += FPA_WG) {
        const int r = i / (KR / 16);
        const int c = i % (KR / 16);
        const uint4 v =
                *(__global const uint4 *)(a_signal + (size_t)(row0 + r) * (size_t)K + k0 + c * 16);
        __local uint *d = sig + r * FPA_SW + c * 4;
        d[0] = v.x;
        d[1] = v.y;
        d[2] = v.z;
        d[3] = v.w;
    }
    for (int i = t; i < FPA_ROWS * (R_RANK / 16); i += FPA_WG) {
        const int r = i / (R_RANK / 16);
        const int c = i % (R_RANK / 16);
        const uint4 v = *(__global const uint4 *)(el_rows + (size_t)(row0 + r) * (R_RANK / 4) +
                                                   c * 4);
        __local uint *d = el + r * FPA_EW + c * 4;
        d[0] = v.x;
        d[1] = v.y;
        d[2] = v.z;
        d[3] = v.w;
    }
    for (int i = t; i < KR; i += FPA_WG) {
        p0[i] = (uchar)pairs[(size_t)(k0 + i) * 2];
        p1[i] = (uchar)pairs[(size_t)(k0 + i) * 2 + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __global uint *dst = a_pre_out + ((size_t)im * (size_t)blocks_k + (size_t)kb) *
                                             (size_t)(MACRO_KB_BLOCK_A / 4);
    for (int w = t; w < FPA_ROWS * K_GROUPS; w += FPA_WG) {
        const int kg = w / FPA_ROWS;
        const int r = w % FPA_ROWS;
        const uint s = sig[r * FPA_SW + kg];
        const __local uchar *e = (const __local uchar *)(el + r * FPA_EW);
        uint out = 0;
        for (int b = 0; b < 4; ++b) {
            const int k = kg * 4 + b;
            const uint v = (s >> (8 * b)) + (uint)e[p0[k]] - (uint)e[p1[k]];
            out |= (v & 0xffu) << (8 * b);
        }
        dst[kg * MACRO_M + rbase + r] = out;
    }
}

/* Pearl noisy matrix in native row layout: A is M×K (row*K+k), B^T is N×K (col*K+k). */
__kernel void ocl_noisy_matrix_rowmajor(__global char *out, __global const uchar *noise_seed,
                                        __global const uint *pairs, int rows, int K, int rank,
                                        int is_b, int has_signal,
                                        __global const char *signal, int out_lda) {
    const int row = (int)get_global_id(0);
    if (row >= rows) {
        return;
    }
    if (out_lda < K) {
        return;
    }
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(row, rank, noise_seed, is_b, el);
    __global char *dst = out + (size_t)row * (size_t)out_lda;
    for (int l = 0; l < K; ++l) {
        const int pos = (int)(char)el[pairs[(size_t)l * 2]];
        const int neg = (int)(char)el[pairs[(size_t)l * 2 + 1]];
        int sig = 0;
        if (has_signal) {
            sig = (int)signal[(size_t)row * (size_t)K + (size_t)l];
        }
        dst[l] = (char)(sig + (pos - neg));
    }
    for (int l = K; l < out_lda; ++l) {
        dst[l] = 0;
    }
}

/* Uniform noise rows for ocl_noisy_matrix_rowmajor_wg: one work item per row,
 * rank bytes per row at el_out + row*rank. */
__kernel void ocl_uniform_rows(__global const uchar *noise_seed, int rows, int rank, int is_b,
                               __global uchar *el_out) {
    const int row = (int)get_global_id(0);
    if (row >= rows) {
        return;
    }
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(row, rank, noise_seed, is_b, el);
    __global uchar *dst = el_out + (size_t)row * (size_t)rank;
    for (int j = 0; j < rank; ++j) {
        dst[j] = el[j];
    }
}

/* Same bytes as ocl_uniform_rows (rank 128): one work item per 32-byte block b = 4 row + j, the
 * keyed single-block BLAKE3 of (1 + b, zeros, label) with every word in registers, then
 * (digest & 63) - 16 per byte. ocl_uniform_rows builds the same hash through private byte arrays
 * and took 8.7 ms per 131072 rows on the A770. */
__kernel void ocl_uniform_rows_b3(__global const uchar *noise_seed, int rows, int is_b,
                                  __global uchar *el_out) {
    const int b = (int)get_global_id(0);
    if (b >= rows * 4) {
        return;
    }
    uint cv[8];
    d_b3_key_words_g(noise_seed, cv);
    uint m[16];
    m[0] = (uint)(1 + b);
    for (int i = 1; i < 8; ++i) {
        m[i] = 0u;
    }
    for (int i = 0; i < 8; ++i) {
        const int o = 4 * i;
        m[8 + i] = is_b ? ((uint)CP_SEED_LABEL_B[o] | ((uint)CP_SEED_LABEL_B[o + 1] << 8) |
                           ((uint)CP_SEED_LABEL_B[o + 2] << 16) | ((uint)CP_SEED_LABEL_B[o + 3] << 24))
                        : ((uint)CP_SEED_LABEL_A[o] | ((uint)CP_SEED_LABEL_A[o + 1] << 8) |
                           ((uint)CP_SEED_LABEL_A[o + 2] << 16) | ((uint)CP_SEED_LABEL_A[o + 3] << 24));
    }
    d_b3_compress_words(cv, m, 64u, 0ul,
                        D_B3_CHUNK_START | D_B3_CHUNK_END | D_B3_ROOT | D_B3_KEYED);
    uchar16 lo = as_uchar16((uint4)(cv[0], cv[1], cv[2], cv[3]));
    uchar16 hi = as_uchar16((uint4)(cv[4], cv[5], cv[6], cv[7]));
    lo = (lo & (uchar16)(CP_RANGE_MASK)) - (uchar16)(CP_ZERO_PT);
    hi = (hi & (uchar16)(CP_RANGE_MASK)) - (uchar16)(CP_ZERO_PT);
    __global uchar *dst = el_out + (size_t)b * 32;
    vstore16(lo, 0, dst);
    vstore16(hi, 1, dst);
}

/* Same result as ocl_noisy_matrix_rowmajor, one work group per row: the row's
 * uniform values sit in local memory and each work item handles 16 consecutive
 * k with vector loads/stores, so neighbouring lanes touch neighbouring bytes. */
__kernel void ocl_noisy_matrix_rowmajor_wg(__global char *out, __global const uchar *el_rows,
                                           __global const uint *pairs, int rows, int K, int rank,
                                           int has_signal, __global const char *signal,
                                           int out_lda) {
    __local uchar el[R_RANK];
    const int row = (int)get_group_id(0);
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    if (row >= rows || out_lda < K) {
        return; /* uniform across the work group */
    }
    for (int j = lid; j < rank; j += lsz) {
        el[j] = el_rows[(size_t)row * (size_t)rank + (size_t)j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __global char *dst = out + (size_t)row * (size_t)out_lda;
    __global const char *src = signal + (size_t)row * (size_t)K;
    for (int l0 = lid * 16; l0 < K; l0 += lsz * 16) {
        const int n = min(16, K - l0);
        char v[16];
        for (int j = 0; j < n; ++j) {
            const int l = l0 + j;
            const int pos = (int)(char)el[pairs[(size_t)l * 2]];
            const int neg = (int)(char)el[pairs[(size_t)l * 2 + 1]];
            const int sig = has_signal ? (int)src[l] : 0;
            v[j] = (char)(sig + (pos - neg));
        }
        if (n == 16) {
            vstore16(vload16(0, v), 0, dst + l0);
        } else {
            for (int j = 0; j < n; ++j) {
                dst[l0 + j] = v[j];
            }
        }
    }
    for (int l = K + lid; l < out_lda; l += lsz) {
        dst[l] = 0;
    }
}

#ifndef NSG_KPL
#define NSG_KPL 4 /* consecutive k per work item per unit: 4 or 16 */
#endif
#ifndef NSG_UNROLL
#define NSG_UNROLL 8 /* units whose signal loads are issued before the first store */
#endif
#ifndef NSG_DIAG
#define NSG_DIAG 0 /* timing diagnostics only (wrong output): 1 = copy, 2 = synthetic pair indices */
#endif
#define NSG_LANES 32                     /* work items per row */
#define NSG_ROWS_PER_WG (256 / NSG_LANES) /* 8 rows per 256-item work group */
#define NSG_UNIT (NSG_LANES * NSG_KPL)
#define NSG_STEP (NSG_UNIT * NSG_UNROLL)
#define NSG_LOOKUP(p) ((int)(char)et[(p)])

/* Same result as ocl_noisy_matrix_rowmajor_wg; plain OpenCL 1.2, no vendor extensions. 32 work
 * items per row, 8 rows per work group, the rows' 128 uniform bytes in local memory. Each work item
 * handles 4 consecutive k per unit, so 32 neighbouring items cover 128 contiguous bytes, and 8 units'
 * loads are issued before their stores. Measured on the A770 (131072 x 4096): the memory pattern
 * decides the time. 16 bytes per item per access copies 1 GB in 5.5 ms, 4 bytes per item with 8 units
 * in flight in 2.7 ms; the full kernel takes 3.0 ms. (A sub-group-shuffle variant of the lookups
 * cost 3 ms more and was removed.) Needs rank == 128 and K % NSG_STEP == 0 (1024 by default). */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
ocl_noisy_matrix_rowmajor_sg(__global char *out, __global const uchar *el_rows,
                             __global const uint *pairs, int rows, int K, int has_signal,
                             __global const char *signal, int out_lda) {
    __local uint tbl32[NSG_ROWS_PER_WG * 32];
    const int lid = (int)get_local_id(0);
    const int rsub = lid / NSG_LANES;
    const int lane = lid % NSG_LANES;
    const int row_raw = (int)get_group_id(0) * NSG_ROWS_PER_WG + rsub;
    const int valid = row_raw < rows && out_lda >= K;
    const int row = min(row_raw, rows - 1);
    /* 32 items x 4 bytes = the row's 128 uniform bytes */
    tbl32[rsub * 32 + lane] = ((const __global uint *)(el_rows + (size_t)row * 128))[lane];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!valid) {
        return; /* after the barrier */
    }
    const __local uchar *et = (const __local uchar *)(tbl32 + rsub * 32);
    __global char *dst = out + (size_t)row * (size_t)out_lda;
    __global const char *src = signal + (size_t)row * (size_t)K;

    for (int kb = 0; kb < K; kb += NSG_STEP) {
#if NSG_KPL == 4
        uchar4 sv[NSG_UNROLL];
        for (int u = 0; u < NSG_UNROLL; ++u) {
            const int k0 = kb + u * NSG_UNIT + lane * 4;
            sv[u] = has_signal ? vload4(0, (__global const uchar *)src + k0) : (uchar4)(0);
        }
        for (int u = 0; u < NSG_UNROLL; ++u) {
            const int k0 = kb + u * NSG_UNIT + lane * 4;
            uchar4 o = sv[u];
#if NSG_DIAG == 2
            const uint4 pa = (uint4)((uint)(k0 * 7 + lane) & 127u, (uint)(k0 * 11 + lane) & 127u,
                                     (uint)(k0 * 13 + lane) & 127u, (uint)(k0 * 17 + lane) & 127u);
            const uint4 pb = pa ^ (uint4)(33u, 65u, 97u, 5u);
#elif NSG_DIAG == 0
            const uint4 pa = *(const __global uint4 *)(pairs + (size_t)k0 * 2);
            const uint4 pb = *(const __global uint4 *)(pairs + (size_t)k0 * 2 + 4);
#endif
#if NSG_DIAG != 1
            o.s0 = (uchar)((int)(char)sv[u].s0 + NSG_LOOKUP(pa.x) - NSG_LOOKUP(pa.y));
            o.s1 = (uchar)((int)(char)sv[u].s1 + NSG_LOOKUP(pa.z) - NSG_LOOKUP(pa.w));
            o.s2 = (uchar)((int)(char)sv[u].s2 + NSG_LOOKUP(pb.x) - NSG_LOOKUP(pb.y));
            o.s3 = (uchar)((int)(char)sv[u].s3 + NSG_LOOKUP(pb.z) - NSG_LOOKUP(pb.w));
#endif
            vstore4(o, 0, (__global uchar *)dst + k0);
        }
#elif NSG_KPL == 16
        uchar16 sv[NSG_UNROLL];
        for (int u = 0; u < NSG_UNROLL; ++u) {
            const int k0 = kb + u * NSG_UNIT + lane * 16;
            sv[u] = has_signal ? vload16(0, (__global const uchar *)src + k0) : (uchar16)(0);
        }
        for (int u = 0; u < NSG_UNROLL; ++u) {
            const int k0 = kb + u * NSG_UNIT + lane * 16;
            uchar16 o = sv[u];
#if NSG_DIAG != 1
            uchar sb[16];
            uchar r[16];
            vstore16(o, 0, sb);
            const __global uint4 *pp = (const __global uint4 *)(pairs + (size_t)k0 * 2);
            for (int q = 0; q < 8; ++q) {
                const uint4 pq = pp[q]; /* pairs of k0+2q (x,y) and k0+2q+1 (z,w) */
                r[2 * q] = (uchar)((int)(char)sb[2 * q] + NSG_LOOKUP(pq.x) - NSG_LOOKUP(pq.y));
                r[2 * q + 1] = (uchar)((int)(char)sb[2 * q + 1] + NSG_LOOKUP(pq.z) - NSG_LOOKUP(pq.w));
            }
            o = vload16(0, r);
#endif
            vstore16(o, 0, (__global uchar *)dst + k0);
        }
#else
#error "NSG_KPL must be 4 or 16"
#endif
    }
    for (int l = K + lane; l < out_lda; l += NSG_LANES) {
        dst[l] = 0;
    }
}

#ifndef NPA_UNROLL
#define NPA_UNROLL 4 /* 64-k units per loop iteration (K % (64 * NPA_UNROLL) == 0) */
#endif
/* Noisy A in gemmstone's packed-columns layout for the XeHPG kernel (MatrixLayout::Pc, 16-row
 * panels, crosspack 4): panel p (rows 16p..16p+15) starts at p * panel_stride_dw dwords and element
 * (m, k) sits at byte (k / 4) * 64 + (m % 16) * 4 + k % 4, i.e. dword (k / 4) * 16 + m % 16 holds
 * k..k+3 of row m. One 256-item work group per panel. Per unit of 64 k the signal tile (16 rows x
 * 64 bytes) is loaded with 16 items per row (coalesced) into local memory, and item w = 16 g + r
 * produces the dword of row r, k-group g, so the group's stores are 1 KB contiguous. Same bytes as
 * ocl_noisy_matrix_rowmajor (CP_OCL_PREP_CHECK=1 compares them). Plain OpenCL 1.2. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
ocl_noisy_matrix_a_packed16(__global uint *out, __global const uint *el_rows,
                            __global const uint *pairs, int rows, int K,
                            __global const char *signal, int panel_stride_dw) {
    __local uint sig[NPA_UNROLL * 16 * 17]; /* 17: no bank conflicts on the transposed read */
    __local uint el32[16 * 33];             /* 33: row stride against bank conflicts */
    __local uchar p0[NPA_UNROLL * 64];
    __local uchar p1[NPA_UNROLL * 64];
    const int w = (int)get_local_id(0);
    const int panel = (int)get_group_id(0);
    const int row0 = panel * 16;
    if (row0 >= rows) {
        return; /* uniform across the work group */
    }
    for (int j = w; j < 512; j += 256) {
        el32[(j >> 5) * 33 + (j & 31)] = el_rows[(size_t)row0 * 32 + (size_t)j];
    }
    const __local uchar *el = (const __local uchar *)el32;
    __global uint *dst = out + (size_t)panel * (size_t)panel_stride_dw;
    const int r = w & 15;  /* compute and store: row in panel */
    const int g = w >> 4;  /* compute and store: k group of 4 inside a 64-k unit */
    const int lr = w >> 4; /* load: row in panel */
    const int ld = w & 15; /* load: dword inside the row's 64 bytes */
    const __local uchar *e = el + r * 132;
    for (int kb = 0; kb < K; kb += 64 * NPA_UNROLL) {
        barrier(CLK_LOCAL_MEM_FENCE); /* the previous iteration's readers are done */
        for (int u = 0; u < NPA_UNROLL; ++u) {
            sig[(u * 16 + lr) * 17 + ld] = *(const __global uint *)(
                    signal + (size_t)(row0 + lr) * (size_t)K + (size_t)(kb + u * 64 + ld * 4));
        }
        for (int i = w; i < 64 * NPA_UNROLL; i += 256) {
            p0[i] = (uchar)pairs[(size_t)(kb + i) * 2];
            p1[i] = (uchar)pairs[(size_t)(kb + i) * 2 + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int u = 0; u < NPA_UNROLL; ++u) {
            const uint sv = sig[(u * 16 + r) * 17 + g];
            const int kl = u * 64 + g * 4;
            uint o = 0;
            for (int b = 0; b < 4; ++b) {
                const uint v = (sv >> (8 * b)) + (uint)e[p0[kl + b]] - (uint)e[p1[kl + b]];
                o |= (v & 0xffu) << (8 * b);
            }
            dst[(size_t)((kb >> 2) + u * 16) * 16 + (size_t)w] = o;
        }
    }
}

/* Per-job B for the oneDNN XeHPG kernel in gemmstone's packed-rows layout (MatrixLayout::Pr, 64-column
 * panels, crosspack 32): panel p (columns 64p..64p+63) starts at byte p * 64 * ldb, element (k, n) sits
 * at (k / 32) * 2048 + (n % 64) * 32 + k % 32 inside it. Input: column-major B (column n at n * ldb),
 * as written by ocl_noisy_matrix_colmajor. One work item per 32-byte chunk; neighbouring items write
 * neighbouring chunks of a panel. Plain OpenCL 1.2. */
/* Noisy B (no signal) written straight into gemmstone's packed-rows layout: panels of Q columns at
 * p * panel_stride_dw, element (k, c) at (k / cp) * Q * cp + (c % Q) * cp + k % cp inside the panel
 * (Q <= NPB_QMAX, cp % 4 == 0, 64 % cp == 0, K % 64 == 0). Same bytes as ocl_noisy_matrix_colmajor
 * followed by a repack, without the column-major copy. One work group of 256 per panel: the panel's
 * uniform rows stay in local memory; inside a 64-k unit the panel's 16 * Q output dwords are contiguous,
 * so consecutive work items write consecutive dwords. */
#ifndef NPB_QMAX
#define NPB_QMAX 64
#endif
__kernel void ocl_noisy_matrix_b_packed(__global uint *out, __global const uint *el_rows,
                                        __global const uint *pairs, int cols, int K, int Q, int cp,
                                        int panel_stride_dw) {
    __local uint el32[NPB_QMAX * 33]; /* 33: row stride against bank conflicts */
    __local uchar p0[64];
    __local uchar p1[64];
    const int w = (int)get_local_id(0);
    const int panel = (int)get_group_id(0);
    const int col0 = panel * Q;
    if (col0 >= cols) {
        return; /* uniform across the work group */
    }
    for (int j = w; j < Q * 32; j += 256) {
        el32[(j >> 5) * 33 + (j & 31)] = el_rows[(size_t)col0 * 32 + (size_t)j];
    }
    const __local uchar *el = (const __local uchar *)el32;
    __global uint *dst = out + (size_t)panel * (size_t)panel_stride_dw;
    const int cpd = cp >> 2;    /* dwords per column in a k block of cp */
    const int blk_dw = Q * cpd; /* dwords per k block */
    for (int kb = 0; kb < K; kb += 64) {
        barrier(CLK_LOCAL_MEM_FENCE); /* the previous iteration's readers are done */
        if (w < 64) {
            p0[w] = (uchar)pairs[(size_t)(kb + w) * 2];
            p1[w] = (uchar)pairs[(size_t)(kb + w) * 2 + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __global uint *unit = dst + (size_t)kb * (size_t)Q / 4;
        for (int j = w; j < 16 * Q; j += 256) {
            const int blk = j / blk_dw;
            const int d = j - blk * blk_dw;
            const int c = d / cpd;
            const int kl = blk * cp + (d - c * cpd) * 4;
            const __local uchar *e = el + c * 132;
            uint o = 0;
            for (int b = 0; b < 4; ++b) {
                const uint v = (uint)e[p0[kl + b]] - (uint)e[p1[kl + b]];
                o |= (v & 0xffu) << (8 * b);
            }
            unit[j] = o;
        }
    }
}

/* A column-major (gemmstone layout N): row-parallel like rowmajor; write out[k*lda+row]. */
__kernel void ocl_noisy_matrix_a_colmajor(__global char *out, __global const uchar *noise_seed,
                                         __global const uint *pairs, int rows, int K, int rank,
                                         __global const char *signal, int out_lda) {
    const int row = (int)get_global_id(0);
    if (row >= rows || out_lda < rows) {
        return;
    }
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(row, rank, noise_seed, 0, el);
    for (int k = 0; k < K; ++k) {
        const int pos = (int)(char)el[pairs[(size_t)k * 2]];
        const int neg = (int)(char)el[pairs[(size_t)k * 2 + 1]];
        const int sig = (int)signal[(size_t)row * (size_t)K + (size_t)k];
        out[(size_t)k * (size_t)out_lda + (size_t)row] = (char)(sig + (pos - neg));
    }
}

/* Zero column-major column tails when lda > valid_rows (one work item per column). */
__kernel void ocl_pad_colmajor_col_tails(__global char *out, int cols, int valid_rows, int lda) {
    const int col = (int)get_global_id(0);
    if (col >= cols || lda <= valid_rows) {
        return;
    }
    __global char *dst = out + (size_t)col * (size_t)lda;
    for (int p = valid_rows; p < lda; ++p) {
        dst[p] = 0;
    }
}

/* gemmstone column-major B (zero signal): column j at out + j*ldb. */
__kernel void ocl_noisy_matrix_colmajor(__global char *out, __global const uchar *noise_seed,
                                        __global const uint *pairs, int cols, int K, int rank,
                                        int is_b, int ldb) {
    const int col = (int)get_global_id(0);
    if (col >= cols || ldb < K) {
        return;
    }
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(col, rank, noise_seed, is_b, el);
    __global char *dst = out + (size_t)col * (size_t)ldb;
    for (int k = 0; k < K; ++k) {
        const int pos = (int)(char)el[pairs[(size_t)k * 2]];
        const int neg = (int)(char)el[pairs[(size_t)k * 2 + 1]];
        dst[k] = (char)(pos - neg);
    }
    for (int k = K; k < ldb; ++k) {
        dst[k] = 0;
    }
}

/* gemmstone row-major B (K×N, zero signal): column-parallel; write out[k*ldb+n]. */
__kernel void ocl_noisy_matrix_b_rowmajor(__global char *out, __global const uchar *noise_seed,
                                          __global const uint *pairs, int K, int N, int rank,
                                          int ldb) {
    const int n = (int)get_global_id(0);
    if (n >= N || ldb < N) {
        return;
    }
    uchar el[R_RANK];
    ocl_generate_uniform_row_glob(n, rank, noise_seed, 1, el);
    for (int k = 0; k < K; ++k) {
        const int pos = (int)(char)el[pairs[(size_t)k * 2]];
        const int neg = (int)(char)el[pairs[(size_t)k * 2 + 1]];
        out[(size_t)k * (size_t)ldb + (size_t)n] = (char)(pos - neg);
    }
}

/* Zero row-major matrix row tails when ldb > valid_cols (one work item per row). */
__kernel void ocl_pad_rowmajor_row_tails(__global char *out, int rows, int valid_cols, int ldb) {
    const int row = (int)get_global_id(0);
    if (row >= rows || ldb <= valid_cols) {
        return;
    }
    __global char *dst = out + (size_t)row * (size_t)ldb;
    for (int p = valid_cols; p < ldb; ++p) {
        dst[p] = 0;
    }
}

/* Align-test: device get_random_hash spot check. */
__kernel void ocl_test_get_random_hash(int index, __global const uchar *seed, int is_b,
                                       int prepend_index, __global uchar *out) {
    uchar digest[32];
    d_get_random_hash_glob(index, seed, is_b, prepend_index, digest);
    for (int i = 0; i < 32; i++) {
        out[i] = digest[i];
    }
}

/* A's noise seed on the device (mirrors pearl_a_noise_seed_from_hash in cp_noise.c):
     root_a = Merkle root of the signal A (32 bytes at root[0..31]);
     salted: root_a = BLAKE3_keyed(SALT_A, root_a || le32(m) || zeros), a 64-byte message;
     a_seed = BLAKE3(b_noise_seed || root_a), a 64-byte message.
   Writes a_seed to seed_out (the noise kernels' seed) and key_out (the scan's jackpot key), and copies
   the root to root_out (the share witness root). One work item. */
constant uint CP_SEED_SALT_A_W[8] = {
    0x6C404982u, 0x1615EDA0u, 0x92F61696u, 0xF876F0FCu, 0x2ADBDB92u, 0x52B82370u, 0x1977D4F0u, 0x7B0190C3u
};

__kernel void ocl_a_noise_seed(__global const uchar *root, uint m, int salted, uint8 b_seed,
                               __global uchar *seed_out, __global uchar *key_out,
                               __global uchar *root_out) {
    if (get_global_id(0) != 0) {
        return;
    }
    const uint flags = D_B3_CHUNK_START | D_B3_CHUNK_END | D_B3_ROOT;
    uint r[8];
    for (int i = 0; i < 8; i++) {
        r[i] = d_b3_load32_g(root + 4 * i);
        d_b3_store32_g(root_out + 4 * i, r[i]);
    }
    uint msg[16];
    if (salted) {
        uint cv[8];
        for (int i = 0; i < 8; i++) {
            cv[i] = CP_SEED_SALT_A_W[i];
            msg[i] = r[i];
        }
        msg[8] = m;
        for (int i = 9; i < 16; i++) {
            msg[i] = 0u;
        }
        d_b3_compress_words(cv, msg, 64u, 0ul, flags | D_B3_KEYED);
        for (int i = 0; i < 8; i++) {
            r[i] = cv[i];
        }
    }
    const uint bw[8] = {b_seed.s0, b_seed.s1, b_seed.s2, b_seed.s3, b_seed.s4, b_seed.s5, b_seed.s6, b_seed.s7};
    uint cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = D_B3_IV[i];
        msg[i] = bw[i];
        msg[8 + i] = r[i];
    }
    d_b3_compress_words(cv, msg, 64u, 0ul, flags);
    for (int i = 0; i < 8; i++) {
        d_b3_store32_g(seed_out + 4 * i, cv[i]);
        d_b3_store32_g(key_out + 4 * i, cv[i]);
    }
}
