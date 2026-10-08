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

/* Same bytes as ocl_fused_prepack_a, one 256-wide work group per (im, kb) block.
 * The rows' uniform noise (el_rows, R_RANK bytes per row, from ocl_uniform_rows) and the
 * signal tile are staged in local memory with a 4-byte row pad, so lanes that read the same
 * k of neighbouring rows hit different banks. Output word w of the block is
 * kg * MACRO_M + r (byte offset kg * MACRO_KG_STRIP_A + r * 4), so stores are coalesced. */
#define FPA_WG 256
#define FPA_SW (KR / 4 + 1)
#define FPA_EW (R_RANK / 4 + 1)
__kernel __attribute__((reqd_work_group_size(FPA_WG, 1, 1))) void
ocl_fused_prepack_a_wg(__global uint *a_pre_out, __global const uint *el_rows,
                       __global const uint *pairs, __global const char *a_signal, int K,
                       int blocks_k) {
    __local uint sig[MACRO_M * FPA_SW];
    __local uint el[MACRO_M * FPA_EW];
    __local uchar p0[KR];
    __local uchar p1[KR];

    const int blk = (int)get_group_id(0);
    const int kb = blk % blocks_k;
    const int im = blk / blocks_k;
    const int t = (int)get_local_id(0);
    const int row0 = im * MACRO_M;
    const int k0 = kb * KR;

    for (int i = t; i < MACRO_M * (KR / 16); i += FPA_WG) {
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
    for (int i = t; i < MACRO_M * (R_RANK / 16); i += FPA_WG) {
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
    for (int w = t; w < MACRO_M * K_GROUPS; w += FPA_WG) {
        const int kg = w / MACRO_M;
        const int r = w % MACRO_M;
        const uint s = sig[r * FPA_SW + kg];
        const __local uchar *e = (const __local uchar *)(el + r * FPA_EW);
        uint out = 0;
        for (int b = 0; b < 4; ++b) {
            const int k = kg * 4 + b;
            const uint v = (s >> (8 * b)) + (uint)e[p0[k]] - (uint)e[p1[k]];
            out |= (v & 0xffu) << (8 * b);
        }
        dst[w] = out;
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
