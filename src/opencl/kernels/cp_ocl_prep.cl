/* OpenCL matrix prep: random A, noise, fused coalesced prepack (signed s8s8). */

#ifndef MR
#define MR 8
#endif
#ifndef NR
#define NR 16
#endif
#ifndef KR
#define KR 256
#endif
#ifndef R_RANK
#define R_RANK 256
#endif

#define CP_RANGE_MASK 63
#define CP_ZERO_PT 16
#define CP_B3_LINES 8
#define K_GROUPS (KR / 4)
#define COLS_PER_GROUP 8
#define KG_BYTES_A (MR * 4)
#define KG_SLICE_B ((NR / COLS_PER_GROUP) * 32)
#define MICRO_M (128 / MR)
#define MICRO_N (128 / NR)
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

    if (col == 0) {
        const size_t block_base =
                ((size_t)jm * (size_t)blocks_k + (size_t)kb) * (size_t)MACRO_KB_BLOCK_B;
        for (int kg = 0; kg < K_GROUPS; ++kg) {
            const size_t dst = block_base + (size_t)kg * (size_t)MACRO_KG_STRIP_B +
                               (size_t)tc * (size_t)KG_SLICE_B;
            for (int jg = 0; jg < NR / COLS_PER_GROUP; ++jg) {
                for (int c = 0; c < COLS_PER_GROUP; ++c) {
                    for (int ko = 0; ko < 4; ++ko) {
                        b_pre_out[dst + (size_t)jg * 32 + (size_t)c * 4 + (size_t)ko] =
                                stripe[(size_t)(jg * COLS_PER_GROUP + c)][(size_t)kg * 4 + (size_t)ko];
                    }
                }
            }
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

    if (row == 0) {
        const size_t block_base =
                ((size_t)im * (size_t)blocks_k + (size_t)kb) * (size_t)MACRO_KB_BLOCK_A;
        for (int kg = 0; kg < K_GROUPS; ++kg) {
            const size_t dst = block_base + (size_t)kg * (size_t)MACRO_KG_STRIP_A +
                               (size_t)tr * (size_t)KG_BYTES_A;
            for (int r = 0; r < MR; ++r) {
                for (int ko = 0; ko < 4; ++ko) {
                    a_pre_out[dst + (size_t)r * 4 + (size_t)ko] =
                            stripe[r][(size_t)kg * 4 + (size_t)ko];
                }
            }
        }
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
