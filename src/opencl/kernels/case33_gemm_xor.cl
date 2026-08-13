// Case 3.3: Case 3.2 int8 GEMM + milestoned MR×NR tile XOR + optional fused device jackpot.
//
// fuse_jackpot=1 (mining): milestone XORs fold online into private msg[16]; BLAKE3 + target
// compare on-device. Host readback is only found_flag (+ t_rows/t_cols on hit).
// fuse_jackpot=0: legacy tile_xor writeback for correctness benchmarks.
//
// Private memory (fuse_jackpot mining path; source arrays + compiler stack):
//   acc[NR×MR]  8×8: 256 B   8×16: 512 B
//   msg[16]           64 B         64 B   (online milestone fold; replaces old ms_xor[32])
//   a_pack[MR]        32 B         32 B
//   digest[8]         32 B         32 B
//   b3_compress64     ~192 B       ~192 B  (inlined v/m/t; Beignet may reserve for whole kernel)
// Measured CL_KERNEL_PRIVATE_MEM_SIZE on Beignet (Haswell GT1, scalar):
//   8×8: 384 B/WI (was 1152 with ms_xor[32] + post-GEMM msg fold)
//   8×16: 1152 B/WI (acc spill dominates; unchanged after ms_xor removal)

#ifndef MR
#define MR 8
#endif
#ifndef NR
#define NR 16
#endif
#ifndef KR
#define KR 128
#endif
#ifndef MACRO_M
#define MACRO_M 128
#endif
#ifndef MACRO_N
#define MACRO_N 128
#endif
#ifndef PANEL_A
#define PANEL_A (KR * MR)
#endif
#ifndef PANEL_B
#define PANEL_B (KR * NR)
#endif
#ifndef COLS_PER_GROUP
#define COLS_PER_GROUP 8
#endif
#ifndef RANK
#define RANK 4
#endif
#ifndef KGROUPS
#define KGROUPS (KR / RANK)
#endif
#ifndef KG_BYTES_A
#define KG_BYTES_A (MR * RANK)
#endif
#ifndef MICRO_M
#define MICRO_M (MACRO_M / MR)
#endif
#ifndef MICRO_N
#define MICRO_N (MACRO_N / NR)
#endif
#ifndef KG_SLICE_B
#define KG_SLICE_B ((NR / COLS_PER_GROUP) * 32)
#endif
#ifndef MACRO_KG_STRIP_A
#define MACRO_KG_STRIP_A (MICRO_M * KG_BYTES_A)
#endif
#ifndef MACRO_KG_STRIP_B
#define MACRO_KG_STRIP_B (MICRO_N * KG_SLICE_B)
#endif
#ifndef MACRO_KB_BLOCK_A
#define MACRO_KB_BLOCK_A (KGROUPS * MACRO_KG_STRIP_A)
#endif
#ifndef MACRO_KB_BLOCK_B
#define MACRO_KB_BLOCK_B (KGROUPS * MACRO_KG_STRIP_B)
#endif

#define PP_JACKPOT_WORDS 16
#define PP_LROT 13
#ifndef R_RANK
#define R_RANK 128
#endif
#ifndef PP_MAX_MILESTONES
#define PP_MAX_MILESTONES 32
#endif
/* KR == R_RANK: one packed K-panel is one jackpot milestone. */

#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define CASE32_USE_DOT 1
#endif
#if defined(CASE32_FORCE_DPI)
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define CASE32_USE_DOT 1
#endif

inline uint pp_rotl32(uint x, int s) { return (x << s) | (x >> (32 - s)); }

inline uint b3_rotr32(uint x, int n) { return (x >> n) | (x << (32 - n)); }

inline void b3_g(uint *v, int a, int b, int c, int d, uint x, uint y) {
    v[a] += v[b] + x;
    v[d] = b3_rotr32(v[d] ^ v[a], 16);
    v[c] += v[d];
    v[b] = b3_rotr32(v[b] ^ v[c], 12);
    v[a] += v[b] + y;
    v[d] = b3_rotr32(v[d] ^ v[a], 8);
    v[c] += v[d];
    v[b] = b3_rotr32(v[b] ^ v[c], 7);
}

inline void b3_compress64(__global const uint *key8, const uint *msg16, uint *out8) {
    const uint kIV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                         0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    uint v[16] = {key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7],
                  kIV[0],  kIV[1],  kIV[2],  kIV[3],  0u,      0u,      64u,     0x1Bu};
    uint m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = msg16[i];
    }
    const uchar kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    for (int round = 0; round < 7; ++round) {
        b3_g(v, 0, 4, 8, 12, m[0], m[1]);
        b3_g(v, 1, 5, 9, 13, m[2], m[3]);
        b3_g(v, 2, 6, 10, 14, m[4], m[5]);
        b3_g(v, 3, 7, 11, 15, m[6], m[7]);
        b3_g(v, 0, 5, 10, 15, m[8], m[9]);
        b3_g(v, 1, 6, 11, 12, m[10], m[11]);
        b3_g(v, 2, 7, 8, 13, m[12], m[13]);
        b3_g(v, 3, 4, 9, 14, m[14], m[15]);
        if (round < 6) {
            uint t[16];
            for (int i = 0; i < 16; ++i) {
                t[i] = m[kPerm[i]];
            }
            for (int i = 0; i < 16; ++i) {
                m[i] = t[i];
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        out8[i] = v[i] ^ v[i + 8];
    }
}

inline bool digest_beats_target(const uint digest[8], __global const uint *bound) {
    for (int w = 7; w >= 0; --w) {
        if (digest[w] < bound[w]) {
            return true;
        }
        if (digest[w] > bound[w]) {
            return false;
        }
    }
    return true;
}

inline int case32_dot4(int acc, int a_pack, int b_pack) {
#if defined(CASE32_USE_ASM_DOT)
    __asm volatile("v_dot4c_i32_i8 %0, %1, %2" : "+v"(acc) : "v"(a_pack), "v"(b_pack));
    return acc;
#elif defined(CASE32_USE_BUILTIN_SDOT4)
    return __builtin_amdgcn_sdot4(a_pack, b_pack, acc, false);
#elif defined(CASE32_USE_DOT)
    return dot_acc_sat(as_char4(a_pack), as_char4(b_pack), acc);
#else
    char4 a = as_char4(a_pack);
    char4 b = as_char4(b_pack);
    acc += (int)a.s0 * (int)b.s0;
    acc += (int)a.s1 * (int)b.s1;
    acc += (int)a.s2 * (int)b.s2;
    acc += (int)a.s3 * (int)b.s3;
    return acc;
#endif
}

__kernel void case33_macro_gemm_xor(__global const char *a_pre, __global const char *b_pre,
                                    __global uint *tile_xor, int N, int blocks_k,
                                    int blocks_per_milestone, int num_milestones, int tile_count,
                                    int macro_rows, int macro_cols, int xor_after_milestone,
                                    int mb_begin, int compact_xor, __global const uint *a_key8,
                                    __global const uint *bound, __global int *found_flag,
                                    __global int *out_t_rows, __global int *out_t_cols,
                                    int fuse_jackpot) {
    if (fuse_jackpot && found_flag != 0 && *found_flag != 0) {
        return;
    }

    const int mb = mb_begin + (int)get_group_id(0);
    const int jm = mb / macro_rows;
    const int im = mb % macro_rows;

    const int col0 = jm * MACRO_N;
    const int tr0 = im * MICRO_M;
    const int tc0 = jm * MICRO_N;

    const int lid = (int)get_local_id(0);
#if CASE32_WI_ROWMAJOR
    const int tr = lid / MICRO_N;
    const int tc = lid % MICRO_N;
#else
    const int tr = lid % MICRO_M;
    const int tc = lid / MICRO_M;
#endif

    const int micro_col0 = col0 + tc * NR;
    const int tr_global = tr0 + tr;
    const int tc_global = tc0 + tc;
    const int tile_cols = N / NR;
    const int spatial_id = tr_global * tile_cols + tc_global;

    int acc[NR * MR];
    for (int i = 0; i < NR * MR; ++i) {
        acc[i] = 0;
    }

    uint msg[PP_JACKPOT_WORDS];
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        msg[i] = 0u;
    }
    int ms = 0;
    for (int kb = 0; kb < blocks_k; ++kb) {
#if defined(CASE32_COALESCE)
        const size_t a_kb_base =
                (size_t)im * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_A +
                (size_t)kb * (size_t)MACRO_KB_BLOCK_A;
        const size_t b_kb_base =
                (size_t)jm * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_B +
                (size_t)kb * (size_t)MACRO_KB_BLOCK_B;

        for (int kg = 0; kg < KGROUPS; ++kg) {
            __global const char *a_kg =
                    a_pre + a_kb_base + (size_t)kg * (size_t)MACRO_KG_STRIP_A +
                    (size_t)tr * (size_t)KG_BYTES_A;
            __global const char *b_kg =
                    b_pre + b_kb_base + (size_t)kg * (size_t)MACRO_KG_STRIP_B +
                    (size_t)tc * (size_t)KG_SLICE_B;

            int a_pack[MR];
            for (int i = 0; i < MR; ++i) {
                a_pack[i] = as_int(vload4(0, a_kg + (size_t)i * RANK));
            }

            for (int jg = 0; jg < NR / COLS_PER_GROUP; ++jg) {
                __global const char *b_jg = b_kg + (size_t)jg * 32;
                for (int col = 0; col < COLS_PER_GROUP; ++col) {
                    const int j = jg * COLS_PER_GROUP + col;
                    const int b0 = as_int(vload4(0, b_jg + (size_t)col * RANK));
                    const int base = j * MR;
                    for (int i = 0; i < MR; ++i) {
                        acc[base + i] = case32_dot4(acc[base + i], a_pack[i], b0);
                    }
                }
            }
        }
#else
        __global const char *a_tile = a_pre + (size_t)tr_global * (size_t)blocks_k *
                                              (size_t)PANEL_A +
                                      (size_t)kb * (size_t)PANEL_A;
        __global const char *b_tile = b_pre + (size_t)tc_global * (size_t)blocks_k *
                                              (size_t)PANEL_B +
                                      (size_t)kb * (size_t)PANEL_B;

        for (int kg = 0; kg < KGROUPS; ++kg) {
            __global const char *a_kg = a_tile + (size_t)kg * (size_t)KG_BYTES_A;
            int a_pack[MR];
            for (int i = 0; i < MR; ++i) {
                a_pack[i] = as_int(vload4(0, a_kg + (size_t)i * RANK));
            }

            for (int jg = 0; jg < NR / COLS_PER_GROUP; ++jg) {
                __global const char *b_jg =
                        b_tile + ((size_t)jg * (size_t)KGROUPS + (size_t)kg) * 32;
                for (int col = 0; col < COLS_PER_GROUP; ++col) {
                    const int j = jg * COLS_PER_GROUP + col;
                    const int b0 = as_int(vload4(0, b_jg + (size_t)col * RANK));
                    const int base = j * MR;
                    for (int i = 0; i < MR; ++i) {
                        acc[base + i] = case32_dot4(acc[base + i], a_pack[i], b0);
                    }
                }
            }
        }
#endif
        /* One milestone per KR panel (KR == R_RANK). Cumulative acc across kb. */
        {
            uint x = 0u;
            for (int i = 0; i < NR * MR; ++i) {
                x ^= as_uint(acc[i]);
            }
            if (xor_after_milestone) {
                if (fuse_jackpot) {
                    if (ms < PP_MAX_MILESTONES) {
                        const int tid = ms % PP_JACKPOT_WORDS;
                        msg[tid] = pp_rotl32(msg[tid], PP_LROT) ^ x;
                    }
                } else if (compact_xor) {
                    const ulong batch_stride =
                            (ulong)get_local_size(0) * (ulong)num_milestones;
                    const ulong out_base = (ulong)get_group_id(0) * batch_stride +
                                           (ulong)lid * (ulong)num_milestones;
                    tile_xor[out_base + (ulong)ms] = x;
                } else {
                    tile_xor[(ulong)ms * (ulong)tile_count + (ulong)spatial_id] = x;
                }
            }
            ++ms;
        }
    }
    (void)blocks_per_milestone;

    if (!fuse_jackpot || !xor_after_milestone || a_key8 == 0 || bound == 0) {
        return;
    }

    uint digest[8];
    b3_compress64(a_key8, msg, digest);
    if (!digest_beats_target(digest, bound)) {
        return;
    }

    if (found_flag == 0) {
        return;
    }
    if (atomic_cmpxchg(found_flag, 0, 1) != 0) {
        return;
    }

    const int t_rows = im * MACRO_M + tr * MR;
    const int t_cols = jm * MACRO_N + tc * NR;
    if (out_t_rows != 0) {
        *out_t_rows = t_rows;
    }
    if (out_t_cols != 0) {
        *out_t_cols = t_cols;
    }
}
