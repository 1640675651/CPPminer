// Case 3.3: Case 3.2 int8 GEMM + milestoned MR×NR tile XOR + optional fused device jackpot.
//
// fuse_jackpot=1 (mining): milestone XORs fold online into private msg[16]; BLAKE3 + target
// compare on-device. Host readback is only found_flag (+ t_rows/t_cols on hit).
// fuse_jackpot=0: legacy tile_xor writeback for correctness benchmarks.
//
// Private memory (fuse_jackpot mining path; source arrays + compiler stack):
//   acc[NR×MR]  4×4: 64 B   8×8: 256 B   8×16: 512 B
//   msg[16]          64 B         64 B         64 B
//   a_pack[MR]       16 B         32 B         32 B
//   digest[8]        32 B         32 B         32 B
//   b3_compress64     ~192 B       ~192 B  (inlined v/m/t; Beignet may reserve for whole kernel)
// Measured CL_KERNEL_PRIVATE_MEM_SIZE on Beignet (Haswell GT1, scalar):
//   8×8: 128 B/WI with CLBlast cpm issue (was 384 with per-element dot4; 1152 with ms_xor)
//   8×16: 1152 B/WI (acc spill dominates)
// Issue shape (matches beignet-fix / case36):
//   !CASE32_PACKED_DOT (scalar / --ocl-issue broadcast): CLBlast GEMMK=0 cpm += aval * bscalar
//   CASE32_PACKED_DOT (--ocl-issue packed or DPI): per-C case32_dot4
//   --ocl-cpm-type int (CASE32_CPM_INT): int4 cpm; default float4 mad. CASE32_NO_DPI blocks
//   auto-enabling KHR DPI when the host requests the scalar cpm nest.

#ifndef MR
#define MR 8
#endif
#ifndef NR
#define NR 16
#endif
#ifndef HASH_NR
#define HASH_NR NR
#endif
#define HASH_REG_TILES_N (HASH_NR / NR)
#if (HASH_NR % NR) != 0 || HASH_REG_TILES_N < 1 || HASH_REG_TILES_N > 2
#error HASH_NR must contain one or two register tiles
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
#ifndef RANK
#define RANK 4
#endif
#ifndef VWM
#define VWM 4
#endif
#ifndef VWN
#define VWN 4
#endif
#define CPM_COLVEC (MR / VWM)
#define CPM_NVEC (NR * CPM_COLVEC)
#if (MR % VWM) != 0 || (NR % VWN) != 0
#error CLBlast-style cpm tile requires MR%VWM==0 and NR%VWN==0
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
#define HASH_MICRO_N (MACRO_N / HASH_NR)
#ifndef KG_SLICE_B
#define KG_SLICE_B (NR * RANK)
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
#ifndef CASE32_USE_LDS
#define CASE32_USE_LDS 0
#endif
#ifndef CASE32_CPM_INT
#define CASE32_CPM_INT 0
#endif
#if CASE32_CPM_INT
typedef int cpm_lane;
typedef int4 cpm_vec;
#define CASE32_CPM_ZERO ((int4)(0))
#define CASE32_CPM_SPLAT(b) ((int4)(b))
#define CASE32_CPM_MAD(aval, bsc, c) ((aval) * CASE32_CPM_SPLAT(bsc) + (c))
#else
typedef float cpm_lane;
typedef float4 cpm_vec;
#define CASE32_CPM_ZERO ((float4)(0.0f))
#define CASE32_CPM_SPLAT(b) ((float4)(b))
#define CASE32_CPM_MAD(aval, bsc, c) mad((aval), CASE32_CPM_SPLAT(bsc), (c))
#endif
#ifndef LDS_KWG
#define LDS_KWG 16
#endif
#define LDS_KGROUPS (LDS_KWG / RANK)
#define LDS_A_STRIDE (MICRO_M * KG_BYTES_A)
#define LDS_B_STRIDE (MICRO_N * KG_SLICE_B)

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
#ifndef CASE32_NO_DPI
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define CASE32_USE_DOT 1
#endif
#endif
#if defined(CASE32_FORCE_DPI)
#ifndef CASE32_NO_DPI
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define CASE32_USE_DOT 1
#endif
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

#if defined(CASE32_USE_ASM_DOT) || defined(CASE32_USE_BUILTIN_SDOT4) || \
        defined(CASE32_USE_DOT) || defined(CASE32_INT_DOT) || \
        defined(CASE32_FORCE_PACKED)
#define CASE32_PACKED_DOT 1
#else
#define CASE32_PACKED_DOT 0
#endif

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

#if !CASE32_PACKED_DOT
/* CLBlast GEMMK=0 issue: cpm[n][m] += apm[m] * bpm[n]
   VWM along M, VWN along N, B broadcast as a scalar. Packed K=4 is the
   unrolled KWI. Default float panel is flushed to int32 each KR (exact:
   |sum|<2^21). CASE32_CPM_INT uses int4 acc (int8 lanes; mul is int32). */

inline cpm_lane case32_char_lane(char4 c, int k) {
    if (k == 0) {
        return (cpm_lane)c.s0;
    }
    if (k == 1) {
        return (cpm_lane)c.s1;
    }
    if (k == 2) {
        return (cpm_lane)c.s2;
    }
    return (cpm_lane)c.s3;
}

inline cpm_vec case32_apm(__private const char4 *ar, int row0, int k) {
    return (cpm_vec)(case32_char_lane(ar[row0], k), case32_char_lane(ar[row0 + 1], k),
                     case32_char_lane(ar[row0 + 2], k), case32_char_lane(ar[row0 + 3], k));
}

inline void case32_cpm_zero(__private cpm_vec *cpm) {
    #pragma unroll
    for (int i = 0; i < CPM_NVEC; ++i) {
        cpm[i] = CASE32_CPM_ZERO;
    }
}

inline void case32_cpm_kgroup(__private cpm_vec *cpm, __private const int *a_pack,
                              __private const int *b_pack) {
    char4 ar[MR];
    char4 br[NR];
    #pragma unroll
    for (int i = 0; i < MR; ++i) {
        ar[i] = as_char4(a_pack[i]);
    }
    #pragma unroll
    for (int j = 0; j < NR; ++j) {
        br[j] = as_char4(b_pack[j]);
    }
    #pragma unroll
    for (int k = 0; k < RANK; ++k) {
        #pragma unroll
        for (int mi = 0; mi < CPM_COLVEC; ++mi) {
            const cpm_vec aval = case32_apm(ar, mi * VWM, k);
            #pragma unroll
            for (int ni = 0; ni < NR / VWN; ++ni) {
                const int j0 = ni * VWN;
                cpm[(j0 + 0) * CPM_COLVEC + mi] = CASE32_CPM_MAD(
                        aval, case32_char_lane(br[j0 + 0], k),
                        cpm[(j0 + 0) * CPM_COLVEC + mi]);
                cpm[(j0 + 1) * CPM_COLVEC + mi] = CASE32_CPM_MAD(
                        aval, case32_char_lane(br[j0 + 1], k),
                        cpm[(j0 + 1) * CPM_COLVEC + mi]);
                cpm[(j0 + 2) * CPM_COLVEC + mi] = CASE32_CPM_MAD(
                        aval, case32_char_lane(br[j0 + 2], k),
                        cpm[(j0 + 2) * CPM_COLVEC + mi]);
                cpm[(j0 + 3) * CPM_COLVEC + mi] = CASE32_CPM_MAD(
                        aval, case32_char_lane(br[j0 + 3], k),
                        cpm[(j0 + 3) * CPM_COLVEC + mi]);
            }
        }
    }
}

inline void case32_cpm_flush(__private int *acc, __private const cpm_vec *cpm) {
    #pragma unroll
    for (int j = 0; j < NR; ++j) {
        #pragma unroll
        for (int mi = 0; mi < CPM_COLVEC; ++mi) {
            const int4 t = convert_int4(cpm[j * CPM_COLVEC + mi]);
            const int base = j * MR + mi * VWM;
            acc[base + 0] += t.s0;
            acc[base + 1] += t.s1;
            acc[base + 2] += t.s2;
            acc[base + 3] += t.s3;
        }
    }
}
#endif

inline void case32_accum_kgroup(__private int *acc, __private cpm_vec *cpm,
                                __private const int *a_pack, __private const int *b_pack) {
#if CASE32_PACKED_DOT
    (void)cpm;
    #pragma unroll
    for (int j = 0; j < NR; ++j) {
        const int base = j * MR;
        const int b0 = b_pack[j];
        #pragma unroll
        for (int i = 0; i < MR; ++i) {
            acc[base + i] = case32_dot4(acc[base + i], a_pack[i], b0);
        }
    }
#else
    (void)acc;
    case32_cpm_kgroup(cpm, a_pack, b_pack);
#endif
}

__kernel void case33_macro_gemm_xor(__global const char *a_pre, __global const char *b_pre,
                                    __global uint *tile_xor, int N, int blocks_k,
                                    int blocks_per_milestone, int num_milestones, int tile_count,
                                    int macro_rows, int macro_cols, int xor_after_milestone,
                                    int mb_begin, int compact_xor, __global const uint *a_key8,
                                    __global const uint *bound, __global int *found_flag,
                                    __global int *out_t_rows, __global int *out_t_cols,
                                    int fuse_jackpot, int micro_m_begin, int micro_m_count) {
    const int lid = (int)get_local_id(0);
#if !CASE32_USE_LDS
    if (fuse_jackpot && found_flag != 0 && *found_flag != 0) {
        return;
    }
#endif

    const int mb = mb_begin + (int)get_group_id(0);
    const int jm = mb / macro_rows;
    const int im = mb % macro_rows;

    const int tr0 = im * MICRO_M;
    const int tc0 = jm * MICRO_N;
    const int hash_tc0 = jm * HASH_MICRO_N;

#if CASE32_WI_ROWMAJOR
    const int tr_in_slice = lid / HASH_MICRO_N;
    const int hash_tc = lid % HASH_MICRO_N;
#else
    const int tr_in_slice = lid % micro_m_count;
    const int hash_tc = lid / micro_m_count;
#endif
    if (tr_in_slice >= micro_m_count) {
#if CASE32_USE_LDS
        /* Padded WIs must still hit barriers; skip GEMM/jackpot via `active`. */
#else
        return;
#endif
    }
    const int tr = micro_m_begin + tr_in_slice;

    const int tr_global = tr0 + tr;
    const int hash_tc_global = hash_tc0 + hash_tc;
    const int hash_tile_cols = N / HASH_NR;
    const int hash_spatial_id = tr_global * hash_tile_cols + hash_tc_global;

    uint msg[PP_JACKPOT_WORDS];
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        msg[i] = 0u;
    }
#if CASE32_USE_LDS && defined(CASE32_COALESCE)
    __local char lds_a[MICRO_M * LDS_KGROUPS * KG_BYTES_A];
    __local char lds_b[MICRO_N * LDS_KGROUPS * KG_SLICE_B];
    const int active = (tr_in_slice < micro_m_count);
#endif
    for (int reg_half = 0; reg_half < HASH_REG_TILES_N; ++reg_half) {
        const int tc = hash_tc * HASH_REG_TILES_N + reg_half;
        const int tc_global = tc0 + tc;
        int acc[NR * MR];
        for (int i = 0; i < NR * MR; ++i) {
            acc[i] = 0;
        }
#if !CASE32_PACKED_DOT
        cpm_vec cpm[CPM_NVEC];
#endif
        int ms = 0;
        for (int kb = 0; kb < blocks_k; ++kb) {
#if !CASE32_PACKED_DOT
        case32_cpm_zero(cpm);
#endif
#if defined(CASE32_COALESCE)
        const size_t a_kb_base =
                (size_t)im * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_A +
                (size_t)kb * (size_t)MACRO_KB_BLOCK_A;
        const size_t b_kb_base =
                (size_t)jm * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_B +
                (size_t)kb * (size_t)MACRO_KB_BLOCK_B;

#if CASE32_USE_LDS
        /* CLBlast-style: stage KWG of A (all macro rows) and B (all macro cols) in
           __local, then every WI reuses the same panels for its MR×NR C tile. */
        for (int kg0 = 0; kg0 < KGROUPS; kg0 += LDS_KGROUPS) {
            const int n_kg = ((kg0 + LDS_KGROUPS) <= KGROUPS) ? LDS_KGROUPS
                                                             : (KGROUPS - kg0);
            if (active) {
                if (hash_tc == 0) {
                    for (int kg_l = 0; kg_l < n_kg; ++kg_l) {
                        __global const char *src =
                                a_pre + a_kb_base +
                                (size_t)(kg0 + kg_l) * (size_t)MACRO_KG_STRIP_A +
                                (size_t)tr * (size_t)KG_BYTES_A;
                        __local char *dst = lds_a + (size_t)kg_l * (size_t)LDS_A_STRIDE +
                                            (size_t)tr * (size_t)KG_BYTES_A;
                        for (int i = 0; i < KG_BYTES_A; i += RANK) {
                            *((__local int *)(dst + i)) = as_int(vload4(0, src + i));
                        }
                    }
                }
                if (tr_in_slice == 0) {
                    for (int kg_l = 0; kg_l < n_kg; ++kg_l) {
                        __global const char *src =
                                b_pre + b_kb_base +
                                (size_t)(kg0 + kg_l) * (size_t)MACRO_KG_STRIP_B +
                                (size_t)tc * (size_t)KG_SLICE_B;
                        __local char *dst = lds_b + (size_t)kg_l * (size_t)LDS_B_STRIDE +
                                            (size_t)tc * (size_t)KG_SLICE_B;
                        for (int i = 0; i < KG_SLICE_B; i += RANK) {
                            *((__local int *)(dst + i)) = as_int(vload4(0, src + i));
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (active) {
                for (int kg_l = 0; kg_l < n_kg; ++kg_l) {
                    __local const char *a_kg = lds_a + (size_t)kg_l * (size_t)LDS_A_STRIDE +
                                               (size_t)tr * (size_t)KG_BYTES_A;
                    __local const char *b_kg = lds_b + (size_t)kg_l * (size_t)LDS_B_STRIDE +
                                               (size_t)tc * (size_t)KG_SLICE_B;
                    int a_pack[MR];
                    int b_pack[NR];
                    #pragma unroll
                    for (int i = 0; i < MR; ++i) {
                        a_pack[i] = as_int(vload4(0, a_kg + (size_t)i * RANK));
                    }
                    #pragma unroll
                    for (int j = 0; j < NR; ++j) {
                        b_pack[j] = as_int(vload4(0, b_kg + (size_t)j * RANK));
                    }
#if CASE32_PACKED_DOT
                    case32_accum_kgroup(acc, (__private cpm_vec *)0, a_pack, b_pack);
#else
                    case32_accum_kgroup(acc, cpm, a_pack, b_pack);
#endif
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
#else
        for (int kg = 0; kg < KGROUPS; ++kg) {
            __global const char *a_kg =
                    a_pre + a_kb_base + (size_t)kg * (size_t)MACRO_KG_STRIP_A +
                    (size_t)tr * (size_t)KG_BYTES_A;
            __global const char *b_kg =
                    b_pre + b_kb_base + (size_t)kg * (size_t)MACRO_KG_STRIP_B +
                    (size_t)tc * (size_t)KG_SLICE_B;

            int a_pack[MR];
            int b_pack[NR];
            #pragma unroll
            for (int i = 0; i < MR; ++i) {
                a_pack[i] = as_int(vload4(0, a_kg + (size_t)i * RANK));
            }

            #pragma unroll
            for (int j = 0; j < NR; ++j) {
                b_pack[j] = as_int(vload4(0, b_kg + (size_t)j * RANK));
            }
#if CASE32_PACKED_DOT
            case32_accum_kgroup(acc, (__private cpm_vec *)0, a_pack, b_pack);
#else
            case32_accum_kgroup(acc, cpm, a_pack, b_pack);
#endif
        }
#endif
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
            int b_pack[NR];
            #pragma unroll
            for (int i = 0; i < MR; ++i) {
                a_pack[i] = as_int(vload4(0, a_kg + (size_t)i * RANK));
            }

            __global const char *b_kg =
                    b_tile + (size_t)kg * (size_t)KG_SLICE_B;
            #pragma unroll
            for (int j = 0; j < NR; ++j) {
                b_pack[j] = as_int(vload4(0, b_kg + (size_t)j * RANK));
            }
#if CASE32_PACKED_DOT
            case32_accum_kgroup(acc, (__private cpm_vec *)0, a_pack, b_pack);
#else
            case32_accum_kgroup(acc, cpm, a_pack, b_pack);
#endif
        }
#endif
#if !CASE32_PACKED_DOT
        case32_cpm_flush(acc, cpm);
#endif
        /* One milestone per KR panel (KR == R_RANK). Cumulative acc across kb. */
        uint x = 0u;
        if (tr_in_slice < micro_m_count) {
            for (int i = 0; i < NR * MR; ++i) {
                x ^= as_uint(acc[i]);
            }
        }
        if (tr_in_slice < micro_m_count) {
            if (xor_after_milestone) {
                if (fuse_jackpot) {
                    if (ms < PP_MAX_MILESTONES) {
                        const int tid = ms % PP_JACKPOT_WORDS;
                        const uint contribution =
                                (ms + PP_JACKPOT_WORDS < num_milestones)
                                        ? pp_rotl32(x, PP_LROT)
                                        : x;
                        msg[tid] ^= contribution;
                    }
                } else {
                    ulong out_idx;
                    if (compact_xor) {
                        const int hash_local_id =
                                tr_in_slice * HASH_MICRO_N + hash_tc;
                        const ulong batch_stride =
                                (ulong)(micro_m_count * HASH_MICRO_N) *
                                (ulong)num_milestones;
                        const ulong out_base = (ulong)get_group_id(0) * batch_stride +
                                               (ulong)hash_local_id *
                                                       (ulong)num_milestones;
                        out_idx = out_base + (ulong)ms;
                    } else {
                        out_idx = (ulong)ms * (ulong)tile_count +
                                  (ulong)hash_spatial_id;
                    }
#if HASH_REG_TILES_N > 1
                    if (reg_half == 0) {
                        tile_xor[out_idx] = x;
                    } else {
                        tile_xor[out_idx] ^= x;
                    }
#else
                    tile_xor[out_idx] = x;
#endif
                }
            }
        }
        ++ms;
        }
    }
    (void)blocks_per_milestone;

#if CASE32_USE_LDS
    if (tr_in_slice >= micro_m_count) {
        return;
    }
#endif
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
    const int t_cols = jm * MACRO_N + hash_tc * HASH_NR;
    if (out_t_rows != 0) {
        *out_t_rows = t_rows;
    }
    if (out_t_cols != 0) {
        *out_t_cols = t_cols;
    }
}
