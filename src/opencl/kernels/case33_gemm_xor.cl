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
/* Register double-buffered k-group loop (coalesced, non-LDS path only; needs an even
   KGROUPS and the flat kb/kg stride, i.e. MACRO_KB_BLOCK_A == KGROUPS*MACRO_KG_STRIP_A).
   -DCASE32_PIPELINE=0 restores the plain load-then-dot loop. */
#ifndef CASE32_PIPELINE
#define CASE32_PIPELINE 1
#endif
#if CASE32_PIPELINE && (!defined(CASE32_COALESCE) || CASE32_USE_LDS || (KGROUPS % 2) != 0)
#undef CASE32_PIPELINE
#define CASE32_PIPELINE 0
#endif
/* Work-group -> macro-block super-tile shape (see kernel body). 1x1 = linear map. */
#ifndef SWZ_IM
#define SWZ_IM 8
#endif
#ifndef SWZ_JM
#define SWZ_JM 4
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
/* How many coalesce kg-strips to stage in LDS at once (Case 3.4; keep SLM modest). */
#ifndef KG_LDS_CHUNK
#define KG_LDS_CHUNK 8
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
        defined(CASE32_USE_BUILTIN_SUDOT4) || \
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
#elif defined(CASE32_USE_BUILTIN_SUDOT4)
    /* RDNA3 (gfx11) spells the int8 dot product V_DOT4_I32_IU8, reached via
     * sudot4 with both operands marked signed. The older sdot4 and v_dot4c
     * forms need dot1-insts, which gfx11 does not have, so on a Radeon 780M
     * the accelerated path fell through to the scalar 4x MAC nest. */
    return __builtin_amdgcn_sudot4(true, a_pack, true, b_pack, acc, false);
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

/* One WI's packed-int8 operands for one k-group (RANK=4 bytes per row/col):
   A: MR*RANK contiguous bytes at tr*KG_BYTES_A, B: NR*RANK bytes at tc*KG_SLICE_B.
   In the coalesced layout both offsets are multiples of 16 (KG_BYTES_A = MR*4,
   KG_SLICE_B = NR*4, MR/NR >= 4) on top of 16 KiB kb blocks and 512 B kg strips, so
   load through 16 B vectors instead of 4 B char4s: the loads are unambiguously
   dwordx4 even for compilers that do not infer alignment through vload4(char*). */
#if (MR % 4) == 0 && (NR % 4) == 0 && RANK == 4
#define CASE32_VEC_LOADS 1
#else
#define CASE32_VEC_LOADS 0
#endif

inline void case32_load_a_pack(__private int *a_pack, __global const char *a_kg) {
#if CASE32_VEC_LOADS
    __global const uint4 *a_v = (__global const uint4 *)a_kg;
    #pragma unroll
    for (int i = 0; i < MR / 4; ++i) {
        const uint4 v = a_v[i];
        a_pack[4 * i + 0] = as_int(v.s0);
        a_pack[4 * i + 1] = as_int(v.s1);
        a_pack[4 * i + 2] = as_int(v.s2);
        a_pack[4 * i + 3] = as_int(v.s3);
    }
#else
    #pragma unroll
    for (int i = 0; i < MR; ++i) {
        a_pack[i] = as_int(vload4(0, a_kg + (size_t)i * RANK));
    }
#endif
}

inline void case32_load_b_pack(__private int *b_pack, __global const char *b_kg) {
#if CASE32_VEC_LOADS
    __global const uint4 *b_v = (__global const uint4 *)b_kg;
    #pragma unroll
    for (int j = 0; j < NR / 4; ++j) {
        const uint4 v = b_v[j];
        b_pack[4 * j + 0] = as_int(v.s0);
        b_pack[4 * j + 1] = as_int(v.s1);
        b_pack[4 * j + 2] = as_int(v.s2);
        b_pack[4 * j + 3] = as_int(v.s3);
    }
#else
    #pragma unroll
    for (int j = 0; j < NR; ++j) {
        b_pack[j] = as_int(vload4(0, b_kg + (size_t)j * RANK));
    }
#endif
}

#if CASE32_USE_LDS
/* Case 3.4: all WIs cooperatively copy A/B kg-strips into SLM (coalesced 16B). */
inline void case32_copy_bytes(__global const uchar *src, __local uchar *dst, int nbytes,
                              int lid, int lsz) {
    for (int off = lid * 16; off + 15 < nbytes; off += lsz * 16) {
        vstore4(vload4(0, src + off), 0, dst + off);
        vstore4(vload4(0, src + off + 4), 0, dst + off + 4);
        vstore4(vload4(0, src + off + 8), 0, dst + off + 8);
        vstore4(vload4(0, src + off + 12), 0, dst + off + 12);
    }
    for (int off = (nbytes & ~15) + lid; off < nbytes; off += lsz) {
        dst[off] = src[off];
    }
}
#endif

#if !defined(CASE32_WMMA)
#ifdef CASE32_REQD_WG
/* Host passes the exact launch local size when one macro block fits a work-group. */
__attribute__((reqd_work_group_size(CASE32_REQD_WG, 1, 1)))
#endif
__kernel void case33_macro_gemm_xor(__global const char *a_pre, __global const char *b_pre,
                                    __global uint *tile_xor, int N, int blocks_k,
                                    int blocks_per_milestone, int num_milestones, int tile_count,
                                    int macro_rows, int macro_cols, int xor_after_milestone,
                                    int mb_begin, int compact_xor, __global const uint *a_key8,
                                    __global const uint *bound, __global int *found_flag,
                                    __global int *out_t_rows, __global int *out_t_cols,
                                    int fuse_jackpot, int micro_m_begin, int micro_m_count) {
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
#if !CASE32_USE_LDS
    if (fuse_jackpot && found_flag != 0 && *found_flag != 0) {
        return;
    }
#endif

    const int mb = mb_begin + (int)get_group_id(0);
    /* Work-group -> macro block. The linear map (im fastest) makes a 1024-launch
       walk every im for one jm: each resident work-group streams its own A strip
       from DRAM and A is never reused from L2 (the whole of A is re-read once per
       jm). Swizzle into SWZ_IM x SWZ_JM super-tiles (im fastest inside, super-tiles
       ordered im-fastest too) so the ~30 co-resident work-groups share SWZ_IM A
       strips and SWZ_JM B strips in L2. Any bijection is correct: t_rows/t_cols and
       all addressing derive from (im, jm). Falls back to the linear map when the
       macro grid is not divisible (uniform branch). */
    int jm;
    int im;
    if ((macro_rows % SWZ_IM) == 0 && (macro_cols % SWZ_JM) == 0) {
        const int super_rows = macro_rows / SWZ_IM;
        const int super_id = mb / (SWZ_IM * SWZ_JM);
        const int within = mb - super_id * (SWZ_IM * SWZ_JM);
        const int super_col = super_id / super_rows;
        const int super_row = super_id - super_col * super_rows;
        im = super_row * SWZ_IM + (within % SWZ_IM);
        jm = super_col * SWZ_JM + (within / SWZ_IM);
    } else {
        jm = mb / macro_rows;
        im = mb - jm * macro_rows;
    }

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
    /* Case 3.4: stage full macro kg-strips; every WI participates in the copy. */
    __local uchar lds_a[KG_LDS_CHUNK * MACRO_KG_STRIP_A];
    __local uchar lds_b[KG_LDS_CHUNK * MACRO_KG_STRIP_B];
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
#if CASE32_PIPELINE
        /* Software pipeline (register double buffer). In the coalesced layout a WI's
           operands for k-group (kb, kg) sit at a_run + (kb*KGROUPS + kg)*MACRO_KG_STRIP_A
           (same for B), i.e. the whole K walk is one flat stride, so the prefetch runs
           across kb boundaries and never drains. Buffer 0 holds the k-group being
           consumed; the loop is unrolled x2 so buffers alternate without copies. */
        __global const char *a_run =
                a_pre + (size_t)im * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_A +
                (size_t)tr * (size_t)KG_BYTES_A;
        __global const char *b_run =
                b_pre + (size_t)jm * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_B +
                (size_t)tc * (size_t)KG_SLICE_B;
        const int kg_last = blocks_k * KGROUPS - 1;
        int kg_flat = 0;
        int a_p0[MR];
        int b_p0[NR];
        int a_p1[MR];
        int b_p1[NR];
        case32_load_a_pack(a_p0, a_run);
        case32_load_b_pack(b_p0, b_run);
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
        /* Case 3.4 cooperative LDS: all WIs copy A/B strips, then reuse from SLM. */
        for (int kg0 = 0; kg0 < KGROUPS; kg0 += KG_LDS_CHUNK) {
            const int nkg = min(KG_LDS_CHUNK, KGROUPS - kg0);
            const int a_bytes = nkg * MACRO_KG_STRIP_A;
            const int b_bytes = nkg * MACRO_KG_STRIP_B;

            case32_copy_bytes((__global const uchar *)(a_pre + a_kb_base +
                                                       (size_t)kg0 * (size_t)MACRO_KG_STRIP_A),
                              lds_a, a_bytes, lid, lsz);
            case32_copy_bytes((__global const uchar *)(b_pre + b_kb_base +
                                                       (size_t)kg0 * (size_t)MACRO_KG_STRIP_B),
                              lds_b, b_bytes, lid, lsz);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (active) {
                for (int kgi = 0; kgi < nkg; ++kgi) {
                    __local const uchar *a_kg =
                            lds_a + (size_t)kgi * (size_t)MACRO_KG_STRIP_A +
                            (size_t)tr * (size_t)KG_BYTES_A;
                    __local const uchar *b_kg =
                            lds_b + (size_t)kgi * (size_t)MACRO_KG_STRIP_B +
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
#elif CASE32_PIPELINE
        (void)a_kb_base;
        (void)b_kb_base;
        for (int kg = 0; kg < KGROUPS; kg += 2) {
            /* kg+1 always exists (KGROUPS even); kg+2 may be the next kb's first
               k-group, clamp at the very end (re-reads the last k-group, harmless). */
            const int kg_n1 = kg_flat + 1;
            const int kg_n2 = (kg_flat + 2 <= kg_last) ? (kg_flat + 2) : kg_last;
            case32_load_a_pack(a_p1, a_run + (size_t)kg_n1 * (size_t)MACRO_KG_STRIP_A);
            case32_load_b_pack(b_p1, b_run + (size_t)kg_n1 * (size_t)MACRO_KG_STRIP_B);
#if CASE32_PACKED_DOT
            case32_accum_kgroup(acc, (__private cpm_vec *)0, a_p0, b_p0);
#else
            case32_accum_kgroup(acc, cpm, a_p0, b_p0);
#endif
            case32_load_a_pack(a_p0, a_run + (size_t)kg_n2 * (size_t)MACRO_KG_STRIP_A);
            case32_load_b_pack(b_p0, b_run + (size_t)kg_n2 * (size_t)MACRO_KG_STRIP_B);
#if CASE32_PACKED_DOT
            case32_accum_kgroup(acc, (__private cpm_vec *)0, a_p1, b_p1);
#else
            case32_accum_kgroup(acc, cpm, a_p1, b_p1);
#endif
            kg_flat += 2;
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
            case32_load_a_pack(a_pack, a_kg);
            case32_load_b_pack(b_pack, b_kg);
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
            __global const char *b_kg =
                    b_tile + (size_t)kg * (size_t)KG_SLICE_B;
            int a_pack[MR];
            int b_pack[NR];
            case32_load_a_pack(a_pack, a_kg);
            case32_load_b_pack(b_pack, b_kg);
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
#else /* CASE32_WMMA */

/* =====================================================================================
 * AMD WMMA (matrix core) path: --ocl-dot wmma, -DCASE32_WMMA=11 (gfx11, RDNA3) or
 * -DCASE32_WMMA=12 (gfx12, RDNA4). Needs wave32, the 8x16 hash tile and RANK=4 prepack.
 *
 * Work split. One wave32 owns a 64x64 C sub-tile of the macro block: 4x4 WMMA blocks of
 * 16x16 = 8x4 hash tiles of 8x16 = 32 hash tiles, i.e. one hash tile per lane for the
 * msg[16] fold and BLAKE3. A 128x128 macro block is 4 waves = 128 WIs, the same local
 * size (MACRO_M/8 * MACRO_N/16) the dot4 kernel launches with; 64x64 is one wave.
 *
 *      macro block (128x128)            one wave's 64x64 (16x16 WMMA blocks bi x bj)
 *   +-------------+-------------+         bj=0    bj=1    bj=2    bj=3
 *   | wave 0      | wave 2      |  bi=0 | t0   | t8   | t16  | t24  |  rows 0..7  (top)
 *   | wm=0 wn=0   | wm=0 wn=1   |       | t1   | t9   | t17  | t25  |  rows 8..15 (bottom)
 *   +-------------+-------------+  bi=1 | t2   | t10  | ...
 *   | wave 1      | wave 3      |       | t3   | t11  | ...
 *   | wm=1 wn=0   | wm=1 wn=1   |  ...       t = bj*8 + bi*2 + half
 *   +-------------+-------------+  after the milestone reduce-scatter lane t holds tile t
 *
 * Operands come from the default coalesced prepack, unchanged: inside one macro block,
 * flat k-group g (4 k) of row r (A) / column c (B) is the dword at byte g*STRIP + r*4,
 * STRIP = MACRO_M*4 (MACRO_KG_STRIP_A/B). A 16-k WMMA step is 4 consecutive k-groups, so
 * every lane gathers its operand with dword loads at a STRIP stride; the 16 lanes of a
 * half-wave read 64 consecutive bytes per k-group (fully coalesced).
 *
 * gfx11 wave32 V_WMMA_I32_16X16X16_IU8 (A, B: 4 VGPRs; C, D: 8 VGPRs):
 *   A: lane l holds row m = l%16, VGPR q = k 4q..4q+3 (byte j of the dword = k 4q+j);
 *      lanes 16..31 must carry a copy of lanes 0..15 (they load the same addresses).
 *   B: lane l holds column n = l%16, same k packing.
 *   D: VGPR e of lane l = D[2e + l/16][l%16]:
 *                 lanes 0..15     lanes 16..31
 *        VGPR 0   row  0          row  1        \
 *        VGPR 1   row  2          row  3         | top hash tile    (rows 0..7)
 *        VGPR 2   row  4          row  5         |
 *        VGPR 3   row  6          row  7        /
 *        VGPR 4   row  8          row  9        \
 *        ...                                     | bottom hash tile (rows 8..15)
 *        VGPR 7   row 14          row 15        /
 *
 * gfx12 wave32 V_WMMA_I32_16X16X16_IU8 (A, B: 2 VGPRs; C, D: 8 VGPRs), h = l/16:
 *   A: lane l holds row m = l%16, no duplication across the halves.
 *      CASE32_WMMA_G12_KSPLIT=1 (default; AMD's RDNA4 WMMA guide: one layout for every
 *      data type, 8 consecutive k per lane):
 *          VGPR0 = k 8h..8h+3, VGPR1 = k 8h+4..8h+7
 *      CASE32_WMMA_G12_KSPLIT=0 (alternative):
 *          VGPR0 = k 4h..4h+3, VGPR1 = k 8+4h..8+4h+3
 *          (lanes 0..15: k 0-3, 8-11; lanes 16..31: k 4-7, 12-15)
 *   B: same with column n = l%16.
 *   D: VGPR e of lane l = D[e + 8*(l/16)][l%16]:
 *                 lanes 0..15     lanes 16..31
 *        VGPR e   row e           row 8+e
 *      -> lanes 0..15 hold the top hash tile, lanes 16..31 the bottom one.
 *   Not yet run on gfx12 hardware: the host runs case33_wmma_selftest for both k mappings
 *   at start-up and uses the one that passes (sudot4 if none does).
 *
 * Milestones. Every KR = 128 k the 32 hash-tile words of a wave are formed in two steps:
 * each lane XORs the D elements it owns per hash tile (2 partials per WMMA block, 32 per
 * lane), then a 5-stage XOR reduce-scatter over ds_swizzle (xor 16, 8, 4, 2, 1; 31
 * swizzles) leaves the complete word of hash tile t in lane t. XOR and int32 wrap-around
 * accumulation are both order-independent, so the words equal the dot4 kernel's exactly.
 * ===================================================================================== */
#if CASE32_WMMA != 11 && CASE32_WMMA != 12
#error CASE32_WMMA must be 11 (gfx11) or 12 (gfx12)
#endif
#if MR != 8 || NR != 16 || HASH_NR != 16 || RANK != 4 || KR != 128
#error WMMA path needs the 8x16 hash tile, RANK=4 prepack and KR=128
#endif
#if (MACRO_M % 64) != 0 || (MACRO_N % 64) != 0 || MACRO_M != MACRO_N
#error WMMA path needs a 64x64 or 128x128 macro block
#endif
#if CASE32_USE_LDS
#error WMMA path reads the coalesced prepack directly (no LDS staging)
#endif
#ifndef CASE32_WMMA_G12_KSPLIT
#define CASE32_WMMA_G12_KSPLIT 1
#endif
/* 1: register double-buffer the 16-k step operands (host default; CP_OCL_WMMA_PIPELINE=0
   builds the plain load-then-WMMA loop). */
#ifndef CASE32_WMMA_PIPELINE
#define CASE32_WMMA_PIPELINE 1
#endif
#if CASE32_WMMA_PIPELINE && (KGROUPS % 8) != 0
#error CASE32_WMMA_PIPELINE needs an even number of 16-k steps per KR panel
#endif
/* Unroll of the 16-k step loop inside a KR panel (0 = compiler default, i.e. full). With
   the full unroll the scheduler hoists the loads of several steps to hide latency and the
   operands of all of them stay live: 256 VGPRs and scratch spills on gfx12. */
#ifndef CASE32_WMMA_KUNROLL
#define CASE32_WMMA_KUNROLL 0
#endif
#define WMMA_PRAGMA_(x) _Pragma(#x)
#define WMMA_UNROLL_(n) WMMA_PRAGMA_(unroll n)
#if CASE32_WMMA_KUNROLL > 0
#define WMMA_KLOOP_UNROLL WMMA_UNROLL_(CASE32_WMMA_KUNROLL)
#else
#define WMMA_KLOOP_UNROLL
#endif

#define WMMA_WAVE_ROWS 64
#define WMMA_WAVE_COLS 64
#define WMMA_BM (WMMA_WAVE_ROWS / 16)
#define WMMA_BN (WMMA_WAVE_COLS / 16)
#define WMMA_WAVES_M (MACRO_M / WMMA_WAVE_ROWS)
#define WMMA_KG_DW_A (MACRO_KG_STRIP_A / 4) /* dwords between consecutive k-groups */
#define WMMA_KG_DW_B (MACRO_KG_STRIP_B / 4)
#define WMMA_KSTEPS (KGROUPS / 4)          /* 16-k WMMA steps per KR panel */
#define WMMA_WG ((MACRO_M / WMMA_WAVE_ROWS) * (MACRO_N / WMMA_WAVE_COLS) * 32) /* WIs per group */
/* 1: the 16 jackpot words per lane sit in LDS (8 KB per 128-WI group) instead of VGPRs. */
#ifndef CASE32_WMMA_MSG_LDS
#define CASE32_WMMA_MSG_LDS 1
#endif

#if CASE32_WMMA == 11
typedef int4 wmma_ab;
#else
typedef int2 wmma_ab;
#endif
typedef int8 wmma_acc;

/* CASE32_WMMA_EMU=1 (host: CP_OCL_WMMA_EMU=11|12): the WMMA instruction and ds_swizzle are
   emulated through LDS with the same lane layouts, so this whole path (prepack indexing,
   k mapping, accumulator layout, milestone reduce-scatter, LDS staging) can be checked with
   --align-test / --verify on any OpenCL device. Slow; for testing only. Every lane of the
   work-group must make the same calls (they contain barriers). */
#ifndef CASE32_WMMA_EMU
#define CASE32_WMMA_EMU 0
#endif
#if CASE32_WMMA_EMU
#define WMMA_EMU_INTS 160 /* per wave: A 16x16 int8 (64), B (64), swizzle (32) */
#define WMMA_EMU_ARG , __local int *emu
#define WMMA_EMU_FWD , emu
#define WMMA_EMU_DECL(wis) __local int wmma_emu_lds_[((wis) / 32) * WMMA_EMU_INTS]
#define WMMA_EMU_PASS , (wmma_emu_lds_ + ((int)get_local_id(0) >> 5) * WMMA_EMU_INTS)
inline int wmma_emu_dot4(int a, int b) {
    return (int)(char)a * (int)(char)b + (int)(char)(a >> 8) * (int)(char)(b >> 8) +
           (int)(char)(a >> 16) * (int)(char)(b >> 16) + (int)(char)(a >> 24) * (int)(char)(b >> 24);
}
#else
#define WMMA_EMU_ARG
#define WMMA_EMU_FWD
#define WMMA_EMU_DECL(wis)
#define WMMA_EMU_PASS
#endif

/* k-group offset of the first operand dword of `lane` (0 on gfx11: full k per lane). */
inline int wmma_lane_kg0(uint lane, int ksplit) {
#if CASE32_WMMA == 11
    (void)lane;
    (void)ksplit;
    return 0;
#else
    const int h = (int)(lane >> 4);
    return ksplit ? 2 * h : h;
#endif
}

/* One lane's A or B operand for a 16-k step. p: the lane's dword for its row/column at
   k-group wmma_lane_kg0(); kg: dword stride between consecutive k-groups. */
inline wmma_ab wmma_load_ab(__global const int *p, int kg, int ksplit) {
#if CASE32_WMMA == 11
    (void)ksplit;
    return (int4)(p[0], p[kg], p[2 * kg], p[3 * kg]);
#else
    return (int2)(p[0], p[(ksplit ? 1 : 2) * kg]);
#endif
}

/* D/C row of accumulator VGPR e in `lane` (column is lane % 16 on both). */
inline int wmma_c_row(int e, uint lane) {
#if CASE32_WMMA == 11
    return 2 * e + (int)(lane >> 4);
#else
    return e + 8 * (int)(lane >> 4);
#endif
}

#if CASE32_WMMA_EMU
/* emu: A/B as [row or column][k-group] dwords in LDS (4 k-groups = 16 k per row). */
inline wmma_acc wmma_mac(wmma_ab a, wmma_ab b, wmma_acc c, __local int *emu, int ksplit) {
    const int lane = (int)get_local_id(0) & 31;
    const int m = lane & 15;
    const int h = lane >> 4;
    __local int *ea = emu;
    __local int *eb = emu + 64;
#if CASE32_WMMA == 11
    (void)ksplit;
    if (h == 0) {
        ea[m * 4 + 0] = a.s0; ea[m * 4 + 1] = a.s1; ea[m * 4 + 2] = a.s2; ea[m * 4 + 3] = a.s3;
        eb[m * 4 + 0] = b.s0; eb[m * 4 + 1] = b.s1; eb[m * 4 + 2] = b.s2; eb[m * 4 + 3] = b.s3;
    }
#else
    const int g0 = ksplit ? 2 * h : h;
    const int g1 = ksplit ? 2 * h + 1 : 2 + h;
    ea[m * 4 + g0] = a.s0;
    ea[m * 4 + g1] = a.s1;
    eb[m * 4 + g0] = b.s0;
    eb[m * 4 + g1] = b.s1;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    int d[8] = {c.s0, c.s1, c.s2, c.s3, c.s4, c.s5, c.s6, c.s7};
    for (int e = 0; e < 8; ++e) {
#if CASE32_WMMA == 11
        const int row = 2 * e + h;
#else
        const int row = e + 8 * h;
#endif
        for (int q = 0; q < 4; ++q) {
            d[e] += wmma_emu_dot4(ea[row * 4 + q], eb[m * 4 + q]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return (wmma_acc)(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}
inline uint wmma_swz_xor(uint v, uint m, __local int *emu) {
    const int lane = (int)get_local_id(0) & 31;
    emu[128 + lane] = as_int(v);
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint r = as_uint(emu[128 + (lane ^ (int)m)]);
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
#define WMMA_MAC(a, b, c) wmma_mac((a), (b), (c), emu, CASE32_WMMA_G12_KSPLIT)
#else
inline wmma_acc wmma_mac(wmma_ab a, wmma_ab b, wmma_acc c) {
#if CASE32_WMMA == 11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#endif
}
#define WMMA_MAC(a, b, c) wmma_mac((a), (b), (c))
#endif

/* This lane's XOR partials of the two 8x16 hash tiles of one 16x16 block. */
inline void wmma_tile_partials(wmma_acc c, uint lane, uint *top, uint *bot) {
#if CASE32_WMMA == 11
    (void)lane;
    *top = as_uint(c.s0 ^ c.s1 ^ c.s2 ^ c.s3); /* rows 0,2,4,6 / 1,3,5,7 */
    *bot = as_uint(c.s4 ^ c.s5 ^ c.s6 ^ c.s7); /* rows 8..14 / 9..15 */
#else
    const uint x = as_uint(c.s0 ^ c.s1 ^ c.s2 ^ c.s3 ^ c.s4 ^ c.s5 ^ c.s6 ^ c.s7);
    const int lower = lane < 16u;
    *top = lower ? x : 0u;
    *bot = lower ? 0u : x;
#endif
}

/* ds_swizzle bit-mask mode (offset[15] = 0): and_mask = 0x1f, or_mask = 0, xor_mask = m.
   Works within groups of 32 lanes, i.e. the whole wave32. */
#if CASE32_WMMA_EMU
#define WMMA_SWZ_XOR(v, m) wmma_swz_xor((v), (m), emu)
#else
#define WMMA_SWZ_XOR(v, m) as_uint(__builtin_amdgcn_ds_swizzle(as_int(v), ((m) << 10) | 0x1f))
#endif
/* Reduce-scatter stage over lane bit m: keep the half of p[0..n) whose index bit
   matches the lane bit, send the other half to lane ^ m, XOR in what comes back. */
/* The halves are picked with a lane mask, not `hi ? p[a] : p[b]`: LLVM turns that select of
   two elements into one load at a select-ed index, which forces p[] out of registers into
   scratch (gfx11/gfx12: ~70 scratch ops per milestone). */
#define WMMA_RS_STAGE(p, lane, n, m)                                                  \
    do {                                                                              \
        const uint hm_ = (((lane) & (m)) != 0u) ? 0xFFFFFFFFu : 0u;                   \
        _Pragma("unroll") for (int j_ = 0; j_ < (n) / 2; ++j_) {                      \
            const uint lo_ = (p)[j_];                                                 \
            const uint up_ = (p)[j_ + (n) / 2];                                       \
            const uint keep_ = (up_ & hm_) | (lo_ & ~hm_);                            \
            const uint send_ = (lo_ & hm_) | (up_ & ~hm_);                            \
            (p)[j_] = keep_ ^ WMMA_SWZ_XOR(send_, m);                                 \
        }                                                                             \
    } while (0)

/* p[t] (t = 0..31): this lane's partial of word t. Returns XOR over all 32 lanes of
   p[lane], i.e. lane t receives the complete word t. Whole wave must be active. */
inline uint wmma_reduce_scatter32(uint *p, uint lane WMMA_EMU_ARG) {
    WMMA_RS_STAGE(p, lane, 32, 16);
    WMMA_RS_STAGE(p, lane, 16, 8);
    WMMA_RS_STAGE(p, lane, 8, 4);
    WMMA_RS_STAGE(p, lane, 4, 2);
    WMMA_RS_STAGE(p, lane, 2, 1);
    return p[0];
}

/* Operands of one 16-k step (k-groups g..g+3) for the wave's 4 row and 4 column blocks. */
inline void wmma_load_step(__private wmma_ab *a, __private wmma_ab *b, __global const int *a_lane,
                           __global const int *b_lane, int g) {
    #pragma unroll
    for (int bi = 0; bi < WMMA_BM; ++bi) {
        a[bi] = wmma_load_ab(a_lane + g * WMMA_KG_DW_A + bi * 16, WMMA_KG_DW_A,
                             CASE32_WMMA_G12_KSPLIT);
    }
    #pragma unroll
    for (int bj = 0; bj < WMMA_BN; ++bj) {
        b[bj] = wmma_load_ab(b_lane + g * WMMA_KG_DW_B + bj * 16, WMMA_KG_DW_B,
                             CASE32_WMMA_G12_KSPLIT);
    }
}

inline void wmma_mac_step(__private wmma_acc *acc, __private const wmma_ab *a,
                          __private const wmma_ab *b WMMA_EMU_ARG) {
    #pragma unroll
    for (int bi = 0; bi < WMMA_BM; ++bi) {
        #pragma unroll
        for (int bj = 0; bj < WMMA_BN; ++bj) {
            acc[bi * WMMA_BN + bj] = WMMA_MAC(a[bi], b[bj], acc[bi * WMMA_BN + bj]);
        }
    }
}

#ifdef CASE32_REQD_WG
__attribute__((reqd_work_group_size(CASE32_REQD_WG, 1, 1)))
#endif
__kernel void case33_macro_gemm_xor(__global const char *a_pre, __global const char *b_pre,
                                    __global uint *tile_xor, int N, int blocks_k,
                                    int blocks_per_milestone, int num_milestones, int tile_count,
                                    int macro_rows, int macro_cols, int xor_after_milestone,
                                    int mb_begin, int compact_xor, __global const uint *a_key8,
                                    __global const uint *bound, __global int *found_flag,
                                    __global int *out_t_rows, __global int *out_t_cols,
                                    int fuse_jackpot, int micro_m_begin, int micro_m_count) {
    /* The host launches exactly one full macro block per work-group here. */
    (void)blocks_per_milestone;
    (void)micro_m_begin;
    (void)micro_m_count;
    WMMA_EMU_DECL(WMMA_WG);
#if !CASE32_WMMA_EMU /* emu: the waves of a group must all reach every barrier */
    if (fuse_jackpot && found_flag != 0 && *found_flag != 0) {
        return;
    }
#endif

    const int mb = mb_begin + (int)get_group_id(0);
    int jm;
    int im;
    if ((macro_rows % SWZ_IM) == 0 && (macro_cols % SWZ_JM) == 0) {
        const int super_rows = macro_rows / SWZ_IM;
        const int super_id = mb / (SWZ_IM * SWZ_JM);
        const int within = mb - super_id * (SWZ_IM * SWZ_JM);
        const int super_col = super_id / super_rows;
        const int super_row = super_id - super_col * super_rows;
        im = super_row * SWZ_IM + (within % SWZ_IM);
        jm = super_col * SWZ_JM + (within / SWZ_IM);
    } else {
        jm = mb / macro_rows;
        im = mb - jm * macro_rows;
    }

    const uint lid = (uint)get_local_id(0);
    const uint lane = lid & 31u;
    const int wave = (int)(lid >> 5);
    const int wm = wave % WMMA_WAVES_M;
    const int wn = wave / WMMA_WAVES_M;

    /* Per-lane operand streams: row (col) wm*64 + lane%16 of the macro block, first
       k-group per the arch layout; block bi (bj) adds 16 rows (cols) = 16 dwords. */
    const int kg0 = wmma_lane_kg0(lane, CASE32_WMMA_G12_KSPLIT);
    __global const int *a_lane =
            (__global const int *)(a_pre + (size_t)im * (size_t)blocks_k *
                                                   (size_t)MACRO_KB_BLOCK_A) +
            (wm * WMMA_WAVE_ROWS + (int)(lane & 15u)) + kg0 * WMMA_KG_DW_A;
    __global const int *b_lane =
            (__global const int *)(b_pre + (size_t)jm * (size_t)blocks_k *
                                                   (size_t)MACRO_KB_BLOCK_B) +
            (wn * WMMA_WAVE_COLS + (int)(lane & 15u)) + kg0 * WMMA_KG_DW_B;

    /* Hash tile owned by this lane after the reduce-scatter: t = bj*8 + bi*2 + half. */
    const int own_bj = (int)(lane >> 3);
    const int own_bi = (int)((lane >> 1) & 3u);
    const int own_half = (int)(lane & 1u);
    const int hash_row = wm * (WMMA_WAVE_ROWS / MR) + own_bi * 2 + own_half; /* in macro */
    const int hash_col = wn * (WMMA_WAVE_COLS / HASH_NR) + own_bj;            /* in macro */
    const int tr_global = im * MICRO_M + hash_row;
    const int hash_tc_global = jm * HASH_MICRO_N + hash_col;
    const int hash_spatial_id = tr_global * (N / HASH_NR) + hash_tc_global;

#if CASE32_WMMA_MSG_LDS
    /* Jackpot words live in LDS, word-major so a wave's lanes hit consecutive banks: the
       128 accumulator VGPRs leave no room for them at the milestone without spills. Each
       lane touches only its own column, so no barrier is needed. */
    __local uint msg_lds[PP_JACKPOT_WORDS * WMMA_WG];
    #pragma unroll
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        msg_lds[i * WMMA_WG + (int)lid] = 0u;
    }
#else
    uint msg[PP_JACKPOT_WORDS];
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        msg[i] = 0u;
    }
#endif
    wmma_acc acc[WMMA_BM * WMMA_BN];
    #pragma unroll
    for (int i = 0; i < WMMA_BM * WMMA_BN; ++i) {
        acc[i] = (wmma_acc)(0);
    }

    int kg_flat = 0;
#if CASE32_WMMA_PIPELINE
    /* Register double buffer: the next 16-k step's operands are in flight while the
       current step's 16 WMMAs issue. The k walk is one flat k-group stride across KR
       panels, so the prefetch crosses milestones; the very last one is clamped (re-reads
       the final step, harmless). */
    const int kg_last_step = blocks_k * KGROUPS - 4;
    wmma_ab a0[WMMA_BM];
    wmma_ab b0[WMMA_BN];
    wmma_ab a1[WMMA_BM];
    wmma_ab b1[WMMA_BN];
    wmma_load_step(a0, b0, a_lane, b_lane, 0);
#endif
    for (int ms = 0; ms < blocks_k; ++ms) {
#if CASE32_WMMA_PIPELINE
        for (int ks = 0; ks < WMMA_KSTEPS; ks += 2) {
            wmma_load_step(a1, b1, a_lane, b_lane, kg_flat + 4);
            wmma_mac_step(acc, a0, b0 WMMA_EMU_PASS);
            const int kg_n2 = (kg_flat + 8 <= kg_last_step) ? (kg_flat + 8) : kg_last_step;
            wmma_load_step(a0, b0, a_lane, b_lane, kg_n2);
            wmma_mac_step(acc, a1, b1 WMMA_EMU_PASS);
            kg_flat += 8;
        }
#else
        WMMA_KLOOP_UNROLL
        for (int ks = 0; ks < WMMA_KSTEPS; ++ks) {
            wmma_ab a[WMMA_BM];
            wmma_ab b[WMMA_BN];
            wmma_load_step(a, b, a_lane, b_lane, kg_flat);
            wmma_mac_step(acc, a, b WMMA_EMU_PASS);
            kg_flat += 4;
        }
#endif

        /* Milestone (one per KR panel, cumulative C): word of hash tile `lane`. */
        uint p[32];
        #pragma unroll
        for (int bi = 0; bi < WMMA_BM; ++bi) {
            #pragma unroll
            for (int bj = 0; bj < WMMA_BN; ++bj) {
                wmma_tile_partials(acc[bi * WMMA_BN + bj], lane, &p[bj * 8 + bi * 2],
                                   &p[bj * 8 + bi * 2 + 1]);
            }
        }
        const uint x = wmma_reduce_scatter32(p, lane WMMA_EMU_PASS);

        if (xor_after_milestone) {
            if (fuse_jackpot) {
                if (ms < PP_MAX_MILESTONES) {
                    const int tid = ms % PP_JACKPOT_WORDS;
                    const uint contribution =
                            (ms + PP_JACKPOT_WORDS < num_milestones) ? pp_rotl32(x, PP_LROT) : x;
#if CASE32_WMMA_MSG_LDS
                    msg_lds[tid * WMMA_WG + (int)lid] ^= contribution;
#else
                    /* static indices keep msg[] in VGPRs (msg[tid] would live in scratch) */
                    #pragma unroll
                    for (int w = 0; w < PP_JACKPOT_WORDS; ++w) {
                        msg[w] ^= (w == tid) ? contribution : 0u;
                    }
#endif
                }
            } else {
                ulong out_idx;
                if (compact_xor) {
                    const int hash_local_id = hash_row * HASH_MICRO_N + hash_col;
                    const ulong batch_stride =
                            (ulong)(MICRO_M * HASH_MICRO_N) * (ulong)num_milestones;
                    out_idx = (ulong)get_group_id(0) * batch_stride +
                              (ulong)hash_local_id * (ulong)num_milestones + (ulong)ms;
                } else {
                    out_idx = (ulong)ms * (ulong)tile_count + (ulong)hash_spatial_id;
                }
                tile_xor[out_idx] = x;
            }
        }
    }

    if (!fuse_jackpot || !xor_after_milestone || a_key8 == 0 || bound == 0) {
        return;
    }

#if CASE32_WMMA_MSG_LDS
    uint msg[PP_JACKPOT_WORDS];
    #pragma unroll
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        msg[i] = msg_lds[i * WMMA_WG + (int)lid];
    }
#endif
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
    if (out_t_rows != 0) {
        *out_t_rows = im * MACRO_M + hash_row * MR;
    }
    if (out_t_cols != 0) {
        *out_t_cols = jm * MACRO_N + hash_col * HASH_NR;
    }
}

/* Layout self-test (host: CP_OCL_WMMA_SELFTEST=1), one wave32, using the same helpers as
   the GEMM kernel. a: 16x16 int8 row-major [m][k]; b: [n][k]; c_in, d_out: int32 [m][n].
   words[0..31]: reduce-scatter of a synthetic pattern; words[32], words[33]: the top and
   bottom 8x16 hash-tile XOR of D via wmma_tile_partials. ksplit: gfx12 A/B k mapping. */
__kernel void case33_wmma_selftest(__global const int *a, __global const int *b,
                                   __global const int *c_in, __global int *d_out,
                                   __global uint *words, int ksplit) {
    WMMA_EMU_DECL(32);
    const uint lane = (uint)get_local_id(0) & 31u;
    const int col = (int)(lane & 15u);
    const int kg0 = wmma_lane_kg0(lane, ksplit);
    /* row/col stride 4 dwords, k-group stride 1 dword */
    const wmma_ab av = wmma_load_ab(a + col * 4 + kg0, 1, ksplit);
    const wmma_ab bv = wmma_load_ab(b + col * 4 + kg0, 1, ksplit);
    wmma_acc c;
    c.s0 = c_in[wmma_c_row(0, lane) * 16 + col];
    c.s1 = c_in[wmma_c_row(1, lane) * 16 + col];
    c.s2 = c_in[wmma_c_row(2, lane) * 16 + col];
    c.s3 = c_in[wmma_c_row(3, lane) * 16 + col];
    c.s4 = c_in[wmma_c_row(4, lane) * 16 + col];
    c.s5 = c_in[wmma_c_row(5, lane) * 16 + col];
    c.s6 = c_in[wmma_c_row(6, lane) * 16 + col];
    c.s7 = c_in[wmma_c_row(7, lane) * 16 + col];
#if CASE32_WMMA_EMU
    const wmma_acc d = wmma_mac(av, bv, c WMMA_EMU_PASS, ksplit);
#else
    const wmma_acc d = wmma_mac(av, bv, c);
#endif
    d_out[wmma_c_row(0, lane) * 16 + col] = d.s0;
    d_out[wmma_c_row(1, lane) * 16 + col] = d.s1;
    d_out[wmma_c_row(2, lane) * 16 + col] = d.s2;
    d_out[wmma_c_row(3, lane) * 16 + col] = d.s3;
    d_out[wmma_c_row(4, lane) * 16 + col] = d.s4;
    d_out[wmma_c_row(5, lane) * 16 + col] = d.s5;
    d_out[wmma_c_row(6, lane) * 16 + col] = d.s6;
    d_out[wmma_c_row(7, lane) * 16 + col] = d.s7;

    uint p[32];
    #pragma unroll
    for (int t = 0; t < 32; ++t) {
        p[t] = (lane * 2654435761u + (uint)t * 2246822519u) ^ ((lane + 1u) * (uint)(t + 3));
    }
    words[lane] = wmma_reduce_scatter32(p, lane WMMA_EMU_PASS);

    uint q[32];
    #pragma unroll
    for (int t = 0; t < 32; ++t) {
        q[t] = 0u;
    }
    wmma_tile_partials(d, lane, &q[0], &q[1]);
    const uint w = wmma_reduce_scatter32(q, lane WMMA_EMU_PASS);
    if (lane < 2u) {
        words[32 + lane] = w;
    }
}
#endif /* CASE32_WMMA */
