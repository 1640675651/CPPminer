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

#if !defined(CASE32_WMMA) && !defined(CASE32_DPAS)
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
#elif defined(CASE32_WMMA)

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

/* Wave sub-tile. 64x64 (default): 16 accumulator blocks = 128 VGPRs, 32 hash tiles, one per
   lane. CASE32_WMMA_WAVE_COLS=32 (host: CP_OCL_WMMA_WAVE_N=32): 64x32, 8 accumulator blocks =
   64 VGPRs and 16 hash tiles per wave, each held by lanes t and t+16 after the reduce-scatter;
   a 128x128 macro block is then 8 waves (256 WIs). Frees the VGPRs that LDS staging or a deeper
   operand buffer needs without spilling. */
#define WMMA_WAVE_ROWS 64
#ifndef CASE32_WMMA_WAVE_COLS
#define CASE32_WMMA_WAVE_COLS 64
#endif
#if CASE32_WMMA_WAVE_COLS != 64 && CASE32_WMMA_WAVE_COLS != 32
#error CASE32_WMMA_WAVE_COLS must be 64 or 32
#endif
#define WMMA_WAVE_COLS CASE32_WMMA_WAVE_COLS
#define WMMA_BM (WMMA_WAVE_ROWS / 16)
#define WMMA_BN (WMMA_WAVE_COLS / 16)
#define WMMA_TILES (WMMA_BM * WMMA_BN * 2) /* 8x16 hash tiles per wave: 32 or 16 */
/* CASE32_WMMA_G12_HALFRS=1 (gfx12, 64x64 wave tile): tile ownership by lane half, milestone
   reduce-scatter over 16 lanes (15 exchanges instead of 31). */
#ifndef CASE32_WMMA_G12_HALFRS
#define CASE32_WMMA_G12_HALFRS 1
#endif
#define WMMA_G12_HALFRS (CASE32_WMMA == 12 && WMMA_TILES == 32 && CASE32_WMMA_G12_HALFRS)
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
/* CASE32_WMMA_RS_MASK (default: on for gfx12, off for gfx11): lane-mask halves (above) or the
   per-element select, which measured faster on gfx11 (Radeon 780M: 6.32 vs 6.17 TMAC/s). */
#ifndef CASE32_WMMA_RS_MASK
#if CASE32_WMMA == 12
#define CASE32_WMMA_RS_MASK 1
#else
#define CASE32_WMMA_RS_MASK 0
#endif
#endif
#if CASE32_WMMA_RS_MASK
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
#else
#define WMMA_RS_STAGE(p, lane, n, m)                                                  \
    do {                                                                              \
        const int hi_ = ((lane) & (m)) != 0u;                                         \
        _Pragma("unroll") for (int j_ = 0; j_ < (n) / 2; ++j_) {                      \
            const uint keep_ = hi_ ? (p)[j_ + (n) / 2] : (p)[j_];                     \
            const uint send_ = hi_ ? (p)[j_] : (p)[j_ + (n) / 2];                     \
            (p)[j_] = keep_ ^ WMMA_SWZ_XOR(send_, m);                                 \
        }                                                                             \
    } while (0)
#endif

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

/* 64x32 wave tile: p[t] (t = 0..15) this lane's partial of word t. Four stages over lane bits
   8..1 leave lane l with word l%16 summed over the 16 lanes of its half-wave; one exchange with
   lane l^16 completes it, so lanes t and t+16 both return the full word t (16 swizzles). */
inline uint wmma_reduce_scatter16(uint *p, uint lane WMMA_EMU_ARG) {
    WMMA_RS_STAGE(p, lane, 16, 8);
    WMMA_RS_STAGE(p, lane, 8, 4);
    WMMA_RS_STAGE(p, lane, 4, 2);
    WMMA_RS_STAGE(p, lane, 2, 1);
    return p[0] ^ WMMA_SWZ_XOR(p[0], 16);
}

/* gfx12 half-wave reduce-scatter (WMMA_G12_HALFRS): p[k] (k = 0..15) this lane's partial of
   word k of its half-wave. Four stages over lane bits 8..1: lane l returns word l%16 summed
   over the 16 lanes of its half (15 exchanges). */
inline uint wmma_reduce_scatter_half16(uint *p, uint lane WMMA_EMU_ARG) {
    WMMA_RS_STAGE(p, lane, 16, 8);
    WMMA_RS_STAGE(p, lane, 8, 4);
    WMMA_RS_STAGE(p, lane, 4, 2);
    WMMA_RS_STAGE(p, lane, 2, 1);
    return p[0];
}

/* Operands of one 16-k step (k-groups g..g+3) for the wave's 4 row and WMMA_BN column blocks. */
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

/* ---- CASE32_WMMA_LDS=1: operands staged through LDS (double buffered).
   One 16-k step of a 128x128 macro block needs k-groups g..g+3 of all 128 rows of A and
   all 128 columns of B: in the coalesced prepack that is one contiguous 2 KB range each
   (k-group stride = STRIP = MACRO_M dwords). The group loads it with one int4 per WI
   (MACRO_M / WG of them), stores it to LDS as [k-group][row] with a padded row stride,
   and every lane then reads its 2 (gfx12) or 4 (gfx11) k-group dwords per block, one
   ds_load_2addr per block. The next step is fetched before this step's WMMAs, so only
   the int4s are in flight (no register double buffer of 16 operands). */
#ifndef CASE32_WMMA_LDS
#define CASE32_WMMA_LDS 0
#endif
#define WMMA_LSTR (MACRO_M + 8)                 /* LDS dwords per k-group row: halves of a wave
                                                   (k-groups 2 apart) land 16 banks apart */
#define WMMA_LDS_BUF (4 * WMMA_LSTR)            /* one step of A (or B) */
/* A step of one operand is MACRO_M int4. With WG <= MACRO_M every WI fetches MACRO_M/WG int4
   of A and of B; with WG == 2*MACRO_M (64x32 wave tile, 256 WIs) the first MACRO_M WIs (whole
   waves) fetch A and the rest fetch B, one int4 each. */
#define WMMA_LDS_SPLIT (WMMA_WG == 2 * MACRO_M)
#if WMMA_LDS_SPLIT
#define WMMA_LDS_V4 1
#else
#define WMMA_LDS_V4 (MACRO_M / WMMA_WG)         /* int4 per WI per step per operand */
#endif
#if CASE32_WMMA_LDS && (WMMA_KG_DW_A != MACRO_M || WMMA_KG_DW_B != MACRO_N || \
                        (!WMMA_LDS_SPLIT && MACRO_M % WMMA_WG != 0))
#error CASE32_WMMA_LDS assumes a k-group strip of exactly MACRO_M (MACRO_N) dwords
#endif

inline void wmma_lds_fetch(__private int4 *na, __private int4 *nb, __global const int4 *a_blk,
                           __global const int4 *b_blk, int g, int lid) {
#if WMMA_LDS_SPLIT
    if (lid < MACRO_M) {
        na[0] = a_blk[g * (WMMA_KG_DW_A / 4) + lid];
    } else {
        nb[0] = b_blk[g * (WMMA_KG_DW_B / 4) + lid - MACRO_M];
    }
#else
    #pragma unroll
    for (int v = 0; v < WMMA_LDS_V4; ++v) {
        na[v] = a_blk[g * (WMMA_KG_DW_A / 4) + lid + v * WMMA_WG];
        nb[v] = b_blk[g * (WMMA_KG_DW_B / 4) + lid + v * WMMA_WG];
    }
#endif
}

inline void wmma_lds_store(__local int *la, __local int *lb, __private const int4 *na,
                           __private const int4 *nb, int lid) {
#if WMMA_LDS_SPLIT
    const int is_b = lid >= MACRO_M;
    const int d = (is_b ? lid - MACRO_M : lid) * 4; /* dword within the 4 k-group strips */
    const int q = d / MACRO_M;
    const int r = d - q * MACRO_M;
    if (is_b) {
        *(__local int4 *)(lb + q * WMMA_LSTR + r) = nb[0];
    } else {
        *(__local int4 *)(la + q * WMMA_LSTR + r) = na[0];
    }
#else
    #pragma unroll
    for (int v = 0; v < WMMA_LDS_V4; ++v) {
        const int d = (lid + v * WMMA_WG) * 4; /* dword within the 4 k-group strips */
        const int q = d / MACRO_M;
        const int r = d - q * MACRO_M;
        *(__local int4 *)(la + q * WMMA_LSTR + r) = na[v];
        *(__local int4 *)(lb + q * WMMA_LSTR + r) = nb[v];
    }
#endif
}

inline void wmma_lds_operands(__private wmma_ab *a, __private wmma_ab *b, __local const int *la,
                              __local const int *lb, int wm, int wn, uint lane) {
    const int m = (int)(lane & 15u);
#if CASE32_WMMA == 12
    const int h = (int)(lane >> 4);
    const int g0 = CASE32_WMMA_G12_KSPLIT ? 2 * h : h;
    const int g1 = CASE32_WMMA_G12_KSPLIT ? 2 * h + 1 : 2 + h;
#endif
    #pragma unroll
    for (int bi = 0; bi < WMMA_BM; ++bi) {
        const int r = wm * WMMA_WAVE_ROWS + bi * 16 + m;
#if CASE32_WMMA == 11
        a[bi] = (int4)(la[r], la[WMMA_LSTR + r], la[2 * WMMA_LSTR + r], la[3 * WMMA_LSTR + r]);
#else
        a[bi] = (int2)(la[g0 * WMMA_LSTR + r], la[g1 * WMMA_LSTR + r]);
#endif
    }
    #pragma unroll
    for (int bj = 0; bj < WMMA_BN; ++bj) {
        const int c = wn * WMMA_WAVE_COLS + bj * 16 + m;
#if CASE32_WMMA == 11
        b[bj] = (int4)(lb[c], lb[WMMA_LSTR + c], lb[2 * WMMA_LSTR + c], lb[3 * WMMA_LSTR + c]);
#else
        b[bj] = (int2)(lb[g0 * WMMA_LSTR + c], lb[g1 * WMMA_LSTR + c]);
#endif
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

    /* Hash tile owned by this lane after the reduce-scatter: t = bj*8 + bi*2 + half. With the
       64x32 wave tile (16 tiles) lanes t and t+16 hold the same tile; only the lower one
       (`owner`) writes results. */
#if WMMA_G12_HALFRS
    /* gfx12 half-wave reduce-scatter: lanes 0..15 own the top tiles, 16..31 the bottom ones;
       lane l owns block k = l%16 (bj = k/4, bi = k%4) */
    const int own_bj = (int)((lane >> 2) & 3u);
    const int own_bi = (int)(lane & 3u);
    const int own_half = (int)(lane >> 4);
#else
    const int own_bj = (int)((lane >> 3) & (uint)(WMMA_BN - 1));
    const int own_bi = (int)((lane >> 1) & 3u);
    const int own_half = (int)(lane & 1u);
#endif
    const int owner = WMMA_TILES == 32 || lane < 16u;
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
#elif CASE32_WMMA_LDS
    (void)a_lane;
    (void)b_lane;
    /* int4 arrays: the staging stores are 16-byte vector stores (an int array need not be
       16-byte aligned) */
    __local int4 lds_a4[2 * WMMA_LDS_BUF / 4];
    __local int4 lds_b4[2 * WMMA_LDS_BUF / 4];
    __local int *lds_a = (__local int *)lds_a4;
    __local int *lds_b = (__local int *)lds_b4;
    __global const int4 *a_blk =
            (__global const int4 *)(a_pre + (size_t)im * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_A);
    __global const int4 *b_blk =
            (__global const int4 *)(b_pre + (size_t)jm * (size_t)blocks_k * (size_t)MACRO_KB_BLOCK_B);
    const int total_steps = blocks_k * WMMA_KSTEPS;
    {
        int4 na[WMMA_LDS_V4];
        int4 nb[WMMA_LDS_V4];
        wmma_lds_fetch(na, nb, a_blk, b_blk, 0, (int)lid);
        wmma_lds_store(lds_a, lds_b, na, nb, (int)lid);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
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
#elif CASE32_WMMA_LDS
        WMMA_KLOOP_UNROLL
        for (int ks = 0; ks < WMMA_KSTEPS; ++ks) {
            const int s = ms * WMMA_KSTEPS + ks;
            const int cur = s & 1;
            const int more = s + 1 < total_steps; /* uniform across the group */
            int4 na[WMMA_LDS_V4];
            int4 nb[WMMA_LDS_V4];
            if (more) {
                wmma_lds_fetch(na, nb, a_blk, b_blk, (s + 1) * 4, (int)lid);
            }
            wmma_ab a[WMMA_BM];
            wmma_ab b[WMMA_BN];
            wmma_lds_operands(a, b, lds_a + cur * WMMA_LDS_BUF, lds_b + cur * WMMA_LDS_BUF, wm, wn,
                              lane);
            wmma_mac_step(acc, a, b WMMA_EMU_PASS);
            if (more) {
                /* buffer cur^1 was last read in step s-1; the barrier ending s-1 freed it */
                wmma_lds_store(lds_a + (cur ^ 1) * WMMA_LDS_BUF, lds_b + (cur ^ 1) * WMMA_LDS_BUF, na,
                               nb, (int)lid);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        kg_flat += KGROUPS;
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

        /* Milestone (one per KR panel, cumulative C): word of hash tile `lane` (% 16). */
#if WMMA_G12_HALFRS
        /* gfx12: a lane's 8 accumulator rows all belong to its half's hash tile, so the lane
           partial of block k is one XOR of the block; the word of tile (k, half) sums over the
           16 lanes of that half only: 15 exchanges instead of 31. */
        uint p[16];
        #pragma unroll
        for (int bi = 0; bi < WMMA_BM; ++bi) {
            #pragma unroll
            for (int bj = 0; bj < WMMA_BN; ++bj) {
                const wmma_acc c = acc[bi * WMMA_BN + bj];
                p[bj * 4 + bi] = as_uint(c.s0 ^ c.s1 ^ c.s2 ^ c.s3 ^ c.s4 ^ c.s5 ^ c.s6 ^ c.s7);
            }
        }
        const uint x = wmma_reduce_scatter_half16(p, lane WMMA_EMU_PASS);
#else
        uint p[WMMA_TILES];
        #pragma unroll
        for (int bi = 0; bi < WMMA_BM; ++bi) {
            #pragma unroll
            for (int bj = 0; bj < WMMA_BN; ++bj) {
                wmma_tile_partials(acc[bi * WMMA_BN + bj], lane, &p[bj * 8 + bi * 2],
                                   &p[bj * 8 + bi * 2 + 1]);
            }
        }
#if WMMA_TILES == 32
        const uint x = wmma_reduce_scatter32(p, lane WMMA_EMU_PASS);
#else
        const uint x = wmma_reduce_scatter16(p, lane WMMA_EMU_PASS);
#endif
#endif /* WMMA_G12_HALFRS */

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
                if (owner) {
                    tile_xor[out_idx] = x;
                }
            }
        }
    }

    /* after the k loop: no barrier follows, the duplicate lanes may leave */
    if (!owner || !fuse_jackpot || !xor_after_milestone || a_key8 == 0 || bound == 0) {
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
#else /* CASE32_DPAS */

/* =====================================================================================
 * Intel XMX (DPAS) path: --ocl-dot dpas, -DCASE32_DPAS=8 (Xe-HPG: Arc A-series DG2,
 * Arrow Lake-H; minimum sub-group size 8) or -DCASE32_DPAS=16 (Xe2: Battlemage B580/B570,
 * Lunar Lake; minimum sub-group size 16). Needs the 8x16 hash tile and the RANK=4 prepack.
 *
 * Builtins: cl_intel_subgroup_matrix_multiply_accumulate (Khronos registry, rev 1.1.0).
 * Quoted from the spec (the parts this path relies on):
 *   "// These functions are available to devices where the minimum subgroup size is 8.
 *    // For these devices, the subgroup size must be 8 (the minimum supported subgroup
 *    // size). [...]
 *    int8 intel_sub_group_i8_i8_matrix_mad_k32(int8  a, int8  b, int8 acc);  // M = 8"
 *   "// These functions are available to devices where the minimum subgroup size is 16.
 *    [...]
 *    int8 intel_sub_group_i8_i8_matrix_mad_k32(short8  a, int8  b, int8 acc);  // M = 8"
 *   "The value for M is determined by the number of vector components in the source
 *    operand a." "[...] since each work item is contributing 32 bits (the size of a uint)
 *    of data per row of this matrix, each work item is contributing four 8-bit integer
 *    values per row." (sub-group size 8)
 *   "b [...] Each work item contributes one column of this matrix. Therefore, the number
 *    of columns N is equivalent to the subgroup size." "[...] each work item must
 *    contribute 256 bits of source data to contribute K values. The 256 bits of source
 *    data are packed and passed as the int8 argument b."
 *   "acc [...] each work item contributes one column of accumulation values." "[...] each
 *    work item will receive one column of result values."
 *   "These functions must be encountered by all work items in the subgroup executing the
 *    kernel." "For 8-bit matrices, K must be equal to 32." "[...] the accumulation value
 *    acc and result value are signed 32-bit integers."
 *   Coding sample (sub-group size 8): result = acc + sum over i = 0..7 of
 *    dot(as_char4(sub_group_broadcast(a_row, i)), as_char4(b.s<i>))
 *    -> lane i's A dword holds k 4i..4i+3; b.s<i> holds k 4i..4i+3 of this lane's column.
 *
 * Layout assumptions (lane = get_sub_group_local_id(); one DPAS = M 8 x N SG x K 32):
 *   SG 8  (int8 a):   a.s<m> = row m, k 4*lane .. 4*lane+3          (spec coding sample)
 *   SG 16 (short8 a): a.s<m> = row m, two k bytes per lane (low byte = lower k):
 *                       CASE32_DPAS_AK=0: k 2*lane, 2*lane+1         (natural extension of
 *                                         the sample: 32 k / 16 lanes, register word = lane)
 *                       CASE32_DPAS_AK=1: k 4*(lane%8) + 2*(lane/8) + {0,1} (alternative,
 *                                         tried by the self-test)
 *   both: b.s<i> = column `lane` of the block, k 4i..4i+3 (byte j = k 4i+j);
 *         acc.s<m> / result.s<m> = C[row m][column `lane`].
 *   Unverified until CP_OCL_DPAS_SELFTEST=1 runs on the device; the self-test checks every
 *   D element and prints which A variant passes.
 *
 * Work split. One sub-group owns DPAS_TM x DPAS_TN hash tiles (8x16 each; defaults 4x1
 * for SG 8 = 32x16 of C, 4x2 for SG 16 = 32x32), i.e. DPAS_TM row blocks x
 * DPAS_TN*16/SG column blocks of DPAS results. The work-group is one macro block of
 * (MACRO_M/(8*DPAS_TM)) * (MACRO_N/(16*DPAS_TN)) sub-groups: 256 WIs for both defaults at
 * 128x128. Per 32-k step a lane loads one 8-dword column slice of B per column block and
 * one 8-row A slice per row block straight from the default coalesced prepack (flat
 * k-group g of row r / column c = dword g*STRIP + r, STRIP = MACRO_M dwords), no new
 * layout: a lane's A slice is 8 consecutive dwords (rows r0..r0+7 at one k-group).
 *
 *   sub-group (SG 16, TM=4, TN=2)       hash tile t = tj*TM + ti  ->  lane t after the
 *             tj=0       tj=1           milestone reduce-scatter (lanes >= TM*TN idle
 *   ti=0  | t0       | t4       |       for msg[16]/BLAKE3 but take part in every DPAS
 *   ti=1  | t1       | t5       |       and shuffle)
 *   ti=2  | t2       | t6       |       SG 8: each 8x16 hash tile = 2 DPAS column blocks
 *   ti=3  | t3       | t7       |       (lanes hold columns h*8 + lane, h = 0, 1)
 *
 * Milestones (every KR = 128 k = 4 DPAS steps): each lane XORs its 8 (SG 16) or 16 (SG 8)
 * accumulator elements per hash tile, then an XOR reduce-scatter over
 * intel_sub_group_shuffle_xor (masks SG/2 .. 1) leaves hash tile t's word in lane t.
 * XOR and int32 wrap-around accumulation are order-independent, so the words equal the
 * dot4 kernel's bit for bit.
 * ===================================================================================== */
#if CASE32_DPAS != 8 && CASE32_DPAS != 16
#error CASE32_DPAS must be 8 (Xe-HPG) or 16 (Xe2)
#endif
#if MR != 8 || NR != 16 || HASH_NR != 16 || RANK != 4 || KR != 128
#error DPAS path needs the 8x16 hash tile, RANK=4 prepack and KR=128
#endif
#if CASE32_USE_LDS
#error DPAS path reads the coalesced prepack directly (no LDS staging)
#endif

#define DPAS_SG CASE32_DPAS

#if defined(CP_DPAS_SYNTAX_CHECK)
/* Host-side syntax check with stock clang (-DCP_DPAS_SYNTAX_CHECK): declare the Intel
   builtins the device compiler provides. */
#define DPAS_OVL __attribute__((overloadable))
int8 DPAS_OVL intel_sub_group_i8_i8_matrix_mad_k32(int8 a, int8 b, int8 acc);
int8 DPAS_OVL intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc);
uint DPAS_OVL intel_sub_group_shuffle_xor(uint x, uint m);
#define DPAS_REQD_SG __attribute__((intel_reqd_sub_group_size(DPAS_SG)))
#elif defined(CP_DPAS_EMULATE)
/* Functional emulation on an AMD GPU (host: CP_OCL_DPAS_EMULATE=8|16 with --ocl-dot dpas).
   A "sub-group" is DPAS_SG consecutive local ids = consecutive hardware lanes of one
   wave; cross-lane reads use ds_bpermute. The DPAS builtin follows the spec's coding
   sample (SG 8) and its natural SG 16 extension (CASE32_DPAS_AK=0 layout), so this
   validates the GEMM indexing, reduce-scatter, milestone words and jackpot of this path
   -- not the Intel hardware layout, which only the self-test on the device can confirm. */
#define DPAS_REQD_SG
#define get_sub_group_local_id() ((uint)get_local_id(0) % (uint)DPAS_SG)
#define get_sub_group_id() ((uint)get_local_id(0) / (uint)DPAS_SG)
#define get_sub_group_size() ((uint)DPAS_SG)
#ifdef CP_DPAS_EMULATE_HOST
/* Value of x in emulated lane `src` of this lane's sub-group: provided by a host-side
   harness that runs the kernel source compiled for the CPU (one thread per lane). */
int dpas_emu_read(int x, uint src);
#else
inline uint dpas_emu_hw_lane(void) {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
/* Value of x in emulated lane `src` of this lane's sub-group. */
inline int dpas_emu_read(int x, uint src) {
    const uint base = dpas_emu_hw_lane() - get_sub_group_local_id();
    return __builtin_amdgcn_ds_bpermute((int)((base + src) << 2), x);
}
#endif
#define intel_sub_group_shuffle_xor(x, m)                                              \
    as_uint(dpas_emu_read(as_int(x), get_sub_group_local_id() ^ (uint)(m)))
inline int dpas_emu_dot4(int a, int b, int acc) {
    const char4 x = as_char4(a);
    const char4 y = as_char4(b);
    return acc + (int)x.s0 * (int)y.s0 + (int)x.s1 * (int)y.s1 + (int)x.s2 * (int)y.s2 +
           (int)x.s3 * (int)y.s3;
}
/* Dword i (k 4i..4i+3) of one A row, gathered across the sub-group. */
inline int dpas_emu_a_dword(int a_lane_bits, int i) {
#if DPAS_SG == 8
    return dpas_emu_read(a_lane_bits, (uint)i);
#else
    const int lo = dpas_emu_read(a_lane_bits, (uint)(2 * i)) & 0xffff;
    const int hi = dpas_emu_read(a_lane_bits, (uint)(2 * i + 1)) << 16;
    return lo | hi;
#endif
}
/* spec __intel_vector_matrix_multiply_accumulate_k32 */
inline int dpas_emu_row(int a_lane_bits, int8 b, int acc) {
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 0), b.s0, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 1), b.s1, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 2), b.s2, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 3), b.s3, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 4), b.s4, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 5), b.s5, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 6), b.s6, acc);
    acc = dpas_emu_dot4(dpas_emu_a_dword(a_lane_bits, 7), b.s7, acc);
    return acc;
}
#if DPAS_SG == 8
inline int8 intel_sub_group_i8_i8_matrix_mad_k32(int8 a, int8 b, int8 acc) {
    const int8 x = a;
#else
inline int8 intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc) {
    const int8 x = convert_int8(a);
#endif
    int8 r;
    r.s0 = dpas_emu_row(x.s0, b, acc.s0);
    r.s1 = dpas_emu_row(x.s1, b, acc.s1);
    r.s2 = dpas_emu_row(x.s2, b, acc.s2);
    r.s3 = dpas_emu_row(x.s3, b, acc.s3);
    r.s4 = dpas_emu_row(x.s4, b, acc.s4);
    r.s5 = dpas_emu_row(x.s5, b, acc.s5);
    r.s6 = dpas_emu_row(x.s6, b, acc.s6);
    r.s7 = dpas_emu_row(x.s7, b, acc.s7);
    return r;
}
#else
/* Intel device compiler: the builtins come with cl_intel_subgroup_matrix_multiply_accumulate
   (the host checked CL_DEVICE_EXTENSIONS). The extension macro is deliberately not
   required, so a driver that omits it still builds; missing builtins fail loudly anyway. */
#define DPAS_REQD_SG __attribute__((intel_reqd_sub_group_size(DPAS_SG)))
#endif

#ifndef DPAS_TM
#define DPAS_TM 4
#endif
#ifndef DPAS_TN
#if DPAS_SG == 8
#define DPAS_TN 1
#else
#define DPAS_TN 2
#endif
#endif
#ifndef CASE32_DPAS_AK
#define CASE32_DPAS_AK 0
#endif
#define DPAS_TILES (DPAS_TM * DPAS_TN) /* hash tiles per sub-group */
#define DPAS_NB (16 / DPAS_SG)         /* DPAS column blocks per 16-wide hash tile */
#define DPAS_BN (DPAS_TN * DPAS_NB)    /* DPAS column blocks per sub-group */
#define DPAS_SG_ROWS (8 * DPAS_TM)
#define DPAS_SG_COLS (16 * DPAS_TN)
#if DPAS_TILES > DPAS_SG || (MACRO_M % DPAS_SG_ROWS) != 0 || (MACRO_N % DPAS_SG_COLS) != 0
#error DPAS_TM x DPAS_TN must fit the sub-group size and divide the macro block
#endif
#define DPAS_SGS_M (MACRO_M / DPAS_SG_ROWS)
#define DPAS_KG_DW_A (MACRO_KG_STRIP_A / 4) /* dwords between consecutive k-groups */
#define DPAS_KG_DW_B (MACRO_KG_STRIP_B / 4)
#define DPAS_KSTEPS (KGROUPS / 8)          /* 32-k DPAS steps per KR panel */

#if DPAS_SG == 8
typedef int8 dpas_a;
#else
typedef short8 dpas_a;
#endif
typedef int8 dpas_b;
typedef int8 dpas_acc;

/* First k-group (relative to the 32-k step) of this lane's A bytes. */
inline int dpas_a_lane_kg(uint lane, int ak) {
#if DPAS_SG == 8
    (void)ak;
    return (int)lane;
#else
    return ak ? (int)(lane & 7u) : (int)(lane >> 1);
#endif
}

/* Which 16-bit half of that k-group dword (SG 16 only). */
inline int dpas_a_lane_half(uint lane, int ak) {
#if DPAS_SG == 8
    (void)lane;
    (void)ak;
    return 0;
#else
    return ak ? (int)(lane >> 3) : (int)(lane & 1u);
#endif
}

/* One lane's A operand: p = dword of row 0 of the 8-row block at this lane's k-group;
   rows are consecutive dwords. */
inline dpas_a dpas_load_a(__global const int *p, int hi16) {
    const int8 v = vload8(0, p);
#if DPAS_SG == 8
    (void)hi16;
    return v;
#else
    /* sign-extend the selected 16-bit half so the conversion is in range */
    const int8 h = hi16 ? (v >> 16) : ((v << 16) >> 16);
    return convert_short8(h);
#endif
}

/* One lane's B operand: p = dword of this lane's column at the step's first k-group;
   kg = dword stride between k-groups. b.s<i> = k-group i = k 4i..4i+3. */
inline dpas_b dpas_load_b(__global const int *p, int kg) {
    return (int8)(p[0], p[kg], p[2 * kg], p[3 * kg], p[4 * kg], p[5 * kg], p[6 * kg],
                  p[7 * kg]);
}

inline dpas_acc dpas_mac(dpas_a a, dpas_b b, dpas_acc c) {
    return intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
}

inline uint dpas_xor8(dpas_acc c) {
    return as_uint(c.s0 ^ c.s1 ^ c.s2 ^ c.s3 ^ c.s4 ^ c.s5 ^ c.s6 ^ c.s7);
}

/* This lane's XOR partial of hash tile (ti, tj): DPAS_NB accumulators of 8 rows each. */
inline uint dpas_tile_partial(__private const dpas_acc *acc, int ti, int tj) {
    uint x = 0u;
    #pragma unroll
    for (int h = 0; h < DPAS_NB; ++h) {
        x ^= dpas_xor8(acc[ti * DPAS_BN + tj * DPAS_NB + h]);
    }
    return x;
}

/* Reduce-scatter stage over lane bit m (see WMMA_RS_STAGE), via cl_intel_subgroups. */
#define DPAS_RS_STAGE(p, lane, n, m)                                                  \
    do {                                                                              \
        const int hi_ = ((lane) & (m)) != 0u;                                         \
        _Pragma("unroll") for (int j_ = 0; j_ < (n) / 2; ++j_) {                      \
            const uint keep_ = hi_ ? (p)[j_ + (n) / 2] : (p)[j_];                     \
            const uint send_ = hi_ ? (p)[j_] : (p)[j_ + (n) / 2];                     \
            (p)[j_] = keep_ ^ intel_sub_group_shuffle_xor(send_, (uint)(m));          \
        }                                                                             \
    } while (0)

/* p[t] (t = 0..SG-1): this lane's partial of word t. Returns the XOR over all lanes of
   p[lane]: lane t receives the complete word t. All lanes of the sub-group must call. */
inline uint dpas_reduce_scatter(uint *p, uint lane) {
#if DPAS_SG == 16
    DPAS_RS_STAGE(p, lane, 16, 8);
#endif
    DPAS_RS_STAGE(p, lane, 8, 4);
    DPAS_RS_STAGE(p, lane, 4, 2);
    DPAS_RS_STAGE(p, lane, 2, 1);
    return p[0];
}

/* One 32-k step (k-groups g..g+7) for the sub-group's DPAS_TM x DPAS_BN blocks. */
inline void dpas_step(__private dpas_acc *acc, __global const int *a_lane,
                      __global const int *b_lane, int g, int a_half) {
    dpas_b b[DPAS_BN];
    #pragma unroll
    for (int j = 0; j < DPAS_BN; ++j) {
        b[j] = dpas_load_b(b_lane + g * DPAS_KG_DW_B + j * DPAS_SG, DPAS_KG_DW_B);
    }
    #pragma unroll
    for (int i = 0; i < DPAS_TM; ++i) {
        const dpas_a a = dpas_load_a(a_lane + g * DPAS_KG_DW_A + i * 8, a_half);
        #pragma unroll
        for (int j = 0; j < DPAS_BN; ++j) {
            acc[i * DPAS_BN + j] = dpas_mac(a, b[j], acc[i * DPAS_BN + j]);
        }
    }
}

DPAS_REQD_SG
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
    if (fuse_jackpot && found_flag != 0 && *found_flag != 0) {
        return;
    }

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

    const uint lane = get_sub_group_local_id();
    const int sg = (int)get_sub_group_id();
    const int sg_m = sg % DPAS_SGS_M;
    const int sg_n = sg / DPAS_SGS_M;

    /* Per-lane operand streams inside the macro block. A: rows sg_m*DPAS_SG_ROWS.. at
       this lane's k-group; B: column sg_n*DPAS_SG_COLS + lane (block j adds j*SG). */
    const int a_half = dpas_a_lane_half(lane, CASE32_DPAS_AK);
    __global const int *a_lane =
            (__global const int *)(a_pre + (size_t)im * (size_t)blocks_k *
                                                   (size_t)MACRO_KB_BLOCK_A) +
            dpas_a_lane_kg(lane, CASE32_DPAS_AK) * DPAS_KG_DW_A + sg_m * DPAS_SG_ROWS;
    __global const int *b_lane =
            (__global const int *)(b_pre + (size_t)jm * (size_t)blocks_k *
                                                   (size_t)MACRO_KB_BLOCK_B) +
            sg_n * DPAS_SG_COLS + (int)lane;

    /* Hash tile owned by this lane after the reduce-scatter: t = tj*DPAS_TM + ti. */
    const int owns_tile = lane < (uint)DPAS_TILES;
    const int own_ti = (int)lane % DPAS_TM;
    const int own_tj = (int)lane / DPAS_TM;
    const int hash_row = sg_m * DPAS_TM + own_ti; /* in macro */
    const int hash_col = sg_n * DPAS_TN + own_tj; /* in macro */
    const int tr_global = im * MICRO_M + hash_row;
    const int hash_tc_global = jm * HASH_MICRO_N + hash_col;
    const int hash_spatial_id = tr_global * (N / HASH_NR) + hash_tc_global;

    uint msg[PP_JACKPOT_WORDS];
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        msg[i] = 0u;
    }
    dpas_acc acc[DPAS_TM * DPAS_BN];
    #pragma unroll
    for (int i = 0; i < DPAS_TM * DPAS_BN; ++i) {
        acc[i] = (dpas_acc)(0);
    }

    int kg_flat = 0;
    for (int ms = 0; ms < blocks_k; ++ms) {
        for (int ks = 0; ks < DPAS_KSTEPS; ++ks) {
            dpas_step(acc, a_lane, b_lane, kg_flat, a_half);
            kg_flat += 8;
        }

        /* Milestone (one per KR panel, cumulative C): word of hash tile `lane`. */
        uint p[DPAS_SG];
        #pragma unroll
        for (int t = 0; t < DPAS_SG; ++t) {
            p[t] = t < DPAS_TILES ? dpas_tile_partial(acc, t % DPAS_TM, t / DPAS_TM) : 0u;
        }
        const uint x = dpas_reduce_scatter(p, lane);

        if (owns_tile && xor_after_milestone) {
            if (fuse_jackpot) {
                if (ms < PP_MAX_MILESTONES) {
                    const int tid = ms % PP_JACKPOT_WORDS;
                    const uint contribution =
                            (ms + PP_JACKPOT_WORDS < num_milestones) ? pp_rotl32(x, PP_LROT) : x;
                    msg[tid] ^= contribution;
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

    if (!owns_tile || !fuse_jackpot || !xor_after_milestone || a_key8 == 0 || bound == 0) {
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
    if (out_t_rows != 0) {
        *out_t_rows = im * MACRO_M + hash_row * MR;
    }
    if (out_t_cols != 0) {
        *out_t_cols = jm * MACRO_N + hash_col * HASH_NR;
    }
}

/* Layout self-test (host: CP_OCL_DPAS_SELFTEST=1), one sub-group, through the same
   helpers as the GEMM kernel. One 8x16 hash tile = DPAS_NB DPAS calls (1 on SG 16, 2 on
   SG 8). a: 8x32 int8 as dwords [k-group q][row m] (q*8 + m, k 4q..4q+3); b: 32x16 int8
   as dwords [q][column n] (q*16 + n); c_in, d_out: int32 [m][n] row-major.
   words[0..SG-1]: reduce-scatter of a synthetic pattern; words[SG]: the hash-tile XOR of D
   via dpas_tile_partial + reduce-scatter (lane 0); words[SG+1]: get_sub_group_size();
   words[SG+2]: lanes whose get_sub_group_local_id() == get_local_id(0) (via shuffles).
   ak: SG 16 A k mapping (CASE32_DPAS_AK). */
DPAS_REQD_SG
__kernel void case33_dpas_selftest(__global const int *a, __global const int *b,
                                   __global const int *c_in, __global int *d_out,
                                   __global uint *words, int ak) {
    const uint lane = get_sub_group_local_id();
    const dpas_a av = dpas_load_a(a + dpas_a_lane_kg(lane, ak) * 8, dpas_a_lane_half(lane, ak));
    dpas_acc d[DPAS_NB];
    #pragma unroll
    for (int h = 0; h < DPAS_NB; ++h) {
        const int col = h * DPAS_SG + (int)lane;
        const dpas_b bv = dpas_load_b(b + col, 16);
        dpas_acc c;
        c.s0 = c_in[0 * 16 + col];
        c.s1 = c_in[1 * 16 + col];
        c.s2 = c_in[2 * 16 + col];
        c.s3 = c_in[3 * 16 + col];
        c.s4 = c_in[4 * 16 + col];
        c.s5 = c_in[5 * 16 + col];
        c.s6 = c_in[6 * 16 + col];
        c.s7 = c_in[7 * 16 + col];
        d[h] = dpas_mac(av, bv, c);
        d_out[0 * 16 + col] = d[h].s0;
        d_out[1 * 16 + col] = d[h].s1;
        d_out[2 * 16 + col] = d[h].s2;
        d_out[3 * 16 + col] = d[h].s3;
        d_out[4 * 16 + col] = d[h].s4;
        d_out[5 * 16 + col] = d[h].s5;
        d_out[6 * 16 + col] = d[h].s6;
        d_out[7 * 16 + col] = d[h].s7;
    }

    uint p[DPAS_SG];
    #pragma unroll
    for (int t = 0; t < DPAS_SG; ++t) {
        p[t] = (lane * 2654435761u + (uint)t * 2246822519u) ^ ((lane + 1u) * (uint)(t + 3));
    }
    words[lane] = dpas_reduce_scatter(p, lane);

    /* hash tile (0, 0) = d[0..DPAS_NB): partial into word 0, reduce-scatter to lane 0 */
    uint q[DPAS_SG];
    #pragma unroll
    for (int t = 0; t < DPAS_SG; ++t) {
        q[t] = 0u;
    }
    q[0] = dpas_tile_partial(d, 0, 0);
    const uint w = dpas_reduce_scatter(q, lane);

    /* Lane-id sanity: bit l set when lane l == get_local_id(0) (distinct bits, so the
       XOR reduction is their OR); the host expects all SG bits. */
    uint same[DPAS_SG];
    #pragma unroll
    for (int t = 0; t < DPAS_SG; ++t) {
        same[t] = 0u;
    }
    same[0] = (lane == (uint)get_local_id(0)) ? (1u << lane) : 0u;
    const uint same_mask = dpas_reduce_scatter(same, lane);
    if (lane == 0u) {
        words[DPAS_SG] = w;
        words[DPAS_SG + 1] = get_sub_group_size();
        words[DPAS_SG + 2] = same_mask;
    }
}
#endif /* CASE32_DPAS */
