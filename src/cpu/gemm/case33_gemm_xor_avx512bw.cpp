#include "case33_gemm_xor_avx512bw.hpp"
#include "case33_gemm_xor_avx2.hpp"
#include "case33_gemm_xor.hpp"

#include <immintrin.h>

#if defined(_MSC_VER)
#define CASE33_FORCEINLINE __forceinline
/* MSVC has no inline asm on x64; its optimizer does not reassociate integer SIMD chains. */
#define CASE33_PIN_ZMM(x) ((void)0)
#else
#define CASE33_FORCEINLINE inline __attribute__((always_inline))
/* Empty asm with a "+v" (any xmm/ymm/zmm, incl. 16-31) operand: keeps GCC's -ftree-reassoc
 * from rewriting acc += p0; acc += p1; ... as acc + (p0 + p1 + ...), which would keep every
 * product of the unrolled body live and spill (same fix as the AVX2 kernel). */
#define CASE33_PIN_ZMM(x) __asm__("" : "+v"(x))
#endif

/* Base AVX-512 (AVX512F + AVX512BW) micro-kernel for CPUs without AVX512_VNNI, mainly
 * Skylake-X / Skylake-SP class parts. Same structure as the AVX512-VNNI zmm pair kernel:
 * two vertically adjacent 8x16 hash tiles (rows r..r+7 and r+8..r+15 of the same 16 columns)
 * held as one 16x16 register block, one zmm accumulator per column with lanes 0-7 = tile 0
 * rows and lanes 8-15 = tile 1 rows, so every B broadcast feeds both tiles.
 *
 * Each rank-4 update is the AVX2 fast-path triple at 512-bit width:
 *   vpmaddubsw (u8 x s8 -> s16 pair sums), vpmaddwd with ones (-> s32), vpaddd.
 * With A stored as A+128 in [1, 254] and B noise in [-63, 63] the pair sums are at most
 * 2 * 254 * 63 = 32004 < 32767, so vpmaddubsw never saturates and every int32 lane equals
 * the exact u8 x s8 dot product, bit-identical to scalar / AVX2 / vpdpbusd.
 *
 * Exact s8s8 mode needs vpsignb, which AVX-512 does not have; that mode runs the AVX2 kernel
 * (every AVX512BW CPU also has AVX2) once per tile. */

namespace {

constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;
constexpr int kKGroups = kKR / kRank;

static_assert(kMR == 8, "AVX512BW ukernel assumes 8-row tiles (one ymm half per tile)");
static_assert(kNR == 2 * kColsPerGroup, "AVX512BW ukernel assumes 16 columns (2 groups)");
static_assert(kKGroups % 2 == 0, "AVX512BW ukernel unrolls k-groups by 2");

CASE33_FORCEINLINE __m512i load_a_pair(const int8_t *a0, const int8_t *a1) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a1));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/* acc += sum of 4 u8 x s8 products per int32 lane (exact for the ranges above). */
CASE33_FORCEINLINE __m512i rank4_maddubs_zmm(const __m512i acc, const __m512i ua,
                                             const __m512i sb, const __m512i ones16) {
    const __m512i pair16 = _mm512_maddubs_epi16(ua, sb);
    __m512i r = _mm512_add_epi32(acc, _mm512_madd_epi16(pair16, ones16));
    CASE33_PIN_ZMM(r);
    return r;
}

CASE33_FORCEINLINE void rank4_kgroup_update8_zmm(__m512i acc[kColsPerGroup], const __m512i ua,
                                                 const int32_t *bp, const __m512i ones16) {
    acc[0] = rank4_maddubs_zmm(acc[0], ua, _mm512_set1_epi32(bp[0]), ones16);
    acc[1] = rank4_maddubs_zmm(acc[1], ua, _mm512_set1_epi32(bp[1]), ones16);
    acc[2] = rank4_maddubs_zmm(acc[2], ua, _mm512_set1_epi32(bp[2]), ones16);
    acc[3] = rank4_maddubs_zmm(acc[3], ua, _mm512_set1_epi32(bp[3]), ones16);
    acc[4] = rank4_maddubs_zmm(acc[4], ua, _mm512_set1_epi32(bp[4]), ones16);
    acc[5] = rank4_maddubs_zmm(acc[5], ua, _mm512_set1_epi32(bp[5]), ones16);
    acc[6] = rank4_maddubs_zmm(acc[6], ua, _mm512_set1_epi32(bp[6]), ones16);
    acc[7] = rank4_maddubs_zmm(acc[7], ua, _mm512_set1_epi32(bp[7]), ones16);
}

CASE33_FORCEINLINE void zmm_pair_kgroups(__m512i acc0[kColsPerGroup], __m512i acc1[kColsPerGroup],
                                         const int8_t *a_tile0, const int8_t *a_tile1,
                                         const int8_t *b_jg0, const int8_t *b_jg1,
                                         const __m512i ones16) {
    for (int kg = 0; kg < kKGroups; ++kg) {
        const __m512i ua = load_a_pair(a_tile0 + kg * 32, a_tile1 + kg * 32);
        const int32_t *bp0 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg) * 32);
        const int32_t *bp1 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg) * 32);
        rank4_kgroup_update8_zmm(acc0, ua, bp0, ones16);
        rank4_kgroup_update8_zmm(acc1, ua, bp1, ones16);
    }
}

CASE33_FORCEINLINE void apply_b_comp_to_acc_zmm(__m512i acc0[kColsPerGroup],
                                                __m512i acc1[kColsPerGroup],
                                                const int32_t *b_comp_slice, int global_col0) {
    for (int c = 0; c < kColsPerGroup; ++c) {
        acc0[c] = _mm512_add_epi32(acc0[c],
                                   _mm512_set1_epi32(b_comp_slice[global_col0 + c]));
        acc1[c] = _mm512_add_epi32(
                acc1[c], _mm512_set1_epi32(b_comp_slice[global_col0 + kColsPerGroup + c]));
    }
}

CASE33_FORCEINLINE uint32_t reduce_xor_epi32(__m256i v) {
    v = _mm256_xor_si256(v, _mm256_permute2x128_si256(v, v, 0x01));
    __m128i x = _mm256_castsi256_si128(v);
    x = _mm_xor_si128(x, _mm_srli_si128(x, 8));
    x = _mm_xor_si128(x, _mm_srli_si128(x, 4));
    return static_cast<uint32_t>(_mm_cvtsi128_si32(x));
}

/* XOR of each 8x16 tile: lanes 0-7 of every accumulator form tile 0, lanes 8-15 tile 1. */
CASE33_FORCEINLINE void xor_pair_acc(const __m512i acc0[kColsPerGroup],
                                     const __m512i acc1[kColsPerGroup], uint32_t *x0,
                                     uint32_t *x1) {
    __m512i x = _mm512_setzero_si512();
    for (int c = 0; c < kColsPerGroup; ++c) {
        x = _mm512_xor_si512(x, acc0[c]);
        x = _mm512_xor_si512(x, acc1[c]);
    }
    *x0 = reduce_xor_epi32(_mm512_castsi512_si256(x));
    *x1 = reduce_xor_epi32(_mm512_extracti64x4_epi64(x, 1));
}

void avx512bw_pair_fast_impl(const int8_t *a_base0, const int8_t *a_base1,
                             const int8_t *b_base, int blocks_k, int N, int global_col0,
                             size_t spatial_tile_id0, size_t spatial_tile_id1,
                             size_t tile_count, const int32_t *b_comp_ms,
                             bool xor_after_milestone, uint32_t *tile_xor_out) {
    const __m512i ones16 = _mm512_set1_epi16(1);
    __m512i acc0[kColsPerGroup];
    __m512i acc1[kColsPerGroup];
    for (int col = 0; col < kColsPerGroup; ++col) {
        acc0[col] = _mm512_setzero_si512();
        acc1[col] = _mm512_setzero_si512();
    }

    int ms = 0;
    for (int kb = 0; kb < blocks_k; ++kb) {
        const int8_t *a_tile0 = a_base0 + static_cast<size_t>(kb) * kPanelA;
        const int8_t *a_tile1 = a_base1 + static_cast<size_t>(kb) * kPanelA;
        const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
        zmm_pair_kgroups(acc0, acc1, a_tile0, a_tile1, b_tile,
                         b_tile + static_cast<size_t>(kKGroups) * 32, ones16);
        if (b_comp_ms) {
            apply_b_comp_to_acc_zmm(acc0, acc1,
                                    b_comp_ms + static_cast<size_t>(ms) * static_cast<size_t>(N),
                                    global_col0);
        }
        if (xor_after_milestone) {
            uint32_t x0 = 0, x1 = 0;
            xor_pair_acc(acc0, acc1, &x0, &x1);
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id0] = x0;
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id1] = x1;
        }
        ++ms;
    }
}

} // namespace

void case33_avx512bw_micro_gemm_xor_fused_k_x2(
        const int8_t *a_base0, const int8_t *a_base1, const int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, int N, int global_col0,
        size_t spatial_tile_id0, size_t spatial_tile_id1, size_t tile_count,
        const int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    if (!use_fast_u8s8) {
        /* Exact s8s8: no vpsignb in AVX-512, use the AVX2 kernel for each tile. */
        case33_avx2_micro_gemm_xor_fused_k(a_base0, b_base, blocks_k, blocks_per_milestone,
                                           num_milestones, N, global_col0, spatial_tile_id0,
                                           tile_count, b_comp_ms, false, xor_after_milestone,
                                           tile_xor_out);
        case33_avx2_micro_gemm_xor_fused_k(a_base1, b_base, blocks_k, blocks_per_milestone,
                                           num_milestones, N, global_col0, spatial_tile_id1,
                                           tile_count, b_comp_ms, false, xor_after_milestone,
                                           tile_xor_out);
        return;
    }
    (void)blocks_per_milestone;
    (void)num_milestones;
    avx512bw_pair_fast_impl(a_base0, a_base1, b_base, blocks_k, N, global_col0,
                            spatial_tile_id0, spatial_tile_id1, tile_count, b_comp_ms,
                            xor_after_milestone, tile_xor_out);
}
