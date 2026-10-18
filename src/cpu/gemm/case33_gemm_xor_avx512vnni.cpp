#include "case33_gemm_xor_avx512vnni.hpp"
#include "case33_gemm_xor.hpp"

#include <immintrin.h>

#include <cstdlib>

#if defined(_MSC_VER)
#define CASE33_FORCEINLINE __forceinline
#else
#define CASE33_FORCEINLINE inline __attribute__((always_inline))
#endif

/* AVX512-VNNI micro-kernel: identical algorithm, data layout, milestone XOR schedule and
 * epilogue to case33_gemm_xor_avxvnni.cpp. Differences are purely in encoding:
 *  - EVEX ymm `vpdpbusd` (_mm256_dpbusd_epi32) instead of the VEX AVX-VNNI form, so the
 *    compiler has ymm16..ymm31 available and the 16 accumulators + A + 8 B broadcasts
 *    never spill (the AVX2 kernel runs out of 16 ymm regs).
 *  - Targets Zen4-class parts that expose AVX512_VNNI but not VEX AVX-VNNI.
 * The hash tile stays 8x16 int32; XOR is lane-order independent so the reduction order
 * does not affect the result. */

namespace {

constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;
constexpr int kKGroups = kKR / kRank;

static_assert(kMR == 8, "AVX512-VNNI ukernel assumes an 8-row (one ymm of 8 i32) tile");
static_assert(kNR == 2 * kColsPerGroup, "AVX512-VNNI ukernel assumes 16 columns (2 groups)");

/* u8 x s8 -> i32, 4 products summed per lane; EVEX-encoded (AVX512VL + AVX512_VNNI). */
CASE33_FORCEINLINE __m256i rank4_dpbusd(__m256i acc, __m256i ua, __m256i sb) {
    return _mm256_dpbusd_epi32(acc, ua, sb);
}

CASE33_FORCEINLINE __m256i broadcast_rank4_b(int32_t packed_b4) {
    return _mm256_set1_epi32(packed_b4);
}

CASE33_FORCEINLINE void rank4_kgroup_update8_fast(__m256i acc[kColsPerGroup], const __m256i ua,
                                                  const int32_t *bp) {
    const __m256i b0 = broadcast_rank4_b(bp[0]);
    const __m256i b1 = broadcast_rank4_b(bp[1]);
    const __m256i b2 = broadcast_rank4_b(bp[2]);
    const __m256i b3 = broadcast_rank4_b(bp[3]);
    const __m256i b4 = broadcast_rank4_b(bp[4]);
    const __m256i b5 = broadcast_rank4_b(bp[5]);
    const __m256i b6 = broadcast_rank4_b(bp[6]);
    const __m256i b7 = broadcast_rank4_b(bp[7]);

    acc[0] = rank4_dpbusd(acc[0], ua, b0);
    acc[1] = rank4_dpbusd(acc[1], ua, b1);
    acc[2] = rank4_dpbusd(acc[2], ua, b2);
    acc[3] = rank4_dpbusd(acc[3], ua, b3);
    acc[4] = rank4_dpbusd(acc[4], ua, b4);
    acc[5] = rank4_dpbusd(acc[5], ua, b5);
    acc[6] = rank4_dpbusd(acc[6], ua, b6);
    acc[7] = rank4_dpbusd(acc[7], ua, b7);
}

CASE33_FORCEINLINE void rank4_kgroup_update8_exact(__m256i acc[kColsPerGroup], const __m256i abs_a,
                                                   const __m256i va, const int32_t *bp) {
    const __m256i b0 = broadcast_rank4_b(bp[0]);
    const __m256i b1 = broadcast_rank4_b(bp[1]);
    const __m256i b2 = broadcast_rank4_b(bp[2]);
    const __m256i b3 = broadcast_rank4_b(bp[3]);
    const __m256i b4 = broadcast_rank4_b(bp[4]);
    const __m256i b5 = broadcast_rank4_b(bp[5]);
    const __m256i b6 = broadcast_rank4_b(bp[6]);
    const __m256i b7 = broadcast_rank4_b(bp[7]);

    acc[0] = rank4_dpbusd(acc[0], abs_a, _mm256_sign_epi8(b0, va));
    acc[1] = rank4_dpbusd(acc[1], abs_a, _mm256_sign_epi8(b1, va));
    acc[2] = rank4_dpbusd(acc[2], abs_a, _mm256_sign_epi8(b2, va));
    acc[3] = rank4_dpbusd(acc[3], abs_a, _mm256_sign_epi8(b3, va));
    acc[4] = rank4_dpbusd(acc[4], abs_a, _mm256_sign_epi8(b4, va));
    acc[5] = rank4_dpbusd(acc[5], abs_a, _mm256_sign_epi8(b5, va));
    acc[6] = rank4_dpbusd(acc[6], abs_a, _mm256_sign_epi8(b6, va));
    acc[7] = rank4_dpbusd(acc[7], abs_a, _mm256_sign_epi8(b7, va));
}

CASE33_FORCEINLINE uint32_t reduce_xor_epi32(__m256i v) {
    v = _mm256_xor_si256(v, _mm256_permute2x128_si256(v, v, 0x01));
    __m128i x = _mm256_castsi256_si128(v);
    x = _mm_xor_si128(x, _mm_srli_si128(x, 8));
    x = _mm_xor_si128(x, _mm_srli_si128(x, 4));
    return static_cast<uint32_t>(_mm_cvtsi128_si32(x));
}

CASE33_FORCEINLINE uint32_t xor_micro_acc(const __m256i acc0[kColsPerGroup],
                                          const __m256i acc1[kColsPerGroup]) {
    __m256i x = _mm256_setzero_si256();
    for (int c = 0; c < kColsPerGroup; ++c) {
        x = _mm256_xor_si256(x, acc0[c]);
        x = _mm256_xor_si256(x, acc1[c]);
    }
    return reduce_xor_epi32(x);
}

CASE33_FORCEINLINE void apply_b_comp_to_acc(__m256i acc0[kColsPerGroup],
                                            __m256i acc1[kColsPerGroup],
                                            const int32_t *b_comp_slice, int global_col0) {
    for (int c = 0; c < kColsPerGroup; ++c) {
        acc0[c] = _mm256_add_epi32(
                acc0[c], _mm256_set1_epi32(b_comp_slice[global_col0 + c]));
        acc1[c] = _mm256_add_epi32(
                acc1[c],
                _mm256_set1_epi32(b_comp_slice[global_col0 + kColsPerGroup + c]));
    }
}

template <typename UpdateFn>
CASE33_FORCEINLINE void avx512vnni_micro_gemm_kgroups(__m256i acc0[kColsPerGroup],
                                                      __m256i acc1[kColsPerGroup],
                                                      const int8_t *a_tile, const int8_t *b_jg0,
                                                      const int8_t *b_jg1, UpdateFn update) {
    int kg = 0;
    for (; kg + 1 < kKGroups; kg += 2) {
        const __m256i va0 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_tile + kg * 32));
        const int32_t *bp0_0 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg) * 32);
        const int32_t *bp1_0 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg) * 32);
        update(acc0, va0, bp0_0);
        update(acc1, va0, bp1_0);

        const int kg1 = kg + 1;
        const __m256i va1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_tile + kg1 * 32));
        const int32_t *bp0_1 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg1) * 32);
        const int32_t *bp1_1 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg1) * 32);
        update(acc0, va1, bp0_1);
        update(acc1, va1, bp1_1);
    }
    for (; kg < kKGroups; ++kg) {
        const __m256i va =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_tile + kg * 32));
        const int32_t *bp0 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg) * 32);
        const int32_t *bp1 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg) * 32);
        update(acc0, va, bp0);
        update(acc1, va, bp1);
    }
}

CASE33_FORCEINLINE void zero_micro_acc(__m256i acc0[kColsPerGroup],
                                       __m256i acc1[kColsPerGroup]) {
    for (int col = 0; col < kColsPerGroup; ++col) {
        acc0[col] = _mm256_setzero_si256();
        acc1[col] = _mm256_setzero_si256();
    }
}

void avx512vnni_micro_gemm_xor_fused_k_impl(const int8_t *a_base, const int8_t *b_base,
                                            int blocks_k, int blocks_per_milestone,
                                            int num_milestones, int N, int global_col0,
                                            size_t spatial_tile_id, size_t tile_count,
                                            const int32_t *b_comp_ms, bool use_fast_u8s8,
                                            bool xor_after_milestone, uint32_t *tile_xor_out) {
    __m256i acc0[kColsPerGroup];
    __m256i acc1[kColsPerGroup];
    zero_micro_acc(acc0, acc1);
    (void)blocks_per_milestone;

    int ms = 0;
    const auto milestone_epilogue = [&](const int32_t *b_comp_slice) {
        if (b_comp_slice) {
            apply_b_comp_to_acc(acc0, acc1, b_comp_slice, global_col0);
        }
        if (xor_after_milestone) {
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id] =
                    xor_micro_acc(acc0, acc1);
        }
        ++ms;
    };

    if (use_fast_u8s8) {
        const auto update_fast = [](__m256i acc[kColsPerGroup], const __m256i ua,
                                    const int32_t *bp) {
            rank4_kgroup_update8_fast(acc, ua, bp);
        };
        for (int kb = 0; kb < blocks_k; ++kb) {
            const int8_t *a_tile = a_base + static_cast<size_t>(kb) * kPanelA;
            const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
            avx512vnni_micro_gemm_kgroups(acc0, acc1, a_tile, b_tile,
                                          b_tile + static_cast<size_t>(kKGroups) * 32,
                                          update_fast);
            const int32_t *b_comp_slice =
                    b_comp_ms ? b_comp_ms + static_cast<size_t>(ms) * static_cast<size_t>(N)
                              : nullptr;
            milestone_epilogue(b_comp_slice);
        }
    } else {
        const auto update_exact = [](__m256i acc[kColsPerGroup], const __m256i va,
                                     const int32_t *bp) {
            const __m256i abs_a = _mm256_sign_epi8(va, va);
            rank4_kgroup_update8_exact(acc, abs_a, va, bp);
        };
        for (int kb = 0; kb < blocks_k; ++kb) {
            const int8_t *a_tile = a_base + static_cast<size_t>(kb) * kPanelA;
            const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
            avx512vnni_micro_gemm_kgroups(acc0, acc1, a_tile, b_tile,
                                          b_tile + static_cast<size_t>(kKGroups) * 32,
                                          update_exact);
            milestone_epilogue(nullptr);
        }
    }
    (void)num_milestones;
}

/* ---------------------------------------------------------------------------------------
 * zmm pair variant: two vertically adjacent 8x16 hash tiles (rows r..r+7 and r+8..r+15 of
 * the same 16 columns) computed together as one 16x16 register block. Each zmm accumulator
 * holds one column: lanes 0-7 = tile 0 rows, lanes 8-15 = tile 1 rows. Both tiles share every
 * B broadcast, so B loads per MAC halve; the A operand is the two tiles' 32-byte k-group rows
 * joined with vinserti64x4. The hash tiles themselves stay 8x16 (low/high zmm halves), and the
 * milestone XOR runs per KR panel exactly as in the single-tile kernel, so the int32 values
 * and XOR schedule are identical to the reference. Fast u8s8 mode only: AVX-512 has no
 * vpsignb, so the exact s8s8 path uses the ymm kernel twice. */

CASE33_FORCEINLINE __m512i load_a_pair(const int8_t *a0, const int8_t *a1) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a1));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

CASE33_FORCEINLINE void rank4_kgroup_update8_fast_zmm(__m512i acc[kColsPerGroup],
                                                      const __m512i ua, const int32_t *bp) {
    const __m512i b0 = _mm512_set1_epi32(bp[0]);
    const __m512i b1 = _mm512_set1_epi32(bp[1]);
    const __m512i b2 = _mm512_set1_epi32(bp[2]);
    const __m512i b3 = _mm512_set1_epi32(bp[3]);
    const __m512i b4 = _mm512_set1_epi32(bp[4]);
    const __m512i b5 = _mm512_set1_epi32(bp[5]);
    const __m512i b6 = _mm512_set1_epi32(bp[6]);
    const __m512i b7 = _mm512_set1_epi32(bp[7]);

    acc[0] = _mm512_dpbusd_epi32(acc[0], ua, b0);
    acc[1] = _mm512_dpbusd_epi32(acc[1], ua, b1);
    acc[2] = _mm512_dpbusd_epi32(acc[2], ua, b2);
    acc[3] = _mm512_dpbusd_epi32(acc[3], ua, b3);
    acc[4] = _mm512_dpbusd_epi32(acc[4], ua, b4);
    acc[5] = _mm512_dpbusd_epi32(acc[5], ua, b5);
    acc[6] = _mm512_dpbusd_epi32(acc[6], ua, b6);
    acc[7] = _mm512_dpbusd_epi32(acc[7], ua, b7);
}

CASE33_FORCEINLINE void zmm_pair_kgroups(__m512i acc0[kColsPerGroup], __m512i acc1[kColsPerGroup],
                                         const int8_t *a_tile0, const int8_t *a_tile1,
                                         const int8_t *b_jg0, const int8_t *b_jg1) {
    for (int kg = 0; kg < kKGroups; ++kg) {
        const __m512i ua = load_a_pair(a_tile0 + kg * 32, a_tile1 + kg * 32);
        const int32_t *bp0 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg) * 32);
        const int32_t *bp1 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg) * 32);
        rank4_kgroup_update8_fast_zmm(acc0, ua, bp0);
        rank4_kgroup_update8_fast_zmm(acc1, ua, bp1);
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

void avx512vnni_pair_fast_impl(const int8_t *a_base0, const int8_t *a_base1,
                               const int8_t *b_base, int blocks_k, int N, int global_col0,
                               size_t spatial_tile_id0, size_t spatial_tile_id1,
                               size_t tile_count, const int32_t *b_comp_ms,
                               bool xor_after_milestone, uint32_t *tile_xor_out) {
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
                         b_tile + static_cast<size_t>(kKGroups) * 32);
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

bool case33_avx512vnni_pair_enabled() {
    static int cached = -1;
    if (cached < 0) {
        /* CP_AVX512_PAIR=0 forces the ymm single-tile kernel (A/B knob). */
        const char *env = std::getenv("CP_AVX512_PAIR");
        cached = env ? (std::atoi(env) != 0 ? 1 : 0) : 1;
    }
    return cached != 0;
}

void case33_avx512vnni_micro_gemm_xor_fused_k_x2(
        const int8_t *a_base0, const int8_t *a_base1, const int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, int N, int global_col0,
        size_t spatial_tile_id0, size_t spatial_tile_id1, size_t tile_count,
        const int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    if (!use_fast_u8s8) {
        avx512vnni_micro_gemm_xor_fused_k_impl(a_base0, b_base, blocks_k, blocks_per_milestone,
                                               num_milestones, N, global_col0, spatial_tile_id0,
                                               tile_count, b_comp_ms, false,
                                               xor_after_milestone, tile_xor_out);
        avx512vnni_micro_gemm_xor_fused_k_impl(a_base1, b_base, blocks_k, blocks_per_milestone,
                                               num_milestones, N, global_col0, spatial_tile_id1,
                                               tile_count, b_comp_ms, false,
                                               xor_after_milestone, tile_xor_out);
        return;
    }
    (void)blocks_per_milestone;
    (void)num_milestones;
    avx512vnni_pair_fast_impl(a_base0, a_base1, b_base, blocks_k, N, global_col0,
                              spatial_tile_id0, spatial_tile_id1, tile_count, b_comp_ms,
                              xor_after_milestone, tile_xor_out);
}

void case33_avx512vnni_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, int N, int global_col0, size_t spatial_tile_id, size_t tile_count,
        const int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    avx512vnni_micro_gemm_xor_fused_k_impl(a_base, b_base, blocks_k, blocks_per_milestone,
                                           num_milestones, N, global_col0, spatial_tile_id,
                                           tile_count, b_comp_ms, use_fast_u8s8,
                                           xor_after_milestone, tile_xor_out);
}
