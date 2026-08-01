#include "case33_gemm_xor_avx2.hpp"
#include "case33_gemm_xor.hpp"

#include <immintrin.h>

#if defined(_MSC_VER)
#define CASE33_FORCEINLINE __forceinline
#else
#define CASE33_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace {

constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kTileRows = Case33GemmXor::kTileRows;
constexpr int kTileCols = Case33GemmXor::kTileCols;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;

CASE33_FORCEINLINE __m256i rank4_maddubs(__m256i acc, __m256i ua, __m256i sb,
                                         __m256i ones16) {
    const __m256i pair16 = _mm256_maddubs_epi16(ua, sb);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pair16, ones16));
}

CASE33_FORCEINLINE __m256i broadcast_rank4_b(int32_t packed_b4) {
    return _mm256_broadcastd_epi32(_mm_cvtsi32_si128(packed_b4));
}

CASE33_FORCEINLINE void rank4_kgroup_update8_fast(__m256i acc[kColsPerGroup], const __m256i ua,
                                                  const int32_t *bp, __m256i ones16) {
    const __m256i b0 = broadcast_rank4_b(bp[0]);
    const __m256i b1 = broadcast_rank4_b(bp[1]);
    const __m256i b2 = broadcast_rank4_b(bp[2]);
    const __m256i b3 = broadcast_rank4_b(bp[3]);
    const __m256i b4 = broadcast_rank4_b(bp[4]);
    const __m256i b5 = broadcast_rank4_b(bp[5]);
    const __m256i b6 = broadcast_rank4_b(bp[6]);
    const __m256i b7 = broadcast_rank4_b(bp[7]);

    acc[0] = rank4_maddubs(acc[0], ua, b0, ones16);
    acc[1] = rank4_maddubs(acc[1], ua, b1, ones16);
    acc[2] = rank4_maddubs(acc[2], ua, b2, ones16);
    acc[3] = rank4_maddubs(acc[3], ua, b3, ones16);
    acc[4] = rank4_maddubs(acc[4], ua, b4, ones16);
    acc[5] = rank4_maddubs(acc[5], ua, b5, ones16);
    acc[6] = rank4_maddubs(acc[6], ua, b6, ones16);
    acc[7] = rank4_maddubs(acc[7], ua, b7, ones16);
}

CASE33_FORCEINLINE void rank4_kgroup_update8_exact(__m256i acc[kColsPerGroup], const __m256i abs_a,
                                                   const __m256i va, const int32_t *bp,
                                                   __m256i ones16) {
    const __m256i b0 = broadcast_rank4_b(bp[0]);
    const __m256i b1 = broadcast_rank4_b(bp[1]);
    const __m256i b2 = broadcast_rank4_b(bp[2]);
    const __m256i b3 = broadcast_rank4_b(bp[3]);
    const __m256i b4 = broadcast_rank4_b(bp[4]);
    const __m256i b5 = broadcast_rank4_b(bp[5]);
    const __m256i b6 = broadcast_rank4_b(bp[6]);
    const __m256i b7 = broadcast_rank4_b(bp[7]);

    acc[0] = rank4_maddubs(acc[0], abs_a, _mm256_sign_epi8(b0, va), ones16);
    acc[1] = rank4_maddubs(acc[1], abs_a, _mm256_sign_epi8(b1, va), ones16);
    acc[2] = rank4_maddubs(acc[2], abs_a, _mm256_sign_epi8(b2, va), ones16);
    acc[3] = rank4_maddubs(acc[3], abs_a, _mm256_sign_epi8(b3, va), ones16);
    acc[4] = rank4_maddubs(acc[4], abs_a, _mm256_sign_epi8(b4, va), ones16);
    acc[5] = rank4_maddubs(acc[5], abs_a, _mm256_sign_epi8(b5, va), ones16);
    acc[6] = rank4_maddubs(acc[6], abs_a, _mm256_sign_epi8(b6, va), ones16);
    acc[7] = rank4_maddubs(acc[7], abs_a, _mm256_sign_epi8(b7, va), ones16);
}

CASE33_FORCEINLINE __m256i accum_col_group_to_tile(const __m256i acc[kColsPerGroup],
                                                   int32_t *tile_c, int local_row0,
                                                   int global_col0, int local_col0,
                                                   const int32_t *b_comp_ms,
                                                   bool xor_epilogue) {
    __m256i xor_vec = _mm256_setzero_si256();
    for (int c = 0; c < kColsPerGroup; ++c) {
        __m256i v = acc[c];
        if (b_comp_ms) {
            v = _mm256_add_epi32(v, _mm256_set1_epi32(b_comp_ms[global_col0 + c]));
        }
        int32_t *dst = tile_c + static_cast<size_t>(local_col0 + c) * kTileRows + local_row0;
        v = _mm256_add_epi32(
                v, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst)));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), v);
        if (xor_epilogue) {
            xor_vec = _mm256_xor_si256(xor_vec, v);
        }
    }
    return xor_vec;
}

CASE33_FORCEINLINE uint32_t reduce_xor_epi32(__m256i v) {
    v = _mm256_xor_si256(v, _mm256_permute2x128_si256(v, v, 0x01));
    __m128i x = _mm256_castsi256_si128(v);
    x = _mm_xor_si128(x, _mm_srli_si128(x, 8));
    x = _mm_xor_si128(x, _mm_srli_si128(x, 4));
    return static_cast<uint32_t>(_mm_cvtsi128_si32(x));
}

template <typename UpdateFn>
CASE33_FORCEINLINE void avx2_micro_gemm_kgroups(__m256i acc0[kColsPerGroup],
                                                __m256i acc1[kColsPerGroup],
                                                const int8_t *a_tile, const int8_t *b_jg0,
                                                const int8_t *b_jg1, UpdateFn update) {
    constexpr int kKGroups = kKR / kRank;
    const __m256i ones16 = _mm256_set1_epi16(1);
    int kg = 0;
    for (; kg + 1 < kKGroups; kg += 2) {
        const __m256i va0 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_tile + kg * 32));
        const int32_t *bp0_0 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg) * 32);
        const int32_t *bp1_0 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg) * 32);
        update(acc0, va0, bp0_0, ones16);
        update(acc1, va0, bp1_0, ones16);

        const int kg1 = kg + 1;
        const __m256i va1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_tile + kg1 * 32));
        const int32_t *bp0_1 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg1) * 32);
        const int32_t *bp1_1 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg1) * 32);
        update(acc0, va1, bp0_1, ones16);
        update(acc1, va1, bp1_1, ones16);
    }
    for (; kg < kKGroups; ++kg) {
        const __m256i va =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_tile + kg * 32));
        const int32_t *bp0 =
                reinterpret_cast<const int32_t *>(b_jg0 + static_cast<size_t>(kg) * 32);
        const int32_t *bp1 =
                reinterpret_cast<const int32_t *>(b_jg1 + static_cast<size_t>(kg) * 32);
        update(acc0, va, bp0, ones16);
        update(acc1, va, bp1, ones16);
    }
}

CASE33_FORCEINLINE void zero_micro_acc(__m256i acc0[kColsPerGroup],
                                       __m256i acc1[kColsPerGroup]) {
    for (int col = 0; col < kColsPerGroup; ++col) {
        acc0[col] = _mm256_setzero_si256();
        acc1[col] = _mm256_setzero_si256();
    }
}

void avx2_micro_gemm_xor_fused_k_impl(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                                      int blocks_per_milestone, int num_milestones, int N,
                                      int global_col0, size_t spatial_tile_id, size_t tile_count,
                                      const int32_t *b_comp_ms, bool use_fast_u8s8,
                                      bool xor_after_milestone, uint32_t *tile_xor_out) {
    constexpr int kKGroups = kKR / kRank;

    int32_t tile_c[kTileRows * kTileCols] = {};
    __m256i acc0[kColsPerGroup];
    __m256i acc1[kColsPerGroup];
    zero_micro_acc(acc0, acc1);

    int ms = 0;
    const auto milestone_epilogue = [&](const int32_t *b_comp_slice) {
        const __m256i xor0 = accum_col_group_to_tile(
                acc0, tile_c, 0, global_col0, 0, b_comp_slice, xor_after_milestone);
        const __m256i xor1 = accum_col_group_to_tile(
                acc1, tile_c, 0, global_col0 + kColsPerGroup, kColsPerGroup,
                b_comp_slice, xor_after_milestone);
        if (xor_after_milestone) {
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id] =
                    reduce_xor_epi32(_mm256_xor_si256(xor0, xor1));
        }
        zero_micro_acc(acc0, acc1);
        ++ms;
    };

    if (use_fast_u8s8) {
        const auto update_fast = [](__m256i acc[kColsPerGroup], const __m256i ua,
                                    const int32_t *bp, const __m256i ones16) {
            rank4_kgroup_update8_fast(acc, ua, bp, ones16);
        };
        for (int kb = 0; kb < blocks_k; ++kb) {
            const int8_t *a_tile = a_base + static_cast<size_t>(kb) * kPanelA;
            const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
            avx2_micro_gemm_kgroups(acc0, acc1, a_tile, b_tile,
                                    b_tile + static_cast<size_t>(kKGroups) * 32,
                                    update_fast);
            if ((kb + 1) % blocks_per_milestone == 0) {
                const int32_t *b_comp_slice =
                        b_comp_ms ? b_comp_ms + static_cast<size_t>(ms) * static_cast<size_t>(N)
                                  : nullptr;
                milestone_epilogue(b_comp_slice);
            }
        }
    } else {
        const auto update_exact = [](__m256i acc[kColsPerGroup], const __m256i va,
                                     const int32_t *bp, const __m256i ones16) {
            const __m256i abs_a = _mm256_sign_epi8(va, va);
            rank4_kgroup_update8_exact(acc, abs_a, va, bp, ones16);
        };
        for (int kb = 0; kb < blocks_k; ++kb) {
            const int8_t *a_tile = a_base + static_cast<size_t>(kb) * kPanelA;
            const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
            avx2_micro_gemm_kgroups(acc0, acc1, a_tile, b_tile,
                                    b_tile + static_cast<size_t>(kKGroups) * 32,
                                    update_exact);
            if ((kb + 1) % blocks_per_milestone == 0) {
                milestone_epilogue(nullptr);
            }
        }
    }
    (void)num_milestones;
}

} // namespace

void case33_avx2_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, int N, int global_col0, size_t spatial_tile_id, size_t tile_count,
        const int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    avx2_micro_gemm_xor_fused_k_impl(a_base, b_base, blocks_k, blocks_per_milestone,
                                     num_milestones, N, global_col0, spatial_tile_id,
                                     tile_count, b_comp_ms, use_fast_u8s8,
                                     xor_after_milestone, tile_xor_out);
}
