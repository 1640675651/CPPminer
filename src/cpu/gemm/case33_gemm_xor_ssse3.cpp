#include "case33_gemm_xor_ssse3.hpp"

#include <emmintrin.h>
#include <tmmintrin.h>

#if defined(_MSC_VER)
#define CASE33_FORCEINLINE __forceinline
#define CASE33_SSSE3
#else
#define CASE33_FORCEINLINE inline __attribute__((always_inline, target("ssse3")))
#define CASE33_SSSE3 __attribute__((target("ssse3")))
#endif

namespace {

constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;
constexpr int kKGroups = kKR / kRank;

uint32_t xor_tile(const int32_t *tile_c) {
    uint32_t x = 0u;
    for (int i = 0; i < kMR * kNR; ++i) x ^= static_cast<uint32_t>(tile_c[i]);
    return x;
}

// SSSE3 path: same 8x16 semantic tile / pack layout as AVX2; selectable register tiles.
CASE33_FORCEINLINE __m128i sse_rank4_maddubs(__m128i acc, __m128i ua, __m128i sb,
                                             __m128i ones16) {
    const __m128i pair16 = _mm_maddubs_epi16(ua, sb);
    return _mm_add_epi32(acc, _mm_madd_epi16(pair16, ones16));
}

CASE33_FORCEINLINE __m128i sse_broadcast_rank4_b(int32_t packed_b4) {
    return _mm_set1_epi32(packed_b4);
}

CASE33_FORCEINLINE void sse_zero_acc_n(__m128i *acc, int n) {
    for (int c = 0; c < n; ++c) {
        acc[c] = _mm_setzero_si128();
    }
}

CASE33_FORCEINLINE void sse_add_acc_to_vals(__m128i *acc, int n_cols, int32_t *vals,
                                           int row0, int col0) {
    for (int c = 0; c < n_cols; ++c) {
        alignas(16) int32_t tmp[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(tmp), acc[c]);
        const int j = col0 + c;
        vals[j * kMR + row0 + 0] += tmp[0];
        vals[j * kMR + row0 + 1] += tmp[1];
        vals[j * kMR + row0 + 2] += tmp[2];
        vals[j * kMR + row0 + 3] += tmp[3];
    }
}

// 4x8: two row-halves x two col-groups (baseline).
CASE33_SSSE3 void sse_panel_accum_fast_4x8(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals, int kg_begin, int kg_end) {
    const __m128i ones16 = _mm_set1_epi16(1);
    for (int row0 = 0; row0 < kMR; row0 += 4) {
        for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
            __m128i acc[kColsPerGroup];
            sse_zero_acc_n(acc, kColsPerGroup);
            const int8_t *b_jg =
                    b_tile + static_cast<size_t>(jg) * static_cast<size_t>(kKGroups) * 32;
            for (int kg = kg_begin; kg < kg_end; ++kg) {
                const __m128i ua = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                        a_tile + static_cast<size_t>(kg) * 32 +
                        static_cast<size_t>(row0) * kRank));
                const int32_t *bp = reinterpret_cast<const int32_t *>(
                        b_jg + static_cast<size_t>(kg) * 32);
                for (int c = 0; c < kColsPerGroup; ++c) {
                    acc[c] = sse_rank4_maddubs(acc[c], ua, sse_broadcast_rank4_b(bp[c]),
                                               ones16);
                }
            }
            sse_add_acc_to_vals(acc, kColsPerGroup, vals, row0, jg * kColsPerGroup);
        }
    }
}

CASE33_SSSE3 void sse_panel_accum_exact_4x8(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals, int kg_begin, int kg_end) {
    const __m128i ones16 = _mm_set1_epi16(1);
    for (int row0 = 0; row0 < kMR; row0 += 4) {
        for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
            __m128i acc[kColsPerGroup];
            sse_zero_acc_n(acc, kColsPerGroup);
            const int8_t *b_jg =
                    b_tile + static_cast<size_t>(jg) * static_cast<size_t>(kKGroups) * 32;
            for (int kg = kg_begin; kg < kg_end; ++kg) {
                const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                        a_tile + static_cast<size_t>(kg) * 32 +
                        static_cast<size_t>(row0) * kRank));
                const __m128i abs_a = _mm_sign_epi8(va, va);
                const int32_t *bp = reinterpret_cast<const int32_t *>(
                        b_jg + static_cast<size_t>(kg) * 32);
                for (int c = 0; c < kColsPerGroup; ++c) {
                    const __m128i sb =
                            _mm_sign_epi8(sse_broadcast_rank4_b(bp[c]), va);
                    acc[c] = sse_rank4_maddubs(acc[c], abs_a, sb, ones16);
                }
            }
            sse_add_acc_to_vals(acc, kColsPerGroup, vals, row0, jg * kColsPerGroup);
        }
    }
}

// 8x8: keep both 4-row halves live across K for one col-group (fewer vals stores).
CASE33_SSSE3 void sse_panel_accum_fast_8x8(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals, int kg_begin, int kg_end) {
    const __m128i ones16 = _mm_set1_epi16(1);
    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
        __m128i acc0[kColsPerGroup];
        __m128i acc1[kColsPerGroup];
        sse_zero_acc_n(acc0, kColsPerGroup);
        sse_zero_acc_n(acc1, kColsPerGroup);
        const int8_t *b_jg =
                b_tile + static_cast<size_t>(jg) * static_cast<size_t>(kKGroups) * 32;
        for (int kg = kg_begin; kg < kg_end; ++kg) {
            const __m128i ua0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    a_tile + static_cast<size_t>(kg) * 32));
            const __m128i ua1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    a_tile + static_cast<size_t>(kg) * 32 + 16));
            const int32_t *bp = reinterpret_cast<const int32_t *>(
                    b_jg + static_cast<size_t>(kg) * 32);
            for (int c = 0; c < kColsPerGroup; ++c) {
                const __m128i sb = sse_broadcast_rank4_b(bp[c]);
                acc0[c] = sse_rank4_maddubs(acc0[c], ua0, sb, ones16);
                acc1[c] = sse_rank4_maddubs(acc1[c], ua1, sb, ones16);
            }
        }
        sse_add_acc_to_vals(acc0, kColsPerGroup, vals, 0, jg * kColsPerGroup);
        sse_add_acc_to_vals(acc1, kColsPerGroup, vals, 4, jg * kColsPerGroup);
    }
}

CASE33_SSSE3 void sse_panel_accum_exact_8x8(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals, int kg_begin, int kg_end) {
    const __m128i ones16 = _mm_set1_epi16(1);
    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
        __m128i acc0[kColsPerGroup];
        __m128i acc1[kColsPerGroup];
        sse_zero_acc_n(acc0, kColsPerGroup);
        sse_zero_acc_n(acc1, kColsPerGroup);
        const int8_t *b_jg =
                b_tile + static_cast<size_t>(jg) * static_cast<size_t>(kKGroups) * 32;
        for (int kg = kg_begin; kg < kg_end; ++kg) {
            const __m128i va0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    a_tile + static_cast<size_t>(kg) * 32));
            const __m128i va1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    a_tile + static_cast<size_t>(kg) * 32 + 16));
            const __m128i abs0 = _mm_sign_epi8(va0, va0);
            const __m128i abs1 = _mm_sign_epi8(va1, va1);
            const int32_t *bp = reinterpret_cast<const int32_t *>(
                    b_jg + static_cast<size_t>(kg) * 32);
            for (int c = 0; c < kColsPerGroup; ++c) {
                const __m128i bcast = sse_broadcast_rank4_b(bp[c]);
                acc0[c] = sse_rank4_maddubs(acc0[c], abs0, _mm_sign_epi8(bcast, va0),
                                            ones16);
                acc1[c] = sse_rank4_maddubs(acc1[c], abs1, _mm_sign_epi8(bcast, va1),
                                            ones16);
            }
        }
        sse_add_acc_to_vals(acc0, kColsPerGroup, vals, 0, jg * kColsPerGroup);
        sse_add_acc_to_vals(acc1, kColsPerGroup, vals, 4, jg * kColsPerGroup);
    }
}

// 4x16: one A load reused across all 16 B columns per row-half.
CASE33_SSSE3 void sse_panel_accum_fast_4x16(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals, int kg_begin, int kg_end) {
    const __m128i ones16 = _mm_set1_epi16(1);
    const int8_t *b_jg1 = b_tile + static_cast<size_t>(kKGroups) * 32;
    for (int row0 = 0; row0 < kMR; row0 += 4) {
        __m128i acc[kNR];
        sse_zero_acc_n(acc, kNR);
        for (int kg = kg_begin; kg < kg_end; ++kg) {
            const __m128i ua = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    a_tile + static_cast<size_t>(kg) * 32 +
                    static_cast<size_t>(row0) * kRank));
            const int32_t *bp0 = reinterpret_cast<const int32_t *>(
                    b_tile + static_cast<size_t>(kg) * 32);
            const int32_t *bp1 = reinterpret_cast<const int32_t *>(
                    b_jg1 + static_cast<size_t>(kg) * 32);
            for (int c = 0; c < kColsPerGroup; ++c) {
                acc[c] = sse_rank4_maddubs(acc[c], ua, sse_broadcast_rank4_b(bp0[c]),
                                           ones16);
                acc[c + kColsPerGroup] = sse_rank4_maddubs(
                        acc[c + kColsPerGroup], ua, sse_broadcast_rank4_b(bp1[c]), ones16);
            }
        }
        sse_add_acc_to_vals(acc, kNR, vals, row0, 0);
    }
}

CASE33_SSSE3 void sse_panel_accum_exact_4x16(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals, int kg_begin, int kg_end) {
    const __m128i ones16 = _mm_set1_epi16(1);
    const int8_t *b_jg1 = b_tile + static_cast<size_t>(kKGroups) * 32;
    for (int row0 = 0; row0 < kMR; row0 += 4) {
        __m128i acc[kNR];
        sse_zero_acc_n(acc, kNR);
        for (int kg = kg_begin; kg < kg_end; ++kg) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    a_tile + static_cast<size_t>(kg) * 32 +
                    static_cast<size_t>(row0) * kRank));
            const __m128i abs_a = _mm_sign_epi8(va, va);
            const int32_t *bp0 = reinterpret_cast<const int32_t *>(
                    b_tile + static_cast<size_t>(kg) * 32);
            const int32_t *bp1 = reinterpret_cast<const int32_t *>(
                    b_jg1 + static_cast<size_t>(kg) * 32);
            for (int c = 0; c < kColsPerGroup; ++c) {
                acc[c] = sse_rank4_maddubs(acc[c], abs_a,
                                           _mm_sign_epi8(sse_broadcast_rank4_b(bp0[c]), va),
                                           ones16);
                acc[c + kColsPerGroup] = sse_rank4_maddubs(
                        acc[c + kColsPerGroup], abs_a,
                        _mm_sign_epi8(sse_broadcast_rank4_b(bp1[c]), va), ones16);
            }
        }
        sse_add_acc_to_vals(acc, kNR, vals, row0, 0);
    }
}

using SsePanelAccumFn = void (*)(const int8_t *, const int8_t *, int32_t *, int, int);

SsePanelAccumFn sse_panel_fn(Case33SseTile tile, bool use_fast_u8s8) {
    switch (tile) {
    case Case33SseTile::R8C8:
        return use_fast_u8s8 ? sse_panel_accum_fast_8x8 : sse_panel_accum_exact_8x8;
    case Case33SseTile::R4C16:
        return use_fast_u8s8 ? sse_panel_accum_fast_4x16 : sse_panel_accum_exact_4x16;
    case Case33SseTile::R4C8:
    default:
        return use_fast_u8s8 ? sse_panel_accum_fast_4x8 : sse_panel_accum_exact_4x8;
    }
}

const char *sse_tile_name(Case33SseTile tile) {
    switch (tile) {
    case Case33SseTile::R8C8:
        return "8x8regs";
    case Case33SseTile::R4C16:
        return "4x16regs";
    case Case33SseTile::R4C8:
    default:
        return "4x8regs";
    }
}

CASE33_SSSE3 void sse_micro_gemm_xor_fused_k(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                                int blocks_per_milestone, int num_milestones, int N,
                                int global_col0, size_t spatial_tile_id, size_t tile_count,
                                const int32_t *b_comp_ms, bool use_fast_u8s8,
                                Case33SseTile sse_tile, bool xor_after_milestone,
                                uint32_t *tile_xor_out) {
    /* Cumulative C tile. One KR panel == one milestone. */
    int32_t vals[kNR * kMR] = {};
    int ms = 0;
    const SsePanelAccumFn panel = sse_panel_fn(sse_tile, use_fast_u8s8);
    (void)blocks_per_milestone;

    for (int kb = 0; kb < blocks_k; ++kb) {
        panel(a_base + static_cast<size_t>(kb) * kPanelA,
              b_base + static_cast<size_t>(kb) * kPanelB, vals, 0, kKGroups);

        if (use_fast_u8s8 && b_comp_ms) {
            const int32_t *b_comp_slice =
                    b_comp_ms + static_cast<size_t>(ms) * static_cast<size_t>(N);
            for (int j = 0; j < kNR; ++j) {
                const int32_t comp = b_comp_slice[global_col0 + j];
                for (int i = 0; i < kMR; ++i) {
                    vals[j * kMR + i] += comp;
                }
            }
        }
        if (xor_after_milestone) {
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id] =
                    xor_tile(vals);
        }
        ++ms;
    }
    (void)num_milestones;
}

} // namespace

void case33_ssse3_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, int n, int global_col0,
        size_t spatial_tile_id, size_t tile_count, const int32_t *b_comp_ms,
        bool use_fast_u8s8, Case33SseTile sse_tile, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    sse_micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone,
            num_milestones, n, global_col0, spatial_tile_id, tile_count, b_comp_ms,
            use_fast_u8s8, sse_tile, xor_after_milestone, tile_xor_out);
}
