#include "case33_gemm_xor.hpp"

#include "cp_job_ctrl.h"
#include "cp_noise.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>
#include <functional>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kTileRows = Case33GemmXor::kTileRows;
constexpr int kTileCols = Case33GemmXor::kTileCols;
constexpr int kMacroM = Case33GemmXor::kMacroM;
constexpr int kMacroN = Case33GemmXor::kMacroN;
constexpr int kMicroPerMacroM = kMacroM / kMR;
constexpr int kMicroPerMacroN = kMacroN / kNR;
constexpr int kNumMilestones = Case33GemmXor::kNumMilestones;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;

bool cpu_has_avx2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

int detect_thread_count() {
#if defined(_OPENMP)
    const int n = omp_get_max_threads();
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

void pack_a_panel(const int8_t *a, int row0, int k_base, int K, int8_t *a_tile,
                  bool offset_a128) {
    constexpr int kRowsPerVector = 8;
    for (int rg = 0; rg < kMR / kRowsPerVector; ++rg) {
        for (int kg = 0; kg < kKR / kRank; ++kg) {
            int8_t *dst = a_tile + (rg * (kKR / kRank) + kg) * 32;
            for (int r = 0; r < kRowsPerVector; ++r) {
                for (int ko = 0; ko < kRank; ++ko) {
                    int8_t v =
                            a[static_cast<size_t>(row0 + rg * kRowsPerVector + r) *
                                      static_cast<size_t>(K) +
                              static_cast<size_t>(k_base + kg * kRank + ko)];
                    if (offset_a128) {
                        v = static_cast<int8_t>(static_cast<uint8_t>(v) + 128u);
                    }
                    dst[r * kRank + ko] = v;
                }
            }
        }
    }
}

void pack_b_panel(const int8_t *b, int col0, int k_base, int K, int8_t *b_tile) {
    constexpr int kKGroups = kKR / kRank;
    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
        for (int kg = 0; kg < kKGroups; ++kg) {
            int8_t *dst = b_tile + (jg * kKGroups + kg) * 32;
            for (int c = 0; c < kColsPerGroup; ++c) {
                for (int ko = 0; ko < kRank; ++ko) {
                    dst[c * kRank + ko] =
                            b[static_cast<size_t>(k_base + kg * kRank + ko) +
                              static_cast<size_t>(col0 + jg * kColsPerGroup + c) *
                                      static_cast<size_t>(K)];
                }
            }
        }
    }
}

void prepack_a_all(const int8_t *a, int M, int K, int blocks_k, bool offset_a128,
                   std::vector<int8_t> *out) {
    const int tile_rows = M / kMR;
    out->assign(static_cast<size_t>(tile_rows) * static_cast<size_t>(blocks_k) *
                        static_cast<size_t>(kPanelA),
                0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int tr = 0; tr < tile_rows; ++tr) {
        for (int kb = 0; kb < blocks_k; ++kb) {
            const size_t idx =
                    (static_cast<size_t>(tr) * static_cast<size_t>(blocks_k) +
                     static_cast<size_t>(kb)) *
                    static_cast<size_t>(kPanelA);
            pack_a_panel(a, tr * kMR, kb * kKR, K, out->data() + idx, offset_a128);
        }
    }
}

void prepack_b_all(const int8_t *b, int N, int K, int blocks_k, int tile_cols,
                   std::vector<int8_t> *out) {
    out->assign(static_cast<size_t>(tile_cols) * static_cast<size_t>(blocks_k) *
                        static_cast<size_t>(kPanelB),
                0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int tc = 0; tc < tile_cols; ++tc) {
        for (int kb = 0; kb < blocks_k; ++kb) {
            const size_t idx =
                    (static_cast<size_t>(tc) * static_cast<size_t>(blocks_k) +
                     static_cast<size_t>(kb)) *
                    static_cast<size_t>(kPanelB);
            pack_b_panel(b, tc * kNR, kb * kKR, K, out->data() + idx);
        }
    }
}

const char *prepack_mode_tag(Case33PrepackMode mode) {
    switch (mode) {
    case Case33PrepackMode::ReuseScanBuf:
        return " reuse-scan";
    case Case33PrepackMode::Fused:
        return " fused-prepack";
    default:
        return "";
    }
}

void compute_b_compensation_milestones(const int8_t *b, int K, int N, int num_milestones,
                                       int milestone_k, std::vector<int32_t> *out) {
    out->assign(static_cast<size_t>(num_milestones) * static_cast<size_t>(N), 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int ms = 0; ms < num_milestones; ++ms) {
        const int k0 = ms * milestone_k;
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = k0; k < k0 + milestone_k; ++k) {
                sum += static_cast<int32_t>(
                        b[static_cast<size_t>(k) + static_cast<size_t>(j) * K]);
            }
            (*out)[static_cast<size_t>(ms) * static_cast<size_t>(N) +
                   static_cast<size_t>(j)] = -128 * sum;
        }
    }
}

uint32_t xor_tile(const int32_t *tile_c) {
    uint32_t x = 0u;
    for (int i = 0; i < kTileRows * kTileCols; ++i) {
        x ^= static_cast<uint32_t>(tile_c[i]);
    }
    return x;
}

void scalar_panel_accum(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals) {
    constexpr int kKGroups = kKR / kRank;
    for (int j = 0; j < kNR; ++j) {
        for (int i = 0; i < kMR; ++i) {
            for (int kk = 0; kk < kKR; ++kk) {
                const int kg = kk / kRank;
                const int ko = kk % kRank;
                const size_t a_idx =
                        static_cast<size_t>(kg) * 32 + static_cast<size_t>(i * kRank + ko);
                vals[j * kMR + i] +=
                        static_cast<int32_t>(a_tile[a_idx]) *
                        static_cast<int32_t>(
                                b_tile[(static_cast<size_t>(j / kColsPerGroup) *
                                                static_cast<size_t>(kKGroups) +
                                        static_cast<size_t>(kg)) *
                                               32 +
                                       static_cast<size_t>(j % kColsPerGroup) * 4 +
                                       static_cast<size_t>(ko)]);
            }
        }
    }
}

void scalar_micro_gemm_xor_fused_k(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                                   int blocks_per_milestone, int num_milestones, int N,
                                   int global_col0, size_t spatial_tile_id, size_t tile_count,
                                   const int32_t *b_comp_ms, bool use_fast_u8s8,
                                   bool xor_after_milestone, uint32_t *tile_xor_out) {
    int32_t tile_c[kTileRows * kTileCols] = {};
    int32_t vals[kNR * kMR] = {};
    int ms = 0;

    for (int kb = 0; kb < blocks_k; ++kb) {
        scalar_panel_accum(a_base + static_cast<size_t>(kb) * kPanelA,
                           b_base + static_cast<size_t>(kb) * kPanelB, vals);

        if ((kb + 1) % blocks_per_milestone != 0) {
            continue;
        }

        const int32_t *b_comp_slice =
                use_fast_u8s8 ? b_comp_ms + static_cast<size_t>(ms) * static_cast<size_t>(N)
                              : nullptr;
        for (int j = 0; j < kNR; ++j) {
            const int32_t comp = b_comp_slice ? b_comp_slice[global_col0 + j] : 0;
            for (int i = 0; i < kMR; ++i) {
                tile_c[static_cast<size_t>(j) * kTileRows + i] +=
                        vals[j * kMR + i] + comp;
            }
        }
        if (xor_after_milestone) {
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id] =
                    xor_tile(tile_c);
        }
        std::memset(vals, 0, sizeof(vals));
        ++ms;
    }
    (void)num_milestones;
}

#if defined(__AVX2__)

#if defined(_MSC_VER)
#define CASE33_FORCEINLINE __forceinline
#else
#define CASE33_FORCEINLINE inline __attribute__((always_inline))
#endif

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

void avx2_micro_gemm_xor_fused_k(const int8_t *a_base, const int8_t *b_base, int blocks_k,
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
#undef CASE33_FORCEINLINE
#endif

void micro_gemm_xor_fused_k(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                            int blocks_per_milestone, int num_milestones, int N,
                            int global_col0, size_t spatial_tile_id, size_t tile_count,
                            const int32_t *b_comp_ms, bool use_fast_u8s8, bool use_avx2,
                            bool xor_after_milestone, uint32_t *tile_xor_out) {
#if defined(__AVX2__)
    if (use_avx2) {
        avx2_micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone,
                                    num_milestones, N, global_col0, spatial_tile_id,
                                    tile_count, b_comp_ms, use_fast_u8s8,
                                    xor_after_milestone, tile_xor_out);
        return;
    }
#else
    (void)use_avx2;
#endif
    scalar_micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone,
                                  num_milestones, N, global_col0, spatial_tile_id,
                                  tile_count, b_comp_ms, use_fast_u8s8, xor_after_milestone,
                                  tile_xor_out);
}

void micro_gemm_xor_milestones(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                               int blocks_per_milestone, int num_milestones, int N,
                               int global_col0, size_t spatial_tile_id, size_t tile_count,
                               const int32_t *b_comp_ms, bool use_fast_u8s8, bool use_avx2,
                               bool xor_after_milestone, uint32_t *tile_xor_out) {
    micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone, num_milestones,
                           N, global_col0, spatial_tile_id, tile_count, b_comp_ms,
                           use_fast_u8s8, use_avx2, xor_after_milestone, tile_xor_out);
}

// Case 3.2 macro schedule: 2D OpenMP over macro blocks, tc outer / tr inner (B reuse).
void run_hardcoded_macro_xor(const int8_t *a_pre, const int8_t *b_pre, int N, int blocks_k,
                             int blocks_per_milestone, int num_milestones, int macro_rows,
                             int macro_cols, int tile_cols, size_t tile_count,
                             const int32_t *b_comp_ms, bool use_fast_u8s8, bool use_avx2,
                             bool xor_after_milestone, std::vector<uint32_t> *out) {
    const size_t a_tile_stride =
            static_cast<size_t>(blocks_k) * static_cast<size_t>(kPanelA);
    const size_t b_tile_stride =
            static_cast<size_t>(blocks_k) * static_cast<size_t>(kPanelB);
    const int macro_blocks = macro_cols * macro_rows;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int mb = 0; mb < macro_blocks; ++mb) {
        const int jm = mb / macro_rows;
        const int im = mb % macro_rows;
        const int col0 = jm * kMacroN;
        const int tc0 = jm * kMicroPerMacroN;
        const int row0 = im * kMacroM;
        const int tr0 = im * kMicroPerMacroM;

        for (int tc = 0; tc < kMicroPerMacroN; ++tc) {
            const int tc_global = tc0 + tc;
            const int micro_col0 = col0 + tc * kNR;
            const int8_t *b_base =
                    b_pre + static_cast<size_t>(tc_global) * b_tile_stride;

            for (int tr = 0; tr < kMicroPerMacroM; ++tr) {
                const int tr_global = tr0 + tr;
                const int micro_row0 = row0 + tr * kMR;
                const int8_t *a_base =
                        a_pre + static_cast<size_t>(tr_global) * a_tile_stride;
                const size_t spatial_tile_id =
                        static_cast<size_t>(tr_global) * static_cast<size_t>(tile_cols) +
                        static_cast<size_t>(tc_global);

                micro_gemm_xor_milestones(
                        a_base, b_base, blocks_k, blocks_per_milestone, num_milestones,
                        N, micro_col0, spatial_tile_id, tile_count, b_comp_ms,
                        use_fast_u8s8, use_avx2, xor_after_milestone, out->data());
            }
        }
    }
}

void run_fused_impl(const int8_t *a_pre, const int8_t *b_pre, int N, int blocks_k,
                    int blocks_per_milestone, int num_milestones, int macro_rows,
                    int macro_cols, int tile_cols, size_t tile_count,
                    const int32_t *b_comp_ms, bool use_fast_u8s8, bool use_avx2,
                    bool xor_after_milestone, std::vector<uint32_t> *out) {
    out->assign(static_cast<size_t>(num_milestones) * tile_count, 0u);
    run_hardcoded_macro_xor(a_pre, b_pre, N, blocks_k, blocks_per_milestone, num_milestones,
                            macro_rows, macro_cols, tile_cols, tile_count, b_comp_ms,
                            use_fast_u8s8, use_avx2, xor_after_milestone, out);
}

bool run_online_tile_scan(
        const int8_t *a_pre, const int8_t *b_pre, int N, int blocks_k,
        int blocks_per_milestone, int num_milestones, int macro_rows, int macro_cols,
        const int32_t *b_comp_ms, bool use_fast_u8s8, bool use_avx2,
        const std::function<bool(const uint32_t *milestone_xor, int t_rows, int t_cols)>
                &on_tile,
        const std::function<bool()> &should_cancel) {
    const size_t a_tile_stride =
            static_cast<size_t>(blocks_k) * static_cast<size_t>(kPanelA);
    const size_t b_tile_stride =
            static_cast<size_t>(blocks_k) * static_cast<size_t>(kPanelB);
    const int macro_blocks = macro_cols * macro_rows;
    std::atomic<int> stop{0};

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int mb = 0; mb < macro_blocks; ++mb) {
        if (stop.load(std::memory_order_relaxed)) {
            continue;
        }
        if (should_cancel && should_cancel()) {
            stop.store(1, std::memory_order_relaxed);
            continue;
        }

        const int jm = mb / macro_rows;
        const int im = mb % macro_rows;
        const int col0 = jm * kMacroN;
        const int tc0 = jm * kMicroPerMacroN;
        const int row0 = im * kMacroM;
        const int tr0 = im * kMicroPerMacroM;

        for (int tc = 0; tc < kMicroPerMacroN && !stop.load(std::memory_order_relaxed);
             ++tc) {
            const int tc_global = tc0 + tc;
            const int micro_col0 = col0 + tc * kNR;
            const int8_t *b_base =
                    b_pre + static_cast<size_t>(tc_global) * b_tile_stride;

            for (int tr = 0; tr < kMicroPerMacroM; ++tr) {
                if (stop.load(std::memory_order_relaxed)) {
                    break;
                }
                if (should_cancel && should_cancel()) {
                    stop.store(1, std::memory_order_relaxed);
                    break;
                }

                const int tr_global = tr0 + tr;
                const int micro_row0 = row0 + tr * kMR;
                const int8_t *a_base =
                        a_pre + static_cast<size_t>(tr_global) * a_tile_stride;

                uint32_t milestone_xor[kNumMilestones] = {};
                micro_gemm_xor_milestones(
                        a_base, b_base, blocks_k, blocks_per_milestone, num_milestones, N,
                        micro_col0, /*spatial_tile_id=*/0, /*tile_count=*/1, b_comp_ms,
                        use_fast_u8s8, use_avx2, true, milestone_xor);

                if (!on_tile(milestone_xor, micro_row0, micro_col0)) {
                    stop.store(1, std::memory_order_relaxed);
                    break;
                }
            }
        }
    }

    return !(should_cancel && should_cancel());
}

int case33_test_inplace_prepack_impl(int M, int N, int K) {
    if (M % kMR != 0 || N % kNR != 0 || K % kKR != 0) {
        return 2;
    }
    const int blocks_k = K / kKR;
    const int tile_cols = N / kNR;

    std::vector<int8_t> row_a(static_cast<size_t>(M) * static_cast<size_t>(K));
    std::vector<int8_t> row_b(static_cast<size_t>(K) * static_cast<size_t>(N));
    for (size_t i = 0; i < row_a.size(); ++i) {
        row_a[i] = static_cast<int8_t>((i * 17u + 3u) & 0xffu);
    }
    for (size_t i = 0; i < row_b.size(); ++i) {
        row_b[i] = static_cast<int8_t>((i * 31u + 7u) & 0xffu);
    }

    std::vector<int8_t> ref_a;
    std::vector<int8_t> ref_b;
    prepack_a_all(row_a.data(), M, K, blocks_k, true, &ref_a);
    prepack_b_all(row_b.data(), N, K, blocks_k, tile_cols, &ref_b);

    std::vector<int8_t> inp_a = row_a;
    std::vector<int8_t> inp_b = row_b;
    std::vector<int8_t> tmp_a;
    std::vector<int8_t> tmp_b;
    prepack_a_all(inp_a.data(), M, K, blocks_k, true, &tmp_a);
    prepack_b_all(inp_b.data(), N, K, blocks_k, tile_cols, &tmp_b);
    std::swap(inp_a, tmp_a);
    std::swap(inp_b, tmp_b);

    if (inp_a != ref_a || inp_b != ref_b) {
        return 1;
    }
    return 0;
}

int case33_test_fused_prepack_impl(int M, int N, int K, int rank) {
    if (M % kMR != 0 || N % kNR != 0 || K % kKR != 0) {
        return 2;
    }
    const int blocks_k = K / kKR;
    const int tile_cols = N / kNR;

    std::vector<int8_t> sig_a(static_cast<size_t>(M) * static_cast<size_t>(K));
    for (size_t i = 0; i < sig_a.size(); ++i) {
        sig_a[i] = static_cast<int8_t>((i * 17u + 3u) & 0x3fu);
    }

    uint8_t seed_a[32]{};
    uint8_t seed_b[32]{};
    for (int i = 0; i < 32; ++i) {
        seed_a[i] = static_cast<uint8_t>(i + 1u);
        seed_b[i] = static_cast<uint8_t>(i + 33u);
    }

    std::vector<int8_t> noisy_a(sig_a.size());
    std::vector<int8_t> noisy_b(static_cast<size_t>(K) * static_cast<size_t>(N));
    if (pearl_build_noisy_a(M, K, rank, seed_a, sig_a.data(), noisy_a.data()) != 0) {
        return 3;
    }
    if (pearl_build_noisy_b(N, K, rank, seed_b, nullptr, noisy_b.data()) != 0) {
        return 3;
    }

    std::vector<int8_t> ref_a;
    std::vector<int8_t> ref_b;
    prepack_a_all(noisy_a.data(), M, K, blocks_k, true, &ref_a);
    prepack_b_all(noisy_b.data(), N, K, blocks_k, tile_cols, &ref_b);

    Case33GemmXor gemm;
    gemm.set_int8_mode(Case32Int8Mode::FastU8S8);
    gemm.set_prepack_mode(Case33PrepackMode::Fused);
    std::vector<int8_t> fused_b;
    std::vector<int8_t> fused_a;
    if (!gemm.prepare_job_b(M, N, K, &fused_b, nullptr, seed_b, rank)) {
        return 4;
    }
    if (!gemm.prepare_attempt_a(&fused_a, sig_a.data(), seed_a, rank)) {
        return 4;
    }
    if (fused_a != ref_a || fused_b != ref_b) {
        return 1;
    }
    return 0;
}

} // namespace

int case33_test_inplace_prepack(int M, int N, int K) {
    return case33_test_inplace_prepack_impl(M, N, K);
}

int case33_test_fused_prepack(int M, int N, int K, int rank) {
    return case33_test_fused_prepack_impl(M, N, K, rank);
}

bool Case33GemmXor::setup_dims_(int M, int N, int K) {
    M_ = M;
    N_ = N;
    K_ = K;
    milestone_k_ = K / kNumMilestones;
    blocks_k_ = K / kKR;
    blocks_per_milestone_ = milestone_k_ / kKR;
    tile_cols_ = N / kTileCols;
    tile_count_ = static_cast<size_t>(M / kTileRows) * static_cast<size_t>(tile_cols_);
    macro_rows_ = M / kMacroM;
    macro_cols_ = N / kMacroN;

    if (M % kMacroM != 0 || N % kMacroN != 0) {
        return false;
    }
    if (M % kTileRows != 0 || N % kTileCols != 0 || K % kNumMilestones != 0) {
        return false;
    }
    if (milestone_k_ % kKR != 0) {
        return false;
    }
    return true;
}

void Case33GemmXor::update_backend_label_() {
    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;
    const bool avx2 = cpu_has_avx2();
    const char *prepack = prepack_mode_tag(prepack_mode_);
#if defined(_OPENMP)
    const char *par = "OpenMP";
#else
    const char *par = "serial";
#endif
    if (avx2 && use_fast) {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K u8s8+XOR, AVX2 8x16 KR=%d%s",
                      par, kMacroM, kMacroN, kKR, prepack);
    } else if (avx2) {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K exact s8s8+XOR, AVX2 8x16 KR=%d%s",
                      par, kMacroM, kMacroN, kKR, prepack);
    } else {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K scalar 8x16+XOR KR=%d%s", par,
                      kMacroM, kMacroN, kKR, prepack);
    }
    backend_ = backend_buf_;
}

void Case33GemmXor::reset() {
    available_ = false;
    b_job_ready_ = false;
    backend_ = "unavailable";
    a_scan_ = nullptr;
    b_scan_ = nullptr;
    a_pre_.clear();
    b_pre_.clear();
    b_comp_ms_.clear();
    tile_xor_.clear();
}

bool Case33GemmXor::fused_noisy_prepack_b_(const int8_t *b_signal,
                                           const uint8_t *b_noise_seed, int rank,
                                           std::vector<int8_t> *scan) {
    if (!b_noise_seed || !scan) {
        return false;
    }
    const int tile_cols = N_ / kNR;
    scan->assign(static_cast<size_t>(tile_cols) * static_cast<size_t>(blocks_k_) *
                         static_cast<size_t>(kPanelB),
                 0);

    std::vector<uint32_t> pairs(static_cast<size_t>(K_) * 2u);
    pearl_build_perm_pairs_b(b_noise_seed, K_, rank, pairs.data());

    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;
    b_comp_ms_.clear();
    if (use_fast) {
        b_comp_ms_.assign(static_cast<size_t>(kNumMilestones) * static_cast<size_t>(N_), 0);
    }

    std::atomic<int> aborted{0};
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<int8_t> el(static_cast<size_t>(rank));
        std::vector<int8_t> nr(static_cast<size_t>(K_));
        std::vector<int8_t> stripe(static_cast<size_t>(kNR) * static_cast<size_t>(K_));
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 4)
#endif
        for (int tc = 0; tc < tile_cols; ++tc) {
            if (aborted.load(std::memory_order_relaxed)) {
                continue;
            }
            if ((tc & 255) == 0 && cp_job_should_cancel()) {
                aborted.store(1, std::memory_order_relaxed);
                continue;
            }
            const int col0 = tc * kNR;
            for (int c = 0; c < kNR; ++c) {
                const int col = col0 + c;
                const int8_t *sig =
                        b_signal ? b_signal + static_cast<size_t>(col) * static_cast<size_t>(K_)
                                 : nullptr;
                pearl_fuse_noise_row_b_buf(col, K_, rank, b_noise_seed, pairs.data(), sig,
                                           stripe.data() + static_cast<size_t>(c) * K_,
                                           el.data(), nr.data());
                if (use_fast) {
                    for (int ms = 0; ms < kNumMilestones; ++ms) {
                        const int k0 = ms * milestone_k_;
                        int32_t sum = 0;
                        for (int l = k0; l < k0 + milestone_k_; ++l) {
                            sum += static_cast<int32_t>(
                                    stripe[static_cast<size_t>(c) * K_ + static_cast<size_t>(l)]);
                        }
                        b_comp_ms_[static_cast<size_t>(ms) * static_cast<size_t>(N_) +
                                   static_cast<size_t>(col)] = -128 * sum;
                    }
                }
            }
            for (int kb = 0; kb < blocks_k_; ++kb) {
                const size_t idx = (static_cast<size_t>(tc) * static_cast<size_t>(blocks_k_) +
                                    static_cast<size_t>(kb)) *
                                   static_cast<size_t>(kPanelB);
                pack_b_panel(stripe.data(), 0, kb * kKR, K_, scan->data() + idx);
            }
        }
    }
    return aborted.load(std::memory_order_relaxed) == 0;
}

bool Case33GemmXor::fused_noisy_prepack_a_(const int8_t *a_signal,
                                           const uint8_t *a_noise_seed, int rank,
                                           std::vector<int8_t> *scan) {
    if (!a_signal || !a_noise_seed || !scan) {
        return false;
    }
    const int tile_rows = M_ / kMR;
    scan->assign(static_cast<size_t>(tile_rows) * static_cast<size_t>(blocks_k_) *
                         static_cast<size_t>(kPanelA),
                 0);

    std::vector<uint32_t> pairs(static_cast<size_t>(K_) * 2u);
    pearl_build_perm_pairs_a(a_noise_seed, K_, rank, pairs.data());
    const bool offset_a128 = int8_mode_ == Case32Int8Mode::FastU8S8;

    std::atomic<int> aborted{0};
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<int8_t> el(static_cast<size_t>(rank));
        std::vector<int8_t> nr(static_cast<size_t>(K_));
        std::vector<int8_t> stripe(static_cast<size_t>(kMR) * static_cast<size_t>(K_));
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 4)
#endif
        for (int tr = 0; tr < tile_rows; ++tr) {
            if (aborted.load(std::memory_order_relaxed)) {
                continue;
            }
            if ((tr & 255) == 0 && cp_job_should_cancel()) {
                aborted.store(1, std::memory_order_relaxed);
                continue;
            }
            const int row0 = tr * kMR;
            for (int r = 0; r < kMR; ++r) {
                const int row = row0 + r;
                pearl_fuse_noise_row_a_buf(
                        row, K_, rank, a_noise_seed, pairs.data(),
                        a_signal + static_cast<size_t>(row) * static_cast<size_t>(K_),
                        stripe.data() + static_cast<size_t>(r) * static_cast<size_t>(K_),
                        el.data(), nr.data());
            }
            for (int kb = 0; kb < blocks_k_; ++kb) {
                const size_t idx = (static_cast<size_t>(tr) * static_cast<size_t>(blocks_k_) +
                                    static_cast<size_t>(kb)) *
                                   static_cast<size_t>(kPanelA);
                pack_a_panel(stripe.data(), 0, kb * kKR, K_, scan->data() + idx, offset_a128);
            }
        }
    }
    return aborted.load(std::memory_order_relaxed) == 0;
}

bool Case33GemmXor::prepare_job_b(int M, int N, int K, std::vector<int8_t> *b_buf,
                                  const int8_t *b_signal, const uint8_t *b_noise_seed,
                                  int rank) {
    available_ = false;
    b_job_ready_ = false;
    num_threads_ = detect_thread_count();

    if (!b_buf || !setup_dims_(M, N, K)) {
        return false;
    }

    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;

    if (prepack_mode_ == Case33PrepackMode::Fused) {
        if (!b_noise_seed || !fused_noisy_prepack_b_(b_signal, b_noise_seed, rank, b_buf)) {
            return false;
        }
        b_pre_.clear();
        b_scan_ = b_buf->data();
    } else {
        if (b_buf->empty()) {
            return false;
        }
        const int8_t *b_row = b_buf->data();
        if (!use_fast) {
            for (size_t i = 0, count = static_cast<size_t>(K_) * N_; i < count; ++i) {
                if (b_row[i] == INT8_MIN) {
                    return false;
                }
            }
        }

        b_comp_ms_.clear();
        if (use_fast) {
            compute_b_compensation_milestones(b_row, K_, N_, kNumMilestones, milestone_k_,
                                              &b_comp_ms_);
        }
        if (prepack_mode_ == Case33PrepackMode::ReuseScanBuf) {
            std::vector<int8_t> prepacked;
            prepack_b_all(b_row, N_, K_, blocks_k_, N_ / kNR, &prepacked);
            std::swap(*b_buf, prepacked);
            b_pre_.clear();
            b_scan_ = b_buf->data();
        } else {
            prepack_b_all(b_row, N_, K_, blocks_k_, N / kNR, &b_pre_);
            b_scan_ = b_pre_.data();
        }
    }
    tile_xor_.clear();
    update_backend_label_();
    b_job_ready_ = true;
    return true;
}

bool Case33GemmXor::prepare_attempt_a(std::vector<int8_t> *a_buf, const int8_t *a_signal,
                                      const uint8_t *a_noise_seed, int rank) {
    if (!b_job_ready_ || !a_buf) {
        return false;
    }

    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;
    if (prepack_mode_ == Case33PrepackMode::Fused) {
        if (!a_signal || !a_noise_seed ||
            !fused_noisy_prepack_a_(a_signal, a_noise_seed, rank, a_buf)) {
            return false;
        }
        a_pre_.clear();
        a_scan_ = a_buf->data();
    } else {
        if (a_buf->empty()) {
            return false;
        }
        if (prepack_mode_ == Case33PrepackMode::ReuseScanBuf) {
            std::vector<int8_t> prepacked;
            prepack_a_all(a_buf->data(), M_, K_, blocks_k_, use_fast, &prepacked);
            std::swap(*a_buf, prepacked);
            a_pre_.clear();
            a_scan_ = a_buf->data();
        } else if (use_fast) {
            prepack_a_all(a_buf->data(), M_, K_, blocks_k_, true, &a_pre_);
            a_scan_ = a_pre_.data();
        } else {
            prepack_a_all(a_buf->data(), M_, K_, blocks_k_, false, &a_pre_);
            a_scan_ = a_pre_.data();
        }
    }
    available_ = true;
    return true;
}

bool Case33GemmXor::init(int M, int N, int K, const int8_t *a, const int8_t *b) {
    reset();
    set_int8_mode(int8_mode_);
    if (!a || !b) {
        return false;
    }
    const Case33PrepackMode saved_mode = prepack_mode_;
    prepack_mode_ = Case33PrepackMode::Separate;

    std::vector<int8_t> b_buf(static_cast<size_t>(K) * static_cast<size_t>(N));
    memcpy(b_buf.data(), b, b_buf.size());
    if (!prepare_job_b(M, N, K, &b_buf, nullptr, nullptr, 256)) {
        prepack_mode_ = saved_mode;
        return false;
    }

    std::vector<int8_t> a_buf(static_cast<size_t>(M) * static_cast<size_t>(K));
    memcpy(a_buf.data(), a, a_buf.size());
    const bool ok = prepare_attempt_a(&a_buf, nullptr, nullptr, 256);
    prepack_mode_ = saved_mode;
    return ok;
}

void Case33GemmXor::run() {
    if (!available_) {
        return;
    }
    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;
    run_fused_impl(a_scan_, b_scan_, N_, blocks_k_, blocks_per_milestone_,
                   kNumMilestones, macro_rows_, macro_cols_, tile_cols_, tile_count_,
                   b_comp_ms_.data(), use_fast, cpu_has_avx2(), true, &tile_xor_);
}

bool Case33GemmXor::scan_tiles(
        const std::function<bool(const uint32_t *milestone_xor, int t_rows, int t_cols)>
                &on_tile,
        const std::function<bool()> &should_cancel) {
    if (!available_ || !on_tile || !a_scan_ || !b_scan_) {
        return false;
    }
    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;
    return run_online_tile_scan(a_scan_, b_scan_, N_, blocks_k_,
                                blocks_per_milestone_, kNumMilestones, macro_rows_,
                                macro_cols_, b_comp_ms_.data(), use_fast, cpu_has_avx2(),
                                on_tile, should_cancel);
}

void Case33GemmXor::run_gemm_only() {
    if (!available_) {
        return;
    }
    const bool use_fast = int8_mode_ == Case32Int8Mode::FastU8S8;
    run_fused_impl(a_scan_, b_scan_, N_, blocks_k_, blocks_per_milestone_,
                   kNumMilestones, macro_rows_, macro_cols_, tile_cols_, tile_count_,
                   b_comp_ms_.data(), use_fast, cpu_has_avx2(), false, &tile_xor_);
}
