#include "case33_gemm_xor.hpp"
#include "case33_cpu_features.hpp"
#include "case33_gemm_xor_neon.hpp"

#if defined(_M_X64) || defined(_M_IX86) || defined(__i386__) || defined(__x86_64__)
#define CASE33_X86 1
#include "case33_gemm_xor_avx2.hpp"
#include "case33_gemm_xor_ssse3.hpp"
#else
#define CASE33_X86 0
#endif

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
constexpr int kKGroups = kKR / kRank;

bool resolve_isa(Case33Isa pref, Case33Isa *out, char *error, size_t error_size) {
    const Case33CpuFeatures features = case33_detect_cpu_features();
    const bool avx2 = CASE33_X86 && features.avx2;
    const bool ssse3 = CASE33_X86 && features.ssse3;
    const bool neon = case33_is_aarch64_build() && features.neon;
    switch (pref) {
    case Case33Isa::Avx2:
        if (avx2) { *out = Case33Isa::Avx2; return true; }
        break;
    case Case33Isa::Sse:
        if (ssse3) { *out = Case33Isa::Sse; return true; }
        break;
    case Case33Isa::Neon:
        if (neon) { *out = Case33Isa::Neon; return true; }
        break;
    case Case33Isa::Scalar:
        *out = Case33Isa::Scalar;
        return true;
    case Case33Isa::Auto:
    default:
        if (avx2) *out = Case33Isa::Avx2;
        else if (ssse3) *out = Case33Isa::Sse;
        else if (neon) *out = Case33Isa::Neon;
        else *out = Case33Isa::Scalar;
        return true;
    }
    std::snprintf(error, error_size, "requested SIMD ISA is unavailable on this build or CPU");
    return false;
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

const char *sse_tile_name(Case33SseTile tile) {
    switch (tile) {
    case Case33SseTile::R8C8: return "8x8regs";
    case Case33SseTile::R4C16: return "4x16regs";
    case Case33SseTile::R4C8:
    default: return "4x8regs";
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

void scalar_panel_accum(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals,
                        bool use_fast_u8s8) {
    (void)use_fast_u8s8;
    for (int j = 0; j < kNR; ++j) {
        for (int i = 0; i < kMR; ++i) {
            for (int kg = 0; kg < kKGroups; ++kg) {
                for (int ko = 0; ko < kRank; ++ko) {
                    const size_t a_idx =
                            static_cast<size_t>(kg) * 32 + static_cast<size_t>(i * kRank + ko);
                    const int32_t a = static_cast<int32_t>(a_tile[a_idx]);
                    const size_t b_idx =
                            (static_cast<size_t>(j / kColsPerGroup) *
                                             static_cast<size_t>(kKGroups) +
                             static_cast<size_t>(kg)) *
                                            32 +
                            static_cast<size_t>(j % kColsPerGroup) * 4 +
                            static_cast<size_t>(ko);
                    vals[j * kMR + i] += a * static_cast<int32_t>(b_tile[b_idx]);
                }
            }
        }
    }
}

void scalar_micro_gemm_xor_fused_k(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                                   int blocks_per_milestone, int num_milestones, int N,
                                   int global_col0, size_t spatial_tile_id, size_t tile_count,
                                   const int32_t *b_comp_ms, bool use_fast_u8s8,
                                   bool xor_after_milestone, uint32_t *tile_xor_out) {
    /* Cumulative C tile (never cleared). One KR panel == one milestone. */
    int32_t vals[kNR * kMR] = {};
    int ms = 0;
    (void)blocks_per_milestone;

    for (int kb = 0; kb < blocks_k; ++kb) {
        scalar_panel_accum(a_base + static_cast<size_t>(kb) * kPanelA,
                           b_base + static_cast<size_t>(kb) * kPanelB, vals, use_fast_u8s8);

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



void micro_gemm_xor_fused_k(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                            int blocks_per_milestone, int num_milestones, int N,
                            int global_col0, size_t spatial_tile_id, size_t tile_count,
                            const int32_t *b_comp_ms, bool use_fast_u8s8, Case33Isa isa,
                            Case33SseTile sse_tile, bool xor_after_milestone,
                            uint32_t *tile_xor_out) {
#if CASE33_X86
    if (isa == Case33Isa::Avx2) {
        case33_avx2_micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone,
                                           num_milestones, N, global_col0, spatial_tile_id,
                                           tile_count, b_comp_ms, use_fast_u8s8,
                                           xor_after_milestone, tile_xor_out);
        return;
    }
    if (isa == Case33Isa::Sse) {
        case33_ssse3_micro_gemm_xor_fused_k(
                a_base, b_base, blocks_k, blocks_per_milestone, num_milestones, N,
                global_col0, spatial_tile_id, tile_count, b_comp_ms, use_fast_u8s8, sse_tile,
                xor_after_milestone, tile_xor_out);
        return;
    }
#endif
    if (isa == Case33Isa::Neon) {
        case33_neon_micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone,
                                            num_milestones, N, global_col0, spatial_tile_id,
                                            tile_count, b_comp_ms, use_fast_u8s8,
                                            xor_after_milestone, tile_xor_out);
        return;
    }
    scalar_micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone,
                                  num_milestones, N, global_col0, spatial_tile_id, tile_count,
                                  b_comp_ms, use_fast_u8s8, xor_after_milestone, tile_xor_out);
}

void micro_gemm_xor_milestones(const int8_t *a_base, const int8_t *b_base, int blocks_k,
                               int blocks_per_milestone, int num_milestones, int N,
                               int global_col0, size_t spatial_tile_id, size_t tile_count,
                               const int32_t *b_comp_ms, bool use_fast_u8s8, Case33Isa isa,
                               Case33SseTile sse_tile, bool xor_after_milestone,
                               uint32_t *tile_xor_out) {
    micro_gemm_xor_fused_k(a_base, b_base, blocks_k, blocks_per_milestone, num_milestones,
                           N, global_col0, spatial_tile_id, tile_count, b_comp_ms,
                           use_fast_u8s8, isa, sse_tile, xor_after_milestone, tile_xor_out);
}

// Case 3.2 macro schedule: 2D OpenMP over macro blocks, tc outer / tr inner (B reuse).
void run_hardcoded_macro_xor(const int8_t *a_pre, const int8_t *b_pre, int N, int blocks_k,
                             int blocks_per_milestone, int num_milestones, int macro_rows,
                             int macro_cols, int tile_cols, size_t tile_count,
                             const int32_t *b_comp_ms, bool use_fast_u8s8, Case33Isa isa,
                             Case33SseTile sse_tile, bool xor_after_milestone,
                             std::vector<uint32_t> *out) {
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
                        use_fast_u8s8, isa, sse_tile, xor_after_milestone, out->data());
            }
        }
    }
}

void run_fused_impl(const int8_t *a_pre, const int8_t *b_pre, int N, int blocks_k,
                    int blocks_per_milestone, int num_milestones, int macro_rows,
                    int macro_cols, int tile_cols, size_t tile_count,
                    const int32_t *b_comp_ms, bool use_fast_u8s8, Case33Isa isa,
                    Case33SseTile sse_tile, bool xor_after_milestone,
                    std::vector<uint32_t> *out) {
    out->assign(static_cast<size_t>(num_milestones) * tile_count, 0u);
    run_hardcoded_macro_xor(a_pre, b_pre, N, blocks_k, blocks_per_milestone, num_milestones,
                            macro_rows, macro_cols, tile_cols, tile_count, b_comp_ms,
                            use_fast_u8s8, isa, sse_tile, xor_after_milestone, out);
}

bool run_online_tile_scan(
        const int8_t *a_pre, const int8_t *b_pre, int N, int blocks_k,
        int blocks_per_milestone, int num_milestones, int macro_rows, int macro_cols,
        const int32_t *b_comp_ms, bool use_fast_u8s8, Case33Isa isa,
        Case33SseTile sse_tile,
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
                        use_fast_u8s8, isa, sse_tile, true, milestone_xor);

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

int case33_test_simd_parity() {
    constexpr int kTestM = Case33GemmXor::kMacroM;
    constexpr int kTestN = Case33GemmXor::kMacroN;
    std::vector<int8_t> a(static_cast<size_t>(kTestM) * K_DIM);
    std::vector<int8_t> b(static_cast<size_t>(K_DIM) * kTestN);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<int8_t>(static_cast<int>(i * 17u % 127u) - 63);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<int8_t>(static_cast<int>(i * 29u % 127u) - 63);

    for (Case32Int8Mode mode : {Case32Int8Mode::FastU8S8, Case32Int8Mode::ExactS8S8}) {
        Case33GemmXor scalar;
        scalar.set_isa(Case33Isa::Scalar);
        scalar.set_int8_mode(mode);
        if (!scalar.init(kTestM, kTestN, K_DIM, a.data(), b.data())) return 1;
        scalar.run();
        const std::vector<uint32_t> expected = scalar.tile_xor();

        for (Case33Isa isa : {Case33Isa::Sse, Case33Isa::Avx2, Case33Isa::Neon}) {
            Case33GemmXor candidate;
            candidate.set_isa(isa);
            candidate.set_int8_mode(mode);
            if (!candidate.resolve_runtime_isa()) continue;
            if (!candidate.init(kTestM, kTestN, K_DIM, a.data(), b.data())) return 1;
            candidate.run();
            if (candidate.tile_xor() != expected) return 2;
        }
    }
    return 0;
}

bool Case33GemmXor::setup_dims_(int M, int N, int K) {
    M_ = M;
    N_ = N;
    K_ = K;
    milestone_k_ = R_RANK;
    blocks_k_ = K / kKR;
    blocks_per_milestone_ = 1;
    tile_cols_ = N / kTileCols;
    tile_count_ = static_cast<size_t>(M / kTileRows) * static_cast<size_t>(tile_cols_);
    macro_rows_ = M / kMacroM;
    macro_cols_ = N / kMacroN;

    if (M % kMacroM != 0 || N % kMacroN != 0) {
        return false;
    }
    if (M % kTileRows != 0 || N % kTileCols != 0 || K % kKR != 0) {
        return false;
    }
    if (K % R_RANK != 0 || (K / R_RANK) != kNumMilestones || kKR != R_RANK) {
        return false;
    }
    return true;
}

bool Case33GemmXor::resolve_runtime_isa() {
    simd_error_[0] = '\0';
    return resolve_isa(isa_pref_, &isa_used_, simd_error_, sizeof(simd_error_));
}

void Case33GemmXor::update_backend_label_() {
    if (!resolve_runtime_isa()) {
        backend_ = "unavailable SIMD ISA";
        return;
    }
    const bool use_fast = use_fast_u8s8_();
    const char *prepack = prepack_mode_tag(prepack_mode_);
#if defined(_OPENMP)
    const char *par = "OpenMP";
#else
    const char *par = "serial";
#endif
    const char *dot = use_fast ? "u8s8" : "exact s8s8";
    if (isa_used_ == Case33Isa::Avx2) {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K %s+XOR, AVX2 8x16 KR=%d%s",
                      par, kMacroM, kMacroN, dot, kKR, prepack);
    } else if (isa_used_ == Case33Isa::Sse) {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K %s+XOR, SSSE3 %s/8x16tile KR=%d%s",
                      par, kMacroM, kMacroN, dot, sse_tile_name(sse_tile_), kKR, prepack);
    } else if (isa_used_ == Case33Isa::Neon) {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K %s+XOR, NEON 8x16 KR=%d%s",
                      par, kMacroM, kMacroN, dot, kKR, prepack);
    } else {
        std::snprintf(backend_buf_, sizeof(backend_buf_),
                      "ukernel %s %dx%d 2D-par fused-K %s+XOR, scalar 8x16 KR=%d%s",
                      par, kMacroM, kMacroN, dot, kKR, prepack);
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

    const bool use_fast = use_fast_u8s8_();
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
    const bool offset_a128 = use_fast_u8s8_();

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
    if (!resolve_runtime_isa()) {
        return false;
    }

    const bool use_fast = use_fast_u8s8_();

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
        if (!use_fast && (isa_used_ == Case33Isa::Avx2 || isa_used_ == Case33Isa::Sse)) {
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

    const bool use_fast = use_fast_u8s8_();
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
    const bool use_fast = use_fast_u8s8_();
    run_fused_impl(a_scan_, b_scan_, N_, blocks_k_, blocks_per_milestone_,
                   kNumMilestones, macro_rows_, macro_cols_, tile_cols_, tile_count_,
                   b_comp_ms_.data(), use_fast, isa_used_, sse_tile_, true, &tile_xor_);
}

bool Case33GemmXor::scan_tiles(
        const std::function<bool(const uint32_t *milestone_xor, int t_rows, int t_cols)>
                &on_tile,
        const std::function<bool()> &should_cancel) {
    if (!available_ || !on_tile || !a_scan_ || !b_scan_) {
        return false;
    }
    const bool use_fast = use_fast_u8s8_();
    return run_online_tile_scan(a_scan_, b_scan_, N_, blocks_k_,
                                blocks_per_milestone_, kNumMilestones, macro_rows_,
                                macro_cols_, b_comp_ms_.data(), use_fast, isa_used_,
                                sse_tile_, on_tile, should_cancel);
}

void Case33GemmXor::run_gemm_only() {
    if (!available_) {
        return;
    }
    const bool use_fast = use_fast_u8s8_();
    run_fused_impl(a_scan_, b_scan_, N_, blocks_k_, blocks_per_milestone_,
                   kNumMilestones, macro_rows_, macro_cols_, tile_cols_, tile_count_,
                   b_comp_ms_.data(), use_fast, isa_used_, sse_tile_, false, &tile_xor_);
}
