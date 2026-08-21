#include "case33_gemm_xor_neon.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
#include "case33_gemm_xor.hpp"

#include <arm_neon.h>

namespace {
constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kKGroups = kKR / 4;

uint32_t xor_tile(const int32_t *tile) {
    uint32_t result = 0;
    for (int i = 0; i < kMR * kNR; ++i) result ^= static_cast<uint32_t>(tile[i]);
    return result;
}

void neon_panel_accum(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals) {
    constexpr int kColsPerGroup = 8;
    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
        int32x4_t lo[kColsPerGroup];
        int32x4_t hi[kColsPerGroup];
        for (int c = 0; c < kColsPerGroup; ++c) lo[c] = hi[c] = vdupq_n_s32(0);

        for (int kg = 0; kg < kKGroups; ++kg) {
            const int8x8x4_t a = vld4_s8(a_tile + static_cast<size_t>(kg) * 32);
            const int8_t *b_group =
                    b_tile + static_cast<size_t>(jg * kKGroups + kg) * 32;
            for (int ko = 0; ko < 4; ++ko) {
                const int16x8_t aw = vmovl_s8(a.val[ko]);
                for (int c = 0; c < kColsPerGroup; ++c) {
                    const int16x8_t bw =
                            vdupq_n_s16(static_cast<int16_t>(b_group[c * 4 + ko]));
                    lo[c] = vmlal_s16(lo[c], vget_low_s16(aw), vget_low_s16(bw));
                    hi[c] = vmlal_s16(hi[c], vget_high_s16(aw), vget_high_s16(bw));
                }
            }
        }
        for (int c = 0; c < kColsPerGroup; ++c) {
            int32_t lanes[8];
            vst1q_s32(lanes, lo[c]);
            vst1q_s32(lanes + 4, hi[c]);
            const int j = jg * kColsPerGroup + c;
            for (int i = 0; i < kMR; ++i) vals[j * kMR + i] += lanes[i];
        }
    }
}
} // namespace

void case33_neon_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, size_t spatial_tile_id, size_t tile_count, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    int32_t vals[kMR * kNR] = {};
    (void)blocks_per_milestone;
    (void)num_milestones;
    for (int ms = 0; ms < blocks_k; ++ms) {
        neon_panel_accum(a_base + static_cast<size_t>(ms) * kPanelA,
                         b_base + static_cast<size_t>(ms) * kPanelB, vals);
        if (xor_after_milestone)
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id] = xor_tile(vals);
    }
}
#else
void case33_neon_micro_gemm_xor_fused_k(
        const std::int8_t *, const std::int8_t *, int, int, int, std::size_t, std::size_t, bool,
        std::uint32_t *) {}
#endif
