#include "case33_gemm_xor_dotprod.hpp"

#if (defined(__aarch64__) || defined(_M_ARM64)) && \
        (defined(__ARM_FEATURE_DOTPROD) || \
         (defined(_MSC_VER) && defined(__ARM_ARCH) && __ARM_ARCH >= 802))
#include "case33_gemm_xor.hpp"

#include <arm_neon.h>

#include <cstring>

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

int8x16_t repeat_b4(const int8_t *b) {
    int32_t packed = 0;
    std::memcpy(&packed, b, sizeof(packed));
    return vreinterpretq_s8_s32(vdupq_n_s32(packed));
}

void dotprod_panel_accum(const int8_t *a_tile, const int8_t *b_tile, int32_t *vals) {
    constexpr int kColsPerGroup = 8;
    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
        int32x4_t lo[kColsPerGroup];
        int32x4_t hi[kColsPerGroup];
        for (int c = 0; c < kColsPerGroup; ++c) lo[c] = hi[c] = vdupq_n_s32(0);

        for (int kg = 0; kg < kKGroups; ++kg) {
            const int8_t *a_group = a_tile + static_cast<size_t>(kg) * 32;
            const int8x16_t a_lo = vld1q_s8(a_group);
            const int8x16_t a_hi = vld1q_s8(a_group + 16);
            const int8_t *b_group =
                    b_tile + static_cast<size_t>(jg * kKGroups + kg) * 32;
            for (int c = 0; c < kColsPerGroup; ++c) {
                const int8x16_t b = repeat_b4(b_group + c * 4);
                lo[c] = vdotq_s32(lo[c], a_lo, b);
                hi[c] = vdotq_s32(hi[c], a_hi, b);
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

void case33_dotprod_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, size_t spatial_tile_id, size_t tile_count, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    int32_t vals[kMR * kNR] = {};
    (void)blocks_per_milestone;
    (void)num_milestones;
    for (int ms = 0; ms < blocks_k; ++ms) {
        dotprod_panel_accum(a_base + static_cast<size_t>(ms) * kPanelA,
                            b_base + static_cast<size_t>(ms) * kPanelB, vals);
        if (xor_after_milestone)
            tile_xor_out[static_cast<size_t>(ms) * tile_count + spatial_tile_id] = xor_tile(vals);
    }
}
#else
void case33_dotprod_micro_gemm_xor_fused_k(
        const std::int8_t *, const std::int8_t *, int, int, int, std::size_t, std::size_t, bool,
        std::uint32_t *) {}
#endif
