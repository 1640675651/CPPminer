#include "case33_gemm_xor_dotprod.hpp"

#if (defined(__aarch64__) || defined(_M_ARM64)) && \
        (defined(__ARM_FEATURE_DOTPROD) || \
         (defined(_MSC_VER) && defined(__ARM_ARCH) && __ARM_ARCH >= 802))
#include "case33_gemm_xor.hpp"

#include <arm_neon.h>

namespace {
constexpr int kMR = Case33GemmXor::kMR;
constexpr int kNR = Case33GemmXor::kNR;
constexpr int kKR = Case33GemmXor::kKR;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kKGroups = kKR / 4;
constexpr int kColsPerGroup = 8;
constexpr int kColGroups = kNR / kColsPerGroup;

static_assert(kMR == 8, "DotProd kernel assumes 8 rows per micro tile (2 x int8x16 A loads)");
static_assert(kNR % kColsPerGroup == 0, "DotProd kernel assumes 8-column groups");

/* XOR all 32-bit lanes of the 16 accumulators of one 8x8 column group. */
inline uint32_t xor_accumulators(const int32x4_t *lo, const int32x4_t *hi) {
    int32x4_t x0 = veorq_s32(veorq_s32(lo[0], lo[1]), veorq_s32(lo[2], lo[3]));
    int32x4_t x1 = veorq_s32(veorq_s32(lo[4], lo[5]), veorq_s32(lo[6], lo[7]));
    int32x4_t x2 = veorq_s32(veorq_s32(hi[0], hi[1]), veorq_s32(hi[2], hi[3]));
    int32x4_t x3 = veorq_s32(veorq_s32(hi[4], hi[5]), veorq_s32(hi[6], hi[7]));
    const uint32x4_t x = vreinterpretq_u32_s32(veorq_s32(veorq_s32(x0, x1), veorq_s32(x2, x3)));
    const uint32x2_t h = veor_u32(vget_low_u32(x), vget_high_u32(x));
    return vget_lane_u32(h, 0) ^ vget_lane_u32(h, 1);
}
} // namespace

/* Exact s8s8 8x16 micro tile over all KR panels of one K sweep, one milestone per panel.
 *
 * Packed layouts (unchanged):
 *   A panel = kKR x kMR bytes; k-group kg (4 k) = 32 bytes = 8 rows x 4 k, row r at +4r.
 *   B panel = kKR x kNR bytes; (jg, kg) group = 32 bytes = 8 cols x 4 k, col c at +4c.
 *
 * Loop order: column group (8 cols) outermost so the 16 int32x4 accumulators
 * (rows 0-3 and 4-7 for each of 8 columns) stay in registers across ALL panels.
 * Per k-group: 2 A loads (rows 0-3 / 4-7 x 4 k), 2 B loads (cols 0-3 / 4-7 x 4 k),
 * 16 sdot with the B column selected by lane (vdotq_laneq_s32).
 *
 * The cumulative tile value C[i][j] after panel ms is the int32 wrap-around sum of
 * all products up to ms, which is order-independent, so the lanes hold exactly the
 * same values the scalar reference holds in vals[]. The per-milestone tile XOR is
 * element-wise over those 128 values; the two column groups cover disjoint
 * elements, so XOR-ing the first pass' result with the second pass' result equals
 * xor_tile(vals) of the reference. A (1 KiB per panel) is re-read from L1 on the
 * second pass. */
void case33_dotprod_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, size_t spatial_tile_id, size_t tile_count, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    (void)blocks_per_milestone;
    (void)num_milestones;
    for (int jg = 0; jg < kColGroups; ++jg) {
        int32x4_t lo[kColsPerGroup];
        int32x4_t hi[kColsPerGroup];
        for (int c = 0; c < kColsPerGroup; ++c) lo[c] = hi[c] = vdupq_n_s32(0);

        for (int ms = 0; ms < blocks_k; ++ms) {
            const int8_t *a_tile = a_base + static_cast<size_t>(ms) * kPanelA;
            const int8_t *b_tile = b_base + static_cast<size_t>(ms) * kPanelB +
                                   static_cast<size_t>(jg) * kKGroups * 32;
            for (int kg = 0; kg < kKGroups; ++kg) {
                const int8_t *a_group = a_tile + static_cast<size_t>(kg) * 32;
                const int8_t *b_group = b_tile + static_cast<size_t>(kg) * 32;
                const int8x16_t a_lo = vld1q_s8(a_group);      /* rows 0-3 x k0-3 */
                const int8x16_t a_hi = vld1q_s8(a_group + 16); /* rows 4-7 x k0-3 */
                const int8x16_t b0 = vld1q_s8(b_group);        /* cols 0-3 x k0-3 */
                const int8x16_t b1 = vld1q_s8(b_group + 16);   /* cols 4-7 x k0-3 */
                lo[0] = vdotq_laneq_s32(lo[0], a_lo, b0, 0);
                hi[0] = vdotq_laneq_s32(hi[0], a_hi, b0, 0);
                lo[1] = vdotq_laneq_s32(lo[1], a_lo, b0, 1);
                hi[1] = vdotq_laneq_s32(hi[1], a_hi, b0, 1);
                lo[2] = vdotq_laneq_s32(lo[2], a_lo, b0, 2);
                hi[2] = vdotq_laneq_s32(hi[2], a_hi, b0, 2);
                lo[3] = vdotq_laneq_s32(lo[3], a_lo, b0, 3);
                hi[3] = vdotq_laneq_s32(hi[3], a_hi, b0, 3);
                lo[4] = vdotq_laneq_s32(lo[4], a_lo, b1, 0);
                hi[4] = vdotq_laneq_s32(hi[4], a_hi, b1, 0);
                lo[5] = vdotq_laneq_s32(lo[5], a_lo, b1, 1);
                hi[5] = vdotq_laneq_s32(hi[5], a_hi, b1, 1);
                lo[6] = vdotq_laneq_s32(lo[6], a_lo, b1, 2);
                hi[6] = vdotq_laneq_s32(hi[6], a_hi, b1, 2);
                lo[7] = vdotq_laneq_s32(lo[7], a_lo, b1, 3);
                hi[7] = vdotq_laneq_s32(hi[7], a_hi, b1, 3);
            }
            if (xor_after_milestone) {
                uint32_t *out = tile_xor_out + static_cast<size_t>(ms) * tile_count + spatial_tile_id;
                const uint32_t x = xor_accumulators(lo, hi);
                if (jg == 0) *out = x;
                else *out ^= x;
            }
        }
    }
}
#else
void case33_dotprod_micro_gemm_xor_fused_k(
        const std::int8_t *, const std::int8_t *, int, int, int, std::size_t, std::size_t, bool,
        std::uint32_t *) {}
#endif
