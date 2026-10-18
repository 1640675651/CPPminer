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
constexpr int kColsPerBGroup = 8; /* packed B group = 8 cols x 4 k */
constexpr int kColsPerPass = 4;   /* register tile = 8 rows x 4 cols */
constexpr int kColPasses = kNR / kColsPerPass;

static_assert(kMR == 8, "NEON kernel assumes 8 rows per micro tile");
static_assert(kNR % kColsPerBGroup == 0 && kColsPerBGroup % kColsPerPass == 0,
              "NEON kernel assumes 8-column B groups split into 4-column passes");
static_assert(kKGroups % 2 == 0, "NEON kernel consumes two 4-k groups (8 k) per step");

/* Broadcast the 4 k bytes of one B column to both halves: [c k0-3 | c k0-3]. */
inline int8x8_t dup_b4(const int8_t *b) {
    return vreinterpret_s8_s32(vld1_dup_s32(reinterpret_cast<const int32_t *>(b)));
}

/* XOR all 32-bit lanes of the 16 accumulators of one 8x4 pass, after folding the
 * two partial sums of every row with vpaddq_s32 (see accumulator layout below). */
inline uint32_t xor_accumulators(const int32x4_t (*acc)[4]) {
    int32x4_t x = vdupq_n_s32(0);
    for (int c = 0; c < kColsPerPass; ++c) {
        /* [r0 r1 r2 r3] and [r4 r5 r6 r7] of column c: vals[j*kMR + i] bit-identical */
        const int32x4_t v0 = vpaddq_s32(acc[c][0], acc[c][1]);
        const int32x4_t v1 = vpaddq_s32(acc[c][2], acc[c][3]);
        x = veorq_s32(x, veorq_s32(v0, v1));
    }
    const uint32x4_t u = vreinterpretq_u32_s32(x);
    const uint32x2_t h = veor_u32(vget_low_u32(u), vget_high_u32(u));
    return vget_lane_u32(h, 0) ^ vget_lane_u32(h, 1);
}
} // namespace

/* Exact s8s8 8x16 micro tile over all KR panels, Advanced SIMD only (no DotProd).
 *
 * gemmlowp-style: smull/smlal int8 -> int16 of two products per lane, then
 * sadalp (vpadalq_s16) pairwise-adds the int16 lanes into int32 accumulators.
 *
 * Value range (cp_noise.c): the noisy operands are signal in [-64, 63]
 * (pearl_generate_ab: chunk % 128 - 64) plus the sparse-permutation noise
 * pos - neg of two uniform values in [-16, 47] (RANGE_MASK 63, ZERO_PT 16), i.e.
 * noise in [-63, 63], so every A and B byte lies in [-127, 126]. A product is
 * therefore at most 127*127 = 16129 and the sum of two products at most 32258,
 * which fits int16 without wrapping. The int32 accumulation is then the exact
 * (wrap-around) sum of all products, order-independent, hence bit-identical to the
 * scalar reference. (Only a -128 * -128 + -128 * -128 pair could overflow int16,
 * and -128 never occurs in the Pearl operands.)
 *
 * Packed layouts (unchanged):
 *   A panel = kKR x kMR bytes; k-group kg (4 k) = 32 bytes = 8 rows x 4 k, row r at +4r.
 *   B panel = kKR x kNR bytes; (jg, kg) group = 32 bytes = 8 cols x 4 k, col c at +4c.
 *
 * Register tile: 8 rows x 4 columns -> 16 int32x4 accumulators acc[c][rp] with
 *   acc[c][rp] = [ C(2rp, c) partial a | C(2rp, c) partial b | C(2rp+1, c) partial a | ... b ]
 * (the pairwise add leaves two int32 partial sums per row: k = 0,1 (+4,5) and 2,3 (+6,7)
 * of every 8-k step). vpaddq_s32 of acc[c][0], acc[c][1] gives rows 0-3 of column c.
 * Four passes over the K sweep (4 cols each); A (1 KiB/panel) is re-read from L1.
 * Per 8 k and 4 columns: 4 A loads, 8 B dup-loads, 16 smull + 16 smlal + 16 sadalp. */
void case33_neon_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, size_t spatial_tile_id, size_t tile_count, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    (void)blocks_per_milestone;
    (void)num_milestones;
    for (int pass = 0; pass < kColPasses; ++pass) {
        const int jg = pass / (kColsPerBGroup / kColsPerPass);
        const int c0 = (pass % (kColsPerBGroup / kColsPerPass)) * kColsPerPass;
        int32x4_t acc[kColsPerPass][4];
        for (int c = 0; c < kColsPerPass; ++c)
            for (int rp = 0; rp < 4; ++rp) acc[c][rp] = vdupq_n_s32(0);

        for (int ms = 0; ms < blocks_k; ++ms) {
            const int8_t *a_tile = a_base + static_cast<size_t>(ms) * kPanelA;
            const int8_t *b_tile = b_base + static_cast<size_t>(ms) * kPanelB +
                                   static_cast<size_t>(jg) * kKGroups * 32 + c0 * 4;
            for (int kg = 0; kg < kKGroups; kg += 2) {
                const int8_t *a_group = a_tile + static_cast<size_t>(kg) * 32;
                const int8_t *b_group = b_tile + static_cast<size_t>(kg) * 32;
                const int8x16_t a0 = vld1q_s8(a_group);      /* rows 0-3 x k0-3 */
                const int8x16_t a1 = vld1q_s8(a_group + 16); /* rows 4-7 x k0-3 */
                const int8x16_t a2 = vld1q_s8(a_group + 32); /* rows 0-3 x k4-7 */
                const int8x16_t a3 = vld1q_s8(a_group + 48); /* rows 4-7 x k4-7 */
                for (int c = 0; c < kColsPerPass; ++c) {
                    const int8x8_t b_lo = dup_b4(b_group + c * 4);      /* col c, k0-3 */
                    const int8x8_t b_hi = dup_b4(b_group + 32 + c * 4); /* col c, k4-7 */
                    int16x8_t p01 = vmull_s8(vget_low_s8(a0), b_lo);
                    int16x8_t p23 = vmull_s8(vget_high_s8(a0), b_lo);
                    int16x8_t p45 = vmull_s8(vget_low_s8(a1), b_lo);
                    int16x8_t p67 = vmull_s8(vget_high_s8(a1), b_lo);
                    p01 = vmlal_s8(p01, vget_low_s8(a2), b_hi);
                    p23 = vmlal_s8(p23, vget_high_s8(a2), b_hi);
                    p45 = vmlal_s8(p45, vget_low_s8(a3), b_hi);
                    p67 = vmlal_s8(p67, vget_high_s8(a3), b_hi);
                    acc[c][0] = vpadalq_s16(acc[c][0], p01);
                    acc[c][1] = vpadalq_s16(acc[c][1], p23);
                    acc[c][2] = vpadalq_s16(acc[c][2], p45);
                    acc[c][3] = vpadalq_s16(acc[c][3], p67);
                }
            }
            if (xor_after_milestone) {
                uint32_t *out = tile_xor_out + static_cast<size_t>(ms) * tile_count + spatial_tile_id;
                const uint32_t x = xor_accumulators(acc);
                if (pass == 0) *out = x;
                else *out ^= x;
            }
        }
    }
}
#else
void case33_neon_micro_gemm_xor_fused_k(
        const std::int8_t *, const std::int8_t *, int, int, int, std::size_t, std::size_t, bool,
        std::uint32_t *) {}
#endif
