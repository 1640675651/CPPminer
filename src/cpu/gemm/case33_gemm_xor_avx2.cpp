#include "case33_gemm_xor_avx2.hpp"
#include "case33_gemm_xor.hpp"

#include <immintrin.h>

#if defined(_MSC_VER)
#define CASE33_FORCEINLINE __forceinline
#define CASE33_NOINLINE __declspec(noinline)
#else
#define CASE33_FORCEINLINE inline __attribute__((always_inline))
#define CASE33_NOINLINE __attribute__((noinline))
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
constexpr int kKGroupBytes = kColsPerGroup * kRank; /* 32 B of A rows / B cols per k-group */
constexpr int kKUnroll = 4;

static_assert(kMR == 8 && kNR == 2 * kColsPerGroup, "AVX2 ukernel assumes an 8x16 tile");
static_assert(kKGroups % kKUnroll == 0, "k-group loop is unrolled by kKUnroll");

/* Register budget (16 ymm).
 *
 * Accumulating the whole 8x16 tile at once needs 16 int32 accumulators plus
 * the A vector, the vpmaddwd ones constant and broadcast/pair temporaries
 * (19+ live values), so GCC kept the accumulators on the stack and every
 * k-step became a load-op + store per accumulator (108 vmovdqu %ymm,(%rsp)
 * in the hot loop). Instead each KR=128 panel is swept twice over all 32
 * k-groups: columns 0-7 first, then columns 8-15. One sweep keeps 8
 * accumulators + A + ones + 2-3 temporaries (~12 live) and stays entirely in
 * registers; A (1 KiB per panel) is re-read from L1 on the second sweep.
 * int32 wrap-around addition is associative, so every tile cell sums exactly
 * the same products and the milestone XOR is bit-identical to the single
 * sweep. */
struct Acc8 {
    __m256i c0, c1, c2, c3, c4, c5, c6, c7;
};

CASE33_FORCEINLINE void zero_acc8(Acc8 &acc) {
    const __m256i z = _mm256_setzero_si256();
    acc.c0 = z;
    acc.c1 = z;
    acc.c2 = z;
    acc.c3 = z;
    acc.c4 = z;
    acc.c5 = z;
    acc.c6 = z;
    acc.c7 = z;
}

/* Pin a value to a ymm register at this point in the dependency chain.
 * Without it GCC's -ftree-reassoc rewrites the unrolled acc += p0; acc += p1;
 * ... chain as acc + ((p0 + p1) + ...), making every product of the unrolled
 * body live at once (32 ymm) and spilling them all to the stack. The empty asm
 * is free (no instruction) and keeps the chain acc -> add -> add -> ... */
CASE33_FORCEINLINE __m256i pin_reg(__m256i v) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__("" : "+x"(v));
#endif
    return v;
}

CASE33_FORCEINLINE __m256i rank4_maddubs(__m256i acc, __m256i ua, __m256i sb) {
    const __m256i pair16 = _mm256_maddubs_epi16(ua, sb);
    /* Constant; the compiler materialises it once per sweep. */
    const __m256i ones16 = _mm256_set1_epi16(1);
    return pin_reg(_mm256_add_epi32(acc, _mm256_madd_epi16(pair16, ones16)));
}

/* Column c's rank-4 B bytes broadcast to all 8 lanes (folds into vpbroadcastd mem). */
CASE33_FORCEINLINE __m256i broadcast_rank4_b(const int32_t *bp) {
    return _mm256_broadcastd_epi32(_mm_cvtsi32_si128(*bp));
}

/* One rank-4 k-group for an 8x8 half: acc.c[j] += A(8 rows x 4 k) . B(col j, 4 k).
 * Fast path: A is u8 (compensated via b_comp), B is s8.
 * Exact path: |A| as u8 against sign(B, A) so each product keeps its sign. */
template <bool kExact>
CASE33_FORCEINLINE void kgroup_update8(Acc8 &acc, const int8_t *a, const int32_t *bp) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
    if constexpr (kExact) {
        const __m256i abs_a = _mm256_sign_epi8(va, va);
        acc.c0 = rank4_maddubs(acc.c0, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 0), va));
        acc.c1 = rank4_maddubs(acc.c1, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 1), va));
        acc.c2 = rank4_maddubs(acc.c2, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 2), va));
        acc.c3 = rank4_maddubs(acc.c3, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 3), va));
        acc.c4 = rank4_maddubs(acc.c4, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 4), va));
        acc.c5 = rank4_maddubs(acc.c5, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 5), va));
        acc.c6 = rank4_maddubs(acc.c6, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 6), va));
        acc.c7 = rank4_maddubs(acc.c7, abs_a, _mm256_sign_epi8(broadcast_rank4_b(bp + 7), va));
    } else {
        acc.c0 = rank4_maddubs(acc.c0, va, broadcast_rank4_b(bp + 0));
        acc.c1 = rank4_maddubs(acc.c1, va, broadcast_rank4_b(bp + 1));
        acc.c2 = rank4_maddubs(acc.c2, va, broadcast_rank4_b(bp + 2));
        acc.c3 = rank4_maddubs(acc.c3, va, broadcast_rank4_b(bp + 3));
        acc.c4 = rank4_maddubs(acc.c4, va, broadcast_rank4_b(bp + 4));
        acc.c5 = rank4_maddubs(acc.c5, va, broadcast_rank4_b(bp + 5));
        acc.c6 = rank4_maddubs(acc.c6, va, broadcast_rank4_b(bp + 6));
        acc.c7 = rank4_maddubs(acc.c7, va, broadcast_rank4_b(bp + 7));
    }
}

CASE33_FORCEINLINE __m256i xor_acc8(const Acc8 &acc) {
    __m256i x = _mm256_xor_si256(acc.c0, acc.c1);
    x = _mm256_xor_si256(x, acc.c2);
    x = _mm256_xor_si256(x, acc.c3);
    x = _mm256_xor_si256(x, acc.c4);
    x = _mm256_xor_si256(x, acc.c5);
    x = _mm256_xor_si256(x, acc.c6);
    x = _mm256_xor_si256(x, acc.c7);
    return x;
}

/* Fold this milestone's FastU8S8 column compensation (8 consecutive columns) into
 * the live register accs. */
CASE33_FORCEINLINE void apply_b_comp_acc8(Acc8 &acc, const int32_t *comp8) {
    acc.c0 = _mm256_add_epi32(acc.c0, _mm256_set1_epi32(comp8[0]));
    acc.c1 = _mm256_add_epi32(acc.c1, _mm256_set1_epi32(comp8[1]));
    acc.c2 = _mm256_add_epi32(acc.c2, _mm256_set1_epi32(comp8[2]));
    acc.c3 = _mm256_add_epi32(acc.c3, _mm256_set1_epi32(comp8[3]));
    acc.c4 = _mm256_add_epi32(acc.c4, _mm256_set1_epi32(comp8[4]));
    acc.c5 = _mm256_add_epi32(acc.c5, _mm256_set1_epi32(comp8[5]));
    acc.c6 = _mm256_add_epi32(acc.c6, _mm256_set1_epi32(comp8[6]));
    acc.c7 = _mm256_add_epi32(acc.c7, _mm256_set1_epi32(comp8[7]));
}

/* Sweep all k-groups of one KR panel for one 8-column half, then fold this
 * milestone's FastU8S8 column compensation (comp8: 8 consecutive columns, may be
 * null) into the accumulators and return their 8-lane XOR fold.
 *
 * Deliberately NOT inlined: as an out-of-line function nothing but these 8
 * accumulators, A, ones16 and a few temporaries is live inside the k-loop, so
 * the register allocator has no live-through values (the other half's
 * accumulators) competing for the 16 ymm registers. Inlined, GCC kept 2-3 of
 * the other half's accumulators in registers and round-tripped 2-3 of ours
 * through the stack on every unrolled iteration. The call + 8 loads + 8 stores
 * per 256 vpmaddubsw is noise. */
template <bool kExact>
CASE33_NOINLINE __m256i half_panel_kgroups(Acc8 *acc_io, const int8_t *a_tile,
                                           const int8_t *b_half, const int32_t *comp8) {
    Acc8 acc = *acc_io;
    for (int kg = 0; kg < kKGroups; kg += kKUnroll) {
        const int8_t *a = a_tile + static_cast<size_t>(kg) * kKGroupBytes;
        const int32_t *bp =
                reinterpret_cast<const int32_t *>(b_half + static_cast<size_t>(kg) * kKGroupBytes);
        kgroup_update8<kExact>(acc, a, bp);
        kgroup_update8<kExact>(acc, a + kKGroupBytes, bp + kColsPerGroup);
        kgroup_update8<kExact>(acc, a + 2 * kKGroupBytes, bp + 2 * kColsPerGroup);
        kgroup_update8<kExact>(acc, a + 3 * kKGroupBytes, bp + 3 * kColsPerGroup);
    }
    if (comp8) {
        apply_b_comp_acc8(acc, comp8);
    }
    *acc_io = acc;
    return xor_acc8(acc);
}

CASE33_FORCEINLINE uint32_t reduce_xor_epi32(__m256i v) {
    v = _mm256_xor_si256(v, _mm256_permute2x128_si256(v, v, 0x01));
    __m128i x = _mm256_castsi256_si128(v);
    x = _mm_xor_si128(x, _mm_srli_si128(x, 8));
    x = _mm_xor_si128(x, _mm_srli_si128(x, 4));
    return static_cast<uint32_t>(_mm_cvtsi128_si32(x));
}

template <bool kExact>
CASE33_FORCEINLINE void avx2_fused_k_sweep(const int8_t *a_base, const int8_t *b_base,
                                           int blocks_k, int N, int global_col0,
                                           size_t spatial_tile_id, size_t tile_count,
                                           const int32_t *b_comp_ms, bool xor_after_milestone,
                                           uint32_t *tile_xor_out) {
    Acc8 lo; /* tile columns 0-7 */
    Acc8 hi; /* tile columns 8-15 */
    zero_acc8(lo);
    zero_acc8(hi);

    /* One milestone per KR panel (blocks_per_milestone == 1), so ms == kb. */
    for (int kb = 0; kb < blocks_k; ++kb) {
        const int8_t *a_tile = a_base + static_cast<size_t>(kb) * kPanelA;
        const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
        const int32_t *comp =
                b_comp_ms ? b_comp_ms + static_cast<size_t>(kb) * static_cast<size_t>(N) +
                                    static_cast<size_t>(global_col0)
                          : nullptr;
        const __m256i x_lo = half_panel_kgroups<kExact>(&lo, a_tile, b_tile, comp);
        const __m256i x_hi = half_panel_kgroups<kExact>(
                &hi, a_tile, b_tile + static_cast<size_t>(kKGroups) * kKGroupBytes,
                comp ? comp + kColsPerGroup : nullptr);
        if (xor_after_milestone) {
            /* XOR-fold of all 8x16 cumulative C cells after this milestone. */
            tile_xor_out[static_cast<size_t>(kb) * tile_count + spatial_tile_id] =
                    reduce_xor_epi32(_mm256_xor_si256(x_lo, x_hi));
        }
    }
}

} // namespace

void case33_avx2_micro_gemm_xor_fused_k(
        const int8_t *a_base, const int8_t *b_base, int blocks_k, int blocks_per_milestone,
        int num_milestones, int N, int global_col0, size_t spatial_tile_id, size_t tile_count,
        const int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        uint32_t *tile_xor_out) {
    (void)blocks_per_milestone;
    (void)num_milestones;
    if (use_fast_u8s8) {
        avx2_fused_k_sweep<false>(a_base, b_base, blocks_k, N, global_col0, spatial_tile_id,
                                  tile_count, b_comp_ms, xor_after_milestone, tile_xor_out);
    } else {
        /* Exact s8s8 needs no column compensation. */
        avx2_fused_k_sweep<true>(a_base, b_base, blocks_k, N, global_col0, spatial_tile_id,
                                 tile_count, nullptr, xor_after_milestone, tile_xor_out);
    }
}
