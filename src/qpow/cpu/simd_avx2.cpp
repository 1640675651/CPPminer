#include "qpow/miner.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace qpow {
namespace {

// AVX2_ATTR: non-inlined functions that may use AVX2. AVX2_INL: helpers that are
// always inlined into them, so the 12-vector state can stay in registers and no
// per-call xmm6-xmm15 save/restore (Windows ABI) or state reload is paid per round.
#if defined(__GNUC__) || defined(__clang__)
#define AVX2_ATTR __attribute__((target("avx2")))
#define AVX2_INL __attribute__((target("avx2"), always_inline)) inline
#elif defined(_MSC_VER)
#define AVX2_ATTR
#define AVX2_INL __forceinline
#else
#define AVX2_ATTR
#define AVX2_INL inline
#endif

AVX2_INL __m256i u64_lt(__m256i a, __m256i b) {
    const __m256i k = _mm256_set1_epi64x((long long)0x8000000000000000ull);
    return _mm256_cmpgt_epi64(_mm256_xor_si256(b, k), _mm256_xor_si256(a, k));
}

// Field add on non-canonical lanes: +EPS on carry, rare second +EPS. The second
// wrap can only follow a first one (without it s1 == s0 and the compare is
// false), so it needs no conjunction with the first carry mask.
AVX2_INL __m256i avx_add(__m256i a, __m256i b) {
    const __m256i eps = _mm256_set1_epi64x((long long)EPS64);
    __m256i s0 = _mm256_add_epi64(a, b);
    __m256i s1 = _mm256_add_epi64(s0, _mm256_and_si256(u64_lt(s0, a), eps));
    return _mm256_add_epi64(s1, _mm256_and_si256(u64_lt(s1, s0), eps));
}

AVX2_INL __m256i avx_canon(__m256i a) {
    const __m256i p = _mm256_set1_epi64x((long long)P64);
    __m256i ge = _mm256_or_si256(u64_lt(p, a), _mm256_cmpeq_epi64(a, p));
    return _mm256_sub_epi64(a, _mm256_and_si256(ge, p));
}

AVX2_INL __m256i avx_reduce(__m256i lo, __m256i hi) {
    const __m256i eps = _mm256_set1_epi64x((long long)EPS64);
    __m256i hi_hi = _mm256_srli_epi64(hi, 32);
    __m256i hi_lo = _mm256_and_si256(hi, eps);
    __m256i t0 = _mm256_sub_epi64(lo, hi_hi);
    t0 = _mm256_sub_epi64(t0, _mm256_and_si256(u64_lt(lo, hi_hi), eps));
    __m256i t1 = _mm256_sub_epi64(_mm256_slli_epi64(hi_lo, 32), hi_lo);
    __m256i t2 = _mm256_add_epi64(t0, t1);
    return _mm256_add_epi64(t2, _mm256_and_si256(u64_lt(t2, t0), eps));
}

// 64x64 -> 128 per lane from four 32x32 products. The cross terms are split
// into 32-bit halves before they are summed, so no partial sum can carry:
// mid < 3*2^32, and hh + three values below 2^32 still fits in 64 bits. This
// avoids the two unsigned-compare carry checks (each 2 xor + a port-5-only
// vpcmpgtq) that a 64-bit cross-term sum needs.
AVX2_INL __m256i avx_mul(__m256i a, __m256i b) {
    const __m256i m32 = _mm256_set1_epi64x((long long)EPS64);
    __m256i a_hi = _mm256_srli_epi64(a, 32);
    __m256i b_hi = _mm256_srli_epi64(b, 32);
    __m256i ll = _mm256_mul_epu32(a, b);
    __m256i lh = _mm256_mul_epu32(a, b_hi);
    __m256i hl = _mm256_mul_epu32(a_hi, b);
    __m256i hh = _mm256_mul_epu32(a_hi, b_hi);
    __m256i mid = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_srli_epi64(ll, 32), _mm256_and_si256(lh, m32)),
        _mm256_and_si256(hl, m32));
    __m256i lo = _mm256_or_si256(_mm256_and_si256(ll, m32), _mm256_slli_epi64(mid, 32));
    __m256i hi = _mm256_add_epi64(
        _mm256_add_epi64(hh, _mm256_srli_epi64(lh, 32)),
        _mm256_add_epi64(_mm256_srli_epi64(hl, 32), _mm256_srli_epi64(mid, 32)));
    return avx_reduce(lo, hi);
}

AVX2_INL __m256i avx_sbox(__m256i x) {
    __m256i x2 = avx_mul(x, x);
    __m256i x4 = avx_mul(x2, x2);
    return avx_mul(avx_mul(x4, x2), x);
}

AVX2_INL void avx_ext(__m256i s[WIDTH]) {
    for (int chunk = 0; chunk < 3; chunk++) {
        int o = chunk * 4;
        __m256i x0 = s[o], x1 = s[o + 1], x2 = s[o + 2], x3 = s[o + 3];
        __m256i t01 = avx_add(x0, x1);
        __m256i t23 = avx_add(x2, x3);
        __m256i t0123 = avx_add(t01, t23);
        __m256i t01123 = avx_add(t0123, x1);
        __m256i t01233 = avx_add(t0123, x3);
        s[o + 3] = avx_add(t01233, avx_add(x0, x0));
        s[o + 1] = avx_add(t01123, avx_add(x2, x2));
        s[o] = avx_add(t01123, t01);
        s[o + 2] = avx_add(t01233, t23);
    }
    // Unrolled so the four column sums stay in registers (no indexed stack array).
    __m256i sum0 = avx_add(avx_add(s[0], s[4]), s[8]);
    __m256i sum1 = avx_add(avx_add(s[1], s[5]), s[9]);
    __m256i sum2 = avx_add(avx_add(s[2], s[6]), s[10]);
    __m256i sum3 = avx_add(avx_add(s[3], s[7]), s[11]);
    s[0] = avx_add(s[0], sum0);
    s[1] = avx_add(s[1], sum1);
    s[2] = avx_add(s[2], sum2);
    s[3] = avx_add(s[3], sum3);
    s[4] = avx_add(s[4], sum0);
    s[5] = avx_add(s[5], sum1);
    s[6] = avx_add(s[6], sum2);
    s[7] = avx_add(s[7], sum3);
    s[8] = avx_add(s[8], sum0);
    s[9] = avx_add(s[9], sum1);
    s[10] = avx_add(s[10], sum2);
    s[11] = avx_add(s[11], sum3);
}

AVX2_INL void avx_int(__m256i s[WIDTH]) {
    // Tree-shaped sum: 4 dependent adds deep instead of 11. The internal round is
    // a serial chain (S-box -> sum -> multiply-add -> next S-box), so depth matters.
    __m256i t01 = avx_add(s[0], s[1]);
    __m256i t23 = avx_add(s[2], s[3]);
    __m256i t45 = avx_add(s[4], s[5]);
    __m256i t67 = avx_add(s[6], s[7]);
    __m256i t89 = avx_add(s[8], s[9]);
    __m256i tab = avx_add(s[10], s[11]);
    __m256i sum = avx_add(avx_add(avx_add(t01, t23), avx_add(t45, t67)), avx_add(t89, tab));
    for (int i = 0; i < WIDTH; i++)
        s[i] = avx_add(avx_mul(s[i], _mm256_set1_epi64x((long long)MDS_DIAG[i])), sum);
}

AVX2_ATTR void avx_permute(__m256i s[WIDTH]) {
    avx_ext(s);
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < WIDTH; i++)
            s[i] = avx_add(s[i], _mm256_set1_epi64x((long long)RC_INITIAL[r][i]));
        for (int i = 0; i < WIDTH; i++) s[i] = avx_sbox(s[i]);
        avx_ext(s);
    }
    for (int r = 0; r < 22; r++) {
        s[0] = avx_sbox(avx_add(s[0], _mm256_set1_epi64x((long long)RC_INTERNAL[r])));
        avx_int(s);
    }
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < WIDTH; i++)
            s[i] = avx_add(s[i], _mm256_set1_epi64x((long long)RC_TERMINAL[r][i]));
        for (int i = 0; i < WIDTH; i++) s[i] = avx_sbox(s[i]);
        avx_ext(s);
    }
}

AVX2_ATTR void avx_absorb32(__m256i s[WIDTH], const uint8_t lows[4][32]) {
    for (int i = 0; i < 8; i++) {
        alignas(32) uint64_t w[4];
        for (int lane = 0; lane < 4; lane++) {
            uint32_t t;
            std::memcpy(&t, lows[lane] + i * 4, 4);
            w[lane] = t;
        }
        s[i] = avx_add(s[i], _mm256_load_si256((const __m256i*)w));
    }
}

// Returns a bitmask of lanes that still need the second squeeze
// (high 32 bytes of hash are <= target high 32). Full hashes are only
// valid for those lanes after the optional third permute.
AVX2_ATTR int avx_hash4_impl(const uint64_t mid[WIDTH], const uint8_t lows[4][32],
                             const uint8_t target[64], uint8_t hashes[4][64]) {
    __m256i s[WIDTH];
    for (int i = 0; i < WIDTH; i++) s[i] = _mm256_set1_epi64x((long long)mid[i]);
    avx_absorb32(s, lows);
    avx_permute(s);
    s[0] = avx_add(s[0], _mm256_set1_epi64x(1));
    s[1] = avx_add(s[1], _mm256_set1_epi64x(1));
    avx_permute(s);

    alignas(32) uint64_t tmp[4];
    uint64_t felts[4][WIDTH];
    for (int i = 0; i < WIDTH; i++) {
        _mm256_store_si256((__m256i*)tmp, avx_canon(s[i]));
        for (int lane = 0; lane < 4; lane++) felts[lane][i] = tmp[lane];
    }
    int need = 0;
    for (int lane = 0; lane < 4; lane++) {
        squeeze32(hashes[lane], felts[lane]);
        int cmp = std::memcmp(hashes[lane], target, 32);
        if (cmp > 0) {
            std::memset(hashes[lane] + 32, 0xff, 32);
        } else {
            need |= 1 << lane;
        }
    }
    if (need) {
        avx_permute(s);
        for (int i = 0; i < WIDTH; i++) {
            _mm256_store_si256((__m256i*)tmp, avx_canon(s[i]));
            for (int lane = 0; lane < 4; lane++) felts[lane][i] = tmp[lane];
        }
        for (int lane = 0; lane < 4; lane++) {
            if (need & (1 << lane)) squeeze32(hashes[lane] + 32, felts[lane]);
        }
    }
    _mm256_zeroupper();
    return need;
}

}  // namespace

bool cpu_has_avx2() {
#if defined(_MSC_VER)
    int c[4];
    __cpuid(c, 1);
    if ((c[2] & (1 << 27)) == 0) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(c, 7, 0);
    return (c[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

SearchResult search_range_avx2(const uint8_t header[32], const uint8_t start[64],
                               uint64_t count, const uint8_t target[64]) {
    if (!cpu_has_avx2()) return search_range_scalar(header, start, count, target);

    SearchResult r{};
    uint8_t nonce[64];
    std::memcpy(nonce, start, 64);
    uint64_t mid[WIDTH];
    mining_midstate(header, nonce, mid);
    uint8_t high[32];
    std::memcpy(high, nonce, 32);

    uint64_t i = 0;
    while (i + 4 <= count) {
        uint8_t batch[4][64];
        uint8_t lows[4][32];
        bool split = false;
        for (int k = 0; k < 4; k++) {
            std::memcpy(batch[k], nonce, 64);
            if (std::memcmp(nonce, high, 32) != 0) {
                split = true;
                break;
            }
            std::memcpy(lows[k], nonce + 32, 32);
            inc_be(nonce);
        }
        if (split) {
            std::memcpy(nonce, batch[0], 64);
            std::memcpy(high, nonce, 32);
            mining_midstate(header, high, mid);
            uint8_t hash[64];
            r.hashes++;
            i++;
            if (hash_if_valid(mid, nonce + 32, target, hash)) {
                r.found = true;
                std::memcpy(r.nonce, nonce, 64);
                std::memcpy(r.hash, hash, 64);
                return r;
            }
            inc_be(nonce);
            continue;
        }

        uint8_t hashes[4][64];
        avx_hash4_impl(mid, lows, target, hashes);
        r.hashes += 4;
        i += 4;
        for (int k = 0; k < 4; k++) {
            if (std::memcmp(hashes[k], target, 64) < 0) {
                r.found = true;
                std::memcpy(r.nonce, batch[k], 64);
                std::memcpy(r.hash, hashes[k], 64);
                return r;
            }
        }
    }
    while (i < count) {
        if (std::memcmp(nonce, high, 32) != 0) {
            std::memcpy(high, nonce, 32);
            mining_midstate(header, high, mid);
        }
        uint8_t hash[64];
        r.hashes++;
        i++;
        if (hash_if_valid(mid, nonce + 32, target, hash)) {
            r.found = true;
            std::memcpy(r.nonce, nonce, 64);
            std::memcpy(r.hash, hash, 64);
            return r;
        }
        inc_be(nonce);
    }
    return r;
}

int test_avx2_field() {
    if (!cpu_has_avx2()) return 0;
    int fail = 0;
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (int t = 0; t < 400; t++) {
        alignas(32) uint64_t a[4], b[4], got_add[4], got_mul[4], got_sbox[4];
        for (int k = 0; k < 4; k++) {
            a[k] = rnd();
            b[k] = rnd();
        }
        __m256i va = _mm256_load_si256((const __m256i*)a);
        __m256i vb = _mm256_load_si256((const __m256i*)b);
        _mm256_store_si256((__m256i*)got_add, avx_add(va, vb));
        _mm256_store_si256((__m256i*)got_mul, avx_mul(va, vb));
        _mm256_store_si256((__m256i*)got_sbox, avx_sbox(va));
        _mm256_zeroupper();
        for (int k = 0; k < 4; k++) {
            if (gf_canon(got_add[k]) != gf_canon(gf_add(a[k], b[k]))) fail++;
            if (gf_canon(got_mul[k]) != gf_canon(gf_mul(a[k], b[k]))) fail++;
            if (gf_canon(got_sbox[k]) != gf_canon(gf_sbox(a[k]))) fail++;
        }
        if (fail) return fail;
    }
    return fail;
}

int test_avx2_hash_parity() {
    if (!cpu_has_avx2()) return 0;
    uint8_t header[32];
    std::memset(header, 7, 32);
    uint8_t start[64];
    std::memset(start, 0, 64);
    start[63] = 0x10;
    uint64_t mid[WIDTH];
    mining_midstate(header, start, mid);
    uint8_t lows[4][32];
    uint8_t nonce[64];
    std::memcpy(nonce, start, 64);
    for (int k = 0; k < 4; k++) {
        std::memcpy(lows[k], nonce + 32, 32);
        inc_be(nonce);
    }
    uint8_t avx_h[4][64];
    uint8_t all_ff[64];
    std::memset(all_ff, 0xff, 64);
    avx_hash4_impl(mid, lows, all_ff, avx_h);
    std::memcpy(nonce, start, 64);
    int fail = 0;
    for (int k = 0; k < 4; k++) {
        uint8_t sc[64];
        get_nonce_hash(header, nonce, sc);
        if (std::memcmp(avx_h[k], sc, 64) != 0) fail++;
        inc_be(nonce);
    }
    return fail;
}

}  // namespace qpow

#endif  // x86
