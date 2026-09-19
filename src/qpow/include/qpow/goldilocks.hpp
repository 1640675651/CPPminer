#pragma once

#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace qpow {

#if defined(_MSC_VER)
#define QPOW_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define QPOW_INLINE inline __attribute__((always_inline))
#else
#define QPOW_INLINE inline
#endif

constexpr uint64_t P64 = 0xFFFFFFFF00000001ull;
constexpr uint64_t EPS64 = 0xFFFFFFFFull;
constexpr int WIDTH = 12;

// Official qp-poseidon-core add: overflowing_add, then +EPS on carry, rare
// second +EPS. Result may be in [P, 2^64). The common path is branchless.
QPOW_INLINE uint64_t gf_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    uint64_t c1 = (uint64_t)(sum < a);
    uint64_t sum2 = sum + (c1 * EPS64);
    if (sum2 < sum) sum2 += EPS64;
    return sum2;
}

QPOW_INLINE uint64_t gf_canon(uint64_t a) { return a >= P64 ? a - P64 : a; }

QPOW_INLINE void mul64x64(uint64_t a, uint64_t b, uint64_t* lo, uint64_t* hi) {
#if defined(_MSC_VER) && defined(_M_X64)
    *lo = _umul128(a, b, hi);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 p = (unsigned __int128)a * b;
    *lo = (uint64_t)p;
    *hi = (uint64_t)(p >> 64);
#else
    uint64_t a_lo = (uint32_t)a, a_hi = a >> 32;
    uint64_t b_lo = (uint32_t)b, b_hi = b >> 32;
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    uint64_t mid = (p0 >> 32) + (uint32_t)p1 + (uint32_t)p2;
    *lo = (p0 & EPS64) | (mid << 32);
    *hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
#endif
}

// Official reduce128: rare borrow branch, then add/sbb-style wrap (+0 or +EPS).
QPOW_INLINE uint64_t gf_reduce(uint64_t lo, uint64_t hi) {
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & EPS64;
    uint64_t t0 = lo - hi_hi;
    if (lo < hi_hi) t0 -= EPS64;
    uint64_t t1 = (hi_lo << 32) - hi_lo;
    uint64_t t2 = t0 + t1;
    return t2 + ((uint64_t)(t2 < t0) * EPS64);
}

QPOW_INLINE uint64_t gf_mul(uint64_t a, uint64_t b) {
    uint64_t lo, hi;
    mul64x64(a, b, &lo, &hi);
    return gf_reduce(lo, hi);
}

QPOW_INLINE uint64_t gf_sqr(uint64_t a) { return gf_mul(a, a); }

QPOW_INLINE uint64_t gf_sbox(uint64_t x) {
    uint64_t x2 = gf_sqr(x);
    uint64_t x3 = gf_mul(x2, x);
    uint64_t x4 = gf_sqr(x2);
    return gf_mul(x3, x4);
}

}  // namespace qpow
