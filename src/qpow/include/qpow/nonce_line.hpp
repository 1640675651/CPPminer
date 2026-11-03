#pragma once

/* Structured nonce search ("nonce line").
 *
 * The miners count 64-byte big-endian values ("counters") and hash, for each counter, the nonce
 * map() gives. Bytes 0..31 pass through unchanged (they feed the midstate). Bytes 32..63 are read
 * as a 256-bit counter C = k * 2^26 + t and become the eight little-endian absorb words
 *
 *     w = B(k) + t * v,   B_i = 2^31 + bits [29i, 29i + 29) of k,
 *     v = (u, -u),        u = (4, -17, 11, -3) = column 0 of adj(M4).
 *
 * M_ext * v is zero except lane 0 (-35) and lane 4 (+35), so after the absorb and the first linear
 * layer only those two lanes depend on t: the first full round needs two S-boxes, and the other
 * ten are folded once per k into the launch constants. |17 t| < 2^30.1 keeps every word inside
 * [0, 2^32). Counters of one job differ in k only in its low words, never by a multiple of v, so
 * distinct counters give distinct nonces. Any 64-byte nonce is valid for the pool.
 */

#include "poseidon2.hpp"

#include <cstdint>
#include <cstring>

namespace qpow {
namespace nonce_line {

constexpr int kTBits = 26;
constexpr uint64_t kTSpan = 1ull << kTBits;
constexpr int kV[8] = {4, -17, 11, -3, -4, 17, -11, 3};

/* bits [pos, pos + len) of the 256-bit big-endian number b[0..31] */
inline uint32_t bits256(const uint8_t b[32], int pos, int len) {
    uint32_t r = 0;
    for (int i = len - 1; i >= 0; i--) {
        const int bit = pos + i;
        r = (r << 1) | (bit < 256 ? (uint32_t)(b[31 - bit / 8] >> (bit % 8)) & 1u : 0u);
    }
    return r;
}

inline uint32_t t_of(const uint8_t ctr[64]) { return bits256(ctr + 32, 0, kTBits); }

/* ctr += v (64-byte big-endian) */
inline void add_be(uint8_t ctr[64], uint64_t v) {
    for (int i = 63; i >= 0 && v; i--) {
        v += ctr[i];
        ctr[i] = (uint8_t)v;
        v >>= 8;
    }
}

inline void base_words(const uint8_t ctr[64], uint32_t b[8]) {
    for (int i = 0; i < 8; i++) b[i] = 0x80000000u | bits256(ctr + 32, kTBits + 29 * i, 29);
}

/* The nonce that counter `ctr` stands for. */
inline void map(const uint8_t ctr[64], uint8_t out[64]) {
    uint32_t b[8];
    base_words(ctr, b);
    const uint32_t t = t_of(ctr);
    std::memcpy(out, ctr, 32);
    for (int i = 0; i < 8; i++) {
        const uint32_t w = b[i] + (uint32_t)((int64_t)kV[i] * t);
        std::memcpy(out + 32 + 4 * i, &w, 4);
    }
}

/* Constants for the counters that share ctr's k (mid = midstate of ctr's bytes 0..31):
 * pk[0], pk[1] = lanes 0 and 4 of ext_layer(mid + B) + RC_INITIAL[0] at t = 0 (canonical);
 * pk[2 + i] = K_i, the state entering round 1 with lanes 0 and 4 left out of round 0:
 * ext_layer(sbox of the other ten lanes, zero at 0 and 4) + RC_INITIAL[1] (canonical). */
constexpr int kParams = 14;
inline void launch_params(const uint64_t mid[WIDTH], const uint8_t ctr[64], uint64_t pk[kParams]) {
    uint32_t b[8];
    base_words(ctr, b);
    uint64_t s[WIDTH];
    std::memcpy(s, mid, sizeof(s));
    for (int i = 0; i < 8; i++) s[i] = gf_add(s[i], (uint64_t)b[i]);
    ext_layer(s);
    for (int i = 0; i < WIDTH; i++) s[i] = gf_canon(gf_add(s[i], RC_INITIAL[0][i]));
    pk[0] = s[0];
    pk[1] = s[4];
    uint64_t k[WIDTH];
    for (int i = 0; i < WIDTH; i++) k[i] = (i == 0 || i == 4) ? 0 : gf_sbox(s[i]);
    ext_layer(k);
    for (int i = 0; i < WIDTH; i++) pk[2 + i] = gf_canon(gf_add(k[i], RC_INITIAL[1][i]));
}

/* State entering round 1 (round constants included) for line position t. Column 0 of M_ext is
 * m0 * (2, 1, 1) over the three blocks and column 4 is m0 * (1, 2, 1), m0 = M4 column 0. */
inline void round1_state(const uint64_t pk[kParams], uint32_t t, uint64_t s[WIDTH]) {
    const uint64_t d = 35ull * t;
    const uint64_t f = gf_sbox(gf_add(pk[0], P64 - d));
    const uint64_t g = gf_sbox(gf_add(pk[1], d));
    const uint64_t c2 = gf_add(f, g);
    const uint64_t c[3] = {gf_add(c2, f), gf_add(c2, g), c2};
    const int m0[4] = {2, 1, 1, 3};
    for (int blk = 0; blk < 3; blk++)
        for (int j = 0; j < 4; j++) {
            uint64_t v = c[blk];
            for (int r = 1; r < m0[j]; r++) v = gf_add(v, c[blk]);
            s[4 * blk + j] = gf_add(pk[2 + 4 * blk + j], v);
        }
}

/* Rest of the first permutation from the round-1 state. */
inline void permute_from_round1(uint64_t s[WIDTH]) {
    for (int i = 0; i < WIDTH; i++) s[i] = gf_sbox(s[i]);
    ext_layer(s);
    for (int r = 2; r < 4; r++) external_round(s, RC_INITIAL[r]);
    for (int r = 0; r < 22; r++) {
        s[0] = gf_sbox(gf_add(s[0], RC_INTERNAL[r]));
        int_layer(s);
    }
    for (int r = 0; r < 4; r++) external_round(s, RC_TERMINAL[r]);
}

/* hash_if_valid() for line position t (same result as for the mapped nonce). */
inline bool hash_if_valid(const uint64_t pk[kParams], uint32_t t, const uint8_t target[64],
                          uint8_t hash[64]) {
    uint64_t s[WIDTH];
    round1_state(pk, t, s);
    permute_from_round1(s);
    s[0] = gf_add(s[0], 1);
    s[1] = gf_add(s[1], 1);
    permute(s);
    squeeze32(hash, s);
    if (std::memcmp(hash, target, 32) > 0) return false;
    permute(s);
    squeeze32(hash + 32, s);
    return std::memcmp(hash, target, 64) < 0;
}

}  // namespace nonce_line
}  // namespace qpow
