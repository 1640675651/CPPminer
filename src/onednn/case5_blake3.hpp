#pragma once

// Case 5.6: keyed BLAKE3 single-block compress on 16 folded tile_xor words.
// Matches Pearl CPPminer cp_jackpot.hpp / cp_onednn_jackpot.cl (b3_compress64).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace case5_ngen {

constexpr int kBlake3JackpotWords = 16;
constexpr int kBlake3DigestWords = 8;
constexpr int kBlake3KeyWords = 8;

inline uint32_t blake3_rotl32(uint32_t x, int s) {
    return (x << s) | (x >> (32 - s));
}

inline void blake3_g(uint32_t *v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    auto rotr = [](uint32_t w, int n) { return (w >> n) | (w << (32 - n)); };
    v[a] += v[b] + x;
    v[d] = rotr(v[d] ^ v[a], 16);
    v[c] += v[d];
    v[b] = rotr(v[b] ^ v[c], 12);
    v[a] += v[b] + y;
    v[d] = rotr(v[d] ^ v[a], 8);
    v[c] += v[d];
    v[b] = rotr(v[b] ^ v[c], 7);
}

inline void blake3_compress64(const uint32_t key8[kBlake3KeyWords],
                             const uint32_t msg16[kBlake3JackpotWords],
                             uint32_t out8[kBlake3DigestWords]) {
    static const uint32_t kIV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    uint32_t v[16] = {key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7],
                      kIV[0],  kIV[1],  kIV[2],  kIV[3],  0,       0,       64u,     0x1Bu};
    uint32_t m[16];
    std::memcpy(m, msg16, sizeof(m));
    static const uint8_t kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    for (int round = 0; round < 7; ++round) {
        blake3_g(v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(v, 3, 7, 11, 15, m[6], m[7]);
        blake3_g(v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(v, 3, 4, 9, 14, m[14], m[15]);
        if (round < 6) {
            uint32_t t[16];
            for (int i = 0; i < 16; ++i) {
                t[i] = m[kPerm[i]];
            }
            std::memcpy(m, t, sizeof(m));
        }
    }
    for (int i = 0; i < 8; ++i) {
        out8[i] = v[i] ^ v[i + 8];
    }
}

// Default test key (CASE5_BLAKE3_KEY unset): deterministic non-zero pattern.
inline void blake3_default_key(uint32_t key8[kBlake3KeyWords]) {
    for (int i = 0; i < kBlake3KeyWords; ++i) {
        key8[i] = 0xA5A5A5A5u ^ static_cast<uint32_t>(i * 0x9E3779B9u);
    }
}

inline void blake3_parse_key_env(uint32_t key8[kBlake3KeyWords]) {
    blake3_default_key(key8);
    const char *v = std::getenv("CASE5_BLAKE3_KEY");
    if (!v || v[0] == '\0') {
        return;
    }
    unsigned long parsed = 0;
    if (std::sscanf(v, "%lx", &parsed) == 1) {
        for (int i = 0; i < kBlake3KeyWords; ++i) {
            key8[i] = static_cast<uint32_t>(parsed) ^ static_cast<uint32_t>(i * 0x9E3779B9u);
        }
    }
}

// blake3_out layout: digest[spatial_id * 8 + w], w in [0,7].
bool verify_case5_tile_blake3(const std::vector<uint32_t> &tile_xor,
                              const std::vector<uint32_t> &blake3_out, int output_ms,
                              int tile_count, const uint32_t key8[kBlake3KeyWords]);

} // namespace case5_ngen
