#pragma once

#include <cstdint>
#include <cstring>

/* Host keyed BLAKE3 single-block compress (matches device b3_compress64). */
namespace cp_jackpot {

inline uint32_t rotl32(uint32_t x, int s) {
    return (x << s) | (x >> (32 - s));
}

inline void b3_g(uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
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

inline void b3_compress64(const uint32_t* key8, const uint32_t* msg16, uint32_t* out8) {
    static const uint32_t kIV[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    uint32_t v[16] = {
        key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7],
        kIV[0],  kIV[1],  kIV[2],  kIV[3],  0,        0,        64u,     0x1Bu};
    uint32_t m[16];
    std::memcpy(m, msg16, sizeof(m));
    static const uint8_t kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    for (int round = 0; round < 7; ++round) {
        b3_g(v, 0, 4, 8, 12, m[0], m[1]);
        b3_g(v, 1, 5, 9, 13, m[2], m[3]);
        b3_g(v, 2, 6, 10, 14, m[4], m[5]);
        b3_g(v, 3, 7, 11, 15, m[6], m[7]);
        b3_g(v, 0, 5, 10, 15, m[8], m[9]);
        b3_g(v, 1, 6, 11, 12, m[10], m[11]);
        b3_g(v, 2, 7, 8, 13, m[12], m[13]);
        b3_g(v, 3, 4, 9, 14, m[14], m[15]);
        if (round < 6) {
            uint32_t t[16];
            for (int i = 0; i < 16; ++i) t[i] = m[kPerm[i]];
            std::memcpy(m, t, sizeof(m));
        }
    }
    for (int i = 0; i < 8; ++i) out8[i] = v[i] ^ v[i + 8];
}

constexpr int kJackpotWords = 16;
constexpr int kLrot = 13;

/* Fold milestone tile XORs into jackpot words (plain_proof / Case 3.3 layout). */
inline void fold_milestones(const uint32_t* milestone_xor, int num_milestones,
                            uint32_t out_msg[kJackpotWords]) {
    for (int i = 0; i < kJackpotWords; ++i) out_msg[i] = 0u;
    for (int step = 0; step < num_milestones; ++step) {
        const int tid = step % kJackpotWords;
        out_msg[tid] = rotl32(out_msg[tid], kLrot) ^ milestone_xor[step];
    }
}

inline bool digest_beats_target(const uint32_t digest[8], const uint32_t bound[8]) {
    for (int w = 7; w >= 0; --w) {
        if (digest[w] < bound[w]) return true;
        if (digest[w] > bound[w]) return false;
    }
    return true;
}

inline bool tile_beats_target(const uint32_t* milestone_xor, int num_milestones,
                              const uint32_t a_key8[8], const uint32_t bound[8]) {
    uint32_t msg[kJackpotWords];
    fold_milestones(milestone_xor, num_milestones, msg);
    uint32_t digest[8];
    b3_compress64(a_key8, msg, digest);
    return digest_beats_target(digest, bound);
}

} // namespace cp_jackpot
