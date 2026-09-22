#pragma once

#include "goldilocks.hpp"

#include <cstring>

namespace qpow {

inline const uint64_t RC_INTERNAL[22] = {
    0x97f7798a784ad863ull, 0xd1d2bf082f60d4f0ull, 0x69a377a79f9ad206ull,
    0xa9d06906a3858e24ull, 0x295275001eede5b5ull, 0x5874e441117bd746ull,
    0x8a084bbba8ed86ccull, 0x3defd7645cde6425ull, 0x3998cfe6871cc137ull,
    0x3e52ef8bca48314aull, 0x964a209f85dc9eccu, 0x3fcc9ee82cc4577eull,
    0x8e79b4a5d0096d6dull, 0x8492362ad2392556ull, 0xee72f470262574d6ull,
    0x1e0e18496da2444aull, 0x0f3a74bf215eaac6ull, 0x1b061b76a1c0ded3ull,
    0x192c42d86803d7a6ull, 0xf6d49ff997ae0260ull, 0x3ec372e7a0fa3786ull,
    0x5538cdf4f23445d3ull};

inline const uint64_t RC_INITIAL[4][12] = {
    {0xc002e770975b1607ull, 0xbca51a8dfe14593aull, 0x72938dfbe774f7f9ull,
     0xe4f2fe29e03234acull, 0xd5e0ba2f541b6449ull, 0xec33b868f3cc46c1ull,
     0x486dcb55419d475aull, 0x6c1cb2a358cc24f1ull, 0xe3f30d509a1436bbull,
     0xd9a64f068dca7c29ull, 0xe59b3f57aabba1aeull, 0x2a3dd4505b478fdcull},
    {0xada1f8dc7676ed25ull, 0x2711aa8b5509d516ull, 0x4ae6acd0c9c92897ull,
     0x56eb3d6b5256d67aull, 0x1f7a9d55923bf51eull, 0x3600427d397a7f68ull,
     0xe5076df75b72c3d0ull, 0xfcd59aa12c6090adull, 0xcd895e8c68b57a9eull,
     0x41df7ef9d730ae3eull, 0xee3e2b889abe977dull, 0xd29bb7edbeb9c405ull},
    {0x7d5c08eef608e382ull, 0x89ae889caaf0802cull, 0xb35a8e976d2af617ull,
     0xdb14234eafaf5173ull, 0x78f04462d48b1c98ull, 0x265293b0e47ce88aull,
     0x999a649b69b9d32full, 0x64b0a186698e01d3ull, 0xee0b22d0dfae8bb8ull,
     0x4fd53e50ca04a7eeull, 0x5762bfe181f25047ull, 0xf51593e2beb5e3bdull},
    {0x1e5e2b5760e32477ull, 0x622462a1f9aaaeedull, 0xaa284b3ecdb222aeull,
     0x63c8e72f542bf3fcull, 0x3ba588cacb43b5e0ull, 0x23eda6f3c99150ddull,
     0xaad3bea4baac9a5aull, 0xe9da8d699b94184aull, 0xcdb13f4cd93e024cull,
     0x902cbd0956f655e3ull, 0x5b4e40ffc759532full, 0xde795c20a2357af7ull}};

inline const uint64_t RC_TERMINAL[4][12] = {
    {0x7b72c539e0ea4c6eull, 0x144573dae2ce9976ull, 0x802028b68f35fc88ull,
     0x6d36c5022c4fe7c2ull, 0xa205d0ffa9b9def3ull, 0xf6e7e38b1ea6ba2full,
     0x34f7909ae5258d64ull, 0xb0464d9d77b97fcaull, 0x64ddb9d5de7e00a6ull,
     0x0ed0d75c27975d97ull, 0x1cbb36f11127338bull, 0x6673e505cfd0b6baull},
    {0x605f902830872e01ull, 0x3fd5eb927e95fe4full, 0xe81025b5a24c69cdull,
     0xf7d0ce75de23f74eull, 0xf39942b6a8585089ull, 0x6d808a08f7b71df6ull,
     0xf8806b6588f49a8bull, 0x57df2d8c2a32107aull, 0x16e7c2074d654a2dull,
     0x213de241fcf33835ull, 0xb0f2b8905a0976f6ull, 0xd8e3cf2bbd355417ull},
    {0xe498691679d9330full, 0x763b45d2a3821b28ull, 0x0908bf65eb0a1f0dull,
     0x7691eb2d194b24f4ull, 0x0e43551233ae13b2ull, 0x93c393dbfc2fe76full,
     0x98f607485d48cdeaull, 0xe3d95f30309819c0ull, 0x1ef581a93eaf6acfull,
     0x0b24c1b7a030fca4ull, 0x624370be5670b327ull, 0x5f1e28615a11e486ull},
    {0xfe04051f909e042bull, 0x7257e5b147fd3803ull, 0xe6ae134bb82f2e78ull,
     0x5711fd5cf4784511ull, 0xf83a42660c08c0bcull, 0x2cd8c96d9a3ce855ull,
     0x7d2ffb1bb0e17271ull, 0x85ae1528caea3811ull, 0x52a345d5c7adb0b8ull,
     0x504c4c51f3faee94ull, 0xbce34a649cfccaf9ull, 0xe0a3389266fb6dc9ull}};

inline const uint64_t MDS_DIAG[12] = {
    0xc3b6c08e23ba9300ull, 0xd84b5de94a324fb6ull, 0x0d0c371c5b35b84full,
    0x7964f570e7188037ull, 0x5daf18bbd996604bull, 0x6743bc47b9595257ull,
    0x5528b9362c59bb70ull, 0xac45e25b7127b68bull, 0xa2077d7dfbb606b5ull,
    0xf3faac6faee378aeull, 0x0c6388b51545e883ull, 0xd27dbb6944917b60ull};

QPOW_INLINE void apply_mat4(uint64_t x[4]) {
    uint64_t t01 = gf_add(x[0], x[1]);
    uint64_t t23 = gf_add(x[2], x[3]);
    uint64_t t0123 = gf_add(t01, t23);
    uint64_t t01123 = gf_add(t0123, x[1]);
    uint64_t t01233 = gf_add(t0123, x[3]);
    x[3] = gf_add(t01233, gf_add(x[0], x[0]));
    x[1] = gf_add(t01123, gf_add(x[2], x[2]));
    x[0] = gf_add(t01123, t01);
    x[2] = gf_add(t01233, t23);
}

QPOW_INLINE void ext_layer(uint64_t s[WIDTH]) {
    apply_mat4(s);
    apply_mat4(s + 4);
    apply_mat4(s + 8);
    uint64_t sums[4] = {
        gf_add(gf_add(s[0], s[4]), s[8]),
        gf_add(gf_add(s[1], s[5]), s[9]),
        gf_add(gf_add(s[2], s[6]), s[10]),
        gf_add(gf_add(s[3], s[7]), s[11]),
    };
    s[0] = gf_add(s[0], sums[0]);
    s[1] = gf_add(s[1], sums[1]);
    s[2] = gf_add(s[2], sums[2]);
    s[3] = gf_add(s[3], sums[3]);
    s[4] = gf_add(s[4], sums[0]);
    s[5] = gf_add(s[5], sums[1]);
    s[6] = gf_add(s[6], sums[2]);
    s[7] = gf_add(s[7], sums[3]);
    s[8] = gf_add(s[8], sums[0]);
    s[9] = gf_add(s[9], sums[1]);
    s[10] = gf_add(s[10], sums[2]);
    s[11] = gf_add(s[11], sums[3]);
}

QPOW_INLINE void int_layer(uint64_t s[WIDTH]) {
    uint64_t sl = s[0], sh = 0;
    for (int i = 1; i < WIDTH; i++) {
        uint64_t t = sl + s[i];
        sh += (uint64_t)(t < sl);
        sl = t;
    }
    uint64_t sum = gf_reduce(sl, sh);
    for (int i = 0; i < WIDTH; i++) s[i] = gf_add(gf_mul(s[i], MDS_DIAG[i]), sum);
}

QPOW_INLINE void external_round(uint64_t s[WIDTH], const uint64_t rc[WIDTH]) {
    for (int i = 0; i < WIDTH; i++) s[i] = gf_sbox(gf_add(s[i], rc[i]));
    ext_layer(s);
}

inline void permute(uint64_t s[WIDTH]) {
    ext_layer(s);
    for (int r = 0; r < 4; r++) external_round(s, RC_INITIAL[r]);
    for (int r = 0; r < 22; r++) {
        s[0] = gf_sbox(gf_add(s[0], RC_INTERNAL[r]));
        int_layer(s);
    }
    for (int r = 0; r < 4; r++) external_round(s, RC_TERMINAL[r]);
}

QPOW_INLINE void absorb32(uint64_t s[WIDTH], const uint8_t bytes[32]) {
    for (int i = 0; i < 8; i++) {
        uint32_t w;
        std::memcpy(&w, bytes + i * 4, 4);
        s[i] = gf_add(s[i], (uint64_t)w);
    }
}

QPOW_INLINE void squeeze32(uint8_t out[32], const uint64_t s[WIDTH]) {
    for (int i = 0; i < 4; i++) {
        uint64_t c = gf_canon(s[i]);
        std::memcpy(out + i * 8, &c, 8);
    }
}

inline void mining_midstate(const uint8_t header[32], const uint8_t nonce_high[32],
                            uint64_t mid[WIDTH]) {
    std::memset(mid, 0, WIDTH * sizeof(uint64_t));
    absorb32(mid, header);
    permute(mid);
    absorb32(mid, nonce_high);
    permute(mid);
    for (int i = 0; i < WIDTH; i++) mid[i] = gf_canon(mid[i]);
}

inline void hash_squeeze_twice_96(const uint8_t in[96], uint8_t out[64]) {
    uint64_t s[WIDTH]{};
    absorb32(s, in);
    permute(s);
    absorb32(s, in + 32);
    permute(s);
    absorb32(s, in + 64);
    permute(s);
    s[0] = gf_add(s[0], 1);
    s[1] = gf_add(s[1], 1);
    permute(s);
    squeeze32(out, s);
    permute(s);
    squeeze32(out + 32, s);
}

inline void get_nonce_hash(const uint8_t header[32], const uint8_t nonce_be[64],
                           uint8_t hash[64]) {
    uint8_t in[96];
    std::memcpy(in, header, 32);
    std::memcpy(in + 32, nonce_be, 64);
    hash_squeeze_twice_96(in, hash);
}

inline void get_nonce_hash_from_mid(const uint64_t mid[WIDTH], const uint8_t nonce_low[32],
                                    uint8_t hash[64]) {
    uint64_t s[WIDTH];
    std::memcpy(s, mid, WIDTH * sizeof(uint64_t));
    absorb32(s, nonce_low);
    permute(s);
    s[0] = gf_add(s[0], 1);
    s[1] = gf_add(s[1], 1);
    permute(s);
    squeeze32(hash, s);
    permute(s);
    squeeze32(hash + 32, s);
}

inline bool hash_if_valid(const uint64_t mid[WIDTH], const uint8_t nonce_low[32],
                          const uint8_t target[64], uint8_t hash[64]) {
    uint64_t s[WIDTH];
    std::memcpy(s, mid, WIDTH * sizeof(uint64_t));
    absorb32(s, nonce_low);
    permute(s);
    s[0] = gf_add(s[0], 1);
    s[1] = gf_add(s[1], 1);
    permute(s);
    squeeze32(hash, s);
    if (std::memcmp(hash, target, 32) > 0) return false;
    permute(s);
    squeeze32(hash + 32, s);
    return std::memcmp(hash, target, 64) < 0;
}

}  // namespace qpow
