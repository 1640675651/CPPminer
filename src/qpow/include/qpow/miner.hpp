#pragma once

#include "poseidon2.hpp"

#include <cstdint>
#include <cstring>

namespace qpow {

inline void inc_be(uint8_t n[64]) {
    for (int i = 63; i >= 0; --i) {
        if (++n[i] != 0) break;
    }
}

struct SearchResult {
    bool found = false;
    uint8_t nonce[64]{};
    uint8_t hash[64]{};
    uint64_t hashes = 0;
};

enum class Isa { Auto, Scalar, Avx2 };

bool cpu_has_avx2();
SearchResult search_range_avx2(const uint8_t header[32], const uint8_t start[64],
                               uint64_t count, const uint8_t target[64]);
int test_avx2_field();
int test_avx2_hash_parity();

inline SearchResult search_range_scalar(const uint8_t header[32], const uint8_t start[64],
                                        uint64_t count, const uint8_t target[64]) {
    SearchResult r{};
    uint8_t nonce[64];
    std::memcpy(nonce, start, 64);
    uint64_t mid[WIDTH];
    mining_midstate(header, nonce, mid);
    uint8_t high[32];
    std::memcpy(high, nonce, 32);
    for (uint64_t i = 0; i < count; i++) {
        if (std::memcmp(nonce, high, 32) != 0) {
            std::memcpy(high, nonce, 32);
            mining_midstate(header, high, mid);
        }
        r.hashes++;
        uint8_t hash[64];
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

SearchResult search_range(const uint8_t header[32], const uint8_t start[64], uint64_t count,
                          const uint8_t target[64], Isa isa);

}  // namespace qpow
