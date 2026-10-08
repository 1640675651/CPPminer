#pragma once

#include "nonce_line.hpp"
#include "poseidon2.hpp"

#include <cstdint>
#include <cstring>

namespace qpow {

inline void inc_be(uint8_t n[64]) {
    for (int i = 63; i >= 0; --i) {
        if (++n[i] != 0) break;
    }
}

/* The search functions count 64-byte big-endian values and hash the nonce each one stands for
 * (nonce_line.hpp). nonce is what to submit; counter is the searched value it came from. */
struct SearchResult {
    bool found = false;
    uint8_t nonce[64]{};
    uint8_t counter[64]{};
    uint8_t hash[64]{};
    uint64_t hashes = 0;
};

enum class Isa { Auto, Scalar, Avx2 };

bool cpu_has_avx2();
SearchResult search_range_avx2(const uint8_t header[32], const uint8_t start[64],
                               uint64_t count, const uint8_t target[64]);
int test_avx2_field();
int test_avx2_hash_parity();

/* Launch constants of the nonce line that counter ctr lies on, refreshed only when ctr leaves
 * the cached line (new bytes 0..31 or new k). */
struct LineCache {
    bool valid = false;
    uint8_t high[32]{};
    uint32_t base[8]{};
    uint64_t mid[WIDTH]{};
    uint64_t pk[nonce_line::kParams]{};

    void update(const uint8_t header[32], const uint8_t ctr[64]) {
        uint32_t b[8];
        nonce_line::base_words(ctr, b);
        const bool same_high = valid && std::memcmp(high, ctr, 32) == 0;
        if (same_high && std::memcmp(base, b, sizeof(b)) == 0) return;
        if (!same_high) {
            std::memcpy(high, ctr, 32);
            mining_midstate(header, ctr, mid);
        }
        std::memcpy(base, b, sizeof(b));
        nonce_line::launch_params(mid, ctr, pk);
        valid = true;
    }
};

inline SearchResult search_range_scalar(const uint8_t header[32], const uint8_t start[64],
                                        uint64_t count, const uint8_t target[64]) {
    SearchResult r{};
    uint8_t ctr[64];
    std::memcpy(ctr, start, 64);
    LineCache line;
    uint64_t i = 0;
    while (i < count) {
        /* one stretch of a single line: t .. t + n - 1 without a wrap of t */
        line.update(header, ctr);
        const uint32_t t = nonce_line::t_of(ctr);
        uint64_t n = nonce_line::kTSpan - t;
        if (n > count - i) n = count - i;
        for (uint64_t j = 0; j < n; j++) {
            r.hashes++;
            uint8_t hash[64];
            if (nonce_line::hash_if_valid(line.pk, t + (uint32_t)j, target, hash)) {
                nonce_line::add_be(ctr, j);
                r.found = true;
                nonce_line::map(ctr, r.nonce);
                std::memcpy(r.counter, ctr, 64);
                std::memcpy(r.hash, hash, 64);
                return r;
            }
        }
        nonce_line::add_be(ctr, n);
        i += n;
    }
    return r;
}

SearchResult search_range(const uint8_t header[32], const uint8_t start[64], uint64_t count,
                          const uint8_t target[64], Isa isa);

}  // namespace qpow
