#include "qpow/miner.hpp"

namespace qpow {

#if !(defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
bool cpu_has_avx2() { return false; }

SearchResult search_range_avx2(const uint8_t header[32], const uint8_t start[64],
                               uint64_t count, const uint8_t target[64]) {
    return search_range_scalar(header, start, count, target);
}

int test_avx2_field() { return 0; }
int test_avx2_hash_parity() { return 0; }
#endif

SearchResult search_range(const uint8_t header[32], const uint8_t start[64], uint64_t count,
                          const uint8_t target[64], Isa isa) {
    bool use_avx = false;
    if (isa == Isa::Avx2) use_avx = cpu_has_avx2();
    else if (isa == Isa::Auto) use_avx = cpu_has_avx2();
    if (use_avx) return search_range_avx2(header, start, count, target);
    return search_range_scalar(header, start, count, target);
}

}  // namespace qpow
