#include "case5_blake3.hpp"

#include <cstdio>

namespace case5_ngen {

bool verify_case5_tile_blake3(const std::vector<uint32_t> &tile_xor,
                              const std::vector<uint32_t> &blake3_out, int output_ms,
                              int tile_count, const uint32_t key8[kBlake3KeyWords]) {
    if (output_ms != kBlake3JackpotWords) {
        std::fprintf(stderr, "Case5.6 BLAKE3 verify: output_ms=%d (expected %d)\n", output_ms,
                     kBlake3JackpotWords);
        return false;
    }
    const size_t need_xor = static_cast<size_t>(output_ms) * static_cast<size_t>(tile_count);
    const size_t need_out = static_cast<size_t>(tile_count) * kBlake3DigestWords;
    if (tile_xor.size() < need_xor || blake3_out.size() < need_out) {
        std::fprintf(stderr, "Case5.6 BLAKE3 verify: buffer size tile_xor=%zu blake3=%zu\n",
                     tile_xor.size(), blake3_out.size());
        return false;
    }

    uint32_t msg[kBlake3JackpotWords];
    uint32_t expect[kBlake3DigestWords];
    for (int sid = 0; sid < tile_count; ++sid) {
        for (int w = 0; w < kBlake3JackpotWords; ++w) {
            msg[w] = tile_xor[static_cast<size_t>(w) * static_cast<size_t>(tile_count) + sid];
        }
        blake3_compress64(key8, msg, expect);
        for (int w = 0; w < kBlake3DigestWords; ++w) {
            const uint32_t got =
                    blake3_out[static_cast<size_t>(sid) * kBlake3DigestWords + w];
            if (got != expect[w]) {
                std::fprintf(stderr,
                             "Case5.6 BLAKE3 mismatch tile=%d word=%d: got=0x%08x expect=0x%08x\n",
                             sid, w, got, expect[w]);
                std::fprintf(stderr, "  digest expect:");
                for (int j = 0; j < kBlake3DigestWords; ++j) {
                    std::fprintf(stderr, " %08x", expect[j]);
                }
                std::fprintf(stderr, "\n  digest got:   ");
                for (int j = 0; j < kBlake3DigestWords; ++j) {
                    std::fprintf(stderr, " %08x",
                                 blake3_out[static_cast<size_t>(sid) * kBlake3DigestWords + j]);
                }
                std::fprintf(stderr, "\n");
                return false;
            }
        }
    }
    return true;
}

} // namespace case5_ngen
