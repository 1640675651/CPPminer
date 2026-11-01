#include "case5_blake3.hpp"

#include <cstdio>

namespace case5_ngen {

bool verify_case5_tile_blake3(const std::vector<uint32_t> &tile_xor,
                              const std::vector<uint32_t> &blake3_out, int output_ms,
                              int tile_count, const uint32_t key8[kBlake3KeyWords],
                              const std::vector<uint8_t> *active) {
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
    size_t skipped = 0;
    for (int sid = 0; sid < tile_count; ++sid) {
        if (active && !(*active)[static_cast<size_t>(sid)]) {
            // A tile without any element inside M x N belongs to a thread row/column the kernel may
            // skip entirely (gemmstone's empty-tile epilogue bypass); accept an unwritten digest there.
            bool unwritten = true;
            for (int w = 0; w < kBlake3DigestWords && unwritten; ++w)
                unwritten = blake3_out[static_cast<size_t>(sid) * kBlake3DigestWords + w] == 0;
            if (unwritten) { ++skipped; continue; }
        }
        for (int w = 0; w < kBlake3JackpotWords; ++w) {
            // tile_xor is tile-major: the 16 folded words of a tile are contiguous.
            msg[w] = tile_xor[static_cast<size_t>(sid) * kBlake3JackpotWords + static_cast<size_t>(w)];
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
    if (skipped)
        std::fprintf(stderr, "Case5.6 BLAKE3 verify: %zu padded tiles left unwritten by the kernel (ok)%c", skipped, 10);
    return true;
}

} // namespace case5_ngen
