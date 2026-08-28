#pragma once

#include "cp_jackpot.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace case5_ngen {

// Per-thread XOR sub-tile geometry (see CASE5_XOR_TILE.md).
struct XorSubtileDims {
    int subM = 0;
    int subN = 0;
    int subGridM = 1;
    int subGridN = 1;
    int ownedM = 0;
    int ownedN = 0;
    bool split = false;
};

enum class XorSubtileSplitMode {
    Native,   // no split — one XOR per full unroll panel (debug / Core Ultra verify)
    Vertical, // halve M only until subM*subN <= 256 (e.g. 32x32 -> 4x 8x32)
    Both,     // halve M and N until both <= 16 (legacy 2D split)
};

inline XorSubtileSplitMode case5_xor_subtile_split_mode() {
    static const XorSubtileSplitMode mode = [] {
        const char *v = std::getenv("CASE5_XOR_SPLIT");
        if (!v || v[0] == '\0') {
            return XorSubtileSplitMode::Both;
        }
        if (std::strcmp(v, "native") == 0 || std::strcmp(v, "none") == 0
                || std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0) {
            return XorSubtileSplitMode::Native;
        }
        if (std::strcmp(v, "vertical") == 0) {
            return XorSubtileSplitMode::Vertical;
        }
        if (std::strcmp(v, "both") == 0 || std::strcmp(v, "2d") == 0) {
            return XorSubtileSplitMode::Both;
        }
        return XorSubtileSplitMode::Both;
    }();
    return mode;
}

inline const char *case5_xor_subtile_split_mode_name() {
    switch (case5_xor_subtile_split_mode()) {
        case XorSubtileSplitMode::Native: return "native";
        case XorSubtileSplitMode::Both: return "both";
        case XorSubtileSplitMode::Vertical: return "vertical";
    }
    return "both";
}

// One XOR word per thread over the full unrollM x unrollN panel (no sub-tiling).
inline void case5_native_xor_subtile_dims(int unrollM, int unrollN, XorSubtileDims &d) {
    d.ownedM = unrollM;
    d.ownedN = unrollN;
    d.subM = unrollM;
    d.subN = unrollN;
    d.subGridM = 1;
    d.subGridN = 1;
    d.split = false;
}

// Halve each unroll dimension until both <= 16.
inline void case5_halve_xor_subtile_dims_both(int unrollM, int unrollN, XorSubtileDims &d) {
    d.ownedM = unrollM;
    d.ownedN = unrollN;
    d.subM = unrollM;
    d.subN = unrollN;
    d.split = true;
    while (d.subM > 16) {
        d.subM /= 2;
    }
    while (d.subN > 16) {
        d.subN /= 2;
    }
    d.subGridM = unrollM / d.subM;
    d.subGridN = unrollN / d.subN;
}

// Halve M only until each sub-tile has at most 256 elements (subN stays full unrollN).
inline void case5_halve_xor_subtile_dims_vertical(int unrollM, int unrollN, XorSubtileDims &d) {
    d.ownedM = unrollM;
    d.ownedN = unrollN;
    d.subM = unrollM;
    d.subN = unrollN;
    d.split = true;
    while (d.subM * d.subN > 256) {
        d.subM /= 2;
    }
    d.subGridM = unrollM / d.subM;
    d.subGridN = 1;
}

// Each thread owns the full unrollM x unrollN register panel. If the panel has fewer than
// 256 elements, one XOR word per thread; otherwise 2D-split sub-tiles (both <= 16) by default.
inline XorSubtileDims compute_xor_subtile_dims(int unrollM, int unrollN) {
    XorSubtileDims d;
    if (case5_xor_subtile_split_mode() == XorSubtileSplitMode::Native) {
        case5_native_xor_subtile_dims(unrollM, unrollN, d);
        return d;
    }
    d.ownedM = unrollM;
    d.ownedN = unrollN;
    if (unrollM * unrollN < 256) {
        d.subM = unrollM;
        d.subN = unrollN;
        d.subGridM = 1;
        d.subGridN = 1;
        d.split = false;
        return d;
    }
    if (case5_xor_subtile_split_mode() == XorSubtileSplitMode::Both) {
        case5_halve_xor_subtile_dims_both(unrollM, unrollN, d);
    } else {
        case5_halve_xor_subtile_dims_vertical(unrollM, unrollN, d);
    }
    return d;
}

// Map logical tile (idM, idN) to global C origin — full unroll panel per thread.
inline void case5_thread_c_origin(int id_m, int id_n, int unroll_m, int unroll_n, int &row0,
                                  int &col0) {
    row0 = id_m * unroll_m;
    col0 = id_n * unroll_n;
}

// Set CASE5_DUMP_TILE_XOR=1 to print tile_xor[ms * tile_count + lr * tile_cols + lc] after readback.
// Optional CASE5_DUMP_TILE_XOR_VERBOSE=1 lists every non-zero slot with (ms, lr, lc, idx).
inline bool case5_dump_tile_xor_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CASE5_DUMP_TILE_XOR");
        return v && v[0] != '0';
    }();
    return on;
}

inline bool case5_dump_tile_xor_verbose() {
    static const bool on = [] {
        const char *v = std::getenv("CASE5_DUMP_TILE_XOR_VERBOSE");
        return v && v[0] != '0';
    }();
    return on;
}

inline void dump_case5_tile_xor(const std::vector<uint32_t> &got, int num_ms, int tile_rows,
                                int tile_cols, int tile_count, uint32_t expect_match = 0,
                                bool mark_expected = false) {
    if (!case5_dump_tile_xor_enabled()) {
        return;
    }

    std::fprintf(stderr,
                 "Case5 tile_xor dump: tile_rows=%d tile_cols=%d tile_count=%d milestones=%d "
                 "size=%zu",
                 tile_rows, tile_cols, tile_count, num_ms, got.size());
    if (mark_expected) {
        std::fprintf(stderr, " expect=0x%08x (O=match X=wrong .=zero)", expect_match);
    } else {
        std::fprintf(stderr, " (.=zero +=nonzero)");
    }
    std::fprintf(stderr, "\n");

    for (int ms = 0; ms < num_ms; ++ms) {
        std::fprintf(stderr, "--- ms=%d (idx base %d) ---\n", ms, ms * tile_count);
        for (int lr = 0; lr < tile_rows; ++lr) {
            std::fputc(' ', stderr);
            for (int lc = 0; lc < tile_cols; ++lc) {
                const size_t idx = static_cast<size_t>(ms) * static_cast<size_t>(tile_count) +
                                   static_cast<size_t>(lr) * static_cast<size_t>(tile_cols) +
                                   static_cast<size_t>(lc);
                const uint32_t v = idx < got.size() ? got[idx] : 0u;
                char mark = '.';
                if (mark_expected) {
                    if (v == 0u) {
                        mark = '.';
                    } else if (v == expect_match) {
                        mark = 'O';
                    } else {
                        mark = 'X';
                    }
                } else if (v != 0u) {
                    mark = '+';
                }
                std::fputc(mark, stderr);
            }
            std::fputc('\n', stderr);
        }
    }

    if (case5_dump_tile_xor_verbose()) {
        std::fprintf(stderr, "non-zero entries:\n");
        for (size_t idx = 0; idx < got.size(); ++idx) {
            const uint32_t v = got[idx];
            if (v == 0u) {
                continue;
            }
            const size_t ms = idx / static_cast<size_t>(tile_count);
            const size_t rem = idx % static_cast<size_t>(tile_count);
            const size_t lr = rem / static_cast<size_t>(tile_cols);
            const size_t lc = rem % static_cast<size_t>(tile_cols);
            std::fprintf(stderr, "  idx=%zu ms=%zu lr=%zu lc=%zu val=0x%08x\n", idx, ms, lr, lc,
                         v);
        }
    }

    std::fflush(stderr);
}

inline bool verify_case5_tile_xor(const std::vector<uint32_t> &got, const int8_t *a,
                                  const int8_t *b, int M, int N, int K, int num_ms,
                                  int milestone_k, int unroll_m, int unroll_n, int /*wg_m*/,
                                  int /*wg_n*/, bool /*coop_split_a*/, bool /*coop_split_b*/,
                                  int sub_m, int sub_n, int sub_grid_m, int sub_grid_n,
                                  int tile_rows, int tile_cols, int tile_count) {
    std::vector<int32_t> c(static_cast<size_t>(M) * static_cast<size_t>(N), 0);

    const int active_tr = (M + unroll_m - 1) / unroll_m;
    const int active_tc = (N + unroll_n - 1) / unroll_n;

    for (int ms = 0; ms < num_ms; ++ms) {
        const int k_end = (ms + 1) * milestone_k;
        std::fill(c.begin(), c.end(), 0);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int32_t s = 0;
                for (int k = 0; k < k_end; ++k) {
                    s += int32_t(a[static_cast<size_t>(i) * K + k]) *
                         int32_t(b[static_cast<size_t>(k) * N + j]);
                }
                c[static_cast<size_t>(i) * N + j] = s;
            }
        }

        for (int tr = 0; tr < active_tr; ++tr) {
            for (int tc = 0; tc < active_tc; ++tc) {
                const int id_m = tr;
                const int id_n = tc;

                int row0 = 0;
                int col0 = 0;
                case5_thread_c_origin(id_m, id_n, unroll_m, unroll_n, row0, col0);

                for (int sr = 0; sr < sub_grid_m; ++sr) {
                    for (int sc = 0; sc < sub_grid_n; ++sc) {
                        uint32_t x = 0;
                        bool any_in_bounds = false;
                        for (int i = 0; i < sub_m; ++i) {
                            for (int j = 0; j < sub_n; ++j) {
                                const int gr = row0 + sr * sub_m + i;
                                const int gc = col0 + sc * sub_n + j;
                                if (gr < 0 || gr >= M || gc < 0 || gc >= N) {
                                    continue;
                                }
                                any_in_bounds = true;
                                const int32_t v = c[static_cast<size_t>(gr) * N + gc];
                                uint32_t u;
                                std::memcpy(&u, &v, sizeof(u));
                                x ^= u;
                            }
                        }
                        if (!any_in_bounds) {
                            continue;
                        }
                        const int lr = id_m * sub_grid_m + sr;
                        const int lc = id_n * sub_grid_n + sc;
                        if (lr < 0 || lr >= tile_rows || lc < 0 || lc >= tile_cols) {
                            continue;
                        }
                        const size_t idx =
                                static_cast<size_t>(ms) * static_cast<size_t>(tile_count) +
                                static_cast<size_t>(lr) * static_cast<size_t>(tile_cols) +
                                static_cast<size_t>(lc);
                        if (idx >= got.size() || got[idx] != x) {
                            std::fprintf(stderr,
                                         "tile_xor mismatch ms=%d idM=%d idN=%d sr=%d sc=%d "
                                         "lr=%d lc=%d got=0x%08x expected=0x%08x\n",
                                         ms, id_m, id_n, sr, sc, lr, lc,
                                         idx < got.size() ? got[idx] : 0u, x);
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace case5_ngen
