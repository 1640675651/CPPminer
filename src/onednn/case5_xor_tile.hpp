#pragma once

#include "cp_jackpot.hpp"

#include <cctype>
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

// Device matrix layouts (C always column-major N). Names: TN, TT, NT, NN (A then B).
// Host A is M×K row-major; host B^T is N×K row-major. Device packing converts layout.
inline bool case5_rowmajor_from_env(const char *env_name, bool default_row) {
    const char *v = std::getenv(env_name);
    if (!v || v[0] == '\0') {
        return default_row;
    }
    if (std::strcmp(v, "col") == 0 || std::strcmp(v, "column") == 0 || std::strcmp(v, "N") == 0
            || std::strcmp(v, "n") == 0) {
        return false;
    }
    return true;
}

inline bool case5_resolve_rowmajor(int cli_layout, const char *env_name, bool default_row) {
    if (cli_layout < 0) {
        return case5_rowmajor_from_env(env_name, default_row);
    }
    return cli_layout == 0;
}

inline const char *case5_layout_trans_name(bool row_major) {
    return row_major ? "T" : "N";
}

inline void case5_gemm_layout_name(bool a_row_major, bool b_row_major, char *out, size_t out_sz) {
    std::snprintf(out, out_sz, "%s%s", case5_layout_trans_name(a_row_major),
                  case5_layout_trans_name(b_row_major));
}

inline const char *case5_device_layout_name(bool a_row_major, bool b_row_major) {
    static thread_local char buf[8];
    case5_gemm_layout_name(a_row_major, b_row_major, buf, sizeof(buf));
    return buf;
}

inline bool case5_parse_gemm_layout_env(bool &a_row_major, bool &b_row_major) {
    const char *v = std::getenv("CASE5_GEMM_LAYOUT");
    if (!v || v[0] == '\0') {
        return false;
    }
    char a = '\0';
    char b = '\0';
    if (std::strlen(v) >= 3 && (v[2] == 'N' || v[2] == 'n')) {
        a = static_cast<char>(std::toupper(static_cast<unsigned char>(v[0])));
        b = static_cast<char>(std::toupper(static_cast<unsigned char>(v[1])));
    } else if (std::strlen(v) == 2) {
        a = static_cast<char>(std::toupper(static_cast<unsigned char>(v[0])));
        b = static_cast<char>(std::toupper(static_cast<unsigned char>(v[1])));
    } else {
        return false;
    }
    if ((a != 'T' && a != 'N') || (b != 'T' && b != 'N')) {
        return false;
    }
    a_row_major = (a == 'T');
    b_row_major = (b == 'T');
    return true;
}

inline bool case5_parse_gemm_layout_name(const char *name, bool &a_row_major, bool &b_row_major) {
    if (!name || name[0] == '\0') {
        return false;
    }
    char a = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));
    char b = static_cast<char>(std::toupper(static_cast<unsigned char>(name[1])));
    if ((a != 'T' && a != 'N') || (b != 'T' && b != 'N')) {
        return false;
    }
    if (name[2] != '\0' && name[2] != 'N' && name[2] != 'n') {
        return false;
    }
    a_row_major = (a == 'T');
    b_row_major = (b == 'T');
    return true;
}

inline void case5_resolve_gemm_layouts(int cli_a_layout, int cli_b_layout, bool &a_row_major,
                                       bool &b_row_major) {
    a_row_major = case5_resolve_rowmajor(cli_a_layout, "CASE5_A_LAYOUT", true);
    b_row_major = case5_resolve_rowmajor(cli_b_layout, "CASE5_B_LAYOUT", false);
    bool env_a = a_row_major;
    bool env_b = b_row_major;
    if (case5_parse_gemm_layout_env(env_a, env_b)) {
        if (cli_a_layout < 0) {
            a_row_major = env_a;
        }
        if (cli_b_layout < 0) {
            b_row_major = env_b;
        }
    }
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

// CPU reference for one hash tile: milestone XORs, folded msg, BLAKE3 digest (matches cp_jackpot).
// A is row-major (lda>=K); B is column-major (ldb>=K), same layout as Case33GemmOnednn device buffers.
inline void compute_single_hash_tile_jackpot(
        const int8_t *a, int lda, const int8_t *b_colmajor, int ldb, int M, int N, int K,
        int num_ms, int milestone_k, int unroll_m, int unroll_n, int sub_m, int sub_n,
        int sub_grid_m, int sub_grid_n, int lr, int lc, const uint32_t key8[8],
        uint32_t out_milestones[], uint32_t out_msg[cp_jackpot::kJackpotWords],
        uint32_t out_digest[8]) {
    const int id_m = lr / sub_grid_m;
    const int sr = lr % sub_grid_m;
    const int id_n = lc / sub_grid_n;
    const int sc = lc % sub_grid_n;

    int row0 = 0;
    int col0 = 0;
    case5_thread_c_origin(id_m, id_n, unroll_m, unroll_n, row0, col0);

    const int row0_sub = row0 + sr * sub_m;
    const int col0_sub = col0 + sc * sub_n;
    const int n_cells = sub_m * sub_n;
    std::vector<int32_t> cell_c(static_cast<size_t>(n_cells), 0);
    uint32_t local_ms[64] = {};
    uint32_t *ms_buf = out_milestones ? out_milestones : local_ms;

    for (int ms = 0; ms < num_ms; ++ms) {
        const int k_begin = ms * milestone_k;
        const int k_end = (ms + 1) * milestone_k;
        for (int k = k_begin; k < k_end; ++k) {
            int idx = 0;
            for (int i = 0; i < sub_m; ++i) {
                for (int j = 0; j < sub_n; ++j) {
                    const int gr = row0_sub + i;
                    const int gc = col0_sub + j;
                    if (gr < 0 || gr >= M || gc < 0 || gc >= N) {
                        ++idx;
                        continue;
                    }
                    const int32_t av = static_cast<int32_t>(a[static_cast<size_t>(gr) * lda + k]);
                    const int32_t bv =
                            static_cast<int32_t>(b_colmajor[static_cast<size_t>(gc) * ldb + k]);
                    cell_c[static_cast<size_t>(idx)] += av * bv;
                    ++idx;
                }
            }
        }

        uint32_t x = 0u;
        for (int i = 0; i < n_cells; ++i) {
            uint32_t u = 0u;
            std::memcpy(&u, &cell_c[static_cast<size_t>(i)], sizeof(u));
            x ^= u;
        }
        ms_buf[ms] = x;
    }

    cp_jackpot::fold_milestones(ms_buf, num_ms, out_msg);
    cp_jackpot::b3_compress64(key8, out_msg, out_digest);
}

inline void fprint_jackpot_words_hex(FILE *out, const char *label, const uint32_t *words,
                                     int count) {
    std::fprintf(out, "%s", label);
    for (int i = 0; i < count; ++i) {
        std::fprintf(out, "%08x", words[i]);
    }
    std::fprintf(out, "\n");
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
