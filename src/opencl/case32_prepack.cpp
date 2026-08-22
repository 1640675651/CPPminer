#include "case32_prepack.hpp"

#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace case32 {

void pack_a_panel(const int8_t *a, int row0, int k_base, int K, int8_t *a_tile,
                  bool offset_a128) {
    for (int kg = 0; kg < kKGroups; ++kg) {
        int8_t *dst = a_tile + kg * kKgBytesA;
        for (int r = 0; r < kMR; ++r) {
            for (int ko = 0; ko < kRank; ++ko) {
                int8_t v = a[static_cast<size_t>(row0 + r) * static_cast<size_t>(K) +
                             static_cast<size_t>(k_base + kg * kRank + ko)];
                if (offset_a128) {
                    v = static_cast<int8_t>(static_cast<uint8_t>(v) + 128u);
                }
                dst[r * kRank + ko] = v;
            }
        }
    }
}

void pack_b_panel(const int8_t *b, int col0, int k_base, int K, int8_t *b_tile) {
    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
        for (int kg = 0; kg < kKGroups; ++kg) {
            int8_t *dst = b_tile + (jg * kKGroups + kg) * kKgGroupBytes;
            for (int c = 0; c < kColsPerGroup; ++c) {
                for (int ko = 0; ko < kRank; ++ko) {
                    dst[c * kRank + ko] =
                            b[static_cast<size_t>(k_base + kg * kRank + ko) +
                              static_cast<size_t>(col0 + jg * kColsPerGroup + c) *
                                      static_cast<size_t>(K)];
                }
            }
        }
    }
}

void prepack_a_all(const int8_t *a, int M, int K, int blocks_k, bool offset_a128,
                   std::vector<int8_t> *out) {
    const int tile_rows = M / kMR;
    out->assign(static_cast<size_t>(tile_rows) * static_cast<size_t>(blocks_k) *
                        static_cast<size_t>(kPanelA),
                0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int tr = 0; tr < tile_rows; ++tr) {
        for (int kb = 0; kb < blocks_k; ++kb) {
            const size_t idx =
                    (static_cast<size_t>(tr) * static_cast<size_t>(blocks_k) +
                     static_cast<size_t>(kb)) *
                    static_cast<size_t>(kPanelA);
            pack_a_panel(a, tr * kMR, kb * kKR, K, out->data() + idx, offset_a128);
        }
    }
}

void compute_b_compensation(const int8_t *b, int K, int N, std::vector<int32_t> *out) {
    out->assign(static_cast<size_t>(N), 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += static_cast<int32_t>(
                    b[static_cast<size_t>(k) + static_cast<size_t>(j) * K]);
        }
        (*out)[static_cast<size_t>(j)] = -128 * sum;
    }
}

void compute_b_compensation_milestones(const int8_t *b, int K, int N, int num_milestones,
                                       int milestone_k, std::vector<int32_t> *out) {
    out->assign(static_cast<size_t>(num_milestones) * static_cast<size_t>(N), 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int ms = 0; ms < num_milestones; ++ms) {
        const int k0 = ms * milestone_k;
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = k0; k < k0 + milestone_k; ++k) {
                sum += static_cast<int32_t>(
                        b[static_cast<size_t>(k) + static_cast<size_t>(j) * K]);
            }
            (*out)[static_cast<size_t>(ms) * static_cast<size_t>(N) +
                   static_cast<size_t>(j)] = -128 * sum;
        }
    }
}

uint32_t xor_tile_host(const int32_t *tile_c, int tile_elems) {
    uint32_t x = 0u;
    for (int i = 0; i < tile_elems; ++i) {
        x ^= static_cast<uint32_t>(tile_c[i]);
    }
    return x;
}

void reference_milestone_tile_xor(const int8_t *a, const int8_t *b, uint32_t *tile_xor,
                                  int M, int N, int K, int num_milestones,
                                  int milestone_k, size_t tile_count) {
    const int hash_mr = hash_tile_mr();
    const int hash_nr = hash_tile_nr();
    const int tile_rows = M / hash_mr;
    const int tile_cols = N / hash_nr;
    const int tile_elems = hash_mr * hash_nr;
    std::vector<int32_t> partial(static_cast<size_t>(M) * static_cast<size_t>(N), 0);

    for (int ms = 0; ms < num_milestones; ++ms) {
        const int k0 = ms * milestone_k;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int32_t acc = partial[static_cast<size_t>(i) * static_cast<size_t>(N) + j];
                for (int k = k0; k < k0 + milestone_k; ++k) {
                    acc += static_cast<int32_t>(a[static_cast<size_t>(i) * K + k]) *
                           static_cast<int32_t>(
                                   b[static_cast<size_t>(k) +
                                     static_cast<size_t>(j) * K]);
                }
                partial[static_cast<size_t>(i) * static_cast<size_t>(N) + j] = acc;
            }
        }
        for (int tr = 0; tr < tile_rows; ++tr) {
            for (int tc = 0; tc < tile_cols; ++tc) {
                std::vector<int32_t> tile_c(static_cast<size_t>(tile_elems), 0);
                const int row0 = tr * hash_mr;
                const int col0 = tc * hash_nr;
                for (int j = 0; j < hash_nr; ++j) {
                    for (int i = 0; i < hash_mr; ++i) {
                        tile_c[static_cast<size_t>(j) * static_cast<size_t>(hash_mr) +
                               static_cast<size_t>(i)] =
                                partial[static_cast<size_t>(row0 + i) * N + (col0 + j)];
                    }
                }
                const size_t spatial_id =
                        static_cast<size_t>(tr) * static_cast<size_t>(tile_cols) +
                        static_cast<size_t>(tc);
                tile_xor[static_cast<size_t>(ms) * tile_count + spatial_id] =
                        xor_tile_host(tile_c.data(), tile_elems);
            }
        }
    }
}

void prepack_b_all(const int8_t *b, int N, int K, int blocks_k, int tile_cols,
                   std::vector<int8_t> *out) {
    out->assign(static_cast<size_t>(tile_cols) * static_cast<size_t>(blocks_k) *
                        static_cast<size_t>(kPanelB),
                0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int tc = 0; tc < tile_cols; ++tc) {
        for (int kb = 0; kb < blocks_k; ++kb) {
            const size_t idx =
                    (static_cast<size_t>(tc) * static_cast<size_t>(blocks_k) +
                     static_cast<size_t>(kb)) *
                    static_cast<size_t>(kPanelB);
            pack_b_panel(b, tc * kNR, kb * kKR, K, out->data() + idx);
        }
    }
}

void prepack_a_coalesced(const int8_t *a, int M, int K, int blocks_k, int macro_rows,
                         bool offset_a128, std::vector<int8_t> *out) {
    const int tile_rows = M / kMR;
    out->assign(static_cast<size_t>(tile_rows) * static_cast<size_t>(blocks_k) *
                        static_cast<size_t>(kPanelA),
                0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int im = 0; im < macro_rows; ++im) {
        const int tr0 = im * kMicroPerMacroM;
        for (int kb = 0; kb < blocks_k; ++kb) {
            for (int kg = 0; kg < kKGroups; ++kg) {
                for (int tr = 0; tr < kMicroPerMacroM; ++tr) {
                    const int tr_global = tr0 + tr;
                    const size_t dst =
                            (static_cast<size_t>(im) * static_cast<size_t>(blocks_k) +
                             static_cast<size_t>(kb)) *
                                    static_cast<size_t>(kMacroKbBlockA) +
                            static_cast<size_t>(kg) * static_cast<size_t>(kMacroKgStripA) +
                            static_cast<size_t>(tr) * static_cast<size_t>(kKgBytesA);
                    std::vector<int8_t> tmp(static_cast<size_t>(kPanelA));
                    pack_a_panel(a, tr_global * kMR, kb * kKR, K, tmp.data(), offset_a128);
                    std::memcpy(out->data() + dst, tmp.data() + kg * kKgBytesA, kKgBytesA);
                }
            }
        }
    }
}

void prepack_b_coalesced(const int8_t *b, int N, int K, int blocks_k, int macro_cols,
                         std::vector<int8_t> *out) {
    const int tile_cols = N / kNR;
    out->assign(static_cast<size_t>(tile_cols) * static_cast<size_t>(blocks_k) *
                        static_cast<size_t>(kPanelB),
                0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int jm = 0; jm < macro_cols; ++jm) {
        const int tc0 = jm * kMicroPerMacroN;
        for (int kb = 0; kb < blocks_k; ++kb) {
            for (int kg = 0; kg < kKGroups; ++kg) {
                for (int tc = 0; tc < kMicroPerMacroN; ++tc) {
                    const int tc_global = tc0 + tc;
                    const size_t dst =
                            (static_cast<size_t>(jm) * static_cast<size_t>(blocks_k) +
                             static_cast<size_t>(kb)) *
                                    static_cast<size_t>(kMacroKbBlockB) +
                            static_cast<size_t>(kg) * static_cast<size_t>(kMacroKgStripB) +
                            static_cast<size_t>(tc) * static_cast<size_t>(kKgSliceB);
                    std::vector<int8_t> tmp(static_cast<size_t>(kPanelB));
                    pack_b_panel(b, tc_global * kNR, kb * kKR, K, tmp.data());
                    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
                        std::memcpy(out->data() + dst +
                                            static_cast<size_t>(jg) * kKgGroupBytes,
                                    tmp.data() +
                                            static_cast<size_t>(jg * kKGroups + kg) *
                                                    kKgGroupBytes,
                                    static_cast<size_t>(kKgGroupBytes));
                    }
                }
            }
        }
    }
}

void reference_macro_gemm(const int8_t *a_pre, const int8_t *b_pre, int32_t *c, int N,
                          int blocks_k, int macro_rows, int macro_cols, bool use_fast_u8s8,
                          const int32_t *b_comp) {
    const size_t a_tile_stride =
            static_cast<size_t>(blocks_k) * static_cast<size_t>(kPanelA);
    const size_t b_tile_stride =
            static_cast<size_t>(blocks_k) * static_cast<size_t>(kPanelB);
    const int macro_blocks = macro_cols * macro_rows;

    for (int mb = 0; mb < macro_blocks; ++mb) {
        const int jm = mb / macro_rows;
        const int im = mb % macro_rows;
        const int col0 = jm * kMacroN;
        const int tc0 = jm * kMicroPerMacroN;
        const int row0 = im * kMacroM;
        const int tr0 = im * kMicroPerMacroM;

        for (int lid = 0; lid < kMicroPerMacroM * kMicroPerMacroN; ++lid) {
            const int tr = lid % kMicroPerMacroM;
            const int tc = lid / kMicroPerMacroM;
            const int micro_row0 = row0 + tr * kMR;
            const int micro_col0 = col0 + tc * kNR;
            const int tr_global = tr0 + tr;
            const int tc_global = tc0 + tc;

            const int8_t *a_base =
                    a_pre + static_cast<size_t>(tr_global) * a_tile_stride;
            const int8_t *b_base =
                    b_pre + static_cast<size_t>(tc_global) * b_tile_stride;

            std::vector<int32_t> acc(static_cast<size_t>(kNR * kMR), 0);
            for (int kb = 0; kb < blocks_k; ++kb) {
                const int8_t *a_tile = a_base + static_cast<size_t>(kb) * kPanelA;
                const int8_t *b_tile = b_base + static_cast<size_t>(kb) * kPanelB;
                for (int kg = 0; kg < kKGroups; ++kg) {
                    const int8_t *a_kg = a_tile + static_cast<size_t>(kg) * kKgBytesA;
                    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
                        const int8_t *b_jg =
                                b_tile + (static_cast<size_t>(jg) * kKGroups +
                                          static_cast<size_t>(kg)) *
                                                 static_cast<size_t>(kKgGroupBytes);
                        for (int col = 0; col < kColsPerGroup; ++col) {
                            const int j = jg * kColsPerGroup + col;
                            const int8_t *bp = b_jg + static_cast<size_t>(col) * kRank;
                            for (int i = 0; i < kMR; ++i) {
                                for (int ko = 0; ko < kRank; ++ko) {
                                    acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                        static_cast<size_t>(i)] +=
                                            static_cast<int32_t>(a_kg[i * kRank + ko]) *
                                            static_cast<int32_t>(bp[ko]);
                                }
                            }
                        }
                    }
                }
            }
            if (use_fast_u8s8 && b_comp) {
                for (int j = 0; j < kNR; ++j) {
                    const int32_t comp = b_comp[micro_col0 + j];
                    for (int i = 0; i < kMR; ++i) {
                        acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                            static_cast<size_t>(i)] += comp;
                    }
                }
            }
            for (int j = 0; j < kNR; ++j) {
                for (int i = 0; i < kMR; ++i) {
                    c[(micro_row0 + i) * N + (micro_col0 + j)] =
                            acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                static_cast<size_t>(i)];
                }
            }
        }
    }
}

void reference_macro_gemm_coalesced(const int8_t *a_pre, const int8_t *b_pre, int32_t *c,
                                    int N, int blocks_k, int macro_rows, int macro_cols,
                                    bool use_fast_u8s8, const int32_t *b_comp) {
    const int macro_blocks = macro_cols * macro_rows;

    for (int mb = 0; mb < macro_blocks; ++mb) {
        const int jm = mb / macro_rows;
        const int im = mb % macro_rows;
        const int col0 = jm * kMacroN;
        const int row0 = im * kMacroM;

        for (int lid = 0; lid < kMicroPerMacroM * kMicroPerMacroN; ++lid) {
#if CASE32_WI_ROWMAJOR
            const int tr = lid / kMicroPerMacroN;
            const int tc = lid % kMicroPerMacroN;
#else
            const int tr = lid % kMicroPerMacroM;
            const int tc = lid / kMicroPerMacroM;
#endif
            const int micro_row0 = row0 + tr * kMR;
            const int micro_col0 = col0 + tc * kNR;

            std::vector<int32_t> acc(static_cast<size_t>(kNR * kMR), 0);
            for (int kb = 0; kb < blocks_k; ++kb) {
                const size_t a_kb_base =
                        (static_cast<size_t>(im) * static_cast<size_t>(blocks_k) +
                         static_cast<size_t>(kb)) *
                        static_cast<size_t>(kMacroKbBlockA);
                const size_t b_kb_base =
                        (static_cast<size_t>(jm) * static_cast<size_t>(blocks_k) +
                         static_cast<size_t>(kb)) *
                        static_cast<size_t>(kMacroKbBlockB);
                for (int kg = 0; kg < kKGroups; ++kg) {
                    const int8_t *a_kg =
                            a_pre + a_kb_base +
                            static_cast<size_t>(kg) * static_cast<size_t>(kMacroKgStripA) +
                            static_cast<size_t>(tr) * static_cast<size_t>(kKgBytesA);
                    const int8_t *b_kg =
                            b_pre + b_kb_base +
                            static_cast<size_t>(kg) * static_cast<size_t>(kMacroKgStripB) +
                            static_cast<size_t>(tc) * static_cast<size_t>(kKgSliceB);
                    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
                        const int8_t *b_jg =
                                b_kg + static_cast<size_t>(jg) * kKgGroupBytes;
                        for (int col = 0; col < kColsPerGroup; ++col) {
                            const int j = jg * kColsPerGroup + col;
                            const int8_t *bp = b_jg + static_cast<size_t>(col) * kRank;
                            for (int i = 0; i < kMR; ++i) {
                                for (int ko = 0; ko < kRank; ++ko) {
                                    acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                        static_cast<size_t>(i)] +=
                                            static_cast<int32_t>(a_kg[i * kRank + ko]) *
                                            static_cast<int32_t>(bp[ko]);
                                }
                            }
                        }
                    }
                }
            }
            if (use_fast_u8s8 && b_comp) {
                for (int j = 0; j < kNR; ++j) {
                    const int32_t comp = b_comp[micro_col0 + j];
                    for (int i = 0; i < kMR; ++i) {
                        acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                            static_cast<size_t>(i)] += comp;
                    }
                }
            }
            for (int j = 0; j < kNR; ++j) {
                for (int i = 0; i < kMR; ++i) {
                    c[(micro_row0 + i) * N + (micro_col0 + j)] =
                            acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                static_cast<size_t>(i)];
                }
            }
        }
    }
}

void reference_macro_tile_xor_coalesced(const int8_t *a_pre, const int8_t *b_pre,
                                        uint32_t *tile_xor, int N, int blocks_k,
                                        int blocks_per_milestone, int num_milestones,
                                        int macro_rows, int macro_cols, int tile_cols,
                                        size_t tile_count, const int32_t *b_comp_ms,
                                        bool use_fast_u8s8) {
    const int macro_blocks = macro_cols * macro_rows;
    const int register_tiles_per_hash = hash_tile_nr() / kNR;

    for (int mb = 0; mb < macro_blocks; ++mb) {
        const int jm = mb / macro_rows;
        const int im = mb % macro_rows;
        const int col0 = jm * kMacroN;
        const int tr0 = im * kMicroPerMacroM;
        const int tc0 = jm * kMicroPerMacroN;

        for (int lid = 0; lid < kMicroPerMacroM * kMicroPerMacroN; ++lid) {
#if CASE32_WI_ROWMAJOR
            const int tr = lid / kMicroPerMacroN;
            const int tc = lid % kMicroPerMacroN;
#else
            const int tr = lid % kMicroPerMacroM;
            const int tc = lid / kMicroPerMacroM;
#endif
            const int micro_col0 = col0 + tc * kNR;
            const int tr_global = tr0 + tr;
            const int tc_global = tc0 + tc;
            const int hash_tc_global = tc_global / register_tiles_per_hash;
            const size_t spatial_id =
                    static_cast<size_t>(tr_global) * static_cast<size_t>(tile_cols) +
                    static_cast<size_t>(hash_tc_global);

            std::vector<int32_t> tile_c(static_cast<size_t>(kNR * kMR), 0);
            std::vector<int32_t> panel_acc(static_cast<size_t>(kNR * kMR), 0);
            int ms = 0;

            for (int kb = 0; kb < blocks_k; ++kb) {
                const size_t a_kb_base =
                        (static_cast<size_t>(im) * static_cast<size_t>(blocks_k) +
                         static_cast<size_t>(kb)) *
                        static_cast<size_t>(kMacroKbBlockA);
                const size_t b_kb_base =
                        (static_cast<size_t>(jm) * static_cast<size_t>(blocks_k) +
                         static_cast<size_t>(kb)) *
                        static_cast<size_t>(kMacroKbBlockB);
                for (int kg = 0; kg < kKGroups; ++kg) {
                    const int8_t *a_kg =
                            a_pre + a_kb_base +
                            static_cast<size_t>(kg) * static_cast<size_t>(kMacroKgStripA) +
                            static_cast<size_t>(tr) * static_cast<size_t>(kKgBytesA);
                    const int8_t *b_kg =
                            b_pre + b_kb_base +
                            static_cast<size_t>(kg) * static_cast<size_t>(kMacroKgStripB) +
                            static_cast<size_t>(tc) * static_cast<size_t>(kKgSliceB);
                    for (int jg = 0; jg < kNR / kColsPerGroup; ++jg) {
                        const int8_t *b_jg =
                                b_kg + static_cast<size_t>(jg) * kKgGroupBytes;
                        for (int col = 0; col < kColsPerGroup; ++col) {
                            const int j = jg * kColsPerGroup + col;
                            const int8_t *bp = b_jg + static_cast<size_t>(col) * kRank;
                            for (int i = 0; i < kMR; ++i) {
                                for (int ko = 0; ko < kRank; ++ko) {
                                    panel_acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                              static_cast<size_t>(i)] +=
                                            static_cast<int32_t>(a_kg[i * kRank + ko]) *
                                            static_cast<int32_t>(bp[ko]);
                                }
                            }
                        }
                    }
                }

                if (use_fast_u8s8 && b_comp_ms) {
                    for (int j = 0; j < kNR; ++j) {
                        const int32_t comp =
                                b_comp_ms[static_cast<size_t>(ms) * static_cast<size_t>(N) +
                                          static_cast<size_t>(micro_col0 + j)];
                        for (int i = 0; i < kMR; ++i) {
                            tile_c[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                   static_cast<size_t>(i)] +=
                                    panel_acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                              static_cast<size_t>(i)] + comp;
                        }
                    }
                } else {
                    for (int j = 0; j < kNR; ++j) {
                        for (int i = 0; i < kMR; ++i) {
                            tile_c[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                   static_cast<size_t>(i)] +=
                                    panel_acc[static_cast<size_t>(j) * static_cast<size_t>(kMR) +
                                              static_cast<size_t>(i)];
                        }
                    }
                }
                uint32_t &hash_xor =
                        tile_xor[static_cast<size_t>(ms) * tile_count + spatial_id];
                const uint32_t register_xor = xor_tile_host(tile_c.data(), kNR * kMR);
                if ((tc % register_tiles_per_hash) == 0) {
                    hash_xor = register_xor;
                } else {
                    hash_xor ^= register_xor;
                }
                std::fill(panel_acc.begin(), panel_acc.end(), 0);
                ++ms;
            }
            (void)blocks_per_milestone;
            (void)num_milestones;
        }
    }
}

} // namespace case32
