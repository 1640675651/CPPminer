#pragma once

#include "case32_layout.hpp"

#include <cstdint>
#include <vector>

namespace case32 {

void pack_a_panel(const int8_t *a, int row0, int k_base, int K, int8_t *a_tile,
                  bool offset_a128);

void pack_b_panel(const int8_t *b, int col0, int k_base, int K, int8_t *b_tile);

void prepack_a_all(const int8_t *a, int M, int K, int blocks_k, bool offset_a128,
                   std::vector<int8_t> *out);

void prepack_b_all(const int8_t *b, int N, int K, int blocks_k, int tile_cols,
                   std::vector<int8_t> *out);

// Macro-block layout: within each (im,jm,kb,kg), micro-tiles are contiguous for wave loads.
void prepack_a_coalesced(const int8_t *a, int M, int K, int blocks_k, int macro_rows,
                         bool offset_a128, std::vector<int8_t> *out);
void prepack_b_coalesced(const int8_t *b, int N, int K, int blocks_k, int macro_cols,
                         std::vector<int8_t> *out);

void compute_b_compensation(const int8_t *b, int K, int N, std::vector<int32_t> *out);

void compute_b_compensation_milestones(const int8_t *b, int K, int N, int num_milestones,
                                       int milestone_k, std::vector<int32_t> *out);

void reference_macro_gemm(const int8_t *a_pre, const int8_t *b_pre, int32_t *c, int N,
                          int blocks_k, int macro_rows, int macro_cols, bool use_fast_u8s8,
                          const int32_t *b_comp);

void reference_macro_gemm_coalesced(const int8_t *a_pre, const int8_t *b_pre, int32_t *c,
                                    int N, int blocks_k, int macro_rows, int macro_cols,
                                    bool use_fast_u8s8, const int32_t *b_comp);

// Host reference: milestoned naive GEMM + 8x16 tile XOR (Case 3.3 semantics).
void reference_milestone_tile_xor(const int8_t *a, const int8_t *b, uint32_t *tile_xor,
                                  int M, int N, int K, int num_milestones,
                                  int milestone_k, size_t tile_count);

/* Naive signed GEMM milestone XOR for one 8x16 hash tile (pool/proof reference). */
void compute_tile_milestone_xor_naive(const int8_t *a_rowmajor, const int8_t *b_colmajor,
                                      int M, int N, int K, int t_rows, int t_cols,
                                      int num_milestones, int milestone_k,
                                      uint32_t *milestone_xor_out);

void reference_macro_tile_xor_coalesced(const int8_t *a_pre, const int8_t *b_pre,
                                        uint32_t *tile_xor, int N, int blocks_k,
                                        int blocks_per_milestone, int num_milestones,
                                        int macro_rows, int macro_cols, int tile_cols,
                                        size_t tile_count, const int32_t *b_comp_ms,
                                        bool use_fast_u8s8);

} // namespace case32
