#pragma once

#include "case33_gemm_xor.hpp"

#include <cstddef>
#include <cstdint>

void case33_ssse3_micro_gemm_xor_fused_k(
        const std::int8_t *a_base, const std::int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, int n, int global_col0,
        std::size_t spatial_tile_id, std::size_t tile_count, const std::int32_t *b_comp_ms,
        bool use_fast_u8s8, Case33SseTile sse_tile, bool xor_after_milestone,
        std::uint32_t *tile_xor_out);
