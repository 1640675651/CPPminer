#pragma once

#include <cstddef>
#include <cstdint>

/* AVX-VNNI ukernel entry (compiled with /arch:AVX2 + __AVXVNNI__ or -mavxvnni).
 * Call only after runtime CPUID confirms AVX2 + AVX-VNNI. */
void case33_avxvnni_micro_gemm_xor_fused_k(
        const std::int8_t *a_base, const std::int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, int N, int global_col0,
        std::size_t spatial_tile_id, std::size_t tile_count, const std::int32_t *b_comp_ms,
        bool use_fast_u8s8, bool xor_after_milestone, std::uint32_t *tile_xor_out);
