#pragma once

#include <cstddef>
#include <cstdint>

/* AArch64 DotProd entry. The implementation is compiled in a dedicated
 * ARMv8.2+DotProd translation unit and called only after runtime detection. */
void case33_dotprod_micro_gemm_xor_fused_k(
        const std::int8_t *a_base, const std::int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, std::size_t spatial_tile_id,
        std::size_t tile_count, bool xor_after_milestone, std::uint32_t *tile_xor_out);
