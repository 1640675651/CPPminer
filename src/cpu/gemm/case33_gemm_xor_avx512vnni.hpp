#pragma once

#include <cstddef>
#include <cstdint>

/* AVX512-VNNI ukernel entry (compiled with -mavx512f -mavx512bw -mavx512vl -mavx512vnni,
 * MSVC /arch:AVX512). Call only after runtime CPUID confirms AVX512F/BW/VL/VNNI and
 * XCR0 ZMM/opmask state. Same packed A/B layout, 8x16 hash tile, KR milestone XOR
 * schedule and epilogue as the AVX-VNNI kernel; only the encoding (EVEX, 32 regs) differs. */
void case33_avx512vnni_micro_gemm_xor_fused_k(
        const std::int8_t *a_base, const std::int8_t *b_base, int blocks_k,
        int blocks_per_milestone, int num_milestones, int N, int global_col0,
        std::size_t spatial_tile_id, std::size_t tile_count, const std::int32_t *b_comp_ms,
        bool use_fast_u8s8, bool xor_after_milestone, std::uint32_t *tile_xor_out);

/* zmm variant: two vertically adjacent 8x16 tiles (a_base0 rows r..r+7, a_base1 rows
 * r+8..r+15, same B panel / columns) as one 16x16 block sharing every B broadcast. Writes
 * tile_xor_out[ms * tile_count + spatial_tile_id0] and [... + spatial_tile_id1]. Exact s8s8
 * mode falls back to two single-tile calls. */
void case33_avx512vnni_micro_gemm_xor_fused_k_x2(
        const std::int8_t *a_base0, const std::int8_t *a_base1, const std::int8_t *b_base,
        int blocks_k, int blocks_per_milestone, int num_milestones, int N, int global_col0,
        std::size_t spatial_tile_id0, std::size_t spatial_tile_id1, std::size_t tile_count,
        const std::int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        std::uint32_t *tile_xor_out);

/* True unless CP_AVX512_PAIR=0 (A/B knob: forces the ymm single-tile kernel). */
bool case33_avx512vnni_pair_enabled();
