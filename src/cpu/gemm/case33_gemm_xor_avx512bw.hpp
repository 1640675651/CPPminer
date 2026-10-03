#pragma once

#include <cstddef>
#include <cstdint>

/* Base AVX-512 (AVX512F + AVX512BW, no VNNI) ukernel, compiled with -mavx512f -mavx512bw
 * -mavx512vl (MSVC /arch:AVX512). Call only after runtime CPUID confirms AVX512F/BW/VL and
 * XCR0 ZMM/opmask state (such CPUs always have AVX2).
 *
 * Two vertically adjacent 8x16 tiles (a_base0 rows r..r+7, a_base1 rows r+8..r+15, same B
 * panel / columns) computed as one 16x16 zmm block sharing every B broadcast, exactly like
 * case33_avx512vnni_micro_gemm_xor_fused_k_x2 but with vpmaddubsw + vpmaddwd + vpaddd per
 * rank-4 update instead of vpdpbusd. Writes tile_xor_out[ms * tile_count + spatial_tile_id0]
 * and [... + spatial_tile_id1]. Exact s8s8 mode runs the AVX2 kernel once per tile. Single
 * tiles (pair disabled, odd tail) use the AVX2 kernel from the dispatcher. */
void case33_avx512bw_micro_gemm_xor_fused_k_x2(
        const std::int8_t *a_base0, const std::int8_t *a_base1, const std::int8_t *b_base,
        int blocks_k, int blocks_per_milestone, int num_milestones, int N, int global_col0,
        std::size_t spatial_tile_id0, std::size_t spatial_tile_id1, std::size_t tile_count,
        const std::int32_t *b_comp_ms, bool use_fast_u8s8, bool xor_after_milestone,
        std::uint32_t *tile_xor_out);
