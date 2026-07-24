// MMA accumulator lane partition for SIMT int8 GemmShape<128,128,32> / WarpShape<32,64,32>.
// FragmentC is 64 cells as a Cartesian 8x8 hash tile, but spatially interleaved as four
// 4x4 blocks (row stride 16, col stride 32) — LaneMmaShape<4,4> + Policy WarpShape<4,8>.
// Case 9: 4 warps in M x 2 warps in N. Proof offsets: rows [0,1,2,3,16..19], cols [0,1,2,3,32..35].

#pragma once

#include <cstdint>

struct MmaLaneTile128x128 {
  static constexpr int kThreadblockM = 128;
  static constexpr int kThreadblockN = 128;
  static constexpr int kThreadsPerCta = 256;
  static constexpr int kCellsPerThread = 64;
  static constexpr int kHashH = 8;
  static constexpr int kHashW = 8;
  static constexpr int kWarpsM = 4;
  static constexpr int kWarpsN = 2;

#if defined(__CUDA_ARCH__)
#define MMA_LANE_TILE_FN __device__ __forceinline__
#else
#define MMA_LANE_TILE_FN inline
#endif

  /* Origin (min row, min col) of this thread's interleaved 8x8 hash tile within the CTA. */
  MMA_LANE_TILE_FN static void thread_block_origin(int thread_idx, int& row0,
                                                   int& col0) {
    int const warp = thread_idx / 32;
    int const lane = thread_idx % 32;
    int const warp_m = warp % kWarpsM;
    int const warp_n = warp / kWarpsM;

    /* ColumnMajorInterleaved<2>::inverse({4,8}, lane); LaneMmaShape spacing is 4. */
    int const col_major = lane / 8;
    int const residual = lane % 8;
    int const group_m = residual / 2;
    int const group_n = col_major * 2 + (residual % 2);

    row0 = warp_m * 32 + group_m * 4;
    col0 = warp_n * 64 + group_n * 4;
  }

  MMA_LANE_TILE_FN static void thread_cell_global(int cta_row0, int cta_col0,
                                                  int thread_idx, int& row,
                                                  int& col) {
    int row0 = 0;
    int col0 = 0;
    thread_block_origin(thread_idx, row0, col0);
    row = cta_row0 + row0;
    col = cta_col0 + col0;
  }

#undef MMA_LANE_TILE_FN
};
