// Scattered output tile for one thread in a SIMT int8 epilogue (DefaultThreadMapSimt).
//
// Parameterized by CTA shape and epilogue visit counts so Case 7 (128x128) and Case 7.1
// (128x256) share the same coordinate logic. Each thread owns
// kEpilogueSteps * kFragsPerStep int32 C cells visited in epilogue order.

#pragma once

#include <cstddef>
#include <cstdint>

template <int kThreadblockM_, int kThreadblockN_, int kEpilogueSteps_,
          int kFragsPerStep_, int kColumnGroups_, int kThreadsPerCta_ = 256,
          int kScatteredColStride_ = 32, int kScatteredRowBandStride_ = 8>
struct ScatteredThreadTile {
  static constexpr int kThreadblockM = kThreadblockM_;
  static constexpr int kThreadblockN = kThreadblockN_;
  static constexpr int kEpilogueSteps = kEpilogueSteps_;
  static constexpr int kFragsPerStep = kFragsPerStep_;
  static constexpr int kColumnGroups = kColumnGroups_;
  static constexpr int kThreadsPerCta = kThreadsPerCta_;
  static constexpr int kScatteredColStride = kScatteredColStride_;
  static constexpr int kScatteredRowBandStride = kScatteredRowBandStride_;
  static constexpr int kCellsPerThread = kEpilogueSteps * kFragsPerStep;
  static constexpr int kRowGroupsPerStep = kFragsPerStep / kColumnGroups;

  static_assert(kFragsPerStep % kColumnGroups == 0,
                "kFragsPerStep must be a multiple of kColumnGroups");
  static_assert(kThreadblockM * kThreadblockN ==
                    kThreadsPerCta * kCellsPerThread,
                "one scattered tile per thread");

#if defined(__CUDA_ARCH__)
#define SCATTERED_TILE_FN __device__ __forceinline__
#else
#define SCATTERED_TILE_FN inline
#endif

  SCATTERED_TILE_FN static int thread_row_base(int thread_idx) {
    int const warp = thread_idx / 32;
    return (warp / 2) * 32 + (warp % 2) * 4;
  }

  SCATTERED_TILE_FN static int thread_lane(int thread_idx) {
    return thread_idx % 32;
  }

  SCATTERED_TILE_FN static int advance_epilogue_row_anchor(int row_base, int step) {
    int state[3] = {0, 0, 0};
    int row = row_base;
    for (int s = 0; s < step; ++s) {
      state[0] += 1;
      row += 1;
      if (state[0] == 4) {
        state[0] = 0;
        state[1] += 1;
        row += 12;
        if (state[1] == 2) {
          state[1] = 0;
          state[2] += 1;
          row += 32;
          if (state[2] == 1) {
            state[2] = 0;
            row += 16;
          }
        }
      }
    }
    return row;
  }

  SCATTERED_TILE_FN static void frag_offset(int frag_idx, int &drow, int &dcol) {
    int const column_idx = frag_idx % kColumnGroups;
    int residual = frag_idx / kColumnGroups;
    int const group_idx = residual % kRowGroupsPerStep;
    int const tile_idx = residual / kRowGroupsPerStep;
    drow = group_idx * kScatteredRowBandStride + tile_idx;
    dcol = column_idx * kScatteredColStride;
  }

  SCATTERED_TILE_FN static void thread_cell_local(int thread_idx, int step,
                                                   int frag_idx, int &row,
                                                   int &col) {
    int const row_base = thread_row_base(thread_idx);
    int const col_base = thread_lane(thread_idx);
    int drow = 0;
    int dcol = 0;
    frag_offset(frag_idx, drow, dcol);
    row = advance_epilogue_row_anchor(row_base, step) + drow;
    col = col_base + dcol;
  }

  SCATTERED_TILE_FN static void thread_cell_global(int cta_row0, int cta_col0,
                                                    int thread_idx, int step,
                                                    int frag_idx, int &row,
                                                    int &col) {
    thread_cell_local(thread_idx, step, frag_idx, row, col);
    row += cta_row0;
    col += cta_col0;
  }

#undef SCATTERED_TILE_FN

  static inline std::size_t cta_count_per_row(int N) {
    return static_cast<std::size_t>(N / kThreadblockN);
  }

  static inline std::size_t tile_count(int M, int N) {
    return static_cast<std::size_t>(M / kThreadblockM) * cta_count_per_row(N) *
           static_cast<std::size_t>(kThreadsPerCta);
  }

  static inline std::size_t thread_tile_id(int cta_row0, int cta_col0, int N,
                                           int thread_idx) {
    int const cta_tr = cta_row0 / kThreadblockM;
    int const cta_tc = cta_col0 / kThreadblockN;
    std::size_t const cta_id =
        static_cast<std::size_t>(cta_tr) * cta_count_per_row(N) +
        static_cast<std::size_t>(cta_tc);
    return cta_id * static_cast<std::size_t>(kThreadsPerCta) +
           static_cast<std::size_t>(thread_idx);
  }
};

// Case 7: 128x128 CTA, 8 epilogue steps x 8 frags (4 column groups x 2 row bands).
using ScatteredThreadTile128x128 =
    ScatteredThreadTile<128, 128, 8, 8, 4>;

namespace scattered_thread_tile {
constexpr int kCellsPerThread = ScatteredThreadTile128x128::kCellsPerThread;
constexpr int kEpilogueSteps = ScatteredThreadTile128x128::kEpilogueSteps;
constexpr int kFragsPerStep = ScatteredThreadTile128x128::kFragsPerStep;
constexpr int kScatteredColStride = ScatteredThreadTile128x128::kScatteredColStride;
constexpr int kScatteredRowBandStride =
    ScatteredThreadTile128x128::kScatteredRowBandStride;
constexpr int kThreadblockM = ScatteredThreadTile128x128::kThreadblockM;
constexpr int kThreadblockN = ScatteredThreadTile128x128::kThreadblockN;
constexpr int kThreadsPerCta = ScatteredThreadTile128x128::kThreadsPerCta;
inline std::size_t cta_count_per_row(int N) {
  return ScatteredThreadTile128x128::cta_count_per_row(N);
}
inline std::size_t tile_count(int M, int N) {
  return ScatteredThreadTile128x128::tile_count(M, N);
}
inline std::size_t thread_tile_id(int cta_row0, int cta_col0, int N,
                                  int thread_idx) {
  return ScatteredThreadTile128x128::thread_tile_id(cta_row0, cta_col0, N,
                                                    thread_idx);
}
inline void thread_cell_global(int cta_row0, int cta_col0, int thread_idx,
                               int step, int frag_idx, int &row, int &col) {
  ScatteredThreadTile128x128::thread_cell_global(cta_row0, cta_col0, thread_idx,
                                                 step, frag_idx, row, col);
}
}  // namespace scattered_thread_tile
