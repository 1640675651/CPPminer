#pragma once
/*
 * OpenVINO graph of one Intel NPU launch: the exact int8 GEMM of a 512-row panel of noisy A with a
 * 512-column block of noisy B over all of K, with the int32 accumulator XOR-reduced per 8x16 tile
 * after every K step of 128 (the pearl milestones). Built from the gemm_xor_benchmark NPU design:
 * per-panel int32 GEMMs on the DPU feed the TileXorAcc SHAVE kernel of the patched compiler, which
 * keeps the running sum and XORs every tile in one pass.
 *
 * Inputs (u8, value + 128, which the compiler folds into a cast to int8):
 *   0  a_panels   [K/128, rows, 128]   A rows of the launch, one 128-wide K panel after another
 *   1  bt_panels  [K/128, cols, 128]   B columns (rows of B^T), same panel order
 * Output: int32 [rows/8, (K/128) * cols/16], the tile words of each tile row in launch order;
 * word_index() maps (tile column, milestone) to the word's position within a tile row.
 */

#include <memory>
#include <vector>

#include <openvino/core/model.hpp>

namespace cp_npu {

constexpr int kTileRows = 8;
constexpr int kTileCols = 16;
constexpr int kKr = 128;

struct LaunchShape {
    int rows = 512;     /* one accumulator row block: B blocks are then used by exactly one GEMM */
    int cols = 512;     /* 8 column chains of 64, run in lockstep (A panels freed after use) */
    int k = 4096;
    int chunk_n = 64;   /* columns per accumulator chain */
    int group = 8;      /* panel GEMMs per TileXorAcc op */

    int milestones() const { return k / kKr; }
    int tile_rows() const { return rows / kTileRows; }
    int tile_cols() const { return cols / kTileCols; }
    int row_words() const { return milestones() * tile_cols(); }
};

/* Builds the launch model. word_index receives tile_cols * milestones entries: entry
 * tc * milestones + ms is the position of tile column tc's word for milestone ms in a tile row. */
std::shared_ptr<ov::Model> build_launch_model(const LaunchShape& shape, std::vector<int>* word_index);

} // namespace cp_npu
