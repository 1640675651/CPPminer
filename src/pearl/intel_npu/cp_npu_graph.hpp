#pragma once
/*
 * OpenVINO graph of one Intel NPU launch: the exact int8 GEMM of a panel of noisy A rows with a block
 * of noisy B columns over all of K, with the int32 accumulator XOR-reduced per 16x16 tile after every
 * K step of 128 (the pearl milestones). Built from the gemm_xor_benchmark NPU design: per-panel int32
 * GEMMs on the DPU feed the TileXorAcc SHAVE kernel of the patched compiler, which keeps the running
 * sum and XORs every tile in one pass.
 *
 * Inputs (A and B as u8, value + 128, which the compiler folds into a cast to int8):
 *   0       a_panels   [K/128, rows, 128]   A rows of the launch, one 128-wide K panel after another
 *   1 .. b  bt_panels  [K/128, cols, 128]   B columns (rows of B^T), same panel order; b_inputs()
 *                                           copies that the caller binds to the same tensor
 *   b + 1   jackpot_params, int32 [1, 1, chunk_m/16, 16], fused_jackpot only: every row the 8 key and
 *           8 bound words of the attempt
 * Output: int32 [rows/16, (K/128) * cols/16], the tile words of each tile row in launch order;
 * word_index() maps (tile column, milestone) to the word's position within a tile row.
 * With fused_jackpot the TileJackpot SHAVE op hashes the tile words on the NPU instead, and the output
 * is int32 [rows/16]: per tile row 1 + the lowest tile column that beats the bound, or 0.
 */

#include <memory>
#include <vector>

#include <openvino/core/model.hpp>

namespace cp_npu {

/* Hash tile. The compiler and kernel also do 8-row tiles; NPU_EXPERIMENTAL_TILE_XOR_ROWS selects the height
 * (set_compiler_environment). 16 halves the tile words the NPU writes and the host hashes. */
constexpr int kTileRows = 16;
constexpr int kTileCols = 16;
constexpr int kKr = 128;
constexpr int kJackpotParamWords = 16; /* key[8], bound[8] */

struct LaunchShape {
    int rows = 512;     /* row blocks of chunk_m, one after another */
    int cols = 512;     /* column chains of chunk_n, run in lockstep within a row block (A panels freed
                           after use); at most 8 chains */
    int k = 4096;
    int chunk_m = 512;  /* rows per accumulator chain */
    int chunk_n = 64;   /* columns per accumulator chain */
    int group = 8;      /* panel GEMMs per TileXorAcc op */
    bool split_b = true; /* one B input per row block: the compiler merges identical slices of one input,
                            so a shared B block would stay in CMX across all row blocks */
    bool fused_jackpot = false; /* BLAKE3 jackpot on the SHAVEs: one hit word per tile row comes back */

    int milestones() const { return k / kKr; }
    int tile_rows() const { return rows / kTileRows; }
    int tile_cols() const { return cols / kTileCols; }
    int row_words() const { return milestones() * tile_cols(); }
    int row_blocks() const { return rows / chunk_m; }
    int b_inputs() const { return split_b ? row_blocks() : 1; }
    int jackpot_input() const { return 1 + b_inputs(); } /* index of jackpot_params (fused_jackpot) */
};

/* Why a shape cannot be built, or nullptr. */
const char* launch_shape_error(const LaunchShape& shape);

/* Builds the launch model. word_index receives tile_cols * milestones entries: entry
 * tc * milestones + ms is the position of tile column tc's word for milestone ms in a tile row. */
std::shared_ptr<ov::Model> build_launch_model(const LaunchShape& shape, std::vector<int>* word_index);

} // namespace cp_npu
