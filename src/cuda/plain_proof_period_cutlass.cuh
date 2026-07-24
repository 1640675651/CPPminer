/* Legacy separate jackpot from tile_xor (unused when Case 9 in-kernel jackpot is on). */
#ifndef PLAIN_PROOF_PERIOD_CUTLASS_CUH
#define PLAIN_PROOF_PERIOD_CUTLASS_CUH

#include "cp_config.h"
#include "cutlass/mma_lane_tile.h"
#include "plain_proof_kernel.cuh"

using CutlassJackpotTile = MmaLaneTile128x128;

__device__ __forceinline__ void cp_cutlass_tile_origin_from_xor(
    int row_period, int col_period0, int batch_idx, int thread_idx,
    int* out_t_rows, int* out_t_cols)
{
    const int cta_row0 = row_period * CP_CUTLASS_CTA_M;
    const int cta_col0 = (col_period0 + batch_idx) * CP_CUTLASS_CTA_N;
    int row = 0;
    int col = 0;
    CutlassJackpotTile::thread_cell_global(cta_row0, cta_col0, thread_idx, row, col);
    *out_t_rows = row;
    *out_t_cols = col;
}

#endif /* PLAIN_PROOF_PERIOD_CUTLASS_CUH */
