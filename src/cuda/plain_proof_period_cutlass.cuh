/* Period scan jackpot from CUTLASS fused tile_xor (one hash tile per thread). */
#ifndef PLAIN_PROOF_PERIOD_CUTLASS_CUH
#define PLAIN_PROOF_PERIOD_CUTLASS_CUH

#include "cp_config.h"
#include "cutlass/cp_cutlass_layout.h"
#include "plain_proof_kernel.cuh"

using CutlassScatteredTile = cp_cutlass::CutlassScatteredTile128x256;

__device__ __forceinline__ void cp_cutlass_tile_origin(
    int row_period, int col_period0, int batch_idx, int thread_idx,
    int* out_t_rows, int* out_t_cols)
{
    const int cta_row0 = row_period * PP_ROW_PERIOD;
    const int cta_col0 = (col_period0 + batch_idx) * PP_COL_PERIOD;
    int row = 0;
    int col = 0;
    CutlassScatteredTile::thread_cell_global(cta_row0, cta_col0, thread_idx, 0,
                                           0, row, col);
    *out_t_rows = row;
    *out_t_cols = col;
}

/*
 * One block per hash tile (256 per 128x256 period plane).
 * tile_xor layout: [step * tiles_per_batch + tile_id] uint32 partial XORs.
 */
__global__ void plain_proof_period_cutlass_jackpot_kernel(
    const uint32_t* __restrict__ tile_xor,
    size_t tiles_per_batch,
    int batch_count,
    int num_steps,
    int row_period, int col_period0,
    int M, int N,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    const uint32_t* __restrict__ a_key8,
    int* __restrict__ out_t_rows,
    int* __restrict__ out_t_cols,
    int* __restrict__ found_flag)
{
    const int tile_id = (int)blockIdx.x;
    const int tiles_in_launch = batch_count * (int)CutlassScatteredTile::kThreadsPerCta;
    if(tile_id >= tiles_in_launch) return;
    if(*found_flag) return;

    const int batch_idx = tile_id / CutlassScatteredTile::kThreadsPerCta;
    const int thread_idx = tile_id % CutlassScatteredTile::kThreadsPerCta;

    uint32_t jackpot[PP_JACKPOT_WORDS];
    for(int i = 0; i < PP_JACKPOT_WORDS; i++) jackpot[i] = 0u;

    for(int step = 0; step < num_steps; step++){
        const size_t idx =
            (size_t)step * tiles_per_batch + (size_t)tile_id;
        const uint32_t xored = tile_xor[idx];
        const int tid = step % PP_JACKPOT_WORDS;
        jackpot[tid] = pp_rotl32(jackpot[tid], PP_LROT) ^ xored;
    }

    uint32_t msg[PP_JACKPOT_WORDS];
    for(int i = 0; i < PP_JACKPOT_WORDS; i++) msg[i] = jackpot[i];

    uint32_t digest[8];
    b3_compress64(a_key8, msg, digest);

    bool ok = false;
    uint32_t tgt[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    for(int w = 7; w >= 0; w--){
        if(digest[w] < tgt[w]){ ok = true; break; }
        if(digest[w] > tgt[w]){ ok = false; break; }
        if(w == 0) ok = true;
    }

    if(ok && atomicCAS(found_flag, 0, 1) == 0){
        int t_rows = 0;
        int t_cols = 0;
        cp_cutlass_tile_origin(row_period, col_period0, batch_idx, thread_idx,
                               &t_rows, &t_cols);
        (void)M;
        (void)N;
        *out_t_rows = t_rows;
        *out_t_cols = t_cols;
    }
}

#endif /* PLAIN_PROOF_PERIOD_CUTLASS_CUH */
