/* In-kernel jackpot finalize for CUTLASS fused period GEMM (mainloop tail). */
#pragma once

#include "cp_config.h"
#include "cp_gpu.cuh"
#include "scattered_thread_tile.h"

#define CP_CUTLASS_JACKPOT_WORDS 16
#define CP_CUTLASS_JACKPOT_LROT 13

using CutlassJackpotTile = ScatteredThreadTile<128, 256, 8, 16, 8>;

__device__ __forceinline__ uint32_t cp_cutlass_rotl32(uint32_t x, int s)
{
    return (x << s) | (x >> (32 - s));
}

__device__ __forceinline__ void cp_cutlass_jackpot_fold_step(
    uint32_t* jackpot_words, int step, uint32_t partial_xor)
{
    const int tid = step % CP_CUTLASS_JACKPOT_WORDS;
    jackpot_words[tid] =
        cp_cutlass_rotl32(jackpot_words[tid], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor;
}

__device__ __forceinline__ bool cp_cutlass_jackpot_target_ok(
    const uint32_t digest[8], const uint32_t bound[8])
{
    for(int w = 7; w >= 0; w--){
        if(digest[w] < bound[w]) return true;
        if(digest[w] > bound[w]) return false;
    }
    return true;
}

__device__ __forceinline__ void cp_cutlass_tile_origin(
    int row_period, int col_period0, int batch_idx, int thread_idx,
    int* out_t_rows, int* out_t_cols)
{
    const int cta_row0 = row_period * PP_ROW_PERIOD;
    const int cta_col0 = (col_period0 + batch_idx) * PP_COL_PERIOD;
    int row = 0;
    int col = 0;
    CutlassJackpotTile::thread_cell_global(
        cta_row0, cta_col0, thread_idx, 0, 0, row, col);
    *out_t_rows = row;
    *out_t_cols = col;
}

__device__ __forceinline__ void cp_cutlass_jackpot_try(
    const uint32_t jackpot_words[CP_CUTLASS_JACKPOT_WORDS],
    const uint32_t* a_key8,
    const uint32_t bound[8],
    int row_period,
    int col_period0,
    int batch_idx,
    int thread_idx,
    int* found_flag,
    int* out_t_rows,
    int* out_t_cols)
{
    uint32_t msg[CP_CUTLASS_JACKPOT_WORDS];
    for(int i = 0; i < CP_CUTLASS_JACKPOT_WORDS; i++)
        msg[i] = jackpot_words[i];

    uint32_t digest[8];
    b3_compress64(a_key8, msg, digest);

    if(!cp_cutlass_jackpot_target_ok(digest, bound))
        return;

    if(atomicCAS(found_flag, 0, 1) != 0)
        return;

    int t_rows = 0;
    int t_cols = 0;
    cp_cutlass_tile_origin(row_period, col_period0, batch_idx, thread_idx,
                           &t_rows, &t_cols);
    *out_t_rows = t_rows;
    *out_t_cols = t_cols;
}
