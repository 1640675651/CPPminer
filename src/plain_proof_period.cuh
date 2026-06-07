/* Per-period (128×256) cuBLAS GEMM + scattered 8×16 hash-tile jackpot scan. */
#ifndef PLAIN_PROOF_PERIOD_CUH
#define PLAIN_PROOF_PERIOD_CUH

#include "cp_gpu.cuh"
#include "plain_proof_kernel.cuh"
#include "cp_config.h"

#define PP_TILES_PER_PERIOD 256

__device__ __forceinline__ int pp_period_row_base(int local_rp){
    const int base[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
    };
    return base[local_rp & 15];
}

__device__ __forceinline__ int pp_period_col_base(int local_cp){
    const int base[16] = {
        0, 2, 4, 6, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28, 30
    };
    return base[local_cp & 15];
}

/* Fallback when cuBLAS int8 GEMM is unavailable (e.g. Pascal sm_61). */
__global__ void plain_proof_period_gemm_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int K, int R,
    int row_period, int col_period,
    int32_t* __restrict__ C_hist)
{
    const int r = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if(r >= PP_ROW_PERIOD || c >= PP_COL_PERIOD) return;

    const int8_t* a_row = A + (size_t)(row_period * PP_ROW_PERIOD + r) * (size_t)K;
    const int8_t* b_row = B + (size_t)(col_period * PP_COL_PERIOD + c) * (size_t)K;
    const size_t plane = (size_t)PP_ROW_PERIOD * (size_t)PP_COL_PERIOD;

    int32_t cell = 0;
    for(int ll = R; ll <= K; ll += R){
        for(int l = ll - R; l < ll; l += 4){
            cell = pp_dot_i8_rows(cell, a_row, b_row, l, l + 4);
        }
        C_hist[(size_t)(ll / R - 1) * plane + (size_t)r * PP_COL_PERIOD + (size_t)c] = cell;
    }
}

/* One block per period, one thread per hash tile (256 threads/block). */
__global__ void __launch_bounds__(PP_TILES_PER_PERIOD)
plain_proof_period_jackpot_kernel(
    const int32_t* __restrict__ C_hist_batch,
    int K, int R,
    int row_period, int col_period0,
    int M, int N,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    const uint32_t* __restrict__ a_key8,
    int* __restrict__ out_t_rows,
    int* __restrict__ out_t_cols,
    int* __restrict__ found_flag)
{
    const int b = (int)blockIdx.x;
    const int tile = (int)threadIdx.x;
    if(tile >= PP_TILES_PER_PERIOD) return;

    const int local_rp = tile / 16;
    const int local_cp = tile % 16;
    const int row_base = pp_period_row_base(local_rp);
    const int col_base = pp_period_col_base(local_cp);
    const int col_period = col_period0 + b;
    const int t_rows = row_period * PP_ROW_PERIOD + row_base;
    const int t_cols = col_period * PP_COL_PERIOD + col_base;

    const size_t plane = (size_t)PP_ROW_PERIOD * (size_t)PP_COL_PERIOD;
    const size_t hist_per_item = (size_t)(K / R) * plane;
    const int32_t* C_hist = C_hist_batch + (size_t)b * hist_per_item;

    if(*found_flag) return;

    const int num_steps = K / R;
    uint32_t jackpot[PP_JACKPOT_WORDS];
    for(int i = 0; i < PP_JACKPOT_WORDS; i++) jackpot[i] = 0u;

    for(int step = 0; step < num_steps; step++){
        const size_t step_off = (size_t)step * plane;
        uint32_t xored = 0u;
        for(int u = 0; u < PP_HASH_H; u++){
            const int rel_r = row_base + PP_ROW_PAT[u];
            if(row_period * PP_ROW_PERIOD + rel_r >= M) continue;
            for(int v = 0; v < PP_HASH_W; v++){
                const int rel_c = col_base + PP_COL_PAT[v];
                if(col_period * PP_COL_PERIOD + rel_c >= N) continue;
                const int32_t cell = C_hist[step_off
                    + (size_t)rel_r * PP_COL_PERIOD + (size_t)rel_c];
                xored ^= (uint32_t)cell;
            }
        }
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
        *out_t_rows = t_rows;
        *out_t_cols = t_cols;
    }
}

#endif /* PLAIN_PROOF_PERIOD_CUH */
