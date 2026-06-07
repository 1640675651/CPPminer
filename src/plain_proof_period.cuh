/* Per-period (128×256) cuBLAS GEMM + scattered 8×16 hash-tile jackpot scan. */
#ifndef PLAIN_PROOF_PERIOD_CUH
#define PLAIN_PROOF_PERIOD_CUH

#include "cp_gpu.cuh"
#include "plain_proof_kernel.cuh"
#include "cp_config.h"

#define PP_TILES_PER_PERIOD 256
#define PP_PERIOD_PLANE_ELEMS (PP_ROW_PERIOD * PP_COL_PERIOD)

/* C_hist[step][row][batch_col] with batch_col = period_in_batch * 256 + col. */
__device__ __forceinline__ size_t pp_c_hist_index(
    int step, int rel_r, int batch_idx, int rel_c, int batch_count)
{
    const int batch_cols = batch_count * PP_COL_PERIOD;
    const size_t step_plane = (size_t)PP_ROW_PERIOD * (size_t)batch_cols;
    return (size_t)step * step_plane
         + (size_t)rel_r * (size_t)batch_cols
         + (size_t)batch_idx * (size_t)PP_COL_PERIOD
         + (size_t)rel_c;
}

__device__ __forceinline__ size_t pp_c_hist_step_plane_elems(int batch_count)
{
    return (size_t)PP_ROW_PERIOD * (size_t)batch_count * (size_t)PP_COL_PERIOD;
}

/* C += prev (element-wise). */
__global__ void plain_proof_plane_add_kernel(
    int32_t* __restrict__ C,
    const int32_t* __restrict__ prev,
    int n)
{
    const int i = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if(i < n) C[i] += prev[i];
}

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
    int batch_idx, int batch_count,
    int32_t* __restrict__ C_hist)
{
    const int r = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if(r >= PP_ROW_PERIOD || c >= PP_COL_PERIOD) return;

    const int8_t* a_row = A + (size_t)(row_period * PP_ROW_PERIOD + r) * (size_t)K;
    const int8_t* b_row = B + (size_t)(col_period * PP_COL_PERIOD + c) * (size_t)K;

    int32_t cell = 0;
    for(int ll = R; ll <= K; ll += R){
        for(int l = ll - R; l < ll; l += 4){
            cell = pp_dot_i8_rows(cell, a_row, b_row, l, l + 4);
        }
        const int step = ll / R - 1;
        C_hist[pp_c_hist_index(step, r, batch_idx, c, batch_count)] = cell;
    }
}

/* One block per period, one thread per hash tile (256 threads/block). */
__global__ void __launch_bounds__(PP_TILES_PER_PERIOD)
plain_proof_period_jackpot_kernel(
    const int32_t* __restrict__ C_hist,
    int batch_count,
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

    if(*found_flag) return;

    const int num_steps = K / R;
    uint32_t jackpot[PP_JACKPOT_WORDS];
    for(int i = 0; i < PP_JACKPOT_WORDS; i++) jackpot[i] = 0u;

    for(int step = 0; step < num_steps; step++){
        uint32_t xored = 0u;
        for(int u = 0; u < PP_HASH_H; u++){
            const int rel_r = row_base + PP_ROW_PAT[u];
            if(row_period * PP_ROW_PERIOD + rel_r >= M) continue;
            for(int v = 0; v < PP_HASH_W; v++){
                const int rel_c = col_base + PP_COL_PAT[v];
                if(col_period * PP_COL_PERIOD + rel_c >= N) continue;
                const int32_t cell = C_hist[
                    pp_c_hist_index(step, rel_r, b, rel_c, batch_count)];
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
