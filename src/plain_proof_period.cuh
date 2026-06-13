/* Per-period cuBLAS GEMM + cooperative 8x16 hash-tile jackpot (reads C_hist). */
#ifndef PLAIN_PROOF_PERIOD_CUH
#define PLAIN_PROOF_PERIOD_CUH

#include "cp_gpu.cuh"
#include "plain_proof_kernel.cuh"
#include "cp_config.h"

#define PP_TILES_PER_PERIOD 256
#define PP_PERIOD_PLANE_ELEMS (PP_ROW_PERIOD * PP_COL_PERIOD)

/* C_hist fat panel: row_batch*128 rows x col_batch*256 cols per step. */
__device__ __forceinline__ size_t pp_c_hist_panel_index(
    int step, int rel_r, int rel_c, int row_batch_count, int col_batch_count)
{
    const int panel_rows = row_batch_count * PP_ROW_PERIOD;
    const int panel_cols = col_batch_count * PP_COL_PERIOD;
    const size_t step_plane = (size_t)panel_rows * (size_t)panel_cols;
    return (size_t)step * step_plane
         + (size_t)rel_r * (size_t)panel_cols
         + (size_t)rel_c;
}

__device__ __forceinline__ size_t pp_c_hist_step_plane_elems(
    int row_batch_count, int col_batch_count)
{
    return (size_t)(row_batch_count * PP_ROW_PERIOD)
         * (size_t)(col_batch_count * PP_COL_PERIOD);
}

/* Legacy col-only batch layout (row_batch_count=1). */
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

/* C_hist[step] holds rank partials; jackpot cumulates across steps. */

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
    int m, int n, int K, int R,
    int row_period, int col_period,
    int row_in_batch, int col_in_batch,
    int row_batch_count, int col_batch_count,
    int32_t* __restrict__ C_hist)
{
    const int r = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if(r >= PP_ROW_PERIOD || c >= PP_COL_PERIOD) return;

    const int grow = row_period * PP_ROW_PERIOD + r;
    const int gcol = col_period * PP_COL_PERIOD + c;
    const int rel_r = row_in_batch * PP_ROW_PERIOD + r;
    const int rel_c = col_in_batch * PP_COL_PERIOD + c;

    for(int ll = R; ll <= K; ll += R){
        const int step = ll / R - 1;
        const int8_t* a_row = pp_ap_panel_ptr(A, step, grow, m, K, R);
        const int8_t* b_row = pp_bp_panel_ptr(B, step, gcol, n, K, R);
        int32_t cell = pp_dot_i8_panel(0, a_row, b_row, 0, R);
        C_hist[pp_c_hist_panel_index(step, rel_r, rel_c,
                                     row_batch_count, col_batch_count)] = cell;
    }
}

/* One block per hash tile (8x16 threads); reads C_hist rank partials, one BLAKE3/block. */
__global__ void plain_proof_period_jackpot_kernel(
    const int32_t* __restrict__ C_hist,
    int row_batch_count, int col_batch_count,
    int K, int R,
    int row_period0, int col_period0,
    int M, int N,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    const uint32_t* __restrict__ a_key8,
    int* __restrict__ out_t_rows,
    int* __restrict__ out_t_cols,
    int* __restrict__ found_flag)
{
    const int blk = (int)blockIdx.x;
    const int tiles_per_plane = PP_TILES_PER_PERIOD;
    const int planes = row_batch_count * col_batch_count;
    if(blk >= planes * tiles_per_plane) return;

    if(*found_flag) return;

    const int plane = blk / tiles_per_plane;
    const int tile = blk % tiles_per_plane;
    const int row_in_batch = plane / col_batch_count;
    const int col_in_batch = plane % col_batch_count;
    const int row_period = row_period0 + row_in_batch;
    const int col_period = col_period0 + col_in_batch;

    const int u = (int)threadIdx.y;
    const int v = (int)threadIdx.x;

    const int local_rp = tile / 16;
    const int local_cp = tile % 16;
    const int row_base = pp_period_row_base(local_rp);
    const int col_base = pp_period_col_base(local_cp);
    const int t_rows = row_period * PP_ROW_PERIOD + row_base;
    const int t_cols = col_period * PP_COL_PERIOD + col_base;

    const int rel_r = row_in_batch * PP_ROW_PERIOD + row_base + PP_ROW_PAT[u];
    const int rel_c = col_in_batch * PP_COL_PERIOD + col_base + PP_COL_PAT[v];
    const bool valid = rel_r < M && rel_c < N;

    __shared__ int32_t s_tile[PP_HASH_H * PP_HASH_W];
    __shared__ uint32_t s_jackpot[PP_JACKPOT_WORDS];

    if(u == 0 && v == 0)
        for(int i = 0; i < PP_JACKPOT_WORDS; i++) s_jackpot[i] = 0u;
    s_tile[u * PP_HASH_W + v] = 0;
    __syncthreads();

    int32_t cell = 0;
    const int num_steps = K / R;
    for(int step = 0; step < num_steps; step++){
        if(valid){
            const int32_t partial = C_hist[
                pp_c_hist_panel_index(step, rel_r, rel_c,
                                      row_batch_count, col_batch_count)];
            cell += partial;
        }
        s_tile[u * PP_HASH_W + v] = cell;
        __syncthreads();

        if(u == 0 && v == 0){
            uint32_t xored = 0u;
            for(int i = 0; i < PP_HASH_H * PP_HASH_W; i++)
                xored ^= (uint32_t)s_tile[i];
            const int tid = step % PP_JACKPOT_WORDS;
            s_jackpot[tid] = pp_rotl32(s_jackpot[tid], PP_LROT) ^ xored;
        }
        __syncthreads();
    }

    if(u != 0 || v != 0) return;

    uint32_t msg[PP_JACKPOT_WORDS];
    for(int i = 0; i < PP_JACKPOT_WORDS; i++) msg[i] = s_jackpot[i];

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
