#include "cp_gpu.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cp_gpu.cuh"
#include "plain_proof_kernel.cuh"

#define CU_CHECK(call) do { \
    cudaError_t _e = (call); \
    if(_e != cudaSuccess){ \
        fprintf(stderr,"[CUDA] %s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

typedef struct {
    int     dev;
    int8_t* d_Ap;
    int8_t* d_BpT;
    int*    d_found;
    int*    d_out_t_rows;
    int*    d_out_t_cols;
    uint32_t* d_a_key8;
} GpuCtx;

static GpuCtx g_gpus[MAX_GPUS];
static int g_ngpu = 0;
static int g_contiguous = 0;

static void sync_tile_config(void)
{
    static const int scattered_row[PP_HASH_H] = {
        0, 8, 32, 40, 64, 72, 96, 104
    };
    static const int scattered_col[PP_HASH_W] = {
        0, 1, 32, 33, 64, 65, 96, 97,
        128, 129, 160, 161, 192, 193, 224, 225
    };
    int row_pat[PP_HASH_H];
    int col_pat[PP_HASH_W];
    int mode = g_contiguous ? 1 : 0;
    if(g_contiguous){
        for(int i = 0; i < PP_HASH_H; i++) row_pat[i] = i;
        for(int i = 0; i < PP_HASH_W; i++) col_pat[i] = i;
    } else {
        memcpy(row_pat, scattered_row, sizeof(row_pat));
        memcpy(col_pat, scattered_col, sizeof(col_pat));
    }
    for(int i = 0; i < g_ngpu; i++){
        CU_CHECK(cudaSetDevice(g_gpus[i].dev));
        CU_CHECK(cudaMemcpyToSymbol(PP_ROW_PAT, row_pat, sizeof(row_pat)));
        CU_CHECK(cudaMemcpyToSymbol(PP_COL_PAT, col_pat, sizeof(col_pat)));
        CU_CHECK(cudaMemcpyToSymbol(PP_CONTIGUOUS_MODE, &mode, sizeof(mode)));
    }
}

void cp_gpu_set_contiguous_tiles(int on)
{
    g_contiguous = on;
    if(g_ngpu > 0) sync_tile_config();
}

void cp_gpu_init(int* devs, int ndev)
{
    g_ngpu = ndev;
    printf("[gpu] Initializing %d GPU(s)...\n", ndev);
    fflush(stdout);
    for(int i = 0; i < ndev; i++){
        GpuCtx* g = &g_gpus[i];
        g->dev = devs[i];
        CU_CHECK(cudaSetDevice(g->dev));
        CU_CHECK(cudaMalloc(&g->d_found, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_out_t_rows, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_out_t_cols, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_a_key8, 8*sizeof(uint32_t)));
        printf("[gpu] GPU%d OK\n", g->dev); fflush(stdout);
    }
    sync_tile_config();
}

void cp_gpu_shutdown(void)
{
    for(int i = 0; i < g_ngpu; i++){
        GpuCtx* g = &g_gpus[i];
        CU_CHECK(cudaSetDevice(g->dev));
        if(g->d_Ap) cudaFree(g->d_Ap);
        if(g->d_BpT) cudaFree(g->d_BpT);
        if(g->d_found) cudaFree(g->d_found);
        if(g->d_out_t_rows) cudaFree(g->d_out_t_rows);
        if(g->d_out_t_cols) cudaFree(g->d_out_t_cols);
        if(g->d_a_key8) cudaFree(g->d_a_key8);
    }
    g_ngpu = 0;
}

static void ensure_buffers(GpuCtx* g, int m, int n)
{
    size_t szAp  = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    CU_CHECK(cudaSetDevice(g->dev));
    if(!g->d_Ap){
        CU_CHECK(cudaMalloc(&g->d_Ap, szAp));
        CU_CHECK(cudaMalloc(&g->d_BpT, szBpT));
    }
}

int cp_gpu_mine_plain_proof(
    const int8_t* h_A, const int8_t* h_B,
    const uint8_t* a_key, const uint32_t pool_tgt[8],
    int m, int n,
    int* out_t_rows, int* out_t_cols)
{
    size_t szAp  = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;

    uint32_t bound[8];
    cp_scale_jackpot_target(pool_tgt, bound);
    uint32_t a_key32[8];
    memcpy(a_key32, a_key, 32);

    const int row_parts = cp_pp_num_row_parts(m, g_contiguous);
    const int col_parts = cp_pp_num_col_parts(n, g_contiguous);
    const int batch = 64;
    dim3 block(PP_HASH_W, PP_HASH_H);
    int zero = 0;
    int found = 0;
    const int total_tiles = row_parts * col_parts;
    double scan_t0 = cp_now_sec();

    sync_tile_config();

    printf("[gpu] plain_proof scan %dx%d hash tiles, bound scaled by %d\n",
           row_parts, col_parts, PP_HASH_H * PP_HASH_W * K_DIM);
    printf("[gpu] jackpot target LE: %08X %08X ...\n", bound[0], bound[1]);
    fflush(stdout);

    for(int i = 0; i < g_ngpu; i++){
        GpuCtx* g = &g_gpus[i];
        ensure_buffers(g, m, n);
        CU_CHECK(cudaMemcpy(g->d_Ap,  h_A,  szAp,  cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g->d_BpT, h_B,  szBpT, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g->d_a_key8, a_key32, 32, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }

    for(int rp0 = 0; rp0 < row_parts && !found; rp0 += batch){
        if(cp_job_should_cancel()) return -1;
        int rpb = batch;
        if(rp0 + rpb > row_parts) rpb = row_parts - rp0;
        for(int cp0 = 0; cp0 < col_parts && !found; cp0 += batch){
            if(cp_job_should_cancel()) return -1;
            int cpb = batch;
            if(cp0 + cpb > col_parts) cpb = col_parts - cp0;
            dim3 grid(cpb, rpb);

            for(int i = 0; i < g_ngpu; i++){
                GpuCtx* g = &g_gpus[i];
                CU_CHECK(cudaSetDevice(g->dev));
                plain_proof_jackpot_kernel<<<grid, block>>>(
                    g->d_Ap, g->d_BpT,
                    m, n, K_DIM, R_RANK,
                    rp0, cp0, row_parts, col_parts,
                    bound[0], bound[1], bound[2], bound[3],
                    bound[4], bound[5], bound[6], bound[7],
                    g->d_a_key8,
                    g->d_out_t_rows, g->d_out_t_cols, g->d_found
                );
                CU_CHECK(cudaGetLastError());
            }

            for(int i = 0; i < g_ngpu; i++){
                GpuCtx* g = &g_gpus[i];
                CU_CHECK(cudaSetDevice(g->dev));
                CU_CHECK(cudaDeviceSynchronize());
                int f = 0;
                CU_CHECK(cudaMemcpy(&f, g->d_found, sizeof(int), cudaMemcpyDeviceToHost));
                if(f && !found){
                    found = 1;
                    CU_CHECK(cudaMemcpy(out_t_rows, g->d_out_t_rows, sizeof(int), cudaMemcpyDeviceToHost));
                    CU_CHECK(cudaMemcpy(out_t_cols, g->d_out_t_cols, sizeof(int), cudaMemcpyDeviceToHost));
                    printf("[gpu] GPU%d: plain_proof SHARE t_rows=%d t_cols=%d\n",
                           g->dev, *out_t_rows, *out_t_cols);
                    fflush(stdout);
                }
            }
        }
        if((rp0 / batch) % 4 == 0){
            int tiles_done = (rp0 + rpb) * col_parts;
            double scan_sec = cp_now_sec() - scan_t0;
            if(scan_sec < 1e-9) scan_sec = 1e-9;
            double scan_mac_s = (double)tiles_done * cp_pp_macs_per_hash_tile() / scan_sec;
            char mac_buf[32];
            cp_pp_fmt_mac_rate(scan_mac_s, mac_buf, sizeof(mac_buf));
            printf("[gpu] plain_proof progress: row parts %d/%d tiles %d/%d (%.1f%%) %s\n",
                   rp0 + rpb, row_parts, tiles_done, total_tiles,
                   100.0 * (double)tiles_done / (double)total_tiles, mac_buf);
            fflush(stdout);
        }
    }
    return found;
}
