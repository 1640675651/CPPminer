#include "cp_gpu.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_state.h"
#include "cp_util.h"

#include <cuda_runtime.h>
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "cp_gpu.cuh"
#include "cp_gpu_gen.cuh"
#include "cp_noise_phase.cuh"
#include "cp_merkle_tree.cuh"
#include "cp_noise.h"
#include "cp_cutlass.h"
#include "plain_proof_kernel.cuh"
#include "plain_proof_period.cuh"

#define CU_CHECK(call) do { \
    cudaError_t _e = (call); \
    if(_e != cudaSuccess){ \
        fprintf(stderr,"[CUDA] %s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _e = (call); \
    if(_e != CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"[CUBLAS] %s:%d %s: status %d\n",__FILE__,__LINE__,#call,(int)_e); \
        exit(1); \
    } \
} while(0)
#endif

typedef struct {
    int       dev;
    int8_t*   d_Ap;
    int8_t*   d_BpT;
    int8_t*   d_A_sig;
    int8_t*   d_Bt_sig;
    uint32_t* d_e_ar;
    uint32_t* d_e_bl;
    int8_t*   d_eal;
    int8_t*   d_ebr;
    size_t    noise_m_cap;
    size_t    noise_n_cap;
    uint8_t*  d_merkle_roots;
    size_t    merkle_roots_cap;
    uint8_t*  d_seed_a;
    uint8_t*  d_seed_b;
    uint8_t*  d_job_key;
    int*      d_found;
    int*      d_out_t_rows;
    int*      d_out_t_cols;
    uint32_t* d_a_key8;
    int32_t*  d_C_hist;
    size_t    C_hist_cap;
    uint32_t* d_tile_xor;
    size_t    tile_xor_cap;
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    cublasHandle_t cublas;
#endif
    int       use_cublas_period;
    int       use_cutlass_fused;
} GpuCtx;

static GpuCtx g_gpus[MAX_GPUS];
static int g_ngpu = 0;
static int g_contiguous = 0;
static int g_period_gemm = 1;
static int g_row_period_batch = CP_ROW_PERIOD_BATCH_DEFAULT;
static int g_col_period_batch = CP_PERIOD_BATCH_DEFAULT;
static int g_step_major_ap = 0; /* Case 9 default; main sets 1 for cuBLAS period */
static int g_cutlass_fused = 0;

static size_t pp_hist_batch_int32s(int row_batch_count, int col_batch_count)
{
    return (size_t)(K_DIM / R_RANK)
         * (size_t)(row_batch_count * PP_ROW_PERIOD)
         * (size_t)(col_batch_count * PP_COL_PERIOD);
}

static size_t pp_hist_batch_bytes(int row_batch_count, int col_batch_count)
{
    return pp_hist_batch_int32s(row_batch_count, col_batch_count) * sizeof(int32_t);
}

static int pp_clamp_row_period_batch(int batch)
{
    if(batch < 1) batch = 1;
    if(batch > CP_ROW_PERIOD_BATCH_MAX) batch = CP_ROW_PERIOD_BATCH_MAX;
    return batch;
}

static int pp_clamp_col_period_batch(int batch)
{
    if(batch < 1) batch = 1;
    if(batch > CP_PERIOD_BATCH_MAX) batch = CP_PERIOD_BATCH_MAX;
    return batch;
}

static int pp_batch_hash_tiles(int row_batch_count, int col_batch_count)
{
    return row_batch_count * col_batch_count * PP_TILES_PER_PERIOD;
}

static int gpu_num_row_periods(int m)
{
    if(g_cutlass_fused) return m / CP_CUTLASS_CTA_M;
    return cp_pp_num_row_periods(m, g_contiguous);
}

static int gpu_num_col_periods(int n)
{
    if(g_cutlass_fused) return n / CP_CUTLASS_CTA_N;
    return cp_pp_num_col_periods(n, g_contiguous);
}

#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
static const char* cublas_status_str(cublasStatus_t st)
{
    switch(st){
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
    default: return "UNKNOWN";
    }
}

/* Row-major C[M×N] = A[M×R] * B[N×R]^T (B = Bt rows). Same API as matmul_benchmark.cu. */
static cublasStatus_t pp_cublas_gemm_i8_bt(
    cublasHandle_t handle,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc,
    int M, int N, int R,
    int32_t beta)
{
    const int32_t alpha = 1;
    return cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, R,
        &alpha,
        B, CUDA_R_8I, ldb,
        A, CUDA_R_8I, lda,
        &beta, C, CUDA_R_32I, ldc,
        CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
}

/* Probe production-sized int8 GEMM (must use CUDA_R_32I compute type in .cu). */
static int gpu_probe_cublas_int8(GpuCtx* g)
{
    const int M = PP_ROW_PERIOD;
    const int N = PP_COL_PERIOD;
    const int R = R_RANK;
    int8_t *dA = NULL, *dB = NULL;
    int32_t *dC = NULL;
    cublasStatus_t st;

    CU_CHECK(cudaSetDevice(g->dev));
    CUBLAS_CHECK(cublasSetPointerMode(g->cublas, CUBLAS_POINTER_MODE_HOST));
    CU_CHECK(cudaMalloc(&dA, (size_t)M * (size_t)R));
    CU_CHECK(cudaMalloc(&dB, (size_t)2 * (size_t)N * (size_t)R));
    CU_CHECK(cudaMalloc(&dC, (size_t)M * (size_t)N * 2 * sizeof(int32_t)));
    CU_CHECK(cudaMemset(dA, 0, (size_t)M * (size_t)R));
    CU_CHECK(cudaMemset(dB, 0, (size_t)2 * (size_t)N * (size_t)R));

    st = pp_cublas_gemm_i8_bt(g->cublas, dA, R, dB, R, dC, N, M, N, R, 0);
    if(st == CUBLAS_STATUS_SUCCESS){
        st = pp_cublas_gemm_i8_bt(
            g->cublas, dA, R, dB, R, dC, N * 2, M, N * 2, R, 0);
    }
    CU_CHECK(cudaDeviceSynchronize());

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    if(st != CUBLAS_STATUS_SUCCESS){
        cudaDeviceProp prop;
        CU_CHECK(cudaGetDeviceProperties(&prop, g->dev));
        printf("[gpu] GPU%d: cuBLAS int8 GEMM probe failed (%s, status %d), "
               "sm_%d%d -> CUDA period GEMM fallback\n",
               g->dev, cublas_status_str(st), (int)st,
               prop.major, prop.minor);
        fflush(stdout);
        return 0;
    }
    return 1;
}
#endif /* CP_ENABLE_CUBLAS */

static void sync_ap_layout(void)
{
    int mode = g_step_major_ap ? 1 : 0;
    for(int i = 0; i < g_ngpu; i++){
        CU_CHECK(cudaSetDevice(g_gpus[i].dev));
        CU_CHECK(cudaMemcpyToSymbol(PP_STEP_MAJOR_AP, &mode, sizeof(mode)));
    }
}

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
    if(on) g_period_gemm = 0;
    if(g_ngpu > 0) sync_tile_config();
}

void cp_gpu_set_period_gemm(int on)
{
    g_period_gemm = on ? 1 : 0;
}

void cp_gpu_set_period_batch(int batch)
{
    cp_gpu_set_col_period_batch(batch);
}

void cp_gpu_set_row_period_batch(int batch)
{
    g_row_period_batch = pp_clamp_row_period_batch(batch);
}

void cp_gpu_set_col_period_batch(int batch)
{
    g_col_period_batch = pp_clamp_col_period_batch(batch);
}

void cp_gpu_set_step_major_ap(int on)
{
    g_step_major_ap = on ? 1 : 0;
    if(g_ngpu > 0) sync_ap_layout();
}

void cp_gpu_set_cutlass_fused(int on)
{
    g_cutlass_fused = on ? 1 : 0;
    for(int i = 0; i < g_ngpu; i++)
        g_gpus[i].use_cutlass_fused = g_cutlass_fused;
}

void cp_gpu_init(int* devs, int ndev)
{
    g_ngpu = ndev;
    printf("[gpu] Initializing %d GPU(s)...\n", ndev);
    fflush(stdout);
    /* Blocking sync: large period-batch kernels sleep the CPU instead of
     * spin-waiting in cudaDeviceSynchronize (default WDDM schedule).
     * Set before any device is current so flags apply at primary-context init. */
    CU_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    for(int i = 0; i < ndev; i++){
        GpuCtx* g = &g_gpus[i];
        g->dev = devs[i];
        CU_CHECK(cudaSetDevice(g->dev));
        if(g_cutlass_fused && !cp_cutlass_device_ok(g->dev)){
            fprintf(stderr,
                    "[gpu] GPU%d: --cutlass-fused needs Pascal SIMT (compute capability <= 7.5)\n",
                    g->dev);
            exit(1);
        }
        CU_CHECK(cudaMalloc(&g->d_found, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_out_t_rows, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_out_t_cols, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_a_key8, 8*sizeof(uint32_t)));
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
        CUBLAS_CHECK(cublasCreate(&g->cublas));
        g->use_cublas_period = gpu_probe_cublas_int8(g);
#else
        g->use_cublas_period = 0;
#endif
        g->use_cutlass_fused = g_cutlass_fused;
        printf("[gpu] GPU%d OK (%s, blocking sync)\n", g->dev,
               g->use_cutlass_fused ? "CUTLASS fused period GEMM"
               : (g->use_cublas_period ? "cuBLAS int8 period GEMM"
                                       : "CUDA period GEMM"));
        fflush(stdout);
    }
    sync_tile_config();
    sync_ap_layout();
}

int cp_gpu_list_devices(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess || count <= 0){
        printf("[cuda] no CUDA devices found (%s)\n",
               err == cudaSuccess ? "count=0" : cudaGetErrorString(err));
        return 0;
    }
    printf("[cuda] CUDA devices (use --devices N[,M]):\n");
    for(int i = 0; i < count; i++){
        cudaDeviceProp prop{};
        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess){
            printf("  [%d] <unavailable>\n", i);
            continue;
        }
        printf("  [%d] %s  (sm_%d%d, %.1f GiB)\n", i, prop.name, prop.major,
               prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    return count;
}

void cp_gpu_shutdown(void)
{
    for(int i = 0; i < g_ngpu; i++){
        GpuCtx* g = &g_gpus[i];
        CU_CHECK(cudaSetDevice(g->dev));
        if(g->d_Ap) cudaFree(g->d_Ap);
        if(g->d_BpT) cudaFree(g->d_BpT);
        if(g->d_A_sig) cudaFree(g->d_A_sig);
        if(g->d_Bt_sig) cudaFree(g->d_Bt_sig);
        if(g->d_e_ar) cudaFree(g->d_e_ar);
        if(g->d_e_bl) cudaFree(g->d_e_bl);
        if(g->d_eal) cudaFree(g->d_eal);
        if(g->d_ebr) cudaFree(g->d_ebr);
        g->noise_m_cap = 0;
        g->noise_n_cap = 0;
        if(g->d_merkle_roots) cudaFree(g->d_merkle_roots);
        if(g->d_seed_a) cudaFree(g->d_seed_a);
        if(g->d_seed_b) cudaFree(g->d_seed_b);
        if(g->d_job_key) cudaFree(g->d_job_key);
        g->merkle_roots_cap = 0;
        if(g->d_found) cudaFree(g->d_found);
        if(g->d_out_t_rows) cudaFree(g->d_out_t_rows);
        if(g->d_out_t_cols) cudaFree(g->d_out_t_cols);
        if(g->d_a_key8) cudaFree(g->d_a_key8);
        if(g->d_C_hist) cudaFree(g->d_C_hist);
        if(g->d_tile_xor) cudaFree(g->d_tile_xor);
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
        if(g->cublas){ cublasDestroy(g->cublas); g->cublas = NULL; }
#endif
    }
    g_ngpu = 0;
}

static void ensure_buffers(GpuCtx* g, int m, int n)
{
    size_t szAp  = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    size_t raw_a = szAp;
    size_t raw_b = szBpT;
    size_t pad_a = (raw_a + 1023) / 1024 * 1024;
    size_t pad_b = (raw_b + 1023) / 1024 * 1024;
    size_t chunks_a = pad_a / 1024;
    size_t chunks_b = pad_b / 1024;
    size_t chunks_max = chunks_a > chunks_b ? chunks_a : chunks_b;
    size_t merkle_need = ((chunks_max + CP_MT_THREADS - 1) / CP_MT_THREADS) * 32;
    if(merkle_need < 32) merkle_need = 32;

    CU_CHECK(cudaSetDevice(g->dev));
    if(!g->d_Ap){
        /* d_Ap/d_BpT: noisy mats for GEMM/jackpot; layout set by PP_STEP_MAJOR_AP. */
        CU_CHECK(cudaMalloc(&g->d_Ap, szAp));
        CU_CHECK(cudaMalloc(&g->d_BpT, szBpT));
        CU_CHECK(cudaMalloc(&g->d_A_sig, szAp));
        CU_CHECK(cudaMalloc(&g->d_Bt_sig, szBpT));
        CU_CHECK(cudaMalloc(&g->d_e_ar, (size_t)K_DIM * 2 * sizeof(uint32_t)));
        CU_CHECK(cudaMalloc(&g->d_e_bl, (size_t)K_DIM * 2 * sizeof(uint32_t)));
        CU_CHECK(cudaMalloc(&g->d_eal, (size_t)m * R_RANK));
        CU_CHECK(cudaMalloc(&g->d_ebr, (size_t)n * R_RANK));
        g->noise_m_cap = (size_t)m * R_RANK;
        g->noise_n_cap = (size_t)n * R_RANK;
        CU_CHECK(cudaMalloc(&g->d_seed_a, 32));
        CU_CHECK(cudaMalloc(&g->d_seed_b, 32));
        CU_CHECK(cudaMalloc(&g->d_job_key, 32));
    }
    if(merkle_need > g->merkle_roots_cap){
        if(g->d_merkle_roots) cudaFree(g->d_merkle_roots);
        CU_CHECK(cudaMalloc(&g->d_merkle_roots, merkle_need));
        g->merkle_roots_cap = merkle_need;
    }
    {
        size_t eal_need = (size_t)m * R_RANK;
        size_t ebr_need = (size_t)n * R_RANK;
        if(eal_need > g->noise_m_cap){
            if(g->d_eal) cudaFree(g->d_eal);
            CU_CHECK(cudaMalloc(&g->d_eal, eal_need));
            g->noise_m_cap = eal_need;
        }
        if(ebr_need > g->noise_n_cap){
            if(g->d_ebr) cudaFree(g->d_ebr);
            CU_CHECK(cudaMalloc(&g->d_ebr, ebr_need));
            g->noise_n_cap = ebr_need;
        }
    }
    {
        if(g->use_cutlass_fused){
        /* Jackpot runs in CUTLASS mainloop tail; no tile_xor buffer. */
    } else {
            size_t hist_need = pp_hist_batch_bytes(
                g_row_period_batch, g_col_period_batch);
            if(hist_need > g->C_hist_cap){
                if(g->d_C_hist) cudaFree(g->d_C_hist);
                CU_CHECK(cudaMalloc(&g->d_C_hist, hist_need));
                g->C_hist_cap = hist_need;
            }
        }
    }
}

static size_t pp_ap_step_plane(int dim)
{
    return (size_t)dim * (size_t)R_RANK;
}

static void gpu_noise_generate(GpuCtx* g, int m, int n)
{
    const int tpb = 256;
    const int tpr = R_RANK / 32;
    const int rows_per_block = tpb / tpr;
    const int perm_blocks = (K_DIM + CP_B3_LINES * tpb - 1) / (CP_B3_LINES * tpb);

    cp_gen_dense_noise_kernel<<<(m + rows_per_block - 1) / rows_per_block, tpb>>>(
        0, m, R_RANK, g->d_seed_a, g->d_eal);
    cp_gen_dense_noise_kernel<<<(n + rows_per_block - 1) / rows_per_block, tpb>>>(
        1, n, R_RANK, g->d_seed_b, g->d_ebr);
    cp_build_perm_pairs_par_kernel<<<perm_blocks, tpb>>>(
        0, g->d_seed_a, K_DIM, R_RANK, g->d_e_ar);
    cp_build_perm_pairs_par_kernel<<<perm_blocks, tpb>>>(
        1, g->d_seed_b, K_DIM, R_RANK, g->d_e_bl);
    CU_CHECK(cudaGetLastError());
}

static void gpu_noise_apply(GpuCtx* g, int m, int n)
{
    const int tpb = 256;
    const size_t smem = (size_t)R_RANK + (size_t)K_DIM;

    if(g_step_major_ap){
        cp_apply_noise_a_kernel<<<m, tpb, smem>>>(
            g->d_A_sig, g->d_eal, g->d_Ap, m, K_DIM, R_RANK, g->d_e_ar);
        cp_apply_noise_b_kernel<<<n, tpb, smem>>>(
            g->d_Bt_sig, g->d_ebr, g->d_BpT, n, K_DIM, R_RANK, g->d_e_bl);
    } else {
        cp_apply_noise_a_rowmajor_kernel<<<m, tpb, smem>>>(
            g->d_A_sig, g->d_eal, g->d_Ap, m, K_DIM, R_RANK, g->d_e_ar);
        cp_apply_noise_b_rowmajor_kernel<<<n, tpb, smem>>>(
            g->d_Bt_sig, g->d_ebr, g->d_BpT, n, K_DIM, R_RANK, g->d_e_bl);
    }
    CU_CHECK(cudaGetLastError());
}

static void gpu_upload_rowmajor_noisy(
    GpuCtx* g, const int8_t* h_a, const int8_t* h_b, int m, int n)
{
    const int tpb = 256;
    size_t szAp = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    CU_CHECK(cudaMemcpy(g->d_A_sig, h_a, szAp, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(g->d_Bt_sig, h_b, szBpT, cudaMemcpyHostToDevice));
    if(g_step_major_ap){
        cp_pack_rowmajor_to_step_kernel<<<(int)((szAp + tpb - 1) / tpb), tpb>>>(
            g->d_A_sig, g->d_Ap, m, K_DIM, R_RANK);
        cp_pack_rowmajor_to_step_kernel<<<(int)((szBpT + tpb - 1) / tpb), tpb>>>(
            g->d_Bt_sig, g->d_BpT, n, K_DIM, R_RANK);
    } else {
        CU_CHECK(cudaMemcpy(g->d_Ap, g->d_A_sig, szAp, cudaMemcpyDeviceToDevice));
        CU_CHECK(cudaMemcpy(g->d_BpT, g->d_Bt_sig, szBpT, cudaMemcpyDeviceToDevice));
    }
    CU_CHECK(cudaGetLastError());
}

static void gpu_period_gemm_panel_ptrs(
    const int8_t* d_Ap, const int8_t* d_BpT,
    int m, int n, int step, size_t row_base, size_t col_base,
    const int8_t** Ap, const int8_t** Bp0, int* lda, int* ldb)
{
    if(g_step_major_ap){
        const size_t ap_step_plane = pp_ap_step_plane(m);
        const size_t bp_step_plane = pp_ap_step_plane(n);
        *Ap = d_Ap + (size_t)step * ap_step_plane + row_base * (size_t)R_RANK;
        *Bp0 = d_BpT + (size_t)step * bp_step_plane + col_base * (size_t)R_RANK;
        *lda = R_RANK;
        *ldb = R_RANK;
    } else {
        *Ap = d_Ap + row_base * (size_t)K_DIM + (size_t)step * (size_t)R_RANK;
        *Bp0 = d_BpT + col_base * (size_t)K_DIM + (size_t)step * (size_t)R_RANK;
        *lda = K_DIM;
        *ldb = K_DIM;
    }
}

#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
/*
 * cuBLAS: one fat GemmEx per rank step into C_hist[step] (rank partials only).
 * Jackpot cumulates partials across steps (no plane_add).
 * Ap/BpT layout: row-major lda=K_DIM (default) or step-major lda=R_RANK (--step-major).
 */
static void gpu_period_gemm_cublas_batch(
    GpuCtx* g, int m, int n, int row_period0, int col_period0,
    int row_batch_count, int col_batch_count)
{
    const int M = row_batch_count * PP_ROW_PERIOD;
    const int N = PP_COL_PERIOD;
    const int R = R_RANK;
    const int num_steps = K_DIM / R;
    const int N_fat = col_batch_count * N;
    const size_t step_plane = (size_t)M * (size_t)N_fat;
    const size_t row_base = (size_t)row_period0 * (size_t)PP_ROW_PERIOD;
    const size_t col_base = (size_t)col_period0 * (size_t)N;

    for(int s = 0; s < num_steps; s++){
        const int8_t *Ap = NULL, *Bp0 = NULL;
        int lda = 0, ldb = 0;
        gpu_period_gemm_panel_ptrs(
            g->d_Ap, g->d_BpT, m, n, s, row_base, col_base, &Ap, &Bp0, &lda, &ldb);
        int32_t* Cp = g->d_C_hist + (size_t)s * step_plane;

        cublasStatus_t st = pp_cublas_gemm_i8_bt(
            g->cublas, Ap, lda, Bp0, ldb, Cp, N_fat, M, N_fat, R, 0);
        if(st != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "[CUBLAS] GemmEx failed: %s (%d)\n",
                    cublas_status_str(st), (int)st);
            exit(1);
        }
    }
}

typedef struct {
    float gemm_ex_ms;
} PeriodCublasBreakdown;

/* Profile only: per-step GemmEx timing (no plane_add). */
static PeriodCublasBreakdown gpu_period_gemm_cublas_batch_timed(
    GpuCtx* g, int m, int n, int row_period0, int col_period0,
    int row_batch_count, int col_batch_count)
{
    const int M = row_batch_count * PP_ROW_PERIOD;
    const int N = PP_COL_PERIOD;
    const int R = R_RANK;
    const int num_steps = K_DIM / R;
    const int N_fat = col_batch_count * N;
    const size_t step_plane = (size_t)M * (size_t)N_fat;
    const size_t row_base = (size_t)row_period0 * (size_t)PP_ROW_PERIOD;
    const size_t col_base = (size_t)col_period0 * (size_t)N;
    PeriodCublasBreakdown out = {0.f};
    static cudaEvent_t e0, e1;
    static int ev_ready = 0;
    float ms = 0.f;

    if(!ev_ready){
        CU_CHECK(cudaEventCreate(&e0));
        CU_CHECK(cudaEventCreate(&e1));
        ev_ready = 1;
    }

    for(int s = 0; s < num_steps; s++){
        const int8_t *Ap = NULL, *Bp0 = NULL;
        int lda = 0, ldb = 0;
        gpu_period_gemm_panel_ptrs(
            g->d_Ap, g->d_BpT, m, n, s, row_base, col_base, &Ap, &Bp0, &lda, &ldb);
        int32_t* Cp = g->d_C_hist + (size_t)s * step_plane;

        CU_CHECK(cudaEventRecord(e0));
        cublasStatus_t st = pp_cublas_gemm_i8_bt(
            g->cublas, Ap, lda, Bp0, ldb, Cp, N_fat, M, N_fat, R, 0);
        CU_CHECK(cudaEventRecord(e1));
        CU_CHECK(cudaEventSynchronize(e1));
        CU_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        out.gemm_ex_ms += ms;
        if(st != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "[CUBLAS] GemmEx failed: %s (%d)\n",
                    cublas_status_str(st), (int)st);
            exit(1);
        }
    }

    return out;
}
#endif /* CP_ENABLE_CUBLAS */

static void gpu_period_gemm_cuda_batch(
    GpuCtx* g, int m, int n, int row_period0, int col_period0,
    int row_batch_count, int col_batch_count)
{
    const dim3 grid(PP_COL_PERIOD / 16, PP_ROW_PERIOD / 16);
    const dim3 block(16, 16);

    for(int rb = 0; rb < row_batch_count; rb++){
        for(int cb = 0; cb < col_batch_count; cb++){
            plain_proof_period_gemm_kernel<<<grid, block>>>(
                g->d_Ap, g->d_BpT,
                m, n, K_DIM, R_RANK,
                row_period0 + rb, col_period0 + cb,
                rb, cb, row_batch_count, col_batch_count,
                g->d_C_hist);
        }
    }
    CU_CHECK(cudaGetLastError());
}

static void gpu_period_gemm_batch(
    GpuCtx* g, int m, int n, int row_period0, int col_period0,
    int row_batch_count, int col_batch_count,
    const uint32_t bound[8])
{
    if(g->use_cutlass_fused){
        const size_t tiles_per_batch = cp_cutlass_tiles_per_batch(
            row_batch_count, col_batch_count);
        CpCutlassJackpotLaunch jp;
        for(int i = 0; i < 8; i++)
            jp.bound[i] = bound[i];
        jp.d_a_key8 = g->d_a_key8;
        jp.d_found = g->d_found;
        jp.d_out_t_rows = g->d_out_t_rows;
        jp.d_out_t_cols = g->d_out_t_cols;
        jp.row_period0 = row_period0;
        jp.col_period0 = col_period0;
        if(cp_cutlass_period_batch(
               g->dev, g->d_Ap, g->d_BpT, m, n, row_period0, col_period0,
               row_batch_count, col_batch_count, g_step_major_ap, nullptr,
               tiles_per_batch, &jp) != 0){
            fprintf(stderr, "[cutlass] period batch failed\n");
            exit(1);
        }
        return;
    }
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    if(g->use_cublas_period)
        gpu_period_gemm_cublas_batch(
            g, m, n, row_period0, col_period0, row_batch_count, col_batch_count);
    else
#endif
        gpu_period_gemm_cuda_batch(
            g, m, n, row_period0, col_period0, row_batch_count, col_batch_count);
}

static void cp_gpu_merkle_finish_root(
    const uint8_t* d_job_key, uint8_t* d_roots, int num_subroots)
{
    const int smem = CP_MT_SMEM_BYTES;
    int num_mt_blocks = (num_subroots + CP_MT_THREADS - 1) / CP_MT_THREADS;
    if(num_mt_blocks == 1){
        cp_compute_blake_mt_kernel<CP_MT_THREADS, true>
            <<<1, CP_MT_THREADS, smem>>>(d_job_key, d_roots, num_subroots);
    }else{
        cp_compute_blake_mt_kernel<CP_MT_THREADS, false>
            <<<num_mt_blocks, CP_MT_THREADS, smem>>>(d_job_key, d_roots, num_subroots);
        cp_reduce_roots_kernel<CP_MT_THREADS>
            <<<1, CP_MT_THREADS, smem>>>(d_job_key, d_roots, num_mt_blocks);
    }
    CU_CHECK(cudaGetLastError());
}

static int gpu_matrix_keyed_hash(GpuCtx* g, const int8_t* d_mat,
                                 size_t raw_len, size_t pad_len,
                                 const uint8_t job_key[32], uint8_t out[32])
{
    int num_chunks = (int)(pad_len / 1024);
    if(num_chunks == 1){
        uint8_t* tmp = (uint8_t*)malloc(pad_len);
        if(!tmp) return -1;
        CU_CHECK(cudaMemcpy(tmp, d_mat, raw_len, cudaMemcpyDeviceToHost));
        if(pad_len > raw_len) memset(tmp + raw_len, 0, pad_len - raw_len);
        pearl_keyed_matrix_digest(tmp, pad_len, job_key, out);
        free(tmp);
        return 0;
    }
    CU_CHECK(cudaMemcpy(g->d_job_key, job_key, 32, cudaMemcpyHostToDevice));
    {
        int num_subroots = (num_chunks + CP_MT_THREADS - 1) / CP_MT_THREADS;
        cp_keyed_chunk_roots_kernel<<<num_subroots, CP_MT_THREADS, CP_MT_SMEM_BYTES>>>(
            (const uint8_t*)d_mat, raw_len, pad_len, g->d_job_key,
            g->d_merkle_roots, num_chunks);
        CU_CHECK(cudaGetLastError());
        cp_gpu_merkle_finish_root(g->d_job_key, g->d_merkle_roots, num_subroots);
    }
    CU_CHECK(cudaDeviceSynchronize());
    CU_CHECK(cudaMemcpy(out, g->d_merkle_roots, 32, cudaMemcpyDeviceToHost));
    return 0;
}

static uint64_t cp_gpu_fresh_rng_seed(void)
{
    uint64_t s = 0;
    if(cp_random_u64(&s) == 0)
        return s;
    /* Fallback if CSPRNG unavailable */
    s = (uint64_t)(cp_now_sec() * 1e9);
#ifdef _WIN32
    s ^= (uint64_t)GetTickCount64();
#endif
    s ^= (uint64_t)(uintptr_t)&s;
    return s ? s : 1ULL;
}

static int gpu_prepare_noisy_matrices(
    GpuCtx* g, uint64_t rng_seed,
    const uint8_t job_key[32], int m, int n,
    uint8_t a_key_out[32])
{
    size_t szAp = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    size_t pad_a = (szAp + 1023) / 1024 * 1024;
    size_t pad_b = (szBpT + 1023) / 1024 * 1024;
    const int tpb = 256;
    int total_a = m * K_DIM;
    int total_b = n * K_DIM;
    uint8_t hash_a[32], hash_b[32], b_seed[32];
    double t_step, t_total;

    CU_CHECK(cudaSetDevice(g->dev));

    t_total = cp_now_sec();

    t_step = cp_now_sec();
    cp_gen_random_matrix_kernel<<<(total_a + tpb - 1) / tpb, tpb>>>(
        rng_seed, 0, total_a, g->d_A_sig);
    cp_gen_random_matrix_kernel<<<(total_b + tpb - 1) / tpb, tpb>>>(
        rng_seed, 1, total_b, g->d_Bt_sig);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu-prep] random A/B gen %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    if(cp_job_should_cancel()) return -1;

    t_step = cp_now_sec();
    if(gpu_matrix_keyed_hash(g, g->d_A_sig, szAp, pad_a, job_key, hash_a) != 0) return -1;
    printf("[gpu-prep] keyed hash A %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    t_step = cp_now_sec();
    if(gpu_matrix_keyed_hash(g, g->d_Bt_sig, szBpT, pad_b, job_key, hash_b) != 0) return -1;
    printf("[gpu-prep] keyed hash B %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    t_step = cp_now_sec();
    pearl_derive_noise_seeds(job_key, hash_a, hash_b, b_seed, a_key_out);
    CU_CHECK(cudaMemcpy(g->d_seed_a, a_key_out, 32, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(g->d_seed_b, b_seed, 32, cudaMemcpyHostToDevice));
    printf("[gpu-prep] noise seeds + H2D %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    t_step = cp_now_sec();
    gpu_noise_generate(g, m, n);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu-prep] noise gen (EAL/EBR/perm) %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    t_step = cp_now_sec();
    gpu_noise_apply(g, m, n);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu-prep] noise apply (matvec+fuse) %.3fs\n", cp_now_sec() - t_step);
    printf("[gpu-prep] total %.3fs\n", cp_now_sec() - t_total);
    fflush(stdout);

    return cp_job_should_cancel() ? -1 : 0;
}

static int compare_digest(const char* label, const uint8_t a[32], const uint8_t b[32])
{
    if(memcmp(a, b, 32) == 0) return 0;
    fprintf(stderr, "[align-test-prod] %s mismatch\n", label);
    fprintf(stderr, "  gpu/cpu ref: ");
    for(int i = 0; i < 32; i++) fprintf(stderr, "%02x", a[i]);
    fprintf(stderr, "\n  other:       ");
    for(int i = 0; i < 32; i++) fprintf(stderr, "%02x", b[i]);
    fprintf(stderr, "\n");
    return -1;
}

int cp_gpu_run_alignment_tests(int dev, int m, int n)
{
    int devs[1] = {dev};
    size_t szAp = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    size_t pad_a = (szAp + 1023) / 1024 * 1024;
    size_t pad_b = (szBpT + 1023) / 1024 * 1024;
    const int tpb = 256;
    const uint64_t rng_seed = 0xC0FFEE1234567890ULL;
    uint8_t job_key[32];
    uint8_t hash_a_gpu[32], hash_b_gpu[32];
    uint8_t hash_a_cpu[32], hash_b_cpu[32];
    uint8_t b_seed_gpu[32], a_key_gpu[32];
    uint8_t b_seed_cpu[32], a_key_cpu[32];
    int8_t* h_A = NULL;
    int8_t* h_Bt = NULL;
    uint32_t* h_e_ar = NULL;
    int8_t gpu_row[4096];
    int8_t cpu_row[4096];
    int rc = -1;
    double t0;

    for(int i = 0; i < 32; i++) job_key[i] = (uint8_t)((i * 11 + 7) & 0xff);

    printf("[align-test-prod] GPU vs CPU m=%d n=%d k=%d (device %d)\n",
           m, n, K_DIM, dev);
    fflush(stdout);

    cp_gpu_init(devs, 1);
    GpuCtx* g = &g_gpus[0];
    ensure_buffers(g, m, n);
    CU_CHECK(cudaSetDevice(g->dev));

    t0 = cp_now_sec();
    cp_gen_random_matrix_kernel<<<(m * K_DIM + tpb - 1) / tpb, tpb>>>(
        rng_seed, 0, m * K_DIM, g->d_A_sig);
    cp_gen_random_matrix_kernel<<<(n * K_DIM + tpb - 1) / tpb, tpb>>>(
        rng_seed, 1, n * K_DIM, g->d_Bt_sig);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    printf("[align-test-prod] random A,B gen %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    t0 = cp_now_sec();
    if(gpu_matrix_keyed_hash(g, g->d_A_sig, szAp, pad_a, job_key, hash_a_gpu) != 0)
        goto done;
    if(gpu_matrix_keyed_hash(g, g->d_Bt_sig, szBpT, pad_b, job_key, hash_b_gpu) != 0)
        goto done;
    printf("[align-test-prod] GPU keyed hash %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    h_A = (int8_t*)malloc(szAp);
    h_Bt = (int8_t*)malloc(szBpT);
    if(!h_A || !h_Bt){
        fprintf(stderr, "[align-test-prod] OOM host matrix buffers\n");
        goto done;
    }

    t0 = cp_now_sec();
    printf("[align-test-prod] D2H A (%.1f MiB)...\n", (double)szAp / (1024.0 * 1024.0));
    fflush(stdout);
    CU_CHECK(cudaMemcpy(h_A, g->d_A_sig, szAp, cudaMemcpyDeviceToHost));
    printf("[align-test-prod] D2H B^T (%.1f MiB)...\n", (double)szBpT / (1024.0 * 1024.0));
    fflush(stdout);
    CU_CHECK(cudaMemcpy(h_Bt, g->d_Bt_sig, szBpT, cudaMemcpyDeviceToHost));
    printf("[align-test-prod] D2H done %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    t0 = cp_now_sec();
    pearl_keyed_digest_int8(h_A, szAp, job_key, hash_a_cpu);
    pearl_keyed_digest_int8(h_Bt, szBpT, job_key, hash_b_cpu);
    printf("[align-test-prod] CPU keyed digest %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    if(compare_digest("hash_a", hash_a_gpu, hash_a_cpu) != 0) goto done;
    if(compare_digest("hash_b", hash_b_gpu, hash_b_cpu) != 0) goto done;
    printf("[align-test-prod] GPU/CPU matrix hash OK\n");
    fflush(stdout);

    pearl_derive_noise_seeds(job_key, hash_a_gpu, hash_b_gpu, b_seed_gpu, a_key_gpu);
    pearl_commitment_seeds(job_key, h_A, h_Bt, m, n, K_DIM, b_seed_cpu, a_key_cpu);
    if(compare_digest("b_noise_seed", b_seed_gpu, b_seed_cpu) != 0) goto done;
    if(compare_digest("a_noise_seed", a_key_gpu, a_key_cpu) != 0) goto done;
    printf("[align-test-prod] noise seeds OK\n");
    fflush(stdout);

    CU_CHECK(cudaMemcpy(g->d_seed_a, a_key_gpu, 32, cudaMemcpyHostToDevice));
    {
        uint8_t gpu_digest[32], cpu_digest[32];
        cp_test_perm_hash_kernel<<<1, 1>>>(0, g->d_seed_a, g->d_job_key);
        CU_CHECK(cudaGetLastError());
        CU_CHECK(cudaDeviceSynchronize());
        CU_CHECK(cudaMemcpy(gpu_digest, g->d_job_key, 32, cudaMemcpyDeviceToHost));
        pearl_get_random_hash(0, PEARL_SEED_LABEL_A, a_key_gpu, 1, cpu_digest);
        if(memcmp(gpu_digest, cpu_digest, 32) != 0){
            fprintf(stderr, "[align-test-prod] GPU get_random_hash(0) mismatch\n");
            compare_digest("perm_hash0", gpu_digest, cpu_digest);
            goto done;
        }
        printf("[align-test-prod] GPU get_random_hash spot check OK\n");
        fflush(stdout);
    }

    CU_CHECK(cudaMemcpy(g->d_seed_b, b_seed_gpu, 32, cudaMemcpyHostToDevice));
    gpu_noise_generate(g, m, n);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());

    h_e_ar = (uint32_t*)malloc((size_t)K_DIM * 2 * sizeof(uint32_t));
    if(!h_e_ar){
        fprintf(stderr, "[align-test-prod] OOM perm buffer\n");
        goto done;
    }
    CU_CHECK(cudaMemcpy(h_e_ar, g->d_e_ar, (size_t)K_DIM * 2 * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
    {
        uint32_t* cpu_e_ar = (uint32_t*)malloc((size_t)K_DIM * 2 * sizeof(uint32_t));
        if(!cpu_e_ar) goto done;
        pearl_build_perm_pairs_a(a_key_gpu, K_DIM, R_RANK, cpu_e_ar);
        if(memcmp(h_e_ar, cpu_e_ar, (size_t)K_DIM * 2 * sizeof(uint32_t)) != 0){
            size_t words = (size_t)K_DIM * 2;
            for(size_t wi = 0; wi < words; wi++){
                if(h_e_ar[wi] != cpu_e_ar[wi]){
                    fprintf(stderr, "[align-test-prod] perm pairs A mismatch at word %zu (col %zu)\n",
                            wi, wi / 2);
                    break;
                }
            }
            free(cpu_e_ar);
            goto done;
        }
        free(cpu_e_ar);
    }
    printf("[align-test-prod] perm pairs A OK\n");
    fflush(stdout);

    t0 = cp_now_sec();
    gpu_noise_apply(g, m, n);
    CU_CHECK(cudaDeviceSynchronize());
    printf("[align-test-prod] GPU noise apply %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    {
        static const int sample_rows[] = {0, 1, 17, 4096, 8192};
        const int tpb = 256;
        for(size_t si = 0; si < sizeof(sample_rows) / sizeof(sample_rows[0]); si++){
            int row = sample_rows[si];
            if(row >= m) continue;
            if(g_step_major_ap){
                cp_gather_ap_row_kernel<<<(K_DIM + tpb - 1) / tpb, tpb>>>(
                    g->d_Ap, row, m, K_DIM, R_RANK,
                    g->d_A_sig + (size_t)row * K_DIM);
                CU_CHECK(cudaGetLastError());
                CU_CHECK(cudaMemcpy(gpu_row, g->d_A_sig + (size_t)row * K_DIM,
                                    (size_t)K_DIM, cudaMemcpyDeviceToHost));
            } else {
                CU_CHECK(cudaMemcpy(gpu_row, g->d_Ap + (size_t)row * K_DIM,
                                    (size_t)K_DIM, cudaMemcpyDeviceToHost));
            }
            pearl_fuse_noise_row_a(row, K_DIM, R_RANK, a_key_gpu, h_e_ar,
                                   h_A + (size_t)row * K_DIM, cpu_row);
            if(memcmp(gpu_row, cpu_row, (size_t)K_DIM) != 0){
                fprintf(stderr, "[align-test-prod] noisy A row %d mismatch\n", row);
                goto done;
            }
        }
    }
    printf("[align-test-prod] noisy A sample rows OK\n");
    fflush(stdout);

    rc = 0;
done:
    free(h_A);
    free(h_Bt);
    free(h_e_ar);
    cp_gpu_shutdown();
    if(rc == 0){
        printf("[align-test-prod] GPU pipeline OK\n");
        fflush(stdout);
    }
    return rc;
}

typedef struct {
    float gemm_ex_ms;
    float jackpot_ms;
    float sync_ms;
} PeriodBatchTimes;

static void scan_profile_ensure_events(cudaEvent_t ev[4])
{
    static int ready = 0;
    if(ready) return;
    for(int i = 0; i < 4; i++)
        CU_CHECK(cudaEventCreate(&ev[i]));
    ready = 1;
}

static void launch_jackpot_batch(
    GpuCtx* g, int row_batch_count, int col_batch_count,
    int row_period0, int col_period0, int m, int n,
    const uint32_t bound[8])
{
    if(g->use_cutlass_fused)
        return;

    const int num_blocks = pp_batch_hash_tiles(row_batch_count, col_batch_count);
    const dim3 block(PP_HASH_W, PP_HASH_H);
    plain_proof_period_jackpot_kernel<<<num_blocks, block>>>(
        g->d_C_hist,
        row_batch_count, col_batch_count,
        K_DIM, R_RANK,
        row_period0, col_period0,
        m, n,
        bound[0], bound[1], bound[2], bound[3],
        bound[4], bound[5], bound[6], bound[7],
        g->d_a_key8,
        g->d_out_t_rows, g->d_out_t_cols, g->d_found);
    CU_CHECK(cudaGetLastError());
}

static PeriodBatchTimes profile_period_batch_timed(
    GpuCtx* g, int rpi0, int cpi0, int row_batch_count, int col_batch_count,
    int m, int n, const uint32_t bound[8], cudaEvent_t ev[4])
{
    PeriodBatchTimes t = {0.f, 0.f, 0.f};
    int zero = 0;
    CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));

#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    if(g->use_cublas_period && !g->use_cutlass_fused){
        PeriodCublasBreakdown cb = gpu_period_gemm_cublas_batch_timed(
            g, m, n, rpi0, cpi0, row_batch_count, col_batch_count);
        t.gemm_ex_ms = cb.gemm_ex_ms;
    } else
#endif
    if(!g->use_cutlass_fused) {
        CU_CHECK(cudaEventRecord(ev[0]));
        gpu_period_gemm_cuda_batch(
            g, m, n, rpi0, cpi0, row_batch_count, col_batch_count);
        CU_CHECK(cudaEventRecord(ev[1]));
        CU_CHECK(cudaEventSynchronize(ev[1]));
        CU_CHECK(cudaEventElapsedTime(&t.gemm_ex_ms, ev[0], ev[1]));
    } else {
        CU_CHECK(cudaEventRecord(ev[0]));
        gpu_period_gemm_batch(
            g, m, n, rpi0, cpi0, row_batch_count, col_batch_count, bound);
        CU_CHECK(cudaEventRecord(ev[1]));
        CU_CHECK(cudaEventSynchronize(ev[1]));
        CU_CHECK(cudaEventElapsedTime(&t.gemm_ex_ms, ev[0], ev[1]));
    }

    CU_CHECK(cudaEventRecord(ev[1]));
    if(g->use_cutlass_fused){
        t.jackpot_ms = 0.f;
    } else {
        launch_jackpot_batch(
            g, row_batch_count, col_batch_count, rpi0, cpi0, m, n, bound);
        CU_CHECK(cudaEventRecord(ev[2]));
        CU_CHECK(cudaEventSynchronize(ev[2]));
        CU_CHECK(cudaEventElapsedTime(&t.jackpot_ms, ev[1], ev[2]));
    }

    CU_CHECK(cudaEventRecord(ev[3]));
    CU_CHECK(cudaDeviceSynchronize());
    int found = 0;
    CU_CHECK(cudaMemcpy(&found, g->d_found, sizeof(int), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaEventRecord(ev[0]));
    CU_CHECK(cudaEventSynchronize(ev[0]));
    CU_CHECK(cudaEventElapsedTime(&t.sync_ms, ev[3], ev[0]));
    (void)found;
    return t;
}

/* Same launch order as gpu_scan_device_period; timed with CUDA events. */
static float profile_period_batch_cuda_ms(
    GpuCtx* g, int rpi0, int cpi0, int row_batch_count, int col_batch_count,
    int m, int n, const uint32_t bound[8], cudaEvent_t e0, cudaEvent_t e1)
{
    CU_CHECK(cudaEventRecord(e0));
    gpu_period_gemm_batch(
        g, m, n, rpi0, cpi0, row_batch_count, col_batch_count, bound);
    launch_jackpot_batch(
        g, row_batch_count, col_batch_count, rpi0, cpi0, m, n, bound);
    for(int i = 0; i < g_ngpu; i++){
        CU_CHECK(cudaSetDevice(g_gpus[i].dev));
        CU_CHECK(cudaDeviceSynchronize());
        int f = 0;
        CU_CHECK(cudaMemcpy(&f, g_gpus[i].d_found, sizeof(int), cudaMemcpyDeviceToHost));
        (void)f;
    }
    CU_CHECK(cudaEventRecord(e1));
    CU_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CU_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

typedef struct {
    cudaEvent_t batch_start;
    cudaEvent_t post_launch;
    cudaEvent_t batch_end;
    int         ready;
    uint64_t    batches;
    uint64_t    batch_tiles;
    double      gpu_ms_sum;
    double      sync_ms_sum;
    double      wall_ms_sum;
} ScanBatchTiming;

static void scan_batch_timing_init(ScanBatchTiming* st)
{
    memset(st, 0, sizeof(*st));
    CU_CHECK(cudaSetDevice(g_gpus[0].dev));
    CU_CHECK(cudaEventCreate(&st->batch_start));
    CU_CHECK(cudaEventCreate(&st->post_launch));
    CU_CHECK(cudaEventCreate(&st->batch_end));
    st->ready = 1;
}

static void scan_batch_timing_destroy(ScanBatchTiming* st)
{
    if(!st->ready) return;
    CU_CHECK(cudaSetDevice(g_gpus[0].dev));
    CU_CHECK(cudaEventDestroy(st->batch_start));
    CU_CHECK(cudaEventDestroy(st->post_launch));
    CU_CHECK(cudaEventDestroy(st->batch_end));
    st->ready = 0;
}

/* Returns sync segment ms (DeviceSynchronize + found D2H); accumulates batch stats. */
static float scan_batch_timing_finish(
    ScanBatchTiming* st, int batch_tiles, double wall_ms)
{
    float gpu_ms = 0.f, sync_ms = 0.f;
    CU_CHECK(cudaSetDevice(g_gpus[0].dev));
    CU_CHECK(cudaEventRecord(st->batch_end));
    CU_CHECK(cudaEventSynchronize(st->batch_end));
    CU_CHECK(cudaEventElapsedTime(&gpu_ms, st->batch_start, st->batch_end));
    CU_CHECK(cudaEventElapsedTime(&sync_ms, st->post_launch, st->batch_end));
    st->batches++;
    st->batch_tiles += (uint64_t)batch_tiles;
    st->gpu_ms_sum += (double)gpu_ms;
    st->sync_ms_sum += (double)sync_ms;
    st->wall_ms_sum += wall_ms;
    return sync_ms;
}

static void scan_profile_print_summary(
    int row_batch_count, int col_batch_count, int runs, double macs_per_batch,
    double gemm_ex_ms, double jackpot_ms, double sync_ms,
    double wall_ms, double sweep_ms, int sweep_batches,
    uint64_t sweep_tiles, double sweep_gpu_ms, double sweep_sync_ms,
    const char* gemm_mode)
{
    const double total_ms = gemm_ex_ms + jackpot_ms + sync_ms;
    const double gemm_ex_sec = gemm_ex_ms * 1e-3;
    const double total_sec = total_ms * 1e-3;
    const size_t plane_int32s = (size_t)(g_row_period_batch * PP_ROW_PERIOD)
                              * (size_t)(g_col_period_batch * PP_COL_PERIOD);
    const double plane_mib = (double)plane_int32s * sizeof(int32_t) / (1024.0 * 1024.0);
    char gemm_ex_rate[32], total_rate[32], wall_rate[32];
    char sweep_wall_rate[32], sweep_gpu_rate[32];

    if(gemm_ex_sec > 0.0)
        cp_pp_fmt_mac_rate(macs_per_batch / gemm_ex_sec, gemm_ex_rate, sizeof(gemm_ex_rate));
    else
        snprintf(gemm_ex_rate, sizeof(gemm_ex_rate), "n/a");
    if(total_sec > 0.0)
        cp_pp_fmt_mac_rate(macs_per_batch / total_sec, total_rate, sizeof(total_rate));
    else
        snprintf(total_rate, sizeof(total_rate), "n/a");
    if(wall_ms > 0.0)
        cp_pp_fmt_mac_rate(macs_per_batch / (wall_ms * 1e-3), wall_rate, sizeof(wall_rate));
    else
        snprintf(wall_rate, sizeof(wall_rate), "n/a");
    if(sweep_ms > 0.0 && sweep_tiles > 0)
        cp_pp_fmt_mac_rate(
            cp_pp_mac_rate_from_tiles(sweep_tiles, sweep_ms * 1e-3),
            sweep_wall_rate, sizeof(sweep_wall_rate));
    else
        snprintf(sweep_wall_rate, sizeof(sweep_wall_rate), "n/a");
    if(sweep_gpu_ms > 0.0 && sweep_tiles > 0)
        cp_pp_fmt_mac_rate(
            cp_pp_mac_rate_from_tiles(sweep_tiles, sweep_gpu_ms * 1e-3),
            sweep_gpu_rate, sizeof(sweep_gpu_rate));
    else
        snprintf(sweep_gpu_rate, sizeof(sweep_gpu_rate), "n/a");

    printf("\n[profile-scan] %s  row_batch=%d col_batch=%d  runs=%d\n",
           gemm_mode, row_batch_count, col_batch_count, runs);
    printf("[profile-scan] MACs/batch: %.3f GMAC (%.3f TMAC)\n",
           macs_per_batch / 1e9, macs_per_batch / 1e12);
    printf("[profile-scan] C_hist: %.2f MiB/step x %d steps (partials; cumsum in jackpot)\n",
           plane_mib, K_DIM / R_RANK);
    printf("[profile-scan] avg per batch:\n");
    printf("  gemm_ex:   %7.3f ms  %5.1f%%  %s (16x GemmEx)\n",
           gemm_ex_ms, 100.0 * gemm_ex_ms / total_ms, gemm_ex_rate);
    printf("  jackpot:   %7.3f ms  %5.1f%%  (cumsum partials + BLAKE3)\n",
           jackpot_ms, 100.0 * jackpot_ms / total_ms);
    printf("  sync:      %7.3f ms  %5.1f%%  (DeviceSynchronize + found D2H)\n",
           sync_ms, 100.0 * sync_ms / total_ms);
    printf("  total:     %7.3f ms  %s (CUDA events, per-step GemmEx sync)\n",
           total_ms, total_rate);
    printf("[profile-scan] production batch (mining launch path, rpi=0):\n");
    printf("  batch:     %7.3f ms/batch  %s\n", wall_ms, wall_rate);
    if(sweep_batches > 0){
        const double avg_gpu_ms = sweep_gpu_ms / (double)sweep_batches;
        const double avg_wall_ms = sweep_ms / (double)sweep_batches;
        const double avg_sync_ms = sweep_sync_ms / (double)sweep_batches;
        printf("[profile-scan] full sweep (%d batches, mining loop shape):\n",
               sweep_batches);
        printf("  wall:     %8.1f ms total  %s  (%.3f ms/batch)\n",
               sweep_ms, sweep_wall_rate, avg_wall_ms);
        printf("  gpu:      %8.1f ms total  %s  (%.3f ms/batch)\n",
               sweep_gpu_ms, sweep_gpu_rate, avg_gpu_ms);
        printf("  sync:     %8.3f ms/batch   overhead %.3f ms/batch\n",
               avg_sync_ms, avg_wall_ms - avg_gpu_ms);
    }
    fflush(stdout);
}

int cp_gpu_run_scan_profile(int dev, int m, int n, int warmup, int runs)
{
    int devs[1] = {dev};
    uint8_t job_key[32];
    uint8_t a_key[32];
    uint32_t pool_tgt[8];
    uint32_t bound[8];
    cudaEvent_t ev[4];
    double prep_t0;
    double gemm_ex_sum = 0.0;
    double jackpot_sum = 0.0, sync_sum = 0.0;
    double wall_sum = 0.0;
    float sweep_ms = 0.f;
    int sweep_batches = 0;
    uint64_t sweep_tiles = 0;
    double sweep_gpu_ms = 0.0;
    double sweep_sync_ms = 0.0;
    ScanBatchTiming batch_tm = {0};
    int rc = -1;

    if(warmup < 0) warmup = 0;
    if(runs < 1) runs = 1;

    for(int i = 0; i < 32; i++) job_key[i] = (uint8_t)((i * 13 + 5) & 0xff);
    /*
     * Unbeatable (all-zero) target: found_flag stays 0 for the whole sweep,
     * so the jackpot kernel does full cumsum+BLAKE3 work every batch -- the
     * same path as live mining. An easy target would let found_flag latch and
     * make every later jackpot early-out (if(*found_flag) return), under-
     * measuring jackpot and making the sweep non-representative.
     */
    memset(pool_tgt, 0, sizeof(uint32_t) * 8);
    cp_scale_jackpot_target(pool_tgt, bound);

    const int col_periods = gpu_num_col_periods(n);
    const int row_periods = gpu_num_row_periods(m);
    int row_batch_count = g_row_period_batch;
    int col_batch_count = g_col_period_batch;
    if(row_batch_count > row_periods) row_batch_count = row_periods;
    if(col_batch_count > col_periods) col_batch_count = col_periods;
    if(row_batch_count < 1 || col_batch_count < 1){
        fprintf(stderr, "[profile-scan] invalid batch for m=%d n=%d\n", m, n);
        return -1;
    }

    const double macs_per_batch = (double)pp_batch_hash_tiles(
                                      row_batch_count, col_batch_count)
                                * cp_pp_macs_per_hash_tile();

    printf("[profile-scan] m=%d n=%d k=%d r=%d row_batch=%d col_batch=%d "
           "(panel %dx%d periods)\n",
           m, n, K_DIM, R_RANK, g_row_period_batch, g_col_period_batch,
           row_batch_count, col_batch_count);
    printf("[profile-scan] Ap/BpT: %s (cuBLAS lda=%d)\n",
           g_step_major_ap ? "step-major" : "row-major strided",
           g_step_major_ap ? R_RANK : K_DIM);
    printf("[profile-scan] warmup=%d timed=%d\n", warmup, runs);
    fflush(stdout);

    cp_gpu_init(devs, 1);
    GpuCtx* g = &g_gpus[0];
    ensure_buffers(g, m, n);
    CU_CHECK(cudaSetDevice(g->dev));
    scan_profile_ensure_events(ev);

    prep_t0 = cp_now_sec();
    if(gpu_prepare_noisy_matrices(g, cp_gpu_fresh_rng_seed(), job_key, m, n, a_key) != 0)
        goto done;
    printf("[profile-scan] matrix prep %.2fs (excluded from batch timings)\n",
           cp_now_sec() - prep_t0);
    fflush(stdout);

    {
        uint32_t a_key32[8];
        memcpy(a_key32, a_key, 32);
        int zero = 0;
        CU_CHECK(cudaMemcpy(g->d_a_key8, a_key32, 32, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }

    const char* gemm_mode = g->use_cutlass_fused ? "CUTLASS fused GEMM"
                            : (g->use_cublas_period ? "cuBLAS int8 fat"
                                                    : "CUDA period GEMM");
    const int rpi0 = 0;
    const int cpi0 = 0;

    for(int i = 0; i < warmup; i++)
        (void)profile_period_batch_timed(
            g, rpi0, cpi0, row_batch_count, col_batch_count, m, n, bound, ev);

    for(int i = 0; i < runs; i++){
        PeriodBatchTimes t = profile_period_batch_timed(
            g, rpi0, cpi0, row_batch_count, col_batch_count, m, n, bound, ev);
        gemm_ex_sum += t.gemm_ex_ms;
        jackpot_sum += t.jackpot_ms;
        sync_sum += t.sync_ms;
    }

    for(int i = 0; i < warmup; i++)
        (void)profile_period_batch_cuda_ms(
            g, rpi0, cpi0, row_batch_count, col_batch_count, m, n, bound,
            ev[0], ev[1]);

    for(int i = 0; i < runs; i++)
        wall_sum += profile_period_batch_cuda_ms(
            g, rpi0, cpi0, row_batch_count, col_batch_count, m, n, bound,
            ev[0], ev[1]);

    /*
     * Full-scan sweep: identical loop shape to gpu_scan_device_period
     * (nested rpi x cpi0, same launch/sync pattern, same ScanBatchTiming),
     * so profile numbers are directly comparable to live mining.
     */
    {
        const int row_periods = gpu_num_row_periods(m);
        const int col_periods = gpu_num_col_periods(n);
        const int row_parts = cp_pp_num_row_parts(m, g_contiguous);
        const int col_parts = cp_pp_num_col_parts(n, g_contiguous);
        const int total_tiles = row_parts * col_parts;
        uint64_t tiles_scanned = 0;
        double sweep_t0 = cp_now_sec();

        int zero = 0;
        CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));

        scan_batch_timing_init(&batch_tm);
        printf("[profile-scan] full sweep (mining loop shape): "
               "%d row periods x %d col periods, row_batch=%d col_batch=%d\n",
               row_periods, col_periods, g_row_period_batch, g_col_period_batch);
        fflush(stdout);

        for(int rpi0 = 0; rpi0 < row_periods; rpi0 += g_row_period_batch){
            int rb = g_row_period_batch;
            if(rpi0 + rb > row_periods) rb = row_periods - rpi0;
            for(int cpi0 = 0; cpi0 < col_periods; cpi0 += g_col_period_batch){
                int cb = g_col_period_batch;
                if(cpi0 + cb > col_periods) cb = col_periods - cpi0;

                const int batch_tiles = pp_batch_hash_tiles(rb, cb);
                const double wall_t0 = cp_now_sec();
                CU_CHECK(cudaSetDevice(g->dev));
                CU_CHECK(cudaEventRecord(batch_tm.batch_start));

                gpu_period_gemm_batch(g, m, n, rpi0, cpi0, rb, cb, bound);
                launch_jackpot_batch(g, rb, cb, rpi0, cpi0, m, n, bound);

                CU_CHECK(cudaEventRecord(batch_tm.post_launch));
                CU_CHECK(cudaDeviceSynchronize());
                int f = 0;
                CU_CHECK(cudaMemcpy(&f, g->d_found, sizeof(int),
                                    cudaMemcpyDeviceToHost));
                (void)f;

                const double wall_ms = (cp_now_sec() - wall_t0) * 1000.0;
                (void)scan_batch_timing_finish(&batch_tm, batch_tiles, wall_ms);
                tiles_scanned += (uint64_t)batch_tiles;
            }
            if((rpi0 / g_row_period_batch) % 16 == 0){
                double sweep_sec = cp_now_sec() - sweep_t0;
                if(sweep_sec < 1e-9) sweep_sec = 1e-9;
                char wall_buf[32], gpu_buf[32];
                cp_pp_fmt_mac_rate(
                    cp_pp_mac_rate_from_tiles(tiles_scanned, sweep_sec),
                    wall_buf, sizeof(wall_buf));
                if(batch_tm.gpu_ms_sum > 0.0)
                    cp_pp_fmt_mac_rate(
                        cp_pp_mac_rate_from_tiles(batch_tm.batch_tiles,
                                                  batch_tm.gpu_ms_sum * 1e-3),
                        gpu_buf, sizeof(gpu_buf));
                else
                    snprintf(gpu_buf, sizeof(gpu_buf), "n/a");
                printf("[profile-scan] sweep progress: row periods %d/%d "
                       "tiles %llu/%d (%.1f%%) %s wall | %s gpu\n",
                       rpi0 + rb, row_periods,
                       (unsigned long long)tiles_scanned, total_tiles,
                       100.0 * (double)tiles_scanned / (double)total_tiles,
                       wall_buf, gpu_buf);
                fflush(stdout);
            }
        }

        sweep_ms = (float)((cp_now_sec() - sweep_t0) * 1000.0);
        sweep_batches = (int)batch_tm.batches;
        sweep_tiles = batch_tm.batch_tiles;
        sweep_gpu_ms = batch_tm.gpu_ms_sum;
        sweep_sync_ms = batch_tm.sync_ms_sum;
        scan_batch_timing_destroy(&batch_tm);
    }

    scan_profile_print_summary(
        row_batch_count, col_batch_count, runs, macs_per_batch,
        gemm_ex_sum / runs, jackpot_sum / runs, sync_sum / runs,
        wall_sum / runs, (double)sweep_ms, sweep_batches,
        sweep_tiles, sweep_gpu_ms, sweep_sync_ms, gemm_mode);
    rc = 0;

done:
    cp_gpu_shutdown();
    return rc;
}

static int gpu_scan_device_period(
    const uint8_t* a_key, const uint32_t pool_tgt[8],
    int m, int n,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned)
{
    uint32_t bound[8];
    cp_scale_jackpot_target(pool_tgt, bound);

    const int row_periods = gpu_num_row_periods(m);
    const int col_periods = gpu_num_col_periods(n);
    const int row_parts = cp_pp_num_row_parts(m, g_contiguous);
    const int col_parts = cp_pp_num_col_parts(n, g_contiguous);
    const int total_tiles = row_parts * col_parts;
    int found = 0;
    uint64_t tiles_scanned = 0;
    double scan_t0 = cp_now_sec();

    if(out_tiles_scanned) *out_tiles_scanned = 0;

    printf("[gpu] plain_proof period-GEMM scan %dx%d periods "
           "(row_batch=%d col_batch=%d, %d hash tiles), difficulty scaled by %d\n",
           row_periods, col_periods, g_row_period_batch, g_col_period_batch,
           total_tiles,
           (g_cutlass_fused ? CP_CUTLASS_HASH_H * CP_CUTLASS_HASH_W
                            : PP_HASH_H * PP_HASH_W) * K_DIM);
    fflush(stdout);

    for(int rpi0 = 0; rpi0 < row_periods && !found; rpi0 += g_row_period_batch){
        if(cp_job_should_cancel()){
            if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
            return -1;
        }
        int row_batch = g_row_period_batch;
        if(rpi0 + row_batch > row_periods)
            row_batch = row_periods - rpi0;

        for(int cpi0 = 0; cpi0 < col_periods && !found; cpi0 += g_col_period_batch){
            int col_batch = g_col_period_batch;
            if(cpi0 + col_batch > col_periods)
                col_batch = col_periods - cpi0;

            const int batch_tiles = pp_batch_hash_tiles(row_batch, col_batch);

            for(int i = 0; i < g_ngpu; i++){
                GpuCtx* g = &g_gpus[i];
                CU_CHECK(cudaSetDevice(g->dev));
                gpu_period_gemm_batch(
                    g, m, n, rpi0, cpi0, row_batch, col_batch, bound);
                launch_jackpot_batch(
                    g, row_batch, col_batch, rpi0, cpi0, m, n, bound);
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

            tiles_scanned += (uint64_t)batch_tiles;
        }
        if(rpi0 % 128 == 0 && !found){
            double scan_sec = cp_now_sec() - scan_t0;
            if(scan_sec < 1e-9) scan_sec = 1e-9;
            double scan_mac_s = cp_pp_mac_rate_from_tiles(tiles_scanned, scan_sec);
            char mac_buf[32];
            cp_pp_fmt_mac_rate(scan_mac_s, mac_buf, sizeof(mac_buf));
            printf("[gpu] plain_proof progress: row periods %d/%d tiles %llu/%d (%.1f%%) %s\n",
                   rpi0 + row_batch, row_periods,
                   (unsigned long long)tiles_scanned, total_tiles,
                   100.0 * (double)tiles_scanned / (double)total_tiles, mac_buf);
            fflush(stdout);
        }
    }
    if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
    return found;
}

static int gpu_scan_device(
    const uint8_t* a_key, const uint32_t pool_tgt[8],
    int m, int n,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned)
{
    if(g_period_gemm && !g_contiguous)
        return gpu_scan_device_period(a_key, pool_tgt, m, n,
                                      out_t_rows, out_t_cols, out_tiles_scanned);

    uint32_t bound[8];
    cp_scale_jackpot_target(pool_tgt, bound);

    const int row_parts = cp_pp_num_row_parts(m, g_contiguous);
    const int col_parts = cp_pp_num_col_parts(n, g_contiguous);
    const int batch = 64;
    dim3 block(PP_HASH_W, PP_HASH_H);
    int found = 0;
    const int total_tiles = row_parts * col_parts;
    uint64_t tiles_scanned = 0;
    double scan_t0 = cp_now_sec();

    if(out_tiles_scanned) *out_tiles_scanned = 0;

    printf("[gpu] plain_proof scan %dx%d hash tiles, difficulty scaled by %d\n",
           row_parts, col_parts, PP_HASH_H * PP_HASH_W * K_DIM);
    //printf("[gpu] jackpot target LE: %08X %08X ...\n", bound[0], bound[1]);
    fflush(stdout);

    for(int rp0 = 0; rp0 < row_parts && !found; rp0 += batch){
        if(cp_job_should_cancel()){
            if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
            return -1;
        }
        int rpb = batch;
        if(rp0 + rpb > row_parts) rpb = row_parts - rp0;
        for(int cp0 = 0; cp0 < col_parts && !found; cp0 += batch){
            if(cp_job_should_cancel()){
                if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
                return -1;
            }
            int cpb = batch;
            if(cp0 + cpb > col_parts) cpb = col_parts - cp0;
            dim3 grid(cpb, rpb);
            const uint64_t batch_tiles = (uint64_t)rpb * (uint64_t)cpb;

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

            tiles_scanned += batch_tiles;
        }
        if((rp0 / batch) % 4 == 0 && !found){
            double scan_sec = cp_now_sec() - scan_t0;
            if(scan_sec < 1e-9) scan_sec = 1e-9;
            double scan_mac_s = cp_pp_mac_rate_from_tiles(tiles_scanned, scan_sec);
            char mac_buf[32];
            cp_pp_fmt_mac_rate(scan_mac_s, mac_buf, sizeof(mac_buf));
            printf("[gpu] plain_proof progress: row parts %d/%d tiles %llu/%d (%.1f%%) %s\n",
                   rp0 + rpb, row_parts,
                   (unsigned long long)tiles_scanned, total_tiles,
                   100.0 * (double)tiles_scanned / (double)total_tiles, mac_buf);
            fflush(stdout);
        }
    }
    if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
    return found;
}

int cp_gpu_mine_plain_proof(
    const int8_t* h_A, const int8_t* h_B,
    const uint8_t* a_key, const uint32_t pool_tgt[8],
    int m, int n,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned)
{
    size_t szAp  = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    uint32_t a_key32[8];
    memcpy(a_key32, a_key, 32);
    int zero = 0;

    sync_tile_config();
    for(int i = 0; i < g_ngpu; i++){
        GpuCtx* g = &g_gpus[i];
        ensure_buffers(g, m, n);
        CU_CHECK(cudaSetDevice(g->dev));
        gpu_upload_rowmajor_noisy(g, h_A, h_B, m, n);
        CU_CHECK(cudaMemcpy(g->d_a_key8, a_key32, 32, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }
    return gpu_scan_device(a_key, pool_tgt, m, n, out_t_rows, out_t_cols, out_tiles_scanned);
}

int cp_gpu_mine_attempt(
    const uint8_t* ab_seed, int ab_seed_len,
    const uint8_t job_key[32],
    const uint32_t pool_tgt[8],
    int m, int n,
    int cpu_matrices,
    const int8_t* h_A_noisy, const int8_t* h_B_noisy,
    const uint8_t* a_key,
    int8_t* h_A_sig, int8_t* h_Bt_sig,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned)
{
    if(g_ngpu <= 0) return -1;
    (void)h_A_sig;
    (void)h_Bt_sig;
    const double attempt_t0 = cp_now_sec();
    size_t szAp = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    uint8_t a_key_local[32];
    const uint8_t* scan_key = a_key;
    int zero = 0;

    sync_tile_config();
    GpuCtx* g0 = &g_gpus[0];
    ensure_buffers(g0, m, n);

    if(cpu_matrices){
        if(!h_A_noisy || !h_B_noisy || !a_key) return -1;
        scan_key = a_key;
        for(int i = 0; i < g_ngpu; i++){
            GpuCtx* g = &g_gpus[i];
            ensure_buffers(g, m, n);
            CU_CHECK(cudaSetDevice(g->dev));
            CU_CHECK(cudaMemcpy(g->d_a_key8, a_key, 32, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
            gpu_upload_rowmajor_noisy(g, h_A_noisy, h_B_noisy, m, n);
        }
    } else {
        if(gpu_prepare_noisy_matrices(g0, cp_gpu_fresh_rng_seed(), job_key, m, n,
                                      a_key_local) != 0)
            return -1;
        scan_key = a_key_local;
        uint32_t a_key32[8];
        memcpy(a_key32, scan_key, 32);
        CU_CHECK(cudaSetDevice(g0->dev));
        CU_CHECK(cudaMemcpy(g0->d_a_key8, a_key32, 32, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g0->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
        for(int i = 1; i < g_ngpu; i++){
            GpuCtx* g = &g_gpus[i];
            ensure_buffers(g, m, n);
            CU_CHECK(cudaSetDevice(g->dev));
            CU_CHECK(cudaMemcpy(g->d_Ap, g0->d_Ap, szAp, cudaMemcpyDeviceToDevice));
            CU_CHECK(cudaMemcpy(g->d_BpT, g0->d_BpT, szBpT, cudaMemcpyDeviceToDevice));
            CU_CHECK(cudaMemcpy(g->d_a_key8, a_key32, 32, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    const double prep_sec = cp_now_sec() - attempt_t0;
    const double scan_t0 = cp_now_sec();
    int found = gpu_scan_device(scan_key, pool_tgt, m, n, out_t_rows, out_t_cols, out_tiles_scanned);
    const double scan_sec = cp_now_sec() - scan_t0;
    /* Device→host download deferred to cp_gpu_fetch_share_signals after host buffer reclaim. */
    const uint64_t tiles_done = out_tiles_scanned ? *out_tiles_scanned : 0;
    cp_log_attempt_timing("gpu", prep_sec, scan_sec, tiles_done, 0.0);
    return found;
}

int cp_gpu_fetch_share_signals(int8_t* h_A_sig, int8_t* h_Bt_sig)
{
    if(g_ngpu <= 0 || !h_A_sig) return -1;
    GpuCtx* g0 = &g_gpus[0];
    if(!g0->d_A_sig) return -1;
    const int m = g_m_active;
    const int n = g_n_active;
    if(m <= 0 || n <= 0) return -1;
    size_t szAp = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    CU_CHECK(cudaSetDevice(g0->dev));
    CU_CHECK(cudaMemcpy(h_A_sig, g0->d_A_sig, szAp, cudaMemcpyDeviceToHost));
    if(h_Bt_sig && g0->d_Bt_sig){
        CU_CHECK(cudaMemcpy(h_Bt_sig, g0->d_Bt_sig, szBpT, cudaMemcpyDeviceToHost));
    }
    return 0;
}
