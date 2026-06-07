#include "cp_gpu.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_util.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "cp_gpu.cuh"
#include "cp_gpu_gen.cuh"
#include "cp_noise.h"
#include "plain_proof_kernel.cuh"
#include "plain_proof_period.cuh"

#define CU_CHECK(call) do { \
    cudaError_t _e = (call); \
    if(_e != cudaSuccess){ \
        fprintf(stderr,"[CUDA] %s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _e = (call); \
    if(_e != CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"[CUBLAS] %s:%d %s: status %d\n",__FILE__,__LINE__,#call,(int)_e); \
        exit(1); \
    } \
} while(0)

typedef struct {
    int       dev;
    int8_t*   d_Ap;
    int8_t*   d_BpT;
    int8_t*   d_A_sig;
    int8_t*   d_Bt_sig;
    uint32_t* d_e_ar;
    uint32_t* d_e_bl;
    uint8_t*  d_chunk_cvs;
    uint8_t*  h_chunk_cvs;
    uint8_t*  d_seed_a;
    uint8_t*  d_seed_b;
    uint8_t*  d_job_key;
    size_t    chunk_cv_cap;
    int*      d_found;
    int*      d_out_t_rows;
    int*      d_out_t_cols;
    uint32_t* d_a_key8;
    int32_t*  d_C_hist;
    size_t    C_hist_cap;
    cublasHandle_t cublas;
    int       use_cublas_period;
} GpuCtx;

static GpuCtx g_gpus[MAX_GPUS];
static int g_ngpu = 0;
static int g_contiguous = 0;
static int g_period_gemm = 1;
static int g_period_batch = CP_PERIOD_BATCH_DEFAULT;

static size_t pp_hist_batch_int32s(int batch_count)
{
    return (size_t)(K_DIM / R_RANK) * (size_t)PP_ROW_PERIOD
         * (size_t)batch_count * (size_t)PP_COL_PERIOD;
}

static size_t pp_hist_batch_bytes(int batch_count)
{
    return pp_hist_batch_int32s(batch_count) * sizeof(int32_t);
}

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

static void pp_launch_plane_add(int32_t* dst, const int32_t* prev, int n)
{
    const int tpb = 256;
    const int blocks = (n + tpb - 1) / tpb;
    plain_proof_plane_add_kernel<<<blocks, tpb>>>(dst, prev, n);
    CU_CHECK(cudaGetLastError());
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
    CU_CHECK(cudaMalloc(&dA, (size_t)M * (size_t)K_DIM));
    CU_CHECK(cudaMalloc(&dB, (size_t)2 * (size_t)N * (size_t)K_DIM));
    CU_CHECK(cudaMalloc(&dC, (size_t)M * (size_t)N * 2 * sizeof(int32_t)));
    CU_CHECK(cudaMemset(dA, 0, (size_t)M * (size_t)K_DIM));
    CU_CHECK(cudaMemset(dB, 0, (size_t)2 * (size_t)N * (size_t)K_DIM));

    st = pp_cublas_gemm_i8_bt(g->cublas, dA, K_DIM, dB, K_DIM, dC, N, M, N, R, 0);
    if(st == CUBLAS_STATUS_SUCCESS){
        st = pp_cublas_gemm_i8_bt(
            g->cublas, dA, K_DIM, dB, K_DIM, dC, N * 2, M, N * 2, R, 0);
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
    if(batch < 1) batch = 1;
    if(batch > CP_PERIOD_BATCH_MAX) batch = CP_PERIOD_BATCH_MAX;
    g_period_batch = batch;
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
        CUBLAS_CHECK(cublasCreate(&g->cublas));
        g->use_cublas_period = gpu_probe_cublas_int8(g);
        printf("[gpu] GPU%d OK (period GEMM: %s)\n", g->dev,
               g->use_cublas_period ? "cuBLAS int8 fat" : "CUDA __dp4a");
        fflush(stdout);
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
        if(g->d_A_sig) cudaFree(g->d_A_sig);
        if(g->d_Bt_sig) cudaFree(g->d_Bt_sig);
        if(g->d_e_ar) cudaFree(g->d_e_ar);
        if(g->d_e_bl) cudaFree(g->d_e_bl);
        if(g->d_chunk_cvs) cudaFree(g->d_chunk_cvs);
        if(g->d_seed_a) cudaFree(g->d_seed_a);
        if(g->d_seed_b) cudaFree(g->d_seed_b);
        if(g->d_job_key) cudaFree(g->d_job_key);
        free(g->h_chunk_cvs);
        g->h_chunk_cvs = NULL;
        g->chunk_cv_cap = 0;
        if(g->d_found) cudaFree(g->d_found);
        if(g->d_out_t_rows) cudaFree(g->d_out_t_rows);
        if(g->d_out_t_cols) cudaFree(g->d_out_t_cols);
        if(g->d_a_key8) cudaFree(g->d_a_key8);
        if(g->d_C_hist) cudaFree(g->d_C_hist);
        if(g->cublas){ cublasDestroy(g->cublas); g->cublas = NULL; }
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
    size_t cv_need = (chunks_a > chunks_b ? chunks_a : chunks_b) * 32;

    CU_CHECK(cudaSetDevice(g->dev));
    if(!g->d_Ap){
        CU_CHECK(cudaMalloc(&g->d_Ap, szAp));
        CU_CHECK(cudaMalloc(&g->d_BpT, szBpT));
        CU_CHECK(cudaMalloc(&g->d_A_sig, szAp));
        CU_CHECK(cudaMalloc(&g->d_Bt_sig, szBpT));
        CU_CHECK(cudaMalloc(&g->d_e_ar, (size_t)K_DIM * 2 * sizeof(uint32_t)));
        CU_CHECK(cudaMalloc(&g->d_e_bl, (size_t)K_DIM * 2 * sizeof(uint32_t)));
        CU_CHECK(cudaMalloc(&g->d_seed_a, 32));
        CU_CHECK(cudaMalloc(&g->d_seed_b, 32));
        CU_CHECK(cudaMalloc(&g->d_job_key, 32));
    }
    if(cv_need > g->chunk_cv_cap){
        if(g->d_chunk_cvs) cudaFree(g->d_chunk_cvs);
        free(g->h_chunk_cvs);
        CU_CHECK(cudaMalloc(&g->d_chunk_cvs, cv_need));
        g->h_chunk_cvs = (uint8_t*)malloc(cv_need);
        if(!g->h_chunk_cvs){
            fprintf(stderr, "[gpu] OOM chunk CV host buffer\n");
            exit(1);
        }
        g->chunk_cv_cap = cv_need;
    }
    {
        size_t hist_need = pp_hist_batch_bytes(g_period_batch);
        if(hist_need > g->C_hist_cap){
            if(g->d_C_hist) cudaFree(g->d_C_hist);
            CU_CHECK(cudaMalloc(&g->d_C_hist, hist_need));
            g->C_hist_cap = hist_need;
        }
    }
}

/*
 * cuBLAS: one fat GemmEx per rank step into C_hist[step][128][batch*256],
 * then plane add for cumulative history.
 */
static void gpu_period_gemm_cublas_batch(
    GpuCtx* g, int row_period, int col_period0, int batch_count)
{
    const int M = PP_ROW_PERIOD;
    const int N = PP_COL_PERIOD;
    const int R = R_RANK;
    const int num_steps = K_DIM / R;
    const int N_fat = batch_count * N;
    const size_t step_plane = (size_t)M * (size_t)N_fat;
    const size_t row_base = (size_t)row_period * (size_t)M;
    const size_t col_base = (size_t)col_period0 * (size_t)N;
    const int8_t* A0 = g->d_Ap + row_base * (size_t)K_DIM;
    const int8_t* B0 = g->d_BpT + col_base * (size_t)K_DIM;

    for(int s = 0; s < num_steps; s++){
        const int8_t* Ap = A0 + (size_t)s * (size_t)R;
        const int8_t* Bp0 = B0 + (size_t)s * (size_t)R;
        int32_t* Cp = g->d_C_hist + (size_t)s * step_plane;

        cublasStatus_t st = pp_cublas_gemm_i8_bt(
            g->cublas, Ap, K_DIM, Bp0, K_DIM, Cp, N_fat, M, N_fat, R, 0);
        if(st != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "[CUBLAS] GemmEx failed: %s (%d)\n",
                    cublas_status_str(st), (int)st);
            exit(1);
        }

        if(s > 0){
            pp_launch_plane_add(Cp, g->d_C_hist + (size_t)(s - 1) * step_plane,
                                (int)step_plane);
        }
    }
}

static void gpu_period_gemm_cuda_batch(
    GpuCtx* g, int row_period, int col_period0, int batch_count)
{
    const dim3 grid(PP_COL_PERIOD / 16, PP_ROW_PERIOD / 16);
    const dim3 block(16, 16);

    for(int b = 0; b < batch_count; b++){
        plain_proof_period_gemm_kernel<<<grid, block>>>(
            g->d_Ap, g->d_BpT,
            K_DIM, R_RANK,
            row_period, col_period0 + b,
            b, batch_count,
            g->d_C_hist);
    }
    CU_CHECK(cudaGetLastError());
}

static void gpu_period_gemm_batch(
    GpuCtx* g, int row_period, int col_period0, int batch_count)
{
    if(g->use_cublas_period)
        gpu_period_gemm_cublas_batch(g, row_period, col_period0, batch_count);
    else
        gpu_period_gemm_cuda_batch(g, row_period, col_period0, batch_count);
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
        const int tpb = 256;
        int grid = (num_chunks + tpb - 1) / tpb;
        cp_keyed_chunk_cv_kernel<<<grid, tpb>>>(
            (const uint8_t*)d_mat, raw_len, pad_len, g->d_job_key,
            g->d_chunk_cvs, num_chunks);
    }
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    size_t cv_bytes = (size_t)num_chunks * 32;
    CU_CHECK(cudaMemcpy(g->h_chunk_cvs, g->d_chunk_cvs, cv_bytes, cudaMemcpyDeviceToHost));
    return pearl_root_from_chunk_cvs(g->h_chunk_cvs, num_chunks, job_key, out);
}

static uint64_t cp_gpu_fresh_rng_seed(void)
{
    uint64_t s = (uint64_t)(cp_now_sec() * 1e9);
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

    CU_CHECK(cudaSetDevice(g->dev));

    cp_gen_random_matrix_kernel<<<(total_a + tpb - 1) / tpb, tpb>>>(
        rng_seed, 0, total_a, g->d_A_sig);
    cp_gen_random_matrix_kernel<<<(total_b + tpb - 1) / tpb, tpb>>>(
        rng_seed, 1, total_b, g->d_Bt_sig);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());

    if(cp_job_should_cancel()) return -1;

    /* Keyed commitment hash: GPU computes per-chunk CVs; D2H ~16 MiB/matrix
     * 512MiB matrix size / 1KiB chunk size = 524288 chunks
     * (524288 chunks x 32 B at prod) for CPU Merkle root; then 32-byte hash_a/b. */
    if(gpu_matrix_keyed_hash(g, g->d_A_sig, szAp, pad_a, job_key, hash_a) != 0) return -1;
    if(gpu_matrix_keyed_hash(g, g->d_Bt_sig, szBpT, pad_b, job_key, hash_b) != 0) return -1;

    pearl_derive_noise_seeds(job_key, hash_a, hash_b, b_seed, a_key_out);

    CU_CHECK(cudaMemcpy(g->d_seed_a, a_key_out, 32, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(g->d_seed_b, b_seed, 32, cudaMemcpyHostToDevice));

    cp_build_perm_pairs_kernel<<<1, 1>>>(0, g->d_seed_a, K_DIM, R_RANK, g->d_e_ar);
    cp_build_perm_pairs_kernel<<<1, 1>>>(1, g->d_seed_b, K_DIM, R_RANK, g->d_e_bl);
    CU_CHECK(cudaGetLastError());

    cp_fuse_noise_a_kernel<<<m, tpb, (size_t)K_DIM>>>(
        g->d_A_sig, g->d_Ap, m, K_DIM, R_RANK, g->d_seed_a, g->d_e_ar);
    cp_fuse_noise_b_kernel<<<n, tpb, (size_t)K_DIM>>>(
        g->d_Bt_sig, g->d_BpT, n, K_DIM, R_RANK, g->d_seed_b, g->d_e_bl);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
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
    cp_build_perm_pairs_kernel<<<1, 1>>>(0, g->d_seed_a, K_DIM, R_RANK, g->d_e_ar);
    cp_build_perm_pairs_kernel<<<1, 1>>>(1, g->d_seed_b, K_DIM, R_RANK, g->d_e_bl);
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
    cp_fuse_noise_a_kernel<<<m, tpb, (size_t)K_DIM>>>(
        g->d_A_sig, g->d_Ap, m, K_DIM, R_RANK, g->d_seed_a, g->d_e_ar);
    cp_fuse_noise_b_kernel<<<n, tpb, (size_t)K_DIM>>>(
        g->d_Bt_sig, g->d_BpT, n, K_DIM, R_RANK, g->d_seed_b, g->d_e_bl);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    printf("[align-test-prod] GPU noise fuse %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    {
        static const int sample_rows[] = {0, 1, 17, 4096, 8192};
        for(size_t si = 0; si < sizeof(sample_rows) / sizeof(sample_rows[0]); si++){
            int row = sample_rows[si];
            if(row >= m) continue;
            size_t off = (size_t)row * (size_t)K_DIM;
            CU_CHECK(cudaMemcpy(gpu_row, g->d_Ap + off, (size_t)K_DIM, cudaMemcpyDeviceToHost));
            pearl_fuse_noise_row_a(row, K_DIM, R_RANK, a_key_gpu, h_e_ar,
                                   h_A + off, cpu_row);
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
    float gemm_ms;
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
    GpuCtx* g, int batch_count, int rpi, int cpi0, int m, int n,
    const uint32_t bound[8])
{
    plain_proof_period_jackpot_kernel<<<batch_count, PP_TILES_PER_PERIOD>>>(
        g->d_C_hist,
        batch_count,
        K_DIM, R_RANK,
        rpi, cpi0,
        m, n,
        bound[0], bound[1], bound[2], bound[3],
        bound[4], bound[5], bound[6], bound[7],
        g->d_a_key8,
        g->d_out_t_rows, g->d_out_t_cols, g->d_found);
    CU_CHECK(cudaGetLastError());
}

static PeriodBatchTimes profile_period_batch_timed(
    GpuCtx* g, int rpi, int cpi0, int batch_count, int m, int n,
    const uint32_t bound[8], cudaEvent_t ev[4])
{
    PeriodBatchTimes t = {0.f, 0.f, 0.f};
    int zero = 0;
    CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));

    CU_CHECK(cudaEventRecord(ev[0]));
    gpu_period_gemm_batch(g, rpi, cpi0, batch_count);
    CU_CHECK(cudaEventRecord(ev[1]));
    CU_CHECK(cudaEventSynchronize(ev[1]));
    CU_CHECK(cudaEventElapsedTime(&t.gemm_ms, ev[0], ev[1]));

    launch_jackpot_batch(g, batch_count, rpi, cpi0, m, n, bound);
    CU_CHECK(cudaEventRecord(ev[2]));
    CU_CHECK(cudaEventSynchronize(ev[2]));
    CU_CHECK(cudaEventElapsedTime(&t.jackpot_ms, ev[1], ev[2]));

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

static void scan_profile_print_summary(
    int batch_count, int runs, double macs_per_batch,
    double gemm_ms, double jackpot_ms, double sync_ms, const char* gemm_mode)
{
    const double total_ms = gemm_ms + jackpot_ms + sync_ms;
    const double gemm_sec = gemm_ms * 1e-3;
    const double total_sec = total_ms * 1e-3;
    char gemm_rate[32], total_rate[32];

    if(gemm_sec > 0.0)
        cp_pp_fmt_mac_rate(macs_per_batch / gemm_sec, gemm_rate, sizeof(gemm_rate));
    else
        snprintf(gemm_rate, sizeof(gemm_rate), "n/a");
    if(total_sec > 0.0)
        cp_pp_fmt_mac_rate(macs_per_batch / total_sec, total_rate, sizeof(total_rate));
    else
        snprintf(total_rate, sizeof(total_rate), "n/a");

    printf("\n[profile-scan] %s  batch_count=%d  runs=%d\n",
           gemm_mode, batch_count, runs);
    printf("[profile-scan] MACs/batch: %.3f GMAC (%.3f TMAC)\n",
           macs_per_batch / 1e9, macs_per_batch / 1e12);
    printf("[profile-scan] avg per batch:\n");
    printf("  gemm:    %7.3f ms  %5.1f%%  %s (GEMM-only)\n",
           gemm_ms, 100.0 * gemm_ms / total_ms, gemm_rate);
    printf("  jackpot: %7.3f ms  %5.1f%%\n",
           jackpot_ms, 100.0 * jackpot_ms / total_ms);
    printf("  sync:    %7.3f ms  %5.1f%%  (DeviceSynchronize + found D2H)\n",
           sync_ms, 100.0 * sync_ms / total_ms);
    printf("  total:   %7.3f ms  %s (scan-equivalent)\n", total_ms, total_rate);
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
    double gemm_sum = 0.0, jackpot_sum = 0.0, sync_sum = 0.0;
    int rc = -1;

    if(warmup < 0) warmup = 0;
    if(runs < 1) runs = 1;

    for(int i = 0; i < 32; i++) job_key[i] = (uint8_t)((i * 13 + 5) & 0xff);
    cp_target_from_difficulty(1.0, pool_tgt);
    cp_scale_jackpot_target(pool_tgt, bound);

    const int col_periods = cp_pp_num_col_periods(n, g_contiguous);
    int batch_count = g_period_batch;
    if(batch_count > col_periods) batch_count = col_periods;
    if(batch_count < 1){
        fprintf(stderr, "[profile-scan] invalid batch_count for m=%d n=%d\n", m, n);
        return -1;
    }

    const double macs_per_batch = (double)batch_count
                                * (double)PP_TILES_PER_PERIOD
                                * cp_pp_macs_per_hash_tile();

    printf("[profile-scan] m=%d n=%d k=%d r=%d period_batch=%d (using batch_count=%d)\n",
           m, n, K_DIM, R_RANK, g_period_batch, batch_count);
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

    const char* gemm_mode = g->use_cublas_period ? "cuBLAS int8 fat" : "CUDA period GEMM";
    const int rpi = 0;
    const int cpi0 = 0;

    for(int i = 0; i < warmup; i++)
        (void)profile_period_batch_timed(g, rpi, cpi0, batch_count, m, n, bound, ev);

    for(int i = 0; i < runs; i++){
        PeriodBatchTimes t = profile_period_batch_timed(
            g, rpi, cpi0, batch_count, m, n, bound, ev);
        gemm_sum += t.gemm_ms;
        jackpot_sum += t.jackpot_ms;
        sync_sum += t.sync_ms;
    }

    scan_profile_print_summary(
        batch_count, runs, macs_per_batch,
        gemm_sum / runs, jackpot_sum / runs, sync_sum / runs, gemm_mode);
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

    const int row_periods = cp_pp_num_row_periods(m, g_contiguous);
    const int col_periods = cp_pp_num_col_periods(n, g_contiguous);
    const int row_parts = cp_pp_num_row_parts(m, g_contiguous);
    const int col_parts = cp_pp_num_col_parts(n, g_contiguous);
    const int total_tiles = row_parts * col_parts;
    int found = 0;
    uint64_t tiles_scanned = 0;
    double scan_t0 = cp_now_sec();

    if(out_tiles_scanned) *out_tiles_scanned = 0;

    const char* gemm_mode = g_gpus[0].use_cublas_period ? "cuBLAS fat" : "CUDA";

    printf("[gpu] plain_proof period-GEMM scan %dx%d periods "
           "(batch=%d, %s, %d hash tiles), difficulty scaled by %d\n",
           row_periods, col_periods, g_period_batch,
           gemm_mode,
           total_tiles, PP_HASH_H * PP_HASH_W * K_DIM);
    fflush(stdout);

    for(int rpi = 0; rpi < row_periods && !found; rpi++){
        if(cp_job_should_cancel()){
            if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
            return -1;
        }
        // column-batched GEMM
        // 64MiB to store the C_hist for each batch
        for(int cpi0 = 0; cpi0 < col_periods && !found; cpi0 += g_period_batch){
            if(cp_job_should_cancel()){
                if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
                return -1;
            }
            int batch_count = g_period_batch;
            if(cpi0 + batch_count > col_periods)
                batch_count = col_periods - cpi0;

            // GEMM and jackpot scan
            for(int i = 0; i < g_ngpu; i++){
                GpuCtx* g = &g_gpus[i];
                CU_CHECK(cudaSetDevice(g->dev));
                gpu_period_gemm_batch(g, rpi, cpi0, batch_count);
                launch_jackpot_batch(g, batch_count, rpi, cpi0, m, n, bound);
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

            tiles_scanned += (uint64_t)batch_count * (uint64_t)PP_TILES_PER_PERIOD;
        }
        if((rpi % 4) == 0 && !found){
            double scan_sec = cp_now_sec() - scan_t0;
            if(scan_sec < 1e-9) scan_sec = 1e-9;
            double scan_mac_s = cp_pp_mac_rate_from_tiles(tiles_scanned, scan_sec);
            char mac_buf[32];
            cp_pp_fmt_mac_rate(scan_mac_s, mac_buf, sizeof(mac_buf));
            printf("[gpu] plain_proof progress: row periods %d/%d tiles %llu/%d (%.1f%%) %s\n",
                   rpi + 1, row_periods,
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
        CU_CHECK(cudaMemcpy(g->d_Ap,  h_A,  szAp,  cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(g->d_BpT, h_B,  szBpT, cudaMemcpyHostToDevice));
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
            CU_CHECK(cudaMemcpy(g->d_Ap, h_A_noisy, szAp, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(g->d_BpT, h_B_noisy, szBpT, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(g->d_a_key8, a_key, 32, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
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

    int found = gpu_scan_device(scan_key, pool_tgt, m, n, out_t_rows, out_t_cols, out_tiles_scanned);
    if(found == 1 && h_A_sig && h_Bt_sig && !cpu_matrices){
        CU_CHECK(cudaSetDevice(g0->dev));
        CU_CHECK(cudaMemcpy(h_A_sig, g0->d_A_sig, szAp, cudaMemcpyDeviceToHost));
        CU_CHECK(cudaMemcpy(h_Bt_sig, g0->d_Bt_sig, szBpT, cudaMemcpyDeviceToHost));
    }
    return found;
}
