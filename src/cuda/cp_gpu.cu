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

#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

#include "cp_gpu.cuh"
#include "cp_gpu_gen.cuh"
#include "cp_noise_phase.cuh"
#include "cp_merkle_tree.cuh"
#include "cp_noise.h"
#include "cp_cutlass.h"
#include "cp_turing_scan.h"
#include "cp_proof.h"
#include "cp_share_witness.h"
#include "plain_proof_kernel.cuh"
#include "plain_proof_period.cuh"

static_assert(CP_MT_THREADS == CP_WITNESS_BLOCK_CHUNKS,
              "share witness blocks must match the chunk-roots kernel fold width");

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
    /* Signal-A sub-roots of the current attempt, kept for share proofs. */
    uint8_t*  d_a_subroots;
    size_t    a_subroots_cap;
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
    /* CP_CUDA_OVERLAP: second A buffer set (GPU0 only) the next attempt is
     * prepared into while the current one is scanned; swapped with
     * d_A_sig/d_Ap/d_a_subroots when that attempt starts. */
    int8_t*   d_Ap_next;
    int8_t*   d_A_sig_next;
    uint8_t*  d_a_subroots_next;
    size_t    a_subroots_next_cap;
    /* CP_CUDA_OVERLAP pipelined scan: pinned copy of d_found per in-flight
     * batch slot and the event that marks it valid. */
    int*      h_found_pipe;
    cudaEvent_t found_ev[2];
    int       found_pipe_ready;
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    cublasHandle_t cublas;
#endif
    int       use_cublas_period;
    int       use_cutlass_fused;
    /* sm_75 tensor-op: Turing scan kernel on packed noisy operands. */
    int       use_turing;
} GpuCtx;

static GpuCtx g_gpus[MAX_GPUS];
static int g_ngpu = 0;
static int g_contiguous = 0;
static int g_period_gemm = 1;
static int g_row_period_batch = CP_ROW_PERIOD_BATCH_DEFAULT;
static int g_col_period_batch = CP_PERIOD_BATCH_DEFAULT;
static int g_step_major_ap = 0; /* Case 10 default; main sets 1 for cuBLAS period */
/* Noisy A/B^T in the packed layout of cp_turing_layout.cuh (Turing scan kernel). */
static int g_packed_ab = 0;
/* Cert V3: bind Merkle roots with m/n before noise-seed chain. Set by begin_job. */
static int g_salted = 1;

/* Zero-B job cache: signal B^T = 0; noisy B fixed for the job. */
static struct {
    uint8_t job_key[32];
    uint8_t b_noise_seed[32];
    int m;
    int n;
    int ready;
    /* Keyed Merkle sub-roots + root of the all-zero B^T (host, n*K/256 KiB * 32 bytes). */
    uint8_t* bt_subroots;
    int bt_num_subroots;
    uint8_t bt_root[32];
} g_zero_b = {};

/* Signal A commitment of the last prepared attempt (sub-roots stay in g0->d_a_subroots). */
typedef struct {
    uint8_t root[32];
    int num_subroots;
    int valid;
} AttemptACommit;
static AttemptACommit g_attempt_a = {};

/* CP_CUDA_OVERLAP (defined with the attempt prep below). */
static int gpu_overlap_enabled(void);
static void gpu_overlap_shutdown(void);

static int zero_b_cache_matches(const uint8_t job_key[32], int m, int n)
{
    return g_zero_b.ready && g_zero_b.m == m && g_zero_b.n == n &&
           memcmp(g_zero_b.job_key, job_key, 32) == 0;
}

static int gpu_prepare_job_b(GpuCtx* g, const uint8_t job_key[32], int m, int n);

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

/* Packed operands + the Turing scan kernel when every GPU in use runs the
 * fused sm_75 tensor-op path. CP_CUDA_PACKED=0 keeps the CUTLASS kernel.
 * Must be settled before a job builds its noisy B^T. */
static void update_packed_layout(void)
{
    const char* env = getenv("CP_CUDA_PACKED");
    int on = g_ngpu > 0 && g_cutlass_fused && !g_step_major_ap && !g_contiguous &&
             !(env && !strcmp(env, "0"));
    for(int i = 0; i < g_ngpu && on; i++){
        const int kind = cp_cutlass_mma_kind(g_gpus[i].dev);
        on = (kind == CP_CUTLASS_MMA_TENSOROP || kind == CP_CUTLASS_MMA_TENSOROP80) &&
             cp_turing_scan_supported(g_gpus[i].dev);
    }
    for(int i = 0; i < g_ngpu; i++)
        g_gpus[i].use_turing = on;
    if(on && (g_row_period_batch & 1)){
        /* 256-row CTAs: row periods are scanned in pairs. */
        g_row_period_batch = g_row_period_batch < CP_ROW_PERIOD_BATCH_MAX ?
                             g_row_period_batch + 1 : g_row_period_batch - 1;
    }
    if(on && !g_packed_ab){
        printf("[gpu] packed-operand scan kernel (sm_75 m8n8k16 / sm_8x m16n8k32) "
               "(CP_CUDA_PACKED=0 for the CUTLASS kernel)\n");
        fflush(stdout);
    }
    g_packed_ab = on;
}

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
    update_packed_layout();
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
    update_packed_layout();
}

void cp_gpu_set_cutlass_fused(int on)
{
    g_cutlass_fused = on ? 1 : 0;
    for(int i = 0; i < g_ngpu; i++)
        g_gpus[i].use_cutlass_fused = g_cutlass_fused;
    update_packed_layout();
}

void cp_gpu_set_cuda_mma(int mode)
{
    cp_cutlass_set_mma_mode(mode);
    update_packed_layout();
}

void cp_gpu_begin_job(const uint8_t job_key[32], int m, int n, uint32_t cert_version)
{
    g_salted = (cert_version >= 3) ? 1 : 0;
    g_zero_b.ready = 0;
    printf("[gpu] zero-B: cache noisy B per job (salted=%d cert_version=%u)\n", g_salted,
           (unsigned)cert_version);
    fflush(stdout);
    if(g_ngpu <= 0 || !job_key || m <= 0 || n <= 0)
        return;
    GpuCtx* g0 = &g_gpus[0];
    if(gpu_prepare_job_b(g0, job_key, m, n) != 0){
        fprintf(stderr, "[gpu] zero-B job B prep failed\n");
        fflush(stderr);
    }
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
                    "[gpu] GPU%d: --cutlass-fused --cuda-mma %s not supported here "
                    "(simt needs sm_61+, tensorop/tensoropms sm_75+, tensorop80 sm_80+ "
                    "and a binary built for an sm_80+ arch, e.g. --cuda-arch '75;86;89')\n",
                    g->dev, cp_cutlass_mma_mode_name(cp_cutlass_mma_kind(g->dev)));
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
        if(g->use_cutlass_fused){
            cudaDeviceProp prop;
            CU_CHECK(cudaGetDeviceProperties(&prop, g->dev));
            printf("[gpu] GPU%d: sm_%d%d -> CUTLASS mma %s%s\n", g->dev,
                   prop.major, prop.minor,
                   cp_cutlass_mma_kind_name(cp_cutlass_mma_kind(g->dev)),
                   cp_cutlass_mma_mode() == CP_CUTLASS_MMA_AUTO ? " (auto)" : "");
        }
        fflush(stdout);
    }
    if(gpu_overlap_enabled()){
        printf("[gpu] CP_CUDA_OVERLAP=1: next attempt's A prepared on a second stream "
               "during the scan; scan batches pipelined (event-based found check)\n");
        fflush(stdout);
    }
    sync_tile_config();
    sync_ap_layout();
    update_packed_layout();
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
    gpu_overlap_shutdown();
    for(int i = 0; i < g_ngpu; i++){
        GpuCtx* g = &g_gpus[i];
        CU_CHECK(cudaSetDevice(g->dev));
        if(g->d_Ap_next) cudaFree(g->d_Ap_next);
        if(g->d_A_sig_next) cudaFree(g->d_A_sig_next);
        if(g->d_a_subroots_next) cudaFree(g->d_a_subroots_next);
        g->d_Ap_next = nullptr;
        g->d_A_sig_next = nullptr;
        g->d_a_subroots_next = nullptr;
        g->a_subroots_next_cap = 0;
        if(g->found_pipe_ready){
            cudaFreeHost(g->h_found_pipe);
            cudaEventDestroy(g->found_ev[0]);
            cudaEventDestroy(g->found_ev[1]);
            g->h_found_pipe = nullptr;
            g->found_pipe_ready = 0;
        }
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
        if(g->d_a_subroots) cudaFree(g->d_a_subroots);
        g->d_a_subroots = nullptr;
        g->a_subroots_cap = 0;
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
    free(g_zero_b.bt_subroots);
    g_zero_b.bt_subroots = nullptr;
    g_zero_b.bt_num_subroots = 0;
    g_zero_b.ready = 0;
    g_attempt_a.valid = 0;
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
        /* d_Ap/d_BpT: noisy mats for GEMM/jackpot; d_A_sig: signal A for hash/proof.
         * Zero-B path skips d_Bt_sig (~512 MiB); allocate on demand for --cpu-gen / tests. */
        CU_CHECK(cudaMalloc(&g->d_Ap, szAp));
        CU_CHECK(cudaMalloc(&g->d_BpT, szBpT));
        CU_CHECK(cudaMalloc(&g->d_A_sig, szAp));
        g->d_Bt_sig = nullptr;
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
        size_t a_sub_need = ((chunks_a + CP_MT_THREADS - 1) / CP_MT_THREADS) * 32;
        if(a_sub_need > g->a_subroots_cap){
            if(g->d_a_subroots) cudaFree(g->d_a_subroots);
            CU_CHECK(cudaMalloc(&g->d_a_subroots, a_sub_need));
            g->a_subroots_cap = a_sub_need;
        }
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

static void ensure_bt_sig(GpuCtx* g, size_t szBpT)
{
    if(!g->d_Bt_sig){
        CU_CHECK(cudaSetDevice(g->dev));
        CU_CHECK(cudaMalloc(&g->d_Bt_sig, szBpT));
    }
}

static void gpu_noise_generate_a(GpuCtx* g, int m, cudaStream_t st = 0)
{
    const int tpb = 256;
    const int tpr = R_RANK / 32;
    const int rows_per_block = tpb / tpr;
    const int perm_blocks = (K_DIM + CP_B3_LINES * tpb - 1) / (CP_B3_LINES * tpb);

    cp_gen_dense_noise_kernel<<<(m + rows_per_block - 1) / rows_per_block, tpb, 0, st>>>(
        0, m, R_RANK, g->d_seed_a, g->d_eal);
    cp_build_perm_pairs_par_kernel<<<perm_blocks, tpb, 0, st>>>(
        0, g->d_seed_a, K_DIM, R_RANK, g->d_e_ar);
    CU_CHECK(cudaGetLastError());
}

static void gpu_noise_generate_b(GpuCtx* g, int n)
{
    const int tpb = 256;
    const int tpr = R_RANK / 32;
    const int rows_per_block = tpb / tpr;
    const int perm_blocks = (K_DIM + CP_B3_LINES * tpb - 1) / (CP_B3_LINES * tpb);

    cp_gen_dense_noise_kernel<<<(n + rows_per_block - 1) / rows_per_block, tpb>>>(
        1, n, R_RANK, g->d_seed_b, g->d_ebr);
    cp_build_perm_pairs_par_kernel<<<perm_blocks, tpb>>>(
        1, g->d_seed_b, K_DIM, R_RANK, g->d_e_bl);
    CU_CHECK(cudaGetLastError());
}

static void gpu_noise_generate(GpuCtx* g, int m, int n)
{
    gpu_noise_generate_a(g, m);
    gpu_noise_generate_b(g, n);
}

/* a_sig/ap default to the current buffers g->d_A_sig/g->d_Ap. */
static void gpu_noise_apply_a(GpuCtx* g, int m, cudaStream_t st = 0,
                              const int8_t* a_sig = NULL, int8_t* ap = NULL)
{
    const int tpb = 256;
    const size_t smem = (size_t)R_RANK + (size_t)K_DIM;
    if(!a_sig) a_sig = g->d_A_sig;
    if(!ap) ap = g->d_Ap;

    if(g_packed_ab){
        cp_apply_noise_packed_kernel<<<m, tpb, smem, st>>>(
            a_sig, g->d_eal, ap, m, K_DIM, R_RANK, g->d_e_ar, CP_TURING_A_BLK);
    } else if(g_step_major_ap){
        cp_apply_noise_a_kernel<<<m, tpb, smem, st>>>(
            a_sig, g->d_eal, ap, m, K_DIM, R_RANK, g->d_e_ar);
    } else {
        cp_apply_noise_a_rowmajor_kernel<<<m, tpb, smem, st>>>(
            a_sig, g->d_eal, ap, m, K_DIM, R_RANK, g->d_e_ar);
    }
    CU_CHECK(cudaGetLastError());
}

/* signal may be NULL (zero-B: noise-only into d_BpT). */
static void gpu_noise_apply_b(GpuCtx* g, int n, const int8_t* d_bt_sig)
{
    const int tpb = 256;
    const size_t smem = (size_t)R_RANK + (size_t)K_DIM;

    if(g_packed_ab){
        cp_apply_noise_packed_kernel<<<n, tpb, smem>>>(
            d_bt_sig, g->d_ebr, g->d_BpT, n, K_DIM, R_RANK, g->d_e_bl, CP_TURING_B_BLK);
    } else if(g_step_major_ap){
        cp_apply_noise_b_kernel<<<n, tpb, smem>>>(
            d_bt_sig, g->d_ebr, g->d_BpT, n, K_DIM, R_RANK, g->d_e_bl);
    } else {
        cp_apply_noise_b_rowmajor_kernel<<<n, tpb, smem>>>(
            d_bt_sig, g->d_ebr, g->d_BpT, n, K_DIM, R_RANK, g->d_e_bl);
    }
    CU_CHECK(cudaGetLastError());
}

static void gpu_noise_apply(GpuCtx* g, int m, int n)
{
    gpu_noise_apply_a(g, m);
    gpu_noise_apply_b(g, n, g->d_Bt_sig);
}

static void gpu_upload_rowmajor_noisy(
    GpuCtx* g, const int8_t* h_a, const int8_t* h_b, int m, int n)
{
    const int tpb = 256;
    size_t szAp = (size_t)m * K_DIM;
    size_t szBpT = (size_t)n * K_DIM;
    ensure_bt_sig(g, szBpT);
    CU_CHECK(cudaMemcpy(g->d_A_sig, h_a, szAp, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(g->d_Bt_sig, h_b, szBpT, cudaMemcpyHostToDevice));
    if(g_step_major_ap){
        cp_pack_rowmajor_to_step_kernel<<<(int)((szAp + tpb - 1) / tpb), tpb>>>(
            g->d_A_sig, g->d_Ap, m, K_DIM, R_RANK);
        cp_pack_rowmajor_to_step_kernel<<<(int)((szBpT + tpb - 1) / tpb), tpb>>>(
            g->d_Bt_sig, g->d_BpT, n, K_DIM, R_RANK);
    } else if(g_packed_ab){
        if(cp_turing_pack(g->d_A_sig, g->d_Ap, m, CP_TURING_A_BLK) != 0 ||
           cp_turing_pack(g->d_Bt_sig, g->d_BpT, n, CP_TURING_B_BLK) != 0){
            fprintf(stderr, "[gpu] packed upload failed\n");
            exit(1);
        }
    } else {
        CU_CHECK(cudaMemcpy(g->d_Ap, g->d_A_sig, szAp, cudaMemcpyDeviceToDevice));
        CU_CHECK(cudaMemcpy(g->d_BpT, g->d_Bt_sig, szBpT, cudaMemcpyDeviceToDevice));
    }
    CU_CHECK(cudaGetLastError());
}

#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
static size_t pp_ap_step_plane(int dim)
{
    return (size_t)dim * (size_t)R_RANK;
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
        if(g->use_turing && g_packed_ab){
            if(cp_turing_period_batch(
                   g->dev, g->d_Ap, g->d_BpT, m, n, row_period0, col_period0,
                   row_batch_count, col_batch_count, &jp, nullptr) != 0){
                fprintf(stderr, "[turing] period batch failed\n");
                exit(1);
            }
            return;
        }
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
    const uint8_t* d_job_key, uint8_t* d_roots, int num_subroots, cudaStream_t st = 0)
{
    const int smem = CP_MT_SMEM_BYTES;
    int num_mt_blocks = (num_subroots + CP_MT_THREADS - 1) / CP_MT_THREADS;
    if(num_mt_blocks == 1){
        cp_compute_blake_mt_kernel<CP_MT_THREADS, true>
            <<<1, CP_MT_THREADS, smem, st>>>(d_job_key, d_roots, num_subroots);
    }else{
        cp_compute_blake_mt_kernel<CP_MT_THREADS, false>
            <<<num_mt_blocks, CP_MT_THREADS, smem, st>>>(d_job_key, d_roots, num_subroots);
        cp_reduce_roots_kernel<CP_MT_THREADS>
            <<<1, CP_MT_THREADS, smem, st>>>(d_job_key, d_roots, num_mt_blocks);
    }
    CU_CHECK(cudaGetLastError());
}

/* Keyed Merkle root of d_mat. d_mat may be NULL with raw_len 0 for an all-zero matrix.
 * With subroots_out (device or host), also returns the per-CP_MT_THREADS-chunk sub-roots
 * the finish pass folds in place; *out_num_subroots is 0 when the matrix is a single chunk. */
static int gpu_matrix_keyed_hash_ex(GpuCtx* g, const int8_t* d_mat,
                                    size_t raw_len, size_t pad_len,
                                    const uint8_t job_key[32], uint8_t out[32],
                                    void* subroots_out, cudaMemcpyKind subroots_kind,
                                    int* out_num_subroots)
{
    int num_chunks = (int)(pad_len / 1024);
    if(out_num_subroots) *out_num_subroots = 0;
    if(num_chunks == 1){
        uint8_t* tmp = (uint8_t*)calloc(1, pad_len);
        if(!tmp) return -1;
        if(d_mat && raw_len > 0)
            CU_CHECK(cudaMemcpy(tmp, d_mat, raw_len, cudaMemcpyDeviceToHost));
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
        if(subroots_out){
            CU_CHECK(cudaMemcpy(subroots_out, g->d_merkle_roots, (size_t)num_subroots * 32,
                                subroots_kind));
            if(out_num_subroots) *out_num_subroots = num_subroots;
        }
        cp_gpu_merkle_finish_root(g->d_job_key, g->d_merkle_roots, num_subroots);
    }
    CU_CHECK(cudaDeviceSynchronize());
    CU_CHECK(cudaMemcpy(out, g->d_merkle_roots, 32, cudaMemcpyDeviceToHost));
    return 0;
}

static int gpu_matrix_keyed_hash(GpuCtx* g, const int8_t* d_mat,
                                 size_t raw_len, size_t pad_len,
                                 const uint8_t job_key[32], uint8_t out[32])
{
    return gpu_matrix_keyed_hash_ex(g, d_mat, raw_len, pad_len, job_key, out,
                                    nullptr, cudaMemcpyDeviceToDevice, nullptr);
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
    ensure_bt_sig(g, szBpT);

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
    pearl_derive_noise_seeds(job_key, hash_a, hash_b, (uint32_t)m, (uint32_t)n, g_salted,
                             b_seed, a_key_out);
    CU_CHECK(cudaMemcpy(g->d_seed_a, a_key_out, 32, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(g->d_seed_b, b_seed, 32, cudaMemcpyHostToDevice));
    printf("[gpu-prep] noise seeds + H2D %.3fs (salted=%d)\n", cp_now_sec() - t_step,
           g_salted);
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

/* Job-level zero-B: hash empty B^T, build noisy B into d_BpT (no d_Bt_sig). */
static int gpu_prepare_job_b(GpuCtx* g, const uint8_t job_key[32], int m, int n)
{
    size_t szBpT = (size_t)n * K_DIM;
    double t0 = cp_now_sec();

    CU_CHECK(cudaSetDevice(g->dev));
    ensure_buffers(g, m, n);

    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_salted, g_zero_b.b_noise_seed);
    CU_CHECK(cudaMemcpy(g->d_seed_b, g_zero_b.b_noise_seed, 32, cudaMemcpyHostToDevice));

    double t_step = cp_now_sec();
    gpu_noise_generate_b(g, n);
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu] zero-B noise gen (EBR/perm) %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    t_step = cp_now_sec();
    gpu_noise_apply_b(g, n, /*d_bt_sig=*/NULL);
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu] zero-B noise apply (B only) %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    {
        /* Zero B^T commitment for share proofs: sub-roots are a few KiB, no host matrix. */
        size_t pad_b = (szBpT + 1023) / 1024 * 1024;
        size_t num_chunks_b = pad_b / 1024;
        size_t sub_bytes = ((num_chunks_b + CP_MT_THREADS - 1) / CP_MT_THREADS) * 32;
        uint8_t* sub = (uint8_t*)realloc(g_zero_b.bt_subroots, sub_bytes);
        if(!sub) return -1;
        g_zero_b.bt_subroots = sub;
        t_step = cp_now_sec();
        if(gpu_matrix_keyed_hash_ex(g, NULL, 0, pad_b, job_key, g_zero_b.bt_root,
                                    sub, cudaMemcpyDeviceToHost,
                                    &g_zero_b.bt_num_subroots) != 0)
            return -1;
        printf("[gpu] zero-B Merkle sub-roots (%d) %.3fs\n", g_zero_b.bt_num_subroots,
               cp_now_sec() - t_step);
        fflush(stdout);
    }

    for(int i = 1; i < g_ngpu; i++){
        GpuCtx* gi = &g_gpus[i];
        ensure_buffers(gi, m, n);
        CU_CHECK(cudaSetDevice(gi->dev));
        CU_CHECK(cudaMemcpy(gi->d_BpT, g->d_BpT, szBpT, cudaMemcpyDeviceToDevice));
    }
    CU_CHECK(cudaSetDevice(g->dev));

    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;
    g_zero_b.ready = 1;
    printf("[gpu] zero-B job B cached on device (%.3fs total)\n", cp_now_sec() - t0);
    fflush(stdout);
    return cp_job_should_cancel() ? -1 : 0;
}

/* Per-nonce: random A + hash + A-side noise into d_Ap. Reuses cached b_noise_seed. */
static int gpu_prepare_attempt_a(GpuCtx* g, uint64_t rng_seed, const uint8_t job_key[32],
                                 int m, int n, uint8_t a_key_out[32])
{
    size_t szAp = (size_t)m * K_DIM;
    size_t pad_a = (szAp + 1023) / 1024 * 1024;
    const int tpb = 256;
    int total_a = m * K_DIM;
    uint8_t hash_a[32];
    double t_step, t_total;

    CU_CHECK(cudaSetDevice(g->dev));
    t_total = cp_now_sec();
    g_attempt_a.valid = 0;

    t_step = cp_now_sec();
    cp_gen_random_matrix_kernel<<<(total_a + tpb - 1) / tpb, tpb>>>(
        rng_seed, 0, total_a, g->d_A_sig);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu-prep] random A gen %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    if(cp_job_should_cancel()) return -1;

    t_step = cp_now_sec();
    if(gpu_matrix_keyed_hash_ex(g, g->d_A_sig, szAp, pad_a, job_key, hash_a,
                                g->d_a_subroots, cudaMemcpyDeviceToDevice,
                                &g_attempt_a.num_subroots) != 0)
        return -1;
    memcpy(g_attempt_a.root, hash_a, 32);
    g_attempt_a.valid = 1;
    printf("[gpu-prep] keyed hash A %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    pearl_a_noise_seed_from_hash(g_zero_b.b_noise_seed, hash_a, (uint32_t)m, g_salted,
                                 a_key_out);
    CU_CHECK(cudaMemcpy(g->d_seed_a, a_key_out, 32, cudaMemcpyHostToDevice));

    t_step = cp_now_sec();
    gpu_noise_generate_a(g, m);
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu-prep] noise gen (EAL/perm) %.3fs\n", cp_now_sec() - t_step);
    fflush(stdout);

    t_step = cp_now_sec();
    gpu_noise_apply_a(g, m);
    CU_CHECK(cudaDeviceSynchronize());
    printf("[gpu-prep] noise apply (A only) %.3fs\n", cp_now_sec() - t_step);
    printf("[gpu-prep] attempt total %.3fs\n", cp_now_sec() - t_total);
    fflush(stdout);

    (void)n;
    return cp_job_should_cancel() ? -1 : 0;
}

/* ---- CP_CUDA_OVERLAP=1: next-attempt A prep overlapped with the scan ------
 *
 * While attempt N is scanned (legacy default stream), a host thread prepares
 * attempt N+1 -- random A, keyed Merkle hash + sub-roots, A noise seed, A
 * noise -- on a cudaStreamNonBlocking stream into GPU0's second buffer set
 * (d_A_sig_next / d_Ap_next / d_a_subroots_next, +2 x m*K bytes of VRAM).
 * The thread is joined before cp_gpu_mine_attempt returns, so share witness /
 * signal fetches after a hit (cp_gpu_fetch_share_*) still read attempt N's
 * buffers; attempt N+1 then swaps the sets (buffers and g_attempt_a commit)
 * instead of preparing A again. A prefetch is used only if job key, m, n,
 * the job's B noise seed and the salt mode still match; otherwise the attempt
 * prepares synchronously as before. The prep path issues no device-wide
 * syncs: the only host waits are the 32-byte Merkle root (needed on the host
 * to derive the A noise seed) and the end of the prep stream. */
static int g_overlap = -1;

static int gpu_overlap_enabled(void)
{
    if(g_overlap < 0){
        /* On by default (CMP 50HX and RTX 3090: same or slightly better
         * effective rate, align-test and mock verify pass); CP_CUDA_OVERLAP=0
         * restores the serial prep + synchronous scan loop. */
        const char* e = getenv("CP_CUDA_OVERLAP");
        g_overlap = (e && e[0] != '\0') ? (atoi(e) > 0 ? 1 : 0) : 1;
    }
    return g_overlap;
}

typedef struct {
    std::thread   th;
    cudaStream_t  stream;     /* non-blocking prep stream (GPU0) */
    uint8_t*      h_pin;      /* pinned: [0,32) job key, [32,64) root, [64,96) seed */
    int           ready;      /* next buffers hold a complete attempt */
    int           rc;
    uint8_t       job_key[32];
    uint8_t       b_noise_seed[32];
    int           salted;
    int           m;
    int           n;
    uint8_t       a_key[32];
    AttemptACommit commit;
    double        sec;
    uint64_t      hits;
    uint64_t      misses;
} AttemptPrefetch;
static AttemptPrefetch g_pf;

/* Keyed Merkle root (+ sub-roots into d_subroots) of an A buffer on stream
 * st. Same kernels as gpu_matrix_keyed_hash_ex; returns -1 for the
 * single-chunk case (host digest), which the caller handles synchronously. */
static int gpu_keyed_hash_a_stream(GpuCtx* g, cudaStream_t st, uint8_t* h_pin,
                                   const int8_t* d_mat, size_t raw_len, size_t pad_len,
                                   const uint8_t job_key[32], uint8_t* d_subroots,
                                   int* out_num_subroots, uint8_t out[32])
{
    const int num_chunks = (int)(pad_len / 1024);
    if(num_chunks <= 1) return -1;
    memcpy(h_pin, job_key, 32);
    CU_CHECK(cudaMemcpyAsync(g->d_job_key, h_pin, 32, cudaMemcpyHostToDevice, st));
    const int num_subroots = (num_chunks + CP_MT_THREADS - 1) / CP_MT_THREADS;
    cp_keyed_chunk_roots_kernel<<<num_subroots, CP_MT_THREADS, CP_MT_SMEM_BYTES, st>>>(
        (const uint8_t*)d_mat, raw_len, pad_len, g->d_job_key,
        g->d_merkle_roots, num_chunks);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaMemcpyAsync(d_subroots, g->d_merkle_roots, (size_t)num_subroots * 32,
                             cudaMemcpyDeviceToDevice, st));
    *out_num_subroots = num_subroots;
    cp_gpu_merkle_finish_root(g->d_job_key, g->d_merkle_roots, num_subroots, st);
    CU_CHECK(cudaMemcpyAsync(h_pin + 32, g->d_merkle_roots, 32, cudaMemcpyDeviceToHost, st));
    CU_CHECK(cudaStreamSynchronize(st));
    memcpy(out, h_pin + 32, 32);
    return 0;
}

/* gpu_prepare_attempt_a on stream st into explicit buffers; no device-wide
 * syncs. Returns 0 when the buffers and *commit hold a complete attempt. */
static int gpu_prepare_attempt_a_stream(GpuCtx* g, cudaStream_t st, uint8_t* h_pin,
                                        int8_t* d_a_sig, int8_t* d_ap, uint8_t* d_subroots,
                                        AttemptACommit* commit, uint64_t rng_seed,
                                        const uint8_t job_key[32],
                                        const uint8_t b_noise_seed[32], int salted,
                                        int m, uint8_t a_key_out[32])
{
    const size_t szAp = (size_t)m * K_DIM;
    const size_t pad_a = (szAp + 1023) / 1024 * 1024;
    const int tpb = 256;
    const int total_a = m * K_DIM;
    uint8_t hash_a[32];

    commit->valid = 0;
    cp_gen_random_matrix_kernel<<<(total_a + tpb - 1) / tpb, tpb, 0, st>>>(
        rng_seed, 0, total_a, d_a_sig);
    CU_CHECK(cudaGetLastError());
    if(gpu_keyed_hash_a_stream(g, st, h_pin, d_a_sig, szAp, pad_a, job_key, d_subroots,
                               &commit->num_subroots, hash_a) != 0)
        return -1;
    memcpy(commit->root, hash_a, 32);

    pearl_a_noise_seed_from_hash(b_noise_seed, hash_a, (uint32_t)m, salted, a_key_out);
    memcpy(h_pin + 64, a_key_out, 32);
    CU_CHECK(cudaMemcpyAsync(g->d_seed_a, h_pin + 64, 32, cudaMemcpyHostToDevice, st));
    gpu_noise_generate_a(g, m, st);
    gpu_noise_apply_a(g, m, st, d_a_sig, d_ap);
    CU_CHECK(cudaStreamSynchronize(st));
    commit->valid = 1;
    return 0;
}

static void gpu_overlap_ensure(GpuCtx* g, int m)
{
    const size_t szAp = (size_t)m * K_DIM;
    const size_t pad_a = (szAp + 1023) / 1024 * 1024;
    const size_t chunks_a = pad_a / 1024;
    const size_t a_sub_need = ((chunks_a + CP_MT_THREADS - 1) / CP_MT_THREADS) * 32;
    CU_CHECK(cudaSetDevice(g->dev));
    if(!g_pf.stream){
        CU_CHECK(cudaStreamCreateWithFlags(&g_pf.stream, cudaStreamNonBlocking));
        CU_CHECK(cudaHostAlloc((void**)&g_pf.h_pin, 128, cudaHostAllocDefault));
    }
    if(!g->d_Ap_next){
        CU_CHECK(cudaMalloc(&g->d_Ap_next, szAp));
        CU_CHECK(cudaMalloc(&g->d_A_sig_next, szAp));
        printf("[gpu] CP_CUDA_OVERLAP: +%.0f MiB for the next attempt's A buffers\n",
               2.0 * (double)szAp / (1024.0 * 1024.0));
        fflush(stdout);
    }
    if(a_sub_need > g->a_subroots_next_cap){
        if(g->d_a_subroots_next) cudaFree(g->d_a_subroots_next);
        CU_CHECK(cudaMalloc(&g->d_a_subroots_next, a_sub_need));
        g->a_subroots_next_cap = a_sub_need;
    }
}

static void gpu_prefetch_join(void)
{
    if(g_pf.th.joinable())
        g_pf.th.join();
}

/* Starts preparing the attempt after the current one into GPU0's next
 * buffers (which hold the previous, already finished attempt). */
static void gpu_prefetch_start(GpuCtx* g, const uint8_t job_key[32], int m, int n)
{
    gpu_prefetch_join();
    gpu_overlap_ensure(g, m);
    g_pf.ready = 0;
    memcpy(g_pf.job_key, job_key, 32);
    memcpy(g_pf.b_noise_seed, g_zero_b.b_noise_seed, 32);
    g_pf.salted = g_salted;
    g_pf.m = m;
    g_pf.n = n;
    const uint64_t seed = cp_gpu_fresh_rng_seed();
    g_pf.th = std::thread([g, seed, m]() {
        const double t0 = cp_now_sec();
        cudaSetDevice(g->dev);
        g_pf.rc = gpu_prepare_attempt_a_stream(
            g, g_pf.stream, g_pf.h_pin, g->d_A_sig_next, g->d_Ap_next,
            g->d_a_subroots_next, &g_pf.commit, seed, g_pf.job_key, g_pf.b_noise_seed,
            g_pf.salted, m, g_pf.a_key);
        g_pf.sec = cp_now_sec() - t0;
        g_pf.ready = (g_pf.rc == 0);
    });
}

/* Makes the prefetched attempt current (swaps GPU0's A buffer sets and the
 * signal-A commitment) if it matches this attempt. Returns 1 on success. */
static int gpu_prefetch_take(GpuCtx* g, const uint8_t job_key[32], int m, int n,
                             uint8_t a_key_out[32])
{
    gpu_prefetch_join();
    const int ok = g_pf.ready && g_pf.m == m && g_pf.n == n &&
                   g_pf.salted == g_salted && g_zero_b.ready &&
                   memcmp(g_pf.job_key, job_key, 32) == 0 &&
                   memcmp(g_pf.b_noise_seed, g_zero_b.b_noise_seed, 32) == 0;
    g_pf.ready = 0;
    if(!ok){
        g_pf.misses++;
        return 0;
    }
    int8_t* t8;
    t8 = g->d_Ap; g->d_Ap = g->d_Ap_next; g->d_Ap_next = t8;
    t8 = g->d_A_sig; g->d_A_sig = g->d_A_sig_next; g->d_A_sig_next = t8;
    uint8_t* tu = g->d_a_subroots; g->d_a_subroots = g->d_a_subroots_next;
    g->d_a_subroots_next = tu;
    size_t tc = g->a_subroots_cap; g->a_subroots_cap = g->a_subroots_next_cap;
    g->a_subroots_next_cap = tc;
    g_attempt_a = g_pf.commit;
    memcpy(a_key_out, g_pf.a_key, 32);
    g_pf.hits++;
    if(g_pf.hits <= 2 || g_pf.hits % 64 == 0){
        printf("[gpu-prep] overlap: attempt A prepared during the previous scan "
               "(%.3fs on the prep stream; %llu used, %llu discarded)\n",
               g_pf.sec, (unsigned long long)g_pf.hits, (unsigned long long)g_pf.misses);
        fflush(stdout);
    }
    return 1;
}

static void gpu_overlap_shutdown(void)
{
    gpu_prefetch_join();
    if(g_pf.stream){
        cudaStreamDestroy(g_pf.stream);
        g_pf.stream = NULL;
    }
    if(g_pf.h_pin){
        cudaFreeHost(g_pf.h_pin);
        g_pf.h_pin = NULL;
    }
    g_pf.ready = 0;
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

/* One fused-kernel variant of the --align-test-prod cross-check. */
typedef struct {
    int kind;
    int tb;
} CutlassVariant;

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
    /* The checks below read noisy rows directly: keep the row-major layout and
     * test the Turing kernel on packed copies instead. */
    g_packed_ab = 0;
    g->use_turing = 0;
    ensure_buffers(g, m, n);
    ensure_bt_sig(g, szBpT);
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

    /* Match GPU derive vs host commitment for both legacy and cert-V3 salted. */
    for(int salted = 0; salted <= 1; salted++){
        pearl_derive_noise_seeds(job_key, hash_a_gpu, hash_b_gpu, (uint32_t)m, (uint32_t)n,
                                 salted, b_seed_gpu, a_key_gpu);
        pearl_commitment_seeds(job_key, h_A, h_Bt, m, n, K_DIM, salted, b_seed_cpu,
                               a_key_cpu);
        char tag_b[32], tag_a[32];
        snprintf(tag_b, sizeof(tag_b), "b_noise_seed(salted=%d)", salted);
        snprintf(tag_a, sizeof(tag_a), "a_noise_seed(salted=%d)", salted);
        if(compare_digest(tag_b, b_seed_gpu, b_seed_cpu) != 0) goto done;
        if(compare_digest(tag_a, a_key_gpu, a_key_cpu) != 0) goto done;
    }
    printf("[align-test-prod] noise seeds OK (legacy + salted)\n");
    fflush(stdout);

    /* Continue noise-apply checks with salted seeds (cert V3 default). */
    pearl_derive_noise_seeds(job_key, hash_a_gpu, hash_b_gpu, (uint32_t)m, (uint32_t)n, 1,
                             b_seed_gpu, a_key_gpu);

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

    /* CUTLASS kernel cross-check: the per-hash-tile milestone XOR words
     * ([step][cta][virtual SIMT tile]) feed the proof, so every kernel
     * variant -- kind (simt, tensorop, tensoropms, tensorop80 where supported)
     * x threadblock tile (128x128, and 256x128 / 128x256 = two virtual
     * 128x128 CTAs for tensorop/tensorop80) -- must produce bit-identical
     * dumps on the same noisy Ap/BpT panels, and hash tile 0 must match a CPU
     * prefix GEMM. */
    if(g_cutlass_fused && !g_step_major_ap){
        cudaDeviceProp prop;
        CU_CHECK(cudaGetDeviceProperties(&prop, g->dev));
        /* Batch at period (1,1): even in both dimensions so the 256x128 and
         * 128x256 variants really run their large tiles (an odd batch would
         * fall back to 128x128). 2x4 at the minimum --m/--n of 1024. */
        int rb = m / CP_CUTLASS_CTA_M - 1;
        int cb = n / CP_CUTLASS_CTA_N - 1;
        if(rb > 2) rb = 2;
        if(cb > 4) cb = 4;
        rb &= ~1;
        cb &= ~1;
        static const CutlassVariant all_variants[] = {
            {CP_CUTLASS_MMA_SIMT, CP_CUTLASS_TB_128x128},
            {CP_CUTLASS_MMA_TENSOROP, CP_CUTLASS_TB_128x128},
            {CP_CUTLASS_MMA_TENSOROP, CP_CUTLASS_TB_256x128},
            {CP_CUTLASS_MMA_TENSOROP, CP_CUTLASS_TB_128x256},
            {CP_CUTLASS_MMA_TENSOROP_MS, CP_CUTLASS_TB_128x128},
            {CP_CUTLASS_MMA_TENSOROP80, CP_CUTLASS_TB_128x128},
            {CP_CUTLASS_MMA_TENSOROP80, CP_CUTLASS_TB_256x128},
            {CP_CUTLASS_MMA_TENSOROP80, CP_CUTLASS_TB_128x256},
        };
        enum { kMaxVariants = 8 };
        static_assert(sizeof(all_variants) / sizeof(all_variants[0]) == kMaxVariants,
                      "variant table size");
        int kinds[kMaxVariants];
        int tbs[kMaxVariants];
        const char* vnames[kMaxVariants];
        int nk = 0;
        for(int i = 0; i < kMaxVariants; i++){
            const int kind = all_variants[i].kind;
            const int tb = all_variants[i].tb;
            if(cp_cutlass_kind_supported(g->dev, kind)){
                kinds[nk] = kind;
                tbs[nk] = tb;
                vnames[nk] = cp_cutlass_variant_name(kind, tb);
                nk++;
            } else {
                printf("[align-test-prod] sm_%d%d: CUTLASS %s %s not available, skipped\n",
                       prop.major, prop.minor, cp_cutlass_mma_mode_name(kind),
                       cp_cutlass_tb_name(tb));
            }
        }
        int xrc = 0;
        /* Hash-tile policy vs CUTLASS's accumulator iterator (no MMA executed,
         * so the m16n8k32 policies are checked even on sm_75). */
        for(int i = 1; i < kMaxVariants && xrc == 0; i++){
            int ntiles = 0;
            const int bad = cp_cutlass_hash_policy_selftest(
                g->dev, all_variants[i].kind, all_variants[i].tb, &ntiles);
            printf("[align-test-prod] CUTLASS %s %s hash-tile policy vs IteratorC: ",
                   cp_cutlass_mma_mode_name(all_variants[i].kind),
                   cp_cutlass_tb_name(all_variants[i].tb));
            if(bad == 0){
                printf("OK (%d/%d tiles)\n", ntiles, ntiles);
            } else {
                printf("%s %d\n", bad < 0 ? "CUDA error" : "bad tiles", bad);
                xrc = -1;
            }
            fflush(stdout);
        }
        if(xrc == 0 && nk > 1 && rb > 0 && cb > 0){
            const size_t tiles = cp_cutlass_tiles_per_batch(rb, cb);
            const size_t bytes = cp_cutlass_tile_xor_bytes(rb, cb);
            const size_t words = bytes / sizeof(uint32_t);
            const int num_steps = K_DIM / R_RANK;
            const int saved_mode = cp_cutlass_mma_mode();
            const int saved_tb = cp_cutlass_tb();
            uint32_t* d_x = NULL;
            uint32_t* h_x[kMaxVariants] = {NULL};
            CU_CHECK(cudaMalloc(&d_x, bytes));
            for(int k = 0; k < nk && xrc == 0; k++){
                h_x[k] = (uint32_t*)malloc(bytes);
                if(!h_x[k]){ xrc = -1; break; }
                cp_cutlass_set_mma_mode(kinds[k]);
                cp_cutlass_set_tb(tbs[k]);
                CU_CHECK(cudaMemset(d_x, 0, bytes));
                t0 = cp_now_sec();
                if(cp_cutlass_period_batch(g->dev, g->d_Ap, g->d_BpT, m, n,
                                           /*row_period0=*/1, /*col_period0=*/1,
                                           rb, cb, 0, d_x, tiles, NULL) != 0){
                    xrc = -1; break;
                }
                CU_CHECK(cudaDeviceSynchronize());
                printf("[align-test-prod] CUTLASS %s tile-xor dump %dx%d CTAs %.3fs\n",
                       vnames[k], rb, cb, cp_now_sec() - t0);
                CU_CHECK(cudaMemcpy(h_x[k], d_x, bytes, cudaMemcpyDeviceToHost));
            }
            cp_cutlass_set_mma_mode(saved_mode);
            cp_cutlass_set_tb(saved_tb);
            if(xrc == 0){
                /* Independent CPU reference for hash tile 0 of CTA (0,0) of the
                 * batch (SIMT thread 0: rows {0..3,16..19}, cols {0..3,32..35}).
                 * Accumulators persist across milestones, so step s is the XOR of
                 * the prefix GEMM over k < R_RANK*(s+1). */
                int8_t* h_ar = (int8_t*)malloc((size_t)16 * K_DIM);
                if(h_ar){
                    static const int offs[8] = {0, 1, 2, 3, 16, 17, 18, 19};
                    static const int coffs[8] = {0, 1, 2, 3, 32, 33, 34, 35};
                    int32_t acc[64];
                    int ref_bad[kMaxVariants] = {0};
                    for(int i = 0; i < 8; i++){
                        CU_CHECK(cudaMemcpy(h_ar + (size_t)i * K_DIM,
                                            g->d_Ap + ((size_t)1 * CP_CUTLASS_CTA_M + offs[i]) * K_DIM,
                                            (size_t)K_DIM, cudaMemcpyDeviceToHost));
                        CU_CHECK(cudaMemcpy(h_ar + (size_t)(8 + i) * K_DIM,
                                            g->d_BpT + ((size_t)1 * CP_CUTLASS_CTA_N + coffs[i]) * K_DIM,
                                            (size_t)K_DIM, cudaMemcpyDeviceToHost));
                    }
                    memset(acc, 0, sizeof(acc));
                    for(int s = 0; s < num_steps; s++){
                        uint32_t xv = 0;
                        for(int r = 0; r < 8; r++)
                            for(int c = 0; c < 8; c++){
                                const int8_t* a = h_ar + (size_t)r * K_DIM + (size_t)s * R_RANK;
                                const int8_t* b = h_ar + (size_t)(8 + c) * K_DIM + (size_t)s * R_RANK;
                                for(int k = 0; k < R_RANK; k++)
                                    acc[r * 8 + c] += (int32_t)a[k] * (int32_t)b[k];
                                xv ^= (uint32_t)acc[r * 8 + c];
                            }
                        for(int k = 0; k < nk; k++)
                            if(h_x[k][(size_t)s * tiles] != xv) ref_bad[k]++;
                    }
                    free(h_ar);
                    printf("[align-test-prod] CUTLASS hash tile 0 vs CPU prefix GEMM:");
                    for(int k = 0; k < nk; k++){
                        printf(" %s/%s %d/%d", cp_cutlass_mma_mode_name(kinds[k]),
                               cp_cutlass_tb_name(tbs[k]), ref_bad[k], num_steps);
                        if(ref_bad[k]) xrc = -1;
                    }
                    printf(" steps differ\n");
                    fflush(stdout);
                }
                size_t nz = 0;
                for(size_t i = 0; i < words; i++)
                    if(h_x[0][i]) nz++;
                if(nz == 0){
                    fprintf(stderr, "[align-test-prod] CUTLASS tile-xor dump is all zero\n");
                    xrc = -1;
                }
                for(int k = 1; k < nk && nz; k++){
                    char ka[48], kb[48];
                    snprintf(ka, sizeof(ka), "%s/%s", cp_cutlass_mma_mode_name(kinds[0]),
                             cp_cutlass_tb_name(tbs[0]));
                    snprintf(kb, sizeof(kb), "%s/%s", cp_cutlass_mma_mode_name(kinds[k]),
                             cp_cutlass_tb_name(tbs[k]));
                    size_t bad = 0, first = (size_t)-1;
                    for(size_t i = 0; i < words; i++)
                        if(h_x[0][i] != h_x[k][i]){ if(!bad) first = i; bad++; }
                    if(bad){
                        const size_t step = first / tiles;
                        const size_t rem = first % tiles;
                        fprintf(stderr,
                                "[align-test-prod] CUTLASS %s vs %s tile-xor mismatch: "
                                "%zu/%zu words differ, first at step %zu cta %zu tile %zu "
                                "(%s %08x %s %08x)\n",
                                ka, kb, bad, words, step, rem / 256, rem % 256,
                                ka, h_x[0][first], kb, h_x[k][first]);
                        /* XOR of a CTA's 256 words is mapping-independent (all 16384
                         * cells): equal => hash-tile mapping bug, else data/K schedule. */
                        uint32_t cw[2] = {0, 0};
                        for(int t = 0; t < 256; t++){
                            cw[0] ^= h_x[0][step * tiles + (rem / 256) * 256 + t];
                            cw[1] ^= h_x[k][step * tiles + (rem / 256) * 256 + t];
                        }
                        fprintf(stderr, "[align-test-prod]   CTA-wide XOR there: %s %08x %s %08x (%s)\n",
                                ka, cw[0], kb, cw[1],
                                cw[0] == cw[1] ? "equal: mapping bug" : "differ: data/K-schedule bug");
                        xrc = -1;
                    } else {
                        printf("[align-test-prod] CUTLASS %s vs %s tile-xor OK "
                               "(%zu words, %d steps, %zu hash tiles)\n",
                               ka, kb, words, num_steps, tiles);
                        fflush(stdout);
                    }
                }
            }
            if(xrc == 0 && cp_turing_scan_supported(g->dev) &&
               2 + rb <= m / CP_CUTLASS_CTA_M){
                /* Turing kernel on packed copies of the operands vs the SIMT CUTLASS
                 * dump of the same panel, folded into the 16 transcript words of
                 * every hash tile. Its row periods come in pairs, so the panel
                 * starts at row period 2. */
                const int trp0 = 2;
                const size_t twords = tiles * 16;
                int8_t* d_pa = NULL;
                int8_t* d_pb = NULL;
                uint32_t* d_tw = NULL;
                uint32_t* h_ref = (uint32_t*)malloc(bytes);
                uint32_t* h_tw = (uint32_t*)malloc(twords * sizeof(uint32_t));
                CU_CHECK(cudaMalloc(&d_pa, szAp));
                CU_CHECK(cudaMalloc(&d_pb, szBpT));
                CU_CHECK(cudaMalloc(&d_tw, twords * sizeof(uint32_t)));
                cp_cutlass_set_mma_mode(CP_CUTLASS_MMA_SIMT);
                cp_cutlass_set_tb(CP_CUTLASS_TB_128x128);
                CU_CHECK(cudaMemset(d_x, 0, bytes));
                int trc = cp_cutlass_period_batch(g->dev, g->d_Ap, g->d_BpT, m, n,
                                                  trp0, 1, rb, cb, 0, d_x, tiles, NULL);
                cp_cutlass_set_mma_mode(saved_mode);
                cp_cutlass_set_tb(saved_tb);
                if(trc == 0) trc = cp_turing_pack(g->d_Ap, d_pa, m, CP_TURING_A_BLK);
                if(trc == 0) trc = cp_turing_pack(g->d_BpT, d_pb, n, CP_TURING_B_BLK);
                CU_CHECK(cudaMemset(d_tw, 0xAB, twords * sizeof(uint32_t)));
                t0 = cp_now_sec();
                if(trc == 0)
                    trc = cp_turing_period_batch(g->dev, d_pa, d_pb, m, n, trp0, 1, rb, cb,
                                                 NULL, d_tw);
                CU_CHECK(cudaDeviceSynchronize());
                if(trc != 0 || !h_ref || !h_tw){
                    fprintf(stderr, "[align-test-prod] packed kernel check could not run\n");
                    xrc = -1;
                } else {
                    CU_CHECK(cudaMemcpy(h_ref, d_x, bytes, cudaMemcpyDeviceToHost));
                    CU_CHECK(cudaMemcpy(h_tw, d_tw, twords * sizeof(uint32_t),
                                        cudaMemcpyDeviceToHost));
                    size_t bad = 0;
                    for(size_t t = 0; t < tiles; t++){
                        uint32_t w[16] = {0};
                        for(int s = 0; s < num_steps; s++)
                            w[s % 16] = ((w[s % 16] << 13) | (w[s % 16] >> 19)) ^
                                        h_ref[(size_t)s * tiles + t];
                        if(memcmp(w, h_tw + t * 16, sizeof(w)) != 0){
                            if(!bad)
                                fprintf(stderr, "[align-test-prod] Turing tile %zu (cta %zu vt %zu) "
                                        "word0 %08x expected %08x\n", t, t / 256, t % 256,
                                        h_tw[t * 16], w[0]);
                            bad++;
                        }
                    }
                    if(bad){
                        fprintf(stderr, "[align-test-prod] packed scan kernel vs CUTLASS simt: "
                                "%zu/%zu hash tiles differ\n", bad, tiles);
                        xrc = -1;
                    } else {
                        printf("[align-test-prod] packed scan kernel vs CUTLASS simt transcript "
                               "OK (%zu hash tiles x 16 words, %.3fs)\n", tiles, cp_now_sec() - t0);
                        fflush(stdout);
                    }
                }
                cudaFree(d_pa);
                cudaFree(d_pb);
                cudaFree(d_tw);
                free(h_ref);
                free(h_tw);
            }
            cudaFree(d_x);
            for(int k = 0; k < kMaxVariants; k++)
                free(h_x[k]);
        } else if(xrc == 0 && nk <= 1){
            printf("[align-test-prod] sm_%d%d: no int8 tensor cores, kernel cross-check skipped\n",
                   prop.major, prop.minor);
        }
        if(xrc != 0) goto done;
    }

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
    if(gpu_prepare_noisy_matrices(g, cp_gpu_fresh_rng_seed(), job_key, m, n, a_key) != 0){
        cp_gpu_shutdown();
        return rc;
    }
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

    cp_gpu_shutdown();
    return rc;
}

/* CP_CUDA_OVERLAP=1 scan loop: same batches, launch order and early exit as
 * gpu_scan_device_period, but batch i+1 is enqueued before the host waits for
 * batch i. After each batch the 4-byte found flag is copied (async, legacy
 * stream) into a pinned slot and an event is recorded; the host waits on that
 * event instead of cudaDeviceSynchronize (which would also wait for the next
 * attempt's prep on the overlap stream). A hit in batch i is read while batch
 * i+1 is queued: its CTAs see *found != 0 and return at entry, and found is
 * only ever set by atomicCAS(0 -> 1), so t_rows/t_cols stay batch i's. Tiles
 * are counted for waited batches only, as in the synchronous loop. */
static void gpu_found_pipe_ensure(GpuCtx* g)
{
    if(g->found_pipe_ready) return;
    CU_CHECK(cudaSetDevice(g->dev));
    CU_CHECK(cudaHostAlloc((void**)&g->h_found_pipe, 2 * sizeof(int), cudaHostAllocDefault));
    for(int s = 0; s < 2; s++)
        CU_CHECK(cudaEventCreateWithFlags(&g->found_ev[s],
                                          cudaEventDisableTiming | cudaEventBlockingSync));
    g->found_pipe_ready = 1;
}

static int gpu_scan_device_period_pipelined(
    const uint8_t* a_key, const uint32_t pool_tgt[8],
    int m, int n,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned)
{
    uint32_t bound[8];
    cp_scale_jackpot_target(pool_tgt, bound);
    (void)a_key;

    const int row_periods = gpu_num_row_periods(m);
    const int col_periods = gpu_num_col_periods(n);
    const int row_parts = cp_pp_num_row_parts(m, g_contiguous);
    const int col_parts = cp_pp_num_col_parts(n, g_contiguous);
    const int total_tiles = row_parts * col_parts;
    int found = 0;
    int cancelled = 0;
    uint64_t tiles_scanned = 0;
    double scan_t0 = cp_now_sec();
    int slot = 0;
    int pending = 0;          /* a launched batch not yet waited for */
    int pending_tiles = 0;

    if(out_tiles_scanned) *out_tiles_scanned = 0;
    for(int i = 0; i < g_ngpu; i++)
        gpu_found_pipe_ensure(&g_gpus[i]);

    printf("[gpu] plain_proof period-GEMM scan %dx%d periods "
           "(row_batch=%d col_batch=%d, %d hash tiles, pipelined), difficulty scaled by %llu\n",
           row_periods, col_periods, g_row_period_batch, g_col_period_batch,
           total_tiles,
           (unsigned long long)cp_jackpot_scale_factor());
    fflush(stdout);

    /* Waits for the batch in slot s on every GPU and checks its found flag. */
    auto wait_slot = [&](int s) {
        for(int i = 0; i < g_ngpu; i++){
            GpuCtx* g = &g_gpus[i];
            CU_CHECK(cudaSetDevice(g->dev));
            CU_CHECK(cudaEventSynchronize(g->found_ev[s]));
            const int f = ((volatile int*)g->h_found_pipe)[s];
            if(f && !found){
                found = 1;
                CU_CHECK(cudaMemcpy(out_t_rows, g->d_out_t_rows, sizeof(int), cudaMemcpyDeviceToHost));
                CU_CHECK(cudaMemcpy(out_t_cols, g->d_out_t_cols, sizeof(int), cudaMemcpyDeviceToHost));
                printf("[gpu] GPU%d: plain_proof SHARE t_rows=%d t_cols=%d\n",
                       g->dev, *out_t_rows, *out_t_cols);
                fflush(stdout);
            }
        }
    };

    for(int rpi0 = 0; rpi0 < row_periods && !found && !cancelled; rpi0 += g_row_period_batch){
        if(cp_job_should_cancel()){
            cancelled = 1;
            break;
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
                CU_CHECK(cudaMemcpyAsync(&g->h_found_pipe[slot], g->d_found, sizeof(int),
                                         cudaMemcpyDeviceToHost, 0));
                CU_CHECK(cudaEventRecord(g->found_ev[slot], 0));
            }

            if(pending){
                wait_slot(slot ^ 1);
                tiles_scanned += (uint64_t)pending_tiles;
            }
            pending = 1;
            pending_tiles = batch_tiles;
            slot ^= 1;
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
    if(pending && !found){
        /* Last batch (or the one in flight at cancel). */
        wait_slot(slot ^ 1);
        tiles_scanned += (uint64_t)pending_tiles;
    }
    /* After a hit one more batch may be in flight (it exits at entry):
     * leave the legacy streams idle like the synchronous loop does. */
    for(int i = 0; i < g_ngpu; i++){
        CU_CHECK(cudaSetDevice(g_gpus[i].dev));
        CU_CHECK(cudaStreamSynchronize(0));
    }
    if(out_tiles_scanned) *out_tiles_scanned = tiles_scanned;
    if(cancelled && !found) return -1;
    return found;
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
           "(row_batch=%d col_batch=%d, %d hash tiles), difficulty scaled by %llu\n",
           row_periods, col_periods, g_row_period_batch, g_col_period_batch,
           total_tiles,
           (unsigned long long)cp_jackpot_scale_factor());
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
    if(g_period_gemm && !g_contiguous){
        if(gpu_overlap_enabled())
            return gpu_scan_device_period_pipelined(a_key, pool_tgt, m, n,
                                                    out_t_rows, out_t_cols,
                                                    out_tiles_scanned);
        return gpu_scan_device_period(a_key, pool_tgt, m, n,
                                      out_t_rows, out_t_cols, out_tiles_scanned);
    }

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

    printf("[gpu] plain_proof scan %dx%d hash tiles, difficulty scaled by %llu\n",
           row_parts, col_parts, (unsigned long long)cp_jackpot_scale_factor());
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
    (void)ab_seed;
    (void)ab_seed_len;
    (void)h_A_sig;
    (void)h_Bt_sig;
    const double attempt_t0 = cp_now_sec();
    size_t szAp = (size_t)m * K_DIM;
    uint8_t a_key_local[32];
    const uint8_t* scan_key = a_key;
    int zero = 0;
    int overlap = 0;

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
        overlap = gpu_overlap_enabled();
        if(overlap)
            gpu_prefetch_join();
        if(!zero_b_cache_matches(job_key, m, n)){
            if(gpu_prepare_job_b(g0, job_key, m, n) != 0)
                return -1;
        }
        if(overlap){
            /* Use the A prepared during the previous scan, else prepare it
             * now on the overlap stream (no device-wide syncs). */
            if(!gpu_prefetch_take(g0, job_key, m, n, a_key_local)){
                const double t_prep = cp_now_sec();
                gpu_overlap_ensure(g0, m);
                g_attempt_a.valid = 0;
                if(gpu_prepare_attempt_a_stream(
                       g0, g_pf.stream, g_pf.h_pin, g0->d_A_sig, g0->d_Ap,
                       g0->d_a_subroots, &g_attempt_a, cp_gpu_fresh_rng_seed(),
                       job_key, g_zero_b.b_noise_seed, g_salted, m, a_key_local) != 0){
                    if(gpu_prepare_attempt_a(g0, cp_gpu_fresh_rng_seed(), job_key, m, n,
                                             a_key_local) != 0)
                        return -1;
                } else {
                    printf("[gpu-prep] attempt A (overlap stream, not prefetched) %.3fs\n",
                           cp_now_sec() - t_prep);
                    fflush(stdout);
                }
            }
            if(cp_job_should_cancel()) return -1;
        } else if(gpu_prepare_attempt_a(g0, cp_gpu_fresh_rng_seed(), job_key, m, n,
                                        a_key_local) != 0){
            return -1;
        }
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
            /* d_BpT already mirrored in gpu_prepare_job_b. */
            CU_CHECK(cudaMemcpy(g->d_a_key8, a_key32, 32, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(g->d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    /* Overlap: prepare the next attempt's A while this one is scanned. The
     * prep reads none of the buffers the scan uses (d_Ap/d_BpT/d_a_key8/
     * d_found) and writes only GPU0's next buffer set and prep scratch. */
    if(overlap)
        gpu_prefetch_start(g0, job_key, m, n);

    const double prep_sec = cp_now_sec() - attempt_t0;
    const double scan_t0 = cp_now_sec();
    int found = gpu_scan_device(scan_key, pool_tgt, m, n, out_t_rows, out_t_cols, out_tiles_scanned);
    const double scan_sec = cp_now_sec() - scan_t0;
    /* Hit handling (witness/signal fetch) reads this attempt's buffers and
     * the next job may re-prepare B: no prep may still be running. */
    if(overlap)
        gpu_prefetch_join();
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
    /* Zero-B: host h_BpT stays launch-zeroed; only copy B if a signal buffer exists. */
    if(h_Bt_sig && g0->d_Bt_sig){
        CU_CHECK(cudaMemcpy(h_Bt_sig, g0->d_Bt_sig, szBpT, cudaMemcpyDeviceToHost));
    }
    return 0;
}

int cp_gpu_fetch_share_witness(int t_rows, int t_cols, int tile_layout, CpShareWitness** out)
{
    (void)t_cols;
    if(!out) return -1;
    *out = NULL;
    if(g_ngpu <= 0 || !g_attempt_a.valid || !g_zero_b.ready || !g_zero_b.bt_subroots)
        return -1;
    GpuCtx* g0 = &g_gpus[0];
    const int m = g_m_active;
    if(m <= 0 || !g0->d_A_sig) return -1;
    const size_t szAp = (size_t)m * K_DIM;

    CpShareWitness* w = (CpShareWitness*)calloc(1, sizeof(CpShareWitness));
    if(!w) return -1;
    w->tile_layout = tile_layout;

    int nb = cp_proof_witness_blocks(tile_layout, 0, t_rows, m, K_DIM, w->a_block_idx,
                                     CP_WITNESS_MAX_BLOCKS);
    if(nb <= 0){
        fprintf(stderr, "[gpu] share witness: no A blocks for t_rows=%d layout=%d\n",
                t_rows, tile_layout);
        cp_share_witness_free(w);
        return -1;
    }
    w->a_num_blocks = (size_t)nb;
    w->a_blocks = (uint8_t*)calloc((size_t)nb, CP_WITNESS_BLOCK_BYTES);
    if(!w->a_blocks){
        cp_share_witness_free(w);
        return -1;
    }

    CU_CHECK(cudaSetDevice(g0->dev));
    for(int i = 0; i < nb; i++){
        size_t off = (size_t)w->a_block_idx[i] * CP_WITNESS_BLOCK_BYTES;
        if(off >= szAp) continue; /* padding-only block stays zero */
        size_t len = szAp - off;
        if(len > CP_WITNESS_BLOCK_BYTES) len = CP_WITNESS_BLOCK_BYTES;
        CU_CHECK(cudaMemcpy(w->a_blocks + (size_t)i * CP_WITNESS_BLOCK_BYTES,
                            g0->d_A_sig + off, len, cudaMemcpyDeviceToHost));
    }

    if(g_attempt_a.num_subroots > 0){
        size_t bytes = (size_t)g_attempt_a.num_subroots * 32;
        w->a_subroots = (uint8_t*)malloc(bytes);
        if(!w->a_subroots){
            cp_share_witness_free(w);
            return -1;
        }
        CU_CHECK(cudaMemcpy(w->a_subroots, g0->d_a_subroots, bytes, cudaMemcpyDeviceToHost));
        w->a_num_subroots = (size_t)g_attempt_a.num_subroots;
    }
    memcpy(w->a_root, g_attempt_a.root, 32);

    if(g_zero_b.bt_num_subroots > 0){
        size_t bytes = (size_t)g_zero_b.bt_num_subroots * 32;
        w->bt_subroots = (uint8_t*)malloc(bytes);
        if(!w->bt_subroots){
            cp_share_witness_free(w);
            return -1;
        }
        memcpy(w->bt_subroots, g_zero_b.bt_subroots, bytes);
        w->bt_num_subroots = (size_t)g_zero_b.bt_num_subroots;
    }
    memcpy(w->bt_root, g_zero_b.bt_root, 32);

    *out = w;
    return 0;
}
