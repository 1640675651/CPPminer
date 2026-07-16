#ifndef CP_CUDA_WORKER_H
#define CP_CUDA_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Thin adapter over cp_gpu_* for the unified worker API. */
void cp_cuda_worker_init(int* devices, int ndev);
void cp_cuda_worker_shutdown(void);
void cp_cuda_worker_set_contiguous_tiles(int on);
void cp_cuda_worker_set_period_gemm(int on);
void cp_cuda_worker_set_period_batch(int batch);
void cp_cuda_worker_set_row_period_batch(int batch);
void cp_cuda_worker_set_col_period_batch(int batch);
void cp_cuda_worker_set_step_major_ap(int on);
void cp_cuda_worker_set_cutlass_fused(int on);

int cp_cuda_worker_mine_attempt(
    const uint8_t* ab_seed, int ab_seed_len,
    const uint8_t job_key[32],
    const uint32_t pool_tgt[8],
    int m, int n,
    int cpu_matrices,
    const int8_t* h_A_noisy, const int8_t* h_B_noisy,
    const uint8_t* a_key,
    int8_t* h_A_sig, int8_t* h_Bt_sig,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned);

int cp_cuda_worker_fetch_share_signals(int8_t* h_A_sig, int8_t* h_Bt_sig);

#ifdef __cplusplus
}
#endif

#endif /* CP_CUDA_WORKER_H */
