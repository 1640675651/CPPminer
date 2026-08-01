#ifndef CP_GPU_H
#define CP_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cp_gpu_init(int* devs, int ndev);
void cp_gpu_shutdown(void);
/* Print CUDA devices; returns count. */
int cp_gpu_list_devices(void);
void cp_gpu_set_contiguous_tiles(int on);
void cp_gpu_set_period_gemm(int on);
void cp_gpu_set_period_batch(int batch);
void cp_gpu_set_row_period_batch(int batch);
void cp_gpu_set_col_period_batch(int batch);
void cp_gpu_set_step_major_ap(int on);
void cp_gpu_set_cutlass_fused(int on);

/* CPU matrix path: upload host noisy matrices and scan. */
int cp_gpu_mine_plain_proof(const int8_t* h_A, const int8_t* h_B,
                            const uint8_t* a_key, const uint32_t pool_tgt[8],
                            int m, int n,
                            int* out_t_rows, int* out_t_cols,
                            uint64_t* out_tiles_scanned);

/* GPU matrix path (default): random fill + GPU commitment/noise, then scan.
 * On share, copies signal A/B^T to h_A_sig/h_Bt_sig for proof build. */
/* End-to-end GPU vs CPU check at given m,n (production: m=n=131072). */
int cp_gpu_run_alignment_tests(int dev, int m, int n);

/* Time one period batch: cuBLAS rank GEMMs vs jackpot kernel (no pool). */
int cp_gpu_run_scan_profile(int dev, int m, int n, int warmup, int runs);

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
    uint64_t* out_tiles_scanned);

/* Device → host signal A/B after a share (for deferred handoff). */
int cp_gpu_fetch_share_signals(int8_t* h_A_sig, int8_t* h_Bt_sig);

#ifdef __cplusplus
}
#endif

#endif /* CP_GPU_H */
