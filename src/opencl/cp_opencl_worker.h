#ifndef CP_OPENCL_WORKER_H
#define CP_OPENCL_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cp_opencl_worker_init(int *devices, int ndev);
void cp_opencl_worker_shutdown(void);
void cp_opencl_worker_set_macro_batch(int batch);
int cp_opencl_worker_handles_matrix_prep(void);
void cp_opencl_worker_begin_job(const uint8_t job_key[32], int m, int n);

int cp_opencl_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned);

#ifdef __cplusplus
}
#endif

#endif /* CP_OPENCL_WORKER_H */
