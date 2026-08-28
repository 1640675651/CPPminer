#ifndef CP_ONEDNN_WORKER_H
#define CP_ONEDNN_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cp_onednn_worker_init(int *devices, int ndev);
int cp_onednn_worker_is_ready(void);
void cp_onednn_worker_shutdown(void);
void cp_onednn_worker_set_row_period_batch(int batch);
void cp_onednn_worker_set_col_period_batch(int batch);
void cp_onednn_worker_set_platform(int platform_index);
int cp_onednn_worker_list_devices(void);
int cp_onednn_worker_handles_matrix_prep(void);
void cp_onednn_worker_begin_job(const uint8_t job_key[32], int m, int n, uint32_t cert_version);

int cp_onednn_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned);

int cp_onednn_worker_fetch_share_signals(int8_t *h_A_sig, int8_t *h_Bt_sig);

int cp_onednn_hash_tile_mr(void);
int cp_onednn_hash_tile_w(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_ONEDNN_WORKER_H */
