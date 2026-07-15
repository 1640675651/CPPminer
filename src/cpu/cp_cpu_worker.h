#ifndef CP_CPU_WORKER_H
#define CP_CPU_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cp_cpu_worker_init(void);
void cp_cpu_worker_shutdown(void);

/* Reuse noisy buffers as scan layout via prepack+swap (~2 GiB steady vs ~3 GiB). */
void cp_cpu_worker_set_inplace_prepack(int on);

/* Zero-B path: worker builds matrices; skip host gen in cp_mine when non-zero. */
int cp_cpu_worker_handles_matrix_prep(void);

void cp_cpu_worker_begin_job(const uint8_t job_key[32], int m, int n);

int cp_cpu_worker_mine_attempt(
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

#ifdef __cplusplus
}
#endif

#endif /* CP_CPU_WORKER_H */
