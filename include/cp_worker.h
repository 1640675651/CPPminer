#ifndef CP_WORKER_H
#define CP_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CP_BACKEND_NONE   = 0,
    CP_BACKEND_CPU    = 1,
    CP_BACKEND_CUDA   = 2,
    CP_BACKEND_OPENCL = 3
} CpBackendId;

/* Compile-time availability (1 if linked). */
int cp_worker_has_cpu(void);
int cp_worker_has_cuda(void);
int cp_worker_has_opencl(void);

const char* cp_worker_backend_name(void);
CpBackendId cp_worker_backend_id(void);

/* Select backend before init when several are compiled. Returns 0 on ok. */
int cp_worker_select(CpBackendId id);

void cp_worker_init(int* devices, int ndev);
void cp_worker_shutdown(void);

void cp_worker_apply_backend_defaults(void);
int cp_worker_uses_contiguous_tiles(void);
void cp_worker_set_period_gemm(int on);
void cp_worker_set_period_batch(int batch);
void cp_worker_set_row_period_batch(int batch);
void cp_worker_set_col_period_batch(int batch);
void cp_worker_set_step_major_ap(int on);
void cp_worker_set_cutlass_fused(int on);

/* Prefer host matrix path when non-zero (CPU backend always uses host matrices). */
int cp_worker_prefers_host_matrices(void);

/* Worker generates noisy matrices internally (CPU zero-B). */
int cp_worker_worker_handles_matrix_prep(void);
void cp_worker_begin_job(const uint8_t job_key[32], int m, int n);

/* Default tile layout for proof build (matches CP_TILE_LAYOUT_* in cp_proof.h). */
int cp_worker_default_tile_layout(void);

/*
 * One matrix attempt: prepare noisy A/B (host or device), scan for jackpot.
 * Returns 1 on share, 0 on miss, -1 on cancel/error.
 * On share with device-generated matrices, copies signal A/B^T to h_A_sig/h_Bt_sig.
 */
int cp_worker_mine_attempt(
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

#endif /* CP_WORKER_H */
