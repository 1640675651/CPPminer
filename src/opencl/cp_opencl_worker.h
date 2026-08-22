#ifndef CP_OPENCL_WORKER_H
#define CP_OPENCL_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cp_opencl_worker_init(int *devices, int ndev);
void cp_opencl_worker_shutdown(void);
void cp_opencl_worker_set_macro_batch(int batch);
/* Restrict OpenCL enumeration to platform index (default -1 = all). */
void cp_opencl_worker_set_platform(int platform_index);
/* OpenCL hash tile MR (4 or 8) and width (8 or 16) after configure. */
int cp_opencl_hash_tile_mr(void);
int cp_opencl_hash_tile_w(void);
/* OpenCL register tile size. Pass mr<=0 to restore auto (8x8 default, 8x16 on AMD). */
void cp_opencl_worker_set_tile(int mr, int nr);
/* OpenCL GEMM issue: 0 = auto (DPI then cpm), 1 = broadcast/cpm, 2 = packed. */
void cp_opencl_worker_set_issue_mode(int mode);
/* Legacy: on → broadcast (1), off → auto (0). */
void cp_opencl_worker_set_issue_broadcast(int on);
/* Broadcast cpm type: 0 = float (default), 1 = int32. Requires broadcast issue. */
void cp_opencl_worker_set_cpm_int(int on);
/* Stage A/B in __local (default off). */
void cp_opencl_worker_set_use_lds(int on);
/* Apply tile override or auto-detect for device before kernel build. */
void cp_opencl_configure_tile(int device_index, int platform_filter);
/* Same as above using worker platform filter (-1 = all platforms). */
void cp_opencl_configure_tile_for_worker(int device_index);
/* Print OpenCL devices; returns count. */
int cp_opencl_worker_list_devices(void);
int cp_opencl_worker_handles_matrix_prep(void);
void cp_opencl_worker_begin_job(const uint8_t job_key[32], int m, int n,
                                uint32_t cert_version);

int cp_opencl_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned);

int cp_opencl_worker_fetch_share_signals(int8_t *h_A_sig, int8_t *h_Bt_sig);

#ifdef __cplusplus
}
#endif

#endif /* CP_OPENCL_WORKER_H */
