#ifndef CP_PEARL_NPU_WORKER_H
#define CP_PEARL_NPU_WORKER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pearl on the Intel NPU (NPU3720 class, Core Ultra): fused int8 GEMM + per-tile milestone XOR through
 * OpenVINO and the patched in-plugin NPU compiler (src/pearl/intel_npu/README.md). */
/* Launch in 128x128 macro blocks, rows x columns (<= 0: default); before init. */
void cp_pearl_npu_worker_set_launch(int row_macros, int col_macros);
/* The launch in rows x columns: the compiled one after init, else the one init will pick. */
void cp_pearl_npu_worker_launch_shape(int *rows, int *cols);
/* --fused-jackpot: the BLAKE3 jackpot on the NPU's SHAVEs; one hit word per tile row comes back. Before init. */
void cp_pearl_npu_worker_set_fused_jackpot(int on);
void cp_pearl_npu_worker_init(int *devices, int ndev);
void cp_pearl_npu_worker_shutdown(void);
int cp_pearl_npu_worker_is_ready(void);
int cp_pearl_npu_worker_list_devices(void);
void cp_pearl_npu_worker_begin_job(const uint8_t job_key[32], int m, int n, uint32_t cert_version);

int cp_pearl_npu_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned);

#ifdef __cplusplus
}
#endif

#endif /* CP_PEARL_NPU_WORKER_H */
