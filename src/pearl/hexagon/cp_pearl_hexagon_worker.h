#ifndef CP_PEARL_HEXAGON_WORKER_H
#define CP_PEARL_HEXAGON_WORKER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pearl on the Hexagon cDSP (HVX): fused GEMM + milestone XOR in pearlx (dsp/), 4x64
 * hash tiles, noise generated on the DSP (zero-B), host BLAKE3 jackpot. */
void cp_pearl_hexagon_worker_init(void);
void cp_pearl_hexagon_worker_shutdown(void);
int cp_pearl_hexagon_worker_is_ready(void);
int cp_pearl_hexagon_worker_list_devices(void);
/* Host signal A in memory shared with the DSP (zero-filled), which reads each panel's
 * signal rows in place to generate noisy A. Needs the worker initialized. */
int8_t *cp_pearl_hexagon_worker_alloc_signal_a(size_t bytes);
void cp_pearl_hexagon_worker_free_signal_a(int8_t *p);
/* DSP launch size in 128x128 macro blocks: row_macros x col_macros (<= 0 restores the
 * default, CP_HEXAGON_LAUNCH_MACROS_DEFAULT each). */
void cp_pearl_hexagon_worker_set_launch(int row_macros, int col_macros);
/* Launch rows x cols for an m x n matrix (clipped to the matrix). */
void cp_pearl_hexagon_worker_launch(int m, int n, int *rows, int *cols);
void cp_pearl_hexagon_worker_begin_job(const uint8_t job_key[32], int m, int n,
                                       uint32_t cert_version);

int cp_pearl_hexagon_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned);

#ifdef __cplusplus
}
#endif

#endif /* CP_PEARL_HEXAGON_WORKER_H */
