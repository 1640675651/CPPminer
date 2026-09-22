#ifndef CP_PEARL_WGPU_H
#define CP_PEARL_WGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for cp_pearl_wgpu_scan: 1 found, 0 exhausted, -1 cancel/error. */

int cp_pearl_wgpu_list_devices(void);
int cp_pearl_wgpu_init(const int* devices, int ndev); /* 0 ok */
void cp_pearl_wgpu_shutdown(void);
int cp_pearl_wgpu_is_ready(void);

/* Begin job: GPU prep B. b_noise_seed is 32 bytes from host pearl_b_noise_seed_from_bt. */
int cp_pearl_wgpu_begin_job(int m, int n, int k, const uint8_t b_noise_seed[32]);

/* Prep A: gen_random + merkle → hash_a out. Does NOT prepack yet. */
int cp_pearl_wgpu_prep_a_signal(
    const uint8_t* ab_seed,
    int ab_seed_len,
    const uint8_t job_key[32],
    uint8_t hash_a_out[32]);

/* Host helper: derive a_noise_seed from hash_a (mirrors pearl_a_noise_seed_from_hash). */
void cp_pearl_wgpu_a_noise_seed_from_hash(
    const uint8_t b_noise_seed[32],
    const uint8_t hash_a[32],
    uint32_t m,
    int salted,
    uint8_t a_noise_seed[32]);

/* After host derives a_key from hash_a: upload a_key as noise seed, fused_prepack_a. */
int cp_pearl_wgpu_prepack_a(const uint8_t a_noise_seed[32]);

/* Scan: batches of macro blocks. Returns 1 found, 0 exhausted, -1 cancel/error.
 * on_progress may be NULL; called after each macro batch with cumulative tiles. */
int cp_pearl_wgpu_scan(
    const uint32_t a_key8[8],
    const uint32_t bound[8],
    int macro_batch,
    int* out_found,
    int* out_t_rows,
    int* out_t_cols,
    uint64_t* out_tiles_scanned,
    int (*cancel_check)(void),
    void (*on_progress)(uint64_t tiles_scanned));

/* Download A signal matrix: M*K packed s8 bytes. */
int cp_pearl_wgpu_download_a_sig(int8_t* out, size_t elems);

#ifdef __cplusplus
}
#endif

#endif /* CP_PEARL_WGPU_H */
