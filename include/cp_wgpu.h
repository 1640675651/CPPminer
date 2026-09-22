#ifndef CP_WGPU_H
#define CP_WGPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for cp_wgpu_search_range */
#define CP_WGPU_OK_FOUND     1
#define CP_WGPU_OK_EXHAUSTED 0
#define CP_WGPU_CANCELLED   (-1)
#define CP_WGPU_ERROR       (-2)
#define CP_WGPU_DEVICE_LOST (-3)

/* Print mining adapters; returns count (0 if none). Indices match --devices. */
int cp_wgpu_list_devices(void);

/*
 * 0 on success, -1 on failure. batch_size 0 → 1e6.
 * devices/ndev: optional indices from cp_wgpu_list_devices; NULL/0 → auto-all.
 */
int cp_wgpu_init(uint32_t batch_size, int allow_integrated,
                 const int* devices, int ndev);
void cp_wgpu_shutdown(void);
int cp_wgpu_is_ready(void);

/*
 * Search nonces [start, start+count). target_be / start_nonce_be are 64-byte BE.
 * cancel_check may be NULL; non-zero return cancels.
 * out_hashes always written. out_nonce_be/out_hash_be written on found.
 */
int cp_wgpu_search_range(
    const uint8_t* header,
    uint64_t difficulty_u64,
    const uint8_t* target_be,
    const uint8_t* start_nonce_be,
    uint64_t count,
    uint8_t* out_nonce_be,
    uint8_t* out_hash_be,
    uint64_t* out_hashes,
    int (*cancel_check)(void));

#ifdef __cplusplus
}
#endif

#endif /* CP_WGPU_H */
