#ifndef CP_WGPU_WORKER_H
#define CP_WGPU_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Nonces per GpuEngine launch; 0 → 1e6. Call before init. */
void cp_wgpu_worker_set_batch_size(uint32_t batch);
uint32_t cp_wgpu_worker_batch_size(void);

int cp_wgpu_worker_init(int* devices, int ndev);
void cp_wgpu_worker_shutdown(void);
int cp_wgpu_worker_is_ready(void);
int cp_wgpu_worker_list_devices(void);

int cp_wgpu_worker_search(
    const uint8_t header[32],
    uint64_t difficulty_u64,
    const uint8_t target[64],
    const uint8_t start[64],
    uint64_t count,
    uint8_t out_nonce[64],
    uint8_t out_hash[64],
    uint64_t* out_hashes);

#ifdef __cplusplus
}
#endif

#endif /* CP_WGPU_WORKER_H */
