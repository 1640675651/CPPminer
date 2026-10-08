#ifndef CP_QPOW_CUDA_WORKER_H
#define CP_QPOW_CUDA_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes align with CP_QPOW_OCL_* / CP_WGPU_* for mine-loop sharing. */
#define CP_QPOW_CUDA_OK_FOUND     1
#define CP_QPOW_CUDA_OK_EXHAUSTED 0
#define CP_QPOW_CUDA_CANCELLED   (-1)
#define CP_QPOW_CUDA_ERROR       (-2)

/* devices: CUDA ordinals; ndev 0 -> device 0. Runs a hash self-test per device. */
int cp_qpow_cuda_worker_init(const int* devices, int ndev);
void cp_qpow_cuda_worker_shutdown(void);
int cp_qpow_cuda_worker_is_ready(void);

/* Nonces per launch per device; 0 -> automatic (about 100 ms per launch). Call before init. */
void cp_qpow_cuda_worker_set_batch_size(uint32_t batch);
/* Nonces one search step covers on all devices (for the mine loop's chunking). */
uint64_t cp_qpow_cuda_worker_batch_size(void);

/* Search the counters [start_be, start_be + count). Each counter stands for one nonce (bytes
 * 0..31 as given, bytes 32..63 mapped onto a structured nonce family), so out_nonce_be is the
 * nonce to submit, not a counter. Every GPU candidate is re-hashed on the host; OK_FOUND only
 * for a nonce whose full 64-byte hash is below the target. */
int cp_qpow_cuda_worker_search(
    const uint8_t header[32],
    const uint8_t target_be[64],
    const uint8_t start_be[64],
    uint64_t count,
    uint8_t out_nonce_be[64],
    uint8_t out_hash_be[64],
    uint64_t* out_hashes);

/* The counter to continue from after search() reported found_be. */
void cp_qpow_cuda_worker_resume(const uint8_t found_be[64], uint8_t next_be[64]);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_CUDA_WORKER_H */
