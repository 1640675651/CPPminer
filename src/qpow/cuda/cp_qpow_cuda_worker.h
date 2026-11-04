#ifndef CP_QPOW_CUDA_WORKER_H
#define CP_QPOW_CUDA_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes: the shared GPU mine loop asserts they equal its QPOW_GPU_* codes. */
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
/* Nonces one search() call covers: one launch on every device. */
uint64_t cp_qpow_cuda_worker_batch_size(void);

/* Search counters from start_be. Each counter stands for one nonce (bytes 0..31 as given, bytes
 * 32..63 mapped onto a structured nonce family), so out_nonces_be holds nonces to submit, not
 * counters. Every GPU candidate is re-hashed on the host; a share is a nonce whose full 64-byte
 * hash is below the target.
 * Each call hands back one round, one launch on every device: *out_hashes counters from start_be,
 * with every share among them (up to max_shares) in out_nonces_be[0 .. *out_nshares). The next
 * round is already queued: a call for the same header and target at start_be + *out_hashes
 * picks it up, any other call drops it and starts over. */
int cp_qpow_cuda_worker_search(
    const uint8_t header[32],
    const uint8_t target_be[64],
    const uint8_t start_be[64],
    uint8_t out_nonces_be[][64],
    int max_shares,
    int* out_nshares,
    uint64_t* out_hashes);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_CUDA_WORKER_H */
