#ifndef CP_QPOW_OPENCL_WORKER_H
#define CP_QPOW_OPENCL_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes align with CP_WGPU_* for mine-loop sharing. */
#define CP_QPOW_OCL_OK_FOUND     1
#define CP_QPOW_OCL_OK_EXHAUSTED 0
#define CP_QPOW_OCL_CANCELLED   (-1)
#define CP_QPOW_OCL_ERROR       (-2)

int cp_qpow_opencl_worker_init(int* devices, int ndev);
void cp_qpow_opencl_worker_shutdown(void);
int cp_qpow_opencl_worker_is_ready(void);
int cp_qpow_opencl_worker_list_devices(void);

/* Nonces per launch; 0 = automatic (about 100 ms per launch). Call before init. */
void cp_qpow_opencl_worker_set_batch_size(uint32_t batch);
uint32_t cp_qpow_opencl_worker_batch_size(void);

/* Search [start_be, start_be+count). Writes BE nonce/hash on found. */
int cp_qpow_opencl_worker_search(
    const uint8_t header[32],
    const uint8_t target_be[64],
    const uint8_t start_be[64],
    uint64_t count,
    uint8_t out_nonce_be[64],
    uint8_t out_hash_be[64],
    uint64_t* out_hashes);

/* search() counts 64-byte values that stand for nonces of qpow/nonce_line.hpp, so out_nonce_be is
 * the nonce to submit; this gives the counter to continue from after it. */
void cp_qpow_opencl_worker_resume(const uint8_t found_be[64], uint8_t next_be[64]);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_OPENCL_WORKER_H */
