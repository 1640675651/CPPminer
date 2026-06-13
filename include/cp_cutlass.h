#ifndef CP_CUTLASS_H
#define CP_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 if CUTLASS fused GEMM can run on the current device. */
int cp_cutlass_device_ok(int dev);

/* Fused GEMM + per-thread tile XOR for one period batch.
 * Ap/BpT must match step_major (1 = step panels, 0 = row-major m×k). */
int cp_cutlass_period_batch(
    int dev,
    const int8_t* d_Ap,
    const int8_t* d_BpT,
    int m,
    int n,
    int row_period,
    int col_period0,
    int batch_count,
    int step_major,
    uint32_t* d_tile_xor,
    size_t tiles_per_batch);

size_t cp_cutlass_tiles_per_batch(int batch_count);

size_t cp_cutlass_tile_xor_bytes(int batch_count);

#ifdef __cplusplus
}
#endif

#endif /* CP_CUTLASS_H */
