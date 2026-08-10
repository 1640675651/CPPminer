#ifndef CP_CUTLASS_H
#define CP_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 if CUTLASS fused GEMM can run on the current device. */
int cp_cutlass_device_ok(int dev);

/* Fused GEMM + in-register milestone XOR for one period batch panel.
 * Panel covers row_batch x col_batch CTAs of 128x128 (Case 10 / MMA lane).
 * When jackpot is non-NULL, BLAKE3/target check runs in the GEMM kernel tail
 * and d_tile_xor may be NULL. */
typedef struct {
    uint32_t bound[8];
    const uint32_t* d_a_key8;
    int* d_found;
    int* d_out_t_rows;
    int* d_out_t_cols;
    int row_period0;
    int col_period0;
} CpCutlassJackpotLaunch;

int cp_cutlass_period_batch(
    int dev,
    const int8_t* d_Ap,
    const int8_t* d_BpT,
    int m,
    int n,
    int row_period0,
    int col_period0,
    int row_batch_count,
    int col_batch_count,
    int step_major,
    uint32_t* d_tile_xor,
    size_t tiles_per_batch,
    const CpCutlassJackpotLaunch* jackpot);

size_t cp_cutlass_tiles_per_batch(int row_batch_count, int col_batch_count);

size_t cp_cutlass_tile_xor_bytes(int row_batch_count, int col_batch_count);

#ifdef __cplusplus
}
#endif

#endif /* CP_CUTLASS_H */
