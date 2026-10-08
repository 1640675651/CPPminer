#ifndef CP_CUTLASS_H
#define CP_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Warp-level MMA selection for the fused Case-10 kernel (--cuda-mma). */
#define CP_CUTLASS_MMA_AUTO        0  /* tensorop80 on sm_80+, tensorop on sm_75+,
                                         simt otherwise */
#define CP_CUTLASS_MMA_SIMT        1  /* dp4a SIMT (Pascal path) */
#define CP_CUTLASS_MMA_TENSOROP    2  /* Sm75: mma.sync.m8n8k16, 2-stage */
#define CP_CUTLASS_MMA_TENSOROP80  3  /* Sm80+: mma.sync.m16n8k32, multistage
                                         cp.async mainloop */
#define CP_CUTLASS_MMA_TENSOROP_MS 4  /* multistage mainloop + m8n8k16 (sm_75+;
                                         A/B and sm_75 validation) */
#define CP_CUTLASS_MMA_KINDS       5

void cp_cutlass_set_mma_mode(int mode);
int cp_cutlass_mma_mode(void);
/* Kernel actually used on this device under the current mode (a kind other
 * than AUTO). */
int cp_cutlass_mma_kind(int dev);
const char* cp_cutlass_mma_kind_name(int kind);
/* CLI name of a kind/mode ("auto", "simt", "tensorop", "tensorop80", ...). */
const char* cp_cutlass_mma_mode_name(int mode);
/* Nonzero if kernel `kind` can run on dev (arch and, for tensorop80, an
 * sm_80+ kernel image in the binary). */
int cp_cutlass_kind_supported(int dev, int kind);

/* Returns nonzero if the fused GEMM selected by the MMA mode can run on dev. */
int cp_cutlass_device_ok(int dev);

/* Checks the hash-tile policy of tensor-op kind `kind` against CUTLASS's own
 * accumulator iterator (no MMA executed, runs on any sm_61+ device): random
 * fragments are stored through the warp IteratorC and the 256 virtual SIMT
 * hash-tile XORs are recomputed on the host. Returns 0 if all 256 words match
 * and each tile is emitted exactly once, the number of bad tiles otherwise,
 * -1 on CUDA error. */
int cp_cutlass_hash_policy_selftest(int dev, int kind);

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
