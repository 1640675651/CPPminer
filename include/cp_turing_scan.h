/* Turing (sm_75) fused scan kernel: INT8 mma.sync.m8n8k16 GEMM with the 32
 * cumulative milestone folds, keyed BLAKE3 and target check, on noisy operands
 * stored in the packed layout of cp_turing_layout.cuh. Produces the same hash
 * tiles, words and hit coordinates as the CUTLASS fused kernel. */
#ifndef CP_TURING_SCAN_H
#define CP_TURING_SCAN_H

#include <stddef.h>
#include <stdint.h>

#include "cp_cutlass.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Nonzero if dev is sm_75 (the kernel is tuned for Turing; Ampere and newer
 * keep the CUTLASS tensorop80 path). */
int cp_turing_scan_supported(int dev);

/* One period batch: rows [row_period0, +row_batch) x cols [col_period0,
 * +col_batch) in 128-row/col periods of packed d_Ap (m x K) and d_BpT (n x K).
 * row_period0 and row_batch must be even. Launches on the legacy stream of the
 * current device. d_dump_words (optional) receives the 16 transcript words of
 * every hash tile, indexed like the CUTLASS tile dump: (cta * 256 + vt) * 16
 * with cta = local virtual row period * col_batch + local col period. */
int cp_turing_period_batch(int dev, const int8_t* d_Ap, const int8_t* d_BpT, int m, int n,
                           int row_period0, int col_period0, int row_batch, int col_batch,
                           const CpCutlassJackpotLaunch* jackpot, uint32_t* d_dump_words);

/* Row-major rows x K -> packed (blk = 256 for A, 128 for B^T), async on stream 0. */
int cp_turing_pack(const int8_t* d_src, int8_t* d_dst, int rows, int blk);

#ifdef __cplusplus
}
#endif

#endif /* CP_TURING_SCAN_H */
