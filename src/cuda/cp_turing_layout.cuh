/* Packed noisy-operand layout of the Turing (sm_75) scan kernel.
 *
 * Rows are grouped in blocks of `blk` rows (256 for A, 128 for B^T). Every
 * (block, k-tile of 32) is one contiguous blk x 32-byte record, so a CTA reads
 * each k-tile of its operands as a single contiguous block:
 *
 *   offset(row, l) = ((row / blk) * (K / 32) + l / 32) * (blk * 32)
 *                    + (row % blk) * 32 + l % 32
 */
#ifndef CP_TURING_LAYOUT_CUH
#define CP_TURING_LAYOUT_CUH

#include <stddef.h>

#define CP_TURING_BK    32
#define CP_TURING_A_BLK 256
#define CP_TURING_B_BLK 128

__host__ __device__ __forceinline__ size_t cp_turing_packed_offset(int row, int l, int k, int blk)
{
    return ((size_t)(row / blk) * (size_t)(k / CP_TURING_BK) + (size_t)(l / CP_TURING_BK)) *
               ((size_t)blk * CP_TURING_BK) +
           (size_t)(row % blk) * CP_TURING_BK + (size_t)(l % CP_TURING_BK);
}

#endif /* CP_TURING_LAYOUT_CUH */
