#ifndef CP_PROOF_H
#define CP_PROOF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* tile_layout for cp_proof_build:
 *   0 = BzMiner scattered (8 A rows + 16 B^T rows)
 *   1 = contiguous debug (8 + 16)
 *   2 = CUTLASS Case 10 MMA lane 8x8 interleaved (128x128 CTA, 64 cells/thread)
 *   3 = contiguous 8x8 (8 A rows + 8 B^T rows)
 */
#define CP_TILE_LAYOUT_SCATTERED  0
#define CP_TILE_LAYOUT_CONTIGUOUS 1
#define CP_TILE_LAYOUT_CUTLASS    2
#define CP_TILE_LAYOUT_CONTIGUOUS_8x8 3

/* Build plain_proof base64 in-process (Rust/pearl-blake3). Returns 0 on ok, -1 on error.
 * mining_config is retained for ABI compatibility but job_key is derived from tile_layout. */
int cp_proof_build(
    const uint8_t* header,
    size_t header_len,
    const uint8_t* mining_config,
    size_t config_len,
    const int8_t* a,
    const int8_t* bt,
    int m,
    int n,
    int k,
    int rank,
    int t_rows,
    int t_cols,
    int tile_layout,
    char* out_b64,
    size_t out_cap,
    char* err,
    size_t err_cap);

/* Verify plain_proof base64 against pool share target (32-byte BE U256, unscaled).
 * Returns 0 on ok. */
int cp_proof_verify(
    const uint8_t* header,
    size_t header_len,
    const uint8_t* proof_b64,
    size_t proof_b64_len,
    const uint8_t* pool_target_be,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* CP_PROOF_H */
