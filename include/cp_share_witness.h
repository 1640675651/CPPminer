#ifndef CP_SHARE_WITNESS_H
#define CP_SHARE_WITNESS_H

#include <stdint.h>
#include <stdlib.h>

#include "cp_proof.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Host-owned proof inputs for one share (malloc'd; release with cp_share_witness_free).
 * Signal A: device sub-roots + the blocks covering the tile rows.
 * Signal B^T: all-zero, so only its sub-roots; blocks are synthesized by the proof builder. */
typedef struct CpShareWitness {
    int tile_layout;
    uint8_t a_root[32];
    uint8_t* a_subroots;
    size_t a_num_subroots;
    uint8_t* a_blocks;
    uint32_t a_block_idx[CP_WITNESS_MAX_BLOCKS];
    size_t a_num_blocks;
    uint8_t bt_root[32];
    uint8_t* bt_subroots;
    size_t bt_num_subroots;
} CpShareWitness;

static inline void cp_share_witness_free(CpShareWitness* w)
{
    if(!w) return;
    free(w->a_subroots);
    free(w->a_blocks);
    free(w->bt_subroots);
    free(w);
}

#ifdef __cplusplus
}
#endif

#endif /* CP_SHARE_WITNESS_H */
