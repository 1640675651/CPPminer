#ifndef CP_SHARE_QUEUE_H
#define CP_SHARE_QUEUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CpShareQueue CpShareQueue;

typedef struct CpShareJobCtx {
    int sock;
    int *msg_id;
    int m;
    int n;
    const char *hdr_path;
    const char *proof_path;
} CpShareJobCtx;

typedef struct CpShareHit {
    uint64_t nonce;
    int t_rows;
    int t_cols;
    /* Tiles scanned since the previous share (or job start for the first). */
    uint64_t tiles_since_prev;
    /* Wall seconds since the previous share (or job start for the first). */
    double interval_sec;
    /* If non-zero, also hand off B (bt_io). Zero-B GPU paths leave B shared as zeros. */
    int handoff_bt;
} CpShareHit;

CpShareQueue *cp_share_queue_create(int max_depth);
void cp_share_queue_destroy(CpShareQueue *q);

void cp_share_queue_begin_job(CpShareQueue *q, const CpShareJobCtx *ctx, const char *job_key);
void cp_share_queue_end_job(CpShareQueue *q);

/*
 * Hands off host signal matrices to the proof worker (no memcpy).
 * Takes ownership of *a_io (required) and *bt_io when hit->handoff_bt.
 * Sets handed-off pointers to NULL. Blocks if a prior handoff is still in use
 * (single-slot / depth-1 ownership).
 */
int cp_share_queue_enqueue_hit(CpShareQueue *q, const CpShareHit *hit, const uint8_t *header,
                               int hlen, const char *job_id, const char *target_hex,
                               int8_t **a_io, size_t sz_a, int8_t **bt_io, size_t sz_bt);

/*
 * Waits until any loaned matrices are returned, then restores them into *a_io / *bt_io
 * when those pointers are NULL. No-op if the miner already holds the buffers.
 */
void cp_share_queue_reclaim_matrices(CpShareQueue *q, int8_t **a_io, int8_t **bt_io);

#ifdef __cplusplus
}
#endif

#endif /* CP_SHARE_QUEUE_H */
