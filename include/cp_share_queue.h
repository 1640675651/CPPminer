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
    uint64_t tiles_at_hit;
    double hit_elapsed_sec;
    int copy_bt;
} CpShareHit;

CpShareQueue *cp_share_queue_create(int max_depth);
void cp_share_queue_destroy(CpShareQueue *q);

void cp_share_queue_begin_job(CpShareQueue *q, const CpShareJobCtx *ctx, const char *job_key);
void cp_share_queue_end_job(CpShareQueue *q);

/* Copies signal A (and optionally B) then enqueues proof work. Blocks if queue is full. */
int cp_share_queue_enqueue_hit(CpShareQueue *q, const CpShareHit *hit, const uint8_t *header,
                               int hlen, const char *job_id, const char *target_hex,
                               const int8_t *a_src, size_t sz_a, const int8_t *bt_src,
                               size_t sz_bt);

#ifdef __cplusplus
}
#endif

#endif /* CP_SHARE_QUEUE_H */
