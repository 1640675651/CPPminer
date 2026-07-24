#ifndef CP_MINE_H
#define CP_MINE_H

#include <stdint.h>

#include "cp_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void cp_mine_init_host_buffers(void);
void cp_mine_free_host_buffers(void);

/* After a job finishes (end_job drained), last share outcome from the proof queue. */
int cp_mine_last_share_outcome(void);

int cp_mine_job(const uint8_t* header, int hlen,
                const char* job_id,
                const char* target_hex,
                const uint32_t pool_tgt[8],
                int sock, int* msg_id);

#ifdef __cplusplus
}
#endif

#endif /* CP_MINE_H */
