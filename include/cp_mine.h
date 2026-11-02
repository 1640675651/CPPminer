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
                uint32_t cert_version,
                int sock, int* msg_id);

/* Offline --mock: fixed job, mine until the first share, verify it, free the host buffers
 * and shut the worker down. Returns 0 on PASS, 1 on FAIL. */
int cp_mine_mock(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_MINE_H */
