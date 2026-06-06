#ifndef CP_POOL_H
#define CP_POOL_H

#include <stddef.h>
#include <stdint.h>

#include "cp_config.h"

#ifdef __cplusplus
extern "C" {
#endif

int cp_pool_connect(const char* host, int port);
void cp_pool_disconnect(void);
int cp_pool_socket(void);

int cp_pool_send_authorize(int msg_id, const char* wallet,
                           const char* worker, const char* agent);
int cp_pool_send_plain_proof_submit(int sock, int msg_id, const char* job_id,
                                    const char* plain_b64, double hs);

void cp_pool_reader_start(void);
void cp_pool_reader_stop(void);
void cp_pool_inbox_clear(void);

/* 1 = line copied, 0 = timeout, -1 = connection lost */
int cp_pool_wait_line(char* out, size_t out_cap, int timeout_ms);
int cp_pool_conn_lost(void);

void cp_pool_set_submit_inflight(int on);
void cp_pool_log_share_submit_outcome(void);

int cp_pool_parse_notify(const char* json,
                         char* job_id, int job_len,
                         char* header_hex, int header_len,
                         char* target_hex, int target_len);

typedef struct {
    char job_id[128];
    char job_key[320];
    char target_hex[80];
    uint8_t header[INCOMPLETE_HEADER_BYTES];
    uint32_t tgt[8];
} CpPendingJob;

int cp_pool_take_pending_job(CpPendingJob* out);

double cp_pool_difficulty(void);
void cp_pool_set_difficulty(double d);

#ifdef __cplusplus
}
#endif

#endif /* CP_POOL_H */
