#ifndef CP_POOL_H
#define CP_POOL_H

#include <stddef.h>
#include <stdint.h>

#include "cp_config.h"
#include "cp_job_ctrl.h"

#ifdef __cplusplus
extern "C" {
#endif

int cp_pool_connect(const char* host, int port);
void cp_pool_disconnect(void);
int cp_pool_socket(void);

int cp_pool_send_authorize(int msg_id, const char* wallet,
                           const char* worker, const char* agent);
/* Reader-side: the response to the pending authorize id. Returns 1 if the
 * line was that response (handled). */
int cp_pool_on_authorize_response(const char* line);
/* Reader enforces authorize/first-job deadlines while this waits for success. */
int cp_pool_wait_authorized(void);
int cp_pool_send_plain_proof_submit(int sock, int msg_id, const char* job_id,
                                    const char* plain_b64, double hs);
/* Registers the id before sending, and removes it on failure or its own ACK. */
int cp_pool_send_tracked_submit(int sock, int msg_id, const char* json);

void cp_pool_reader_start(void);
void cp_pool_reader_stop(void);
void cp_pool_inbox_clear(void);

/* Blocking read of one JSON object (reader must be stopped). 1=ok, 0=timeout, -1=lost */
int cp_pool_recv_one(char* out, size_t out_cap, int timeout_ms);

/* 1 = line copied, 0 = timeout, -1 = connection lost */
int cp_pool_wait_line(char* out, size_t out_cap, int timeout_ms);
int cp_pool_conn_lost(void);

void cp_pool_log_share_submit_outcome(void);
/* Positive finite difficulty from direct params or a singleton array. */
int cp_pool_parse_difficulty(const char* json, double* difficulty_out);

int cp_pool_parse_notify(const char* json,
                         char* job_id, int job_len,
                         char* header_hex, int header_len,
                         char* target_hex, int target_len,
                         uint32_t* cert_version_out);

typedef struct {
    char job_id[128];
    char job_key[CP_JOB_KEY_CAP];
    char target_hex[80];
    uint8_t header[INCOMPLETE_HEADER_BYTES];
    uint32_t tgt[8];
    uint32_t cert_version; /* 1/2=legacy seeds, 3=salted */
} CpPendingJob;

int cp_pool_take_pending_job(CpPendingJob* out);

double cp_pool_difficulty(void);
void cp_pool_set_difficulty(double d);

#ifdef __cplusplus
}
#endif

#endif /* CP_POOL_H */
