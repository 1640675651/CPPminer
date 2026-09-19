#ifndef CP_QPOW_POOL_H
#define CP_QPOW_POOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CP_QPOW_HEADER_BYTES 32
#define CP_QPOW_NONCE_BYTES  64
#define CP_QPOW_TARGET_BYTES 64
#define CP_QPOW_EXTRANONCE_MAX 32

typedef struct {
    char job_id[128];
    char job_key[160];
    uint8_t mining_hash[CP_QPOW_HEADER_BYTES];
    uint8_t target[CP_QPOW_TARGET_BYTES];
    uint8_t extranonce[CP_QPOW_EXTRANONCE_MAX];
    int extranonce_len;
    double difficulty;
    uint64_t seq;
    int clean_jobs;
} CpQpowJob;

/* Enable Quantus job dispatch in the shared pool reader (0 = Pearl). */
void cp_qpow_pool_set_active(int on);

void cp_qpow_pool_clear(void);

/* Session UUID from login result; used in every submit. */
void cp_qpow_pool_set_session_id(const char* id);
const char* cp_qpow_pool_session_id(void);

int cp_qpow_pool_send_login(int msg_id, const char* login, const char* worker,
                            const char* agent);

int cp_qpow_pool_send_submit(int sock, int msg_id, const char* job_id,
                             const uint8_t nonce[CP_QPOW_NONCE_BYTES]);

/* Parse login ack or job notify. Returns 1 on success. */
int cp_qpow_pool_parse_login_result(const char* json, char* session_out, int session_len,
                                    CpQpowJob* job_out);
int cp_qpow_pool_parse_job(const char* json, CpQpowJob* out);

/* Called from Pearl pool reader when Quantus mode is active. Returns 1 if handled. */
int cp_qpow_pool_on_line(const char* line);

int cp_qpow_pool_take_pending(CpQpowJob* out);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_POOL_H */
