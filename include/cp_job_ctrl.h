#ifndef CP_JOB_CTRL_H
#define CP_JOB_CTRL_H

/* Full Pearl or Quantus work fields, plus a 127-byte pool job id. */
#define CP_JOB_KEY_CAP 512

#ifdef __cplusplus
extern "C" {
#endif

void cp_job_mine_begin(const char* job_key);
/* Publish before waking the consumer; a stale begin is cancelled atomically. */
void cp_job_publish_work(const char* job_key);
void cp_job_reset_work(void);
void cp_job_mine_end(void);
int cp_job_should_cancel(void);

/* Used by pool reader when a newer job arrives during mining. */
void cp_job_request_cancel(void);
int cp_job_mining_active(void);
/* Thread-local snapshot; remains valid until this thread calls it again. */
const char* cp_job_mining_key(void);
int cp_job_key_matches(const char* job_key);

#ifdef __cplusplus
}
#endif

#endif /* CP_JOB_CTRL_H */
