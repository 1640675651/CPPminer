#ifndef CP_JOB_CTRL_H
#define CP_JOB_CTRL_H

#ifdef __cplusplus
extern "C" {
#endif

void cp_job_mine_begin(const char* job_key);
void cp_job_mine_end(void);
int cp_job_should_cancel(void);

/* Used by pool reader when a newer job arrives during mining. */
void cp_job_request_cancel(void);
int cp_job_mining_active(void);
const char* cp_job_mining_key(void);
int cp_job_key_matches(const char* job_key);

#ifdef __cplusplus
}
#endif

#endif /* CP_JOB_CTRL_H */
