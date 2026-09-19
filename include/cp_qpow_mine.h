#ifndef CP_QPOW_MINE_H
#define CP_QPOW_MINE_H

#include "cp_qpow_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mine until share submitted, fee switch, cancel, or connection loss.
 * Returns CP_JOB_NONE, CP_JOB_FEE_SWITCH, or CP_JOB_CANCELLED. */
int cp_qpow_mine_job(const CpQpowJob* job, int sock, int* msg_id,
                     const char* worker_name);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_MINE_H */
