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

/* Offline --mock: fixed job, mine until first share, Poseidon2-verify, exit.
 * Returns 0 on PASS, 1 on FAIL. Difficulty via cp_resolve_mock_diff (U512). */
int cp_qpow_mine_mock(const char* worker_name);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_MINE_H */
