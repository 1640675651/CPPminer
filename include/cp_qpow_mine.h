#ifndef CP_QPOW_MINE_H
#define CP_QPOW_MINE_H

#include "cp_qpow_pool.h"
#include "cp_worker.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --simd for the Quantus CPU path (Poseidon2 has scalar and AVX2 kernels):
 *   hybrid  one scalar + one AVX2 worker per physical core when AVX2 and SMT
 *           are present (the two paths load different execution ports),
 *           otherwise scalar;
 *   auto    best available: currently identical to hybrid, may pick a wider
 *           kernel (e.g. AVX-512) in the future;
 *   avx2 / avxvnni   AVX2 on every thread (fails if the CPU lacks AVX2);
 *   scalar / ssse3 / neon / dotprod   scalar on every thread.
 * Returns 0, or -1 when an explicit AVX2 request cannot be honoured. */
int cp_qpow_set_simd_isa(CpSimdIsa isa);

/* Mine until share submitted, fee switch, cancel, or connection loss.
 * Returns CP_JOB_NONE, CP_JOB_FEE_SWITCH, or CP_JOB_CANCELLED. */
int cp_qpow_mine_job(const CpQpowJob* job, int sock, int* msg_id,
                     const char* worker_name);

/* Offline --mock: fixed job, mine until first share, Poseidon2-verify, exit.
 * Returns 0 on PASS, 1 on FAIL. Difficulty via cp_resolve_mock_diff (U512). */
int cp_qpow_mine_mock(const char* worker_name);

/* Returns 0 if CPU thread nonce ranges remain distinct for all extranonce lengths. */
int cp_qpow_nonce_thread_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_QPOW_MINE_H */
