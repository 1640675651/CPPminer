#ifndef CP_FEE_H
#define CP_FEE_H

#include <stdint.h>

#include "cp_algo.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Developer fee (debt model). The work unit depends on the algo:
 *   - Pearl:   hash tiles; T = tiles in one full matrix scan
 *   - Quantus: microseconds of active hashing; T = CP_FEE_QPOW_TURN_US
 * User work: debt += units
 * When debt >= 100 * T, run fee work; fee work: debt -= 100 * units (clamped at 0)
 * Leave fee mode when debt < 100*T (Pearl: one fee matrix, never split) or when
 * debt reaches 0 (Quantus: a full T-long turn, so wallet switches stay rare).
 * Seed debt = 50 * T so the first fee lands mid-period.
 * Reconnect + re-authorize/login when the wanted wallet changes.
 *
 * Fee sessions go to fixed fee pools (LuckyPool, then Kryptex), tried in order, then to the
 * user's pool. A session that gets no job (connect, authorize or first job failed)
 * moves on to the next pool; when the user's pool fails too, the fee cycle is
 * skipped and counted as paid, so a pool that rejects the fee wallet never stops
 * the user's mining. Repeated failures back off over later fee cycles.
 */

#define CP_FEE_PERIOD 100
#define CP_FEE_QPOW_TURN_US (10ull * 1000000ull) /* 10 s fee per 1000 s mining */

void cp_fee_init(const char* user_wallet, int enable, CpAlgoId algo);

/* T in the algo's unit: Pearl full-matrix hash-tile count for the active
 * backend/layout/dims (call after backend + g_m_active/g_n_active are known, and
 * again if they change); Quantus CP_FEE_QPOW_TURN_US. Seeds debt = 50*T on the
 * first non-zero T while fee is enabled. */
void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix);

void cp_fee_on_authorized(void);

const char* cp_fee_wallet(void);

/* Enter fee mode if debt threshold hit; call at each matrix boundary before mine. */
void cp_fee_prepare_matrix(void);

int cp_fee_next_is_dev(void);
int cp_fee_needs_switch(void);

/* Charge work in the algo's unit: Pearl tiles from a scan (complete or cancelled
 * partial); Quantus microseconds spent hashing (excludes connect/login/job waits). */
void cp_fee_note_tiles(uint64_t tiles);

/* Debt-style value in the algo's unit, for logs ("123 tiles" / "12.3 s"). */
const char* cp_fee_format(uint64_t units, char* buf, int cap);

uint64_t cp_fee_debt(void);
uint64_t cp_fee_tiles_per_matrix(void);
uint64_t cp_fee_threshold(void);
int cp_fee_enabled(void);
/* 1 while the current pool session mines for the dev fee */
int cp_fee_session_is_dev(void);

/* Prints the "[mode] dev fee" line (no-op when the fee is disabled). */
void cp_fee_print_mode(void);

/* Pool for the next session: a fee session gets the current fee pool (or the user's
 * pool as the fallback), a user session the user's pool. Call before each connect. */
void cp_fee_session_pool(const char* user_host, int user_port, const char** host, int* port);
/* 1 if the session from cp_fee_session_pool is on a fixed fee pool: authorize there
 * with the default password, not the user's --pool-pass. */
int cp_fee_session_on_fee_pool(void);
/* The session got a job. */
void cp_fee_session_ok(void);
/* The session ended (connect failed or reconnecting). A fee session without a job
 * moves to the next pool or skips the fee cycle; returns 1 when the next connect
 * should go out at once instead of after the reconnect backoff. */
int cp_fee_session_end(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_FEE_H */
