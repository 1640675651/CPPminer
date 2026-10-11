#ifndef CP_FEE_H
#define CP_FEE_H

#include <stdint.h>

#include "cp_algo.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Same-pool developer fee (debt model). The work unit depends on the algo:
 *   - Pearl:   hash tiles; T = tiles in one full matrix scan
 *   - Quantus: microseconds of active hashing; T = CP_FEE_QPOW_TURN_US
 * User work: debt += units
 * When debt >= 100 * T, run fee work; fee work: debt -= 100 * units (clamped at 0)
 * Leave fee mode when debt < 100*T (Pearl: one fee matrix, never split) or when
 * debt reaches 0 (Quantus: a full T-long turn, so wallet switches stay rare).
 * Seed debt = 50 * T so the first fee lands mid-period.
 * Reconnect + re-authorize/login when the wanted wallet changes.
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

#ifdef __cplusplus
}
#endif

#endif /* CP_FEE_H */
