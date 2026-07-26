#ifndef CP_FEE_H
#define CP_FEE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Same-pool developer fee (tile-debt model):
 *   - T = hash tiles in one full matrix scan (backend/layout/dims; not matrix dim M)
 *   - User scans: debt += tiles
 *   - When debt >= 100 * T, run fee matrices until debt is paid down
 *   - Fee scans: debt -= 100 * tiles (clamped at 0); leave fee mode when debt < 100*T
 *   - Seed debt = 50 * T so the first fee lands mid-period
 * Reconnect + re-authorize when the wanted wallet changes.
 */

#define CP_FEE_PERIOD 100

void cp_fee_init(const char* user_wallet, int enable);

/* Full-matrix hash-tile count T for the active backend/layout/dims. Call after
 * backend + g_m_active/g_n_active are known (and again if they change). Seeds
 * debt = 50*T on the first non-zero T while fee is enabled. */
void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix);

void cp_fee_on_authorized(void);

const char* cp_fee_wallet(void);

/* Enter fee mode if debt threshold hit; call at each matrix boundary before mine. */
void cp_fee_prepare_matrix(void);

int cp_fee_next_is_dev(void);
int cp_fee_needs_switch(void);

/* Charge tiles from a scan (complete or cancelled partial). */
void cp_fee_note_tiles(uint64_t tiles);

uint64_t cp_fee_debt(void);
uint64_t cp_fee_tiles_per_matrix(void);
uint64_t cp_fee_threshold(void);
int cp_fee_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_FEE_H */
