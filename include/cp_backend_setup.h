#ifndef CP_BACKEND_SETUP_H
#define CP_BACKEND_SETUP_H

#include "cp_options.h"

/* --list-devices; returns the exit status. */
int cp_list_devices(const CpOptions* o);

/* Selects o->backend_sel and passes the backend options to it; resolves cutlass_fused and
 * step_major_ap. Returns nonzero if the backend is not available. */
int cp_apply_backend_options(CpOptions* o);

/* Quantus: launch batch, backend init, CPU affinity and SIMD. Returns nonzero on failure. */
int cp_quantus_backend_init(CpOptions* o);
void cp_quantus_backend_shutdown(void);

/* Pearl: batches, jackpot mode and per-backend launch settings, oneDNN init, SIMD.
 * Returns nonzero on failure. */
int cp_pearl_backend_setup(CpOptions* o);

/* The [mode] lines describing the Pearl scan. */
void cp_print_pearl_mode(const CpOptions* o);

#endif /* CP_BACKEND_SETUP_H */
