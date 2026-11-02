#ifndef CP_DIAGNOSTICS_H
#define CP_DIAGNOSTICS_H

#include "cp_options.h"

/* Self-tests and profiling modes; each returns the process exit status. */
int cp_qpow_selftest(void);       /* --qpow-selftest */
int cp_qpow_simd_test(void);      /* --simd-test with --algo quantus */
int cp_pearl_simd_test(void);     /* --simd-test */
int cp_prepack_test(void);        /* --prepack-test */

/* --align-test, --profile-scan, --profile-prep: -1 when none was requested. */
int cp_run_test_modes(CpOptions* o);

#endif /* CP_DIAGNOSTICS_H */
