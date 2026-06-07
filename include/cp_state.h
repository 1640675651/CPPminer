#ifndef CP_STATE_H
#define CP_STATE_H

#include <stdint.h>

#include "cp_config.h"
#include "cp_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int g_dev_dims;
extern int g_contiguous_tiles;
extern int g_m_active;
extern int g_n_active;
extern char g_workdir[MAX_PATH];
extern char g_python_exe[512];
extern char g_host_bridge[512];
extern int8_t* h_Ap_global;
extern int8_t* h_BpT_global;
extern char wallet_global[256];
extern char worker_global[64];
extern char agent_global[64];
extern int g_dry_run;
extern int g_plain_verify;
extern int g_cpu_matrix_gen;
extern int g_max_nonce;

#ifdef __cplusplus
}
#endif

#endif /* CP_STATE_H */
