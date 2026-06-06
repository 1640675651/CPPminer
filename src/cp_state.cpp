#include "cp_state.h"

int g_dev_dims = 0;
int g_contiguous_tiles = 0;
int g_m_active = M_DIM;
int g_n_active = N_DIM;
char g_workdir[MAX_PATH] = ".";
char g_python_exe[512] = "python";
char g_host_bridge[512] = "plain_proof_host.py";
int8_t* h_Ap_global = NULL;
int8_t* h_BpT_global = NULL;
char wallet_global[256] = {0};
char worker_global[64] = "rig01";
char agent_global[64] = "cpminer/1.0";
int g_dry_run = 0;
int g_plain_verify = 0;
int g_max_nonce = 0;
