#include "cp_state.h"

int g_dev_dims = 0;
int g_cutlass_fused = 0;
int g_m_active = M_DIM;
int g_n_active = N_DIM;
char g_workdir[MAX_PATH] = ".";
char g_python_exe[512] = "python";
char g_host_bridge[512] = "plain_proof_host.py";
int8_t* h_Ap_global = NULL;
int8_t* h_BpT_global = NULL;
char wallet_global[256] = {0};
char worker_global[64] = "rig01";
char agent_global[64] = "cppminer/0.2.1";
int g_dry_run = 0;
int g_plain_verify = 0;
int g_mock = 0;
/* Mock scan difficulty (cp_target_from_difficulty). Higher = rarer shares / longer run.
 * ~58 is typically a few–tens of seconds on --dev before the first share. */
double g_mock_diff = 58.0;
int g_cpu_matrix_gen = 0;
int g_max_nonce = 0;
