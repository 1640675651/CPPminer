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
char agent_global[64] = "cppminer/0.4";
int g_dry_run = 0;
int g_plain_verify = 0;
int g_mock = 0;
/* Set only via --mock-diff; ignored unless g_mock_diff_forced. */
double g_mock_diff = 0.0;
int g_mock_diff_forced = 0;
uint32_t g_cert_version = 3;
int g_cert_version_forced = 0;
int g_cpu_matrix_gen = 0;
int g_max_nonce = 0;
int g_qpow_threads = 0;

uint32_t cp_resolve_cert_version(uint32_t notify_cert_version)
{
    if(g_cert_version_forced)
        return g_cert_version;
    if(notify_cert_version >= 1 && notify_cert_version <= 3)
        return notify_cert_version;
    return g_cert_version;
}

double cp_resolve_mock_diff(int algo_quantus)
{
    if(g_mock_diff_forced){
        double d = g_mock_diff;
        if(d < 1.0) d = 1.0;
        return d;
    }
    return algo_quantus ? CP_MOCK_DIFF_QUANTUS_DEFAULT : CP_MOCK_DIFF_PEARL_DEFAULT;
}
