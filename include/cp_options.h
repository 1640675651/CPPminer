#ifndef CP_OPTIONS_H
#define CP_OPTIONS_H

#include "cp_algo.h"
#include "cp_config.h"
#include "cp_worker.h"

#include <stddef.h>

/* Command-line settings of one run; defaults are the values main() starts from. */
struct CpOptions {
    const char* pool_host = "pearl-cpu-eu1.luckypool.io";
    int pool_port = 3370;
    int pool_specified = 0;
    const char* wallet = NULL;
    CpAlgoId algo_sel = CP_ALGO_PEARL;
    int devs[MAX_GPUS] = {0};
    int ndev = 0;
    int devices_specified = 0;
    int align_test = 0;
    int align_test_prod = 0;
    int no_period_gemm = 0;
    int m_units = 0; /* --m / --n, 0 = production default */
    int n_units = 0;
    int batch_size = CP_PERIOD_BATCH_DEFAULT;
    int batch_size_set = 0;
    int row_period_batch_set = 0;
    int row_period_batch = CP_ROW_PERIOD_BATCH_DEFAULT;
    int step_major_ap = -1; /* -1 = unset; CUTLASS→row-major, cuBLAS period→step-major */
    /* -1 = unset; CUDA defaults to fused CUTLASS, other backends force off. */
    int cutlass_fused = -1;
    /* --cuda-mma: 0 auto (tensor cores on sm_75+), 1 simt dp4a, 2 tensorop. */
    int cuda_mma = 0;
    int fused_jackpot = -1;   /* unset: on for --backend onednn, off for hexagon and intel_npu */
    const char *onednn_layout = nullptr;
    CpPrepackMode prepack_mode = CP_PREPACK_FUSED;
    CpSimdIsa simd_isa = CP_SIMD_AUTO;
    int profile_scan = 0;
    int profile_runs = 10;
    int profile_prep = 0;
    int profile_prep_runs = 3;
    int simd_test = 0;
    int prepack_test = 0;
    int list_devices = 0;
    int ocl_platform = -1;
    int ocl_tile_mr = 0;
    int ocl_tile_nr = 0;
    int ocl_macro_m = 0;
    int ocl_macro_n = 0;
    int ocl_issue_mode = 0; /* 0=auto, 1=broadcast, 2=packed */
    int ocl_dot_policy = 0; /* Case32OclDotPolicy */
    int ocl_cpm_type = 0; /* Case32OclCpmType: 0=auto, 1=float, 2=int, 3=mad24 */
    int ocl_lds = 0;
    int ocl_double_buff = 0;
    int ocl_throttle = 0; /* cl_khr_throttle_hints: 0 = off, 1 = high, 2 = med, 3 = low */
    int wgpu_lds = -1;
    int wgpu_tile_mr = 0;
    int wgpu_tile_nr = 0;
    int wgpu_macro_m = 0;
    int wgpu_macro_n = 0;
    CpBackendId backend_sel = CP_BACKEND_NONE;
    int api_port = 0;              /* --api-port, 0 = no stats API */
    const char* api_bind = NULL;   /* --api-bind, NULL = 127.0.0.1 */
};

/* CP_SIMD / CASE33_ISA, then argv; sets g_m_active / g_n_active from --m / --n.
 * Returns -1 to continue, else the exit status (--help, --qpow-selftest, invalid input). */
int cp_options_parse(int argc, char** argv, CpOptions* o);

#endif /* CP_OPTIONS_H */
