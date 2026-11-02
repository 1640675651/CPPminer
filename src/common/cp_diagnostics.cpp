#include "cp_diagnostics.h"
#include "cp_config.h"
#include "cp_noise.h"
#include "cp_qpow_mine.h"
#include "cp_qpow_pool.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#include "qpow/miner.hpp"

#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
#include "gemm/case33_gemm_xor.hpp"
#endif

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
#include "cp_gpu.h"
#endif

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
#include "cp_opencl_align.h"
#include "cp_opencl_prep_profile.h"
#endif

#include <stdio.h>
#include <string.h>

int cp_qpow_selftest(void)
{
    const char* login =
        "{\"id\":1,\"result\":{\"extensions\":[\"keepalive\"],"
        "\"id\":\"d5adbda4-fd6c-4f33-8924-b3c1ae35dcbc\","
        "\"job\":{\"algo\":\"qpow-poseidon2\",\"difficulty\":10000000000,"
        "\"extranonce\":\"00131609\",\"job_id\":\"70655-10000000000\","
        "\"mining_hash\":\"1925e1ae2f620d7162e637fae74fa790ca9526e878e01b52c43c1a762354c961\","
        "\"seq\":70655,"
        "\"target\":\"000000006df37f675ef6eadf5ab9a2072d44268d97df837e6748956e5c6c2117501e68855669e4b8356cf292464d9e16cc8d4655f8fb96f429b149fc87d74da4\"},"
        "\"status\":\"OK\"}}";
    const char* jobline =
        "{\"jsonrpc\":\"2.0\",\"method\":\"job\",\"params\":{\"clean_jobs\":true,"
        "\"job\":{\"algo\":\"qpow-poseidon2\",\"difficulty\":10000000000,"
        "\"extranonce\":\"0012e986\",\"job_id\":\"70257-10000000000\","
        "\"mining_hash\":\"93e8f5baa1b97f5945c69ec8736cda957f920aeae7e659af14679235a4ff7a51\","
        "\"seq\":70257,"
        "\"target\":\"000000006df37f675ef6eadf5ab9a2072d44268d97df837e6748956e5c6c2117501e68855669e4b8356cf292464d9e16cc8d4655f8fb96f429b149fc87d74da4\"}}}";
    char session[80];
    CpQpowJob j0, j1;
    int fail = 0;
    if(!cp_qpow_pool_parse_login_result(login, session, (int)sizeof(session), &j0)){
        fprintf(stderr, "FAIL login parse\n");
        fail++;
    } else if(strcmp(session, "d5adbda4-fd6c-4f33-8924-b3c1ae35dcbc") != 0){
        fprintf(stderr, "FAIL session id\n");
        fail++;
    } else if(j0.extranonce_len != 4 || j0.extranonce[0] != 0x00 ||
              j0.extranonce[1] != 0x13){
        fprintf(stderr, "FAIL login extranonce\n");
        fail++;
    }
    if(!cp_qpow_pool_parse_job(jobline, &j1)){
        fprintf(stderr, "FAIL job parse\n");
        fail++;
    } else if(strcmp(j1.job_id, "70257-10000000000") != 0 || !j1.clean_jobs){
        fprintf(stderr, "FAIL job fields\n");
        fail++;
    }
    if(cp_qpow_nonce_thread_selftest()){
        fprintf(stderr, "FAIL nonce thread separation\n");
        fail++;
    }
    /* One Poseidon2 hash against known midstate/header path. */
    {
        uint8_t header[32], nonce[64], hash[64];
        memset(nonce, 0, 64);
        if(cp_hex_to_bytes(
               "0000000000000000000000000000000000000000000000000000000000000000",
               header, 32) != 32)
            fail++;
        else {
            qpow::get_nonce_hash(header, nonce, hash);
            char hx[129];
            cp_bin_to_hex(hash, 64, hx);
            static const char* want =
                "8e64e3d8e0f38f882e8501f9e525df0a95d2e91e9cfc32c9248d756fb07780e2"
                "f8fdca2c5a54441e6fcd8d774a5f6aae72f36d1c76bc19f691a0d4f6c607e8cc";
            if(strcmp(hx, want) != 0){
                fprintf(stderr, "FAIL golden hash\n  got %s\n", hx);
                fail++;
            }
        }
    }
    if(fail){
        printf("%d qpow selftest failure(s)\n", fail);
        return 1;
    }
    printf("qpow selftest passed\n");
    return 0;
}

int cp_qpow_simd_test(void)
{
    /* Quantus has scalar and AVX2 Poseidon2 kernels; compare them and exit. */
    const int field_fail = qpow::test_avx2_field();
    const int parity_fail = qpow::test_avx2_hash_parity();
    printf("[qpow] AVX2 Poseidon2 vs scalar: field ops %s, hash parity %s (%s)\n",
           field_fail == 0 ? "passed" : "failed",
           parity_fail == 0 ? "passed" : "failed",
           qpow::cpu_has_avx2() ? "avx2 available" : "no avx2: scalar only");
    return (field_fail == 0 && parity_fail == 0) ? 0 : 1;
}

int cp_run_test_modes(CpOptions* o)
{
#if (defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA) || (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL)
    if(o->align_test){
        const CpBackendId bid = cp_worker_backend_id();
        int gpu_ok = 0;
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
        if(bid == CP_BACKEND_CUDA) gpu_ok = 1;
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
        if(bid == CP_BACKEND_OPENCL) gpu_ok = 1;
#endif
        if(!gpu_ok){
            fprintf(stderr, "--align-test requires CUDA or OpenCL backend\n");
            return 1;
        }
        if(!o->ndev){ o->devs[0] = 0; o->ndev = 1; }
        cp_worker_apply_backend_defaults();
        cp_worker_set_period_gemm(!o->no_period_gemm);
        cp_worker_set_period_batch(o->batch_size);
        cp_worker_set_row_period_batch(o->row_period_batch);
        cp_worker_set_step_major_ap(o->step_major_ap);
        cp_worker_set_cutlass_fused(o->cutlass_fused);
        pearl_set_cutlass_fused(o->cutlass_fused);
        g_cutlass_fused = o->cutlass_fused;
        if(pearl_run_alignment_tests() != 0) return 1;
        if(o->align_test_prod){
            const int pm = g_m_active;
            const int pn = g_n_active;
            printf("[align-test-prod] m=%d n=%d\n", pm, pn);
            fflush(stdout);
            if(pearl_run_alignment_tests_prod(pm, pn, K_DIM) != 0) return 1;
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
            if(bid == CP_BACKEND_CUDA &&
               cp_gpu_run_alignment_tests(o->devs[0], pm, pn) != 0) return 1;
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
            if(bid == CP_BACKEND_OPENCL &&
               cp_opencl_run_alignment_tests(o->devs[0], pm, pn) != 0) return 1;
#endif
        }
        printf("[align-test] all tests passed\n");
        return 0;
    }
#endif

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(o->profile_scan){
        if(cp_worker_backend_id() != CP_BACKEND_CUDA){
            fprintf(stderr, "--profile-scan requires CUDA backend\n");
            return 1;
        }
        if(!o->ndev){ o->devs[0] = 0; o->ndev = 1; }
        if(o->no_period_gemm){
            fprintf(stderr, "--profile-scan requires period GEMM (omit --no-period-gemm)\n");
            return 1;
        }
        cp_worker_apply_backend_defaults();
        cp_worker_set_period_gemm(1);
        cp_worker_set_period_batch(o->batch_size);
        cp_worker_set_row_period_batch(o->row_period_batch);
        cp_worker_set_step_major_ap(o->step_major_ap);
        cp_worker_set_cutlass_fused(o->cutlass_fused);
        pearl_set_cutlass_fused(o->cutlass_fused);
        printf("[profile-scan] m=%d n=%d\n", g_m_active, g_n_active);
        fflush(stdout);
        return cp_gpu_run_scan_profile(o->devs[0], g_m_active, g_n_active, 2, o->profile_runs) != 0;
    }
#else
    (void)o->profile_scan;
    (void)o->profile_runs;
#endif

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(o->profile_prep){
        if(cp_worker_backend_id() != CP_BACKEND_OPENCL){
            fprintf(stderr, "--profile-prep requires OpenCL backend\n");
            return 1;
        }
        if(!o->ndev){ o->devs[0] = 0; o->ndev = 1; }
        printf("[profile-prep] m=%d n=%d\n", g_m_active, g_n_active);
        fflush(stdout);
        const int warmup = o->profile_prep_runs > 1 ? 1 : 0;
        return cp_opencl_run_prep_profile(o->devs[0], g_m_active, g_n_active, warmup,
                                          o->profile_prep_runs) != 0;
    }
#else
    (void)o->profile_prep;
    (void)o->profile_prep_runs;
#endif

#if !((defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA) || (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL))
    if(o->align_test){
        fprintf(stderr, "--align-test requires CUDA or OpenCL backend (rebuild with -Backend Cuda/OpenCl)\n");
        return 1;
    }
    (void)o->align_test_prod;
#endif
#if !(defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA)
    if(o->profile_scan){
        fprintf(stderr, "--profile-scan requires CUDA backend\n");
        return 1;
    }
    (void)o->profile_runs;
#endif
#if !(defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL)
    if(o->profile_prep){
        fprintf(stderr, "--profile-prep requires OpenCL backend (rebuild with -Backend OpenCl)\n");
        return 1;
    }
    (void)o->profile_prep_runs;
#endif
    return -1;
}

int cp_pearl_simd_test(void)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    const int rc = case33_test_simd_parity();
    printf("[cpu] SIMD parity test: %s\n", rc == 0 ? "passed" : "failed");
    return rc == 0 ? 0 : 1;
#else
    fprintf(stderr, "--simd-test requires a CPU-enabled build\n");
    return 1;
#endif
}

int cp_prepack_test(void)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    const int rc_reuse = case33_test_inplace_prepack(CP_PREPACK_TEST_DIM, CP_PREPACK_TEST_DIM,
                                                     K_DIM);
    printf("[cpu] reuse prepack test (m=n=%d): %s (rc=%d)\n", CP_PREPACK_TEST_DIM,
           rc_reuse == 0 ? "passed" : "failed", rc_reuse);
    fflush(stdout);
    const int rc_fused = case33_test_fused_prepack(CP_PREPACK_TEST_DIM, CP_PREPACK_TEST_DIM,
                                                   K_DIM, R_RANK);
    printf("[cpu] fused prepack test (m=n=%d): %s (rc=%d)\n", CP_PREPACK_TEST_DIM,
           rc_fused == 0 ? "passed" : "failed", rc_fused);
    return (rc_reuse == 0 && rc_fused == 0) ? 0 : 1;
#else
    fprintf(stderr, "--prepack-test requires a CPU-enabled build\n");
    return 1;
#endif
}
