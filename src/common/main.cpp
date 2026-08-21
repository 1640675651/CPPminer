/*
 * CPminer — cross-platform LuckyPool plain_proof miner (CPU / CUDA / …).
 */
#include "cp_config.h"
#include "cp_fee.h"
#include "cp_mine.h"
#include "cp_noise.h"
#include "cp_pool.h"
#include "cp_platform.h"
#include "cp_proof.h"
#include "cp_share_queue.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
#include "gemm/case33_gemm_xor.hpp"
#endif

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
#include "cp_gpu.h"
#include "cp_cutlass.h"
#endif

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
#include "cp_opencl_align.h"
#include "cp_opencl_prep_profile.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(void)
{
    printf("CPminer — LuckyPool plain_proof miner\n");
    printf("  --pool URI         stratum+tcp://host:port\n");
    printf("  --wallet ADDR      wallet address\n");
    printf("  --worker NAME      worker name (default: rig01)\n");
    printf("  --agent NAME       agent string (default: cpminer/1.0)\n");
    printf("  --backend NAME     cpu");
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    printf("|cuda");
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("|opencl");
#endif
    printf(" (built: ");
    {
        int first = 1;
        if(cp_worker_has_cpu()){ printf("%scpu", first ? "" : ","); first = 0; }
        if(cp_worker_has_cuda()){ printf("%scuda", first ? "" : ","); first = 0; }
        if(cp_worker_has_opencl()){ printf("%sopencl", first ? "" : ","); first = 0; }
        if(first) printf("none");
    }
    printf(")\n");
    printf("  --devices N[,M]    device index(es): CUDA ids, or OpenCL flat index\n");
    printf("                     (default: 0; OpenCL prefers discrete GPU first)\n");
    printf("  --list-devices     list devices for the selected backend and exit\n");
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("  --ocl-platform P   OpenCL: only enumerate platform index P\n");
    printf("  --ocl-tile MxN     OpenCL hash tile: 8x8 (default), 4x8, or 8x16 (auto on AMD)\n");
    printf("  --ocl-issue MODE   OpenCL GEMM issue: auto (default), broadcast, or packed\n");
    printf("  --ocl-cpm-type T   OpenCL broadcast accumulate type: float (default) or int\n");
#endif
    printf("  --dev                m=n=8192 for testing\n");
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    printf("  --no-period-gemm     per-tile scan instead of period GEMM (CUDA debug)\n");
    printf("  --period-batch N     col-period batch (CUDA) / macro-block batch (OpenCL, default %d)\n",
           CP_PERIOD_BATCH_DEFAULT);
    printf("  --col-period-batch N alias for --period-batch\n");
    printf("  --row-period-batch N row-period batch size (default %d, max %d)\n",
           CP_ROW_PERIOD_BATCH_DEFAULT, CP_ROW_PERIOD_BATCH_MAX);
    printf("  --row-major-ap       row-major Ap/BpT (lda=%d; CUTLASS default)\n",
           K_DIM);
    printf("  --step-major         step-major Ap/BpT panels (lda=%d; cuBLAS period default)\n",
           R_RANK);
    printf("  --cutlass-fused      fused CUTLASS GEMM + jackpot (CUDA default)\n");
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    printf("  --cublas-period      debug: cuBLAS period GEMM + separate XOR/jackpot\n");
#endif
    printf("  --no-cutlass-fused   debug: non-CUTLASS period path (CUDA GEMM%s)\n",
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
           " or cuBLAS if probed"
#else
           ""
#endif
           );
    printf("  --cpu-gen            host matrix prep (OpenCL ~1 GiB VRAM; CUDA debug)\n");
    printf("  --align-test         run CPU/GPU hash alignment self-test and exit\n");
    printf("  --align-test-prod    include production m=n=%d checks (~1 GiB RAM, slow)\n",
           M_DIM);
    printf("  --profile-scan [N]   time GEMM vs jackpot per period batch (default N=10)\n");
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("  --profile-prep [N]   time OpenCL matrix prep phases (default N=3)\n");
#endif
    printf("  --max-nonce N        stop after N matrix attempts per job\n");
    printf("  --python EXE         Python for proof build/verify (CP_PYTHON env)\n");
    printf("  --host-bridge PATH   plain_proof_host.py path\n");
    printf("  --dry-run            build proof but do not submit\n");
    printf("  --verify             run in-process zk-pow verify before submit\n");
    printf("  --cert-version N     force certificate version for verify (1/2=legacy, 3=salted;\n");
    printf("                       default 3; without this flag, pool notify cert_version wins)\n");
    printf("  --mock / -mock       offline: fixed job, mine until first share, verify, exit\n");
    printf("  --mock-diff D        mock difficulty (default %.0f; higher = longer before share)\n",
           g_mock_diff);
    printf("  --prepack MODE       CPU prepack: separate (default), reuse, fused\n");
    printf("  --inplace-prepack    alias for --prepack reuse\n");
    printf("  --simd ISA           CPU SIMD: auto (default), avx2, ssse3, dotprod, neon, scalar\n");
    printf("                       (also CP_SIMD / CASE33_ISA env)\n");
    printf("  --simd-test          compare every available CPU SIMD kernel with scalar and exit\n");
}

static int handle_notify_line(const char* line, int* msg_id, char* cur_job_key)
{
    char job_id[128] = {0};
    char header_hex[320] = {0};
    char target_hex[80] = {0};
    uint32_t cert_version = 0;
    if(!cp_pool_parse_notify(line, job_id, sizeof(job_id),
                            header_hex, sizeof(header_hex),
                            target_hex, sizeof(target_hex),
                            &cert_version)){
        printf("[pool] mining.notify parse failed\n"); fflush(stdout);
        return CP_JOB_NONE;
    }
    cert_version = cp_resolve_cert_version(cert_version);

    char job_key[320];
    snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, header_hex);
    if(!strcmp(job_key, cur_job_key)){
        printf("[pool] duplicate notify ignored job=%s\n", job_id); fflush(stdout);
        return CP_JOB_NONE;
    }
    strncpy(cur_job_key, job_key, sizeof(cur_job_key) - 1);
    cur_job_key[319] = 0;

    uint8_t header[INCOMPLETE_HEADER_BYTES];
    int hlen = cp_hex_to_bytes(header_hex, header, INCOMPLETE_HEADER_BYTES);
    if(hlen != INCOMPLETE_HEADER_BYTES){
        printf("[pool] bad header length %d (need %d)\n", hlen, INCOMPLETE_HEADER_BYTES);
        fflush(stdout);
        return CP_JOB_NONE;
    }

    uint32_t tgt[8];
    memset(tgt, 0, sizeof(tgt));
    if(target_hex[0] && cp_be_target_hex_to_le_words(target_hex, tgt)){
        printf("[job] notify id=%s header=%.16s... pool_target (unscaled) cert_version=%u\n",
               job_id, header_hex, (unsigned)cert_version);
    } else {
        cp_target_from_difficulty(cp_pool_difficulty(), tgt);
        printf("[job] notify id=%s header=%.16s... diff=%.1f (no target in notify) cert_version=%u\n",
               job_id, header_hex, cp_pool_difficulty(), (unsigned)cert_version);
    }
    fflush(stdout);

    printf("[plain] mining job=%s%s...\n", job_id,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);
    int rc = cp_mine_job(header, hlen, job_id, target_hex, tgt, cert_version,
                         cp_pool_socket(), msg_id);
    if(rc == CP_JOB_FEE_SWITCH){
        printf("[fee] pausing job for wallet switch\n"); fflush(stdout);
        return rc;
    }
    if(rc == CP_JOB_CANCELLED){
        printf("[plain] job ended (new notify or disconnect)\n"); fflush(stdout);
    } else if(rc == CP_JOB_NONE){
        printf("[plain] job stopped (max_nonce or error)\n"); fflush(stdout);
    }

    CpPendingJob pj;
    while(rc == CP_JOB_CANCELLED && cp_pool_take_pending_job(&pj)){
        strncpy(cur_job_key, pj.job_key, 320);
        cur_job_key[319] = 0;
        printf("[plain] mining queued job=%s%s...\n", pj.job_id,
               cp_fee_next_is_dev() ? " [DEV FEE]" : "");
        fflush(stdout);
        rc = cp_mine_job(pj.header, INCOMPLETE_HEADER_BYTES, pj.job_id,
                         pj.target_hex, pj.tgt, pj.cert_version, cp_pool_socket(), msg_id);
        if(rc == CP_JOB_FEE_SWITCH){
            printf("[fee] pausing job for wallet switch\n"); fflush(stdout);
            return rc;
        }
        if(rc == CP_JOB_CANCELLED){
            printf("[plain] job ended (new notify or disconnect)\n"); fflush(stdout);
        } else if(rc == CP_JOB_NONE){
            printf("[plain] job stopped (max_nonce or error)\n"); fflush(stdout);
        }
    }
    return rc;
}

int main(int argc, char** argv)
{
    const char* pool_host = "pearl-cpu-eu1.luckypool.io";
    int pool_port = 3370;
    const char* wallet = NULL;
    int devs[MAX_GPUS] = {0};
    int ndev = 0;
    int align_test = 0;
    int align_test_prod = 0;
    int no_period_gemm = 0;
    int period_batch = CP_PERIOD_BATCH_DEFAULT;
    int row_period_batch = CP_ROW_PERIOD_BATCH_DEFAULT;
    int step_major_ap = -1; /* -1 = unset; CUTLASS→row-major, cuBLAS period→step-major */
    /* -1 = unset; CUDA defaults to fused CUTLASS, other backends force off. */
    int cutlass_fused = -1;
    CpPrepackMode prepack_mode = CP_PREPACK_SEPARATE;
    CpSimdIsa simd_isa = CP_SIMD_AUTO;
    int simd_env_invalid = 0;
    {
        const char* env = getenv("CP_SIMD");
        if(!env) env = getenv("CASE33_ISA");
        if(env){
            if(!strcmp(env, "avx2")) simd_isa = CP_SIMD_AVX2;
            else if(!strcmp(env, "sse") || !strcmp(env, "ssse3"))
                simd_isa = CP_SIMD_SSE;
            else if(!strcmp(env, "dotprod")) simd_isa = CP_SIMD_DOTPROD;
            else if(!strcmp(env, "neon")) simd_isa = CP_SIMD_NEON;
            else if(!strcmp(env, "scalar")) simd_isa = CP_SIMD_SCALAR;
            else if(!strcmp(env, "auto")) simd_isa = CP_SIMD_AUTO;
            else {
                fprintf(stderr, "unknown CP_SIMD/CASE33_ISA value %s\n", env);
                simd_env_invalid = 1;
            }
        }
    }
    int profile_scan = 0;
    int profile_runs = 10;
    int profile_prep = 0;
    int profile_prep_runs = 3;
    int simd_test = 0;
    int list_devices = 0;
    int ocl_platform = -1;
    int ocl_tile_mr = 0;
    int ocl_tile_nr = 0;
    int ocl_issue_mode = 0; /* 0=auto, 1=broadcast, 2=packed */
    int ocl_cpm_int = 0;
    CpBackendId backend_sel = CP_BACKEND_NONE;

    if(simd_env_invalid)
        return 1;

    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--pool") && i + 1 < argc){
            const char* u = argv[++i];
            const char* h = strstr(u, "://");
            if(h){
                h += 3;
                const char* colon = strchr(h, ':');
                if(colon){
                    int hlen = (int)(colon - h);
                    static char hbuf[256];
                    strncpy(hbuf, h, hlen); hbuf[hlen] = 0;
                    pool_host = hbuf;
                    pool_port = atoi(colon + 1);
                }
            }
        } else if(!strcmp(argv[i], "--wallet") && i + 1 < argc){
            wallet = argv[++i];
        } else if(!strcmp(argv[i], "--backend") && i + 1 < argc){
            const char* b = argv[++i];
            if(!strcmp(b, "cpu")) backend_sel = CP_BACKEND_CPU;
            else if(!strcmp(b, "cuda")) backend_sel = CP_BACKEND_CUDA;
            else if(!strcmp(b, "opencl")) backend_sel = CP_BACKEND_OPENCL;
            else {
                fprintf(stderr, "unknown --backend %s\n", b);
                return 1;
            }
        } else if((!strcmp(argv[i], "--device") || !strcmp(argv[i], "--devices")) && i + 1 < argc){
            const char* s = argv[++i];
            char tmp[256];
            strncpy(tmp, s, 255); tmp[255] = 0;
            char* tok = strtok(tmp, ",");
            while(tok && ndev < MAX_GPUS){ devs[ndev++] = atoi(tok); tok = strtok(NULL, ","); }
        } else if(!strcmp(argv[i], "--list-devices")){
            list_devices = 1;
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
        } else if(!strcmp(argv[i], "--ocl-platform") && i + 1 < argc){
            ocl_platform = atoi(argv[++i]);
        } else if(!strncmp(argv[i], "--ocl-tile", 10)){
            const char* v = argv[i] + 10;
            if(*v == '=') v++;
            else if(*v == '\0' && i + 1 < argc) v = argv[++i];
            else {
                fprintf(stderr, "--ocl-tile requires MxN (e.g. 4x8, 8x8, or 8x16)\n");
                return 1;
            }
            if(sscanf(v, "%dx%d", &ocl_tile_mr, &ocl_tile_nr) != 2 ||
               !((ocl_tile_mr == 4 && ocl_tile_nr == 8) ||
                 (ocl_tile_mr == 8 && (ocl_tile_nr == 8 || ocl_tile_nr == 16)))){
                fprintf(stderr, "invalid --ocl-tile %s (expected 4x8, 8x8, or 8x16)\n", v);
                return 1;
            }
        } else if(!strncmp(argv[i], "--ocl-issue", 11)){
            const char* v = argv[i] + 11;
            if(*v == '=') v++;
            else if(*v == '\0' && i + 1 < argc) v = argv[++i];
            else {
                fprintf(stderr, "--ocl-issue requires auto, broadcast, or packed\n");
                return 1;
            }
            if(!strcmp(v, "auto")){
                ocl_issue_mode = 0;
            } else if(!strcmp(v, "broadcast")){
                ocl_issue_mode = 1;
            } else if(!strcmp(v, "packed") || !strcmp(v, "dot4")){
                ocl_issue_mode = 2;
            } else {
                fprintf(stderr, "invalid --ocl-issue %s (expected auto, broadcast, or packed)\n", v);
                return 1;
            }
        } else if(!strncmp(argv[i], "--ocl-cpm-type", 14)){
            const char* v = argv[i] + 14;
            if(*v == '=') v++;
            else if(*v == '\0' && i + 1 < argc) v = argv[++i];
            else {
                fprintf(stderr, "--ocl-cpm-type requires float or int\n");
                return 1;
            }
            if(!strcmp(v, "float") || !strcmp(v, "fp32")){
                ocl_cpm_int = 0;
            } else if(!strcmp(v, "int") || !strcmp(v, "int32")){
                ocl_cpm_int = 1;
            } else {
                fprintf(stderr, "invalid --ocl-cpm-type %s (expected float or int)\n", v);
                return 1;
            }
#endif
        } else if(!strcmp(argv[i], "--dev")){
            g_dev_dims = 1;
        } else if(!strcmp(argv[i], "--no-period-gemm")){
            no_period_gemm = 1;
        } else if(!strncmp(argv[i], "--period-batch", 14)){
            const char* v = argv[i] + 14;
            if(*v == '=') period_batch = atoi(v + 1);
            else if(i + 1 < argc) period_batch = atoi(argv[++i]);
        } else if(!strncmp(argv[i], "--col-period-batch", 18)){
            const char* v = argv[i] + 18;
            if(*v == '=') period_batch = atoi(v + 1);
            else if(i + 1 < argc) period_batch = atoi(argv[++i]);
        } else if(!strncmp(argv[i], "--row-period-batch", 18)){
            const char* v = argv[i] + 18;
            if(*v == '=') row_period_batch = atoi(v + 1);
            else if(i + 1 < argc) row_period_batch = atoi(argv[++i]);
        } else if(!strcmp(argv[i], "--row-major-ap")){
            step_major_ap = 0;
        } else if(!strcmp(argv[i], "--step-major")){
            step_major_ap = 1;
        } else if(!strcmp(argv[i], "--cutlass-fused")){
            cutlass_fused = 1;
        } else if(!strcmp(argv[i], "--cublas-period")){
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
            cutlass_fused = 0;
#else
            fprintf(stderr,
                    "--cublas-period requires rebuild with -DCP_ENABLE_CUBLAS=ON "
                    "(or build.ps1 -EnableCublas)\n");
            return 1;
#endif
        } else if(!strcmp(argv[i], "--no-cutlass-fused")){
            cutlass_fused = 0;
        } else if(!strcmp(argv[i], "--cpu-gen")){
            g_cpu_matrix_gen = 1;
        } else if(!strcmp(argv[i], "--inplace-prepack")){
            prepack_mode = CP_PREPACK_REUSE;
        } else if(!strcmp(argv[i], "--prepack") && i + 1 < argc){
            const char* mode = argv[++i];
            if(!strcmp(mode, "separate"))
                prepack_mode = CP_PREPACK_SEPARATE;
            else if(!strcmp(mode, "reuse") || !strcmp(mode, "inplace"))
                prepack_mode = CP_PREPACK_REUSE;
            else if(!strcmp(mode, "fused"))
                prepack_mode = CP_PREPACK_FUSED;
            else {
                fprintf(stderr, "unknown --prepack mode %s (separate|reuse|fused)\n", mode);
                return 1;
            }
        } else if(!strcmp(argv[i], "--simd") && i + 1 < argc){
            const char* isa = argv[++i];
            if(!strcmp(isa, "auto"))
                simd_isa = CP_SIMD_AUTO;
            else if(!strcmp(isa, "avx2"))
                simd_isa = CP_SIMD_AVX2;
            else if(!strcmp(isa, "sse") || !strcmp(isa, "ssse3"))
                simd_isa = CP_SIMD_SSE;
            else if(!strcmp(isa, "neon"))
                simd_isa = CP_SIMD_NEON;
            else if(!strcmp(isa, "dotprod"))
                simd_isa = CP_SIMD_DOTPROD;
            else if(!strcmp(isa, "scalar"))
                simd_isa = CP_SIMD_SCALAR;
            else {
                fprintf(stderr,
                        "unknown --simd %s (auto|avx2|ssse3|dotprod|neon|scalar)\n",
                        isa);
                return 1;
            }
        } else if(!strcmp(argv[i], "--simd-test")){
            simd_test = 1;
        } else if(!strcmp(argv[i], "--max-nonce") && i + 1 < argc){
            g_max_nonce = atoi(argv[++i]);
        } else if(!strcmp(argv[i], "--python") && i + 1 < argc){
            strncpy(g_python_exe, argv[++i], sizeof(g_python_exe) - 1);
            g_python_exe[sizeof(g_python_exe) - 1] = 0;
        } else if(!strcmp(argv[i], "--host-bridge") && i + 1 < argc){
            strncpy(g_host_bridge, argv[++i], sizeof(g_host_bridge) - 1);
            g_host_bridge[sizeof(g_host_bridge) - 1] = 0;
        } else if(!strcmp(argv[i], "--worker") && i + 1 < argc){
            strncpy(worker_global, argv[++i], sizeof(worker_global) - 1);
            worker_global[sizeof(worker_global) - 1] = 0;
        } else if(!strcmp(argv[i], "--agent") && i + 1 < argc){
            strncpy(agent_global, argv[++i], sizeof(agent_global) - 1);
            agent_global[sizeof(agent_global) - 1] = 0;
        } else if(!strcmp(argv[i], "--dry-run")){
            g_dry_run = 1;
        } else if(!strcmp(argv[i], "--verify")){
            g_plain_verify = 1;
        } else if(!strcmp(argv[i], "--cert-version") && i + 1 < argc){
            int v = atoi(argv[++i]);
            if(v < 1 || v > 3){
                fprintf(stderr, "--cert-version must be 1, 2, or 3 (got %d)\n", v);
                return 1;
            }
            g_cert_version = (uint32_t)v;
            g_cert_version_forced = 1;
        } else if(!strcmp(argv[i], "--mock") || !strcmp(argv[i], "-mock")){
            g_mock = 1;
        } else if(!strcmp(argv[i], "--mock-diff") && i + 1 < argc){
            g_mock_diff = atof(argv[++i]);
            if(g_mock_diff < 1.0) g_mock_diff = 1.0;
        } else if(!strcmp(argv[i], "--align-test")){
            align_test = 1;
        } else if(!strcmp(argv[i], "--align-test-prod")){
            align_test = 1;
            align_test_prod = 1;
        } else if(!strncmp(argv[i], "--profile-scan", 14)){
            profile_scan = 1;
            const char* v = argv[i] + 14;
            if(*v == '=') profile_runs = atoi(v + 1);
            else if(i + 1 < argc && argv[i + 1][0] != '-')
                profile_runs = atoi(argv[++i]);
            if(profile_runs < 1) profile_runs = 1;
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
        } else if(!strncmp(argv[i], "--profile-prep", 14)){
            profile_prep = 1;
            const char* v = argv[i] + 14;
            if(*v == '=') profile_prep_runs = atoi(v + 1);
            else if(i + 1 < argc && argv[i + 1][0] != '-')
                profile_prep_runs = atoi(argv[++i]);
            if(profile_prep_runs < 1) profile_prep_runs = 1;
#endif
        } else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
            print_usage();
            return 0;
        }
    }

    if(list_devices){
        int n = 0;
        if(backend_sel == CP_BACKEND_NONE){
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
            if(cp_worker_has_cuda()){
                if(cp_worker_select(CP_BACKEND_CUDA) != 0) return 1;
                n += cp_worker_list_devices();
            }
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
            if(cp_worker_has_opencl()){
                if(ocl_platform >= 0)
                    cp_worker_set_ocl_platform(ocl_platform);
                if(cp_worker_select(CP_BACKEND_OPENCL) != 0) return 1;
                n += cp_worker_list_devices();
            }
#endif
            if(n <= 0){
                printf("[list-devices] no CUDA/OpenCL backends in this build\n");
                return 1;
            }
            return 0;
        }
        if(cp_worker_select(backend_sel) != 0) return 1;
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
        if(ocl_platform >= 0)
            cp_worker_set_ocl_platform(ocl_platform);
#endif
        n = cp_worker_list_devices();
        return n > 0 ? 0 : 1;
    }

    if(cp_worker_select(backend_sel) != 0) return 1;

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(ocl_platform >= 0)
        cp_worker_set_ocl_platform(ocl_platform);
    if(ocl_tile_mr > 0)
        cp_worker_set_ocl_tile(ocl_tile_mr, ocl_tile_nr);
    if(ocl_issue_mode != 0)
        cp_worker_set_ocl_issue_mode(ocl_issue_mode);
    if(ocl_cpm_int)
        cp_worker_set_ocl_cpm_int(1);
#endif

    if(cp_worker_backend_id() == CP_BACKEND_CUDA){
        if(cutlass_fused < 0) cutlass_fused = 1;
    } else {
        cutlass_fused = 0;
    }
    /* Case 10 (CUTLASS default) needs contiguous K (row-major Ap/BpT).
     * Step-major falls back to Case 9 wind_down; cuBLAS period / Case 7.2 packing. */
    if(step_major_ap < 0)
        step_major_ap = cutlass_fused ? 0 : 1;

    cp_worker_apply_backend_defaults();

#if (defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA) || (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL)
    if(align_test){
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
        if(!ndev){ devs[0] = 0; ndev = 1; }
        cp_worker_apply_backend_defaults();
        cp_worker_set_period_gemm(!no_period_gemm);
        cp_worker_set_period_batch(period_batch);
        cp_worker_set_row_period_batch(row_period_batch);
        cp_worker_set_step_major_ap(step_major_ap);
        cp_worker_set_cutlass_fused(cutlass_fused);
        pearl_set_cutlass_fused(cutlass_fused);
        g_cutlass_fused = cutlass_fused;
        if(pearl_run_alignment_tests() != 0) return 1;
        if(align_test_prod){
            const int pm = g_dev_dims ? DEV_M_DIM : M_DIM;
            const int pn = g_dev_dims ? DEV_N_DIM : N_DIM;
            if(g_dev_dims){
                printf("[align-test-prod] DEV m=n=%d (omit --dev for production)\n", DEV_M_DIM);
            }
            if(pearl_run_alignment_tests_prod(pm, pm, K_DIM) != 0) return 1;
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
            if(bid == CP_BACKEND_CUDA &&
               cp_gpu_run_alignment_tests(devs[0], pm, pn) != 0) return 1;
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
            if(bid == CP_BACKEND_OPENCL &&
               cp_opencl_run_alignment_tests(devs[0], pm, pn) != 0) return 1;
#endif
        }
        printf("[align-test] all tests passed\n");
        return 0;
    }
#endif

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(profile_scan){
        if(cp_worker_backend_id() != CP_BACKEND_CUDA){
            fprintf(stderr, "--profile-scan requires CUDA backend\n");
            return 1;
        }
        if(!ndev){ devs[0] = 0; ndev = 1; }
        if(no_period_gemm){
            fprintf(stderr, "--profile-scan requires period GEMM (omit --no-period-gemm)\n");
            return 1;
        }
        cp_worker_apply_backend_defaults();
        cp_worker_set_period_gemm(1);
        cp_worker_set_period_batch(period_batch);
        cp_worker_set_row_period_batch(row_period_batch);
        cp_worker_set_step_major_ap(step_major_ap);
        cp_worker_set_cutlass_fused(cutlass_fused);
        pearl_set_cutlass_fused(cutlass_fused);
        int pm = g_dev_dims ? DEV_M_DIM : M_DIM;
        int pn = g_dev_dims ? DEV_N_DIM : N_DIM;
        if(g_dev_dims){
            printf("[profile-scan] DEV m=n=%d (omit --dev for production)\n", DEV_M_DIM);
        }
        return cp_gpu_run_scan_profile(devs[0], pm, pn, 2, profile_runs) != 0;
    }
#else
    (void)profile_scan;
    (void)profile_runs;
#endif

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(profile_prep){
        if(cp_worker_backend_id() != CP_BACKEND_OPENCL){
            fprintf(stderr, "--profile-prep requires OpenCL backend\n");
            return 1;
        }
        if(!ndev){ devs[0] = 0; ndev = 1; }
        int pm = g_dev_dims ? DEV_M_DIM : M_DIM;
        int pn = g_dev_dims ? DEV_N_DIM : N_DIM;
        if(g_dev_dims){
            printf("[profile-prep] DEV m=n=%d (omit --dev for production)\n", DEV_M_DIM);
        }
        const int warmup = profile_prep_runs > 1 ? 1 : 0;
        return cp_opencl_run_prep_profile(devs[0], pm, pn, warmup, profile_prep_runs) != 0;
    }
#else
    (void)profile_prep;
    (void)profile_prep_runs;
#endif

#if !((defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA) || (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL))
    if(align_test){
        fprintf(stderr, "--align-test requires CUDA or OpenCL backend (rebuild with -Backend Cuda/OpenCl)\n");
        return 1;
    }
    (void)align_test_prod;
#endif
#if !(defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA)
    if(profile_scan){
        fprintf(stderr, "--profile-scan requires CUDA backend\n");
        return 1;
    }
    (void)profile_runs;
#endif
#if !(defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL)
    if(profile_prep){
        fprintf(stderr, "--profile-prep requires OpenCL backend (rebuild with -Backend OpenCl)\n");
        return 1;
    }
    (void)profile_prep_runs;
#endif

    if(!wallet){
        if(g_mock){
            wallet = "mock-wallet";
        } else {
            fprintf(stderr, "--wallet required\n");
            return 1;
        }
    }
    if(!ndev){ devs[0] = 0; ndev = 1; }

    if(g_mock){
        /* Offline self-test: no pool submit; always verify the first share. */
        g_dry_run = 1;
        g_plain_verify = 1;
    }

    strncpy(wallet_global, wallet, sizeof(wallet_global) - 1);
    wallet_global[sizeof(wallet_global) - 1] = 0;

    /* Offline mock skips the pool; no fee reconnects. */
    cp_fee_init(wallet_global, g_mock ? 0 : 1);

    cp_worker_apply_backend_defaults();
    cp_worker_set_period_gemm(!no_period_gemm);
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL
       && period_batch == CP_PERIOD_BATCH_DEFAULT){
        period_batch = CP_MACRO_BATCH_DEFAULT;
    }
    /* Resolve tile (incl. broadcast auto 4x8) before mode banner / fee tile counts. */
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL){
        cp_worker_configure_ocl_tile(devs[0]);
        cp_worker_apply_backend_defaults();
    }
#endif
    cp_worker_set_period_batch(period_batch);
    cp_worker_set_row_period_batch(row_period_batch);
    cp_worker_set_step_major_ap(step_major_ap);
    cp_worker_set_cutlass_fused(cutlass_fused);
    pearl_set_cutlass_fused(cutlass_fused);
    g_cutlass_fused = cutlass_fused;
    cp_worker_set_prepack_mode(prepack_mode);
    if(cp_worker_set_simd_isa(simd_isa) != 0)
        return 1;
    if(simd_test){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
        const int rc = case33_test_simd_parity();
        printf("[cpu] SIMD parity test: %s\n", rc == 0 ? "passed" : "failed");
        return rc == 0 ? 0 : 1;
#else
        fprintf(stderr, "--simd-test requires a CPU-enabled build\n");
        return 1;
#endif
    }

    if(cutlass_fused){
        if(no_period_gemm){
            fprintf(stderr, "CUTLASS fused path requires period GEMM (omit --no-period-gemm)\n");
            return 1;
        }
    }

    if(g_dev_dims){
        g_m_active = DEV_M_DIM;
        g_n_active = DEV_N_DIM;
        printf("[mode] DEV m=n=%d (omit --dev for production m=n=%d)\n",
               DEV_M_DIM, M_DIM);
    } else {
        g_m_active = M_DIM;
        g_n_active = N_DIM;
    }

    cp_init_workdir();
    cp_resolve_paths(argc, argv);

    {
        double host_mib = ((double)g_m_active * K_DIM + (double)g_n_active * K_DIM)
                        / (1024.0 * 1024.0);
        const int contiguous = cp_worker_uses_contiguous_tiles();
        const int tile_layout = cp_worker_default_tile_layout();
        int row_parts = cp_pp_num_row_parts(g_m_active, contiguous);
        int col_parts = cp_pp_num_col_parts(g_n_active, contiguous);
        const char *tile_layout_name =
            cutlass_fused ? "CUTLASS MMA lane 8x8 interleaved (128x128 CTA)"
            : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_4x8) ? "contiguous 4x8 blocks"
            : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_8x8) ? "contiguous 8x8 blocks"
            : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS) ? "contiguous 8x16 blocks"
            : "BzMiner periodic scattered 8x16";
        printf("[mode] backend=%s\n", cp_worker_backend_name());
        printf("[mode] plain_proof m=%d n=%d k=%d r=%d%s\n",
               g_m_active, g_n_active, K_DIM, R_RANK,
               g_dev_dims ? " (dev)" : " (production)");
        printf("[mode] tile layout: %s\n", tile_layout_name);
        if(cp_worker_backend_id() == CP_BACKEND_CPU){
            printf("[mode] scan: fused GEMM + XOR + host jackpot\n");
            if(prepack_mode == CP_PREPACK_FUSED)
                printf("[mode] matrix steady: ~%.0f MiB signal + scan buffers (fused prepack)\n",
                       host_mib * 2.0);
            else if(prepack_mode == CP_PREPACK_REUSE)
                printf("[mode] matrix steady: ~%.0f MiB signal + scan buffers (reuse prepack)\n",
                       host_mib * 2.0);
            else
                printf("[mode] matrix peak: ~%.0f MiB host signal + ~%.0f MiB prepack\n",
                       host_mib, host_mib * 2.0);
        } else if(cp_worker_backend_id() == CP_BACKEND_OPENCL){
            const int tiles_per_macro =
                (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_4x8) ? (128 / 4) * (128 / 8)
                : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_8x8) ? (128 / 8) * (128 / 8)
                : (128 / 8) * (128 / 16);
            printf("[mode] scan: OpenCL fused GEMM + XOR + device jackpot\n");
            printf("[mode] macro batch: %d (%d hash tiles/launch, --period-batch)\n",
                   period_batch, period_batch * tiles_per_macro);
            printf("[mode] host signal ~%.0f MiB; noisy B cached on GPU per job\n", host_mib);
        } else if(cutlass_fused){
            printf("[mode] proof rows/cols: 8 A + 8 B^T (interleaved 4x4)\n");
            printf("[mode] scan: CUTLASS Case 10 fused GEMM + inline XOR jackpot\n");
        } else {
            printf("[mode] scan: %s\n",
                   (contiguous || no_period_gemm) ? "per-tile kernel"
                                                  : "period GEMM + batched jackpot");
        }
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
        if(cp_worker_backend_id() == CP_BACKEND_CUDA
           && !contiguous && !no_period_gemm){
            printf("[mode] Ap/BpT layout: %s (lda=%d)\n",
                   step_major_ap ? "step-major panels" : "row-major strided",
                   step_major_ap ? R_RANK : K_DIM);
            if(cutlass_fused){
                printf("[mode] jackpot: fused in GEMM kernel (no tile_xor / C_hist)\n");
                printf("[mode] period batch: row=%d col=%d\n",
                       row_period_batch, period_batch);
            } else {
                printf("[mode] jackpot: separate XOR kernel (period GEMM)\n");
                printf("[mode] period batch: row=%d col=%d (~%.0f MiB C_hist/GPU)\n",
                       row_period_batch, period_batch,
                       (double)row_period_batch * (double)period_batch
                       * (double)(K_DIM / R_RANK)
                       * (double)PP_ROW_PERIOD * (double)PP_COL_PERIOD
                       * (double)sizeof(int32_t) / (1024.0 * 1024.0));
            }
        }
#endif
        printf("[mode] hash_tiles=%dx%d (%d total)\n",
               row_parts, col_parts, row_parts * col_parts);
        printf("[mode] host~%.0f MiB (signal A+B)\n", host_mib);
        printf("[mode] matrix gen: %s\n",
               (g_cpu_matrix_gen || cp_worker_prefers_host_matrices())
                   ? "host BLAKE3 + noise"
                   : "device random + commitment/noise");
        printf("[mode] verify=%d dry_run=%d max_nonce=%d mock=%d cert_version=%u%s\n",
               g_plain_verify, g_dry_run, g_max_nonce, g_mock,
               (unsigned)g_cert_version,
               g_cert_version_forced ? " (forced)" : "");
        if(cp_fee_enabled()){
            printf("[mode] dev fee: 1%%\n");
        }
    }
    fflush(stdout);

    cp_worker_init(devs, ndev);
    {
        const int contiguous = cp_worker_uses_contiguous_tiles();
        const uint64_t t_tiles =
            (uint64_t)cp_pp_num_row_parts(g_m_active, contiguous) *
            (uint64_t)cp_pp_num_col_parts(g_n_active, contiguous);
        cp_fee_set_tiles_per_matrix(t_tiles);
    }
    cp_mine_init_host_buffers();

    if(g_mock){
        /* Fixed legal stratum-style job id + deterministic 76-byte incomplete header. */
        static const char k_mock_job_id[] = "00000000-0000-4000-8000-000000000001";
        uint8_t header[INCOMPLETE_HEADER_BYTES];
        memset(header, 0, sizeof(header));
        /* Minimal non-zero fields so the blob is not all-zero (version + tag). */
        header[0] = 0x01;
        header[1] = 0x00;
        header[2] = 0x00;
        header[3] = 0x00;
        memcpy(header + 4, "CPMOCK", 6);
        header[10] = 0x01; /* mock revision */

        /* Mock difficulty → pool target (same path as mining.set_difficulty). */
        uint32_t tgt[8];
        cp_target_from_difficulty(g_mock_diff, tgt);
        char target_hex[65];
        cp_le_words_to_be_target_hex(tgt, target_hex);

        printf("[mock] job_id=%s (offline, no pool)\n", k_mock_job_id);
        printf("[mock] difficulty=%.1f target=%.16s... cert_version=%u%s\n",
               g_mock_diff, target_hex, (unsigned)g_cert_version,
               g_cert_version_forced ? " (forced)" : "");
        printf("[mock] mining until first share + zk-pow verify...\n");
        fflush(stdout);

        const int rc = cp_mine_job(header, INCOMPLETE_HEADER_BYTES, k_mock_job_id, target_hex, tgt,
                                   g_cert_version, -1, NULL);
        const int outcome = cp_mine_last_share_outcome();
        cp_mine_free_host_buffers();
        cp_worker_shutdown();

        if(rc == CP_JOB_CANCELLED){
            fprintf(stderr, "[mock] cancelled before share\n");
            return 1;
        }
        if(outcome == CP_SHARE_OUTCOME_OK){
            printf("[mock] PASS: first share built and verified\n");
            fflush(stdout);
            return 0;
        }
        if(outcome == CP_SHARE_OUTCOME_NONE){
            fprintf(stderr, "[mock] FAIL: no share produced\n");
        } else if(outcome == CP_SHARE_OUTCOME_VERIFY_FAIL){
            fprintf(stderr, "[mock] FAIL: share verify failed\n");
        } else if(outcome == CP_SHARE_OUTCOME_PROOF_FAIL){
            fprintf(stderr, "[mock] FAIL: proof build failed\n");
        } else {
            fprintf(stderr, "[mock] FAIL: share outcome=%d\n", outcome);
        }
        return 1;
    }

    char cur_job_key[320] = {0};
    int msg_id = 1;

reconnect:
    cp_pool_reader_stop();
    cp_pool_disconnect();
    cp_pool_inbox_clear();
    cur_job_key[0] = 0;

    printf("[main] Connecting to %s:%d...\n", pool_host, pool_port);
    while(1){
        if(cp_pool_connect(pool_host, pool_port) >= 0) break;
        printf("[main] Reconnecting in 5 sec...\n"); fflush(stdout);
        cp_sleep(5);
    }

    if(!cp_pool_send_authorize(msg_id++, cp_fee_wallet(), worker_global, agent_global))
        goto reconnect;
    cp_fee_on_authorized();
    if(cp_fee_enabled()){
        printf("[fee] authorized as %s (debt=%llu / 100*T=%llu)\n",
               cp_fee_next_is_dev() ? "DEV FEE wallet" : "your wallet",
               (unsigned long long)cp_fee_debt(),
               (unsigned long long)cp_fee_threshold());
        fflush(stdout);
    }

    cp_pool_reader_start();

    while(1){
        char line_buf[65536];
        int got = cp_pool_wait_line(line_buf, sizeof(line_buf), -1);
        if(got < 0){
            printf("[net] Connection lost, reconnecting...\n"); fflush(stdout);
            goto reconnect;
        }
        if(got == 0) continue;

        if(strstr(line_buf, "mining.notify")){
            int rc = handle_notify_line(line_buf, &msg_id, cur_job_key);
            if(rc == CP_JOB_FEE_SWITCH || cp_pool_conn_lost()) goto reconnect;
            continue;
        }

        if(strstr(line_buf, "mining.set_difficulty")){
            double d = cp_json_num(line_buf, "params");
            if(!d){
                const char* p = strstr(line_buf, "\"params\":[");
                if(p){
                    p = strchr(p, '[');
                    if(p) d = atof(p + 1);
                }
            }
            if(d > 0.0){
                cp_pool_set_difficulty(d);
                printf("[pool] mining.set_difficulty %.0f\n", d); fflush(stdout);
            }
            continue;
        }

        if(strstr(line_buf, "result") || strstr(line_buf, "error")){
            printf("[pool] jsonrpc: %s\n", line_buf); fflush(stdout);
            continue;
        }

        printf("[pool] (unhandled) %s\n", line_buf); fflush(stdout);
    }

    cp_mine_free_host_buffers();
    cp_worker_shutdown();
    return 0;
}
