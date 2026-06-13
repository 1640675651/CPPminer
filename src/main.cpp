/*
 * CPminer — cross-platform LuckyPool plain_proof GPU miner.
 */
#include "cp_config.h"
#include "cp_cutlass.h"
#include "cp_gpu.h"
#include "cp_mine.h"
#include "cp_noise.h"
#include "cp_pool.h"
#include "cp_platform.h"
#include "cp_state.h"
#include "cp_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(void)
{
    printf("CPminer — LuckyPool plain_proof GPU miner\n");
    printf("  --pool URI         stratum+tcp://host:port\n");
    printf("  --wallet ADDR      wallet address\n");
    printf("  --worker NAME      worker name (default: rig01)\n");
    printf("  --agent NAME       agent string (default: cpminer/1.0)\n");
    printf("  --devices N[,M]    CUDA devices (default: 0)\n");
    printf("  --dev                m=n=8192 for testing\n");
    printf("  --contiguous-tiles   8x16 contiguous hash tiles (debug)\n");
    printf("  --no-period-gemm     per-tile scan instead of period GEMM (debug)\n");
    printf("  --period-batch N     col-period batch size for cuBLAS (default %d, max %d)\n",
           CP_PERIOD_BATCH_DEFAULT, CP_PERIOD_BATCH_MAX);
    printf("  --row-major-ap       row-major Ap/BpT (lda=%d; default step-major lda=%d)\n",
           K_DIM, R_RANK);
    printf("  --cutlass-fused      fused CUTLASS GEMM + per-thread tile XOR (Pascal SIMT)\n");
    printf("  --cpu-gen            CPU BLAKE3 matrix gen (debug; default GPU random)\n");
    printf("  --max-nonce N        stop after N matrix attempts per job\n");
    printf("  --python EXE         Python for proof build/verify (CP_PYTHON env)\n");
    printf("  --host-bridge PATH   plain_proof_host.py path\n");
    printf("  --dry-run            build proof but do not submit\n");
    printf("  --verify             run in-process zk-pow verify before submit\n");
    printf("  --align-test         run CPU/GPU hash alignment self-test and exit\n");
    printf("  --align-test-prod    include production m=n=%d checks (~1 GiB RAM, slow)\n",
           M_DIM);
    printf("  --profile-scan [N]   time GEMM vs jackpot per period batch (default N=10)\n");
}

static void handle_notify_line(const char* line, int* msg_id, char* cur_job_key)
{
    char job_id[128] = {0};
    char header_hex[320] = {0};
    char target_hex[80] = {0};
    if(!cp_pool_parse_notify(line, job_id, sizeof(job_id),
                            header_hex, sizeof(header_hex),
                            target_hex, sizeof(target_hex))){
        printf("[pool] mining.notify parse failed\n"); fflush(stdout);
        return;
    }

    char job_key[320];
    snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, header_hex);
    if(!strcmp(job_key, cur_job_key)){
        printf("[pool] duplicate notify ignored job=%s\n", job_id); fflush(stdout);
        return;
    }
    strncpy(cur_job_key, job_key, sizeof(cur_job_key) - 1);
    cur_job_key[319] = 0;

    uint8_t header[INCOMPLETE_HEADER_BYTES];
    int hlen = cp_hex_to_bytes(header_hex, header, INCOMPLETE_HEADER_BYTES);
    if(hlen != INCOMPLETE_HEADER_BYTES){
        printf("[pool] bad header length %d (need %d)\n", hlen, INCOMPLETE_HEADER_BYTES);
        fflush(stdout);
        return;
    }

    uint32_t tgt[8];
    memset(tgt, 0, sizeof(tgt));
    if(target_hex[0] && cp_be_target_hex_to_le_words(target_hex, tgt)){
        printf("[job] notify id=%s header=%.16s... pool_target (unscaled)\n",
               job_id, header_hex);
    } else {
        cp_target_from_difficulty(cp_pool_difficulty(), tgt);
        printf("[job] notify id=%s header=%.16s... diff=%.1f (no target in notify)\n",
               job_id, header_hex, cp_pool_difficulty());
    }
    fflush(stdout);

    printf("[plain] mining job=%s...\n", job_id); fflush(stdout);
    int rc = cp_mine_job(header, hlen, job_id, target_hex, tgt,
                         cp_pool_socket(), msg_id);
    if(rc == CP_JOB_CANCELLED){
        printf("[plain] job ended (new notify or disconnect)\n"); fflush(stdout);
    } else if(rc == CP_JOB_NONE){
        printf("[plain] job stopped (max_nonce or error)\n"); fflush(stdout);
    }

    CpPendingJob pj;
    while(rc == CP_JOB_CANCELLED && cp_pool_take_pending_job(&pj)){
        strncpy(cur_job_key, pj.job_key, 320);
        cur_job_key[319] = 0;
        printf("[plain] mining queued job=%s...\n", pj.job_id); fflush(stdout);
        rc = cp_mine_job(pj.header, INCOMPLETE_HEADER_BYTES, pj.job_id,
                         pj.target_hex, pj.tgt, cp_pool_socket(), msg_id);
        if(rc == CP_JOB_CANCELLED){
            printf("[plain] job ended (new notify or disconnect)\n"); fflush(stdout);
        } else if(rc == CP_JOB_NONE){
            printf("[plain] job stopped (max_nonce or error)\n"); fflush(stdout);
        }
    }
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
    int step_major_ap = 1;
    int cutlass_fused = 0;
    int profile_scan = 0;
    int profile_runs = 10;

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
        } else if((!strcmp(argv[i], "--device") || !strcmp(argv[i], "--devices")) && i + 1 < argc){
            const char* s = argv[++i];
            char tmp[256];
            strncpy(tmp, s, 255); tmp[255] = 0;
            char* tok = strtok(tmp, ",");
            while(tok && ndev < MAX_GPUS){ devs[ndev++] = atoi(tok); tok = strtok(NULL, ","); }
        } else if(!strcmp(argv[i], "--dev")){
            g_dev_dims = 1;
        } else if(!strcmp(argv[i], "--contiguous-tiles")){
            g_contiguous_tiles = 1;
        } else if(!strcmp(argv[i], "--no-period-gemm")){
            no_period_gemm = 1;
        } else if(!strncmp(argv[i], "--period-batch", 14)){
            const char* v = argv[i] + 14;
            if(*v == '=') period_batch = atoi(v + 1);
            else if(i + 1 < argc) period_batch = atoi(argv[++i]);
        } else if(!strcmp(argv[i], "--row-major-ap")){
            step_major_ap = 0;
        } else if(!strcmp(argv[i], "--step-major")){
            step_major_ap = 1;
        } else if(!strcmp(argv[i], "--cutlass-fused")){
            cutlass_fused = 1;
        } else if(!strcmp(argv[i], "--cpu-gen")){
            g_cpu_matrix_gen = 1;
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
        } else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
            print_usage();
            return 0;
        }
    }

    if(align_test){
        if(!ndev){ devs[0] = 0; ndev = 1; }
        pearl_set_contiguous_tiles(g_contiguous_tiles);
        cp_gpu_set_contiguous_tiles(g_contiguous_tiles);
        cp_gpu_set_period_gemm(!no_period_gemm);
        cp_gpu_set_period_batch(period_batch);
        cp_gpu_set_step_major_ap(step_major_ap);
        cp_gpu_set_cutlass_fused(cutlass_fused);
        pearl_set_cutlass_fused(cutlass_fused);
        g_cutlass_fused = cutlass_fused;
        if(pearl_run_alignment_tests() != 0) return 1;
        if(align_test_prod){
            if(pearl_run_alignment_tests_prod(M_DIM, M_DIM, K_DIM) != 0) return 1;
            if(cp_gpu_run_alignment_tests(devs[0], M_DIM, N_DIM) != 0) return 1;
        }
        printf("[align-test] all tests passed\n");
        return 0;
    }

    if(profile_scan){
        if(!ndev){ devs[0] = 0; ndev = 1; }
        if(no_period_gemm){
            fprintf(stderr, "--profile-scan requires period GEMM (omit --no-period-gemm)\n");
            return 1;
        }
        pearl_set_contiguous_tiles(g_contiguous_tiles);
        cp_gpu_set_contiguous_tiles(g_contiguous_tiles);
        cp_gpu_set_period_gemm(1);
        cp_gpu_set_period_batch(period_batch);
        cp_gpu_set_step_major_ap(step_major_ap);
        cp_gpu_set_cutlass_fused(cutlass_fused);
        pearl_set_cutlass_fused(cutlass_fused);
        int pm = g_dev_dims ? DEV_M_DIM : M_DIM;
        int pn = g_dev_dims ? DEV_N_DIM : N_DIM;
        if(g_dev_dims){
            printf("[profile-scan] DEV m=n=%d (omit --dev for production)\n", DEV_M_DIM);
        }
        return cp_gpu_run_scan_profile(devs[0], pm, pn, 2, profile_runs) != 0;
    }

    if(!wallet){ fprintf(stderr, "--wallet required\n"); return 1; }
    if(!ndev){ devs[0] = 0; ndev = 1; }

    strncpy(wallet_global, wallet, sizeof(wallet_global) - 1);
    wallet_global[sizeof(wallet_global) - 1] = 0;

    pearl_set_contiguous_tiles(g_contiguous_tiles);
    cp_gpu_set_contiguous_tiles(g_contiguous_tiles);
    cp_gpu_set_period_gemm(!no_period_gemm);
    cp_gpu_set_period_batch(period_batch);
    cp_gpu_set_step_major_ap(step_major_ap);
    cp_gpu_set_cutlass_fused(cutlass_fused);
    pearl_set_cutlass_fused(cutlass_fused);
    g_cutlass_fused = cutlass_fused;

    if(cutlass_fused){
        if(g_contiguous_tiles || no_period_gemm){
            fprintf(stderr, "--cutlass-fused requires period GEMM (omit --contiguous-tiles and --no-period-gemm)\n");
            return 1;
        }
        fprintf(stderr,
            "[warn] --cutlass-fused uses a non-BzMiner 16x8 hash tile; pool must accept this layout.\n"
            "       For production LuckyPool mining, omit --cutlass-fused (default BzMiner 8x16).\n");
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
        int row_parts = cp_pp_num_row_parts(g_m_active, g_contiguous_tiles);
        int col_parts = cp_pp_num_col_parts(g_n_active, g_contiguous_tiles);
        printf("[mode] plain_proof m=%d n=%d k=%d r=%d%s\n",
               g_m_active, g_n_active, K_DIM, R_RANK,
               g_dev_dims ? " (dev)" : " (production)");
        printf("[mode] tile layout: %s\n",
               cutlass_fused ? "CUTLASS epilogue scatter (16x8 / 128 cells per thread)"
               : (g_contiguous_tiles ? "contiguous 8x16 blocks"
                                     : "BzMiner periodic scattered 8x16"));
        if(cutlass_fused){
            printf("[mode] proof rows/cols: 16 A + 8 B^T (CUTLASS Case 7.1 offsets)\n");
        }
        printf("[mode] scan: %s\n",
               cutlass_fused ? "CUTLASS fused GEMM + tile_xor jackpot"
               : ((g_contiguous_tiles || no_period_gemm) ? "per-tile kernel"
                  : "period GEMM + batched jackpot"));
        if(!g_contiguous_tiles && !no_period_gemm){
            printf("[mode] Ap/BpT layout: %s (cuBLAS lda=%d)\n",
                   step_major_ap ? "step-major panels" : "row-major strided",
                   step_major_ap ? R_RANK : K_DIM);
            if(cutlass_fused){
                printf("[mode] tile_xor/batch: ~%.1f KiB (fused; no C_hist)\n",
                       (double)cp_cutlass_tile_xor_bytes(period_batch)
                           / 1024.0);
            } else {
                printf("[mode] period_batch=%d (~%.0f MiB C_hist/GPU)\n",
                       period_batch,
                       (double)period_batch * (double)(K_DIM / R_RANK)
                       * (double)PP_ROW_PERIOD * (double)PP_COL_PERIOD
                       * (double)sizeof(int32_t) / (1024.0 * 1024.0));
            }
        }
        printf("[mode] hash_tiles=%dx%d (%d total)\n",
               row_parts, col_parts, row_parts * col_parts);
        printf("[mode] host~%.0f MiB (signal A+B)\n", host_mib);
        printf("[mode] matrix gen: %s\n",
               g_cpu_matrix_gen ? "CPU BLAKE3 (debug --cpu-gen)"
                                : "GPU random + GPU commitment/noise");
        printf("[mode] verify=%d dry_run=%d max_nonce=%d\n",
               g_plain_verify, g_dry_run, g_max_nonce);
    }
    fflush(stdout);

    cp_gpu_init(devs, ndev);
    cp_mine_init_host_buffers();

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

    if(!cp_pool_send_authorize(msg_id++, wallet_global, worker_global, agent_global))
        goto reconnect;

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
            handle_notify_line(line_buf, &msg_id, cur_job_key);
            if(cp_pool_conn_lost()) goto reconnect;
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
    cp_gpu_shutdown();
    return 0;
}
