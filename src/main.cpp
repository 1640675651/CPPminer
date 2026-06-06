/*
 * CPminer — cross-platform LuckyPool plain_proof GPU miner.
 */
#include "cp_config.h"
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
    printf("  --max-nonce N        stop after N matrix attempts per job\n");
    printf("  --python EXE         Python for proof build/verify (CP_PYTHON env)\n");
    printf("  --host-bridge PATH   plain_proof_host.py path\n");
    printf("  --dry-run            build/verify proof but do not submit\n");
    printf("  --no-verify          skip verify before submit\n");
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
        } else if(!strcmp(argv[i], "--no-verify")){
            g_plain_verify = 0;
        } else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
            print_usage();
            return 0;
        }
    }

    if(!wallet){ fprintf(stderr, "--wallet required\n"); return 1; }
    if(!ndev){ devs[0] = 0; ndev = 1; }

    strncpy(wallet_global, wallet, sizeof(wallet_global) - 1);
    wallet_global[sizeof(wallet_global) - 1] = 0;

    pearl_set_contiguous_tiles(g_contiguous_tiles);
    cp_gpu_set_contiguous_tiles(g_contiguous_tiles);

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
               g_contiguous_tiles ? "contiguous 8x16 blocks" : "BzMiner periodic");
        printf("[mode] hash_tiles=%dx%d (%d total)\n",
               row_parts, col_parts, row_parts * col_parts);
        printf("[mode] host~%.0f MiB (signal A+B)\n", host_mib);
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
