#include "cp_mine.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_pool.h"
#include "cp_proof.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cp_verify_proof_file(
    const char* hdr_path,
    const char* target_hex,
    const char* proof_path)
{
    uint8_t header[INCOMPLETE_HEADER_BYTES];
    uint8_t target_be[32];
    char errbuf[4096];
    char* b64 = NULL;
    size_t cap = 0;
    size_t n = 0;
    FILE* pf = NULL;
    int rc = -1;

    if(!target_hex[0]) return -1;

    FILE* hf = fopen(hdr_path, "rb");
    if(!hf){
        perror("verify: header open");
        return -1;
    }
    if(fread(header, 1, sizeof(header), hf) != sizeof(header)){
        fprintf(stderr, "verify: header must be %d bytes\n", INCOMPLETE_HEADER_BYTES);
        fclose(hf);
        return -1;
    }
    fclose(hf);

    if(cp_hex_to_bytes(target_hex, target_be, 32) != 32){
        fprintf(stderr, "verify: invalid target hex\n");
        return -1;
    }

    pf = fopen(proof_path, "rb");
    if(!pf){
        perror("verify: proof open");
        return -1;
    }
    fseek(pf, 0, SEEK_END);
    long fsz = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    if(fsz <= 0 || fsz > (long)PLAIN_PROOF_B64_MAX){
        fprintf(stderr, "verify: invalid proof size %ld\n", fsz);
        fclose(pf);
        return -1;
    }
    cap = (size_t)fsz + 1;
    b64 = (char*)malloc(cap);
    if(!b64){
        fclose(pf);
        return -1;
    }
    n = fread(b64, 1, (size_t)fsz, pf);
    fclose(pf);
    if(n != (size_t)fsz){
        free(b64);
        return -1;
    }
    while(n > 0 && (b64[n - 1] == '\n' || b64[n - 1] == '\r'))
        b64[--n] = 0;

    errbuf[0] = 0;
    if(cp_proof_verify(header, sizeof(header), (const uint8_t*)b64, n,
                       target_be, errbuf, sizeof(errbuf)) != 0){
        fprintf(stderr, "verify FAIL: %s\n", errbuf[0] ? errbuf : "unknown");
        free(b64);
        return -1;
    }
    free(b64);
    rc = 0;
    return rc;
}

void cp_mine_init_host_buffers(void)
{
    size_t szAp  = (size_t)g_m_active * K_DIM;
    size_t szBpT = (size_t)g_n_active * K_DIM;
    h_Ap_global = (int8_t*)malloc(szAp);
    h_BpT_global = (int8_t*)malloc(szBpT);
    if(!h_Ap_global || !h_BpT_global){
        fprintf(stderr, "OOM host matrices\n");
        exit(1);
    }
}

void cp_mine_free_host_buffers(void)
{
    free(h_Ap_global);
    free(h_BpT_global);
    h_Ap_global = NULL;
    h_BpT_global = NULL;
}

int cp_mine_job(
    const uint8_t* header, int hlen,
    const char* job_id,
    const char* target_hex,
    const uint32_t pool_tgt[8],
    int sock, int* msg_id)
{
    int rc = CP_JOB_NONE;
    char job_key[320];
    char hdr_prefix[20];
    cp_bin_to_hex(header, 8, hdr_prefix);
    snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, hdr_prefix);

    cp_job_mine_begin(job_key);

    const char* tmp = g_dev_dims ? "pp_dev" : "pp_prod";
    char hdr_path[512], proof_path[512];
#ifdef _WIN32
    snprintf(hdr_path, sizeof(hdr_path), "%s\\%s_header.bin", g_workdir, tmp);
    snprintf(proof_path, sizeof(proof_path), "%s\\%s_proof.b64", g_workdir, tmp);
#else
    snprintf(hdr_path, sizeof(hdr_path), "%s/%s_header.bin", g_workdir, tmp);
    snprintf(proof_path, sizeof(proof_path), "%s/%s_proof.b64", g_workdir, tmp);
#endif
    cp_path_abs(hdr_path, sizeof(hdr_path));
    cp_path_abs(proof_path, sizeof(proof_path));
    cp_path_to_posix(hdr_path);
    cp_path_to_posix(proof_path);

    double t0 = cp_now_sec();
    size_t szAp = (size_t)g_m_active * K_DIM;
    size_t szBpT = (size_t)g_n_active * K_DIM;
    int8_t* h_A_scan = NULL;
    int8_t* h_B_scan = NULL;
    uint8_t ab_seed[128];
    uint8_t a_key[32];
    uint8_t job_key_bytes[32];
    uint8_t b_seed[32];
    int t_rows = -1;
    int t_cols = -1;
    int found = 0;
    int bn = 0;
    int ab_len = 0;
    int tiles_per_attempt = 0;
    uint64_t nonce = 0;
    uint64_t attempts = 0;
    uint64_t tiles_scanned_total = 0;
    double elapsed = 0.0;
    double hs = 0.0;
    double last_report = 0.0;
    char* b64 = NULL;

    FILE* hf = fopen(hdr_path, "wb");
    if(!hf){
        perror("header tmp");
        rc = CP_JOB_NONE;
        goto job_done;
    }
    fwrite(header, 1, (size_t)hlen, hf);
    fclose(hf);

    h_A_scan = NULL;
    h_B_scan = NULL;
    const int host_matrices = (g_cpu_matrix_gen || cp_worker_prefers_host_matrices())
                           && !cp_worker_worker_handles_matrix_prep();
    if(host_matrices){
        h_A_scan = (int8_t*)malloc(szAp);
        h_B_scan = (int8_t*)malloc(szBpT);
        if(!h_A_scan || !h_B_scan){
            fprintf(stderr,"OOM scan buffers\n");
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }
    }

    pearl_job_key(header, hlen, job_key_bytes);
    if(cp_worker_worker_handles_matrix_prep()){
        memset(h_BpT_global, 0, szBpT);
    }
    cp_worker_begin_job(job_key_bytes, g_m_active, g_n_active);
    tiles_per_attempt = cp_pp_num_row_parts(g_m_active, cp_worker_uses_contiguous_tiles())
                      * cp_pp_num_col_parts(g_n_active, cp_worker_uses_contiguous_tiles());
    last_report = cp_now_sec();

    for(;;){
        if(cp_job_should_cancel()){
            printf("[plain] job cancelled\n"); fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }
        if(g_max_nonce > 0 && nonce >= (uint64_t)g_max_nonce){
            printf("[plain] stopped after max_nonce=%d\n", g_max_nonce); fflush(stdout);
            rc = CP_JOB_NONE;
            goto job_done;
        }

        ab_len = pearl_effective_seed(header, hlen, nonce, ab_seed, (int)sizeof(ab_seed));
        if(ab_len < 0){
            fprintf(stderr, "[plain] effective_seed failed nonce=%llu\n",
                    (unsigned long long)nonce);
            rc = CP_JOB_NONE;
            goto job_done;
        }

        if(host_matrices){
            if(nonce < 3 || nonce % 16 == 0){
                printf("[gen] nonce=%llu: host A/B + noise (%s)...\n",
                       (unsigned long long)nonce, cp_worker_backend_name());
                fflush(stdout);
            }

            if(pearl_generate_ab(ab_seed, ab_len, g_m_active, g_n_active, K_DIM,
                                h_Ap_global, h_BpT_global) != 0){
                printf("[plain] job cancelled during A,B generation\n"); fflush(stdout);
                rc = CP_JOB_CANCELLED;
                goto job_done;
            }

            pearl_commitment_seeds(job_key_bytes, h_Ap_global, h_BpT_global,
                                   g_m_active, g_n_active, K_DIM, b_seed, a_key);

            if(pearl_build_noisy_matrices(g_m_active, g_n_active, K_DIM, R_RANK,
                                          b_seed, a_key, h_Ap_global, h_BpT_global,
                                          h_A_scan, h_B_scan) != 0){
                if(cp_job_should_cancel()){
                    printf("[plain] job cancelled during noise fusion\n"); fflush(stdout);
                    rc = CP_JOB_CANCELLED;
                } else {
                    printf("[gen] noisy matrix build failed\n"); fflush(stdout);
                }
                goto job_done;
            }
        } else if(nonce < 3 || nonce % 16 == 0){
            if(cp_worker_worker_handles_matrix_prep()){
                printf("[gen] nonce=%llu: zero-B random A + A-noise (%s)...\n",
                       (unsigned long long)nonce, cp_worker_backend_name());
            } else {
                printf("[gen] nonce=%llu: device matrix gen + noise...\n",
                       (unsigned long long)nonce);
            }
            fflush(stdout);
        }

        uint64_t scan_tiles = 0;
        const int worker_cpu_prep =
                cp_worker_worker_handles_matrix_prep() ? g_cpu_matrix_gen : host_matrices;
        found = cp_worker_mine_attempt(
            ab_seed, ab_len, job_key_bytes, pool_tgt,
            g_m_active, g_n_active,
            worker_cpu_prep,
            host_matrices ? h_A_scan : NULL,
            host_matrices ? h_B_scan : NULL,
            host_matrices ? a_key : NULL,
            h_Ap_global,
            h_BpT_global,
            &t_rows, &t_cols, &scan_tiles);
        tiles_scanned_total += scan_tiles;
        attempts++;

        if(found < 0){
            printf("[plain] job cancelled during %s scan\n", cp_worker_backend_name());
            fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }
        if(found == 0){
            double now = cp_now_sec();
            if(now - last_report >= 10.0){
                double sec = now - t0;
                if(sec < 1e-3) sec = 1e-3;
                char mac_buf[32];
                cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(tiles_scanned_total, sec),
                                   mac_buf, sizeof(mac_buf));
                printf("[plain] nonce=%llu attempts=%llu (%.2f/s) %s no share yet\n",
                       (unsigned long long)nonce,
                       (unsigned long long)attempts,
                       (double)attempts / sec, mac_buf);
                fflush(stdout);
                last_report = now;
            }
            nonce++;
            continue;
        }

        if(cp_job_should_cancel()){
            printf("[plain] job cancelled after %s hit (stale)\n",
                   cp_worker_backend_name());
            fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }

        printf("[plain] %s hit nonce=%llu t_rows=%d t_cols=%d - building proof...\n",
               cp_worker_backend_name(),
               (unsigned long long)nonce, t_rows, t_cols);
        fflush(stdout);

        free(b64);
        b64 = NULL;
        b64 = (char*)malloc(PLAIN_PROOF_B64_MAX);
        if(!b64){
            fprintf(stderr, "[plain] OOM proof b64 buffer\n");
            nonce++;
            continue;
        }

        {
            int tile_layout = cp_worker_default_tile_layout();
            if(g_cutlass_fused)
                tile_layout = CP_TILE_LAYOUT_CUTLASS;

            const uint8_t* mining_cfg = PEARL_SCATTERED_CONFIG;
            if(tile_layout == CP_TILE_LAYOUT_CUTLASS)
                mining_cfg = PEARL_CUTLASS_CONFIG;
            else if(tile_layout == CP_TILE_LAYOUT_CONTIGUOUS)
                mining_cfg = PEARL_CONTIGUOUS_CONFIG;
            char errbuf[512];
            int prc = cp_proof_build(
                header, (size_t)hlen,
                mining_cfg, 52,
                h_Ap_global, h_BpT_global,
                g_m_active, g_n_active, K_DIM, R_RANK,
                t_rows, t_cols, tile_layout,
                b64, PLAIN_PROOF_B64_MAX,
                errbuf, sizeof(errbuf));
            if(prc != 0){
                printf("[plain] proof build failed (nonce=%llu): %s\n",
                       (unsigned long long)nonce, errbuf[0] ? errbuf : "unknown");
                fflush(stdout);
                free(b64);
                b64 = NULL;
                nonce++;
                continue;
            }
        }

        bn = (int)strlen(b64);
        if(bn < 32){
            printf("[plain] proof too short (%d), continuing job\n", bn);
            fflush(stdout);
            free(b64);
            b64 = NULL;
            nonce++;
            continue;
        }

        if(g_dry_run || g_plain_verify){
            FILE* pf = fopen(proof_path, "wb");
            if(pf){
                fwrite(b64, 1, (size_t)bn, pf);
                fclose(pf);
            }
        }

        if(cp_job_should_cancel()){
            printf("[plain] job cancelled before verify/submit\n"); fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }

        if(g_plain_verify && target_hex[0]){
            if(cp_verify_proof_file(hdr_path, target_hex, proof_path) != 0){
                printf("[plain] verify failed (nonce=%llu), continuing job\n",
                       (unsigned long long)nonce);
                fflush(stdout);
                free(b64);
                b64 = NULL;
                nonce++;
                continue;
            }
            printf("[plain] verify OK\n"); fflush(stdout);
        }

        elapsed = cp_now_sec() - t0;
        if(elapsed < 1e-3) elapsed = 1e-3;
        hs = cp_pp_mac_rate_from_tiles(tiles_scanned_total, elapsed);
        {
            char mac_buf[32];
            cp_pp_fmt_mac_rate(hs, mac_buf, sizeof(mac_buf));
            printf("[plain] proof ready (%d chars) nonce=%llu attempts=%llu elapsed=%.2fs %s (hs=%.0f)\n",
                   bn, (unsigned long long)nonce, (unsigned long long)attempts,
                   elapsed, mac_buf, hs);
        }
        fflush(stdout);

        if(g_dry_run){
            printf("[plain] dry-run: proof saved to %s, continuing job\n", proof_path);
            fflush(stdout);
            free(b64);
            b64 = NULL;
            if(cp_job_should_cancel()){
                rc = CP_JOB_CANCELLED;
                goto job_done;
            }
            nonce++;
            continue;
        }

        if(!cp_pool_send_plain_proof_submit(sock, (*msg_id)++, job_id, b64, hs)){
            printf("[plain] submit failed (nonce=%llu), continuing job\n",
                   (unsigned long long)nonce);
            fflush(stdout);
            free(b64);
            b64 = NULL;
            nonce++;
            continue;
        }
        cp_pool_set_submit_inflight(1);
        printf("[net] plain_proof submit sent\n");
        cp_pool_log_share_submit_outcome();
        free(b64);
        b64 = NULL;

        if(cp_job_should_cancel()){
            printf("[plain] job cancelled after submit (new notify)\n"); fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }

        nonce++;
    }

job_done:
    free(h_A_scan);
    free(h_B_scan);
    free(b64);
    cp_job_mine_end();
    if(cp_pool_conn_lost()) return CP_JOB_CANCELLED;
    return rc;
}
