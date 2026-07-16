#include "cp_mine.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_pool.h"
#include "cp_share_queue.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static CpShareQueue *g_share_queue = NULL;

void cp_mine_init_host_buffers(void)
{
    size_t szAp = (size_t)g_m_active * K_DIM;
    size_t szBpT = (size_t)g_n_active * K_DIM;
    h_Ap_global = (int8_t *)malloc(szAp);
    h_BpT_global = (int8_t *)malloc(szBpT);
    if (!h_Ap_global || !h_BpT_global) {
        fprintf(stderr, "OOM host matrices\n");
        exit(1);
    }
    if (!g_share_queue) {
        /* Depth 1: single host A/B handoff slot (no snapshot memcpy). */
        g_share_queue = cp_share_queue_create(1);
        if (!g_share_queue) {
            fprintf(stderr, "OOM share proof queue\n");
            exit(1);
        }
    }
}

void cp_mine_free_host_buffers(void)
{
    if (g_share_queue) {
        cp_share_queue_destroy(g_share_queue);
        g_share_queue = NULL;
    }
    free(h_Ap_global);
    free(h_BpT_global);
    h_Ap_global = NULL;
    h_BpT_global = NULL;
}

int cp_mine_job(const uint8_t *header, int hlen, const char *job_id, const char *target_hex,
                const uint32_t pool_tgt[8], int sock, int *msg_id) {
    int rc = CP_JOB_NONE;
    char job_key[320];
    char hdr_prefix[20];
    cp_bin_to_hex(header, 8, hdr_prefix);
    snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, hdr_prefix);

    cp_job_mine_begin(job_key);

    const char *tmp = g_dev_dims ? "pp_dev" : "pp_prod";
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
    const size_t szAp = static_cast<size_t>(g_m_active) * K_DIM;
    const size_t szBpT = static_cast<size_t>(g_n_active) * K_DIM;
    int8_t *h_A_scan = NULL;
    int8_t *h_B_scan = NULL;
    uint8_t ab_seed[128];
    uint8_t a_key[32];
    uint8_t job_key_bytes[32];
    uint8_t b_seed[32];
    int t_rows = -1;
    int t_cols = -1;
    int found = 0;
    int ab_len = 0;
    int tiles_per_attempt = 0;
    uint64_t nonce = 0;
    uint64_t attempts = 0;
    uint64_t tiles_scanned_total = 0;
    uint64_t tiles_at_prev_share = 0;
    double last_report = 0.0;
    double t_prev_share = 0.0;

    FILE *hf = fopen(hdr_path, "wb");
    if (!hf) {
        perror("header tmp");
        rc = CP_JOB_NONE;
        goto job_done;
    }
    fwrite(header, 1, (size_t)hlen, hf);
    fclose(hf);

    const int host_matrices = (g_cpu_matrix_gen || cp_worker_prefers_host_matrices()) &&
                              !cp_worker_worker_handles_matrix_prep();
    if (host_matrices) {
        h_A_scan = (int8_t *)malloc(szAp);
        h_B_scan = (int8_t *)malloc(szBpT);
        if (!h_A_scan || !h_B_scan) {
            fprintf(stderr, "OOM scan buffers\n");
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }
    }

    pearl_job_key(header, hlen, job_key_bytes);
    if (cp_worker_worker_handles_matrix_prep()) {
        memset(h_BpT_global, 0, szBpT);
    }
    cp_worker_begin_job(job_key_bytes, g_m_active, g_n_active);
    tiles_per_attempt = cp_pp_num_row_parts(g_m_active, cp_worker_uses_contiguous_tiles()) *
                        cp_pp_num_col_parts(g_n_active, cp_worker_uses_contiguous_tiles());
    last_report = cp_now_sec();
    t_prev_share = t0;
    tiles_at_prev_share = 0;

    if (g_share_queue) {
        const CpShareJobCtx share_ctx = {sock, msg_id, g_m_active, g_n_active, hdr_path,
                                         proof_path};
        cp_share_queue_begin_job(g_share_queue, &share_ctx, job_key);
    }

    const int zero_b_gpu = cp_worker_worker_handles_matrix_prep() && !g_cpu_matrix_gen;
    const int handoff_bt = host_matrices || !zero_b_gpu;
    /* Defer reclaim only when host A is unused between attempts (OpenCL/CUDA GPU prep).
     * CPU writes h_Ap_global every nonce and must reclaim before the next attempt. */
    const int defer_host_reclaim =
            zero_b_gpu && !cp_worker_prefers_host_matrices();

    for (;;) {
        if (cp_job_should_cancel()) {
            printf("[plain] job cancelled\n");
            fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }
        if (g_max_nonce > 0 && nonce >= (uint64_t)g_max_nonce) {
            printf("[plain] stopped after max_nonce=%d\n", g_max_nonce);
            fflush(stdout);
            rc = CP_JOB_NONE;
            goto job_done;
        }

        /* Reclaim host matrices when this attempt will write them (CPU / host-gen).
         * GPU prep defers reclaim until a share hit so scanning can continue while
         * proof holds the single A buffer. */
        if (g_share_queue && !defer_host_reclaim) {
            cp_share_queue_reclaim_matrices(g_share_queue, &h_Ap_global,
                                           handoff_bt ? &h_BpT_global : NULL);
            if (!h_Ap_global || (handoff_bt && !h_BpT_global)) {
                fprintf(stderr, "[plain] host matrix buffers missing after reclaim\n");
                rc = CP_JOB_NONE;
                goto job_done;
            }
        }

        ab_len = pearl_effective_seed(header, hlen, nonce, ab_seed, (int)sizeof(ab_seed));
        if (ab_len < 0) {
            fprintf(stderr, "[plain] effective_seed failed nonce=%llu\n",
                    (unsigned long long)nonce);
            rc = CP_JOB_NONE;
            goto job_done;
        }

        if (host_matrices) {
            if (nonce < 3 || nonce % 16 == 0) {
                printf("[gen] nonce=%llu: host A/B + noise (%s)...\n",
                       (unsigned long long)nonce, cp_worker_backend_name());
                fflush(stdout);
            }

            if (pearl_generate_ab(ab_seed, ab_len, g_m_active, g_n_active, K_DIM, h_Ap_global,
                                  h_BpT_global) != 0) {
                printf("[plain] job cancelled during A,B generation\n");
                fflush(stdout);
                rc = CP_JOB_CANCELLED;
                goto job_done;
            }

            pearl_commitment_seeds(job_key_bytes, h_Ap_global, h_BpT_global, g_m_active, g_n_active,
                                   K_DIM, b_seed, a_key);

            if (pearl_build_noisy_matrices(g_m_active, g_n_active, K_DIM, R_RANK, b_seed, a_key,
                                           h_Ap_global, h_BpT_global, h_A_scan, h_B_scan) != 0) {
                if (cp_job_should_cancel()) {
                    printf("[plain] job cancelled during noise fusion\n");
                    fflush(stdout);
                    rc = CP_JOB_CANCELLED;
                } else {
                    printf("[gen] noisy matrix build failed\n");
                    fflush(stdout);
                }
                goto job_done;
            }
        } else if (nonce < 3 || nonce % 16 == 0) {
            if (cp_worker_worker_handles_matrix_prep()) {
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
        found = cp_worker_mine_attempt(ab_seed, ab_len, job_key_bytes, pool_tgt, g_m_active,
                                       g_n_active, worker_cpu_prep,
                                       host_matrices ? h_A_scan : NULL,
                                       host_matrices ? h_B_scan : NULL, host_matrices ? a_key : NULL,
                                       h_Ap_global, h_BpT_global, &t_rows, &t_cols, &scan_tiles);
        tiles_scanned_total += scan_tiles;
        attempts++;

        if (found < 0) {
            if (cp_job_should_cancel()) {
                printf("[plain] job cancelled during %s scan\n", cp_worker_backend_name());
                rc = CP_JOB_CANCELLED;
            } else {
                fprintf(stderr, "[plain] %s mine_attempt failed (rc=%d)\n",
                        cp_worker_backend_name(), found);
                rc = CP_JOB_NONE;
            }
            fflush(stdout);
            goto job_done;
        }
        if (found == 0) {
            double now = cp_now_sec();
            if (now - last_report >= 10.0) {
                double sec = now - t0;
                if (sec < 1e-3) {
                    sec = 1e-3;
                }
                char mac_buf[32];
                cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(tiles_scanned_total, sec), mac_buf,
                                   sizeof(mac_buf));
                printf("[plain] nonce=%llu attempts=%llu (%.2f/s) %s no share yet\n",
                       (unsigned long long)nonce, (unsigned long long)attempts,
                       (double)attempts / sec, mac_buf);
                fflush(stdout);
                last_report = now;
            }
            nonce++;
            continue;
        }

        if (cp_job_should_cancel()) {
            printf("[plain] job cancelled after %s hit (stale)\n", cp_worker_backend_name());
            fflush(stdout);
            rc = CP_JOB_CANCELLED;
            goto job_done;
        }

        if (!g_share_queue) {
            fprintf(stderr, "[plain] share queue unavailable\n");
            nonce++;
            continue;
        }

        {
            const double now_hit = cp_now_sec();
            double interval_sec = now_hit - t_prev_share;
            if (interval_sec < 1e-3) {
                interval_sec = 1e-3;
            }
            const uint64_t tiles_since_prev = tiles_scanned_total - tiles_at_prev_share;

            if (g_share_queue) {
                cp_share_queue_reclaim_matrices(g_share_queue, &h_Ap_global,
                                               handoff_bt ? &h_BpT_global : NULL);
            }
            if (!h_Ap_global || (handoff_bt && !h_BpT_global)) {
                fprintf(stderr, "[plain] host matrix buffers missing before proof handoff\n");
                nonce++;
                continue;
            }
            if (defer_host_reclaim) {
                if (cp_worker_fetch_share_signals(h_Ap_global,
                                                  handoff_bt ? h_BpT_global : NULL) != 0) {
                    fprintf(stderr, "[plain] failed to fetch signal matrices nonce=%llu\n",
                            (unsigned long long)nonce);
                    nonce++;
                    continue;
                }
            }
            const CpShareHit hit = {nonce,
                                    t_rows,
                                    t_cols,
                                    tiles_since_prev,
                                    interval_sec,
                                    handoff_bt};
            if (cp_share_queue_enqueue_hit(g_share_queue, &hit, header, hlen, job_id, target_hex,
                                           &h_Ap_global, szAp, &h_BpT_global, szBpT) != 0) {
                fprintf(stderr, "[plain] failed to enqueue share nonce=%llu\n",
                        (unsigned long long)nonce);
            } else {
                tiles_at_prev_share = tiles_scanned_total;
                t_prev_share = now_hit;
            }
        }

        nonce++;
    }

job_done:
    if (g_share_queue) {
        cp_share_queue_end_job(g_share_queue);
        cp_share_queue_reclaim_matrices(g_share_queue, &h_Ap_global, &h_BpT_global);
    }
    free(h_A_scan);
    free(h_B_scan);
    cp_job_mine_end();
    if (cp_pool_conn_lost()) {
        return CP_JOB_CANCELLED;
    }
    return rc;
}
