#include "cp_qpow_mine.h"

#include "cp_config.h"
#include "cp_fee.h"
#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "cp_qpow_pool.h"
#include "cp_state.h"
#include "cp_util.h"

#include "qpow/miner.hpp"

#include <chrono>
#include <cstring>
#include <stdio.h>
#include <string.h>

/* Fee reconnect quantum: ~40s at 0.25 MH/s. */
static const uint64_t k_qpow_fee_hashes_per_unit = 10000000ull;
static const uint64_t k_search_chunk = 8192ull;

static void build_start_nonce(const CpQpowJob* job, const char* worker_name,
                              uint8_t start[CP_QPOW_NONCE_BYTES])
{
    memset(start, 0, CP_QPOW_NONCE_BYTES);
    int en = job->extranonce_len;
    if(en < 0) en = 0;
    if(en > CP_QPOW_EXTRANONCE_MAX) en = CP_QPOW_EXTRANONCE_MAX;
    if(en > CP_QPOW_NONCE_BYTES) en = CP_QPOW_NONCE_BYTES;
    if(en > 0) memcpy(start, job->extranonce, (size_t)en);

    /* Worker-unique salt in the free mid region so multi-process workers diverge. */
    uint8_t salt[32];
    memset(salt, 0, sizeof(salt));
    if(worker_name && worker_name[0]){
        size_t n = strlen(worker_name);
        if(n > sizeof(salt)) n = sizeof(salt);
        memcpy(salt, worker_name, n);
    }
    uint64_t rnd = 0;
    (void)cp_random_u64(&rnd);
    memcpy(salt + 16, &rnd, sizeof(rnd));
    int free_off = en;
    int free_len = CP_QPOW_NONCE_BYTES - free_off;
    if(free_len > 24){
        /* Keep low 8 bytes as the hot counter; salt into [en, en+24). */
        int copy = free_len - 8;
        if(copy > 24) copy = 24;
        if(copy > (int)sizeof(salt)) copy = (int)sizeof(salt);
        memcpy(start + free_off, salt, (size_t)copy);
    }
}

int cp_qpow_mine_job(const CpQpowJob* job, int sock, int* msg_id,
                     const char* worker_name)
{
    if(!job) return CP_JOB_CANCELLED;

    cp_fee_set_tiles_per_matrix(k_qpow_fee_hashes_per_unit);
    cp_fee_prepare_matrix();
    if(cp_fee_needs_switch()) return CP_JOB_FEE_SWITCH;

    cp_job_mine_begin(job->job_key);
    printf("[qpow] mine job=%s diff=%.0f extranonce_len=%d%s\n", job->job_id,
           job->difficulty, job->extranonce_len,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);

    uint8_t cur[CP_QPOW_NONCE_BYTES];
    build_start_nonce(job, worker_name, cur);

    uint64_t total_hashes = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto t_log = t0;
    int rc = CP_JOB_NONE;

    while(1){
        if(cp_job_should_cancel() || cp_pool_conn_lost()){
            rc = CP_JOB_CANCELLED;
            break;
        }
        cp_fee_prepare_matrix();
        if(cp_fee_needs_switch()){
            rc = CP_JOB_FEE_SWITCH;
            break;
        }

        qpow::SearchResult r = qpow::search_range(
            job->mining_hash, cur, k_search_chunk, job->target, qpow::Isa::Auto);
        total_hashes += r.hashes;
        cp_fee_note_tiles(r.hashes);

        if(r.found){
            if(g_dry_run){
                char nh[CP_QPOW_NONCE_BYTES * 2 + 1];
                cp_bin_to_hex(r.nonce, CP_QPOW_NONCE_BYTES, nh);
                printf("[qpow] dry-run share nonce=%s\n", nh);
                fflush(stdout);
            } else if(sock >= 0 && msg_id){
                int sid = (*msg_id)++;
                if(!cp_qpow_pool_send_submit(sock, sid, job->job_id, r.nonce)){
                    printf("[qpow] submit send failed\n");
                    fflush(stdout);
                } else {
                    cp_pool_log_share_submit_outcome();
                }
            }
            /* Continue mining after share (PPLNS). */
            memcpy(cur, r.nonce, CP_QPOW_NONCE_BYTES);
            qpow::inc_be(cur);
        } else {
            for(uint64_t i = 0; i < k_search_chunk; i++) qpow::inc_be(cur);
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration<double>(now - t_log).count();
        if(elapsed >= 5.0){
            double total_sec =
                std::chrono::duration<double>(now - t0).count();
            double hs = total_sec > 0 ? (double)total_hashes / total_sec : 0;
            printf("[qpow] %.3f MH/s  (%llu hashes in %.1fs)\n", hs / 1e6,
                   (unsigned long long)total_hashes, total_sec);
            fflush(stdout);
            t_log = now;
        }
    }

    cp_job_mine_end();
    return rc;
}
