#include "cp_qpow_mine.h"

#include "cp_config.h"
#include "cp_fee.h"
#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "cp_qpow_pool.h"
#include "cp_state.h"
#include "cp_util.h"

#include "qpow/miner.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Fee reconnect quantum: ~40s at 0.25 MH/s per thread. */
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

    /* Reserve 4 bytes after extranonce for OpenMP thread id (high half). */
    int salt_off = en + 4;
    if(salt_off > 32) salt_off = 32;

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

    int free_len = CP_QPOW_NONCE_BYTES - salt_off;
    if(free_len > 24){
        int copy = free_len - 8; /* keep low 8B as hot counter */
        if(copy > 24) copy = 24;
        if(copy > (int)sizeof(salt)) copy = (int)sizeof(salt);
        if(copy > 0) memcpy(start + salt_off, salt, (size_t)copy);
    }
}

static void stamp_thread_id(uint8_t nonce[CP_QPOW_NONCE_BYTES], int extranonce_len,
                            int tid)
{
    int off = extranonce_len;
    if(off < 0) off = 0;
    if(off + 4 > 32) return; /* stay in high half so midstate is stable per thread */
    nonce[off + 0] = (uint8_t)((tid >> 24) & 0xff);
    nonce[off + 1] = (uint8_t)((tid >> 16) & 0xff);
    nonce[off + 2] = (uint8_t)((tid >> 8) & 0xff);
    nonce[off + 3] = (uint8_t)(tid & 0xff);
}

static int resolve_thread_count(void)
{
    int n = g_qpow_threads;
#ifdef _OPENMP
    if(n <= 0) n = omp_get_max_threads();
    if(n < 1) n = 1;
#else
    (void)n;
    n = 1;
#endif
    return n;
}

int cp_qpow_mine_job(const CpQpowJob* job, int sock, int* msg_id,
                     const char* worker_name)
{
    if(!job) return CP_JOB_CANCELLED;

    cp_fee_set_tiles_per_matrix(k_qpow_fee_hashes_per_unit);
    cp_fee_prepare_matrix();
    if(cp_fee_needs_switch()) return CP_JOB_FEE_SWITCH;

    const int nthreads = resolve_thread_count();

    cp_job_mine_begin(job->job_key);
    printf("[qpow] mine job=%s diff=%.0f extranonce_len=%d threads=%d%s\n",
           job->job_id, job->difficulty, job->extranonce_len, nthreads,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);

    uint8_t base[CP_QPOW_NONCE_BYTES];
    build_start_nonce(job, worker_name, base);

    std::atomic<uint64_t> total_hashes{0};
    std::atomic<int> stop_rc{CP_JOB_NONE}; /* CP_JOB_NONE while running */
    std::atomic<int> running{1};
    std::mutex submit_mx;
    std::mutex fee_mx;

    auto t0 = std::chrono::steady_clock::now();
    auto t_log = t0;

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        uint8_t cur[CP_QPOW_NONCE_BYTES];
        memcpy(cur, base, CP_QPOW_NONCE_BYTES);
        stamp_thread_id(cur, job->extranonce_len, tid);

        while(running.load(std::memory_order_relaxed)){
            if(cp_job_should_cancel() || cp_pool_conn_lost()){
                stop_rc.store(CP_JOB_CANCELLED, std::memory_order_relaxed);
                running.store(0, std::memory_order_relaxed);
                break;
            }

            qpow::SearchResult r = qpow::search_range(
                job->mining_hash, cur, k_search_chunk, job->target, qpow::Isa::Auto);
            total_hashes.fetch_add(r.hashes, std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lk(fee_mx);
                cp_fee_note_tiles(r.hashes);
                cp_fee_prepare_matrix();
                if(cp_fee_needs_switch()){
                    stop_rc.store(CP_JOB_FEE_SWITCH, std::memory_order_relaxed);
                    running.store(0, std::memory_order_relaxed);
                }
            }
            if(!running.load(std::memory_order_relaxed)) break;

            if(r.found){
                std::lock_guard<std::mutex> lk(submit_mx);
                if(g_dry_run){
                    char nh[CP_QPOW_NONCE_BYTES * 2 + 1];
                    cp_bin_to_hex(r.nonce, CP_QPOW_NONCE_BYTES, nh);
                    printf("[qpow] dry-run share nonce=%s (tid=%d)\n", nh, tid);
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
                memcpy(cur, r.nonce, CP_QPOW_NONCE_BYTES);
                qpow::inc_be(cur);
            } else {
                for(uint64_t i = 0; i < k_search_chunk; i++) qpow::inc_be(cur);
            }

            if(tid == 0){
                auto now = std::chrono::steady_clock::now();
                double elapsed =
                    std::chrono::duration<double>(now - t_log).count();
                if(elapsed >= 5.0){
                    double total_sec =
                        std::chrono::duration<double>(now - t0).count();
                    uint64_t th = total_hashes.load(std::memory_order_relaxed);
                    double hs = total_sec > 0 ? (double)th / total_sec : 0;
                    printf("[qpow] %.3f MH/s  (%llu hashes in %.1fs, %d threads)\n",
                           hs / 1e6, (unsigned long long)th, total_sec, nthreads);
                    fflush(stdout);
                    t_log = now;
                }
            }
        }
    }

    int rc = stop_rc.load(std::memory_order_relaxed);
    if(rc == CP_JOB_NONE) rc = CP_JOB_CANCELLED;

    cp_job_mine_end();
    return rc;
}
