#include "cp_qpow_mine.h"
#include "cp_config.h"
#include "cp_fee.h"
#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "cp_qpow_pool.h"
#include "cp_share_queue.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"
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
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
#include "cp_wgpu.h"
#include "cp_wgpu_worker.h"
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
#include "cp_qpow_opencl_worker.h"
#endif
/* Fee reconnect quantum: ~40s at 0.25 MH/s per thread. */
static const uint64_t k_qpow_fee_hashes_per_unit = 10000000ull;
static const uint64_t k_search_chunk = 8192ull;
static const uint64_t k_gpu_search_chunk = 1000000ull;
static std::atomic<int> g_qpow_mock_outcome{CP_SHARE_OUTCOME_NONE};

/* (hi:lo) / d → quotient; remainder via rem_out. Assumes d != 0. */
static uint64_t u128_div_u64(uint64_t hi, uint64_t lo, uint64_t d, uint64_t* rem_out)
{
#if defined(__SIZEOF_INT128__)
    unsigned __int128 v = ((unsigned __int128)hi << 64) | lo;
    *rem_out = (uint64_t)(v % d);
    return (uint64_t)(v / d);
#elif defined(_MSC_VER) && defined(_M_X64)
    /* Returns quotient; writes remainder to *rem_out. */
    return _udiv128(hi, lo, d, rem_out);
#else
    /* Portable restoring division for the uncommon non-x64 MSVC / exotic hosts. */
    uint64_t q = 0;
    uint64_t r = hi;
    for(int i = 0; i < 64; i++){
        const uint64_t rb = r >> 63;
        r = (r << 1) | (lo >> 63);
        lo <<= 1;
        q <<= 1;
        if(rb || r >= d){
            r -= d;
            q |= 1ull;
        }
    }
    *rem_out = r;
    return q;
#endif
}

/* Bitcoin-style U512: target = (2^512 - 1) / difficulty (big-endian). */
static void qpow_target_from_difficulty(uint64_t difficulty, uint8_t target_be[64])
{
    if(difficulty == 0) difficulty = 1;
    uint8_t num[64];
    memset(num, 0xff, 64);
    uint64_t rem = 0;
    for(int w = 0; w < 8; w++){
        uint64_t limb = 0;
        for(int b = 0; b < 8; b++)
            limb = (limb << 8) | num[w * 8 + b];
        uint64_t qlimb = u128_div_u64(rem, limb, difficulty, &rem);
        for(int b = 7; b >= 0; b--){
            target_be[w * 8 + b] = (uint8_t)(qlimb & 0xff);
            qlimb >>= 8;
        }
    }
}

static void build_start_nonce(const CpQpowJob* job, const char* worker_name,
                              uint8_t start[CP_QPOW_NONCE_BYTES])
{
    memset(start, 0, CP_QPOW_NONCE_BYTES);
    int en = job->extranonce_len;
    if(en < 0) en = 0;
    if(en > CP_QPOW_EXTRANONCE_MAX) en = CP_QPOW_EXTRANONCE_MAX;
    if(en > CP_QPOW_NONCE_BYTES) en = CP_QPOW_NONCE_BYTES;
    if(en > 0) memcpy(start, job->extranonce, (size_t)en);
    /* Mock: keep nonce deterministic (zeros + thread stamp only). */
    if(g_mock) return;
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

static int submit_qpow_share(const CpQpowJob* job, int sock, int* msg_id,
                             const uint8_t nonce[CP_QPOW_NONCE_BYTES], int tid);

/* Handle a found nonce. Returns 1 if mining should stop (mock done). */
static int on_qpow_share_found(const CpQpowJob* job, int sock, int* msg_id,
                               const uint8_t nonce[CP_QPOW_NONCE_BYTES], int tid)
{
    if(g_mock){
        /* Another thread already finished mock verify. */
        if(g_qpow_mock_outcome.load(std::memory_order_relaxed) != CP_SHARE_OUTCOME_NONE)
            return 1;
        uint8_t hash[CP_QPOW_TARGET_BYTES];
        qpow::get_nonce_hash(job->mining_hash, nonce, hash);
        char nh[CP_QPOW_NONCE_BYTES * 2 + 1];
        char hh[CP_QPOW_TARGET_BYTES * 2 + 1];
        char th[CP_QPOW_TARGET_BYTES * 2 + 1];
        cp_bin_to_hex(nonce, CP_QPOW_NONCE_BYTES, nh);
        cp_bin_to_hex(hash, CP_QPOW_TARGET_BYTES, hh);
        cp_bin_to_hex(job->target, CP_QPOW_TARGET_BYTES, th);
        printf("[mock] first share nonce=%s (tid=%d)\n", nh, tid);
        printf("[mock] hash=%s\n", hh);
        printf("[mock] target=%s\n", th);
        fflush(stdout);
        if(memcmp(hash, job->target, CP_QPOW_TARGET_BYTES) < 0){
            g_qpow_mock_outcome.store(CP_SHARE_OUTCOME_OK, std::memory_order_relaxed);
            printf("[mock] Poseidon2 verify OK (hash < target)\n");
            fflush(stdout);
        } else {
            g_qpow_mock_outcome.store(CP_SHARE_OUTCOME_VERIFY_FAIL,
                                      std::memory_order_relaxed);
            fprintf(stderr, "[mock] FAIL: hash does not meet target\n");
        }
        return 1;
    }
    submit_qpow_share(job, sock, msg_id, nonce, tid);
    return 0;
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
static void add_be_u64(uint8_t n[CP_QPOW_NONCE_BYTES], uint64_t add)
{
    uint64_t c = add;
    for(int i = CP_QPOW_NONCE_BYTES - 1; i >= 0; --i){
        c += n[i];
        n[i] = (uint8_t)c;
        c >>= 8;
        if(c == 0) break;
    }
}
static uint64_t difficulty_u64(double d)
{
    if(d < 1.0) return 1ull;
    if(d >= (double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)(d + 0.5);
}
static int submit_qpow_share(const CpQpowJob* job, int sock, int* msg_id,
                             const uint8_t nonce[CP_QPOW_NONCE_BYTES], int tid)
{
    if(g_dry_run){
        char nh[CP_QPOW_NONCE_BYTES * 2 + 1];
        cp_bin_to_hex(nonce, CP_QPOW_NONCE_BYTES, nh);
        printf("[qpow] dry-run share nonce=%s (tid=%d)\n", nh, tid);
        fflush(stdout);
        return 1;
    }
    if(sock < 0 || !msg_id) return 0;
    int sid = (*msg_id)++;
    if(!cp_qpow_pool_send_submit(sock, sid, job->job_id, nonce)){
        printf("[qpow] submit send failed\n");
        fflush(stdout);
        return 0;
    }
    cp_pool_log_share_submit_outcome();
    return 1;
}
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
static int mine_job_wgpu(const CpQpowJob* job, int sock, int* msg_id,
                         const char* worker_name)
{
    if(!cp_wgpu_worker_is_ready()){
        fprintf(stderr, "[qpow] wgpu worker not ready\n");
        return CP_JOB_CANCELLED;
    }
    const uint64_t diff = difficulty_u64(job->difficulty);
    uint8_t cur[CP_QPOW_NONCE_BYTES];
    build_start_nonce(job, worker_name, cur);
    /* Single GPU stream: stamp tid=0 for nonce salt consistency with CPU path. */
    stamp_thread_id(cur, job->extranonce_len, 0);
    printf("[qpow] mine job=%s diff=%.0f extranonce_len=%d backend=wgpu%s\n",
           job->job_id, job->difficulty, job->extranonce_len,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);
    cp_job_mine_begin(job->job_key);
    uint64_t total_hashes = 0;
    int stop_rc = CP_JOB_NONE;
    auto t0 = std::chrono::steady_clock::now();
    auto t_log = t0;
    while(stop_rc == CP_JOB_NONE){
        if(cp_job_should_cancel() || cp_pool_conn_lost()){
            stop_rc = CP_JOB_CANCELLED;
            break;
        }
        uint8_t out_nonce[CP_QPOW_NONCE_BYTES];
        uint8_t out_hash[CP_QPOW_TARGET_BYTES];
        uint64_t hashes = 0;
        const int st = cp_wgpu_worker_search(
            job->mining_hash, diff, job->target, cur, k_gpu_search_chunk,
            out_nonce, out_hash, &hashes);
        total_hashes += hashes;
        cp_fee_note_tiles(hashes);
        cp_fee_prepare_matrix();
        if(cp_fee_needs_switch()){
            stop_rc = CP_JOB_FEE_SWITCH;
            break;
        }
        if(st == CP_WGPU_OK_FOUND){
            if(on_qpow_share_found(job, sock, msg_id, out_nonce, 0)){
                stop_rc = CP_JOB_NONE;
                break;
            }
            memcpy(cur, out_nonce, CP_QPOW_NONCE_BYTES);
            qpow::inc_be(cur);
        } else if(st == CP_WGPU_OK_EXHAUSTED){
            add_be_u64(cur, hashes > 0 ? hashes : k_gpu_search_chunk);
        } else if(st == CP_WGPU_DEVICE_LOST){
            fprintf(stderr, "[qpow] wgpu device lost\n");
            stop_rc = CP_JOB_CANCELLED;
            break;
        } else if(st == CP_WGPU_CANCELLED){
            stop_rc = CP_JOB_CANCELLED;
            break;
        } else {
            fprintf(stderr, "[qpow] wgpu search error (%d)\n", st);
            stop_rc = CP_JOB_CANCELLED;
            break;
        }
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_log).count();
        if(elapsed >= 5.0){
            double total_sec = std::chrono::duration<double>(now - t0).count();
            double hs = total_sec > 0 ? (double)total_hashes / total_sec : 0;
            printf("[qpow] %.3f MH/s  (%llu hashes in %.1fs, wgpu)\n",
                   hs / 1e6, (unsigned long long)total_hashes, total_sec);
            fflush(stdout);
            t_log = now;
        }
    }
    if(stop_rc == CP_JOB_NONE && !g_mock) stop_rc = CP_JOB_CANCELLED;
    cp_job_mine_end();
    return stop_rc;
}
#endif

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
static int mine_job_opencl(const CpQpowJob* job, int sock, int* msg_id,
                           const char* worker_name)
{
    if(!cp_qpow_opencl_worker_is_ready()){
        fprintf(stderr, "[qpow] opencl worker not ready\n");
        return CP_JOB_CANCELLED;
    }
    uint8_t cur[CP_QPOW_NONCE_BYTES];
    build_start_nonce(job, worker_name, cur);
    stamp_thread_id(cur, job->extranonce_len, 0);
    printf("[qpow] mine job=%s diff=%.0f extranonce_len=%d backend=opencl%s\n",
           job->job_id, job->difficulty, job->extranonce_len,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);
    cp_job_mine_begin(job->job_key);
    uint64_t total_hashes = 0;
    int stop_rc = CP_JOB_NONE;
    auto t0 = std::chrono::steady_clock::now();
    auto t_log = t0;
    while(stop_rc == CP_JOB_NONE){
        if(cp_job_should_cancel() || cp_pool_conn_lost()){
            stop_rc = CP_JOB_CANCELLED;
            break;
        }
        uint8_t out_nonce[CP_QPOW_NONCE_BYTES];
        uint8_t out_hash[CP_QPOW_TARGET_BYTES];
        uint64_t hashes = 0;
        const int st = cp_qpow_opencl_worker_search(
            job->mining_hash, job->target, cur, k_gpu_search_chunk,
            out_nonce, out_hash, &hashes);
        total_hashes += hashes;
        cp_fee_note_tiles(hashes);
        cp_fee_prepare_matrix();
        if(cp_fee_needs_switch()){
            stop_rc = CP_JOB_FEE_SWITCH;
            break;
        }
        if(st == CP_QPOW_OCL_OK_FOUND){
            if(on_qpow_share_found(job, sock, msg_id, out_nonce, 0)){
                stop_rc = CP_JOB_NONE;
                break;
            }
            memcpy(cur, out_nonce, CP_QPOW_NONCE_BYTES);
            qpow::inc_be(cur);
        } else if(st == CP_QPOW_OCL_OK_EXHAUSTED){
            add_be_u64(cur, hashes > 0 ? hashes : k_gpu_search_chunk);
        } else if(st == CP_QPOW_OCL_CANCELLED){
            stop_rc = CP_JOB_CANCELLED;
            break;
        } else {
            fprintf(stderr, "[qpow] opencl search error (%d)\n", st);
            stop_rc = CP_JOB_CANCELLED;
            break;
        }
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_log).count();
        if(elapsed >= 5.0){
            double total_sec = std::chrono::duration<double>(now - t0).count();
            double hs = total_sec > 0 ? (double)total_hashes / total_sec : 0;
            printf("[qpow] %.3f MH/s  (%llu hashes in %.1fs, opencl)\n",
                   hs / 1e6, (unsigned long long)total_hashes, total_sec);
            fflush(stdout);
            t_log = now;
        }
    }
    if(stop_rc == CP_JOB_NONE && !g_mock) stop_rc = CP_JOB_CANCELLED;
    cp_job_mine_end();
    return stop_rc;
}
#endif

static int mine_job_cpu(const CpQpowJob* job, int sock, int* msg_id,
                        const char* worker_name)
{
    const int nthreads = resolve_thread_count();
    printf("[qpow] mine job=%s diff=%.0f extranonce_len=%d threads=%d%s\n",
           job->job_id, job->difficulty, job->extranonce_len, nthreads,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);
    uint8_t base[CP_QPOW_NONCE_BYTES];
    build_start_nonce(job, worker_name, base);
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<int> stop_rc{CP_JOB_NONE};
    std::atomic<int> running{1};
    std::mutex submit_mx;
    std::mutex fee_mx;
    auto t0 = std::chrono::steady_clock::now();
    auto t_log = t0;
    cp_job_mine_begin(job->job_key);
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
                if(on_qpow_share_found(job, sock, msg_id, r.nonce, tid)){
                    stop_rc.store(CP_JOB_NONE, std::memory_order_relaxed);
                    running.store(0, std::memory_order_relaxed);
                    break;
                }
                memcpy(cur, r.nonce, CP_QPOW_NONCE_BYTES);
                qpow::inc_be(cur);
            } else {
                add_be_u64(cur, k_search_chunk);
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
    if(rc == CP_JOB_NONE && !g_mock) rc = CP_JOB_CANCELLED;
    cp_job_mine_end();
    return rc;
}
int cp_qpow_mine_job(const CpQpowJob* job, int sock, int* msg_id,
                     const char* worker_name)
{
    if(!job) return CP_JOB_CANCELLED;
    cp_fee_set_tiles_per_matrix(k_qpow_fee_hashes_per_unit);
    cp_fee_prepare_matrix();
    if(cp_fee_needs_switch()) return CP_JOB_FEE_SWITCH;
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
    if(cp_worker_backend_id() == CP_BACKEND_WGPU)
        return mine_job_wgpu(job, sock, msg_id, worker_name);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL)
        return mine_job_opencl(job, sock, msg_id, worker_name);
#endif
    return mine_job_cpu(job, sock, msg_id, worker_name);
}

int cp_qpow_mine_mock(const char* worker_name)
{
    CpQpowJob job;
    memset(&job, 0, sizeof(job));
    static const char k_mock_job_id[] = "00000000-0000-4000-8000-000000000001";
    strncpy(job.job_id, k_mock_job_id, sizeof(job.job_id) - 1);
    snprintf(job.job_key, sizeof(job.job_key), "mock:%s", k_mock_job_id);
    /* Deterministic non-zero header (mirrors Pearl mock blob tagging). */
    memcpy(job.mining_hash, "CPMOCK", 6);
    job.mining_hash[6] = 0x01;
    job.difficulty = cp_resolve_mock_diff(1);
    const uint64_t diff_u64 = difficulty_u64(job.difficulty);
    qpow_target_from_difficulty(diff_u64, job.target);
    job.extranonce_len = 0;
    job.clean_jobs = 1;

    char th[CP_QPOW_TARGET_BYTES * 2 + 1];
    cp_bin_to_hex(job.target, CP_QPOW_TARGET_BYTES, th);
    printf("[mock] algo=quantus job_id=%s (offline, no pool)\n", job.job_id);
    printf("[mock] difficulty=%.0f target=%.16s...\n", job.difficulty, th);
    printf("[mock] mining until first share + Poseidon2 verify...\n");
    fflush(stdout);

    g_qpow_mock_outcome.store(CP_SHARE_OUTCOME_NONE, std::memory_order_relaxed);
    const int rc = cp_qpow_mine_job(&job, -1, NULL, worker_name);
    const int outcome = g_qpow_mock_outcome.load(std::memory_order_relaxed);

    if(rc == CP_JOB_CANCELLED && outcome == CP_SHARE_OUTCOME_NONE){
        fprintf(stderr, "[mock] cancelled before share\n");
        return 1;
    }
    if(outcome == CP_SHARE_OUTCOME_OK){
        printf("[mock] PASS: first share mined and verified\n");
        fflush(stdout);
        return 0;
    }
    if(outcome == CP_SHARE_OUTCOME_NONE){
        fprintf(stderr, "[mock] FAIL: no share produced\n");
    } else if(outcome == CP_SHARE_OUTCOME_VERIFY_FAIL){
        fprintf(stderr, "[mock] FAIL: share verify failed\n");
    } else {
        fprintf(stderr, "[mock] FAIL: share outcome=%d\n", outcome);
    }
    return 1;
}

