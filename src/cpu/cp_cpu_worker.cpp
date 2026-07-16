#include "cp_cpu_worker.h"

#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_util.h"
#include "gemm/case33_gemm_xor.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct ZeroBCache {
    uint8_t job_key[32]{};
    int m = 0;
    int n = 0;
    int ready = 0;
    uint8_t b_noise_seed[32]{};
    std::vector<int8_t> B_noisy;
} g_zero_b;

/* Single copies: A scan buf; a_pre_/b_pre_ only in separate prepack mode. */
static std::vector<int8_t> g_A_noisy;
static Case33GemmXor g_gemm;
static CpPrepackMode g_prepack_mode = CP_PREPACK_SEPARATE;

static void apply_prepack_mode_to_gemm(void)
{
    switch(g_prepack_mode){
    case CP_PREPACK_REUSE:
        g_gemm.set_prepack_mode(Case33PrepackMode::ReuseScanBuf);
        break;
    case CP_PREPACK_FUSED:
        g_gemm.set_prepack_mode(Case33PrepackMode::Fused);
        break;
    default:
        g_gemm.set_prepack_mode(Case33PrepackMode::Separate);
        break;
    }
}

static const char* prepack_mode_name(CpPrepackMode mode)
{
    switch(mode){
    case CP_PREPACK_REUSE: return "reuse";
    case CP_PREPACK_FUSED: return "fused";
    default: return "separate";
    }
}

static int zero_b_cache_matches(const uint8_t job_key[32], int m, int n)
{
    return g_zero_b.ready && g_zero_b.m == m && g_zero_b.n == n
        && memcmp(g_zero_b.job_key, job_key, 32) == 0;
}

static int zero_b_prepare_job(const uint8_t job_key[32], int m, int n)
{
    const int log_step = (m >= 65536 || n >= 65536) ? 1 : 0;
    double t0 = 0.0, t_step = 0.0;
    if(log_step) t0 = cp_now_sec();

    const size_t szB = (size_t)n * (size_t)K_DIM;
    g_zero_b.B_noisy.resize(szB);
    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;

    g_gemm.reset();
    g_gemm.set_int8_mode(Case32Int8Mode::FastU8S8);
    apply_prepack_mode_to_gemm();

    if(log_step) t_step = cp_now_sec();
    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_zero_b.b_noise_seed);
    if(log_step)
        printf("[gen]   zero-B b_noise_seed done in %.1fs\n", cp_now_sec() - t_step);

    if(g_prepack_mode == CP_PREPACK_FUSED){
        if(!g_gemm.prepare_job_b(m, n, K_DIM, &g_zero_b.B_noisy, NULL,
                                g_zero_b.b_noise_seed, R_RANK)){
            g_zero_b.ready = 0;
            fprintf(stderr, "[cpu] prepare_job_b (fused) failed\n");
            return -2;
        }
    } else {
        if(pearl_build_noisy_b(n, K_DIM, R_RANK, g_zero_b.b_noise_seed,
                              NULL, g_zero_b.B_noisy.data()) != 0){
            g_zero_b.ready = 0;
            return cp_job_should_cancel() ? -1 : -2;
        }

        if(!g_gemm.prepare_job_b(m, n, K_DIM, &g_zero_b.B_noisy, NULL, NULL, R_RANK)){
            g_zero_b.ready = 0;
            fprintf(stderr, "[cpu] prepare_job_b failed\n");
            return -2;
        }
    }

    if(log_step)
        printf("[gen]   zero-B job setup done in %.1fs\n", cp_now_sec() - t0);

    g_zero_b.ready = 1;
    return 0;
}

static int zero_b_prepare_attempt(
    const uint8_t* ab_seed, int ab_seed_len,
    const uint8_t job_key[32],
    int m, int n,
    int8_t* h_A_sig,
    uint8_t a_key_out[32])
{
    (void)ab_seed;
    (void)ab_seed_len;
    if(!h_A_sig){
        fprintf(stderr, "[cpu] zero-B requires h_Ap_global (A_sig buffer)\n");
        return -2;
    }

    if(!zero_b_cache_matches(job_key, m, n)){
        if(zero_b_prepare_job(job_key, m, n) != 0)
            return -1;
    }

    const size_t szA = (size_t)m * (size_t)K_DIM;
    g_A_noisy.resize(szA);

    uint8_t a_rng[32];
    if(cp_random_bytes(a_rng, sizeof(a_rng)) != 0){
        fprintf(stderr, "[cpu] CSPRNG failed for random A\n");
        return -2;
    }

    if(pearl_generate_random_a(a_rng, (int)sizeof(a_rng), m, K_DIM, h_A_sig) != 0)
        return -1;

    pearl_a_noise_seed_from_a(job_key, g_zero_b.b_noise_seed,
                              h_A_sig, m, K_DIM, a_key_out);

    if(g_prepack_mode == CP_PREPACK_FUSED){
        if(!g_gemm.prepare_attempt_a(&g_A_noisy, h_A_sig, a_key_out, R_RANK)){
            fprintf(stderr, "[cpu] prepare_attempt_a (fused) failed\n");
            return -2;
        }
    } else {
        if(pearl_build_noisy_a(m, K_DIM, R_RANK, a_key_out,
                               h_A_sig, g_A_noisy.data()) != 0){
            return cp_job_should_cancel() ? -1 : -2;
        }

        if(!g_gemm.prepare_attempt_a(&g_A_noisy, NULL, NULL, R_RANK)){
            fprintf(stderr, "[cpu] prepare_attempt_a failed\n");
            return -2;
        }
    }

    return 0;
}

} /* namespace */

extern "C" int cp_cpu_worker_handles_matrix_prep(void)
{
    return 1;
}

extern "C" void cp_cpu_worker_begin_job(const uint8_t job_key[32], int m, int n)
{
    g_zero_b.ready = 0;
    if(zero_b_prepare_job(job_key, m, n) == 0){
        const char* b_desc = "noisy B + b_pre";
        if(g_prepack_mode == CP_PREPACK_FUSED || g_prepack_mode == CP_PREPACK_REUSE)
            b_desc = "B scan buf";
        printf("[cpu] zero-B: cached %s for job (signal B^T = 0, prepack=%s)\n",
               b_desc, prepack_mode_name(g_prepack_mode));
        fflush(stdout);
    }
}

extern "C" void cp_cpu_worker_set_prepack_mode(CpPrepackMode mode)
{
    if(mode < CP_PREPACK_SEPARATE || mode > CP_PREPACK_FUSED)
        mode = CP_PREPACK_SEPARATE;
    g_prepack_mode = mode;
    apply_prepack_mode_to_gemm();
}

extern "C" void cp_cpu_worker_set_inplace_prepack(int on)
{
    cp_cpu_worker_set_prepack_mode(on ? CP_PREPACK_REUSE : CP_PREPACK_SEPARATE);
}

extern "C" void cp_cpu_worker_init(void)
{
    g_gemm.set_int8_mode(Case32Int8Mode::FastU8S8);
    apply_prepack_mode_to_gemm();
    printf("[cpu] Case 3.3 fused GEMM+XOR worker (contiguous 8x16 tiles, zero-B)\n");
    if(g_prepack_mode != CP_PREPACK_SEPARATE)
        printf("[cpu] prepack mode: %s\n", prepack_mode_name(g_prepack_mode));
    fflush(stdout);
}

extern "C" void cp_cpu_worker_shutdown(void)
{
    g_zero_b.ready = 0;
    g_zero_b.B_noisy.clear();
    g_A_noisy.clear();
    g_gemm.reset();
}

extern "C" int cp_cpu_worker_mine_attempt(
    const uint8_t* ab_seed, int ab_seed_len,
    const uint8_t job_key[32],
    const uint32_t pool_tgt[8],
    int m, int n,
    int cpu_matrices,
    const int8_t* h_A_noisy, const int8_t* h_B_noisy,
    const uint8_t* a_key,
    int8_t* h_A_sig, int8_t* h_Bt_sig,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned)
{
    (void)cpu_matrices;
    (void)h_Bt_sig;

    const double attempt_t0 = cp_now_sec();

    if(out_tiles_scanned) *out_tiles_scanned = 0;
    if(out_t_rows) *out_t_rows = -1;
    if(out_t_cols) *out_t_cols = -1;

    uint8_t a_key_local[32];
    const uint8_t* scan_key = a_key;

    if(!h_A_noisy || !h_B_noisy || !scan_key){
        if(!h_A_sig){
            fprintf(stderr, "[cpu] mine_attempt requires h_Ap_global\n");
            return -2;
        }
        if(zero_b_prepare_attempt(ab_seed, ab_seed_len, job_key, m, n,
                                  h_A_sig, a_key_local) != 0){
            return cp_job_should_cancel() ? -1 : 0;
        }
        scan_key = a_key_local;
    } else {
        g_gemm.reset();
        g_gemm.set_int8_mode(Case32Int8Mode::FastU8S8);
        apply_prepack_mode_to_gemm();
        if(!g_gemm.init(m, n, K_DIM, h_A_noisy, h_B_noisy)){
            fprintf(stderr, "[cpu] Case3.3 init failed (host matrices)\n");
            return -1;
        }
    }

    if(m % Case33GemmXor::kMacroM != 0 || n % Case33GemmXor::kMacroN != 0){
        fprintf(stderr, "[cpu] m,n must be multiples of %dx%d (got %dx%d)\n",
                Case33GemmXor::kMacroM, Case33GemmXor::kMacroN, m, n);
        return -1;
    }

    if(!g_gemm.available()){
        fprintf(stderr, "[cpu] GEMM not ready (backend=%s)\n", g_gemm.backend());
        return -1;
    }

    uint32_t bound[8];
    cp_scale_jackpot_target(pool_tgt, bound);
    uint32_t a_key8[8];
    memcpy(a_key8, scan_key, 32);

    const int row_parts = cp_pp_num_row_parts(m, 1);
    const int col_parts = cp_pp_num_col_parts(n, 1);
    const int total_tiles = row_parts * col_parts;
    const double prep_sec = cp_now_sec() - attempt_t0;
    const double scan_t0 = cp_now_sec();

    printf("[cpu] plain_proof scan %dx%d hash tiles, difficulty scaled by %d\n",
           row_parts, col_parts, PP_HASH_H * PP_HASH_W * K_DIM);
    printf("[cpu] GEMM %s\n", g_gemm.backend());
    fflush(stdout);

    std::atomic<int> found{0};
    std::atomic<uint64_t> tiles{0};
    std::atomic<int> hit_rows{-1};
    std::atomic<int> hit_cols{-1};
    std::atomic<bool> scan_done{false};
    std::mutex progress_mu;
    std::condition_variable progress_cv;
    constexpr auto kProgressInterval = std::chrono::seconds(2);

    std::thread progress_thread([&]() {
        uint64_t last_tiles = 0;
        double last_report = scan_t0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(progress_mu);
                if (progress_cv.wait_for(lock, kProgressInterval, [&] {
                        return scan_done.load(std::memory_order_relaxed);
                    })) {
                    break;
                }
            }
            if (found.load(std::memory_order_relaxed)) {
                break;
            }

            const uint64_t cur = tiles.load(std::memory_order_relaxed);
            const double now = cp_now_sec();
            if (cur == last_tiles && now - last_report < 1.0) {
                continue;
            }
            if (now - last_report < 1.0 && cur - last_tiles < 4096) {
                continue;
            }

            double scan_sec = now - scan_t0;
            if (scan_sec < 1e-9) {
                scan_sec = 1e-9;
            }
            int row_done = col_parts > 0
                ? (int)((cur + (uint64_t)col_parts - 1) / (uint64_t)col_parts)
                : 0;
            if (row_done > row_parts) {
                row_done = row_parts;
            }

            char mac_buf[32];
            cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(cur, scan_sec),
                               mac_buf, sizeof(mac_buf));
            printf("[cpu] plain_proof progress: row parts %d/%d tiles %llu/%d (%.1f%%) %s\n",
                   row_done, row_parts,
                   (unsigned long long)cur, total_tiles,
                   total_tiles > 0 ? 100.0 * (double)cur / (double)total_tiles : 0.0,
                   mac_buf);
            fflush(stdout);
            last_tiles = cur;
            last_report = now;
        }
    });

    const bool ok = g_gemm.scan_tiles(
        [&](const uint32_t* milestone_xor, int t_rows, int t_cols) -> bool {
            tiles.fetch_add(1, std::memory_order_relaxed);
            if(found.load(std::memory_order_relaxed)) return false;
            if(cp_jackpot::tile_beats_target(milestone_xor, Case33GemmXor::kNumMilestones,
                                             a_key8, bound)){
                int expected = 0;
                if(found.compare_exchange_strong(expected, 1)){
                    hit_rows.store(t_rows, std::memory_order_relaxed);
                    hit_cols.store(t_cols, std::memory_order_relaxed);
                }
                return false;
            }
            return true;
        },
        []() -> bool { return cp_job_should_cancel() != 0; });

    const double scan_sec = cp_now_sec() - scan_t0;
    {
        std::lock_guard<std::mutex> lock(progress_mu);
        scan_done.store(true, std::memory_order_relaxed);
    }
    progress_cv.notify_all();
    if (progress_thread.joinable()) {
        progress_thread.join();
    }

    const uint64_t tiles_done = tiles.load();
    if (out_tiles_scanned) {
        *out_tiles_scanned = tiles_done;
    }

    if(cp_job_should_cancel()){
        cp_log_attempt_timing("cpu", prep_sec, scan_sec, tiles_done, 0.0);
        return -1;
    }
    if(!ok && found.load() == 0){
        cp_log_attempt_timing("cpu", prep_sec, scan_sec, tiles_done, 0.0);
        return -1;
    }

    if(found.load()){
        const int tr = hit_rows.load();
        const int tc = hit_cols.load();
        printf("[cpu] plain_proof SHARE t_rows=%d t_cols=%d\n", tr, tc);
        fflush(stdout);
        if(out_t_rows) *out_t_rows = tr;
        if(out_t_cols) *out_t_cols = tc;
        cp_log_attempt_timing("cpu", prep_sec, scan_sec, tiles_done, 0.0);
        return 1;
    }

    cp_log_attempt_timing("cpu", prep_sec, scan_sec, tiles_done, 0.0);
    return 0;
}
