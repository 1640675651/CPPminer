#include "cp_onednn_worker.h"

#include "case33_gemm_onednn.hpp"
#include "case5_xor_tile.hpp"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_jackpot.hpp"
#include "cp_noise.h"
#include "cp_state.h"
#include "cp_util.h"
#include "onednn_intel_devices.hpp"
#include "opencl_context.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
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
    int use_gpu_prep = 0;
    int salted = 0;
    uint8_t b_noise_seed[32]{};
    std::vector<int8_t> B_noisy;
} g_zero_b;

static Case33GemmOnednn g_gemm;
static int g_device_index = 0;
static int g_platform_filter = -1;
static int g_context_ready = 0;
static int g_row_period_batch = CP_ONEDNN_PERIOD_BATCH_DEFAULT;
static int g_col_period_batch = CP_ONEDNN_PERIOD_BATCH_DEFAULT;
static int g_fused_jackpot = 0;
static bool g_a_row_major = true;
static bool g_b_row_major = false;
static int g_gemm_layout_cli = 0;

static int zero_b_cache_matches(const uint8_t job_key[32], int m, int n, int gpu_prep) {
    return g_zero_b.ready && g_zero_b.m == m && g_zero_b.n == n &&
           g_zero_b.use_gpu_prep == gpu_prep && memcmp(g_zero_b.job_key, job_key, 32) == 0;
}

static int zero_b_prepare_job_host(const uint8_t job_key[32], int m, int n) {
    const size_t szB = static_cast<size_t>(n) * static_cast<size_t>(K_DIM);
    g_zero_b.B_noisy.resize(szB);
    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;
    g_zero_b.use_gpu_prep = 0;

    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_zero_b.salted, g_zero_b.b_noise_seed);
    if (pearl_build_noisy_b(n, K_DIM, R_RANK, g_zero_b.b_noise_seed, NULL,
                            g_zero_b.B_noisy.data()) != 0) {
        g_zero_b.ready = 0;
        return cp_job_should_cancel() ? -1 : -2;
    }

    if (!g_gemm.prepare_job(m, n, K_DIM, g_zero_b.B_noisy.data())) {
        g_zero_b.ready = 0;
        fprintf(stderr, "[onednn] prepare_job failed\n");
        return -2;
    }

    g_zero_b.ready = 1;
    return 0;
}

static int zero_b_prepare_job_gpu(const uint8_t job_key[32], int m, int n) {
    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;
    g_zero_b.use_gpu_prep = 1;
    g_zero_b.B_noisy.clear();

    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_zero_b.salted, g_zero_b.b_noise_seed);
    if (!g_gemm.prepare_job_gpu(m, n, K_DIM, g_zero_b.b_noise_seed)) {
        g_zero_b.ready = 0;
        fprintf(stderr, "[onednn] prepare_job_gpu failed\n");
        return -2;
    }

    g_zero_b.ready = 1;
    return 0;
}

static int zero_b_prepare_job(const uint8_t job_key[32], int m, int n) {
    if (g_gemm.gpu_prep_ready()) {
        return zero_b_prepare_job_gpu(job_key, m, n);
    }
    return zero_b_prepare_job_host(job_key, m, n);
}

static int zero_b_prepare_attempt_host(const uint8_t *ab_seed, int ab_seed_len,
                                       const uint8_t job_key[32], int m, int n, int8_t *h_A_sig,
                                       uint8_t a_key_out[32]) {
    if (!h_A_sig) {
        fprintf(stderr, "[onednn] zero-B requires h_Ap_global (A_sig buffer)\n");
        return -2;
    }

    if (!zero_b_cache_matches(job_key, m, n, 0)) {
        if (zero_b_prepare_job_host(job_key, m, n) != 0) {
            return -1;
        }
    }

    uint8_t a_rng[32];
    if (cp_random_bytes(a_rng, sizeof(a_rng)) != 0) {
        fprintf(stderr, "[onednn] CSPRNG failed for random A\n");
        return -2;
    }

    if (pearl_generate_random_a(a_rng, (int)sizeof(a_rng), m, K_DIM, h_A_sig) != 0) {
        return -1;
    }

    pearl_a_noise_seed_from_a(job_key, g_zero_b.b_noise_seed, h_A_sig, m, K_DIM, g_zero_b.salted,
                              a_key_out);

    const size_t szA = static_cast<size_t>(m) * static_cast<size_t>(K_DIM);
    std::vector<int8_t> a_noisy(szA);
    if (pearl_build_noisy_a(m, K_DIM, R_RANK, a_key_out, h_A_sig, a_noisy.data()) != 0) {
        return cp_job_should_cancel() ? -1 : -2;
    }

    if (!g_gemm.prepare_attempt_a(a_noisy.data())) {
        fprintf(stderr, "[onednn] prepare_attempt_a failed\n");
        return -2;
    }

    return 0;
}

static int zero_b_prepare_attempt_gpu(const uint8_t *ab_seed, int ab_seed_len,
                                        const uint8_t job_key[32], int m, int n,
                                        uint8_t a_key_out[32]) {
    if (!zero_b_cache_matches(job_key, m, n, 1)) {
        if (zero_b_prepare_job_gpu(job_key, m, n) != 0) {
            return -1;
        }
    }

    uint8_t a_rng[32];
    if (cp_random_bytes(a_rng, sizeof(a_rng)) != 0) {
        fprintf(stderr, "[onednn] CSPRNG failed for random A\n");
        return -2;
    }

    if (!g_gemm.prepare_attempt_gpu(a_rng, (int)sizeof(a_rng), job_key, g_zero_b.b_noise_seed,
                                    g_zero_b.salted, a_key_out)) {
        fprintf(stderr, "[onednn] prepare_attempt_gpu failed\n");
        return -2;
    }

    return 0;
}

static int zero_b_prepare_attempt(const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
                                  int m, int n, int8_t *h_A_sig, uint8_t a_key_out[32]) {
    (void)h_A_sig;
    if (g_gemm.gpu_prep_ready()) {
        return zero_b_prepare_attempt_gpu(ab_seed, ab_seed_len, job_key, m, n, a_key_out);
    }
    return zero_b_prepare_attempt_host(ab_seed, ab_seed_len, job_key, m, n, h_A_sig, a_key_out);
}

static void configure_hash_tile_from_kernel(void) {
    const auto &di = g_gemm.driver_info();
    cp_pp_set_hash_tile(di.xorSubM, di.xorSubN);
    pearl_set_contiguous_tiles(1);
    pearl_set_contiguous_tile_shape(di.xorSubM, di.xorSubN);
}

} /* namespace */

extern "C" int cp_onednn_worker_handles_matrix_prep(void) { return 1; }

extern "C" void cp_onednn_worker_set_row_period_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_ROW_PERIOD_BATCH_MAX) {
        batch = CP_ROW_PERIOD_BATCH_MAX;
    }
    g_row_period_batch = batch;
    g_gemm.set_row_period_batch(batch);
}

extern "C" void cp_onednn_worker_set_col_period_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_PERIOD_BATCH_MAX) {
        batch = CP_PERIOD_BATCH_MAX;
    }
    g_col_period_batch = batch;
    g_gemm.set_col_period_batch(batch);
}

extern "C" void cp_onednn_worker_set_fused_jackpot(int on) {
    if (g_context_ready) {
        fprintf(stderr, "[onednn] set_fused_jackpot ignored after init\n");
        return;
    }
    g_fused_jackpot = on ? 1 : 0;
}

extern "C" void cp_onednn_worker_set_gemm_layout(const char *name) {
    if (g_context_ready) {
        fprintf(stderr, "[onednn] set_gemm_layout ignored after init\n");
        return;
    }
    if (!name || name[0] == '\0') {
        return;
    }
    bool a_row = g_a_row_major;
    bool b_row = g_b_row_major;
    if (!case5_ngen::case5_parse_gemm_layout_name(name, a_row, b_row)) {
        fprintf(stderr, "[onednn] invalid --onednn-layout %s (use TN, TT, NT, or NN)\n", name);
        return;
    }
    g_a_row_major = a_row;
    g_b_row_major = b_row;
    g_gemm_layout_cli = 1;
}

static void resolve_gemm_layout(void) {
    if (!g_gemm_layout_cli) {
        bool a_row = g_a_row_major;
        bool b_row = g_b_row_major;
        if (case5_ngen::case5_parse_gemm_layout_env(a_row, b_row)) {
            g_a_row_major = a_row;
            g_b_row_major = b_row;
        }
    }
    g_gemm.set_gemm_layout(g_a_row_major, g_b_row_major);
}

extern "C" void cp_onednn_worker_set_platform(int platform_index) {
    g_platform_filter = platform_index;
}

extern "C" int cp_onednn_worker_list_devices(void) {
    return onednn_intel::list_intel_gpus(g_platform_filter);
}

extern "C" int cp_onednn_worker_is_ready(void) { return g_context_ready; }

extern "C" void cp_onednn_worker_init(int *devices, int ndev) {
    if (g_context_ready) {
        return;
    }

    if (devices && ndev > 0) {
        g_device_index = devices[0];
        if (ndev > 1) {
            fprintf(stderr,
                    "[onednn] warning: uses a single Intel GPU; ignoring --devices after %d\n",
                    g_device_index);
        }
    } else {
        g_device_index = 0;
    }

    g_gemm.set_row_period_batch(g_row_period_batch);
    g_gemm.set_col_period_batch(g_col_period_batch);
    g_gemm.set_fused_jackpot(g_fused_jackpot != 0);
    resolve_gemm_layout();
    if (!g_gemm.init_context(g_device_index, g_platform_filter)) {
        fprintf(stderr, "[onednn] init failed (device=%d)\n", g_device_index);
        g_context_ready = 0;
        return;
    }

    g_context_ready = 1;
    configure_hash_tile_from_kernel();
    const auto &di = g_gemm.driver_info();
    printf("[onednn] device[%d]: %s\n", g_gemm.device_index(), g_gemm.device_name());
    printf("[onednn] platform: %s\n", g_gemm.platform_name());
    printf("[onednn] %s\n", g_gemm.backend());
    printf("[onednn] hash tile: %dx%d logical (unroll %dx%d, split=%s)\n", di.xorSubM, di.xorSubN,
           di.unrollM, di.unrollN, case5_ngen::case5_xor_subtile_split_mode_name());
    if (g_gemm.gpu_prep_ready()) {
        printf("[onednn] matrix prep: GPU-only A/B prep (%s, no matrix D2H on scan path)\n",
               case5_ngen::case5_device_layout_name(g_a_row_major, g_b_row_major));
    } else {
        printf("[onednn] matrix prep: CPU random A + noise (%s)\n",
               case5_ngen::case5_device_layout_name(g_a_row_major, g_b_row_major));
    }
    printf("[onednn] period batch: row=%d col=%d (%dx%d hash tiles/panel, %dx%d GEMM m×n)\n",
           g_row_period_batch, g_col_period_batch, g_row_period_batch, g_col_period_batch,
           g_row_period_batch * di.xorSubM, g_col_period_batch * di.xorSubN);
    fflush(stdout);
}

extern "C" void cp_onednn_worker_shutdown(void) {
    g_zero_b.ready = 0;
    g_zero_b.B_noisy.clear();
    g_context_ready = 0;
}

extern "C" void cp_onednn_worker_begin_job(const uint8_t job_key[32], int m, int n,
                                           uint32_t cert_version) {
    if (!g_context_ready) {
        return;
    }
    g_zero_b.ready = 0;
    g_zero_b.salted = (cert_version >= 3) ? 1 : 0;
    if (zero_b_prepare_job(job_key, m, n) == 0) {
        if (g_zero_b.use_gpu_prep) {
            printf("[onednn] zero-B: GPU noisy B on device layout=%s (salted=%d)\n",
                   case5_ngen::case5_device_layout_name(g_a_row_major, g_b_row_major),
                   g_zero_b.salted);
        } else {
            printf("[onednn] zero-B: host noise + device B upload layout=%s (salted=%d)\n",
                   case5_ngen::case5_device_layout_name(g_a_row_major, g_b_row_major),
                   g_zero_b.salted);
        }
        fflush(stdout);
    }
}

extern "C" int cp_onednn_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key, int8_t *h_A_sig,
        int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols, uint64_t *out_tiles_scanned) {
    (void)cpu_matrices;
    (void)h_A_noisy;
    (void)h_B_noisy;
    (void)h_Bt_sig;

    const double attempt_t0 = cp_now_sec();

    if (!g_context_ready) {
        fprintf(stderr, "[onednn] not initialized\n");
        return -1;
    }

    if (out_tiles_scanned) {
        *out_tiles_scanned = 0;
    }
    if (out_t_rows) {
        *out_t_rows = -1;
    }
    if (out_t_cols) {
        *out_t_cols = -1;
    }

    uint8_t a_key_local[32];
    const uint8_t *scan_key = a_key;

    if (!scan_key) {
        if (!h_A_sig && !g_gemm.gpu_prep_ready()) {
            fprintf(stderr, "[onednn] mine_attempt requires h_Ap_global\n");
            return -2;
        }
        const int prep_rc =
                zero_b_prepare_attempt(ab_seed, ab_seed_len, job_key, m, n, h_A_sig, a_key_local);
        if (prep_rc != 0) {
            return cp_job_should_cancel() ? -1 : 0;
        }
        scan_key = a_key_local;
    }

    if (m % g_gemm.driver_info().xorSubM != 0 || n % g_gemm.driver_info().xorSubN != 0) {
        fprintf(stderr, "[onednn] m,n must be multiples of %dx%d (got %dx%d)\n",
                g_gemm.driver_info().xorSubM, g_gemm.driver_info().xorSubN, m, n);
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

    printf("[onednn] plain_proof scan %dx%d hash tiles, difficulty scaled by %llu\n", row_parts,
           col_parts, (unsigned long long)cp_jackpot_scale_factor());
    const auto &di = g_gemm.driver_info();
    printf("[onednn] period batch: row=%d col=%d (%dx%d hash tiles/panel, %dx%d GEMM mxn)\n",
           g_row_period_batch, g_col_period_batch, g_row_period_batch, g_col_period_batch,
           g_row_period_batch * di.xorSubM, g_col_period_batch * di.xorSubN);
    printf("[onednn] GEMM %s\n", g_gemm.backend());
    fflush(stdout);

    std::atomic<uint64_t> tiles{0};
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
                                   ? static_cast<int>((cur + static_cast<uint64_t>(col_parts) - 1) /
                                                      static_cast<uint64_t>(col_parts))
                                   : 0;
            if (row_done > row_parts) {
                row_done = row_parts;
            }

            char mac_buf[32];
            cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(cur, scan_sec), mac_buf, sizeof(mac_buf));
            printf("[onednn] plain_proof progress: row parts %d/%d tiles %llu/%d (%.1f%%) %s\n",
                   row_done, row_parts, static_cast<unsigned long long>(cur), total_tiles,
                   total_tiles > 0 ? 100.0 * static_cast<double>(cur) / total_tiles : 0.0,
                   mac_buf);
            fflush(stdout);
            last_tiles = cur;
            last_report = now;
        }
    });

    int found = 0;
    int hit_rows = -1;
    int hit_cols = -1;
    uint64_t tiles_scanned = 0;

    const bool ok = g_gemm.scan_for_share(
            a_key8, bound, &found, &hit_rows, &hit_cols, &tiles_scanned,
            []() -> bool { return cp_job_should_cancel() != 0; },
            [&](uint64_t cur) { tiles.store(cur, std::memory_order_relaxed); });

    const double scan_sec = cp_now_sec() - scan_t0;
    {
        std::lock_guard<std::mutex> lock(progress_mu);
        scan_done.store(true, std::memory_order_relaxed);
    }
    progress_cv.notify_all();
    if (progress_thread.joinable()) {
        progress_thread.join();
    }

    if (out_tiles_scanned) {
        *out_tiles_scanned = tiles_scanned;
    }

    const double post_sec = 0.0;

    if (cp_job_should_cancel()) {
        cp_log_attempt_timing("onednn", prep_sec, scan_sec, tiles_scanned, post_sec);
        return -1;
    }
    if (!ok) {
        cp_log_attempt_timing("onednn", prep_sec, scan_sec, tiles_scanned, post_sec);
        return -1;
    }

    if (found) {
        printf("[onednn] plain_proof SHARE t_rows=%d t_cols=%d\n", hit_rows, hit_cols);
        fflush(stdout);
        if (out_t_rows) {
            *out_t_rows = hit_rows;
        }
        if (out_t_cols) {
            *out_t_cols = hit_cols;
        }
        cp_log_attempt_timing("onednn", prep_sec, scan_sec, tiles_scanned, post_sec);
        return 1;
    }

    cp_log_attempt_timing("onednn", prep_sec, scan_sec, tiles_scanned, post_sec);
    return 0;
}

extern "C" int cp_onednn_worker_fetch_share_signals(int8_t *h_A_sig, int8_t *h_Bt_sig) {
    (void)h_Bt_sig;
    if (!h_A_sig) {
        return -1;
    }
    if (!g_gemm.gpu_prep_ready()) {
        /* Signal A already lives in h_Ap_global (CPU prep). */
        return 0;
    }
    if (!g_gemm.read_A_sig(h_A_sig)) {
        fprintf(stderr, "[onednn] failed to read A_sig for proof\n");
        return -1;
    }
    return 0;
}

extern "C" int cp_onednn_hash_tile_mr(void) {
    return g_context_ready ? g_gemm.driver_info().xorSubM : PP_HASH_H;
}

extern "C" int cp_onednn_hash_tile_w(void) {
    return g_context_ready ? g_gemm.driver_info().xorSubN : PP_HASH_W;
}
