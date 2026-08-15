#include "cp_opencl_worker.h"

#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_state.h"
#include "cp_util.h"
#include "case33_gemm_ocl.hpp"
#include "case32_layout.hpp"
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
    int use_cpu_prep = 0;
    uint8_t b_noise_seed[32]{};
    std::vector<int8_t> B_noisy;
} g_zero_b;

static Case33GemmOcl g_gemm;
static int g_device_index = 0;
static int g_platform_filter = -1;
static int g_context_ready = 0;
static int g_macro_batch = CP_MACRO_BATCH_DEFAULT;
static int g_tile_mr = 0;
static int g_tile_nr = 0;
static int g_hash_tile_mr = 8;
static int g_hash_tile_w = 8;
static int g_issue_broadcast = 0;

namespace {

bool string_contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle || !needle[0]) {
        return false;
    }
    for (const char *p = haystack; *p; ++p) {
        const char *h = p;
        const char *n = needle;
        while (*h && *n &&
               (static_cast<unsigned char>(*h) == static_cast<unsigned char>(*n) ||
                static_cast<unsigned char>(*h) + 32u == static_cast<unsigned char>(*n) ||
                static_cast<unsigned char>(*h) == static_cast<unsigned char>(*n) + 32u)) {
            ++h;
            ++n;
        }
        if (!*n) {
            return true;
        }
    }
    return false;
}

bool ocl_device_is_amd(const OclDeviceInfo &dev) {
    if (string_contains_ci(dev.vendor_name.c_str(), "advanced micro devices") ||
        string_contains_ci(dev.vendor_name.c_str(), "amd")) {
        return true;
    }
    if (string_contains_ci(dev.platform_name.c_str(), "amd")) {
        return true;
    }
    return string_contains_ci(dev.device_name.c_str(), "radeon") ||
           string_contains_ci(dev.device_name.c_str(), "gfx");
}

} /* namespace */

static int zero_b_cache_matches(const uint8_t job_key[32], int m, int n, int cpu_prep) {
    return g_zero_b.ready && g_zero_b.m == m && g_zero_b.n == n &&
           g_zero_b.use_cpu_prep == cpu_prep &&
           memcmp(g_zero_b.job_key, job_key, 32) == 0;
}

static int zero_b_prepare_job_host(const uint8_t job_key[32], int m, int n) {
    const size_t szB = static_cast<size_t>(n) * static_cast<size_t>(K_DIM);
    g_zero_b.B_noisy.resize(szB);
    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;
    g_zero_b.use_cpu_prep = 1;

    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_zero_b.b_noise_seed);
    if (pearl_build_noisy_b(n, K_DIM, R_RANK, g_zero_b.b_noise_seed, NULL,
                            g_zero_b.B_noisy.data()) != 0) {
        g_zero_b.ready = 0;
        return cp_job_should_cancel() ? -1 : -2;
    }

    if (!g_gemm.prepare_job(m, n, K_DIM, g_zero_b.B_noisy.data())) {
        g_zero_b.ready = 0;
        fprintf(stderr, "[ocl] prepare_job failed\n");
        return -2;
    }

    g_zero_b.ready = 1;
    return 0;
}

static int zero_b_prepare_job_gpu(const uint8_t job_key[32], int m, int n) {
    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;
    g_zero_b.use_cpu_prep = 0;
    g_zero_b.B_noisy.clear();

    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_zero_b.b_noise_seed);
    if (!g_gemm.prepare_job_gpu(m, n, K_DIM, g_zero_b.b_noise_seed)) {
        g_zero_b.ready = 0;
        fprintf(stderr, "[ocl] prepare_job_gpu failed\n");
        return -2;
    }

    g_zero_b.ready = 1;
    return 0;
}

static int zero_b_prepare_job(const uint8_t job_key[32], int m, int n, int cpu_prep) {
    if (cpu_prep) {
        return zero_b_prepare_job_host(job_key, m, n);
    }
    return zero_b_prepare_job_gpu(job_key, m, n);
}

static int zero_b_prepare_attempt_host(const uint8_t *ab_seed, int ab_seed_len,
                                       const uint8_t job_key[32], int m, int n,
                                       int8_t *h_A_sig, uint8_t a_key_out[32]) {
    (void)ab_seed;
    (void)ab_seed_len;
    if (!h_A_sig) {
        fprintf(stderr, "[ocl] zero-B requires h_Ap_global (A_sig buffer)\n");
        return -2;
    }

    if (!zero_b_cache_matches(job_key, m, n, 1)) {
        if (zero_b_prepare_job_host(job_key, m, n) != 0) {
            return -1;
        }
    }

    uint8_t a_rng[32];
    if (cp_random_bytes(a_rng, sizeof(a_rng)) != 0) {
        fprintf(stderr, "[ocl] CSPRNG failed for random A\n");
        return -2;
    }

    if (pearl_generate_random_a(a_rng, (int)sizeof(a_rng), m, K_DIM, h_A_sig) != 0) {
        return -1;
    }

    pearl_a_noise_seed_from_a(job_key, g_zero_b.b_noise_seed, h_A_sig, m, K_DIM, a_key_out);

    const size_t szA = static_cast<size_t>(m) * static_cast<size_t>(K_DIM);
    std::vector<int8_t> a_noisy(szA);
    if (pearl_build_noisy_a(m, K_DIM, R_RANK, a_key_out, h_A_sig, a_noisy.data()) != 0) {
        return cp_job_should_cancel() ? -1 : -2;
    }

    if (!g_gemm.prepare_attempt_a(a_noisy.data())) {
        fprintf(stderr, "[ocl] prepare_attempt_a failed\n");
        return -2;
    }

    return 0;
}

static int zero_b_prepare_attempt_gpu(const uint8_t *ab_seed, int ab_seed_len,
                                        const uint8_t job_key[32], int m, int n,
                                        uint8_t a_key_out[32]) {
    (void)ab_seed;
    (void)ab_seed_len;
    if (!zero_b_cache_matches(job_key, m, n, 0)) {
        if (zero_b_prepare_job_gpu(job_key, m, n) != 0) {
            return -1;
        }
    }

    uint8_t a_rng[32];
    if (cp_random_bytes(a_rng, sizeof(a_rng)) != 0) {
        fprintf(stderr, "[ocl] CSPRNG failed for random A\n");
        return -2;
    }

    if (!g_gemm.prepare_attempt_gpu(a_rng, (int)sizeof(a_rng), job_key, g_zero_b.b_noise_seed,
                                      a_key_out)) {
        fprintf(stderr, "[ocl] prepare_attempt_gpu failed\n");
        return -2;
    }

    return 0;
}

} /* namespace */

extern "C" int cp_opencl_worker_handles_matrix_prep(void) { return 1; }

extern "C" void cp_opencl_worker_set_macro_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_MACRO_BATCH_MAX) {
        batch = CP_MACRO_BATCH_MAX;
    }
    g_macro_batch = batch;
    g_gemm.set_macro_batch(batch);
}

extern "C" void cp_opencl_worker_set_platform(int platform_index) {
    g_platform_filter = platform_index;
}

extern "C" void cp_opencl_worker_set_tile(int mr, int nr) {
    if (mr <= 0 || nr <= 0) {
        g_tile_mr = 0;
        g_tile_nr = 0;
        return;
    }
    g_tile_mr = mr;
    g_tile_nr = nr;
}

extern "C" void cp_opencl_worker_set_issue_broadcast(int on) {
    g_issue_broadcast = on ? 1 : 0;
}

extern "C" void cp_opencl_configure_tile(int device_index, int platform_filter) {
    int tile_mr = PP_HASH_H;
    int tile_nr = 8;
    const char *source = "default 8x8";

    if (g_tile_mr > 0 && g_tile_nr > 0) {
        if (!case32::configure(g_tile_mr, g_tile_nr)) {
            fprintf(stderr, "[ocl] --ocl-tile %dx%d invalid; using 8x8\n", g_tile_mr,
                    g_tile_nr);
            tile_mr = PP_HASH_H;
            tile_nr = 8;
            source = "invalid CLI, using 8x8";
            case32::configure(tile_mr, tile_nr);
        } else {
            tile_mr = g_tile_mr;
            tile_nr = g_tile_nr;
            source = "CLI override";
        }
    } else {
        const std::vector<OclDeviceInfo> devices =
                OpenClContext::enumerate_devices(platform_filter);
        if (device_index >= 0 && device_index < static_cast<int>(devices.size())) {
            const OclDeviceInfo &dev = devices[static_cast<size_t>(device_index)];
            if (ocl_device_is_amd(dev)) {
                tile_nr = 16;
                tile_mr = PP_HASH_H;
                source = "AMD GPU auto 8x16";
            }
        }
        if (!case32::configure(tile_mr, tile_nr)) {
            fprintf(stderr, "[ocl] hash tile configure failed\n");
            return;
        }
    }

    g_hash_tile_mr = case32::kMR;
    g_hash_tile_w = case32::kNR;
    cp_pp_set_hash_tile(case32::kMR, case32::kNR);
    pearl_set_contiguous_tile_shape(case32::kMR, case32::kNR);

    printf("[ocl] OpenCL hash tile: %dx%d (%s, %d WI/macro)\n", case32::kMR, case32::kNR,
           source, case32::kMacroWorkItems);
    fflush(stdout);
}

extern "C" int cp_opencl_hash_tile_mr(void) { return g_hash_tile_mr; }
extern "C" int cp_opencl_hash_tile_w(void) { return g_hash_tile_w; }

extern "C" void cp_opencl_configure_tile_for_worker(int device_index) {
    cp_opencl_configure_tile(device_index, g_platform_filter);
}

extern "C" int cp_opencl_worker_list_devices(void) {
    return OpenClContext::list_devices(g_platform_filter);
}

extern "C" void cp_opencl_worker_init(int *devices, int ndev) {
    if (devices && ndev > 0) {
        g_device_index = devices[0];
        if (ndev > 1) {
            fprintf(stderr,
                    "[ocl] warning: OpenCL uses a single device; ignoring --devices after %d\n",
                    g_device_index);
        }
    } else {
        g_device_index = 0;
    }

    const std::string kernel_path = cp_ocl_resolve_kernel_path();
    g_gemm.set_macro_batch(g_macro_batch);
    g_gemm.set_issue_broadcast(g_issue_broadcast);
    cp_opencl_configure_tile(g_device_index, g_platform_filter);
    if (!g_gemm.init_context(kernel_path.c_str(), g_device_index, g_platform_filter,
                             !g_cpu_matrix_gen)) {
        fprintf(stderr, "[ocl] OpenCL init failed (kernel=%s)\n", kernel_path.c_str());
        g_context_ready = 0;
        return;
    }

    g_context_ready = 1;
    printf("[ocl] device[%d]: %s (%s)\n", g_gemm.device_index(), g_gemm.device_name(),
           g_gemm.discrete_gpu() ? "discrete GPU" : "integrated GPU/CPU");
    printf("[ocl] platform: %s\n", g_gemm.platform_name());
    printf("[ocl] max work-group size: %zu\n", g_gemm.max_work_group_size());
    printf("[ocl] %s\n", g_gemm.backend());
    printf("[ocl] %s\n", g_gemm.dpi_status());
    if (g_issue_broadcast) {
        printf("[ocl] issue: broadcast (--ocl-issue broadcast)\n");
    }
    printf("[ocl] macro batch: %d blocks (%d hash tiles/launch)\n", g_gemm.macro_batch(),
           g_gemm.macro_batch() * case32::hash_tiles_per_macro());
    if (g_cpu_matrix_gen) {
        printf("[ocl] host matrix prep (--cpu-gen)\n");
    } else {
        printf("[ocl] zero-B GPU prep (default); --cpu-gen for host prep (~1 GiB VRAM)\n");
    }
    fflush(stdout);
}

extern "C" void cp_opencl_worker_shutdown(void) {
    g_zero_b.ready = 0;
    g_zero_b.B_noisy.clear();
    g_context_ready = 0;
}

extern "C" void cp_opencl_worker_begin_job(const uint8_t job_key[32], int m, int n) {
    if (!g_context_ready) {
        return;
    }
    g_zero_b.ready = 0;
    const int cpu_prep = g_cpu_matrix_gen;
    if (zero_b_prepare_job(job_key, m, n, cpu_prep) == 0) {
        if (cpu_prep) {
            printf("[ocl] zero-B: host noise + prepack, B cached on device\n");
        } else {
            printf("[ocl] zero-B: GPU noise + fused prepack, B on device\n");
        }
        fflush(stdout);
    }
}

extern "C" int cp_opencl_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned) {
    (void)h_Bt_sig;
    (void)h_A_noisy;
    (void)h_B_noisy;

    const double attempt_t0 = cp_now_sec();
    const int cpu_prep = cpu_matrices || g_cpu_matrix_gen;

    if (!g_context_ready) {
        fprintf(stderr, "[ocl] OpenCL not initialized\n");
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
        int prep_rc;
        if (cpu_prep) {
            if (!h_A_sig) {
                fprintf(stderr, "[ocl] mine_attempt requires h_Ap_global\n");
                return -2;
            }
            prep_rc = zero_b_prepare_attempt_host(ab_seed, ab_seed_len, job_key, m, n, h_A_sig,
                                                  a_key_local);
        } else {
            prep_rc = zero_b_prepare_attempt_gpu(ab_seed, ab_seed_len, job_key, m, n, a_key_local);
        }
        if (prep_rc != 0) {
            return cp_job_should_cancel() ? -1 : 0;
        }
        scan_key = a_key_local;
    }

    if (m % case32::kMacroM != 0 || n % case32::kMacroN != 0) {
        fprintf(stderr, "[ocl] m,n must be multiples of %dx%d (got %dx%d)\n", case32::kMacroM,
                case32::kMacroN, m, n);
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

    printf("[ocl] plain_proof scan %dx%d hash tiles, difficulty scaled by %.0f\n", row_parts,
           col_parts, cp_pp_macs_per_hash_tile());
    printf("[ocl] GEMM %s\n", g_gemm.backend());
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
            printf("[ocl] plain_proof progress: row parts %d/%d tiles %llu/%d (%.1f%%) %s\n",
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

    double post_sec = 0.0;

    if (cp_job_should_cancel()) {
        cp_log_attempt_timing("ocl", prep_sec, scan_sec, tiles_scanned, post_sec);
        return -1;
    }
    if (!ok) {
        cp_log_attempt_timing("ocl", prep_sec, scan_sec, tiles_scanned, post_sec);
        return -1;
    }

    if (found) {
        printf("[ocl] plain_proof SHARE t_rows=%d t_cols=%d\n", hit_rows, hit_cols);
        fflush(stdout);
        if (out_t_rows) {
            *out_t_rows = hit_rows;
        }
        if (out_t_cols) {
            *out_t_cols = hit_cols;
        }
        /* Device→host A download is deferred to cp_worker_fetch_share_signals after
         * the mine loop reclaims the host buffer (single-buffer handoff). */
        cp_log_attempt_timing("ocl", prep_sec, scan_sec, tiles_scanned, post_sec);
        return 1;
    }

    cp_log_attempt_timing("ocl", prep_sec, scan_sec, tiles_scanned, post_sec);
    return 0;
}

extern "C" int cp_opencl_worker_fetch_share_signals(int8_t *h_A_sig, int8_t *h_Bt_sig) {
    (void)h_Bt_sig;
    if (!h_A_sig) {
        return -1;
    }
    if (!g_gemm.read_A_sig(h_A_sig)) {
        fprintf(stderr, "[ocl] failed to read A_sig for proof\n");
        return -1;
    }
    return 0;
}
