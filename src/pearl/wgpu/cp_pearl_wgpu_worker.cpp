#include "cp_pearl_wgpu_worker.h"

#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_pearl_wgpu.h"
#include "cp_proof.h"
#include "cp_share_witness.h"
#include "cp_util.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace {

struct ZeroBCache {
    uint8_t job_key[32]{};
    int m = 0;
    int n = 0;
    int ready = 0;
    int salted = 0;
    uint8_t b_noise_seed[32]{};
} g_zero_b;

static int g_context_ready = 0;
static int g_macro_batch = CP_MACRO_BATCH_DEFAULT;
static int g_tile_mr = 8;
static int g_tile_nr = 8;
static int g_macro = 128;

static int cancel_check_thunk(void) { return cp_job_should_cancel() != 0 ? 1 : 0; }

static std::atomic<uint64_t> *g_scan_progress_tiles = nullptr;

static void scan_progress_thunk(uint64_t tiles) {
    if (g_scan_progress_tiles) {
        g_scan_progress_tiles->store(tiles, std::memory_order_relaxed);
    }
}

static int zero_b_cache_matches(const uint8_t job_key[32], int m, int n) {
    return g_zero_b.ready && g_zero_b.m == m && g_zero_b.n == n &&
           memcmp(g_zero_b.job_key, job_key, 32) == 0;
}

static int zero_b_prepare_job(const uint8_t job_key[32], int m, int n) {
    memcpy(g_zero_b.job_key, job_key, 32);
    g_zero_b.m = m;
    g_zero_b.n = n;

    pearl_b_noise_seed_from_bt(job_key, nullptr, n, K_DIM, g_zero_b.salted,
                               g_zero_b.b_noise_seed);
    if (cp_pearl_wgpu_begin_job(m, n, K_DIM, g_zero_b.b_noise_seed) != 0) {
        g_zero_b.ready = 0;
        fprintf(stderr, "[pearl-wgpu] begin_job / GPU prep B failed\n");
        return -2;
    }
    g_zero_b.ready = 1;
    return 0;
}

} /* namespace */

extern "C" int cp_pearl_wgpu_worker_handles_matrix_prep(void) { return 1; }

extern "C" void cp_pearl_wgpu_worker_set_macro_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_MACRO_BATCH_MAX) {
        batch = CP_MACRO_BATCH_MAX;
    }
    g_macro_batch = batch;
}

extern "C" void cp_pearl_wgpu_worker_set_use_lds(int mode) { cp_pearl_wgpu_set_lds(mode); }

extern "C" void cp_pearl_wgpu_worker_set_tile(int mr, int nr) {
    g_tile_mr = mr;
    g_tile_nr = nr;
}

extern "C" void cp_pearl_wgpu_worker_set_macro(int macro_m, int macro_n) {
    (void)macro_n;
    g_macro = macro_m;
}

extern "C" int cp_pearl_wgpu_worker_hash_tile_mr(void) { return g_tile_mr; }

extern "C" int cp_pearl_wgpu_worker_hash_tile_w(void) {
    return (g_tile_mr == 4 && g_tile_nr == 4) ? 8 : g_tile_nr;
}

extern "C" int cp_pearl_wgpu_worker_list_devices(void) {
    return cp_pearl_wgpu_list_devices();
}

extern "C" void cp_pearl_wgpu_worker_init(int *devices, int ndev) {
    if (cp_pearl_wgpu_set_tile(g_tile_mr, g_tile_nr, g_macro, g_macro) != 0 ||
        cp_pearl_wgpu_init(devices, ndev) != 0) {
        fprintf(stderr, "[pearl-wgpu] init failed\n");
        g_context_ready = 0;
        return;
    }
    g_context_ready = 1;
    /* Jackpot bound + proof layout use the kernel's hash tile (not default PP_HASH_W=16). */
    const int hash_mr = cp_pearl_wgpu_worker_hash_tile_mr();
    const int hash_w = cp_pearl_wgpu_worker_hash_tile_w();
    cp_pp_set_hash_tile(hash_mr, hash_w);
    pearl_set_contiguous_tile_shape(hash_mr, hash_w);
    printf("[pearl-wgpu] register tile %dx%d, hash tile %dx%d, macro %dx%d, KR=128, GPU prep + "
           "fused jackpot\n",
           g_tile_mr, g_tile_nr, hash_mr, hash_w, g_macro, g_macro);
    printf("[pearl-wgpu] jackpot scale tile %dx%d factor=%llu\n", cp_pp_hash_tile_h(),
           cp_pp_hash_tile_w(), (unsigned long long)cp_jackpot_scale_factor());
    printf("[pearl-wgpu] macro batch: %d blocks\n", g_macro_batch);
    fflush(stdout);
}

extern "C" void cp_pearl_wgpu_worker_shutdown(void) {
    g_zero_b.ready = 0;
    g_context_ready = 0;
    cp_pearl_wgpu_shutdown();
}

extern "C" int cp_pearl_wgpu_worker_is_ready(void) {
    return g_context_ready && cp_pearl_wgpu_is_ready() != 0;
}

extern "C" void cp_pearl_wgpu_worker_begin_job(const uint8_t job_key[32], int m, int n,
                                              uint32_t cert_version) {
    if (!g_context_ready) {
        return;
    }
    g_zero_b.ready = 0;
    g_zero_b.salted = (cert_version >= 3) ? 1 : 0;
    if (zero_b_prepare_job(job_key, m, n) == 0) {
        printf("[pearl-wgpu] zero-B: GPU noise + fused prepack (salted=%d)\n",
               g_zero_b.salted);
        fflush(stdout);
    }
}

extern "C" int cp_pearl_wgpu_worker_mine_attempt(
        const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t *h_A_noisy, const int8_t *h_B_noisy, const uint8_t *a_key,
        int8_t *h_A_sig, int8_t *h_Bt_sig, int *out_t_rows, int *out_t_cols,
        uint64_t *out_tiles_scanned) {
    (void)h_Bt_sig;
    (void)h_A_noisy;
    (void)h_B_noisy;
    (void)h_A_sig;
    (void)cpu_matrices;

    const double attempt_t0 = cp_now_sec();

    if (!g_context_ready) {
        fprintf(stderr, "[pearl-wgpu] not initialized\n");
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

    if (m % g_macro != 0 || n % g_macro != 0) {
        fprintf(stderr, "[pearl-wgpu] m,n must be multiples of %d (got %dx%d)\n", g_macro, m, n);
        return -1;
    }

    uint8_t a_key_local[32];
    const uint8_t *scan_key = a_key;

    if (!scan_key) {
        if (!zero_b_cache_matches(job_key, m, n)) {
            if (zero_b_prepare_job(job_key, m, n) != 0) {
                return cp_job_should_cancel() ? -1 : 0;
            }
        }

        uint8_t a_rng[32];
        if (ab_seed && ab_seed_len > 0) {
            /* Prefer caller-provided attempt seed when present. */
            memset(a_rng, 0, sizeof(a_rng));
            const int copy = ab_seed_len < 32 ? ab_seed_len : 32;
            memcpy(a_rng, ab_seed, static_cast<size_t>(copy));
        } else if (cp_random_bytes(a_rng, sizeof(a_rng)) != 0) {
            fprintf(stderr, "[pearl-wgpu] CSPRNG failed for random A\n");
            return -2;
        }

        uint8_t hash_a[32];
        if (cp_pearl_wgpu_prep_a_signal(a_rng, (int)sizeof(a_rng), job_key, hash_a) != 0) {
            fprintf(stderr, "[pearl-wgpu] prep_a_signal failed\n");
            return cp_job_should_cancel() ? -1 : 0;
        }

        pearl_a_noise_seed_from_hash(g_zero_b.b_noise_seed, hash_a, static_cast<uint32_t>(m),
                                     g_zero_b.salted, a_key_local);
        if (cp_pearl_wgpu_prepack_a(a_key_local) != 0) {
            fprintf(stderr, "[pearl-wgpu] prepack_a failed\n");
            return -2;
        }
        scan_key = a_key_local;
    }

    uint32_t bound[8];
    const int hash_mr = cp_pearl_wgpu_worker_hash_tile_mr();
    const int hash_w = cp_pearl_wgpu_worker_hash_tile_w();
    cp_pp_set_hash_tile(hash_mr, hash_w);
    cp_scale_jackpot_target(pool_tgt, bound);
    uint32_t a_key8[8];
    memcpy(a_key8, scan_key, 32);

    const int row_parts = m / hash_mr;
    const int col_parts = n / hash_w;
    const int total_tiles = row_parts * col_parts;
    const double prep_sec = cp_now_sec() - attempt_t0;
    const double scan_t0 = cp_now_sec();

    printf("[pearl-wgpu] plain_proof scan %dx%d hash tiles, difficulty scaled by %llu (tile %dx%d)\n",
           row_parts, col_parts, (unsigned long long)cp_jackpot_scale_factor(),
           cp_pp_hash_tile_h(), cp_pp_hash_tile_w());
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
            double scan_sec = now - scan_t0;
            if (scan_sec < 1e-9) {
                scan_sec = 1e-9;
            }
            int row_done =
                    col_parts > 0
                            ? static_cast<int>((cur + static_cast<uint64_t>(col_parts) - 1) /
                                               static_cast<uint64_t>(col_parts))
                            : 0;
            if (row_done > row_parts) {
                row_done = row_parts;
            }
            char mac_buf[32];
            cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(cur, scan_sec), mac_buf, sizeof(mac_buf));
            printf("[pearl-wgpu] plain_proof progress: row parts %d/%d tiles %llu/%d (%.1f%%) %s\n",
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

    g_scan_progress_tiles = &tiles;
    const int st = cp_pearl_wgpu_scan(a_key8, bound, g_macro_batch, &found, &hit_rows, &hit_cols,
                                      &tiles_scanned, cancel_check_thunk, scan_progress_thunk);
    g_scan_progress_tiles = nullptr;
    tiles.store(tiles_scanned, std::memory_order_relaxed);

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

    if (cp_job_should_cancel() || st < 0) {
        cp_log_attempt_timing("pearl-wgpu", prep_sec, scan_sec, tiles_scanned, 0.0);
        return -1;
    }

    if (found || st == 1) {
        printf("[pearl-wgpu] plain_proof SHARE t_rows=%d t_cols=%d\n", hit_rows, hit_cols);
        fflush(stdout);
        if (out_t_rows) {
            *out_t_rows = hit_rows;
        }
        if (out_t_cols) {
            *out_t_cols = hit_cols;
        }
        cp_log_attempt_timing("pearl-wgpu", prep_sec, scan_sec, tiles_scanned, 0.0);
        return 1;
    }

    cp_log_attempt_timing("pearl-wgpu", prep_sec, scan_sec, tiles_scanned, 0.0);
    return 0;
}

extern "C" int cp_pearl_wgpu_worker_fetch_share_signals(int8_t *h_A_sig, int8_t *h_Bt_sig) {
    (void)h_Bt_sig;
    if (!h_A_sig || !g_zero_b.ready) {
        return -1;
    }
    const size_t elems = static_cast<size_t>(g_zero_b.m) * static_cast<size_t>(K_DIM);
    if (cp_pearl_wgpu_download_a_sig(h_A_sig, elems) != 0) {
        fprintf(stderr, "[pearl-wgpu] failed to read A_sig for proof\n");
        return -1;
    }
    return 0;
}

extern "C" int cp_pearl_wgpu_worker_fetch_share_witness(int t_rows, int t_cols, int tile_layout,
                                                       CpShareWitness **out) {
    (void)t_cols;
    if (!out) {
        return -1;
    }
    *out = nullptr;
    const int num_subroots = cp_pearl_wgpu_a_witness_subroots();
    if (!g_zero_b.ready || num_subroots < 0) {
        return -1;
    }
    const int m = g_zero_b.m;

    CpShareWitness *w = static_cast<CpShareWitness *>(calloc(1, sizeof(CpShareWitness)));
    if (!w) {
        return -1;
    }
    w->tile_layout = tile_layout;

    const int nb = cp_proof_witness_blocks(tile_layout, 0, t_rows, m, K_DIM, w->a_block_idx,
                                           CP_WITNESS_MAX_BLOCKS);
    if (nb <= 0) {
        fprintf(stderr, "[pearl-wgpu] share witness: no A blocks for t_rows=%d layout=%d\n",
                t_rows, tile_layout);
        cp_share_witness_free(w);
        return -1;
    }
    w->a_num_blocks = static_cast<size_t>(nb);
    w->a_blocks = static_cast<uint8_t *>(calloc(static_cast<size_t>(nb), CP_WITNESS_BLOCK_BYTES));
    if (num_subroots > 0) {
        w->a_subroots = static_cast<uint8_t *>(malloc(static_cast<size_t>(num_subroots) * 32));
        w->a_num_subroots = static_cast<size_t>(num_subroots);
    }
    if (!w->a_blocks || (num_subroots > 0 && !w->a_subroots) ||
        cp_pearl_wgpu_read_a_witness(w->a_block_idx, nb, CP_WITNESS_BLOCK_BYTES, w->a_blocks,
                                     w->a_subroots, w->a_root) != 0) {
        fprintf(stderr, "[pearl-wgpu] share witness: failed to read A witness\n");
        cp_share_witness_free(w);
        return -1;
    }
    /* B^T is all-zero: bt_subroots stays NULL and the proof uses host-cached zero sub-roots. */
    *out = w;
    return 0;
}
