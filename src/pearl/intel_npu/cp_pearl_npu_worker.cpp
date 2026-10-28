#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "cp_pearl_npu_worker.h"

#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_npu_engine.hpp"
#include "cp_state.h"
#include "cp_util.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/*
 * Intel NPU worker (zero-B, like the Hexagon and CPU workers). The host generates and packs all the
 * noise; the NPU runs the fused GEMM + tile XOR; the host keeps signal A, its hash, the seeds and the
 * BLAKE3 jackpot.
 *   job:     noisy B (signal B = 0) built on the host and packed per launch column block into tensors
 *            the NPU reads in place.
 *   attempt: sparse random signal A on the host -> a_noise_seed and permutation pairs; then the matrix
 *            is scanned in launches of rows x columns x K (one compiled graph, cp_npu_graph.cpp; the
 *            shape in 128x128 macros from --row-period-batch x --col-period-batch).
 *            A prep thread builds each row panel of noisy A ahead of its launches; the calling thread
 *            keeps one launch per infer request (4 by default) queued on the NPU and checks the
 *            jackpot of each output as it completes, while the NPU runs the launches queued behind it.
 * Hash tile: 8 rows x 16 columns (proof layout CP_TILE_LAYOUT_CONTIGUOUS). A hit is recomputed on the
 * host from the seeds before it can become a share.
 */

namespace {

using cp_npu::kKr;
using cp_npu::kTileCols;
using cp_npu::kTileRows;

constexpr int kMilestones = K_DIM / R_RANK;
static_assert(R_RANK == cp_npu::kKr, "the NPU graph takes one milestone per 128 values of k");
constexpr int kMacro = 128;        /* launch shape unit, as the other backends' period batches */
constexpr int kMaxColMacros = 4;   /* 8 column chains of 64: lockstep keeps all of them live */
constexpr int kRowBlock = 512;     /* rows per accumulator chain; launches above it are made of whole blocks */
constexpr int kPanelSlots = 3; /* A panels in flight: the one being scanned, the next, one spare */

cp_npu::Engine* g_eng = nullptr;
int g_requests = 4;
bool g_fused_jackpot = false; /* --fused-jackpot: BLAKE3 jackpot on the SHAVEs (TileJackpot) */
int g_row_macros = 0; /* <= 0: CP_INTEL_NPU_ROW_MACROS_DEFAULT */
int g_col_macros = 0; /* <= 0: CP_INTEL_NPU_COL_MACROS_DEFAULT */
PearlMatrixHash* g_ahash = nullptr;

struct JobCache {
    uint8_t job_key[32]{};
    int m = 0;
    int n = 0;
    int salted = 0;
    int ready = 0;
    uint8_t b_noise_seed[32]{};
    std::vector<uint32_t> pairs_b;
    std::vector<ov::Tensor> bt; /* n / cols blocks of B^T rows, [K/128, cols, 128] u8 (value + 128) */
} g_job;

/* The launch for row_macros x col_macros 128x128 macros (<= 0: default): the largest buildable shape at
 * or below the request that divides an m x n matrix. Rows above one row block are whole row blocks;
 * columns stop at kMaxColMacros. m, n are multiples of 1024, so 512 x 512 always fits. */
void pick_launch(int row_macros, int col_macros, int m, int n, int* rows, int* cols) {
    if (row_macros <= 0)
        row_macros = CP_INTEL_NPU_ROW_MACROS_DEFAULT;
    if (col_macros <= 0)
        col_macros = CP_INTEL_NPU_COL_MACROS_DEFAULT;
    int r = std::max(1, std::min(row_macros * kMacro, m > 0 ? m : row_macros * kMacro));
    while (r > kMacro && ((r > kRowBlock && r % kRowBlock) || (m > 0 && m % r)))
        r -= kMacro;
    int c = std::max(1, std::min(col_macros, kMaxColMacros)) * kMacro;
    while (c > kMacro && n > 0 && n % c)
        c -= kMacro;
    *rows = r;
    *cols = c;
}

int env_int(const char* name, int def, int lo, int hi) {
    const char* v = std::getenv(name);
    if (!v || !*v)
        return def;
    const int x = std::atoi(v);
    return x < lo ? lo : (x > hi ? hi : x);
}

int hw_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n ? int(n) : 4;
}

/* fn(begin, end) over [0, count) on up to `threads` threads. */
template <class F>
void parallel_for(int count, int threads, F fn) {
    threads = std::max(1, std::min(threads, count));
    if (threads == 1) {
        fn(0, count);
        return;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] { fn(int(int64_t(count) * t / threads), int(int64_t(count) * (t + 1) / threads)); });
    for (auto& th : pool)
        th.join();
}

/* Row r (K int8 values) of a [K/128, rows, 128] u8 panel tensor: value + 128 is value ^ 0x80. */
void pack_row(const int8_t* src, int rows, int r, uint8_t* dst) {
    for (int p = 0; p < kMilestones; ++p) {
        const uint64_t* s = reinterpret_cast<const uint64_t*>(src + size_t(p) * kKr);
        uint64_t* d = reinterpret_cast<uint64_t*>(dst + (size_t(p) * rows + r) * kKr);
        for (int i = 0; i < kKr / 8; ++i)
            d[i] = s[i] ^ 0x8080808080808080ull;
    }
}

int job_matches(const uint8_t job_key[32], int m, int n) {
    return g_job.ready && g_job.m == m && g_job.n == n && memcmp(g_job.job_key, job_key, 32) == 0;
}

/* The job's noisy B (signal B = 0): its seed from the hash of the zero B^T, then every column built
 * and packed into the block tensors. */
int prepare_job(const uint8_t job_key[32], int m, int n) {
    if (!g_eng || !g_eng->ready())
        return -2;
    const int rows = g_eng->shape().rows, cols = g_eng->shape().cols;
    if (m % rows != 0 || n % cols != 0) {
        fprintf(stderr, "[intel-npu] the %dx%d matrix is not a whole number of %dx%d launches "
                        "(--row-period-batch / --col-period-batch)\n", m, n, rows, cols);
        return -2;
    }
    const double t0 = cp_now_sec();
    g_job.ready = 0;
    memcpy(g_job.job_key, job_key, 32);
    g_job.m = m;
    g_job.n = n;
    pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_job.salted, g_job.b_noise_seed);
    const double t_seed = cp_now_sec() - t0;
    g_job.pairs_b.assign(size_t(K_DIM) * 2, 0);
    pearl_build_perm_pairs_b(g_job.b_noise_seed, K_DIM, R_RANK, g_job.pairs_b.data());

    const int blocks = n / cols;
    try {
        if ((int)g_job.bt.size() != blocks) {
            g_job.bt.clear();
            for (int b = 0; b < blocks; ++b)
                g_job.bt.push_back(g_eng->host_tensor(1));
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[intel-npu] allocating noisy B (%d blocks of %d KiB) failed: %s\n", blocks,
                cols * K_DIM / 1024, e.what());
        g_job.bt.clear();
        return -2;
    }
    const double t1 = cp_now_sec();
    parallel_for(n, hw_threads(), [&](int begin, int end) {
        std::vector<int8_t> el(R_RANK), nr(K_DIM), row(K_DIM);
        for (int j = begin; j < end; ++j) {
            pearl_fuse_noise_row_b_buf(j, K_DIM, R_RANK, g_job.b_noise_seed, g_job.pairs_b.data(), NULL,
                                       row.data(), el.data(), nr.data());
            pack_row(row.data(), cols, j % cols, g_job.bt[size_t(j / cols)].data<uint8_t>());
        }
    });
    printf("[intel-npu] job setup %.3fs: B seed %.3fs, noisy B on the host %.3fs (%d blocks, salted=%d)\n",
           cp_now_sec() - t0, t_seed, cp_now_sec() - t1, blocks, g_job.salted);
    fflush(stdout);
    g_job.ready = 1;
    return 0;
}

struct Hit {
    int found = 0;
    int t_rows = -1;
    int t_cols = -1;
    uint32_t xor_words[kMilestones]{}; /* the NPU's tile words (not with the fused jackpot) */
};

/* Recompute a hit tile on the host: its A rows from signal A and B columns from the job seed, with
 * the host's noise code. A device fault can therefore never turn into an invalid share. With the fused
 * jackpot there are no NPU words to compare: the recompute must beat the target itself. */
int verify_hit(const Hit& hit, const int8_t* a_sig, const uint8_t a_seed[32], const uint32_t* pairs_a,
               const uint32_t a_key8[8], const uint32_t bound[8]) {
    std::vector<int8_t> arow(size_t(kTileRows) * K_DIM), bcol(size_t(kTileCols) * K_DIM);
    std::vector<int8_t> el(R_RANK), nr(K_DIM);
    for (int i = 0; i < kTileRows; i++)
        pearl_fuse_noise_row_a_buf(hit.t_rows + i, K_DIM, R_RANK, a_seed, pairs_a,
                                   a_sig + size_t(hit.t_rows + i) * K_DIM, arow.data() + size_t(i) * K_DIM,
                                   el.data(), nr.data());
    for (int j = 0; j < kTileCols; j++)
        pearl_fuse_noise_row_b_buf(hit.t_cols + j, K_DIM, R_RANK, g_job.b_noise_seed, g_job.pairs_b.data(),
                                   NULL, bcol.data() + size_t(j) * K_DIM, el.data(), nr.data());
    uint32_t ref[kMilestones] = {};
    for (int i = 0; i < kTileRows; i++) {
        const int8_t* a = arow.data() + size_t(i) * K_DIM;
        for (int j = 0; j < kTileCols; j++) {
            const int8_t* b = bcol.data() + size_t(j) * K_DIM;
            int32_t acc = 0;
            for (int ms = 0; ms < kMilestones; ms++) {
                for (int l = ms * R_RANK; l < (ms + 1) * R_RANK; l++)
                    acc += int32_t(a[l]) * int32_t(b[l]);
                ref[ms] ^= uint32_t(acc);
            }
        }
    }
    if (!g_eng->shape().fused_jackpot && memcmp(ref, hit.xor_words, sizeof(ref)) != 0) {
        fprintf(stderr, "[intel-npu] NPU tile XOR differs from the host recompute at t_rows=%d t_cols=%d; "
                        "share dropped\n", hit.t_rows, hit.t_cols);
        return -1;
    }
    if (!cp_jackpot::tile_beats_target(ref, kMilestones, a_key8, bound)) {
        if (g_eng->shape().fused_jackpot)
            fprintf(stderr, "[intel-npu] NPU jackpot hit at t_rows=%d t_cols=%d fails the host recompute; share "
                            "dropped\n", hit.t_rows, hit.t_cols);
        return -1;
    }
    return 0;
}

/* One attempt's scan over all launches. */
struct Scan {
    int nb = 0;       /* column blocks */
    int panels = 0;   /* row panels */
    int launches = 0;
    const int8_t* a_sig = nullptr;
    const uint8_t* a_seed = nullptr;
    const uint32_t* pairs_a = nullptr;
    const uint32_t* a_key8 = nullptr;
    const uint32_t* bound = nullptr;

    std::atomic<bool> stop{false};
    uint64_t tiles = 0;

    /* Shared with the prep thread. */
    std::mutex mu;
    std::condition_variable cv;
    int slot_panel[kPanelSlots] = {-1, -1, -1};
    std::vector<int> panel_done;
    ov::Tensor a[kPanelSlots];
    double prep_busy = 0.0;

    Hit hit;
    std::string error;
    double check_busy = 0.0;

    void halt() {
        stop.store(true);
        std::lock_guard<std::mutex> lock(mu);
        cv.notify_all();
    }
};

/* Builds the panels of noisy A in order, each into a free slot, ahead of the launches. */
void prep_panels(Scan& s, int gen_threads) {
    for (int r = 0; r < s.panels; ++r) {
        const int slot = r % kPanelSlots;
        {
            std::unique_lock<std::mutex> lock(s.mu);
            s.cv.wait(lock, [&] { return s.stop.load() || r < kPanelSlots || s.panel_done[size_t(r - kPanelSlots)] == s.nb; });
            if (s.stop.load())
                return;
        }
        const double t = cp_now_sec();
        uint8_t* dst = s.a[slot].data<uint8_t>();
        const int rows = g_eng->shape().rows;
        parallel_for(rows, gen_threads, [&](int begin, int end) {
            std::vector<int8_t> el(R_RANK), nr(K_DIM), row(K_DIM);
            for (int i = begin; i < end; ++i) {
                const int g = r * rows + i;
                pearl_fuse_noise_row_a_buf(g, K_DIM, R_RANK, s.a_seed, s.pairs_a, s.a_sig + size_t(g) * K_DIM,
                                           row.data(), el.data(), nr.data());
                pack_row(row.data(), rows, i, dst);
            }
        });
        std::lock_guard<std::mutex> lock(s.mu);
        s.prep_busy += cp_now_sec() - t;
        s.slot_panel[slot] = r;
        s.cv.notify_all();
    }
}

/* Launch l's jackpot over every tile of its output; the first hit (launches are checked in order). With
 * the fused jackpot the NPU already did it: one word per tile row, 1 + the first hit's tile column. */
void check_launch(Scan& s, ov::InferRequest& req, int l) {
    const cp_npu::LaunchShape& shape = g_eng->shape();
    const std::vector<int>& wi = g_eng->word_index();
    const int tile_rows = shape.tile_rows(), tile_cols = shape.tile_cols(), row_words = shape.row_words();
    const uint32_t* out = reinterpret_cast<const uint32_t*>(req.get_output_tensor(0).data());
    if (shape.fused_jackpot) {
        for (int tr = 0; tr < tile_rows; ++tr) {
            if (out[tr] == 0)
                continue;
            s.hit.found = 1;
            s.hit.t_rows = (l / s.nb) * shape.rows + tr * kTileRows;
            s.hit.t_cols = (l % s.nb) * shape.cols + int(out[tr] - 1) * kTileCols;
            return;
        }
        return;
    }
    uint32_t words[kMilestones];
    for (int tr = 0; tr < tile_rows; ++tr) {
        const uint32_t* row = out + size_t(tr) * row_words;
        for (int tc = 0; tc < tile_cols; ++tc) {
            const int* idx = &wi[size_t(tc) * kMilestones];
            for (int ms = 0; ms < kMilestones; ++ms)
                words[ms] = row[idx[ms]];
            if (!cp_jackpot::tile_beats_target(words, kMilestones, s.a_key8, s.bound))
                continue;
            s.hit.found = 1;
            s.hit.t_rows = (l / s.nb) * shape.rows + tr * kTileRows;
            s.hit.t_cols = (l % s.nb) * shape.cols + tc * kTileCols;
            memcpy(s.hit.xor_words, words, sizeof(words));
            return;
        }
    }
}

/* Runs the scan from the calling thread: up to one launch per infer request queued on the NPU; the
 * oldest is waited for and its jackpot checked (while the NPU runs the rest) before its request takes
 * the next launch. */
void drive_scan(Scan& s, double scan_t0, uint64_t total_tiles) {
    const int tiles_per_launch = g_eng->shape().tile_rows() * g_eng->shape().tile_cols();
    std::vector<int> idle;
    for (int i = g_eng->requests() - 1; i >= 0; --i)
        idle.push_back(i);
    std::deque<std::pair<int, int>> queued; /* (request, launch), oldest first */
    int next = 0;
    double last_report = scan_t0;
    try {
        while (!s.stop.load()) {
            if (cp_job_should_cancel()) {
                s.halt();
                break;
            }
            /* Queue the next launch once its A panel is built; wait for the panel only with nothing
             * queued (the prep thread may be waiting for a queued launch to free a panel slot). */
            if (next < s.launches && !idle.empty()) {
                const int r = next / s.nb, slot = r % kPanelSlots;
                bool ready;
                {
                    std::unique_lock<std::mutex> lock(s.mu);
                    if (queued.empty())
                        s.cv.wait(lock, [&] { return s.stop.load() || s.slot_panel[slot] == r; });
                    ready = s.slot_panel[slot] == r;
                }
                if (ready) {
                    const int i = idle.back();
                    idle.pop_back();
                    g_eng->set_inputs(i, s.a[slot], g_job.bt[size_t(next % s.nb)]);
                    g_eng->request(i).start_async();
                    queued.emplace_back(i, next++);
                    continue;
                }
            }
            if (queued.empty())
                break;
            const auto [i, l] = queued.front();
            queued.pop_front();
            ov::InferRequest& req = g_eng->request(i);
            req.wait();
            const double t = cp_now_sec();
            check_launch(s, req, l);
            s.check_busy += cp_now_sec() - t;
            idle.push_back(i);
            s.tiles += uint64_t(tiles_per_launch);
            {
                std::lock_guard<std::mutex> lock(s.mu);
                s.panel_done[size_t(l / s.nb)]++;
                s.cv.notify_all();
            }
            if (s.hit.found)
                s.halt();
            const double now = cp_now_sec();
            if (now - last_report >= 5.0) {
                char mac_buf[32];
                cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(s.tiles, now - scan_t0), mac_buf, sizeof(mac_buf));
                printf("[intel-npu] plain_proof progress: tiles %llu/%llu (%.1f%%) %s\n", (unsigned long long)s.tiles,
                       (unsigned long long)total_tiles, total_tiles ? 100.0 * double(s.tiles) / double(total_tiles) : 0.0,
                       mac_buf);
                fflush(stdout);
                last_report = now;
            }
        }
    } catch (const std::exception& e) {
        s.error = std::string("NPU inference failed: ") + e.what();
        s.halt();
    }
    /* The panels go away with the scan: let every queued launch finish first. */
    for (const auto& q : queued) {
        try {
            g_eng->request(q.first).wait();
        } catch (const std::exception&) {
        }
    }
}

} /* namespace */

extern "C" void cp_pearl_npu_worker_set_launch(int row_macros, int col_macros) {
    g_row_macros = row_macros;
    g_col_macros = col_macros;
}

extern "C" void cp_pearl_npu_worker_set_fused_jackpot(int on) {
    if (g_eng && g_eng->ready()) {
        fprintf(stderr, "[intel-npu] set_fused_jackpot ignored after init\n");
        return;
    }
    g_fused_jackpot = on != 0;
}

extern "C" void cp_pearl_npu_worker_launch_shape(int* rows, int* cols) {
    if (g_eng && g_eng->ready()) {
        *rows = g_eng->shape().rows;
        *cols = g_eng->shape().cols;
        return;
    }
    pick_launch(g_row_macros, g_col_macros, g_m_active, g_n_active, rows, cols);
}

extern "C" void cp_pearl_npu_worker_init(int* devices, int ndev) {
    (void)devices;
    (void)ndev;
    if (g_eng && g_eng->ready())
        return;
    g_requests = env_int("CP_NPU_REQUESTS", 4, 1, 16);
    cp_npu::LaunchShape shape;
    pick_launch(g_row_macros, g_col_macros, g_m_active, g_n_active, &shape.rows, &shape.cols);
    shape.chunk_m = std::min(shape.rows, kRowBlock);
    shape.k = K_DIM;
    shape.fused_jackpot = g_fused_jackpot;
    if ((g_row_macros > 0 && shape.rows != g_row_macros * kMacro) ||
        (g_col_macros > 0 && shape.cols != g_col_macros * kMacro))
        printf("[intel-npu] launch %dx%d macros requested; using %dx%d (rows a multiple of %d above it and "
               "dividing m=%d, at most %d column macros dividing n=%d)\n",
               g_row_macros > 0 ? g_row_macros : shape.rows / kMacro,
               g_col_macros > 0 ? g_col_macros : shape.cols / kMacro, shape.rows / kMacro, shape.cols / kMacro,
               kRowBlock, g_m_active, kMaxColMacros, g_n_active);
    const std::string kernel_dir = cp_npu::default_kernel_dir();
    const char* cache_env = std::getenv("CP_NPU_CACHE_DIR");
    const std::string cache_root = cache_env && *cache_env ? cache_env : cp_npu::default_cache_root();
    printf("[intel-npu] compiling the %dx%dx%d launch graph (first run of a shape takes seconds to minutes, then "
           "cached in %s)\n",
           shape.rows, shape.cols, shape.k, cache_root.c_str());
    fflush(stdout);
    g_eng = new cp_npu::Engine();
    std::string err;
    if (g_eng->init(shape, kernel_dir, cache_root, g_requests, &err) != 0) {
        fprintf(stderr, "[intel-npu] init failed: %s\n", err.c_str());
        fprintf(stderr, "[intel-npu] needs the patched in-plugin NPU compiler next to openvino.dll "
                        "(src/pearl/intel_npu/prepare_intel_npu_deps.ps1) and the TileXorAcc kernel in %s\n",
                kernel_dir.c_str());
        delete g_eng;
        g_eng = nullptr;
        return;
    }
    cp_pp_set_hash_tile(kTileRows, kTileCols);
    pearl_set_contiguous_tile_shape(kTileRows, kTileCols);
    printf("[intel-npu] %s: fused int8 GEMM + %dx%d tile XOR%s, launch %dx%d (K=%d, %d milestones), compile/load "
           "%.1fs, %d infer requests, %s inputs\n",
           g_eng->device_name().c_str(), kTileRows, kTileCols,
           shape.fused_jackpot ? " + BLAKE3 jackpot on the SHAVEs (--fused-jackpot)" : "", shape.rows, shape.cols,
           shape.k, kMilestones, g_eng->compile_sec(), g_requests, g_eng->zero_copy() ? "zero-copy" : "copied");
    printf("[intel-npu] jackpot scale tile %dx%d factor=%llu\n", cp_pp_hash_tile_h(), cp_pp_hash_tile_w(),
           (unsigned long long)cp_jackpot_scale_factor());
    fflush(stdout);
}

extern "C" void cp_pearl_npu_worker_shutdown(void) {
    g_job.ready = 0;
    g_job.bt.clear();
    g_job.pairs_b.clear();
    pearl_matrix_hash_free(g_ahash);
    g_ahash = nullptr;
    delete g_eng;
    g_eng = nullptr;
}

extern "C" int cp_pearl_npu_worker_is_ready(void) {
    return g_eng && g_eng->ready() ? 1 : 0;
}

extern "C" int cp_pearl_npu_worker_list_devices(void) {
    try {
        ov::Core core;
        int count = 0;
        for (const auto& dev : core.get_available_devices()) {
            if (dev.rfind("NPU", 0) != 0)
                continue;
            std::string name = dev;
            try {
                name = core.get_property(dev, ov::device::full_name);
            } catch (const std::exception&) {
            }
            printf("  [%d] %s (%s)\n", count++, name.c_str(), dev.c_str());
        }
        if (!count)
            printf("[intel-npu] no NPU device found by OpenVINO\n");
        return count;
    } catch (const std::exception& e) {
        fprintf(stderr, "[intel-npu] OpenVINO: %s\n", e.what());
        return 0;
    }
}

extern "C" void cp_pearl_npu_worker_begin_job(const uint8_t job_key[32], int m, int n, uint32_t cert_version) {
    g_job.ready = 0;
    g_job.salted = cert_version >= 3 ? 1 : 0;
    if (g_eng && g_eng->ready())
        prepare_job(job_key, m, n);
}

extern "C" int cp_pearl_npu_worker_mine_attempt(
        const uint8_t* ab_seed, int ab_seed_len, const uint8_t job_key[32], const uint32_t pool_tgt[8], int m, int n,
        int cpu_matrices, const int8_t* h_A_noisy, const int8_t* h_B_noisy, const uint8_t* a_key, int8_t* h_A_sig,
        int8_t* h_Bt_sig, int* out_t_rows, int* out_t_cols, uint64_t* out_tiles_scanned) {
    (void)ab_seed;
    (void)ab_seed_len;
    (void)cpu_matrices;
    (void)h_A_noisy;
    (void)h_B_noisy;
    (void)a_key;
    (void)h_Bt_sig;
    if (out_tiles_scanned)
        *out_tiles_scanned = 0;
    if (out_t_rows)
        *out_t_rows = -1;
    if (out_t_cols)
        *out_t_cols = -1;
    if (!g_eng || !g_eng->ready()) {
        fprintf(stderr, "[intel-npu] NPU not available\n");
        return -2;
    }
    if (!h_A_sig) {
        fprintf(stderr, "[intel-npu] mine_attempt requires h_Ap_global\n");
        return -2;
    }
    const double attempt_t0 = cp_now_sec();
    if (!job_matches(job_key, m, n)) {
        const int rc = prepare_job(job_key, m, n);
        if (rc != 0)
            return rc;
    }

    /* Signal A: ~K_DIM random writes per attempt (CSPRNG), then the attempt's noise seed. */
    uint8_t a_rng[32];
    if (cp_random_bytes(a_rng, sizeof(a_rng)) != 0) {
        fprintf(stderr, "[intel-npu] CSPRNG failed for random A\n");
        return -2;
    }
    std::vector<uint64_t> changed(size_t(K_DIM));
    if (pearl_perturb_random_a_one_per_col_ex(a_rng, int(sizeof(a_rng)), m, K_DIM, h_A_sig, changed.data()) != 0)
        return -1;
    /* hash_a over all of signal A, rehashing only the chunks this attempt wrote (the first attempt of
     * a job hashes everything). */
    const double t_hash = cp_now_sec();
    uint8_t hash_a[32];
    size_t rehashed = 0;
    if (!g_ahash)
        g_ahash = pearl_matrix_hash_create();
    if (!g_ahash || pearl_matrix_hash_digest(g_ahash, h_A_sig, size_t(m) * K_DIM, job_key, changed.data(),
                                             changed.size(), hash_a, &rehashed) != 0) {
        if (g_ahash)
            pearl_matrix_hash_invalidate(g_ahash);
        pearl_keyed_digest_int8(h_A_sig, size_t(m) * K_DIM, job_key, hash_a);
        rehashed = size_t(m) * K_DIM / 1024;
    }
    const double hash_sec = cp_now_sec() - t_hash;
    uint8_t a_seed[32];
    pearl_a_noise_seed_from_hash(g_job.b_noise_seed, hash_a, uint32_t(m), g_job.salted, a_seed);
    std::vector<uint32_t> pairs_a(size_t(K_DIM) * 2);
    pearl_build_perm_pairs_a(a_seed, K_DIM, R_RANK, pairs_a.data());

    uint32_t bound[8];
    cp_pp_set_hash_tile(kTileRows, kTileCols);
    cp_scale_jackpot_target(pool_tgt, bound);
    uint32_t a_key8[8];
    memcpy(a_key8, a_seed, 32);
    if (g_eng->shape().fused_jackpot) {
        /* Every row of the NPU's jackpot parameters: key, bound. No launch is in flight between attempts. */
        ov::Tensor& params = g_eng->jackpot_params();
        uint32_t* p = params.data<uint32_t>();
        for (size_t row = 0; row < params.get_size() / cp_npu::kJackpotParamWords; ++row) {
            memcpy(p + row * cp_npu::kJackpotParamWords, a_key8, sizeof(a_key8));
            memcpy(p + row * cp_npu::kJackpotParamWords + 8, bound, sizeof(bound));
        }
    }

    const int rows = g_eng->shape().rows, cols = g_eng->shape().cols;
    Scan s;
    s.nb = n / cols;
    s.panels = m / rows;
    s.launches = s.nb * s.panels;
    s.a_sig = h_A_sig;
    s.a_seed = a_seed;
    s.pairs_a = pairs_a.data();
    s.a_key8 = a_key8;
    s.bound = bound;
    s.panel_done.assign(size_t(s.panels), 0);
    try {
        for (auto& t : s.a)
            t = g_eng->host_tensor(0);
    } catch (const std::exception& e) {
        fprintf(stderr, "[intel-npu] allocating A panels failed: %s\n", e.what());
        return -2;
    }
    const uint64_t total_tiles = uint64_t(m / kTileRows) * uint64_t(n / kTileCols);
    const double prep_sec = cp_now_sec() - attempt_t0;
    printf("[intel-npu] plain_proof scan %d launches of %dx%d (%d row panels), %llu hash tiles, difficulty "
           "scaled by %llu; hash_a %.3fs (%zu of %zu chunks)\n",
           s.launches, rows, cols, s.panels, (unsigned long long)total_tiles,
           (unsigned long long)cp_jackpot_scale_factor(), hash_sec, rehashed, size_t(m) * K_DIM / 1024);
    fflush(stdout);

    const double scan_t0 = cp_now_sec();
    const int gen_threads = std::max(1, std::min(8, hw_threads() - 1));
    std::thread prep([&] { prep_panels(s, gen_threads); });
    drive_scan(s, scan_t0, total_tiles);
    s.halt(); /* releases the prep thread if the scan stopped early */
    prep.join();

    const double scan_sec = cp_now_sec() - scan_t0;
    const uint64_t tiles_done = s.tiles;
    if (out_tiles_scanned)
        *out_tiles_scanned = tiles_done;
    cp_log_attempt_timing("intel-npu", prep_sec, scan_sec, tiles_done, 0.0);
    printf("[intel-npu] scan %.3fs: host A prep %.3fs, %s %.3fs\n", scan_sec, s.prep_busy,
           g_eng->shape().fused_jackpot ? "hit check" : "jackpot", s.check_busy);
    fflush(stdout);
    if (!s.error.empty()) {
        fprintf(stderr, "[intel-npu] %s\n", s.error.c_str());
        return -2;
    }
    if (s.hit.found) {
        if (verify_hit(s.hit, h_A_sig, a_seed, pairs_a.data(), a_key8, bound) == 0) {
            printf("[intel-npu] plain_proof SHARE t_rows=%d t_cols=%d (host recompute matches NPU)\n", s.hit.t_rows,
                   s.hit.t_cols);
            fflush(stdout);
            if (out_t_rows)
                *out_t_rows = s.hit.t_rows;
            if (out_t_cols)
                *out_t_cols = s.hit.t_cols;
            return 1;
        }
    }
    if (cp_job_should_cancel())
        return -1;
    return 0;
}
