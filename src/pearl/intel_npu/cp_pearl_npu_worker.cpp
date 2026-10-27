#include "cp_pearl_npu_worker.h"

#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_npu_engine.hpp"
#include "cp_util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/*
 * Intel NPU worker (zero-B, like the Hexagon and CPU workers). The host generates and packs all the
 * noise; the NPU runs the fused GEMM + tile XOR; the host keeps signal A, its hash, the seeds and the
 * BLAKE3 jackpot.
 *   job:     noisy B (signal B = 0) built on the host and packed per 512-column block into tensors the
 *            NPU reads in place.
 *   attempt: sparse random signal A on the host -> a_noise_seed and permutation pairs; then the matrix
 *            is scanned in launches of 512 rows x 512 columns x K (one compiled graph, cp_npu_graph.cpp).
 *            A prep thread builds each 512-row panel of noisy A ahead of its launches; each of the
 *            lanes (one infer request and one host thread each, 4 by default) runs launches back to
 *            back and checks the jackpot of its own output while the NPU runs the other lanes'.
 * Hash tile: 8 rows x 16 columns (proof layout CP_TILE_LAYOUT_CONTIGUOUS). A hit is recomputed on the
 * host from the seeds before it can become a share.
 */

namespace {

using cp_npu::kKr;
using cp_npu::kTileCols;
using cp_npu::kTileRows;

constexpr int kMilestones = K_DIM / R_RANK;
static_assert(R_RANK == cp_npu::kKr, "the NPU graph takes one milestone per 128 values of k");
constexpr int kLaunchRows = 512;
constexpr int kLaunchCols = 512;
constexpr int kPanelSlots = 3; /* A panels in flight: the one being scanned, the next, one spare */

cp_npu::Engine* g_eng = nullptr;
int g_lanes = 4;
PearlMatrixHash* g_ahash = nullptr;

struct JobCache {
    uint8_t job_key[32]{};
    int m = 0;
    int n = 0;
    int salted = 0;
    int ready = 0;
    uint8_t b_noise_seed[32]{};
    std::vector<uint32_t> pairs_b;
    std::vector<ov::Tensor> bt; /* n / 512 blocks of B^T rows, [K/128, 512, 128] u8 (value + 128) */
} g_job;

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
    if (m % kLaunchRows != 0 || n % kLaunchCols != 0) {
        fprintf(stderr, "[intel-npu] m and n must be multiples of %d (got %dx%d)\n", kLaunchCols, m, n);
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

    const int blocks = n / kLaunchCols;
    try {
        if ((int)g_job.bt.size() != blocks) {
            g_job.bt.clear();
            for (int b = 0; b < blocks; ++b)
                g_job.bt.push_back(g_eng->host_tensor(1));
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[intel-npu] allocating noisy B (%d blocks of 2 MiB) failed: %s\n", blocks, e.what());
        g_job.bt.clear();
        return -2;
    }
    const double t1 = cp_now_sec();
    parallel_for(n, hw_threads(), [&](int begin, int end) {
        std::vector<int8_t> el(R_RANK), nr(K_DIM), row(K_DIM);
        for (int j = begin; j < end; ++j) {
            pearl_fuse_noise_row_b_buf(j, K_DIM, R_RANK, g_job.b_noise_seed, g_job.pairs_b.data(), NULL,
                                       row.data(), el.data(), nr.data());
            pack_row(row.data(), kLaunchCols, j % kLaunchCols, g_job.bt[size_t(j / kLaunchCols)].data<uint8_t>());
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
    int64_t key = 0; /* launch * tiles per launch + tile: the lowest wins, for a deterministic pick */
    int t_rows = -1;
    int t_cols = -1;
    uint32_t xor_words[kMilestones]{};
};

/* Recompute a hit tile on the host: its A rows from signal A and B columns from the job seed, with
 * the host's noise code. A device fault can therefore never turn into an invalid share. */
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
    if (memcmp(ref, hit.xor_words, sizeof(ref)) != 0) {
        fprintf(stderr, "[intel-npu] NPU tile XOR differs from the host recompute at t_rows=%d t_cols=%d; "
                        "share dropped\n", hit.t_rows, hit.t_cols);
        return -1;
    }
    return cp_jackpot::tile_beats_target(ref, kMilestones, a_key8, bound) ? 0 : -1;
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

    std::atomic<int> next{0};
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> tiles{0};
    std::atomic<int> lanes_left{0};
    std::atomic<bool> found{false};

    std::mutex mu;
    std::condition_variable cv;
    int slot_panel[kPanelSlots] = {-1, -1, -1};
    std::vector<int> panel_done;
    ov::Tensor a[kPanelSlots];

    std::mutex hit_mu;
    Hit hit;
    std::string error;
    double prep_busy = 0.0, check_busy = 0.0;

    void fail(const std::string& msg) {
        {
            std::lock_guard<std::mutex> lock(hit_mu);
            if (error.empty())
                error = msg;
        }
        halt();
    }
    void halt() {
        stop.store(true);
        std::lock_guard<std::mutex> lock(mu);
        cv.notify_all();
    }
};

/* Builds the panels of noisy A in order, each into a free slot, ahead of the lanes. */
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
        parallel_for(kLaunchRows, gen_threads, [&](int begin, int end) {
            std::vector<int8_t> el(R_RANK), nr(K_DIM), row(K_DIM);
            for (int i = begin; i < end; ++i) {
                const int g = r * kLaunchRows + i;
                pearl_fuse_noise_row_a_buf(g, K_DIM, R_RANK, s.a_seed, s.pairs_a, s.a_sig + size_t(g) * K_DIM,
                                           row.data(), el.data(), nr.data());
                pack_row(row.data(), kLaunchRows, i, dst);
            }
        });
        std::lock_guard<std::mutex> lock(s.mu);
        s.prep_busy += cp_now_sec() - t;
        s.slot_panel[slot] = r;
        s.cv.notify_all();
    }
}

/* One lane: launches back to back on its own infer request, checking each output's jackpot. */
void run_lane(Scan& s, int lane) {
    ov::InferRequest& req = g_eng->request(lane);
    const cp_npu::LaunchShape& shape = g_eng->shape();
    const std::vector<int>& wi = g_eng->word_index();
    const int tile_rows = shape.tile_rows(), tile_cols = shape.tile_cols(), row_words = shape.row_words();
    const int tiles_per_launch = tile_rows * tile_cols;
    double check = 0.0;
    for (;;) {
        if (s.stop.load())
            break;
        const int l = s.next.fetch_add(1);
        if (l >= s.launches)
            break;
        const int r = l / s.nb, cb = l % s.nb, slot = r % kPanelSlots;
        {
            std::unique_lock<std::mutex> lock(s.mu);
            s.cv.wait(lock, [&] { return s.stop.load() || s.slot_panel[slot] == r; });
            if (s.stop.load())
                break;
        }
        try {
            req.set_input_tensor(0, s.a[slot]);
            req.set_input_tensor(1, g_job.bt[size_t(cb)]);
            req.infer();
        } catch (const std::exception& e) {
            s.fail(std::string("NPU inference failed: ") + e.what());
            break;
        }
        const double t = cp_now_sec();
        const uint32_t* out = reinterpret_cast<const uint32_t*>(req.get_output_tensor(0).data());
        uint32_t words[kMilestones];
        for (int tr = 0; tr < tile_rows; ++tr) {
            const uint32_t* row = out + size_t(tr) * row_words;
            for (int tc = 0; tc < tile_cols; ++tc) {
                const int* idx = &wi[size_t(tc) * kMilestones];
                for (int ms = 0; ms < kMilestones; ++ms)
                    words[ms] = row[idx[ms]];
                if (!cp_jackpot::tile_beats_target(words, kMilestones, s.a_key8, s.bound))
                    continue;
                const int64_t key = int64_t(l) * tiles_per_launch + tr * tile_cols + tc;
                std::lock_guard<std::mutex> lock(s.hit_mu);
                if (!s.hit.found || key < s.hit.key) {
                    s.hit.found = 1;
                    s.hit.key = key;
                    s.hit.t_rows = r * kLaunchRows + tr * kTileRows;
                    s.hit.t_cols = cb * kLaunchCols + tc * kTileCols;
                    memcpy(s.hit.xor_words, words, sizeof(words));
                    s.found.store(true);
                }
            }
        }
        check += cp_now_sec() - t;
        s.tiles.fetch_add(uint64_t(tiles_per_launch));
        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.panel_done[size_t(r)]++;
            s.cv.notify_all();
        }
        if (s.found.load())
            s.halt();
    }
    std::lock_guard<std::mutex> lock(s.hit_mu);
    s.check_busy += check;
    s.lanes_left.fetch_sub(1);
}

} /* namespace */

extern "C" void cp_pearl_npu_worker_init(int* devices, int ndev) {
    (void)devices;
    (void)ndev;
    if (g_eng && g_eng->ready())
        return;
    g_lanes = env_int("CP_NPU_LANES", 4, 1, 16);
    cp_npu::LaunchShape shape;
    shape.rows = kLaunchRows;
    shape.cols = kLaunchCols;
    shape.k = K_DIM;
    const std::string kernel_dir = cp_npu::default_kernel_dir();
    const char* cache_env = std::getenv("CP_NPU_CACHE_DIR");
    const std::string cache_root = cache_env && *cache_env ? cache_env : cp_npu::default_cache_root();
    printf("[intel-npu] compiling the %dx%dx%d launch graph (first run about 10 s, then cached in %s)\n",
           shape.rows, shape.cols, shape.k, cache_root.c_str());
    fflush(stdout);
    g_eng = new cp_npu::Engine();
    std::string err;
    if (g_eng->init(shape, kernel_dir, cache_root, g_lanes, &err) != 0) {
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
    printf("[intel-npu] %s: fused int8 GEMM + 8x16 tile XOR, launch %dx%d (K=%d, %d milestones), compile/load "
           "%.1fs, %d lanes, %s inputs\n",
           g_eng->device_name().c_str(), shape.rows, shape.cols, shape.k, kMilestones, g_eng->compile_sec(), g_lanes,
           g_eng->zero_copy() ? "zero-copy" : "copied");
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

    Scan s;
    s.nb = n / kLaunchCols;
    s.panels = m / kLaunchRows;
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
           s.launches, kLaunchRows, kLaunchCols, s.panels, (unsigned long long)total_tiles,
           (unsigned long long)cp_jackpot_scale_factor(), hash_sec, rehashed, size_t(m) * K_DIM / 1024);
    fflush(stdout);

    const double scan_t0 = cp_now_sec();
    const int gen_threads = std::max(1, std::min(8, hw_threads() - g_lanes));
    s.lanes_left.store(g_lanes);
    std::thread prep([&] { prep_panels(s, gen_threads); });
    std::vector<std::thread> lanes;
    for (int i = 0; i < g_lanes; ++i)
        lanes.emplace_back([&, i] { run_lane(s, i); });

    double last_report = scan_t0;
    while (s.lanes_left.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (cp_job_should_cancel())
            s.halt();
        const double now = cp_now_sec();
        if (now - last_report >= 5.0) {
            const uint64_t cur = s.tiles.load();
            char mac_buf[32];
            cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(cur, now - scan_t0), mac_buf, sizeof(mac_buf));
            printf("[intel-npu] plain_proof progress: tiles %llu/%llu (%.1f%%) %s\n", (unsigned long long)cur,
                   (unsigned long long)total_tiles, total_tiles ? 100.0 * double(cur) / double(total_tiles) : 0.0,
                   mac_buf);
            fflush(stdout);
            last_report = now;
        }
    }
    for (auto& t : lanes)
        t.join();
    s.halt(); /* releases the prep thread if the lanes stopped early */
    prep.join();

    const double scan_sec = cp_now_sec() - scan_t0;
    const uint64_t tiles_done = s.tiles.load();
    if (out_tiles_scanned)
        *out_tiles_scanned = tiles_done;
    cp_log_attempt_timing("intel-npu", prep_sec, scan_sec, tiles_done, 0.0);
    printf("[intel-npu] scan %.3fs: host A prep %.3fs, jackpot %.3fs (summed over %d lanes)\n", scan_sec,
           s.prep_busy, s.check_busy, g_lanes);
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
