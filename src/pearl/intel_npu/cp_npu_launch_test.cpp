/*
 * cp_npu_launch_test: checks one Intel NPU launch (rows x cols x 4096 tile XOR, default 512 x 512)
 * against a host reference on random int8 data, then measures launches per second with several
 * requests in flight. --fused checks the fused jackpot instead: a random key and a loose bound (about
 * 1 tile in 32 beats it), each tile row's first hit against the host jackpot of the reference words.
 * --profile compiles with PERF_COUNT and prints the summed per-layer time of one launch by engine and
 * op type (NPU_PRINT_PROFILING=JSON NPU_PROFILING_OUTPUT_FILE=<file> also writes the plugin's trace).
 *   cp_npu_launch_test [--rows R] [--cols C] [--group G] [--shared-b] [--fused] [--profile]
 *                      [--launches N] [--requests R] [--kernel-dir DIR] [--cache-dir DIR]
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "cp_jackpot.hpp"
#include "cp_npu_engine.hpp"

using namespace cp_npu;

namespace {

/* [K/128, X, 128] u8 (value + 128) from X rows of K int8 values. */
void pack_panels(const int8_t* rows, int x, int k, uint8_t* dst) {
    for (int r = 0; r < x; ++r)
        for (int p = 0; p < k / kKr; ++p)
            for (int i = 0; i < kKr; ++i)
                dst[(size_t(p) * x + r) * kKr + i] = uint8_t(rows[size_t(r) * k + size_t(p) * kKr + i] ^ 0x80);
}

/* Tile words [milestone][tile row][tile col] of A (rows x k) times B^T (cols x k). */
std::vector<uint32_t> reference(const std::vector<int8_t>& a, const std::vector<int8_t>& bt, const LaunchShape& s) {
    const int ms = s.milestones(), tr_n = s.tile_rows(), tc_n = s.tile_cols();
    std::vector<uint32_t> out(size_t(ms) * tr_n * tc_n);
    std::atomic<int> next{0};
    auto worker = [&] {
        for (int tr; (tr = next++) < tr_n;)
            for (int tc = 0; tc < tc_n; ++tc) {
                int32_t acc[kTileRows][kTileCols] = {};
                for (int p = 0; p < ms; ++p) {
                    uint32_t x = 0;
                    for (int r = 0; r < kTileRows; ++r) {
                        const int8_t* ar = &a[size_t(tr * kTileRows + r) * s.k + size_t(p) * kKr];
                        for (int c = 0; c < kTileCols; ++c) {
                            const int8_t* br = &bt[size_t(tc * kTileCols + c) * s.k + size_t(p) * kKr];
                            int32_t d = 0;
                            for (int i = 0; i < kKr; ++i)
                                d += int32_t(ar[i]) * int32_t(br[i]);
                            acc[r][c] += d;
                            x ^= uint32_t(acc[r][c]);
                        }
                    }
                    out[(size_t(p) * tr_n + tr) * tc_n + tc] = x;
                }
            }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t)
        pool.emplace_back(worker);
    for (auto& t : pool)
        t.join();
    return out;
}

} // namespace

int main(int argc, char** argv) {
    int launches = 400, requests = 4;
    bool profile = false;
    LaunchShape s;
    std::string kernel_dir = default_kernel_dir(), cache_root = default_cache_root();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc)
            s.rows = std::atoi(argv[++i]);
        else if (a == "--cols" && i + 1 < argc)
            s.cols = std::atoi(argv[++i]);
        else if (a == "--group" && i + 1 < argc)
            s.group = std::atoi(argv[++i]);
        else if (a == "--shared-b")
            s.split_b = false;
        else if (a == "--fused")
            s.fused_jackpot = true;
        else if (a == "--profile")
            profile = true;
        else if (a == "--launches" && i + 1 < argc)
            launches = std::atoi(argv[++i]);
        else if (a == "--requests" && i + 1 < argc)
            requests = std::atoi(argv[++i]);
        else if (a == "--kernel-dir" && i + 1 < argc)
            kernel_dir = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc)
            cache_root = argv[++i];
        else {
            std::fprintf(stderr, "usage: cp_npu_launch_test [--rows R] [--cols C] [--group G] [--shared-b] [--fused] "
                                 "[--profile] [--launches N] [--requests R] [--kernel-dir DIR] [--cache-dir DIR]\n");
            return 2;
        }
    }
    s.chunk_m = std::min(s.rows, 512);
    if (const char* why = launch_shape_error(s)) {
        std::fprintf(stderr, "launch %dx%d: %s\n", s.rows, s.cols, why);
        return 2;
    }
    Engine eng;
    std::string err;
    const ov::AnyMap extra = profile ? ov::AnyMap{{"PERF_COUNT", true}} : ov::AnyMap{};
    if (eng.init(s, kernel_dir, cache_root, requests, &err, extra) != 0) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("device %s, launch %dx%dx%d (%d row blocks, group %d, %s B%s), compile/load %.1f s, cache %s, %s "
                "host tensors\n",
                eng.device_name().c_str(), s.rows, s.cols, s.k, s.row_blocks(), s.group,
                s.split_b ? "split" : "shared", s.fused_jackpot ? ", fused jackpot" : "", eng.compile_sec(),
                eng.cache_dir().c_str(), eng.zero_copy() ? "zero-copy" : "plain");

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> a(size_t(s.rows) * s.k), bt(size_t(s.cols) * s.k);
    for (auto& v : a)
        v = int8_t(dist(rng));
    for (auto& v : bt)
        v = int8_t(dist(rng));
    ov::Tensor ta = eng.host_tensor(0), tb = eng.host_tensor(1);
    pack_panels(a.data(), s.rows, s.k, ta.data<uint8_t>());
    pack_panels(bt.data(), s.cols, s.k, tb.data<uint8_t>());

    uint32_t key[8], bound[8];
    if (s.fused_jackpot) {
        for (int i = 0; i < 8; ++i) {
            key[i] = uint32_t(rng());
            bound[i] = uint32_t(rng());
        }
        bound[7] = 0x08000000u; /* digest word 7 below it in about 1 tile of 32 */
        uint32_t* p = eng.jackpot_params().data<uint32_t>();
        for (size_t row = 0; row < eng.jackpot_params().get_size() / kJackpotParamWords; ++row) {
            std::memcpy(p + row * kJackpotParamWords, key, sizeof(key));
            std::memcpy(p + row * kJackpotParamWords + 8, bound, sizeof(bound));
        }
    }

    ov::InferRequest& req = eng.request(0);
    eng.set_inputs(0, ta, tb);
    req.infer();
    const auto* out = reinterpret_cast<const uint32_t*>(req.get_output_tensor(0).data());
    const auto ref = reference(a, bt, s);
    const auto& wi = eng.word_index();
    const int ms = s.milestones(), tr_n = s.tile_rows(), tc_n = s.tile_cols();
    size_t bad = 0;
    if (s.fused_jackpot) {
        /* Each tile row: 1 + the first tile column the host jackpot accepts, or 0. */
        int hit_rows = 0;
        for (int tr = 0; tr < tr_n; ++tr) {
            uint32_t expect = 0;
            for (int tc = 0; tc < tc_n && !expect; ++tc) {
                uint32_t words[64];
                for (int p = 0; p < ms; ++p)
                    words[p] = ref[(size_t(p) * tr_n + tr) * tc_n + tc];
                if (cp_jackpot::tile_beats_target(words, ms, key, bound))
                    expect = uint32_t(tc + 1);
            }
            hit_rows += expect != 0;
            if (out[tr] != expect) {
                if (bad < 5)
                    std::printf("  tile row %d: NPU %u, host %u\n", tr, out[tr], expect);
                ++bad;
            }
        }
        std::printf("correctness: %s (%zu of %d tile rows differ; %d rows with a hit)\n", bad ? "FAIL" : "PASS", bad,
                    tr_n, hit_rows);
    } else {
        for (int tr = 0; tr < tr_n; ++tr)
            for (int tc = 0; tc < tc_n; ++tc)
                for (int p = 0; p < ms; ++p)
                    if (out[size_t(tr) * s.row_words() + wi[size_t(tc * ms + p)]] != ref[(size_t(p) * tr_n + tr) * tc_n + tc])
                        ++bad;
        std::printf("correctness: %s (%zu of %zu tile words differ)\n", bad ? "FAIL" : "PASS", bad, ref.size());
    }
    if (bad)
        return 1;

    if (profile) {
        /* One more launch; per (engine, op type) the summed real time of its layers. */
        req.infer();
        std::vector<std::pair<std::string, double>> sums;
        for (const auto& info : req.get_profiling_info()) {
            const std::string key = info.exec_type + " " + info.node_type;
            const double us = double(info.real_time.count());
            auto it = std::find_if(sums.begin(), sums.end(), [&](const auto& e) { return e.first == key; });
            if (it == sums.end())
                sums.emplace_back(key, us);
            else
                it->second += us;
        }
        std::sort(sums.begin(), sums.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        std::printf("profile (summed per-layer real time of one launch):\n");
        for (const auto& e : sums)
            std::printf("  %-40s %10.1f us\n", e.first.c_str(), e.second);
    }

    /* Throughput: one thread keeps every request queued, resubmitting each as it completes (in order). */
    const int nreq = eng.requests();
    for (int i = 0; i < nreq; ++i)
        eng.set_inputs(i, ta, tb);
    const auto t0 = std::chrono::steady_clock::now();
    int submitted = 0;
    for (; submitted < std::min(nreq, launches); ++submitted)
        eng.request(submitted).start_async();
    for (int done = 0; done < launches; ++done) {
        ov::InferRequest& r = eng.request(done % nreq);
        r.wait();
        if (submitted < launches) {
            r.start_async();
            ++submitted;
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double ops = 2.0 * s.rows * s.cols * s.k * launches;
    std::printf("throughput: %d launches in %.3f s, %.3f ms per launch, %.3f TOPS (%d requests)\n", launches,
                sec, sec / launches * 1e3, ops / sec / 1e12, eng.requests());
    return 0;
}
