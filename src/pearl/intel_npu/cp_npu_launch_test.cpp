/*
 * cp_npu_launch_test: checks one Intel NPU launch (rows x cols x 4096 tile XOR, default 512 x 512)
 * against a host reference on random int8 data, then measures launches per second with several
 * requests in flight.
 *   cp_npu_launch_test [--rows R] [--cols C] [--group G] [--shared-b] [--launches N] [--requests R]
 *                      [--kernel-dir DIR] [--cache-dir DIR]
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
        else if (a == "--launches" && i + 1 < argc)
            launches = std::atoi(argv[++i]);
        else if (a == "--requests" && i + 1 < argc)
            requests = std::atoi(argv[++i]);
        else if (a == "--kernel-dir" && i + 1 < argc)
            kernel_dir = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc)
            cache_root = argv[++i];
        else {
            std::fprintf(stderr, "usage: cp_npu_launch_test [--rows R] [--cols C] [--group G] [--shared-b] "
                                 "[--launches N] [--requests R] [--kernel-dir DIR] [--cache-dir DIR]\n");
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
    if (eng.init(s, kernel_dir, cache_root, requests, &err) != 0) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("device %s, launch %dx%dx%d (%d row blocks, group %d, %s B), compile/load %.1f s, cache %s, %s host "
                "tensors\n",
                eng.device_name().c_str(), s.rows, s.cols, s.k, s.row_blocks(), s.group,
                s.split_b ? "split" : "shared", eng.compile_sec(), eng.cache_dir().c_str(),
                eng.zero_copy() ? "zero-copy" : "plain");

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

    ov::InferRequest& req = eng.request(0);
    eng.set_inputs(0, ta, tb);
    req.infer();
    const auto* out = reinterpret_cast<const uint32_t*>(req.get_output_tensor(0).data());
    const auto ref = reference(a, bt, s);
    const auto& wi = eng.word_index();
    const int ms = s.milestones(), tr_n = s.tile_rows(), tc_n = s.tile_cols();
    size_t bad = 0;
    for (int tr = 0; tr < tr_n; ++tr)
        for (int tc = 0; tc < tc_n; ++tc)
            for (int p = 0; p < ms; ++p)
                if (out[size_t(tr) * s.row_words() + wi[size_t(tc * ms + p)]] != ref[(size_t(p) * tr_n + tr) * tc_n + tc])
                    ++bad;
    std::printf("correctness: %s (%zu of %zu tile words differ)\n", bad ? "FAIL" : "PASS", bad, ref.size());
    if (bad)
        return 1;

    /* Throughput: one host thread per request, each running launches back to back. */
    std::atomic<int> next{0};
    std::vector<std::thread> lanes;
    const auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < eng.requests(); ++l)
        lanes.emplace_back([&, l] {
            ov::InferRequest& r = eng.request(l);
            eng.set_inputs(l, ta, tb);
            while (next++ < launches)
                r.infer();
        });
    for (auto& t : lanes)
        t.join();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double ops = 2.0 * s.rows * s.cols * s.k * launches;
    std::printf("throughput: %d launches in %.3f s, %.3f ms per launch, %.3f TOPS (%d requests)\n", launches,
                sec, sec / launches * 1e3, ops / sec / 1e12, eng.requests());
    return 0;
}
