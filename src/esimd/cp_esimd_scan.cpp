// libcp_esimd: ESIMD XMX scan kernel for the miner (see cp_esimd_scan.h and
// esimd_gemm_bench.cpp, the standalone prototype it was measured with).
//
// Runs on the miner's OpenCL context/queue through SYCL interop, so the
// OpenCL prep that writes the blocked operands and the scan stay ordered on
// one in-order queue.
//
//   icpx -fsycl -O3 -fPIC -shared -o libcp_esimd.so cp_esimd_scan.cpp -lOpenCL \
//        -Wl,--no-as-needed -L<oneapi>/umf/latest/lib -lumf -Wl,-rpath,<that dir>
// libumf is linked directly so the Unified Runtime adapters, which libsycl
// loads while initializing, find it without oneAPI's setvars.sh.
#include "cp_esimd_scan.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;
using esimd::simd;

namespace {

constexpr int K = CP_ESIMD_K;
constexpr int KB = K / 32;
constexpr int MILESTONE_K = 128;
constexpr int STEPS_PER_MS = MILESTONE_K / 32;
constexpr int NUM_MS = K / MILESTONE_K;
constexpr int HT = CP_ESIMD_HASH_TILE;
constexpr int TM = 32, TN = 32; // thread tile
constexpr int WGM = 16, WGN = 2; // threads per work group (A380 sweep)
constexpr int PF = 2;            // L1 prefetch distance, 32-k steps

template <int N>
ESIMD_INLINE uint32_t xor_reduce(simd<uint32_t, N> v) {
    if constexpr (N == 1) {
        return v[0];
    } else {
        simd<uint32_t, N / 2> h = v.template select<N / 2, 1>(0) ^ v.template select<N / 2, 1>(N / 2);
        return xor_reduce<N / 2>(h);
    }
}

template <int L>
ESIMD_INLINE simd<uint32_t, L> rotr(simd<uint32_t, L> x, int n) {
    return (x >> n) | (x << (32 - n));
}

template <int L>
ESIMD_INLINE void b3_g(simd<uint32_t, L> *v, int a, int b, int c, int d, simd<uint32_t, L> x,
                       simd<uint32_t, L> y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr<L>(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr<L>(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr<L>(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr<L>(v[b] ^ v[c], 7);
}

// Keyed BLAKE3 of one 64-byte block per lane: CHUNK_START|CHUNK_END|ROOT|KEYED_HASH.
template <int L>
ESIMD_INLINE void b3_compress(const uint32_t key[8], simd<uint32_t, L> *m, simd<uint32_t, L> *out) {
    constexpr uint8_t sched[7][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
        {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
        {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
        {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
        {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
        {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
    };
    simd<uint32_t, L> v[16];
    for (int i = 0; i < 8; ++i) v[i] = key[i];
    v[8] = 0x6A09E667u; v[9] = 0xBB67AE85u; v[10] = 0x3C6EF372u; v[11] = 0xA54FF53Au;
    v[12] = 0u; v[13] = 0u; v[14] = 64u; v[15] = 0x1Bu;
#pragma unroll
    for (int r = 0; r < 7; ++r) {
        b3_g<L>(v, 0, 4, 8, 12, m[sched[r][0]], m[sched[r][1]]);
        b3_g<L>(v, 1, 5, 9, 13, m[sched[r][2]], m[sched[r][3]]);
        b3_g<L>(v, 2, 6, 10, 14, m[sched[r][4]], m[sched[r][5]]);
        b3_g<L>(v, 3, 7, 11, 15, m[sched[r][6]], m[sched[r][7]]);
        b3_g<L>(v, 0, 5, 10, 15, m[sched[r][8]], m[sched[r][9]]);
        b3_g<L>(v, 1, 6, 11, 12, m[sched[r][10]], m[sched[r][11]]);
        b3_g<L>(v, 2, 7, 8, 13, m[sched[r][12]], m[sched[r][13]]);
        b3_g<L>(v, 3, 4, 9, 14, m[sched[r][14]], m[sched[r][15]]);
    }
    for (int i = 0; i < 8; ++i) out[i] = v[i] ^ v[i + 8];
}

// Block loads/prefetches move at most 64 dwords per message; split larger ones.
template <int N, typename Acc>
ESIMD_INLINE simd<uint32_t, N> load_u32(const Acc &acc, uint32_t off) {
    if constexpr (N <= 64) {
        return esimd::block_load<uint32_t, N>(acc, off);
    } else {
        simd<uint32_t, N> r;
        r.template select<64, 1>(0) = esimd::block_load<uint32_t, 64>(acc, off);
        r.template select<N - 64, 1>(64) = load_u32<N - 64>(acc, off + 256);
        return r;
    }
}

template <int N, typename Acc, typename Props>
ESIMD_INLINE void prefetch_u32(const Acc &acc, uint32_t off, Props props) {
    if constexpr (N <= 64) {
        esimd::prefetch<uint32_t, N>(acc, off, props);
    } else {
        esimd::prefetch<uint32_t, 64>(acc, off, props);
        prefetch_u32<N - 64>(acc, off + 256, props);
    }
}

struct PanelArgs {
    int m0, n0;
    uint32_t key[8];
    uint32_t bound[8];
};

// ES: DPAS execution size (8 on Xe-HPG, 16 on Xe2/Xe-HPC). DPASW: A operand
// shared by the fused thread pair (Xe-HPG only); the pair must share rows.
template <int ES, bool DPASW, typename AccA, typename AccB, typename AccF>
ESIMD_INLINE void scan_thread(const AccA &a, const AccB &b, const AccF &found, const PanelArgs &p,
                              int ti, int tj, int a_half) {
    constexpr int MB = TM / 8, NB = TN / ES;
    constexpr int A_BLK = 8 * 32, B_BLK = 32 * ES, C_BLK = 8 * ES;
    constexpr int A_LD = DPASW ? A_BLK / 2 : A_BLK;
    constexpr int TILES_M = TM / HT, TILES_N = TN / HT, TILES = TILES_M * TILES_N;
    const int m0 = p.m0 + ti * TM, n0 = p.n0 + tj * TN;
    const uint32_t a_base = (uint32_t)(m0 / 8) * KB * A_BLK;
    const uint32_t b_base = (uint32_t)(n0 / ES) * KB * B_BLK;
    constexpr auto pf_props = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>,
                                                esimd::cache_hint_L2<esimd::cache_hint::cached>};

    simd<int32_t, MB * NB * C_BLK> acc = 0;
    simd<uint32_t, 16 * TILES> fold = 0; // word w of all tiles at [w * TILES]

    for (int ms = 0; ms < NUM_MS; ++ms) {
#pragma unroll
        for (int s = 0; s < STEPS_PER_MS; ++s) {
            const int kb = ms * STEPS_PER_MS + s;
            if (kb + PF < KB) {
#pragma unroll
                for (int i = 0; i < MB; ++i)
                    prefetch_u32<A_LD / 4>(a, a_base + ((uint32_t)i * KB + kb + PF) * A_BLK + a_half * A_LD,
                                           pf_props);
#pragma unroll
                for (int j = 0; j < NB; ++j)
                    prefetch_u32<B_BLK / 4>(b, b_base + ((uint32_t)j * KB + kb + PF) * B_BLK, pf_props);
            }
            simd<int8_t, MB * A_LD> av;
            simd<int8_t, NB * B_BLK> bv;
#pragma unroll
            for (int i = 0; i < MB; ++i) {
                simd<uint32_t, A_LD / 4> w =
                        load_u32<A_LD / 4>(a, a_base + ((uint32_t)i * KB + kb) * A_BLK + a_half * A_LD);
                av.template select<A_LD, 1>(i * A_LD) = w.template bit_cast_view<int8_t>();
            }
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                simd<uint32_t, B_BLK / 4> w = load_u32<B_BLK / 4>(b, b_base + ((uint32_t)j * KB + kb) * B_BLK);
                bv.template select<B_BLK, 1>(j * B_BLK) = w.template bit_cast_view<int8_t>();
            }
#pragma unroll
            for (int i = 0; i < MB; ++i)
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    auto c = acc.template select<C_BLK, 1>((i * NB + j) * C_BLK);
                    simd<int8_t, B_BLK> bj = bv.template select<B_BLK, 1>(j * B_BLK);
                    simd<int8_t, A_LD> ai = av.template select<A_LD, 1>(i * A_LD);
                    if constexpr (DPASW)
                        c = xmx::dpasw<8, 8, int32_t>(simd<int32_t, C_BLK>(c), bj, ai);
                    else
                        c = xmx::dpas<8, 8, int32_t>(simd<int32_t, C_BLK>(c), bj, ai);
                }
        }
        simd<uint32_t, TILES> x;
#pragma unroll
        for (int th = 0; th < TILES_M; ++th)
#pragma unroll
            for (int tw = 0; tw < TILES_N; ++tw) {
                simd<uint32_t, C_BLK> t = 0;
#pragma unroll
                for (int bi = 0; bi < HT / 8; ++bi)
#pragma unroll
                    for (int bj = 0; bj < HT / ES; ++bj) {
                        const int i = th * (HT / 8) + bi, j = tw * (HT / ES) + bj;
                        simd<int32_t, C_BLK> cb = acc.template select<C_BLK, 1>((i * NB + j) * C_BLK);
                        t ^= cb.template bit_cast_view<uint32_t>();
                    }
                x[th * TILES_N + tw] = xor_reduce<C_BLK>(t);
            }
        auto f = fold.template select<TILES, 1>((ms % 16) * TILES);
        f = ((simd<uint32_t, TILES>(f) << 13) | (simd<uint32_t, TILES>(f) >> 19)) ^ x;
    }

    simd<uint32_t, TILES> msg[16], digest[8];
#pragma unroll
    for (int w = 0; w < 16; ++w) msg[w] = fold.template select<TILES, 1>(w * TILES);
    b3_compress<TILES>(p.key, msg, digest);
    for (int t = 0; t < TILES; ++t) {
        bool beats = true; // equal to the bound counts as a hit
        for (int w = 7; w >= 0; --w) {
            const uint32_t d = digest[w][t];
            if (d != p.bound[w]) {
                beats = d < p.bound[w];
                break;
            }
        }
        if (beats) {
            // First hit wins the flag; only it writes coordinates, so rows and
            // columns of two different winning tiles can never be mixed.
            simd<uint32_t, 1> flag_off = 0;
            simd<int, 1> old = esimd::atomic_update<esimd::atomic_op::cmpxchg, int, 1>(
                    found, flag_off, simd<int, 1>(1), simd<int, 1>(0));
            if (old[0] == 0) {
                simd<uint32_t, 2> off(4, 4); // byte offsets 4, 8
                simd<int, 2> rc;
                rc[0] = m0 + (t / TILES_N) * HT;
                rc[1] = n0 + (t % TILES_N) * HT;
                esimd::scatter<int, 2>(found, off, rc);
            }
            break;
        }
    }
}

using ByteBuf = sycl::buffer<char, 1>;

} // namespace

struct CpEsimdScan {
    // Built from the miner's OpenCL objects; never default-constructed (that would
    // run SYCL's own device selection).
    CpEsimdScan(cl_context c, cl_command_queue cq)
        : ctx(sycl::make_context<sycl::backend::opencl>(c)),
          q(sycl::make_queue<sycl::backend::opencl>(cq, ctx)) {}
    sycl::context ctx;
    sycl::queue q;
    CpEsimdInfo info{};
    cl_mem a_mem = nullptr, b_mem = nullptr, f_mem = nullptr;
    std::unique_ptr<ByteBuf> a_buf, b_buf, f_buf;

    ByteBuf &wrap(cl_mem mem, cl_mem &cached, std::unique_ptr<ByteBuf> &buf) {
        if (!buf || cached != mem) {
            buf.reset(new ByteBuf(sycl::make_buffer<sycl::backend::opencl, char>(mem, ctx)));
            cached = mem;
        }
        return *buf;
    }
};

extern "C" {

__attribute__((visibility("default"))) int cp_esimd_abi_version(void) {
    return CP_ESIMD_ABI_VERSION;
}

__attribute__((visibility("default"))) CpEsimdScan *cp_esimd_create(void *cl_ctx, void *cl_dev,
                                                                     void *cl_queue, CpEsimdInfo *info,
                                                                     char *err, int err_len) {
    auto fail = [&](const char *msg) -> CpEsimdScan * {
        if (err && err_len > 0) std::snprintf(err, (size_t)err_len, "%s", msg);
        return nullptr;
    };
    try {
        auto s = std::make_unique<CpEsimdScan>((cl_context)cl_ctx, (cl_command_queue)cl_queue);
        const sycl::device dev = s->q.get_device();
        (void)cl_dev;
        if (!dev.has(sycl::aspect::ext_intel_matrix)) return fail("device has no XMX (ext_intel_matrix)");
        const auto sg = dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t min_sg = 1024;
        for (size_t v : sg) min_sg = v < min_sg ? v : min_sg;
        s->info.exec_size = min_sg <= 8 ? 8 : 16;
        s->info.dpasw = s->info.exec_size == 8 ? 1 : 0;
        s->info.tile_m = TM * WGM;
        s->info.tile_n = TN * WGN;
        if (info) *info = s->info;
        return s.release();
    } catch (const std::exception &e) {
        return fail(e.what());
    }
}

__attribute__((visibility("default"))) int cp_esimd_scan_panel(CpEsimdScan *s, void *a_blocked,
                                                               void *bt_vnni, void *found, int m0,
                                                               int n0, int m, int n,
                                                               const uint32_t key[8],
                                                               const uint32_t bound[8],
                                                               void **done) {
    if (!s || m % s->info.tile_m || n % s->info.tile_n) return -1;
    PanelArgs args{m0, n0, {}, {}};
    std::memcpy(args.key, key, sizeof(args.key));
    std::memcpy(args.bound, bound, sizeof(args.bound));
    try {
        ByteBuf &ab = s->wrap((cl_mem)a_blocked, s->a_mem, s->a_buf);
        ByteBuf &bb = s->wrap((cl_mem)bt_vnni, s->b_mem, s->b_buf);
        ByteBuf &fb = s->wrap((cl_mem)found, s->f_mem, s->f_buf);
        const sycl::nd_range<2> r({(size_t)(m / TM), (size_t)(n / TN)}, {WGM, WGN});
        const int es = s->info.exec_size;
        sycl::event ev = s->q.submit([&](sycl::handler &h) {
            auto a = ab.get_access<sycl::access::mode::read>(h);
            auto b = bb.get_access<sycl::access::mode::read>(h);
            auto f = fb.get_access<sycl::access::mode::read_write>(h);
            syclex::properties props{intelex::grf_size<256>};
            if (es == 8) {
                h.parallel_for(r, props, [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                    // A fused pair (consecutive ids) shares rows: dpasw splits A between them.
                    const int lin = (int)it.get_local_linear_id();
                    const int tn = lin % WGN, tm = lin / WGN;
                    scan_thread<8, true>(a, b, f, args, (int)it.get_group(0) * WGM + tm,
                                         (int)it.get_group(1) * WGN + tn, lin & 1);
                });
            } else {
                h.parallel_for(r, props, [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                    const int lin = (int)it.get_local_linear_id();
                    const int tn = lin % WGN, tm = lin / WGN;
                    scan_thread<16, false>(a, b, f, args, (int)it.get_group(0) * WGM + tm,
                                           (int)it.get_group(1) * WGN + tn, 0);
                });
            }
        });
        if (done) {
            *done = nullptr;
            auto natives = sycl::get_native<sycl::backend::opencl>(ev);
            if (!natives.empty()) {
                clRetainEvent(natives.back());
                *done = natives.back();
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[esimd] scan panel failed: %s\n", e.what());
        return -2;
    }
}

__attribute__((visibility("default"))) void cp_esimd_wait(CpEsimdScan *s) {
    if (!s) return;
    try {
        s->q.wait();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[esimd] wait failed: %s\n", e.what());
    }
}

__attribute__((visibility("default"))) void cp_esimd_destroy(CpEsimdScan *s) {
    if (!s) return;
    try {
        s->q.wait();
    } catch (...) {
    }
    delete s;
}

} // extern "C"
