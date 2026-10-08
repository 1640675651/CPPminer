// Standalone ESIMD prototype of the Pearl scan GEMM for Intel XMX GPUs.
//
// C = A * B^T (int8 x int8 -> int32, K = 4096). Every 128 k ("milestone") the
// running C of each 16x16 hash tile is XOR-reduced to one word and folded into
// a 16-word state (state[ms % 16] = rotl(state, 13) ^ x); after the last
// milestone each tile's state is hashed with keyed BLAKE3 and compared with
// the target, all inside the GEMM thread -- no tile_xor buffer, no second pass.
//
// The noisy operands are written by the miner's own prep, so their layout is
// ours to choose: A is stored in 8x32 blocks (one DPAS A operand, 256 B) and
// B^T in 32xES VNNI blocks (one DPAS B operand), each fetched by a single
// block load.
//
//   icpx -fsycl -O3 -o esimd_gemm_bench esimd_gemm_bench.cpp
//   ./esimd_gemm_bench verify            # tile XORs vs CPU reference
//   ./esimd_gemm_bench bench [M] [N]     # TMAC/s on an MxNx4096 product
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;
using esimd::simd;

constexpr int K = 4096;
constexpr int KB = K / 32;          // 32-k DPAS steps
constexpr int MILESTONE_K = 128;
constexpr int STEPS_PER_MS = MILESTONE_K / 32;
constexpr int NUM_MS = K / MILESTONE_K;
constexpr int HT = 16;              // hash tile edge

#ifndef ES
#define ES 8                        // DPAS execution size: 8 on Xe-HPG, 16 on Xe2/Xe-HPC
#endif
#ifndef TM
#define TM 32                       // thread tile rows
#endif
#ifndef TN
#define TN 32                       // thread tile cols
#endif
#ifndef WGM
#define WGM 16                      // threads per work group along M
#endif
#ifndef WGN
#define WGN 2                       // threads per work group along N
#endif
#ifndef DPASW
#define DPASW 1                     // 1: dpasw, A shared by the fused thread pair (Xe-HPG only)
#endif
#ifndef PF
#define PF 2                        // L1 prefetch distance in 32-k steps (0 = off)
#endif

constexpr int MB = TM / 8;          // A blocks (8 rows) per thread
constexpr int NB = TN / ES;         // B blocks (ES cols) per thread
constexpr int A_BLK = 8 * 32;       // bytes per A block
constexpr int B_BLK = 32 * ES;      // bytes per B block
constexpr int C_BLK = 8 * ES;       // int32 results per DPAS
constexpr int A_LD = DPASW ? A_BLK / 2 : A_BLK; // A bytes each thread loads per block
constexpr int TILES_M = TM / HT;
constexpr int TILES_N = TN / HT;
constexpr int TILES = TILES_M * TILES_N;
static_assert(TM % HT == 0 && TN % HT == 0, "thread tile must hold whole hash tiles");

// ---------------------------------------------------------------- host layout
// A block (mb, kb): byte m*32 + k for rows 8mb.., k 32kb..; blocks ordered [mb][kb].
static void pack_a(const int8_t *a, int M, std::vector<int8_t> &out) {
    out.resize((size_t)M * K);
    for (int mb = 0; mb < M / 8; ++mb)
        for (int kb = 0; kb < KB; ++kb) {
            int8_t *dst = &out[((size_t)mb * KB + kb) * A_BLK];
            for (int m = 0; m < 8; ++m)
                std::memcpy(dst + m * 32, a + (size_t)(mb * 8 + m) * K + kb * 32, 32);
        }
}
// B^T block (nb, kb): VNNI, byte ((k/4)*ES + n)*4 + k%4; blocks ordered [nb][kb].
static void pack_bt(const int8_t *bt, int N, std::vector<int8_t> &out) {
    out.resize((size_t)N * K);
    for (int nb = 0; nb < N / ES; ++nb)
        for (int kb = 0; kb < KB; ++kb) {
            int8_t *dst = &out[((size_t)nb * KB + kb) * B_BLK];
            for (int n = 0; n < ES; ++n)
                for (int k = 0; k < 32; ++k)
                    dst[((k / 4) * ES + n) * 4 + (k % 4)] = bt[(size_t)(nb * ES + n) * K + kb * 32 + k];
        }
}

// ------------------------------------------------------------- device helpers
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
ESIMD_INLINE void b3_g(simd<uint32_t, L> *v, int a, int b, int c, int d,
                       simd<uint32_t, L> x, simd<uint32_t, L> y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr<L>(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr<L>(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr<L>(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr<L>(v[b] ^ v[c], 7);
}

// Keyed BLAKE3 compression of one 64-byte block per lane (one lane per tile),
// flags CHUNK_START|CHUNK_END|ROOT|KEYED_HASH, counter 0, block length 64.
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

// ---------------------------------------------------------------- the kernel
struct ScanArgs {
    const int8_t *a;      // packed A, M x K
    const int8_t *bt;     // packed B^T, N x K
    int M, N;
    uint32_t key[8];      // jackpot key
    uint32_t bound[8];    // scaled target, little-endian words
    int *found;           // [0] flag, [1] t_rows, [2] t_cols
    uint32_t *tile_xor;   // verify only: [ms][tile]; nullptr in scan mode
    uint32_t *digests;    // verify only: [tile][8]
};

template <bool kVerify>
void scan_kernel(const ScanArgs &p, int ti, int tj, int a_half) {
    const int m0 = ti * TM, n0 = tj * TN;
    const int8_t *a_base = p.a + (size_t)(m0 / 8) * KB * A_BLK;
    const int8_t *b_base = p.bt + (size_t)(n0 / ES) * KB * B_BLK;

    simd<int32_t, MB * NB * C_BLK> acc = 0;
    simd<uint32_t, 16 * TILES> fold = 0; // word w of all tiles at [w * TILES]

    for (int ms = 0; ms < NUM_MS; ++ms) {
#pragma unroll
        for (int s = 0; s < STEPS_PER_MS; ++s) {
            const int kb = ms * STEPS_PER_MS + s;
            if constexpr (PF > 0) {
                if (kb + PF < KB) {
                    constexpr auto pf_props = esimd::properties{
                            esimd::cache_hint_L1<esimd::cache_hint::cached>,
                            esimd::cache_hint_L2<esimd::cache_hint::cached>};
#pragma unroll
                    for (int i = 0; i < MB; ++i)
                        esimd::prefetch<uint32_t, A_BLK / 4>(
                                (const uint32_t *)(a_base + ((size_t)i * KB + kb + PF) * A_BLK), pf_props);
#pragma unroll
                    for (int j = 0; j < NB; ++j)
                        esimd::prefetch<uint32_t, B_BLK / 4>(
                                (const uint32_t *)(b_base + ((size_t)j * KB + kb + PF) * B_BLK), pf_props);
                }
            }
            simd<int8_t, MB * A_LD> av;
            simd<int8_t, NB * B_BLK> bv;
#pragma unroll
            for (int i = 0; i < MB; ++i) {
                simd<uint32_t, A_LD / 4> w = esimd::block_load<uint32_t, A_LD / 4>(
                        (const uint32_t *)(a_base + ((size_t)i * KB + kb) * A_BLK + a_half * A_LD));
                av.template select<A_LD, 1>(i * A_LD) = w.template bit_cast_view<int8_t>();
            }
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                simd<uint32_t, B_BLK / 4> w = esimd::block_load<uint32_t, B_BLK / 4>(
                        (const uint32_t *)(b_base + ((size_t)j * KB + kb) * B_BLK));
                bv.template select<B_BLK, 1>(j * B_BLK) = w.template bit_cast_view<int8_t>();
            }
#pragma unroll
            for (int i = 0; i < MB; ++i)
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    auto c = acc.template select<C_BLK, 1>((i * NB + j) * C_BLK);
#if DPASW
                    c = xmx::dpasw<8, 8, int32_t>(simd<int32_t, C_BLK>(c),
                                                  simd<int8_t, B_BLK>(bv.template select<B_BLK, 1>(j * B_BLK)),
                                                  simd<int8_t, A_LD>(av.template select<A_LD, 1>(i * A_LD)));
#else
                    c = xmx::dpas<8, 8, int32_t>(simd<int32_t, C_BLK>(c),
                                                 simd<int8_t, B_BLK>(bv.template select<B_BLK, 1>(j * B_BLK)),
                                                 simd<int8_t, A_LD>(av.template select<A_LD, 1>(i * A_LD)));
#endif
                }
        }
        // Milestone: XOR every 16x16 hash tile of the running C to one word.
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
        if constexpr (kVerify) {
            const int tiles_n = p.N / HT;
            const size_t tile_count = (size_t)(p.M / HT) * tiles_n;
            for (int th = 0; th < TILES_M; ++th)
                for (int tw = 0; tw < TILES_N; ++tw) {
                    const size_t tile = (size_t)(m0 / HT + th) * tiles_n + (n0 / HT + tw);
                    simd<uint32_t, 1> one = x[th * TILES_N + tw];
                    esimd::scatter<uint32_t, 1>(p.tile_xor + (size_t)ms * tile_count + tile,
                                                simd<uint32_t, 1>(0), one);
                }
        }
        auto f = fold.template select<TILES, 1>((ms % 16) * TILES);
        f = ((simd<uint32_t, TILES>(f) << 13) | (simd<uint32_t, TILES>(f) >> 19)) ^ x;
    }

    // Jackpot: keyed BLAKE3 over each tile's 16 folded words, lanes = tiles.
    simd<uint32_t, TILES> msg[16], digest[8];
#pragma unroll
    for (int w = 0; w < 16; ++w) msg[w] = fold.template select<TILES, 1>(w * TILES);
    b3_compress<TILES>(p.key, msg, digest);
    if constexpr (kVerify) {
        const int tiles_n = p.N / HT;
        for (int t = 0; t < TILES; ++t) {
            const size_t tile = (size_t)(m0 / HT + t / TILES_N) * tiles_n + (n0 / HT + t % TILES_N);
            simd<uint32_t, 8> d;
            for (int w = 0; w < 8; ++w) d[w] = digest[w][t];
            esimd::block_store<uint32_t, 8>(p.digests + tile * 8, d);
        }
    }
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
            simd<int, 4> hit(0);
            hit[0] = 1;
            hit[1] = m0 + (t / TILES_N) * HT;
            hit[2] = n0 + (t % TILES_N) * HT;
            esimd::block_store<int, 4>(p.found, hit); // benign race: any winner is valid
            break;
        }
    }
}

template <bool kVerify>
sycl::event launch(sycl::queue &q, const ScanArgs &args) {
    const size_t gm = args.M / TM, gn = args.N / TN;
    sycl::nd_range<2> r({gm, gn}, {WGM, WGN});
    syclex::properties props{intelex::grf_size<256>};
    return q.parallel_for(r, props, [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        // Consecutive threads (a fused pair) share M and differ in N, so with
        // dpasw each loads half of the common A block.
        const int lin = (int)it.get_local_linear_id();
        const int tn = lin % WGN, tm = lin / WGN;
        // Default linear group order (N fastest) measured best; band swizzles
        // of 2-32 group rows were 17-66% slower on A380.
        const int ti = (int)it.get_group(0) * WGM + tm, tj = (int)it.get_group(1) * WGN + tn;
        scan_kernel<kVerify>(args, ti, tj, DPASW ? (lin & 1) : 0);
    });
}

// ---------------------------------------------------------------- host checks
static void fill(std::vector<int8_t> &v, int lo, int hi, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> d(lo, hi);
    for (auto &x : v) x = (int8_t)d(g);
}

static uint32_t rotr_h(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void b3_ref(const uint32_t key[8], const uint32_t m[16], uint32_t out[8]) {
    static const uint8_t P[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    uint32_t v[16] = {key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                      0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0, 0, 64, 0x1B};
    uint32_t w[16];
    std::memcpy(w, m, sizeof(w));
    auto g = [&](int a, int b, int c, int d, uint32_t x, uint32_t y) {
        v[a] += v[b] + x; v[d] = rotr_h(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = rotr_h(v[b] ^ v[c], 12);
        v[a] += v[b] + y; v[d] = rotr_h(v[d] ^ v[a], 8); v[c] += v[d]; v[b] = rotr_h(v[b] ^ v[c], 7);
    };
    for (int r = 0; r < 7; ++r) {
        g(0, 4, 8, 12, w[0], w[1]); g(1, 5, 9, 13, w[2], w[3]); g(2, 6, 10, 14, w[4], w[5]);
        g(3, 7, 11, 15, w[6], w[7]); g(0, 5, 10, 15, w[8], w[9]); g(1, 6, 11, 12, w[10], w[11]);
        g(2, 7, 8, 13, w[12], w[13]); g(3, 4, 9, 14, w[14], w[15]);
        uint32_t t[16];
        for (int i = 0; i < 16; ++i) t[i] = w[P[i]];
        std::memcpy(w, t, sizeof(w));
    }
    for (int i = 0; i < 8; ++i) out[i] = v[i] ^ v[i + 8];
}

static int verify(sycl::queue &q) {
    const int M = TM * WGM * 2, N = TN * WGN * 2;
    std::vector<int8_t> a((size_t)M * K), bt((size_t)N * K), ap, bp;
    fill(a, -127, 126, 1);
    fill(bt, -63, 63, 2);
    pack_a(a.data(), M, ap);
    pack_bt(bt.data(), N, bp);
    const size_t tiles = (size_t)(M / HT) * (N / HT);
    auto *da = sycl::malloc_device<int8_t>(ap.size(), q);
    auto *db = sycl::malloc_device<int8_t>(bp.size(), q);
    auto *dx = sycl::malloc_device<uint32_t>(tiles * NUM_MS, q);
    auto *df = sycl::malloc_device<int>(4, q);
    auto *dd = sycl::malloc_device<uint32_t>(tiles * 8, q);
    q.memcpy(da, ap.data(), ap.size()).wait();
    q.memcpy(db, bp.data(), bp.size()).wait();
    q.memset(df, 0, 4 * sizeof(int)).wait();
    const uint32_t key[8] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u,
                             0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u};
    ScanArgs args{da, db, M, N, {}, {}, df, dx, dd};
    std::memcpy(args.key, key, sizeof(key));
    launch<true>(q, args).wait();
    std::vector<uint32_t> got(tiles * NUM_MS), dig(tiles * 8);
    q.memcpy(got.data(), dx, got.size() * 4).wait();
    q.memcpy(dig.data(), dd, dig.size() * 4).wait();

    // CPU reference: running C per milestone, XOR per 16x16 tile.
    std::vector<int32_t> c((size_t)M * N, 0);
    size_t bad = 0;
    for (int ms = 0; ms < NUM_MS; ++ms) {
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                int32_t s = c[(size_t)m * N + n];
                const int8_t *ar = &a[(size_t)m * K + ms * MILESTONE_K];
                const int8_t *br = &bt[(size_t)n * K + ms * MILESTONE_K];
                for (int k = 0; k < MILESTONE_K; ++k) s += (int32_t)ar[k] * br[k];
                c[(size_t)m * N + n] = s;
            }
        for (int tr = 0; tr < M / HT; ++tr)
            for (int tc = 0; tc < N / HT; ++tc) {
                uint32_t x = 0;
                for (int i = 0; i < HT; ++i)
                    for (int j = 0; j < HT; ++j) x ^= (uint32_t)c[(size_t)(tr * HT + i) * N + tc * HT + j];
                if (x != got[(size_t)ms * tiles + (size_t)tr * (N / HT) + tc]) ++bad;
            }
    }
    // Fold the reference milestone words and hash each tile.
    size_t bad_digest = 0;
    for (size_t t = 0; t < tiles; ++t) {
        uint32_t msg[16] = {}, d[8];
        for (int ms = 0; ms < NUM_MS; ++ms) {
            uint32_t &s = msg[ms % 16];
            s = ((s << 13) | (s >> 19)) ^ got[(size_t)ms * tiles + t];
        }
        b3_ref(key, msg, d);
        if (std::memcmp(d, &dig[t * 8], sizeof(d)) != 0) ++bad_digest;
    }
    std::printf("verify %dx%dx%d ES=%d tile %dx%d dpasw %d: %zu / %zu milestone tile words differ, "
                "%zu / %zu digests differ\n", M, N, K, ES, TM, TN, DPASW, bad, tiles * NUM_MS,
                bad_digest, tiles);
    sycl::free(da, q); sycl::free(db, q); sycl::free(dx, q); sycl::free(df, q); sycl::free(dd, q);
    return (bad || bad_digest) ? 1 : 0;
}

static int bench(sycl::queue &q, int M, int N) {
    std::vector<int8_t> ap((size_t)M * K), bp((size_t)N * K);
    fill(ap, -127, 126, 3); // already "packed": layout does not matter for timing
    fill(bp, -63, 63, 4);
    auto *da = sycl::malloc_device<int8_t>(ap.size(), q);
    auto *db = sycl::malloc_device<int8_t>(bp.size(), q);
    auto *df = sycl::malloc_device<int>(4, q);
    q.memcpy(da, ap.data(), ap.size()).wait();
    q.memcpy(db, bp.data(), bp.size()).wait();
    q.memset(df, 0, 4 * sizeof(int)).wait();
    ScanArgs args{da, db, M, N, {1, 2, 3, 4, 5, 6, 7, 8}, {}, df, nullptr, nullptr}; // bound 0: no hits
    launch<false>(q, args).wait();
    const int reps = 5;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) launch<false>(q, args);
    q.wait();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double macs = (double)M * N * K * reps;
    std::printf("bench %dx%dx%d ES=%d tile %dx%d wg %dx%d pf %d dpasw %d: %.2f TMAC/s (%.1f ms/launch)\n",
                M, N, K, ES, TM, TN, WGM, WGN, PF, DPASW, macs / sec / 1e12, sec / reps * 1e3);
    sycl::free(da, q); sycl::free(db, q); sycl::free(df, q);
    return 0;
}

int main(int argc, char **argv) {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const std::string mode = argc > 1 ? argv[1] : "verify";
    if (mode == "verify") return verify(q);
    const int M = argc > 2 ? std::atoi(argv[2]) : 16384;
    const int N = argc > 3 ? std::atoi(argv[3]) : 16384;
    return bench(q, M, N);
}
