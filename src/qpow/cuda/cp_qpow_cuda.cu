/* Quantus QPoW (Poseidon2 over Goldilocks, width 12) mining kernel and CUDA worker.
 *
 * One thread hashes one nonce. Inside a launch the nonces form the line B + t*v (see
 * qpow/nonce_line.hpp), along which only lanes 0 and 4 change after the first linear layer: the host
 * folds the midstate, B, the first linear layer and the other ten S-boxes of the first round
 * into Params, and the kernel runs two S-boxes before the rest of both permutations. Only the
 * first output element is computed after the last S-box layer: it holds the first 8 hash bytes,
 * which decide nearly every comparison. A nonce whose first 8 bytes are <= the target's becomes a candidate and is
 * re-hashed on the host with the reference code before it is reported.
 *
 * Field elements are any 64-bit value (not necessarily < p); the arithmetic is mod p.
 */
#include "cp_qpow_cuda_worker.h"

#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "qpow/nonce_line.hpp"
#include "qpow/poseidon2.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

typedef uint32_t u32;
typedef uint64_t u64;
#define DEV __device__ __forceinline__

constexpr int k_tpb = 256;
constexpr int k_max_candidates = 15;

/* ------------------------------------------------------------------ field arithmetic */

DEV u64 mk64(u32 lo, u32 hi) { u64 r; asm("mov.b64 %0, {%1,%2};" : "=l"(r) : "r"(lo), "r"(hi)); return r; }
DEV void sp64(u64 x, u32& lo, u32& hi) { asm("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(x)); }

/* EPS = 2^32 - 1 = 2^64 mod p, read from constant memory: as a literal, ptxas turns
 * hi(r2 * EPS) into IMAD.HI, which is slow on Turing; as a constant-bank operand it stays one
 * wide multiply-add with carry. */
__constant__ u32 c_eps;

/* (r3:r2:r1:r0) -> 64-bit value congruent mod p.
 * V = lo + r2*2^64 + r3*2^96 = lo + r2*EPS - r3 (mod p). t = lo + r2*EPS with carry c; when c
 * is set, 2^64 = EPS turns -r3 into +EPS - r3 = c*2^32 - (r3 + c), which cannot overflow.
 * Exact except when r2 == 0 and lo < r3 without a carry (probability ~2^-64 for hash data); a
 * miss there only changes the candidate filter, and every candidate is re-hashed on the host.
 * Precondition r3 <= 2^32 - 2: true for any product, and for a*b + c when b < 2^64 - 2^32
 * (the internal-layer diagonal is below 0xF3FB...). */
DEV u64 red128(u32 r0, u32 r1, u32 r2, u32 r3)
{
    u32 z0, z1;
    asm("{.reg .u32 t0, t1, rc, tc;\n\t"
        "mad.lo.cc.u32  t0, %2, %6, %3;\n\t"
        "madc.hi.cc.u32 t1, %2, %6, %4;\n\t"
        "addc.u32       rc, %5, 0;\n\t"
        "addc.u32       tc, t1, 0;\n\t"
        "sub.cc.u32     %0, t0, rc;\n\t"
        "subc.u32       %1, tc, 0;}"
        : "=r"(z0), "=r"(z1) : "r"(r2), "r"(r0), "r"(r1), "r"(r3), "r"(c_eps));
    return mk64(z0, z1);
}

/* a*b as 128 bits: a0*b0, a1*b1, a0*b1 as wide products, a1*b0 accumulated onto a0*b1. */
DEV void mul128(u64 a, u64 b, u32& r0, u32& r1, u32& r2, u32& r3)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u64 P, Q, T;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(P) : "r"(a0), "r"(b0));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(b1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(Q) : "r"(a1), "r"(b1));
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    asm("{.reg .u32 m0, m1, mc;\n\t"
        "mad.lo.cc.u32  m0, %3, %4, %5;\n\t"
        "madc.hi.cc.u32 m1, %3, %4, %6;\n\t"
        "addc.u32       mc, 0, 0;\n\t"
        "add.cc.u32     %0, %7, m0;\n\t"
        "addc.cc.u32    %1, %8, m1;\n\t"
        "addc.u32       %2, %9, mc;}"
        : "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a1), "r"(b0), "r"(t0), "r"(t1), "r"(p1), "r"(q0), "r"(q1));
    r0 = p0;
}

/* CP_QPOW_ADD64: the lazy sums and the squaring's 128-bit add as 64-bit add.cc/addc chains
 * (what a 128-bit integer lowers to) instead of 32-bit ones. ptxas then merges some adds into
 * 3-input IADD3s with two carries and folds the doubled middle product of the square into
 * IMAD.WIDE.U32.X chains, which it never does for 32-bit add.cc chains. RTX 2080 Ti (sm_75):
 * -5% SM cycles per hash with four blocks per SM. The sm_8x SASS shows the same trade (fewer
 * ALU ops, more IMADs). RTX 5070 (sm_120, driver JIT from PTX as shipped): +8..11% cycles, the
 * adds become IADD.64 plus extra moves, so Blackwell keeps the 32-bit chains. Not a 128-bit C
 * integer: MSVC has none, and the Windows build parses this file on the host side too. */
#ifndef CP_QPOW_ADD64
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define CP_QPOW_ADD64 0
#else
#define CP_QPOW_ADD64 1
#endif
#endif

/* a^2 from three wide products: a0^2 + 2*a0*a1*2^32 + a1^2*2^64. Without CP_QPOW_ADD64 the
 * middle product is doubled by a one-bit funnel shift, which leaves a single 3-word add. */
DEV void sqr128(u64 a, u32& r0, u32& r1, u32& r2, u32& r3)
{
    u32 a0, a1; sp64(a, a0, a1);
    u64 P, Q, T;
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(P) : "r"(a0));
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(Q) : "r"(a1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(a1));
#if CP_QPOW_ADD64
    u64 L, H;
    asm("{.reg .u64 s, h;\n\t"
        "shl.b64    s, %2, 33;\n\t"
        "shr.u64    h, %2, 31;\n\t"
        "add.cc.u64 %0, %3, s;\n\t"
        "addc.u64   %1, %4, h;}"
        : "=l"(L), "=l"(H) : "l"(T), "l"(P), "l"(Q));
    sp64(L, r0, r1); sp64(H, r2, r3);
    return;
#endif
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    u32 d0, d1, d2;
    asm("shl.b32 %0, %1, 1;" : "=r"(d0) : "r"(t0));
    asm("shf.l.wrap.b32 %0, %1, %2, 1;" : "=r"(d1) : "r"(t0), "r"(t1));
    asm("shr.b32 %0, %1, 31;" : "=r"(d2) : "r"(t1));
    asm("add.cc.u32  %0, %3, %6;\n\t"
        "addc.cc.u32 %1, %4, %7;\n\t"
        "addc.u32    %2, %5, %8;"
        : "=r"(r1), "=r"(r2), "=r"(r3) : "r"(p1), "r"(q0), "r"(q1), "r"(d0), "r"(d1), "r"(d2));
    r0 = p0;
}

DEV u64 gmul(u64 a, u64 b) { u32 r0, r1, r2, r3; mul128(a, b, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }
DEV u64 gsqr(u64 a) { u32 r0, r1, r2, r3; sqr128(a, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }

/* Lazy sum for the linear layers: value = (hi:lo) + t*2^64. */
#if CP_QPOW_ADD64
struct W { u64 v; u32 t; };
#define W_LO(w) ((u32)(w).v)
#define W_HI(w) ((u32)((w).v >> 32))
#define W_T(w) ((w).t)
DEV W w2(u64 a, u64 b)
{
    W r;
    asm("add.cc.u64 %0, %2, %3;\n\t addc.u32 %1, 0, 0;" : "=l"(r.v), "=r"(r.t) : "l"(a), "l"(b));
    return r;
}
DEV W wadd(W a, W b)
{
    W r;
    asm("add.cc.u64 %0, %2, %4;\n\t addc.u32 %1, %3, %5;" : "=l"(r.v), "=r"(r.t) : "l"(a.v), "r"(a.t), "l"(b.v), "r"(b.t));
    return r;
}
DEV W wadd64(W a, u64 b)
{
    W r;
    asm("add.cc.u64 %0, %2, %4;\n\t addc.u32 %1, %3, 0;" : "=l"(r.v), "=r"(r.t) : "l"(a.v), "r"(a.t), "l"(b));
    return r;
}
#else
struct W { u32 lo, hi, t; };
#define W_LO(w) ((w).lo)
#define W_HI(w) ((w).hi)
#define W_T(w) ((w).t)

DEV W w2(u64 a, u64 b)
{
    W r; u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    asm("add.cc.u32 %0, %3, %5;\n\t addc.cc.u32 %1, %4, %6;\n\t addc.u32 %2, 0, 0;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t) : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    return r;
}
DEV W wadd(W a, W b)
{
    W r;
    asm("add.cc.u32 %0, %3, %6;\n\t addc.cc.u32 %1, %4, %7;\n\t addc.u32 %2, %5, %8;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t)
        : "r"(a.lo), "r"(a.hi), "r"(a.t), "r"(b.lo), "r"(b.hi), "r"(b.t));
    return r;
}
DEV W wadd64(W a, u64 b)
{
    W r; u32 b0, b1; sp64(b, b0, b1);
    asm("add.cc.u32 %0, %3, %6;\n\t addc.cc.u32 %1, %4, %7;\n\t addc.u32 %2, %5, 0;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t) : "r"(a.lo), "r"(a.hi), "r"(a.t), "r"(b0), "r"(b1));
    return r;
}
#endif
/* (hi:lo) + t*2^64 with t < 2^31 -> 64 bits. w = t*EPS + lo < 2^63; adding hi*2^32 carries at
 * most once, and then the high word is < 2^31, so the +EPS fold cannot carry again. */
DEV u64 wred(W a)
{
    u32 w0, w1, c;
    asm("mad.lo.cc.u32  %0, %3, %6, %4;\n\t"
        "madc.hi.u32    %1, %3, %6, 0;\n\t"
        "add.cc.u32     %1, %1, %5;\n\t"
        "addc.u32       %2, 0, 0;\n\t"
        : "=r"(w0), "=r"(w1), "=r"(c) : "r"(W_T(a)), "r"(W_LO(a)), "r"(W_HI(a)), "r"(c_eps));
    u32 n = 0u - c;
    asm("add.cc.u32 %0, %0, %2;\n\t addc.u32 %1, %1, 0;" : "+r"(w0), "+r"(w1) : "r"(n));
    return mk64(w0, w1);
}

DEV u64 gadd(u64 a, u64 b) { return wred(w2(a, b)); }

DEV u64 canon(u64 a) { return a >= 0xFFFFFFFF00000001ull ? a - 0xFFFFFFFF00000001ull : a; }

DEV u64 sbox(u64 x)
{
    u64 x2 = gsqr(x);
    u64 x3 = gmul(x2, x);
    u64 x4 = gsqr(x2);
    return gmul(x3, x4);
}

/* ------------------------------------------------------------------ permutation */

/* Row g (0..14): what is added after the linear layer that follows full round g of the two
 * permutations (next round constants, RC_INTERNAL[0] on s0 before the internal rounds, and the
 * +1/+1 of the second absorb after g == 7); row 15 is RC_INITIAL[0] for the second
 * permutation's first linear layer. */
__constant__ u64 c_post[16][12];
__constant__ u64 c_rci[22];   /* RC_INTERNAL[1..21], 0 */
__constant__ u64 c_rct0[12];  /* RC_TERMINAL[0] */
__constant__ u64 c_diag[12];

struct Params {
    u64 pre0, pre4; /* lanes 0 and 4 of ext(midstate + B) + RC_INITIAL[0] (t = 0) */
    u64 K[12];      /* state after the first full round with lanes 0 and 4 left out */
    u64 t0;         /* first 8 target bytes, big-endian */
    u64* dump;      /* self-test only: canonical first output element per nonce */
    u32* out;       /* [0] = candidate count, [1..15] = nonce index */
    u32 tb;         /* t of nonce index 0 */
    u32 count;
};

DEV void mat4(u64 x0, u64 x1, u64 x2, u64 x3, W& y0, W& y1, W& y2, W& y3)
{
    W t01 = w2(x0, x1), t23 = w2(x2, x3);
    W t0123 = wadd(t01, t23);
    W t01123 = wadd64(t0123, x1);
    W t01233 = wadd64(t0123, x3);
    y3 = wadd(t01233, w2(x0, x0));
    y1 = wadd(t01123, w2(x2, x2));
    y0 = wadd(t01123, t01);
    y2 = wadd(t01233, t23);
}

/* s = M_ext * s + add, M_ext = circ(2*M4, M4, M4). */
DEV void ext_add(u64 s[12], const u64* add)
{
    W y[12];
#pragma unroll
    for (int k = 0; k < 3; k++)
        mat4(s[4 * k], s[4 * k + 1], s[4 * k + 2], s[4 * k + 3], y[4 * k], y[4 * k + 1], y[4 * k + 2], y[4 * k + 3]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        W sum = wadd(wadd(y[j], y[4 + j]), y[8 + j]);
#pragma unroll
        for (int k = 0; k < 3; k++) s[4 * k + j] = wred(wadd64(wadd(y[4 * k + j], sum), add[4 * k + j]));
    }
}

/* a*b + w for a lazy 96-bit sum w: w enters the product's carry chain unreduced.
 * a*b + w < 2^128 - 2^96 for b below 0xF3FB... (the internal diagonal), so red128's
 * r3 <= 2^32 - 2 precondition holds. The cross products a0*b1 + a1*b0 form their own chain on
 * a natural register pair and join the even chain (a0*b0 + w, a1*b1) at word 1: ptxas would
 * otherwise assemble unaligned addend pairs with MOVs for every lane. */
DEV u64 gfma_w(u64 a, u64 b, W w)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u32 r0, r1, r2, r3, x0, x1, x2;
    asm("mul.lo.u32     %0, %3, %5;\n\t"
        "mul.hi.u32     %1, %3, %5;\n\t"
        "mad.lo.cc.u32  %0, %4, %6, %0;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %1;\n\t"
        "addc.u32       %2, 0, 0;"
        : "=r"(x0), "=r"(x1), "=r"(x2) : "r"(a0), "r"(a1), "r"(b1), "r"(b0));
    asm("mad.lo.cc.u32  %0, %4, %6, %8;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
        "madc.lo.cc.u32 %2, %5, %7, %10;\n\t"
        "madc.hi.u32    %3, %5, %7, 0;\n\t"
        "add.cc.u32     %1, %1, %11;\n\t"
        "addc.cc.u32    %2, %2, %12;\n\t"
        "addc.u32       %3, %3, %13;"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(W_LO(w)), "r"(W_HI(w)), "r"(W_T(w)), "r"(x0), "r"(x1), "r"(x2));
    return red128(r0, r1, r2, r3);
}

/* 22 internal rounds; s[0] arrives with RC_INTERNAL[0] added, leaves with RC_TERMINAL[0].
 * The row sum stays a lazy 96-bit value and goes straight into each diagonal product; lanes
 * 1..11 are summed before the S-box so that work overlaps its multiply chain. */
DEV void internal22(u64 s[12])
{
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
        W rest = w2(s[1], s[2]);
#pragma unroll
        for (int i = 3; i < 12; i++) rest = wadd64(rest, s[i]);
        s[0] = sbox(s[0]);
        const W sg = wadd64(rest, s[0]);
        s[0] = gfma_w(s[0], c_diag[0], wadd64(sg, c_rci[r]));
#pragma unroll
        for (int i = 1; i < 12; i++) s[i] = gfma_w(s[i], c_diag[i], sg);
    }
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = gadd(s[i], c_rct0[i]);
}

/* Canonical first element of the second squeeze-permutation output for the nonce B + t*v.
 * After the absorb and the first linear layer only lane 0 (-35t) and lane 4 (+35t) depend on t,
 * so the first full round runs two S-boxes; the other ten are folded into K by the host.
 * Columns 0 and 4 of M_ext are m0 * (2,1,1) and m0 * (1,2,1) over the three blocks, with
 * m0 = M4 column 0 = (2,1,1,3). */
DEV u64 hash_out0(const Params& p, u32 t)
{
    u64 s[12];
    {
        const u64 d = 35ull * t;
        const u64 f = sbox(gadd(p.pre0, 0xFFFFFFFF00000001ull - d));
        const u64 g = sbox(gadd(p.pre4, d));
        const W c2 = w2(f, g);
        const W c[3] = {wadd64(c2, f), wadd64(c2, g), c2};
        const int m0[4] = {2, 1, 1, 3};
#pragma unroll
        for (int k = 0; k < 3; k++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                W v = c[k];
                if (m0[j] >= 2) v = wadd(v, c[k]);
                if (m0[j] == 3) v = wadd(v, c[k]);
                s[4 * k + j] = wred(wadd64(v, p.K[4 * k + j]));
            }
    }
#pragma unroll 1
    for (int g = 1;; g++) {
#pragma unroll
        for (int i = 0; i < 12; i++) s[i] = sbox(s[i]);
        if (g == 15) break;
#pragma unroll 1
        for (int e = 0; e < 1 + (g == 7); e++) ext_add(s, c_post[e ? 15 : g]);
        if ((g & 7) == 3) internal22(s);
    }
    /* out0 = (M_ext s)[0] = 2*y00 + y10 + y20, y_k0 = 2x0 + 3x1 + x2 + x3 of chunk k */
    W acc = w2(s[0], s[0]);
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int m = k == 0 ? 2 : 1;
#pragma unroll
        for (int rep = 0; rep < m; rep++) {
            if (!(k == 0 && rep == 0)) { acc = wadd64(acc, s[4 * k]); acc = wadd64(acc, s[4 * k]); }
            acc = wadd64(acc, s[4 * k + 1]); acc = wadd64(acc, s[4 * k + 1]); acc = wadd64(acc, s[4 * k + 1]);
            acc = wadd64(acc, s[4 * k + 2]); acc = wadd64(acc, s[4 * k + 3]);
        }
    }
    return canon(wred(acc));
}

/* sm_75: four blocks per SM (64 registers) measured 0.8% fewer cycles per hash than three. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
#define CP_QPOW_MIN_BLOCKS 4
#else
#define CP_QPOW_MIN_BLOCKS 1
#endif
__global__ void __launch_bounds__(k_tpb, CP_QPOW_MIN_BLOCKS) qpow_cuda_scan(const Params p)
{
    const u32 stride = gridDim.x * blockDim.x;
    for (u32 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p.count; idx += stride) {
        const u64 o = hash_out0(p, p.tb + idx);
        u32 l, h; sp64(o, l, h);
        const u64 key = mk64(__byte_perm(h, 0, 0x0123), __byte_perm(l, 0, 0x0123));
        if (p.dump) p.dump[idx] = o;
        if (key <= p.t0) {
            const u32 slot = atomicAdd(p.out, 1u);
            if (slot < k_max_candidates) p.out[1 + slot] = idx;
        }
    }
}

/* ------------------------------------------------------------------ host side */

struct Dev {
    int ordinal = -1;
    int sms = 0;
    int blocks = 0;          /* resident blocks for the whole grid */
    uint32_t launch = 0;     /* nonces per launch */
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    u32* d_out = nullptr;
    u32* h_out = nullptr;    /* pinned */
    /* per-step state */
    uint8_t base[64];
    uint32_t n = 0;
};

std::vector<Dev> g_devs;
uint32_t g_batch_req = 0;
int g_ready = 0;
uint8_t g_found_nonce[64]; /* last reported nonce and the search counter it came from */
uint8_t g_found_ctr[64];

bool ck(cudaError_t e, const char* what, int ordinal)
{
    if (e == cudaSuccess) return true;
    fprintf(stderr, "[qpow-cuda] device %d: %s: %s\n", ordinal, what, cudaGetErrorString(e));
    return false;
}

void be_add(uint8_t n[64], u64 add)
{
    for (int i = 63; i >= 0 && add; i--) {
        add += n[i];
        n[i] = (uint8_t)add;
        add >>= 8;
    }
}

bool upload_constants()
{
    u64 post[16][12] = {};
    for (int g = 0; g < 15; g++) {
        const int gg = g & 7;
        if (gg <= 2) memcpy(post[g], qpow::RC_INITIAL[gg + 1], sizeof(post[g]));
        else if (gg == 3) post[g][0] = qpow::RC_INTERNAL[0];
        else if (gg <= 6) memcpy(post[g], qpow::RC_TERMINAL[gg - 3], sizeof(post[g]));
        else { post[g][0] = 1; post[g][1] = 1; }
    }
    memcpy(post[15], qpow::RC_INITIAL[0], sizeof(post[15]));
    u64 rci[22] = {};
    for (int r = 0; r < 21; r++) rci[r] = qpow::RC_INTERNAL[r + 1];
    const u32 eps = 0xFFFFFFFFu;
    return cudaMemcpyToSymbol(c_post, post, sizeof(post)) == cudaSuccess &&
           cudaMemcpyToSymbol(c_rci, rci, sizeof(rci)) == cudaSuccess &&
           cudaMemcpyToSymbol(c_rct0, qpow::RC_TERMINAL[0], sizeof(c_rct0)) == cudaSuccess &&
           cudaMemcpyToSymbol(c_diag, qpow::MDS_DIAG, sizeof(c_diag)) == cudaSuccess &&
           cudaMemcpyToSymbol(c_eps, &eps, sizeof(eps)) == cudaSuccess;
}

using qpow::nonce_line::kTSpan;

void map_nonce(const uint8_t ctr[64], uint8_t out[64]) { qpow::nonce_line::map(ctr, out); }

/* Kernel parameters for counters ctr .. ctr + count - 1 (count <= kTSpan - t: k stays fixed). */
void make_params(const u64 mid[12], const uint8_t ctr[64], Params& p)
{
    u64 pk[qpow::nonce_line::kParams];
    qpow::nonce_line::launch_params(mid, ctr, pk);
    p.pre0 = pk[0];
    p.pre4 = pk[1];
    memcpy(p.K, pk + 2, sizeof(p.K));
    p.tb = qpow::nonce_line::t_of(ctr);
}

/* Reference value of the kernel's output for one nonce. */
u64 cpu_out0(const uint8_t header[32], const uint8_t nonce[64])
{
    u64 s[12];
    qpow::mining_midstate(header, nonce, s);
    qpow::absorb32(s, nonce + 32);
    qpow::permute(s);
    s[0] = qpow::gf_add(s[0], 1);
    s[1] = qpow::gf_add(s[1], 1);
    qpow::permute(s);
    return qpow::gf_canon(s[0]);
}

/* Hash 4096 nonces on the device and compare a spread of them with the reference code. */
bool self_test(Dev& d)
{
    uint8_t header[32], ctr[64];
    for (int i = 0; i < 32; i++) header[i] = (uint8_t)(i * 37 + 11);
    for (int i = 0; i < 64; i++) ctr[i] = (uint8_t)(i * 91 + 5);
    ctr[60] = 0x13;  /* t starts at 0x3000FF0: crosses byte boundaries, stays below 2^26 */
    ctr[61] = 0x00;
    ctr[62] = 0x0F;
    ctr[63] = 0xF0;
    const u32 n = 4096;
    u64 mid[12];
    qpow::mining_midstate(header, ctr, mid);
    Params p{};
    make_params(mid, ctr, p);
    p.count = n;
    p.t0 = 0;
    u64* dump = nullptr;
    if (!ck(cudaMalloc(&dump, n * sizeof(u64)), "self-test alloc", d.ordinal)) return false;
    p.dump = dump;
    p.out = d.d_out;
    bool ok = ck(cudaMemsetAsync(d.d_out, 0, 16 * sizeof(u32), d.stream), "self-test memset", d.ordinal);
    if (ok) {
        qpow_cuda_scan<<<(n + k_tpb - 1) / k_tpb, k_tpb, 0, d.stream>>>(p);
        ok = ck(cudaGetLastError(), "self-test launch", d.ordinal) &&
             ck(cudaStreamSynchronize(d.stream), "self-test run", d.ordinal);
    }
    std::vector<u64> got(n);
    if (ok) ok = ck(cudaMemcpy(got.data(), dump, n * sizeof(u64), cudaMemcpyDeviceToHost), "self-test copy", d.ordinal);
    cudaFree(dump);
    if (!ok) return false;
    /* 128 samples: an unstable overclock that corrupts a few hashes in a thousand still shows up */
    for (u32 i = 0; i < n; i += 32) {
        uint8_t c[64], nn[64];
        memcpy(c, ctr, 64);
        be_add(c, i);
        map_nonce(c, nn);
        const u64 ref = cpu_out0(header, nn);
        if (got[i] != ref) {
            fprintf(stderr, "[qpow-cuda] device %d: self-test mismatch at nonce +%u (got %016llx, want %016llx)\n",
                    d.ordinal, i, (unsigned long long)got[i], (unsigned long long)ref);
            return false;
        }
    }
    return true;
}

void release(Dev& d)
{
    if (d.ordinal < 0) return;
    cudaSetDevice(d.ordinal);
    if (d.d_out) cudaFree(d.d_out);
    if (d.h_out) cudaFreeHost(d.h_out);
    if (d.ev0) cudaEventDestroy(d.ev0);
    if (d.ev1) cudaEventDestroy(d.ev1);
    if (d.stream) cudaStreamDestroy(d.stream);
    d = Dev{};
}

int cancelled()
{
    return cp_job_should_cancel() || cp_pool_conn_lost();
}

}  // namespace

extern "C" void cp_qpow_cuda_worker_set_batch_size(uint32_t batch)
{
    g_batch_req = batch;
}

extern "C" uint64_t cp_qpow_cuda_worker_batch_size(void)
{
    uint64_t total = 0;
    for (const Dev& d : g_devs) total += d.launch;
    return total ? total : 1000000u;
}

extern "C" int cp_qpow_cuda_worker_is_ready(void)
{
    return g_ready;
}

extern "C" void cp_qpow_cuda_worker_shutdown(void)
{
    for (Dev& d : g_devs) release(d);
    g_devs.clear();
    g_ready = 0;
}

extern "C" int cp_qpow_cuda_worker_init(const int* devices, int ndev)
{
    cp_qpow_cuda_worker_shutdown();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        fprintf(stderr, "[qpow-cuda] no CUDA device\n");
        return -1;
    }
    std::vector<int> want;
    if (devices && ndev > 0) want.assign(devices, devices + ndev);
    else want.push_back(0);

    for (int ordinal : want) {
        if (ordinal < 0 || ordinal >= count) {
            fprintf(stderr, "[qpow-cuda] device %d does not exist (%d CUDA devices)\n", ordinal, count);
            cp_qpow_cuda_worker_shutdown();
            return -1;
        }
        Dev d;
        d.ordinal = ordinal;
        cudaDeviceProp prop;
        bool ok = ck(cudaSetDevice(ordinal), "set device", ordinal) &&
                  ck(cudaGetDeviceProperties(&prop, ordinal), "properties", ordinal) &&
                  ck(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking), "stream", ordinal) &&
                  ck(cudaEventCreate(&d.ev0), "event", ordinal) &&
                  ck(cudaEventCreate(&d.ev1), "event", ordinal) &&
                  ck(cudaMalloc(&d.d_out, 16 * sizeof(u32)), "alloc", ordinal) &&
                  ck(cudaMallocHost(&d.h_out, 16 * sizeof(u32)), "pinned alloc", ordinal);
        if (ok && !upload_constants()) {
            fprintf(stderr, "[qpow-cuda] device %d: constant upload failed\n", ordinal);
            ok = false;
        }
        int per_sm = 0;
        if (ok) ok = ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, qpow_cuda_scan, k_tpb, 0),
                        "occupancy", ordinal);
        if (ok && !self_test(d)) {
            fprintf(stderr, "[qpow-cuda] device %d: hash self-test failed, not mining on it\n", ordinal);
            ok = false;
        }
        if (!ok) {
            g_devs.push_back(d);
            cp_qpow_cuda_worker_shutdown();
            return -1;
        }
        d.sms = prop.multiProcessorCount;
        d.blocks = d.sms * (per_sm > 0 ? per_sm : 1);
        /* Start at ~2^22 nonces and let search() settle near 100 ms per launch. */
        d.launch = g_batch_req ? g_batch_req : (uint32_t)d.blocks * k_tpb * 16;
        printf("[qpow-cuda] device[%d]: %s sm_%d%d SMs=%d blocks=%d batch=%s self-test ok\n",
               ordinal, prop.name, prop.major, prop.minor, d.sms, d.blocks,
               g_batch_req ? "fixed" : "auto");
        fflush(stdout);
        g_devs.push_back(d);
    }
    g_ready = 1;
    return 0;
}

extern "C" int cp_qpow_cuda_worker_search(
    const uint8_t header[32],
    const uint8_t target_be[64],
    const uint8_t start_be[64],
    uint64_t count,
    uint8_t out_nonce_be[64],
    uint8_t out_hash_be[64],
    uint64_t* out_hashes)
{
    if (!g_ready || !out_hashes || count == 0) return CP_QPOW_CUDA_ERROR;
    *out_hashes = 0;

    u64 t0 = 0;
    for (int i = 0; i < 8; i++) t0 = (t0 << 8) | target_be[i];

    uint8_t cur[64];
    memcpy(cur, start_be, 64);
    uint8_t mid_high[32];
    u64 mid[12];
    bool have_mid = false;
    uint64_t done = 0;

    while (done < count) {
        if (cancelled()) {
            *out_hashes = done;
            return CP_QPOW_CUDA_CANCELLED;
        }
        /* Split the next step across the devices; no launch crosses a wrap of t (k stays fixed). */
        uint64_t planned = 0;
        for (Dev& d : g_devs) {
            d.n = 0;
            if (done + planned >= count) continue;
            memcpy(d.base, cur, 64);
            uint64_t n = d.launch;
            n = n < count - done - planned ? n : count - done - planned;
            const uint64_t room = kTSpan - qpow::nonce_line::t_of(cur);
            n = n < room ? n : room;
            d.n = (uint32_t)n;
            be_add(cur, n);
            planned += n;
        }
        for (Dev& d : g_devs) {
            if (!d.n) continue;
            if (!have_mid || memcmp(mid_high, d.base, 32) != 0) {
                memcpy(mid_high, d.base, 32);
                qpow::mining_midstate(header, d.base, mid);
                have_mid = true;
            }
            Params p{};
            make_params(mid, d.base, p);
            p.t0 = t0;
            p.count = d.n;
            p.out = d.d_out;
            cudaSetDevice(d.ordinal);
            cudaMemsetAsync(d.d_out, 0, 16 * sizeof(u32), d.stream);
            cudaEventRecord(d.ev0, d.stream);
            const uint64_t want_blocks = ((uint64_t)d.n + k_tpb - 1) / k_tpb;
            const int grid = (int)(want_blocks < (uint64_t)d.blocks ? want_blocks : (uint64_t)d.blocks);
            qpow_cuda_scan<<<grid, k_tpb, 0, d.stream>>>(p);
            cudaEventRecord(d.ev1, d.stream);
            cudaMemcpyAsync(d.h_out, d.d_out, 16 * sizeof(u32), cudaMemcpyDeviceToHost, d.stream);
        }
        int found = 0;
        for (Dev& d : g_devs) {
            if (!d.n) continue;
            cudaSetDevice(d.ordinal);
            if (!ck(cudaStreamSynchronize(d.stream), "scan", d.ordinal)) {
                *out_hashes = done;
                return CP_QPOW_CUDA_ERROR;
            }
            /* Automatic batch: aim at ~100 ms per launch. */
            float ms = 0;
            if (!g_batch_req && d.n == d.launch && cudaEventElapsedTime(&ms, d.ev0, d.ev1) == cudaSuccess && ms > 0) {
                double next = (double)d.launch * 100.0 / ms;
                const double lo = (double)d.blocks * k_tpb, hi = 1u << 30;
                next = next < lo ? lo : (next > hi ? hi : next);
                d.launch = (uint32_t)(0.5 * d.launch + 0.5 * next);
            }
            /* Report the lowest valid nonce of the step: the caller resumes right after it,
             * so later shares of the same step are found again on the next call. Devices
             * hold ascending ranges, so the first device with a share wins. */
            const u32 nc = d.h_out[0] < (u32)k_max_candidates ? d.h_out[0] : (u32)k_max_candidates;
            u32 best = 0xFFFFFFFFu;
            for (u32 k = 0; k < nc && !found; k++) {
                const u32 idx = d.h_out[1 + k];
                if (idx >= best) continue;
                uint8_t c[64], nonce[64], hash[64];
                memcpy(c, d.base, 64);
                be_add(c, idx);
                map_nonce(c, nonce);
                u64 m[12];
                qpow::mining_midstate(header, nonce, m);
                if (qpow::hash_if_valid(m, nonce + 32, target_be, hash)) {
                    best = idx;
                    memcpy(out_nonce_be, nonce, 64);
                    memcpy(out_hash_be, hash, 64);
                    memcpy(g_found_nonce, nonce, 64);
                    memcpy(g_found_ctr, c, 64);
                }
            }
            if (best != 0xFFFFFFFFu) found = 1;
        }
        done += planned;
        *out_hashes = done;
        if (found) return CP_QPOW_CUDA_OK_FOUND;
    }
    return CP_QPOW_CUDA_OK_EXHAUSTED;
}

extern "C" void cp_qpow_cuda_worker_resume(const uint8_t found_be[64], uint8_t next_be[64])
{
    memcpy(next_be, memcmp(found_be, g_found_nonce, 64) == 0 ? g_found_ctr : found_be, 64);
    be_add(next_be, 1);
}
