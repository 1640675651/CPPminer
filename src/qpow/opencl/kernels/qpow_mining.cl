/* Quantus QPoW: Poseidon2 over Goldilocks (p = 2^64 - 2^32 + 1, width 12), one nonce per
 * work-item.
 *
 * A launch covers consecutive positions t of one nonce line (qpow/nonce_line.hpp), along which
 * only lanes 0 and 4 change after the first linear layer. The host folds the midstate, the line
 * base, that layer and the other ten S-boxes of the first round into 14 constants; the kernel
 * runs two S-boxes, then the rest of both permutations. After the last S-box layer only the
 * first output element is computed: it
 * holds the first 8 hash bytes. A nonce whose first 8 bytes are <= the target's becomes a
 * candidate; the host re-hashes every candidate with the reference code before reporting it.
 *
 * Field elements are any 64-bit value (not necessarily < p); all arithmetic is mod p.
 *
 * Build options:
 *   -DQV_MUL=1|2|3   64x64 -> 128 product: 1 = ulong mul + mul_hi, 2 = four 32x32 products
 *                    with explicit carries, 3 = carry-free chain of 32x32 + 64 multiply-adds
 *                    (maps to one mad_u64_u32 each on AMD GCN5+/RDNA)
 *   -DQV_RED=1|2     128 -> 64 reduction: 1 = single carry fold (exact except a ~2^-64 corner),
 *                    2 = signed form with shift-derived borrows (exact; best on Intel Xe-HPG)
 *   -DQV_EXT22=0|1   linear layers in lazy 96-bit sums, or in carry-free 22-bit limbs (default)
 *   -DQV_OVF=0|1     carry detection by compare or by __builtin_add_overflow (default 1 where
 *                    the compiler has the builtin: +3-4% on Intel and NVIDIA, same on AMD)
 *   -DQV_WRED_FAST=0|1  exact lazy-sum reduction, or skip its ~t/2^32 carry fold (default 1)
 *   -DQV_NPW=N       hash N consecutive nonces per work-item in lockstep (default 1; the host
 *                    reads N from CP_QPOW_OCL_OPTS and launches 1/N of the work-items)
 *   -DQV_TEST        also build the field-op test kernel
 */

#ifndef QV_MUL
#define QV_MUL 3
#endif

#define EPS 0xFFFFFFFFUL

/* r = a + b, returns the carry out of bit 63. With -DQV_OVF=1 through the compiler's
 * add-with-overflow builtin (lets the backend reuse the add's own carry), else by compare. */
#ifndef QV_OVF
#if defined(__has_builtin)
#if defined(__AMDGCN__) && __has_builtin(__builtin_addc)
/* gfx1201: 11% fewer VALU instructions in qpow_scan than form 1 (no 64-bit compares) */
#define QV_OVF 2
#elif __has_builtin(__builtin_add_overflow)
#define QV_OVF 1
#endif
#endif
#endif
#ifndef QV_OVF
#define QV_OVF 0
#endif
#if QV_OVF == 2
/* Two chained 32-bit add-with-carry: AMDGPU maps them onto v_add_co_u32 + v_add_co_ci_u32 and can
 * feed the final carry straight into the next add, where the 64-bit uaddo was expanded to the add
 * plus a 64-bit compare and a cndmask. */
uint addc64(ulong* r, ulong a, ulong b)
{
    uint c0, c1;
    const uint lo = __builtin_addc((uint)a, (uint)b, 0u, &c0);
    const uint hi = __builtin_addc((uint)(a >> 32), (uint)(b >> 32), c0, &c1);
    *r = ((ulong)hi << 32) | lo;
    return c1;
}
#define ADDC(r, a, b) addc64(&(r), (ulong)(a), (ulong)(b))
#elif QV_OVF
#define ADDC(r, a, b) ((uint)__builtin_add_overflow((ulong)(a), (ulong)(b), &(r)))
#else
#define ADDC(r, a, b) ((r) = (ulong)(a) + (ulong)(b), (uint)((r) < (ulong)(b)))
#endif

/* ------------------------------------------------------------------ constants */

__constant ulong RC_INITIAL[4][12] = {
    {0xc002e770975b1607UL, 0xbca51a8dfe14593aUL, 0x72938dfbe774f7f9UL, 0xe4f2fe29e03234acUL,
     0xd5e0ba2f541b6449UL, 0xec33b868f3cc46c1UL, 0x486dcb55419d475aUL, 0x6c1cb2a358cc24f1UL,
     0xe3f30d509a1436bbUL, 0xd9a64f068dca7c29UL, 0xe59b3f57aabba1aeUL, 0x2a3dd4505b478fdcUL},
    {0xada1f8dc7676ed25UL, 0x2711aa8b5509d516UL, 0x4ae6acd0c9c92897UL, 0x56eb3d6b5256d67aUL,
     0x1f7a9d55923bf51eUL, 0x3600427d397a7f68UL, 0xe5076df75b72c3d0UL, 0xfcd59aa12c6090adUL,
     0xcd895e8c68b57a9eUL, 0x41df7ef9d730ae3eUL, 0xee3e2b889abe977dUL, 0xd29bb7edbeb9c405UL},
    {0x7d5c08eef608e382UL, 0x89ae889caaf0802cUL, 0xb35a8e976d2af617UL, 0xdb14234eafaf5173UL,
     0x78f04462d48b1c98UL, 0x265293b0e47ce88aUL, 0x999a649b69b9d32fUL, 0x64b0a186698e01d3UL,
     0xee0b22d0dfae8bb8UL, 0x4fd53e50ca04a7eeUL, 0x5762bfe181f25047UL, 0xf51593e2beb5e3bdUL},
    {0x1e5e2b5760e32477UL, 0x622462a1f9aaaeedUL, 0xaa284b3ecdb222aeUL, 0x63c8e72f542bf3fcUL,
     0x3ba588cacb43b5e0UL, 0x23eda6f3c99150ddUL, 0xaad3bea4baac9a5aUL, 0xe9da8d699b94184aUL,
     0xcdb13f4cd93e024cUL, 0x902cbd0956f655e3UL, 0x5b4e40ffc759532fUL, 0xde795c20a2357af7UL}};

__constant ulong RC_TERMINAL[4][12] = {
    {0x7b72c539e0ea4c6eUL, 0x144573dae2ce9976UL, 0x802028b68f35fc88UL, 0x6d36c5022c4fe7c2UL,
     0xa205d0ffa9b9def3UL, 0xf6e7e38b1ea6ba2fUL, 0x34f7909ae5258d64UL, 0xb0464d9d77b97fcaUL,
     0x64ddb9d5de7e00a6UL, 0x0ed0d75c27975d97UL, 0x1cbb36f11127338bUL, 0x6673e505cfd0b6baUL},
    {0x605f902830872e01UL, 0x3fd5eb927e95fe4fUL, 0xe81025b5a24c69cdUL, 0xf7d0ce75de23f74eUL,
     0xf39942b6a8585089UL, 0x6d808a08f7b71df6UL, 0xf8806b6588f49a8bUL, 0x57df2d8c2a32107aUL,
     0x16e7c2074d654a2dUL, 0x213de241fcf33835UL, 0xb0f2b8905a0976f6UL, 0xd8e3cf2bbd355417UL},
    {0xe498691679d9330fUL, 0x763b45d2a3821b28UL, 0x0908bf65eb0a1f0dUL, 0x7691eb2d194b24f4UL,
     0x0e43551233ae13b2UL, 0x93c393dbfc2fe76fUL, 0x98f607485d48cdeaUL, 0xe3d95f30309819c0UL,
     0x1ef581a93eaf6acfUL, 0x0b24c1b7a030fca4UL, 0x624370be5670b327UL, 0x5f1e28615a11e486UL},
    {0xfe04051f909e042bUL, 0x7257e5b147fd3803UL, 0xe6ae134bb82f2e78UL, 0x5711fd5cf4784511UL,
     0xf83a42660c08c0bcUL, 0x2cd8c96d9a3ce855UL, 0x7d2ffb1bb0e17271UL, 0x85ae1528caea3811UL,
     0x52a345d5c7adb0b8UL, 0x504c4c51f3faee94UL, 0xbce34a649cfccaf9UL, 0xe0a3389266fb6dc9UL}};

/* RC_INTERNAL[r + 1] for internal round r (the last one adds nothing). */
__constant ulong RC_INTERNAL_NEXT[22] = {
    0xd1d2bf082f60d4f0UL, 0x69a377a79f9ad206UL, 0xa9d06906a3858e24UL, 0x295275001eede5b5UL,
    0x5874e441117bd746UL, 0x8a084bbba8ed86ccUL, 0x3defd7645cde6425UL, 0x3998cfe6871cc137UL,
    0x3e52ef8bca48314aUL, 0x964a209f85dc9eccUL, 0x3fcc9ee82cc4577eUL, 0x8e79b4a5d0096d6dUL,
    0x8492362ad2392556UL, 0xee72f470262574d6UL, 0x1e0e18496da2444aUL, 0x0f3a74bf215eaac6UL,
    0x1b061b76a1c0ded3UL, 0x192c42d86803d7a6UL, 0xf6d49ff997ae0260UL, 0x3ec372e7a0fa3786UL,
    0x5538cdf4f23445d3UL, 0x0UL};

/* Rows added after a linear layer that has no full-round constants of its own: RC_INTERNAL[0]
 * on lane 0 before the internal rounds, and the second absorb's +1/+1 between permutations. */
__constant ulong ROW_RCI0[12] = {0x97f7798a784ad863UL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__constant ulong ROW_ONES[12] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__constant ulong MDS_DIAG[12] = {
    0xc3b6c08e23ba9300UL, 0xd84b5de94a324fb6UL, 0x0d0c371c5b35b84fUL, 0x7964f570e7188037UL,
    0x5daf18bbd996604bUL, 0x6743bc47b9595257UL, 0x5528b9362c59bb70UL, 0xac45e25b7127b68bUL,
    0xa2077d7dfbb606b5UL, 0xf3faac6faee378aeUL, 0x0c6388b51545e883UL, 0xd27dbb6944917b60UL};

/* ------------------------------------------------------------------ field arithmetic */

/* a*b as 128 bits */
void mul128(ulong a, ulong b, ulong* lo, ulong* hi)
{
#if QV_MUL == 1
    *lo = a * b;
    *hi = mul_hi(a, b);
#elif QV_MUL == 2
    uint a0 = (uint)a, a1 = (uint)(a >> 32), b0 = (uint)b, b1 = (uint)(b >> 32);
    ulong p = (ulong)a0 * b0, q = (ulong)a1 * b1;
    ulong m1 = (ulong)a0 * b1, m2 = (ulong)a1 * b0;
    ulong m, l;
    uint mc = ADDC(m, m1, m2);
    uint lc = ADDC(l, p, m << 32);
    *lo = l;
    *hi = q + (m >> 32) + ((ulong)mc << 32) + lc;
#else
    /* Every step fits 64 bits: (2^32 - 1)^2 + 2 * (2^32 - 1) = 2^64 - 1. */
    uint a0 = (uint)a, a1 = (uint)(a >> 32), b0 = (uint)b, b1 = (uint)(b >> 32);
    ulong p0 = (ulong)a0 * b0;
    ulong m = (ulong)a1 * b0 + (p0 >> 32);
    ulong m2 = (ulong)a0 * b1 + (uint)m;
    *hi = (ulong)a1 * b1 + (m >> 32) + (m2 >> 32);
    *lo = (p0 & EPS) | (m2 << 32);
#endif
}

/* a^2 as 128 bits from three 32x32 products */
void sqr128(ulong a, ulong* lo, ulong* hi)
{
#if QV_MUL == 1
    *lo = a * a;
    *hi = mul_hi(a, a);
#else
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    ulong p = (ulong)a0 * a0, q = (ulong)a1 * a1, t = (ulong)a0 * a1;
    /* p + 2t * 2^32 + q * 2^64 */
    ulong l;
    uint lc = ADDC(l, p, t << 33);
    *lo = l;
    *hi = q + (t >> 31) + lc;
#endif
}

/* 128 -> 64 bits mod p. V = lo + r2*2^64 + r3*2^96 = lo + r2*EPS - r3. t = lo + r2*EPS with
 * carry c; on a carry 2^64 = EPS turns -r3 into EPS - r3 = c*2^32 - (r3 + c), which cannot
 * overflow. Exact except when r2 == 0 and lo < r3 without a carry (probability ~2^-64 for
 * hash data); a miss there only affects the candidate filter, and candidates are re-hashed on
 * the host. Needs r3 <= 2^32 - 2: true for any product, and for a*b + w with b below the
 * internal diagonal's 0xF3FB... */
#ifndef QV_RED
#define QV_RED 1
#endif
ulong red128(ulong lo, ulong hi)
{
    const uint r2 = (uint)hi, r3 = (uint)(hi >> 32);
#if QV_RED == 2
    /* Signed form, exact: V = (l0 - r2 - r3) + (l1 + r2) * 2^32 lies in (-2^33, 2^65). The
     * borrows and carries come from arithmetic shifts, so no carry-out test is needed; the
     * top word v2 in {-1, 0, 1} folds back as v2 * EPS without a second overflow. */
    const long d0 = (long)(uint)lo - (long)r2 - (long)r3;
    const long d1 = (long)(uint)(lo >> 32) + (long)r2 + (d0 >> 32);
    const long v2 = d1 >> 32;
    const ulong z = ((ulong)(uint)d1 << 32) | (uint)d0;
    return z + ((ulong)v2 << 32) - (ulong)v2;
#else
    ulong t;
    const uint c = ADDC(t, lo, ((ulong)r2 << 32) - r2);
    return t + ((ulong)c << 32) - (ulong)(r3 + c);
#endif
}

ulong gmul(ulong a, ulong b) { ulong l, h; mul128(a, b, &l, &h); return red128(l, h); }
ulong gsqr(ulong a) { ulong l, h; sqr128(a, &l, &h); return red128(l, h); }

/* Lazy sum for the linear layers: value = v + t * 2^64. */
typedef struct { ulong v; uint t; } W;

W w2(ulong a, ulong b) { W r; r.t = ADDC(r.v, a, b); return r; }
W wadd(W a, W b) { W r; r.t = a.t + b.t + ADDC(r.v, a.v, b.v); return r; }
W wadd64(W a, ulong b) { W r; r.t = a.t + ADDC(r.v, a.v, b); return r; }

/* v + t*EPS with small t: one carry at most, after which the sum is < 2^37, so +EPS is safe.
 * QV_WRED_FAST (default) skips that fold: the carry fires with probability about t / 2^32 per
 * call, i.e. a couple of hashes in a million come out wrong, which only matters if one of them
 * was a share; every candidate is re-hashed on the host. */
#ifndef QV_WRED_FAST
#define QV_WRED_FAST 1
#endif
ulong wred(W a)
{
#if QV_WRED_FAST
    return a.v + ((ulong)a.t << 32) - a.t;
#else
    ulong s;
    uint c = ADDC(s, a.v, ((ulong)a.t << 32) - a.t);
    return s + (c ? EPS : 0UL);
#endif
}

ulong gadd(ulong a, ulong b) { return wred(w2(a, b)); }

ulong canon(ulong a) { return a >= 0xFFFFFFFF00000001UL ? a - 0xFFFFFFFF00000001UL : a; }

/* a*b + w with w a lazy sum, reduced once.
 * QV_GFMA_CHAIN=1 (with QV_MUL=3): w rides in the 32x32+64 multiply-add chain instead of a
 * separate 128-bit add. Bounds, with every 32-bit word <= 2^32-1:
 *   p0 = a0*b0 + w.lo                               < 2^64
 *   m  = a1*b0 + (p0>>32 + w.hi)  <= (2^32-1)^2 + 2(2^32-1) = 2^64-1
 *   m2 = a0*b1 + (uint)m                            < 2^64
 *   hi = a1*b1 + m>>32 + m2>>32 + t                 (a*b + w < 2^128 for the diagonal b)
 * so every step maps onto one v_mad_u64_u32 with its 64-bit addend and no carry-out tests. */
#ifndef QV_GFMA_CHAIN
#define QV_GFMA_CHAIN 0
#endif
ulong gfma_w(ulong a, ulong b, W w)
{
#if QV_GFMA_CHAIN && QV_MUL == 3
    const uint a0 = (uint)a, a1 = (uint)(a >> 32), b0 = (uint)b, b1 = (uint)(b >> 32);
    const ulong p0 = (ulong)a0 * b0 + (uint)w.v;
    const ulong m = (ulong)a1 * b0 + ((p0 >> 32) + (w.v >> 32));
    const ulong m2 = (ulong)a0 * b1 + (uint)m;
    const ulong hi = (ulong)a1 * b1 + ((m >> 32) + (m2 >> 32) + (ulong)w.t);
    const ulong lo = (p0 & EPS) | (m2 << 32);
    return red128(lo, hi);
#else
    ulong lo, hi;
    mul128(a, b, &lo, &hi);
    ulong l2;
    hi += (ulong)w.t + ADDC(l2, lo, w.v);
    return red128(l2, hi);
#endif
}

ulong sbox(ulong x)
{
    ulong x2 = gsqr(x);
    ulong x3 = gmul(x2, x);
    ulong x4 = gsqr(x2);
    return gmul(x3, x4);
}

/* ------------------------------------------------------------------ permutation */

#ifndef QV_EXT22
#define QV_EXT22 1
#endif

#if QV_EXT22
/* Linear layers in three carry-free limbs, x = a + b*2^22 + c*2^44 (22/22/20 bits). The
 * coefficients of a layer row sum to at most 28, so a limb stays below 2^27 and plain 32-bit
 * adds never overflow; carries are resolved once per output in l3red. Pays off where 64-bit
 * adds and carry-outs are emulated (Intel Xe-HPG). */
typedef struct { uint a, b, c; } L3;

L3 l3(ulong x)
{
    const uint lo = (uint)x, hi = (uint)(x >> 32);
    L3 r;
    r.a = lo & 0x3FFFFFu;
    r.b = (lo >> 22) | ((hi & 0xFFFu) << 10);
    r.c = hi >> 12;
    return r;
}
L3 l3add(L3 x, L3 y) { L3 r; r.a = x.a + y.a; r.b = x.b + y.b; r.c = x.c + y.c; return r; }

/* limbs + add, reduced to 64 bits */
ulong l3red(L3 y, ulong add)
{
    const ulong v0 = (ulong)y.a + ((ulong)y.b << 22);            /* < 2^50 */
    W w;
    w.t = (y.c >> 20) + ADDC(w.v, v0, (ulong)(y.c & 0xFFFFFu) << 44);
    return wred(wadd64(w, add));
}

void mat4l(L3 x0, L3 x1, L3 x2, L3 x3, L3* y0, L3* y1, L3* y2, L3* y3)
{
    L3 t01 = l3add(x0, x1), t23 = l3add(x2, x3);
    L3 t0123 = l3add(t01, t23);
    L3 t01123 = l3add(t0123, x1);
    L3 t01233 = l3add(t0123, x3);
    *y3 = l3add(t01233, l3add(x0, x0));
    *y1 = l3add(t01123, l3add(x2, x2));
    *y0 = l3add(t01123, t01);
    *y2 = l3add(t01233, t23);
}

/* s = M_ext * s + add, M_ext = circ(2*M4, M4, M4) */
void ext_add(ulong* s, __constant const ulong* add)
{
    L3 x[12], y[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) x[i] = l3(s[i]);
    #pragma unroll
    for (int k = 0; k < 3; k++)
        mat4l(x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3], &y[4 * k], &y[4 * k + 1], &y[4 * k + 2], &y[4 * k + 3]);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const L3 sum = l3add(l3add(y[j], y[4 + j]), y[8 + j]);
        #pragma unroll
        for (int k = 0; k < 3; k++) s[4 * k + j] = l3red(l3add(y[4 * k + j], sum), add[4 * k + j]);
    }
}

/* out0 = 2*y00 + y10 + y20 with y_k0 = 2*x0 + 3*x1 + x2 + x3 of chunk k */
ulong out0_of(const ulong* s)
{
    L3 acc = {0, 0, 0};
    const uint coef[12] = {4, 6, 2, 2, 2, 3, 1, 1, 2, 3, 1, 1};
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        const L3 x = l3(s[i]);
        acc.a += coef[i] * x.a; acc.b += coef[i] * x.b; acc.c += coef[i] * x.c;
    }
    return canon(l3red(acc, 0));
}
#else
void mat4(ulong x0, ulong x1, ulong x2, ulong x3, W* y0, W* y1, W* y2, W* y3)
{
    W t01 = w2(x0, x1), t23 = w2(x2, x3);
    W t0123 = wadd(t01, t23);
    W t01123 = wadd64(t0123, x1);
    W t01233 = wadd64(t0123, x3);
    *y3 = wadd(t01233, w2(x0, x0));
    *y1 = wadd(t01123, w2(x2, x2));
    *y0 = wadd(t01123, t01);
    *y2 = wadd(t01233, t23);
}

/* s = M_ext * s + add, M_ext = circ(2*M4, M4, M4) */
void ext_add(ulong* s, __constant const ulong* add)
{
    W y[12];
    #pragma unroll
    for (int k = 0; k < 3; k++)
        mat4(s[4 * k], s[4 * k + 1], s[4 * k + 2], s[4 * k + 3], &y[4 * k], &y[4 * k + 1], &y[4 * k + 2], &y[4 * k + 3]);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        W sum = wadd(wadd(y[j], y[4 + j]), y[8 + j]);
        #pragma unroll
        for (int k = 0; k < 3; k++) s[4 * k + j] = wred(wadd64(wadd(y[4 * k + j], sum), add[4 * k + j]));
    }
}

/* out0 = 2*y00 + y10 + y20 with y_k0 = 2*x0 + 3*x1 + x2 + x3 of chunk k */
ulong out0_of(const ulong* s)
{
    W acc = w2(s[0], s[0]);
    acc = wadd64(acc, s[0]); acc = wadd64(acc, s[0]);
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        const int m = k == 0 ? 2 : 1;
        for (int rep = 0; rep < m; rep++) {
            acc = wadd64(acc, s[4 * k + 1]); acc = wadd64(acc, s[4 * k + 1]); acc = wadd64(acc, s[4 * k + 1]);
            acc = wadd64(acc, s[4 * k + 2]); acc = wadd64(acc, s[4 * k + 3]);
        }
        if (k > 0) { acc = wadd64(acc, s[4 * k]); acc = wadd64(acc, s[4 * k]); }
    }
    return canon(wred(acc));
}
#endif

/* What is added after the linear layer that follows full round g of the two permutations;
 * g == 15 is the second permutation's first linear layer (RC_INITIAL[0]). */
__constant const ulong* post_row(int g)
{
    if (g == 15) return RC_INITIAL[0];
    const int gg = g & 7;
    if (gg <= 2) return RC_INITIAL[gg + 1];
    if (gg == 3) return ROW_RCI0;
    if (gg <= 6) return RC_TERMINAL[gg - 3];
    return ROW_ONES;
}

/* 22 internal rounds: s[0] arrives with RC_INTERNAL[0] added, leaves with RC_TERMINAL[0].
 * The row sum stays a lazy value and enters each diagonal product before its reduction. */
void internal22(ulong* s)
{
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
        W rest = w2(s[1], s[2]);
        #pragma unroll
        for (int i = 3; i < 12; i++) rest = wadd64(rest, s[i]);
        s[0] = sbox(s[0]);
        W sg = wadd64(rest, s[0]);
        s[0] = gfma_w(s[0], MDS_DIAG[0], wadd64(sg, RC_INTERNAL_NEXT[r]));
        #pragma unroll
        for (int i = 1; i < 12; i++) s[i] = gfma_w(s[i], MDS_DIAG[i], sg);
    }
    #pragma unroll
    for (int i = 0; i < 12; i++) s[i] = gadd(s[i], RC_TERMINAL[0][i]);
}

/* Canonical first element of the second permutation's output for nonce-line position t
 * (qpow/nonce_line.hpp). pk[0], pk[1] are lanes 0 and 4 after the first linear layer at t = 0,
 * pk[2..13] the state entering round 1 without them. Only lanes 0 (-35t) and 4 (+35t) depend on
 * t; columns 0 and 4 of M_ext are m0 * (2,1,1) and m0 * (1,2,1) over the blocks, m0 = (2,1,1,3). */
ulong hash_out0(__constant const ulong* pk, uint t)
{
    ulong s[12];
    {
        const ulong d = 35ul * t;
        const ulong f = sbox(gadd(pk[0], 0xFFFFFFFF00000001ul - d));
        const ulong g = sbox(gadd(pk[1], d));
        const W c2 = w2(f, g);
        W c[3];
        c[0] = wadd64(c2, f);
        c[1] = wadd64(c2, g);
        c[2] = c2;
        const int m0[4] = {2, 1, 1, 3};
        #pragma unroll
        for (int k = 0; k < 3; k++)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                W v = c[k];
                if (m0[j] >= 2) v = wadd(v, c[k]);
                if (m0[j] == 3) v = wadd(v, c[k]);
                s[4 * k + j] = wred(wadd64(v, pk[2 + 4 * k + j]));
            }
    }
#pragma unroll 1
    for (int g = 1;; g++) {
        #pragma unroll
        for (int i = 0; i < 12; i++) s[i] = sbox(s[i]);
        if (g == 15) break;
        ext_add(s, post_row(g));
        if (g == 7) ext_add(s, post_row(15));
        if ((g & 7) == 3) internal22(s);
    }
    return out0_of(s);
}

/* QV_NPW > 1: one work-item hashes QV_NPW consecutive line positions in lockstep. The chains are
 * independent, so the compiler can fill carry (VCC) waits and pair instructions (VOPD on RDNA)
 * from the other nonce. Costs registers; the host launches 1/QV_NPW of the work-items. */
#ifndef QV_NPW
#define QV_NPW 1
#endif
#if QV_NPW > 1
void internal22_n(ulong s[QV_NPW][12])
{
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
        W rest[QV_NPW];
        #pragma unroll
        for (int n = 0; n < QV_NPW; n++) {
            rest[n] = w2(s[n][1], s[n][2]);
            #pragma unroll
            for (int i = 3; i < 12; i++) rest[n] = wadd64(rest[n], s[n][i]);
        }
        #pragma unroll
        for (int n = 0; n < QV_NPW; n++) s[n][0] = sbox(s[n][0]);
        #pragma unroll
        for (int n = 0; n < QV_NPW; n++) {
            const W sg = wadd64(rest[n], s[n][0]);
            s[n][0] = gfma_w(s[n][0], MDS_DIAG[0], wadd64(sg, RC_INTERNAL_NEXT[r]));
            #pragma unroll
            for (int i = 1; i < 12; i++) s[n][i] = gfma_w(s[n][i], MDS_DIAG[i], sg);
        }
    }
    #pragma unroll
    for (int n = 0; n < QV_NPW; n++)
        #pragma unroll
        for (int i = 0; i < 12; i++) s[n][i] = gadd(s[n][i], RC_TERMINAL[0][i]);
}

/* hash_out0 for positions t .. t + QV_NPW - 1 */
void hash_out0_n(__constant const ulong* pk, uint t, ulong* o)
{
    ulong s[QV_NPW][12];
    #pragma unroll
    for (int n = 0; n < QV_NPW; n++) {
        const ulong d = 35ul * (t + (uint)n);
        const ulong f = sbox(gadd(pk[0], 0xFFFFFFFF00000001ul - d));
        const ulong g = sbox(gadd(pk[1], d));
        const W c2 = w2(f, g);
        W c[3];
        c[0] = wadd64(c2, f);
        c[1] = wadd64(c2, g);
        c[2] = c2;
        const int m0[4] = {2, 1, 1, 3};
        #pragma unroll
        for (int k = 0; k < 3; k++)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                W v = c[k];
                if (m0[j] >= 2) v = wadd(v, c[k]);
                if (m0[j] == 3) v = wadd(v, c[k]);
                s[n][4 * k + j] = wred(wadd64(v, pk[2 + 4 * k + j]));
            }
    }
#pragma unroll 1
    for (int g = 1;; g++) {
        #pragma unroll
        for (int i = 0; i < 12; i++)
            #pragma unroll
            for (int n = 0; n < QV_NPW; n++) s[n][i] = sbox(s[n][i]);
        if (g == 15) break;
        #pragma unroll
        for (int n = 0; n < QV_NPW; n++) ext_add(s[n], post_row(g));
        if (g == 7) {
            #pragma unroll
            for (int n = 0; n < QV_NPW; n++) ext_add(s[n], post_row(15));
        }
        if ((g & 7) == 3) internal22_n(s);
    }
    #pragma unroll
    for (int n = 0; n < QV_NPW; n++) o[n] = out0_of(s[n]);
}
#endif

uint bswap32(uint v) { return as_uint(as_uchar4(v).s3210); }

#if defined(QV_SG) && defined(cl_intel_subgroups)
#define QV_KERNEL_ATTR __attribute__((intel_reqd_sub_group_size(QV_SG)))
#else
#define QV_KERNEL_ATTR
#endif

/* out[0] = candidate count, out[1..15] = index (line position tb + index) */
__kernel QV_KERNEL_ATTR void qpow_scan(__global volatile uint* out, __global ulong* dump, __constant ulong* pk,
                        ulong t0, uint tb, uint count)
{
    const uint stride = (uint)get_global_size(0);
#if QV_NPW > 1
    /* the host launches about count / QV_NPW work-items */
    for (uint base = (uint)get_global_id(0) * QV_NPW; base < count; base += stride * QV_NPW) {
        ulong os[QV_NPW];
        hash_out0_n(pk, tb + base, os);
        #pragma unroll
        for (int n = 0; n < QV_NPW; n++) {
            const uint idx = base + (uint)n;
            const ulong o = os[n];
            const ulong key = ((ulong)bswap32((uint)o) << 32) | bswap32((uint)(o >> 32));
            if (idx < count) {
                if (dump) dump[idx] = o;
                if (key <= t0) {
                    const uint slot = atomic_inc(&out[0]);
                    if (slot < 15u) out[1 + slot] = idx;
                }
            }
        }
    }
#else
    for (uint idx = (uint)get_global_id(0); idx < count; idx += stride) {
        const ulong o = hash_out0(pk, tb + idx);
        const ulong key = ((ulong)bswap32((uint)o) << 32) | bswap32((uint)(o >> 32));
        if (dump) dump[idx] = o;
        if (key <= t0) {
            const uint slot = atomic_inc(&out[0]);
            if (slot < 15u) out[1 + slot] = idx;
        }
    }
#endif
}

#ifdef QV_TEST
/* out[5*i + k]: k = 0 mul, 1 sqr, 2 fma with a lazy sum (b, c), 3 add, 4 lazy 2a + b + c */
__kernel void field_test(__global const ulong* a, __global const ulong* b, __global const ulong* c,
                         __global ulong* out, uint n)
{
    const uint i = (uint)get_global_id(0);
    if (i >= n) return;
    out[5 * i + 0] = canon(gmul(a[i], b[i]));
    out[5 * i + 1] = canon(gsqr(a[i]));
    W w = w2(c[i], c[i]);
    out[5 * i + 2] = canon(gfma_w(a[i], b[i], w));
    out[5 * i + 3] = canon(gadd(a[i], b[i]));
    out[5 * i + 4] = canon(wred(wadd(w2(a[i], b[i]), w2(c[i], a[i]))));
}
#endif
