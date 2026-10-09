// DSP side of the CPPminer Hexagon backend: Pearl GEMM + milestone XOR on the
// Hexagon v66 cDSP with 4 x 64 hash tiles (gemm_xor Case 1.1). B is packed
// once per job (set_b) and kept in DSP memory; each row panel of A is packed
// once (set_a), then gemm_xor runs it against one block of B's columns at a
// time. See ../../README.md.

#include <stdlib.h>
#include <string.h>
#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "HAP_vtcm_mgr.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "qurt.h"
#include "worker_pool.h"
#include "pearlx.h"

#define VBYTES  128                   // HVX vector length in 128-byte mode
#define MR      4                     // rows per register tile
#define NV      4                     // vectors per register-tile row
#define NT_COLS (NV * 32)             // 128 columns per register tile
#define SPLIT   2                     // hash tiles (4 x 64) per register tile
#define KZ_G    256                   // K block, in groups of 4 values of k (1024)
#define ZSLOT   (KZ_G * NV * VBYTES)  // one thread's B slice in VTCM: 128 KB
#define MS_CH   4                     // chunks (32 values of k) per milestone
// Padding row tiles after each region: after its last tile, pearlx_rows reads
// the next tile's A and starting values and prefetches the one after.
#define PAD_TILES 2

typedef struct {
    worker_pool_context_t pool;
    int clock_vote_err;
    // Packed B of the current job (set_b): [Nt][Kg][NV] vectors.
    HVX_Vector *Bz;
    int n, k;
    // Packed A of the current row panel (set_a); the buffer is reused while it
    // is big enough. a_rows = 0: no panel.
    uint8 *Az;
    size_t az_cap;
    int a_rows, a_threads;
    // VTCM for the B slices, one 128 KB slot per thread, held between calls.
    uint8 *vtcm;
    int vtcm_threads;
    // scan: each thread's records of one column tile, checked on the DSP.
    uint8 *jp_rec;
    size_t jp_cap;
} pearlx_ctx;

// Pin core and bus clocks at their highest corner and keep the DSP out of
// low-power modes, so timings don't depend on DCVS ramp-up.
static int vote_max_clocks(void *ctx) {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = TRUE;
    req.dcvs_v3.dcvs_enable = FALSE;
    req.dcvs_v3.set_core_params = TRUE;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_bus_params = TRUE;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = TRUE;
    req.dcvs_v3.sleep_disable = HAP_DCVS_LPM_LEVEL1;
    int err = HAP_power_set(ctx, &req);
    if (err)
        FARF(ERROR, "pearlx: HAP_power_set failed 0x%x, timings will vary", err);
    return err;
}

int pearlx_open(const char *uri, remote_handle64 *handle) {
    // The context doubles as the HAP_power client handle, so it must be unique.
    pearlx_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return AEE_ENOMEMORY;
    ctx->clock_vote_err = vote_max_clocks(ctx);
    // Without a pool, worker_pool_submit runs jobs on the calling thread.
    if (worker_pool_init(&ctx->pool)) {
        FARF(ERROR, "pearlx: worker_pool_init failed, running single-threaded");
        ctx->pool = NULL;
    }
    *handle = (remote_handle64)ctx;
    return AEE_SUCCESS;
}

int pearlx_close(remote_handle64 handle) {
    pearlx_ctx *ctx = (pearlx_ctx *)handle;
    if (ctx) {
        if (ctx->pool)
            worker_pool_deinit(&ctx->pool);
        free(ctx->Bz);
        free(ctx->Az);
        free(ctx->jp_rec);
        if (ctx->vtcm)
            HAP_release_VTCM(ctx->vtcm);
        HAP_power_request_t req;
        HAP_power_set_dcvs_v3_init(&req);   // restores default DCVS behaviour
        HAP_power_set(ctx, &req);
        free(ctx);
    }
    return AEE_SUCCESS;
}

int pearlx_info(remote_handle64 h, int *hvx_contexts, int *clock_vote_err, int *clock_mhz,
                int *vtcm_kb) {
    unsigned int page_size = 0, page_count = 0;
    *vtcm_kb = HAP_query_total_VTCM(&page_size, &page_count) ? -1 : (int)(page_size * page_count / 1024);
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    *hvx_contexts = num_hvx128_contexts;
    *clock_vote_err = ctx->clock_vote_err;
    // Count processor cycles across a 2 ms busy-wait.
    uint64 c0 = HAP_perf_get_pcycles(), t0 = HAP_perf_get_time_us(), t1;
    while ((t1 = HAP_perf_get_time_us()) - t0 < 2000)
        ;
    *clock_mhz = (int)((HAP_perf_get_pcycles() - c0) / (t1 - t0));
    return AEE_SUCCESS;
}

typedef struct {
    void (*fn)(void *arg);
    void *arg;
    int err;
    worker_synctoken_t *token;
} hvx_job;

static void hvx_job_run(void *p) {
    hvx_job *job = p;
    // On v66 a thread must hold an HVX context before executing HVX code.
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK) {
        job->fn(job->arg);
        qurt_hvx_unlock();
    } else {
        job->err = AEE_EFAILED;
    }
    worker_pool_synctoken_jobdone(job->token);
}

// Runs fn(args[t]) for t in [0, n) on the worker pool, each holding an HVX
// context, and waits for all of them. args is an array of n elements of arg_size bytes.
static int run_hvx_jobs(pearlx_ctx *ctx, int n, void (*fn)(void *), void *args, size_t arg_size) {
    hvx_job jobs[MAX_NUM_WORKERS];
    worker_synctoken_t token;
    if (n < 1 || n > MAX_NUM_WORKERS)
        return AEE_EBADPARM;
    worker_pool_synctoken_init(&token, n);
    for (int t = 0; t < n; t++) {
        jobs[t] = (hvx_job){ fn, (char *)args + t * arg_size, AEE_SUCCESS, &token };
        worker_pool_job_t job = { hvx_job_run, &jobs[t] };
        worker_pool_submit(ctx->pool, job);
    }
    worker_pool_synctoken_wait(&token);
    for (int t = 0; t < n; t++)
        if (jobs[t].err)
            return jobs[t].err;
    return AEE_SUCCESS;
}

static int hvx_threads(int requested) {
    int max_threads = num_hvx128_contexts > 0 ? (int)num_hvx128_contexts : 1;
    if (max_threads > MAX_NUM_WORKERS)
        max_threads = MAX_NUM_WORKERS;
    return requested <= 0 || requested > max_threads ? max_threads : requested;
}

// Asynchronous prefetch of a height x width-byte box (row pitch stride) into L2.
// A new l2fetch replaces one still in flight on the same hardware thread.
static inline void l2fetch(const void *p, uint32 stride, uint32 width, uint32 height) {
    uint64 ctrl = ((uint64)stride << 32) | ((uint64)width << 16) | height;
    __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctrl));
}

struct pearlx_rows_args {
    const uint8 *az;
    int az_step;                    // bytes from one tile's A to the next
    const HVX_Vector *bz;
    int nmile;                      // milestones in this K block, 1..8
    const HVX_Vector *in;
    int in_step;                    // 0: same starting values for every tile
    HVX_Vector *out;
    int out_step;                   // 0: every tile's final values to one place
    uint32 *xo;
    int xo_step;
    int ntiles;
    int rot;                        // vror amount placing the 16 results
    uint32 l2_lo, l2_hi;            // l2fetch box run at az of the next tile
    int qlo, qhi;                   // bytes of the record line to write
};
void pearlx_rows(const struct pearlx_rows_args *a);

typedef struct {
    // Az: one region per K block of kblk chunks (32 values of k each; the last
    // block may be shorter). Region kb holds, for each of Mt + PAD_TILES row
    // tiles (the last ones are padding), gx_rstride(kb) bytes:
    //   [nch][128] A: byte 32r+kk of chunk c = A[4t+r][32(kb*kblk+c)+kk], signed,
    //   then, in regions after the first, [nslots][MR*NV] partial sums: the
    //   starting accumulators of pass kb, written by pass kb - 1.
    // A pass streams its own region; one l2fetch row prefetches a tile's A
    // slice and incoming partial sums together.
    uint8 *Az;
    const HVX_Vector *Bz;   // [Nt][Kg][NV]: vector (g, q) = B[4g..4g+3][128nt+32q+lane], signed
    uint32 *xr;             // [Nt][Mt][lines * 32]: word ms * 2 + h = XOR of hash
                            // tile h of the register tile after milestone ms
    int M, N, K, Kg, Kch;
    int Mt, Nt, kblk, nkb;  // row tiles; column tiles; chunks per K block; K blocks
    int nt0;                // first column tile of a gemm_xor call (Nt counts from it)
    int sbytes;             // partial-sum bytes per tile, in regions after the first
    int lines;              // 128-byte lines per XOR record
} gx_problem;

// Shapes that depend only on n and k.
static void gx_shape(gx_problem *p, int n, int k) {
    p->N = n;
    p->K = k;
    p->Kch = k / 32;                         // k is a multiple of 128
    p->Kg = p->Kch * 8;
    p->kblk = p->Kch < KZ_G / 8 ? p->Kch : KZ_G / 8;
    p->nkb = (p->Kch + p->kblk - 1) / p->kblk;
    p->Nt = n / NT_COLS;
    p->lines = (p->Kch / MS_CH * SPLIT + 31) / 32;
}

// Chunks of A per tile in region kb.
static inline int gx_nch(const gx_problem *p, int kb) {
    const int n = p->Kch - kb * p->kblk;
    return n < p->kblk ? n : p->kblk;
}

static inline int gx_rstride(const gx_problem *p, int kb) {
    return gx_nch(p, kb) * VBYTES + (kb > 0 ? p->sbytes : 0);
}

// Regions before kb are full length; only region 0 lacks partial sums.
static inline uint8 *gx_region(const gx_problem *p, int kb) {
    return p->Az + (size_t)(p->Mt + PAD_TILES) *
                   ((size_t)kb * p->kblk * VBYTES + (size_t)(kb > 0 ? kb - 1 : 0) * p->sbytes);
}

static inline size_t gx_az_bytes(const gx_problem *p) {
    return (size_t)(gx_region(p, p->nkb - 1) - p->Az) +
           (size_t)(p->Mt + PAD_TILES) * gx_rstride(p, p->nkb - 1);
}

// Row tile t's A in region kb.
static inline uint8 *gx_a(const gx_problem *p, int kb, int t) {
    return gx_region(p, kb) + (size_t)t * gx_rstride(p, kb);
}

// Partial sums of row tile t in region kb (kb > 0), for column-tile slot sl.
static inline HVX_Vector *gx_s(const gx_problem *p, int kb, int sl, int t) {
    return (HVX_Vector *)(gx_a(p, kb, t) + gx_nch(p, kb) * VBYTES) + sl * MR * NV;
}

typedef struct {
    const int8 *A;
    const gx_problem *p;
    int t0, t1;              // row tiles
} packx_a_job;

// A (row-major, rows a multiple of 4) into Az: 4 rows x 32 values of k per
// 128-byte chunk.
static void packx_a_run(void *arg) {
    const packx_a_job *j = arg;
    const gx_problem *p = j->p;
    const int K = p->K;

    for (int t = j->t0; t < j->t1; t++) {
        const int i0 = t * MR;
        if (t + 1 < j->t1)
            l2fetch(j->A + (size_t)(i0 + MR) * K, K, K, MR);

        for (int c4 = 0; c4 < p->Kch / 4; c4++) {          // 128 values of k = 4 chunks
            const int k0 = c4 * 128;
            const int kb = 4 * c4 / p->kblk;                // kblk is a multiple of 4
            HVX_Vector *out =
                (HVX_Vector *)(gx_a(p, kb, t) + (size_t)(4 * c4 - kb * p->kblk) * VBYTES);
            HVX_Vector r[MR];
            for (int s = 0; s < MR; s++)
                r[s] = *(const HVX_UVector *)(j->A + (size_t)(i0 + s) * K + k0);
            // Interleave 32-byte pieces of rows 0/1 and 2/3, then 64-byte
            // pairs: chunk c = r0[32c..] r1[32c..] r2[32c..] r3[32c..].
            HVX_VectorPair p01 = Q6_W_vshuff_VVR(r[1], r[0], -32);
            HVX_VectorPair p23 = Q6_W_vshuff_VVR(r[3], r[2], -32);
            HVX_VectorPair q0 = Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
            HVX_VectorPair q1 = Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
            out[0] = Q6_V_lo_W(q0);
            out[1] = Q6_V_hi_W(q0);
            out[2] = Q6_V_lo_W(q1);
            out[3] = Q6_V_hi_W(q1);
        }
    }
}

typedef struct {
    const int8 *Bt;          // row j = column j of B, K values
    const gx_problem *p;
    HVX_Vector *Bz;
    int nt0, nt1;            // column tiles
} packx_b_job;

// B^T into Bz, 32 groups x 32 columns at a time: load 128 bytes (32 words,
// one per group) of each of 32 columns, transpose the 32 x 32 words with five
// rounds of vshuff on row pairs 16, 8, 4, 2, 1 apart, and store one vector per
// group. n and k are multiples of 128, so every block is full.
static void packx_b_run(void *arg) {
    const packx_b_job *j = arg;
    const gx_problem *p = j->p;
    const int K = p->K, Kg = p->Kg;

    for (int nt = j->nt0; nt < j->nt1; nt++) {
        HVX_Vector *out = j->Bz + (size_t)nt * Kg * NV;
        for (int g0 = 0; g0 < Kg; g0 += 32) {               // Kg is a multiple of 32
            for (int q = 0; q < NV; q++) {
                const int col0 = nt * NT_COLS + 32 * q;
                HVX_Vector *o = out + (size_t)g0 * NV + q;
                const int8 *src = j->Bt + (size_t)col0 * K + 4 * g0;
                HVX_Vector r[32];
                for (int c = 0; c < 32; c++)
                    r[c] = *(const HVX_UVector *)(src + (size_t)c * K);
                for (int d = 16; d >= 1; d /= 2)
                    for (int i = 0; i < 32; i++)
                        if (!(i & d)) {
                            HVX_VectorPair w = Q6_W_vshuff_VVR(r[i + d], r[i], -4);
                            r[i] = Q6_V_lo_W(w);
                            r[i + d] = Q6_V_hi_W(w);
                        }
                for (int g = 0; g < 32; g++)
                    o[(size_t)g * NV] = r[g];
            }
        }
    }
}

// ---- Jackpot on the DSP: keyed BLAKE3 of each hash tile's folded milestone XORs ----
// 32 hash tiles per vector, one per 32-bit lane: lane 2r + h is hash tile h of record r
// in a group of 16 records. Matches cp_jackpot.hpp (fold with rotl 13 into 16 words, one
// keyed compression with flags CHUNK_START | CHUNK_END | ROOT | KEYED_HASH, digest <= bound
// as a little-endian 256-bit number).

#define JP_MS 32                    // milestones (k = 4096, one per 128)
#define JP_REC_WORDS 64             // words per register-tile record: ms * 2 + h

typedef struct {
    int hits;                       // hash tiles at or below the bound
    int t, nt, h;                   // the first hit: row tile, column tile, half
    uint32 words[JP_MS];            // its milestone XORs
} jp_result;

static const uint32 b3_iv[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

#define VADD(a, b) Q6_Vw_vadd_VwVw(a, b)
#define VXOR(a, b) Q6_V_vxor_VV(a, b)
#define VROTR(a, r) Q6_Vuw_vrotr_VuwVuw(a, r)

static inline void b3_g(HVX_Vector *v, int a, int b, int c, int d, HVX_Vector x, HVX_Vector y,
                        HVX_Vector r16, HVX_Vector r12, HVX_Vector r8, HVX_Vector r7) {
    v[a] = VADD(VADD(v[a], v[b]), x);
    v[d] = VROTR(VXOR(v[d], v[a]), r16);
    v[c] = VADD(v[c], v[d]);
    v[b] = VROTR(VXOR(v[b], v[c]), r12);
    v[a] = VADD(VADD(v[a], v[b]), y);
    v[d] = VROTR(VXOR(v[d], v[a]), r8);
    v[c] = VADD(v[c], v[d]);
    v[b] = VROTR(VXOR(v[b], v[c]), r7);
}

// Hash tiles of n_rec records (n_rec <= 16) at rec, consecutive JP_REC_WORDS-word records:
// returns a predicate of the lanes at or below the bound.
static HVX_VectorPred jp_group(const uint32 *rec, int n_rec, const uint32 key[8],
                               const uint32 bound[8]) {
    // Each record is two vectors: milestones 0..15 and 16..31, as (h = 0, h = 1) word pairs.
    // A 16 x 16 transpose of those pairs gives x[ms] with lane 2r + h = record r's half h.
    HVX_Vector x[JP_MS];
    for (int half = 0; half < 2; half++) {
        HVX_Vector *r = x + 16 * half;
#pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++)
            r[i] = i < n_rec ? ((const HVX_Vector *)(rec + (size_t)i * JP_REC_WORDS))[half]
                             : Q6_V_vzero();
#pragma clang loop unroll(full)
        for (int d = 8; d >= 1; d /= 2)
#pragma clang loop unroll(full)
            for (int i = 0; i < 16; i++)
                if (!(i & d)) {
                    HVX_VectorPair w = Q6_W_vshuff_VVR(r[i + d], r[i], -8);
                    r[i] = Q6_V_lo_W(w);
                    r[i + d] = Q6_V_hi_W(w);
                }
    }
    const HVX_Vector r19 = Q6_V_vsplat_R(19), r16 = Q6_V_vsplat_R(16), r12 = Q6_V_vsplat_R(12);
    const HVX_Vector r8 = Q6_V_vsplat_R(8), r7 = Q6_V_vsplat_R(7);
    // Fold: msg[i] = rotl(x[i], 13) ^ x[i + 16].
    HVX_Vector m[16];
#pragma clang loop unroll(full)
    for (int i = 0; i < 16; i++)
        m[i] = VXOR(VROTR(x[i], r19), x[i + 16]);
    HVX_Vector v[16];
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++)
        v[i] = Q6_V_vsplat_R(key[i]);
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++)
        v[8 + i] = Q6_V_vsplat_R(b3_iv[i]);
    v[12] = Q6_V_vzero();
    v[13] = Q6_V_vzero();
    v[14] = Q6_V_vsplat_R(64);
    v[15] = Q6_V_vsplat_R(0x1B);
    // Rounds fully unrolled with constant message indices, so the message stays in
    // registers instead of being picked from the stack by a run-time schedule.
#define B3_ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)     b3_g(v, 0, 4, 8, 12, m[s0], m[s1], r16, r12, r8, r7);                                 b3_g(v, 1, 5, 9, 13, m[s2], m[s3], r16, r12, r8, r7);                                 b3_g(v, 2, 6, 10, 14, m[s4], m[s5], r16, r12, r8, r7);                                b3_g(v, 3, 7, 11, 15, m[s6], m[s7], r16, r12, r8, r7);                                b3_g(v, 0, 5, 10, 15, m[s8], m[s9], r16, r12, r8, r7);                                b3_g(v, 1, 6, 11, 12, m[s10], m[s11], r16, r12, r8, r7);                              b3_g(v, 2, 7, 8, 13, m[s12], m[s13], r16, r12, r8, r7);                               b3_g(v, 3, 4, 9, 14, m[s14], m[s15], r16, r12, r8, r7);
    B3_ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    B3_ROUND(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
    B3_ROUND(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1)
    B3_ROUND(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6)
    B3_ROUND(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4)
    B3_ROUND(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7)
    B3_ROUND(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13)
#undef B3_ROUND
    // digest[w] = v[w] ^ v[w + 8]; compare from the most significant word down.
    HVX_VectorPred lt = Q6_Q_vsetq_R(0), gt = Q6_Q_vsetq_R(0);
#pragma clang loop unroll(full)
    for (int w = 7; w >= 0; w--) {
        const HVX_Vector dg = VXOR(v[w], v[w + 8]), bd = Q6_V_vsplat_R(bound[w]);
        const HVX_VectorPred open = Q6_Q_not_Q(Q6_Q_or_QQ(lt, gt));
        lt = Q6_Q_or_QQ(lt, Q6_Q_and_QQ(open, Q6_Q_vcmp_gt_VuwVuw(bd, dg)));
        gt = Q6_Q_or_QQ(gt, Q6_Q_and_QQ(open, Q6_Q_vcmp_gt_VuwVuw(dg, bd)));
    }
    // Lanes past the last record were zero-filled: never hits.
    return Q6_Q_and_QQn(Q6_Q_vsetq2_R(8 * n_rec), gt);   /* vsetq2: 128 means all */
}

// Jackpot over records [0, n) of one column tile (record i = row tile t0 + i).
static void jp_records(const uint32 *rec, int n, int t0, int nt, const uint32 key[8],
                       const uint32 bound[8], jp_result *res) {
    HVX_Vector lanes __attribute__((aligned(VBYTES)));
    for (int r0 = 0; r0 < n; r0 += 16) {
        const int cnt = n - r0 < 16 ? n - r0 : 16;
        const HVX_VectorPred hit = jp_group(rec + (size_t)r0 * JP_REC_WORDS, cnt, key, bound);
        lanes = Q6_V_vand_QR(hit, 0x01010101);
        const uint32 *lw = (const uint32 *)&lanes;
        for (int l = 0; l < 32; l++) {
            if (!lw[l])
                continue;
            const int r = r0 + l / 2, h = l % 2;
            if (res->hits++ == 0 || nt < res->nt || (nt == res->nt && (t0 + r < res->t ||
                                                     (t0 + r == res->t && h < res->h)))) {
                res->t = t0 + r;
                res->nt = nt;
                res->h = h;
                const uint32 *w = rec + (size_t)r * JP_REC_WORDS;
                for (int ms = 0; ms < JP_MS; ms++)
                    res->words[ms] = w[ms * 2 + h];
            }
        }
    }
}

// Starting accumulators of the first K block.
static const HVX_Vector zero_acc[MR * NV];

typedef struct {
    HVX_Vector tmp[MR * NV];        // final accumulators of the last K block go here
    const gx_problem *p;
    HVX_Vector *slot;               // this thread's part of VTCM
    int nt, sl;                     // column tile, and its slot in the partial sums
    int t0, t1;                     // row tiles
    int kb;                         // K block
    uint32 *xr_local;               // jackpot mode: this thread's records, tile t0 first
} gemmx_job;

static void gemmx_job_run(gemmx_job *j) {
    const gx_problem *p = j->p;
    const int kb = j->kb, nch = gx_nch(p, kb), nmile = nch / MS_CH;
    const int last = kb == p->nkb - 1;
    const int rs = gx_rstride(p, kb);

    // This column tile's slice of B is contiguous in Bz: copy it to VTCM.
    const HVX_Vector *src =
        p->Bz + ((size_t)(p->nt0 + j->nt) * p->Kg + (size_t)kb * p->kblk * 8) * NV;
    for (int i = 0; i < nch * 8 * NV; i++)
        j->slot[i] = src[i];

    // The block's milestones 8kb .. 8kb + nmile - 1 land in words
    // 16(kb % 2) .. of record line kb / 2 (two words per milestone). The
    // kernel leaves them in lanes 2(8 - nmile) .. 15, so they move up by s
    // words; vror rotates bytes down.
    const int w = 8 * SPLIT, bpl = 32 / w;
    const int lane0 = w * (kb % bpl);
    const int s = (lane0 - SPLIT * (8 - nmile) + 32) % 32;
    const uint32 width = kb > 0 ? (uint32)rs : (uint32)nch * VBYTES;
    struct pearlx_rows_args a = {
        .az = gx_a(p, kb, j->t0), .az_step = rs, .bz = j->slot, .nmile = nmile,
        .in = kb == 0 ? zero_acc : gx_s(p, kb, j->sl, j->t0), .in_step = kb == 0 ? 0 : rs,
        .out = last ? j->tmp : gx_s(p, kb + 1, j->sl, j->t0),
        .out_step = last ? 0 : gx_rstride(p, kb + 1),
        .xo = j->xr_local ? j->xr_local + (size_t)(kb / bpl) * 32
                          : p->xr + ((size_t)(j->nt * p->Mt + j->t0) * p->lines + kb / bpl) * 32,
        .xo_step = p->lines * VBYTES,
        .ntiles = j->t1 - j->t0,
        .rot = (VBYTES - 4 * s) % VBYTES,
        .l2_lo = width << 16 | 1,
        .l2_hi = width,
        .qlo = 4 * lane0,
        .qhi = 4 * (lane0 + SPLIT * nmile),
    };
    if (a.ntiles > 0)
        pearlx_rows(&a);
}

// One thread's whole share, so a call needs one dispatch. Full rounds give
// thread t its own column tiles (t, t + n, ...); the tiles left over are then
// split by rows. Threads never share rows of a column tile or a partial-sum
// slot, so they need no barrier between rounds.
typedef struct {
    gemmx_job job;          // first, so its tmp vectors stay 128-byte aligned
    int t, n, nfull;        // this thread, thread count, tiles done in full rounds
    // Jackpot mode (job.xr_local set): check each column tile's records as it finishes.
    const uint32 *key, *bound;
    int stop_on_hit;
    volatile int *stop;     // shared: set by the first thread to hit when stop_on_hit
    jp_result res;
} gemmx_thread;

static void gemmx_thread_run(void *arg) {
    gemmx_thread *th = arg;
    gemmx_job *j = &th->job;
    const gx_problem *p = j->p;
    for (int nt = th->nfull > 0 ? th->t : 0; nt < p->Nt; ) {
        if (th->stop && *th->stop)
            break;
        const int full = nt < th->nfull;
        j->nt = nt;
        j->sl = th->t;
        j->t0 = full ? 0 : p->Mt * th->t / th->n;
        j->t1 = full ? p->Mt : p->Mt * (th->t + 1) / th->n;
        for (j->kb = 0; j->kb < p->nkb; j->kb++)
            gemmx_job_run(j);
        if (j->xr_local) {
            jp_records(j->xr_local, j->t1 - j->t0, j->t0, nt, th->key, th->bound, &th->res);
            if (th->stop_on_hit && th->res.hits)
                *th->stop = 1;
        }
        // After the full rounds, every thread takes each leftover tile.
        nt = full && nt + th->n < th->nfull ? nt + th->n : (full ? th->nfull : nt + 1);
    }
}

int pearlx_set_b(remote_handle64 h, const int8 *Bt, int BtLen, int n, int k, uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (n <= 0 || k <= 0 || n % NT_COLS || k % 128 || (int64)n * k > BtLen)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();

    free(ctx->Bz);
    ctx->Bz = NULL;
    ctx->n = ctx->k = 0;
    ctx->a_rows = 0;                        // a panel packed for another k is stale
    gx_problem p;
    memset(&p, 0, sizeof(p));
    gx_shape(&p, n, k);
    HVX_Vector *Bz = memalign(VBYTES, (size_t)n * k);
    if (!Bz) {
        FARF(ERROR, "pearlx: no memory for packed B (%d x %d)", n, k);
        return AEE_ENOMEMORY;
    }

    const int nthreads = hvx_threads(0);
    const int nb = nthreads < p.Nt ? nthreads : p.Nt;
    packx_b_job pb[MAX_NUM_WORKERS];
    for (int t = 0; t < nb; t++)
        pb[t] = (packx_b_job){ Bt, &p, Bz, p.Nt * t / nb, p.Nt * (t + 1) / nb };
    int err = run_hvx_jobs(ctx, nb, packx_b_run, pb, sizeof(pb[0]));
    if (err) {
        free(Bz);
        return err;
    }
    ctx->Bz = Bz;
    ctx->n = n;
    ctx->k = k;
    *dsp_us = HAP_perf_get_time_us() - t0;
    return AEE_SUCCESS;
}


// Shape of the packed row panel (set_a) against the job's B.
static void panel_problem(const pearlx_ctx *ctx, gx_problem *p) {
    memset(p, 0, sizeof(*p));
    gx_shape(p, ctx->n, ctx->k);
    p->M = ctx->a_rows;
    p->Mt = ctx->a_rows / MR;
    p->sbytes = ctx->a_threads * MR * NV * VBYTES;
    p->Bz = ctx->Bz;
    p->Az = ctx->Az;
}

int pearlx_set_a(remote_handle64 h, const int8 *A, int ALen, int rows, int nthreads,
                 uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (!ctx->Bz)
        return AEE_EBADSTATE;
    if (rows <= 0 || rows % MR || (int64)rows * ctx->k > ALen)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    ctx->a_rows = 0;

    gx_problem p;
    ctx->a_threads = hvx_threads(nthreads);
    ctx->a_rows = rows;
    panel_problem(ctx, &p);
    const size_t az_bytes = gx_az_bytes(&p);
    if (az_bytes > ctx->az_cap) {
        free(ctx->Az);
        ctx->az_cap = 0;
        ctx->Az = memalign(VBYTES, az_bytes);
        if (!ctx->Az) {
            ctx->a_rows = 0;
            return AEE_ENOMEMORY;
        }
        ctx->az_cap = az_bytes;
    }
    p.Az = ctx->Az;

    // Each region's padding tiles: the kernel loads one chunk past the last
    // one it uses and runs past its last tile (see PAD_TILES).
    for (int kb = 0; kb < p.nkb; kb++)
        memset(gx_a(&p, kb, p.Mt), 0, (size_t)PAD_TILES * gx_rstride(&p, kb));

    packx_a_job pa[MAX_NUM_WORKERS];
    const int na = ctx->a_threads < p.Mt ? ctx->a_threads : p.Mt;
    for (int t = 0; t < na; t++)
        pa[t] = (packx_a_job){ A, &p, p.Mt * t / na, p.Mt * (t + 1) / na };
    int err = run_hvx_jobs(ctx, na, packx_a_run, pa, sizeof(pa[0]));
    if (err) {
        ctx->a_rows = 0;
        return err;
    }
    *dsp_us = HAP_perf_get_time_us() - t0;
    return AEE_SUCCESS;
}

// VTCM is kept from call to call; ask again only when the thread count grows.
static int ensure_vtcm(pearlx_ctx *ctx, int nthreads) {
    if (ctx->vtcm && ctx->vtcm_threads < nthreads) {
        HAP_release_VTCM(ctx->vtcm);
        ctx->vtcm = NULL;
    }
    if (!ctx->vtcm) {
        ctx->vtcm = HAP_request_VTCM(nthreads * ZSLOT, 1);
        if (!ctx->vtcm) {
            FARF(ERROR, "pearlx: no VTCM (%d bytes)", nthreads * ZSLOT);
            return AEE_EUNSUPPORTED;
        }
        ctx->vtcm_threads = nthreads;
    }
    return AEE_SUCCESS;
}

// The packed row panel against columns [col0, col0 + ncols): records to xr (xr != NULL),
// or the jackpot on the DSP (key and bound; the first hit and the hit count into *res).
static int run_panel(pearlx_ctx *ctx, int col0, int ncols, uint32 *xr, const uint32 *key,
                     const uint32 *bound, int stop_on_hit, jp_result *res) {
    const int nthreads = ctx->a_threads;
    gx_problem p;
    panel_problem(ctx, &p);
    p.nt0 = col0 / NT_COLS;
    p.Nt = ncols / NT_COLS;
    p.xr = xr;
    int err = ensure_vtcm(ctx, nthreads);
    if (err)
        return err;
    // Jackpot mode: one column tile's records per thread at a time.
    const size_t rec_bytes = (size_t)p.Mt * p.lines * VBYTES;
    if (!xr && ctx->jp_cap < rec_bytes * nthreads) {
        free(ctx->jp_rec);
        ctx->jp_cap = 0;
        ctx->jp_rec = memalign(VBYTES, rec_bytes * nthreads);
        if (!ctx->jp_rec)
            return AEE_ENOMEMORY;
        ctx->jp_cap = rec_bytes * nthreads;
    }

    gemmx_thread *jobs = memalign(VBYTES, MAX_NUM_WORKERS * sizeof(gemmx_thread));
    if (!jobs)
        return AEE_ENOMEMORY;
    volatile int stop = 0;
    for (int t = 0; t < nthreads; t++) {
        memset(&jobs[t], 0, sizeof(jobs[t]));
        jobs[t].job.p = &p;
        jobs[t].job.slot = (HVX_Vector *)(ctx->vtcm + (size_t)t * ZSLOT);
        jobs[t].t = t;
        jobs[t].n = nthreads;
        jobs[t].nfull = p.Nt / nthreads * nthreads;
        if (!xr) {
            jobs[t].job.xr_local = (uint32 *)(ctx->jp_rec + rec_bytes * t);
            jobs[t].key = key;
            jobs[t].bound = bound;
            jobs[t].stop_on_hit = stop_on_hit;
            jobs[t].stop = &stop;
        }
    }
    err = run_hvx_jobs(ctx, nthreads, gemmx_thread_run, jobs, sizeof(jobs[0]));
    if (!xr && !err) {
        // The first hit in (column tile, row tile, half) order, and the total.
        memset(res, 0, sizeof(*res));
        for (int t = 0; t < nthreads; t++) {
            const jp_result *r = &jobs[t].res;
            if (!r->hits)
                continue;
            if (!res->hits || r->nt < res->nt || (r->nt == res->nt && (r->t < res->t ||
                                                  (r->t == res->t && r->h < res->h)))) {
                const int total = res->hits;
                *res = *r;
                res->hits = total;
            }
            res->hits += r->hits;
        }
    }
    free(jobs);
    return err;
}

int pearlx_gemm_xor(remote_handle64 h, int col0, int ncols, uint32 *xr, int xrLen,
                    uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (!ctx->Bz || !ctx->a_rows)
        return AEE_EBADSTATE;
    if (col0 < 0 || ncols <= 0 || col0 % NT_COLS || ncols % NT_COLS || col0 + ncols > ctx->n)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    gx_problem p;
    panel_problem(ctx, &p);
    if ((int64)(ncols / NT_COLS) * p.Mt * p.lines * 32 > xrLen || ((uintptr_t)xr & (VBYTES - 1)))
        return AEE_EBADPARM;
    int err = run_panel(ctx, col0, ncols, xr, NULL, NULL, 0, NULL);
    *dsp_us = HAP_perf_get_time_us() - t0;
    return err;
}

int pearlx_scan(remote_handle64 h, int col0, int ncols, const uint32 *key_bound, int kbLen,
                int stop_on_hit, uint32 *hit, int hitLen, uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (!ctx->Bz || !ctx->a_rows)
        return AEE_EBADSTATE;
    if (col0 < 0 || ncols <= 0 || col0 % NT_COLS || ncols % NT_COLS || col0 + ncols > ctx->n ||
        kbLen < 16 || hitLen < 3 + JP_MS || ctx->k != JP_MS * 128)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    jp_result res;
    int err = run_panel(ctx, col0, ncols, NULL, key_bound, key_bound + 8, stop_on_hit, &res);
    if (!err) {
        hit[0] = (uint32)res.hits;
        hit[1] = (uint32)(res.hits ? res.t * MR : 0);
        hit[2] = (uint32)(res.hits ? col0 + res.nt * NT_COLS + res.h * (NT_COLS / SPLIT) : 0);
        for (int ms = 0; ms < JP_MS; ms++)
            hit[3 + ms] = res.hits ? res.words[ms] : 0;
    }
    *dsp_us = HAP_perf_get_time_us() - t0;
    return err;
}
