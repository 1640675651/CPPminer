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

// Starting accumulators of the first K block.
static const HVX_Vector zero_acc[MR * NV];

typedef struct {
    HVX_Vector tmp[MR * NV];        // final accumulators of the last K block go here
    const gx_problem *p;
    HVX_Vector *slot;               // this thread's part of VTCM
    int nt, sl;                     // column tile, and its slot in the partial sums
    int t0, t1;                     // row tiles
    int kb;                         // K block
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
        .xo = p->xr + ((size_t)(j->nt * p->Mt + j->t0) * p->lines + kb / bpl) * 32,
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
} gemmx_thread;

static void gemmx_thread_run(void *arg) {
    gemmx_thread *th = arg;
    gemmx_job *j = &th->job;
    const gx_problem *p = j->p;
    for (int nt = th->nfull > 0 ? th->t : 0; nt < p->Nt; ) {
        const int full = nt < th->nfull;
        j->nt = nt;
        j->sl = th->t;
        j->t0 = full ? 0 : p->Mt * th->t / th->n;
        j->t1 = full ? p->Mt : p->Mt * (th->t + 1) / th->n;
        for (j->kb = 0; j->kb < p->nkb; j->kb++)
            gemmx_job_run(j);
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

int pearlx_gemm_xor(remote_handle64 h, int col0, int ncols, uint32 *xr, int xrLen,
                    uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (!ctx->Bz || !ctx->a_rows)
        return AEE_EBADSTATE;
    if (col0 < 0 || ncols <= 0 || col0 % NT_COLS || ncols % NT_COLS || col0 + ncols > ctx->n)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    const int nthreads = ctx->a_threads;

    gx_problem p;
    panel_problem(ctx, &p);
    p.nt0 = col0 / NT_COLS;
    p.Nt = ncols / NT_COLS;
    if ((int64)p.Nt * p.Mt * p.lines * 32 > xrLen || ((uintptr_t)xr & (VBYTES - 1)))
        return AEE_EBADPARM;
    p.xr = xr;

    // VTCM is kept from call to call; ask again only when the thread count grows.
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

    gemmx_thread *jobs = memalign(VBYTES, MAX_NUM_WORKERS * sizeof(gemmx_thread));
    if (!jobs)
        return AEE_ENOMEMORY;
    for (int t = 0; t < nthreads; t++) {
        memset(&jobs[t], 0, sizeof(jobs[t]));
        jobs[t].job.p = &p;
        jobs[t].job.slot = (HVX_Vector *)(ctx->vtcm + (size_t)t * ZSLOT);
        jobs[t].t = t;
        jobs[t].n = nthreads;
        jobs[t].nfull = p.Nt / nthreads * nthreads;
    }
    int err = run_hvx_jobs(ctx, nthreads, gemmx_thread_run, jobs, sizeof(jobs[0]));
    free(jobs);
    *dsp_us = HAP_perf_get_time_us() - t0;
    return err;
}
