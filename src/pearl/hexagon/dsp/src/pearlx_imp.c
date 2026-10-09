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

// scan_run's control block, shared with the host during the call (see pearlx.idl). Each
// side writes only its own half, a separate cache line, so neither side's write-back can
// overwrite the other's words: the DSP flushes its half after writing it and invalidates
// the host's half before reading it.
#define CTL_LAUNCHED  0
#define CTL_EPOCH_ACK 1
#define CTL_STATE     2
#define CTL_HOST      32                // first word of the host's half
#define CTL_CHECKED   32
#define CTL_STOP      33
#define CTL_EPOCH     34
#define CTL_TEST_IDLE_US 35             // self-test: idle this long after each launch
#define CTL_WORDS     64
enum { CTL_RUNNING = 1, CTL_DONE = 2, CTL_STOPPED = 3, CTL_FAILED = 4 };
#define CTL_POLL_US   100               // while waiting for the host to check a launch

static inline void ctl_pull(volatile uint32 *ctl) {
    qurt_mem_cache_clean((qurt_addr_t)(ctl + CTL_HOST), (CTL_WORDS - CTL_HOST) * 4,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
}

static inline void ctl_push(volatile uint32 *ctl) {
    qurt_mem_cache_clean((qurt_addr_t)ctl, CTL_HOST * 4, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

static inline int ctl_stop_requested(volatile uint32 *ctl) {
    ctl_pull(ctl);
    return ctl[CTL_STOP] != 0;
}

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
    // set_b_gen / set_a_gen: per-thread noise tables and noise columns.
    HVX_Vector *gen_scratch;
    int gen_threads;
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
        free(ctx->gen_scratch);
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

#define B3_CHUNK_START 1
#define B3_CHUNK_END 2
#define B3_ROOT 8
#define B3_KEYED_HASH 16

// One BLAKE3 compression of a 64-byte block per lane: cv[0..7] is the input chaining value
// and gets the output one (the first 8 state words XOR the last 8); counter_lo is each
// lane's block counter (low word; the high word is 0). Rounds fully unrolled with
// constant message indices, so the message stays in registers.
static inline void b3_compress(const HVX_Vector m[16], HVX_Vector cv[8], HVX_Vector counter_lo,
                               uint32 flags) {
    const HVX_Vector r16 = Q6_V_vsplat_R(16), r12 = Q6_V_vsplat_R(12);
    const HVX_Vector r8 = Q6_V_vsplat_R(8), r7 = Q6_V_vsplat_R(7);
    HVX_Vector v[16];
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++)
        v[i] = cv[i];
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++)
        v[8 + i] = Q6_V_vsplat_R(b3_iv[i]);
    v[12] = counter_lo;
    v[13] = Q6_V_vzero();
    v[14] = Q6_V_vsplat_R(64);
    v[15] = Q6_V_vsplat_R(flags);
#define B3_ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) \
    b3_g(v, 0, 4, 8, 12, m[s0], m[s1], r16, r12, r8, r7);                             \
    b3_g(v, 1, 5, 9, 13, m[s2], m[s3], r16, r12, r8, r7);                             \
    b3_g(v, 2, 6, 10, 14, m[s4], m[s5], r16, r12, r8, r7);                            \
    b3_g(v, 3, 7, 11, 15, m[s6], m[s7], r16, r12, r8, r7);                            \
    b3_g(v, 0, 5, 10, 15, m[s8], m[s9], r16, r12, r8, r7);                            \
    b3_g(v, 1, 6, 11, 12, m[s10], m[s11], r16, r12, r8, r7);                          \
    b3_g(v, 2, 7, 8, 13, m[s12], m[s13], r16, r12, r8, r7);                           \
    b3_g(v, 3, 4, 9, 14, m[s14], m[s15], r16, r12, r8, r7);
    B3_ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    B3_ROUND(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
    B3_ROUND(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1)
    B3_ROUND(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6)
    B3_ROUND(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4)
    B3_ROUND(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7)
    B3_ROUND(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13)
#undef B3_ROUND
#pragma clang loop unroll(full)
    for (int w = 0; w < 8; w++)
        cv[w] = VXOR(v[w], v[w + 8]);
}

// Keyed BLAKE3 of one 64-byte message per lane (a single chunk and block: flags
// CHUNK_START | CHUNK_END | ROOT | KEYED_HASH, counter 0): digest words into out[0..7].
static inline void b3_lanes(const HVX_Vector m[16], const uint32 key[8], HVX_Vector out[8]) {
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++)
        out[i] = Q6_V_vsplat_R(key[i]);
    b3_compress(m, out, Q6_V_vzero(), B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED_HASH);
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
    const HVX_Vector r19 = Q6_V_vsplat_R(19);
    // Fold: msg[i] = rotl(x[i], 13) ^ x[i + 16].
    HVX_Vector m[16];
#pragma clang loop unroll(full)
    for (int i = 0; i < 16; i++)
        m[i] = VXOR(VROTR(x[i], r19), x[i + 16]);
    HVX_Vector dgst[8];
    b3_lanes(m, key, dgst);
    // Compare the digest with the bound from the most significant word down.
    HVX_VectorPred lt = Q6_Q_vsetq_R(0), gt = Q6_Q_vsetq_R(0);
#pragma clang loop unroll(full)
    for (int w = 7; w >= 0; w--) {
        const HVX_Vector dg = dgst[w], bd = Q6_V_vsplat_R(bound[w]);
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

    // This column tile's slice of B is contiguous in Bz: copy it to VTCM. Starting an
    // l2fetch of the whole slice first lets the copy's loads run ahead of RAM latency:
    // the copy then costs ~1.1% of the scan at 4096 rows instead of ~1.6%.
    const HVX_Vector *src =
        p->Bz + ((size_t)(p->nt0 + j->nt) * p->Kg + (size_t)kb * p->kblk * 8) * NV;
    const int nv = nch * 8 * NV;
    l2fetch(src, VBYTES, VBYTES, (uint32)nv);
    for (int i = 0; i < nv; i++)
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
    volatile uint32 *ctl;   // scan_run: the host's stop flag, checked per column tile
    int aborted;            // set when ctl's stop ended the run early
} gemmx_thread;

static void gemmx_thread_run(void *arg) {
    gemmx_thread *th = arg;
    gemmx_job *j = &th->job;
    const gx_problem *p = j->p;
    for (int nt = th->nfull > 0 ? th->t : 0; nt < p->Nt; ) {
        if (th->stop && *th->stop)
            break;
        if (th->ctl && ctl_stop_requested(th->ctl)) {
            th->aborted = 1;
            break;
        }
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

// A panel buffer for `rows` rows (sized, padding zeroed); the panel is valid once the
// caller has filled it and set ctx->a_rows = rows.
static int prepare_panel(pearlx_ctx *ctx, int rows, int nthreads, gx_problem *p) {
    ctx->a_rows = 0;
    ctx->a_threads = hvx_threads(nthreads);
    ctx->a_rows = rows;                     // for panel_problem
    panel_problem(ctx, p);
    ctx->a_rows = 0;
    const size_t az_bytes = gx_az_bytes(p);
    if (az_bytes > ctx->az_cap) {
        free(ctx->Az);
        ctx->az_cap = 0;
        ctx->Az = memalign(VBYTES, az_bytes);
        if (!ctx->Az)
            return AEE_ENOMEMORY;
        ctx->az_cap = az_bytes;
    }
    p->Az = ctx->Az;
    // Each region's padding tiles: the kernel loads one chunk past the last
    // one it uses and runs past its last tile (see PAD_TILES).
    for (int kb = 0; kb < p->nkb; kb++)
        memset(gx_a(p, kb, p->Mt), 0, (size_t)PAD_TILES * gx_rstride(p, kb));
    return AEE_SUCCESS;
}

int pearlx_set_a(remote_handle64 h, const int8 *A, int ALen, int rows, int nthreads,
                 uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (!ctx->Bz)
        return AEE_EBADSTATE;
    if (rows <= 0 || rows % MR || (int64)rows * ctx->k > ALen)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    gx_problem p;
    int err = prepare_panel(ctx, rows, nthreads, &p);
    if (err)
        return err;

    packx_a_job pa[MAX_NUM_WORKERS];
    const int na = ctx->a_threads < p.Mt ? ctx->a_threads : p.Mt;
    for (int t = 0; t < na; t++)
        pa[t] = (packx_a_job){ A, &p, p.Mt * t / na, p.Mt * (t + 1) / na };
    err = run_hvx_jobs(ctx, na, packx_a_run, pa, sizeof(pa[0]));
    if (err)
        return err;
    ctx->a_rows = rows;
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
// ctl (scan_run, else NULL): stop early when the host asks, setting *aborted.
static int run_panel(pearlx_ctx *ctx, int col0, int ncols, uint32 *xr, const uint32 *key,
                     const uint32 *bound, int stop_on_hit, jp_result *res,
                     volatile uint32 *ctl, int *aborted) {
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
        jobs[t].ctl = ctl;
        if (!xr) {
            jobs[t].job.xr_local = (uint32 *)(ctx->jp_rec + rec_bytes * t);
            jobs[t].key = key;
            jobs[t].bound = bound;
            jobs[t].stop_on_hit = stop_on_hit;
            jobs[t].stop = &stop;
        }
    }
    err = run_hvx_jobs(ctx, nthreads, gemmx_thread_run, jobs, sizeof(jobs[0]));
    if (aborted) {
        *aborted = 0;
        for (int t = 0; t < nthreads; t++)
            *aborted |= jobs[t].aborted;
    }
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
    int err = run_panel(ctx, col0, ncols, xr, NULL, NULL, 0, NULL, NULL, NULL);
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
    int err = run_panel(ctx, col0, ncols, NULL, key_bound, key_bound + 8, stop_on_hit, &res,
                        NULL, NULL);
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

// ---- Noise generation on the DSP, bit for bit as cp_noise.c ----
// A noise row (a row of A, or a column of B) is a 128-entry table e[j] = (byte_j & 63) - 16
// from 4 keyed BLAKE3 digests (key: the noise seed; message word 0 = 1 + 4 * row + q,
// words 8..15 = the label "A_tensor" / "B_tensor"), then noise[l] = e[first[l]] -
// e[second[l]] over the k permutation pairs. 128 rows at a time: the tables are
// transposed to index-major vectors (byte r of el_t[j] = e_{row0 + r}[j]), so the noise
// of column l for all 128 rows is one byte subtract.

#define GEN_RANK 128                // table entries per noise row
#define GEN_ROWS 128                // noise rows per block
#define GEN_SCRATCH (2 * 128)       // vectors per thread: tables + noise columns

static const uint32 lane_iota[32] __attribute__((aligned(VBYTES))) = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

// In-place transpose of 128 vectors of 128 bytes: afterwards byte i of r[j] is what was
// byte j of r[i]. Seven rounds of vshuff on vector pairs 64, 32, .., 1 apart, done in two
// register-resident passes (each vector loaded and stored twice, not seven times): rounds
// 64, 32, 16 pair only vectors that differ in index bits 4..6, so they run on the 16
// groups of 8 vectors 16 apart; rounds 8, 4, 2, 1 on the 8 groups of 16 neighbours.
static void transpose_bytes_128(HVX_Vector *r) {
    for (int base = 0; base < 16; base++) {
        HVX_Vector v[8];
#pragma clang loop unroll(full)
        for (int j = 0; j < 8; j++)
            v[j] = r[base + 16 * j];
#pragma clang loop unroll(full)
        for (int d = 4; d >= 1; d /= 2)
#pragma clang loop unroll(full)
            for (int j = 0; j < 8; j++)
                if (!(j & d)) {
                    HVX_VectorPair w = Q6_W_vshuff_VVR(v[j + d], v[j], -1);
                    v[j] = Q6_V_lo_W(w);
                    v[j + d] = Q6_V_hi_W(w);
                }
#pragma clang loop unroll(full)
        for (int j = 0; j < 8; j++)
            r[base + 16 * j] = v[j];
    }
    for (int g = 0; g < 8; g++) {
        HVX_Vector v[16];
#pragma clang loop unroll(full)
        for (int j = 0; j < 16; j++)
            v[j] = r[16 * g + j];
#pragma clang loop unroll(full)
        for (int d = 8; d >= 1; d /= 2)
#pragma clang loop unroll(full)
            for (int j = 0; j < 16; j++)
                if (!(j & d)) {
                    HVX_VectorPair w = Q6_W_vshuff_VVR(v[j + d], v[j], -1);
                    v[j] = Q6_V_lo_W(w);
                    v[j + d] = Q6_V_hi_W(w);
                }
#pragma clang loop unroll(full)
        for (int j = 0; j < 16; j++)
            r[16 * g + j] = v[j];
    }
}

// Tables of noise rows [row0, row0 + 128), index-major into el_t[0..127].
// seed_label: the noise seed (8 words), then the label (8 words).
static void gen_tables(int row0, const uint32 *seed_label, HVX_Vector *el_t) {
    HVX_Vector m[16];
    for (int i = 1; i < 8; i++)
        m[i] = Q6_V_vzero();
    for (int i = 0; i < 8; i++)
        m[8 + i] = Q6_V_vsplat_R(seed_label[8 + i]);
    const HVX_Vector iota = *(const HVX_Vector *)lane_iota;
    const HVX_Vector mask = Q6_V_vsplat_R(0x3F3F3F3F), zero_pt = Q6_V_vsplat_R(0x10101010);
    for (int g = 0; g < GEN_ROWS / 8; g++) {
        // One digest per lane: lane i = block 4 (row0 + 8g) + i, so 8 rows x 4 digests.
        m[0] = VADD(Q6_V_vsplat_R(1 + 4 * (row0 + 8 * g)), iota);
        HVX_Vector w[8];
        b3_lanes(m, seed_label, w);
        // 8 x 32 word transpose: vector v = row row0 + 8g + v, word 8q + w = word w of its
        // digest q, i.e. table bytes in order.
#pragma clang loop unroll(full)
        for (int d = 4; d >= 1; d /= 2)
#pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++)
                if (!(i & d)) {
                    HVX_VectorPair x = Q6_W_vshuff_VVR(w[i + d], w[i], -4);
                    w[i] = Q6_V_lo_W(x);
                    w[i + d] = Q6_V_hi_W(x);
                }
        for (int v = 0; v < 8; v++)
            el_t[8 * g + v] = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(w[v], mask), zero_pt);
    }
    transpose_bytes_128(el_t);
}

static int ensure_gen_scratch(pearlx_ctx *ctx, int nthreads) {
    if (ctx->gen_scratch && ctx->gen_threads >= nthreads)
        return AEE_SUCCESS;
    free(ctx->gen_scratch);
    ctx->gen_threads = 0;
    ctx->gen_scratch = memalign(VBYTES, (size_t)nthreads * GEN_SCRATCH * VBYTES);
    if (!ctx->gen_scratch)
        return AEE_ENOMEMORY;
    ctx->gen_threads = nthreads;
    return AEE_SUCCESS;
}

typedef struct {
    const gx_problem *p;
    HVX_Vector *Bz;
    const uint32 *seed_label, *pairs;
    int nt0, nt1;                   // column tiles
    HVX_Vector *scratch;
} gen_b_job;

// Noisy B (signal B = 0) straight into the packed layout: vector (g, q) of column tile nt
// holds the 4 values of k from 4g of columns 128 nt + 32 q + lane, one word per column.
static void gen_b_run(void *arg) {
    const gen_b_job *j = arg;
    const gx_problem *p = j->p;
    HVX_Vector *el_t = j->scratch;
    for (int nt = j->nt0; nt < j->nt1; nt++) {
        gen_tables(nt * NT_COLS, j->seed_label, el_t);
        HVX_Vector *out = j->Bz + (size_t)nt * p->Kg * NV;
        for (int g = 0; g < p->Kg; g++) {
            HVX_Vector nz[4];
            for (int q = 0; q < 4; q++) {
                const uint32 *pr = j->pairs + 2 * (4 * g + q);
                nz[q] = Q6_Vb_vsub_VbVb(el_t[pr[0] & (GEN_RANK - 1)], el_t[pr[1] & (GEN_RANK - 1)]);
            }
            // Bytes of k = 4g .. 4g + 3 into one word per column.
            HVX_VectorPair a = Q6_W_vshuff_VVR(nz[1], nz[0], -1);
            HVX_VectorPair b = Q6_W_vshuff_VVR(nz[3], nz[2], -1);
            HVX_VectorPair c0 = Q6_W_vshuff_VVR(Q6_V_lo_W(b), Q6_V_lo_W(a), -2);
            HVX_VectorPair c1 = Q6_W_vshuff_VVR(Q6_V_hi_W(b), Q6_V_hi_W(a), -2);
            out[(size_t)g * NV + 0] = Q6_V_lo_W(c0);
            out[(size_t)g * NV + 1] = Q6_V_hi_W(c0);
            out[(size_t)g * NV + 2] = Q6_V_lo_W(c1);
            out[(size_t)g * NV + 3] = Q6_V_hi_W(c1);
        }
    }
}

typedef struct {
    const gx_problem *p;
    const int8 *sig;                // the panel's signal rows, row-major
    const uint32 *seed_label, *pairs;
    int row0;                       // global row of the panel's first row
    int b0, b1;                     // 128-row blocks of the panel
    HVX_Vector *scratch;
} gen_a_job;

// Noisy A = signal + noise for 128 rows at a time, straight into the packed panel: per
// 128 values of k, the noise columns are transposed back to rows, the signal added, and
// each 4 rows packed into chunks as packx_a_run does.
static void gen_a_run(void *arg) {
    const gen_a_job *j = arg;
    const gx_problem *p = j->p;
    const int K = p->K;
    HVX_Vector *el_t = j->scratch, *nz = j->scratch + GEN_RANK;
    for (int b = j->b0; b < j->b1; b++) {
        const int r0 = b * GEN_ROWS;
        gen_tables(j->row0 + r0, j->seed_label, el_t);
        for (int c4 = 0; c4 < K / 128; c4++) {
            const uint32 *pairs = j->pairs + 2 * 128 * c4;
            // The two passes of transpose_bytes_128, fused with what comes before and after:
            // pass 1 gathers the noise columns l = 128 c4 + i straight into registers ...
            for (int base = 0; base < 16; base++) {
                HVX_Vector v[8];
#pragma clang loop unroll(full)
                for (int q = 0; q < 8; q++) {
                    const uint32 *pr = pairs + 2 * (base + 16 * q);
                    v[q] = Q6_Vb_vsub_VbVb(el_t[pr[0] & (GEN_RANK - 1)],
                                           el_t[pr[1] & (GEN_RANK - 1)]);
                }
#pragma clang loop unroll(full)
                for (int d = 4; d >= 1; d /= 2)
#pragma clang loop unroll(full)
                    for (int q = 0; q < 8; q++)
                        if (!(q & d)) {
                            HVX_VectorPair w = Q6_W_vshuff_VVR(v[q + d], v[q], -1);
                            v[q] = Q6_V_lo_W(w);
                            v[q + d] = Q6_V_hi_W(w);
                        }
#pragma clang loop unroll(full)
                for (int q = 0; q < 8; q++)
                    nz[base + 16 * q] = v[q];
            }
            // ... and pass 2 leaves finished rows 16 g .. 16 g + 15 in registers: add their
            // signal and pack them, 4 rows to a chunk group as packx_a_run does.
            const int kb = 4 * c4 / p->kblk;
            for (int g = 0; g < GEN_ROWS / 16; g++) {
                HVX_Vector v[16];
#pragma clang loop unroll(full)
                for (int q = 0; q < 16; q++)
                    v[q] = nz[16 * g + q];
#pragma clang loop unroll(full)
                for (int d = 8; d >= 1; d /= 2)
#pragma clang loop unroll(full)
                    for (int q = 0; q < 16; q++)
                        if (!(q & d)) {
                            HVX_VectorPair w = Q6_W_vshuff_VVR(v[q + d], v[q], -1);
                            v[q] = Q6_V_lo_W(w);
                            v[q + d] = Q6_V_hi_W(w);
                        }
#pragma clang loop unroll(full)
                for (int tt = 0; tt < 4; tt++) {
                    HVX_Vector r[MR];
#pragma clang loop unroll(full)
                    for (int q = 0; q < MR; q++) {
                        const int row = r0 + 16 * g + MR * tt + q;
                        const int8 *sig = j->sig + (size_t)row * K + 128 * c4;
                        r[q] = Q6_Vb_vadd_VbVb(v[MR * tt + q], *(const HVX_UVector *)sig);
                    }
                    HVX_VectorPair p01 = Q6_W_vshuff_VVR(r[1], r[0], -32);
                    HVX_VectorPair p23 = Q6_W_vshuff_VVR(r[3], r[2], -32);
                    HVX_VectorPair q0 = Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
                    HVX_VectorPair q1 = Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
                    HVX_Vector *out =
                        (HVX_Vector *)(gx_a(p, kb, (r0 + 16 * g) / MR + tt) +
                                       (size_t)(4 * c4 - kb * p->kblk) * VBYTES);
                    out[0] = Q6_V_lo_W(q0);
                    out[1] = Q6_V_hi_W(q0);
                    out[2] = Q6_V_lo_W(q1);
                    out[3] = Q6_V_hi_W(q1);
                }
            }
        }
    }
}

int pearlx_set_b_gen(remote_handle64 h, int n, int k, const uint32 *seed_label, int slLen,
                     const uint32 *pairs, int pairsLen, uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (n <= 0 || k <= 0 || n % NT_COLS || k % 128 || slLen < 16 || pairsLen < 2 * k)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    free(ctx->Bz);
    ctx->Bz = NULL;
    ctx->n = ctx->k = 0;
    ctx->a_rows = 0;
    gx_problem p;
    memset(&p, 0, sizeof(p));
    gx_shape(&p, n, k);
    HVX_Vector *Bz = memalign(VBYTES, (size_t)n * k);
    if (!Bz) {
        FARF(ERROR, "pearlx: no memory for packed B (%d x %d)", n, k);
        return AEE_ENOMEMORY;
    }
    const int nthreads = hvx_threads(0);
    int err = ensure_gen_scratch(ctx, nthreads);
    if (err) {
        free(Bz);
        return err;
    }
    const int nb = nthreads < p.Nt ? nthreads : p.Nt;
    gen_b_job jb[MAX_NUM_WORKERS];
    for (int t = 0; t < nb; t++)
        jb[t] = (gen_b_job){ &p, Bz, seed_label, pairs, p.Nt * t / nb, p.Nt * (t + 1) / nb,
                             ctx->gen_scratch + (size_t)t * GEN_SCRATCH };
    err = run_hvx_jobs(ctx, nb, gen_b_run, jb, sizeof(jb[0]));
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

// The packed row panel of noisy A from its signal rows A (rows of them, noise rows from
// row0); rows a multiple of GEN_ROWS.
static int gen_a_panel(pearlx_ctx *ctx, const int8 *A, int row0, int rows, int nthreads,
                       const uint32 *seed_label, const uint32 *pairs) {
    gx_problem p;
    int err = prepare_panel(ctx, rows, nthreads, &p);
    if (!err)
        err = ensure_gen_scratch(ctx, ctx->a_threads);
    if (err)
        return err;
    const int blocks = rows / GEN_ROWS;
    const int na = ctx->a_threads < blocks ? ctx->a_threads : blocks;
    gen_a_job ja[MAX_NUM_WORKERS];
    for (int t = 0; t < na; t++)
        ja[t] = (gen_a_job){ &p, A, seed_label, pairs, row0, blocks * t / na,
                             blocks * (t + 1) / na, ctx->gen_scratch + (size_t)t * GEN_SCRATCH };
    err = run_hvx_jobs(ctx, na, gen_a_run, ja, sizeof(ja[0]));
    if (err)
        return err;
    ctx->a_rows = rows;
    return AEE_SUCCESS;
}

int pearlx_set_a_gen(remote_handle64 h, const int8 *A, int ALen, int row0, int rows,
                     int nthreads, const uint32 *seed_label, int slLen, const uint32 *pairs,
                     int pairsLen, uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    if (!ctx->Bz)
        return AEE_EBADSTATE;
    if (row0 < 0 || rows <= 0 || rows % GEN_ROWS || (int64)rows * ctx->k > ALen ||
        slLen < 16 || pairsLen < 2 * ctx->k)
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    int err = gen_a_panel(ctx, A, row0, rows, nthreads, seed_label, pairs);
    *dsp_us = HAP_perf_get_time_us() - t0;
    return err;
}

// An attempt's launches without a call per launch (see pearlx.idl): the host follows
// through ctl, checking each launch's records while the DSP runs the next ones.
int pearlx_scan_run(remote_handle64 h, const int8 *A, int ALen, int row0, int rows,
                    int panel_rows, int launch_cols, int s0, int nbuf, int nthreads,
                    const uint32 *seed_label, int slLen, const uint32 *pairs, int pairsLen,
                    uint32 *ctl_words, int ctlLen, uint32 *xr, int xrLen, uint64 *dsp_us,
                    uint64 *wait_us, int *launched) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    *dsp_us = *wait_us = 0;
    *launched = 0;
    if (!ctx->Bz)
        return AEE_EBADSTATE;
    const int n = ctx->n, k = ctx->k;
    if (row0 < 0 || rows <= 0 || rows % GEN_ROWS || panel_rows <= 0 || panel_rows % GEN_ROWS ||
        launch_cols <= 0 || launch_cols % NT_COLS || s0 < 0 || nbuf < 1 ||
        (int64)rows * k > ALen || slLen < 16 || pairsLen < 2 * k || ctlLen < CTL_WORDS ||
        ((uintptr_t)ctl_words & (VBYTES - 1)) || ((uintptr_t)xr & (VBYTES - 1)))
        return AEE_EBADPARM;
    gx_problem shape;
    memset(&shape, 0, sizeof(shape));
    gx_shape(&shape, n, k);
    // Equal slots, each 128-byte aligned, big enough for the largest launch.
    const size_t slot_words = (size_t)xrLen / nbuf / 32 * 32;
    const int max_rows = panel_rows < rows ? panel_rows : rows;
    const int max_cols = launch_cols < n ? launch_cols : n;
    if ((size_t)(max_cols / NT_COLS) * (max_rows / MR) * shape.lines * 32 > slot_words)
        return AEE_EBADPARM;

    volatile uint32 *ctl = ctl_words;
    ctl_pull(ctl);
    ctl[CTL_LAUNCHED] = (uint32)s0;
    ctl[CTL_EPOCH_ACK] = ctl[CTL_EPOCH];
    ctl[CTL_STATE] = CTL_RUNNING;
    ctl_push(ctl);

    int err = AEE_SUCCESS, stopped = 0, s = s0;
    uint64 busy = 0, waited = 0;
    for (int r = 0; r < rows && !err && !stopped; r += panel_rows) {
        const int prows = rows - r < panel_rows ? rows - r : panel_rows;
        if (ctl_stop_requested(ctl)) {
            stopped = 1;
            break;
        }
        uint64 t0 = HAP_perf_get_time_us();
        err = gen_a_panel(ctx, A + (size_t)r * k, row0 + r, prows, nthreads, seed_label, pairs);
        busy += HAP_perf_get_time_us() - t0;
        for (int col0 = 0; col0 < n && !err && !stopped; col0 += launch_cols, s++) {
            const int cols = n - col0 < launch_cols ? n - col0 : launch_cols;
            // Slot s % nbuf is free once the host has checked launch s - nbuf.
            t0 = HAP_perf_get_time_us();
            for (;;) {
                ctl_pull(ctl);
                if (ctl[CTL_STOP]) {
                    stopped = 1;
                    break;
                }
                if ((int)ctl[CTL_CHECKED] >= s - nbuf + 1)
                    break;
                qurt_timer_sleep(CTL_POLL_US);   // qurt_sleep is not exported to user PDs
            }
            waited += HAP_perf_get_time_us() - t0;
            if (stopped)
                break;
            uint32 *slot = xr + (size_t)(s % nbuf) * slot_words;
            int aborted = 0;
            t0 = HAP_perf_get_time_us();
            err = run_panel(ctx, col0, cols, slot, NULL, NULL, 0, NULL, ctl, &aborted);
            // The records out to memory before the host is told they are there. Cleaning the
            // whole data cache costs ~0.07 ms per launch; cleaning the 8 MiB slot by address
            // costs ~0.44 ms, walking every line though only the cache's worth is dirty.
            if (!err && !aborted)
                qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);
            busy += HAP_perf_get_time_us() - t0;
            if (err || aborted) {
                stopped = aborted;
                break;
            }
            ctl[CTL_LAUNCHED] = (uint32)(s + 1);
            ctl_push(ctl);
            (*launched)++;
            // Self-test only: idle so the launch's lines would still be in the cache while the
            // host reads them, had they not been cleaned (the miner leaves the word at 0).
            ctl_pull(ctl);
            if (ctl[CTL_TEST_IDLE_US])
                qurt_timer_sleep(ctl[CTL_TEST_IDLE_US]);
        }
    }
    ctl[CTL_STATE] = err ? CTL_FAILED : stopped ? CTL_STOPPED : CTL_DONE;
    ctl_push(ctl);
    *dsp_us = busy;
    *wait_us = waited;
    return err;
}

// ---- Keyed BLAKE3 chunk chaining values (the leaves of cp_noise.c's matrix hashes) ----
// 32 chunks of 1 KB per vector, one per lane: 16 compressions per chunk with the chunk
// index as counter, flags KEYED_HASH plus CHUNK_START on the first block and CHUNK_END on
// the last. The host builds the tree above the leaves.

#define CV_GROUP 32                 // chunks per vector

typedef struct {
    const uint8 *data;              // NULL: all-zero data
    const uint32 *key;
    uint32 *cvs;                    // 8 words per chunk
    int g0, g1;                     // groups of CV_GROUP chunks
} cv_job;

static void cv_run(void *arg) {
    const cv_job *j = arg;
    const HVX_Vector iota = *(const HVX_Vector *)lane_iota;
    for (int g = j->g0; g < j->g1; g++) {
        const int chunk0 = g * CV_GROUP;
        HVX_Vector cv[8];
        for (int w = 0; w < 8; w++)
            cv[w] = Q6_V_vsplat_R(j->key[w]);
        const HVX_Vector counter = VADD(Q6_V_vsplat_R(chunk0), iota);
        for (int pair = 0; pair < 8; pair++) {
            const uint32 f0 = B3_KEYED_HASH | (pair == 0 ? B3_CHUNK_START : 0);
            const uint32 f1 = B3_KEYED_HASH | (pair == 7 ? B3_CHUNK_END : 0);
            if (!j->data) {
                HVX_Vector m[16];
                for (int w = 0; w < 16; w++)
                    m[w] = Q6_V_vzero();
                b3_compress(m, cv, counter, f0);
                b3_compress(m, cv, counter, f1);
                continue;
            }
            // Blocks 2 pair and 2 pair + 1 of the 32 chunks, one 128-byte vector per
            // chunk, transposed so that r[w] holds word w of each chunk (16..31: the
            // second block).
            HVX_Vector r[32];
            for (int c = 0; c < 32; c++)
                r[c] = *(const HVX_Vector *)(j->data + (size_t)(chunk0 + c) * 1024 + pair * 128);
            for (int d = 16; d >= 1; d /= 2)
                for (int i = 0; i < 32; i++)
                    if (!(i & d)) {
                        HVX_VectorPair w = Q6_W_vshuff_VVR(r[i + d], r[i], -4);
                        r[i] = Q6_V_lo_W(w);
                        r[i + d] = Q6_V_hi_W(w);
                    }
            b3_compress(r, cv, counter, f0);
            b3_compress(r + 16, cv, counter, f1);
        }
        // 8 x 32 word transpose: vector v = chunks 4v .. 4v + 3, 8 words each, in order.
#pragma clang loop unroll(full)
        for (int d = 4; d >= 1; d /= 2)
#pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++)
                if (!(i & d)) {
                    HVX_VectorPair w = Q6_W_vshuff_VVR(cv[i + d], cv[i], -4);
                    cv[i] = Q6_V_lo_W(w);
                    cv[i + d] = Q6_V_hi_W(w);
                }
        HVX_Vector *out = (HVX_Vector *)(j->cvs + (size_t)chunk0 * 8);
        for (int v = 0; v < 8; v++)
            out[v] = cv[v];
    }
}

int pearlx_chunk_cvs(remote_handle64 h, const int8 *data, int dataLen, int raw_len,
                     const uint32 *key, int keyLen, uint32 *cvs, int cvsLen, uint64 *dsp_us) {
    pearlx_ctx *ctx = (pearlx_ctx *)h;
    const int chunks = raw_len / 1024;
    if (raw_len <= 0 || raw_len % (1024 * CV_GROUP) || keyLen < 8 || cvsLen < chunks * 8 ||
        (dataLen != 0 && dataLen < raw_len) || ((uintptr_t)data & (VBYTES - 1)) ||
        ((uintptr_t)cvs & (VBYTES - 1)))
        return AEE_EBADPARM;
    uint64 t0 = HAP_perf_get_time_us();
    const int groups = chunks / CV_GROUP;
    const int nthreads = hvx_threads(0);
    const int nc = nthreads < groups ? nthreads : groups;
    cv_job jc[MAX_NUM_WORKERS];
    for (int t = 0; t < nc; t++)
        jc[t] = (cv_job){ dataLen ? (const uint8 *)data : NULL, key, cvs, groups * t / nc,
                          groups * (t + 1) / nc };
    int err = run_hvx_jobs(ctx, nc, cv_run, jc, sizeof(jc[0]));
    *dsp_us = HAP_perf_get_time_us() - t0;
    return err;
}
