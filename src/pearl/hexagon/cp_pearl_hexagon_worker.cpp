#include "cp_pearl_hexagon_worker.h"

#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_pearlx_client.h"
#include "cp_state.h"
#include "cp_util.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Hexagon cDSP worker (zero-B, like the CPU worker). The DSP generates and packs all the
 * noise; the host keeps signal A (shared with the DSP), its hash, the seeds and the jackpot.
 *   job:     noisy B (signal B = 0) generated packed in DSP memory (pearlx set_b_gen).
 *   attempt: sparse random signal A on the host -> a_noise_seed and permutation pairs;
 *            then the matrix is scanned in launches of row_macros x col_macros 128x128
 *            macro blocks. The DSP runs the launches itself (scan_run): it generates and
 *            packs each row panel of noisy A from the panel's signal rows, then runs it
 *            against one block of columns per launch. The mining thread follows through a
 *            shared control block, running the BLAKE3 jackpot over each finished launch,
 *            and stops the DSP on a hit or a new job.
 * Hash tile: 4 rows x 64 columns (half a 4x128 HVX register tile), proof layout 4x64.
 */

namespace {

constexpr int kTileRows = 4;
constexpr int kTileCols = 64;
constexpr int kRegCols = 128;
constexpr int kSplit = kRegCols / kTileCols;          // hash tiles per register tile
constexpr int kMacro = 128;                           // macro block: 128 x 128
constexpr int kMilestones = K_DIM / R_RANK;
static_assert(R_RANK == 128, "pearlx takes one milestone per 128 values of k");
constexpr int kRecWords = (kMilestones * kSplit + 31) / 32 * 32;

CpPearlx* g_px = nullptr;
int g_row_macros = CP_HEXAGON_LAUNCH_MACROS_DEFAULT;
int g_col_macros = CP_HEXAGON_LAUNCH_MACROS_DEFAULT;

/* One long-lived thread makes the DSP calls (scan_run blocks for a whole segment). FastRPC
 * pairs every calling thread with a thread on the DSP, so a new caller per call pays that
 * setup each time (~8 ms measured), and the mining thread stays free for the jackpot. */
class DspThread {
public:
    DspThread() : th_([this] { loop(); }) {}
    ~DspThread()
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        th_.join();
    }
    void start(std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            job_ = std::move(job);
            busy_ = true;
        }
        cv_.notify_all();
    }
    void wait()
    {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return !busy_; });
    }

private:
    void loop()
    {
        /* The host's OpenMP threads (jackpot, noisy A) fill every core while the DSP runs;
         * a higher priority lets this thread return each launch's result and start the
         * next one at once instead of waiting for a time slice. */
        errno = 0;
        prio_ok_ = setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), -10) == 0;
        prio_errno_ = errno;
        std::unique_lock<std::mutex> lock(mu_);
        for(;;){
            cv_.wait(lock, [this] { return quit_ || (busy_ && job_); });
            if(quit_) return;
            std::function<void()> job = std::move(job_);
            job_ = nullptr;
            lock.unlock();
            job();
            lock.lock();
            busy_ = false;
            cv_.notify_all();
        }
    }
public:
    bool prio_ok_ = false;
    int prio_errno_ = 0;

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool busy_ = false;
    bool quit_ = false;
    std::thread th_;
};

DspThread* g_dsp = nullptr;

/* Cached hash tree of signal A (incremental hash_a), and CP_HEXAGON_HASH_CHECK=1 to compare
 * it with a full hash every attempt. */
PearlMatrixHash* g_ahash = nullptr;
const bool g_hash_check = [] {
    const char* v = getenv("CP_HEXAGON_HASH_CHECK");
    return v && *v && strcmp(v, "0") != 0;
}();

struct JobCache {
    uint8_t job_key[32]{};
    int m = 0;
    int n = 0;
    int salted = 0;
    int ready = 0;
    uint8_t b_noise_seed[32]{};
    std::vector<uint32_t> pairs_b;   /* B-side permutation, for re-deriving hit columns */
} g_job;

/* Launch s writes its records to slot s % kRecSlots; it starts once the host has checked
 * launch s - kRecSlots, so the host has one launch of slack. */
constexpr int kRecSlots = 2;
/* Signal A per scan_run call: the call maps its rows of A into the DSP. */
constexpr size_t kSegmentBytes = (size_t)128 << 20;

/* Launch records and scan_run's control block (rpcmem, uncached, shared with the DSP). */
struct Buffers {
    int rows = 0;
    int cols = 0;
    size_t xr_words = 0;         /* per slot */
    uint32_t* xr = nullptr;      /* kRecSlots slots */
    uint32_t* ctl = nullptr;     /* CP_PEARLX_CTL_WORDS */
    uint32_t epoch = 0;
} g_buf;

void free_buffers()
{
    cp_pearlx_free(g_px, g_buf.xr);
    cp_pearlx_free(g_px, g_buf.ctl);
    g_buf.xr = nullptr;
    g_buf.ctl = nullptr;
    g_buf.rows = g_buf.cols = 0;
    g_buf.xr_words = 0;
}

/* The control block's words (cp_pearlx_client.h): the host stores only its own half. */
uint32_t ctl_load(int w)
{
    return __atomic_load_n(&g_buf.ctl[w], __ATOMIC_ACQUIRE);
}

void ctl_store(int w, uint32_t v)
{
    __atomic_store_n(&g_buf.ctl[w], v, __ATOMIC_RELEASE);
}

/* Launch size for an m x n matrix (m, n multiples of 1024): rows x cols, at most. */
void launch_shape(int m, int n, int* rows, int* cols)
{
    *rows = g_row_macros * kMacro < m ? g_row_macros * kMacro : m;
    *cols = g_col_macros * kMacro < n ? g_col_macros * kMacro : n;
}

int ensure_buffers(int rows, int cols)
{
    if(g_buf.rows == rows && g_buf.cols == cols)
        return 0;
    free_buffers();
    const size_t xr_words = (size_t)(cols / kRegCols) * (rows / kTileRows) * kRecWords;
    /* Uncached: the host reads the records while the DSP is still running, and the DSP
     * writes them out to memory itself. The jackpot reads them slower, but it runs while
     * the DSP is busy. */
    g_buf.xr = (uint32_t*)cp_pearlx_alloc_uncached(g_px, kRecSlots * xr_words * sizeof(uint32_t));
    g_buf.ctl = (uint32_t*)cp_pearlx_alloc_uncached(g_px, CP_PEARLX_CTL_WORDS * sizeof(uint32_t));
    if(!g_buf.xr || !g_buf.ctl){
        fprintf(stderr, "[hexagon] rpcmem allocation failed (launch %d x %d)\n", rows, cols);
        free_buffers();
        return -1;
    }
    /* Word stores (memset may use cache-zeroing instructions, unsafe on uncached memory).
     * The DSP has not used the block yet, so the host may write both halves this once. */
    for(int w = 0; w < CP_PEARLX_CTL_WORDS; w++)
        ctl_store(w, 0);
    g_buf.rows = rows;
    g_buf.cols = cols;
    g_buf.xr_words = xr_words;
    return 0;
}

int job_matches(const uint8_t job_key[32], int m, int n)
{
    return g_job.ready && g_job.m == m && g_job.n == n && memcmp(g_job.job_key, job_key, 32) == 0;
}

/* Noise seed + label as the 16 words pearlx set_*_gen take. */
void seed_label_words(const uint8_t seed[32], const uint8_t label[32], uint32_t out[16])
{
    memcpy(out, seed, 32);
    memcpy(out + 8, label, 32);
}

/* Keyed matrix hash with the chunk hashes on the DSP and the tree above them on the host:
 * the root of data (NULL: all zeros) into root, and the tree into cache when given. */
int dsp_matrix_hash(const int8_t* data, size_t bytes, const uint8_t key[32], uint8_t* cvs,
                    PearlMatrixHash* cache, uint8_t root[32])
{
    uint32_t key8[8];
    memcpy(key8, key, 32);
    uint64_t us = 0;
    if(cp_pearlx_chunk_cvs(g_px, data, bytes, key8, (uint32_t*)cvs, &us) != 0)
        return -1;
    PearlMatrixHash* h = cache ? cache : pearl_matrix_hash_create();
    const int rc = h ? pearl_matrix_hash_from_cvs(h, data, bytes, key, cvs, root) : -1;
    if(!cache) pearl_matrix_hash_free(h);
    return rc;
}

/* The job's B (signal B = 0): its noise seed from the hash of the zero B^T, then noisy B
 * generated and packed in DSP memory. Then signal A's hash tree under the new job key, so
 * that the job's first attempt rehashes only what it changes. The chunk hashes of both
 * matrices run on the DSP. */
int prepare_job(const uint8_t job_key[32], int m, int n)
{
    if(!g_px) return -2;
    if(n % kMacro != 0 || m % kMacro != 0){
        fprintf(stderr, "[hexagon] m and n must be multiples of %d (got %dx%d)\n", kMacro, m, n);
        return -2;
    }
    const double t0 = cp_now_sec();
    g_job.ready = 0;
    memcpy(g_job.job_key, job_key, 32);
    g_job.m = m;
    g_job.n = n;
    const size_t b_bytes = (size_t)n * K_DIM, a_bytes = (size_t)m * K_DIM;
    uint8_t* cvs = (uint8_t*)cp_pearlx_alloc(g_px, (b_bytes > a_bytes ? b_bytes : a_bytes) / 1024 * 32);

    uint8_t root_b[32];
    if(cvs && dsp_matrix_hash(NULL, b_bytes, job_key, cvs, nullptr, root_b) == 0){
        pearl_b_noise_seed_from_root(job_key, root_b, n, g_job.salted, g_job.b_noise_seed);
    } else {
        fprintf(stderr, "[hexagon] DSP hash of the zero B failed; hashing it on the host\n");
        pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_job.salted, g_job.b_noise_seed);
    }
    if(g_hash_check){
        uint8_t ref[32];
        pearl_b_noise_seed_from_bt(job_key, NULL, n, K_DIM, g_job.salted, ref);
        const bool same = memcmp(ref, g_job.b_noise_seed, 32) == 0;
        printf("[hexagon] hash check: B noise seed from the DSP %s the host's\n",
               same ? "matches" : "DIFFERS FROM");
        if(!same) memcpy(g_job.b_noise_seed, ref, 32);
    }
    const double t_seed = cp_now_sec() - t0;
    g_job.pairs_b.assign((size_t)K_DIM * 2, 0);
    pearl_build_perm_pairs_b(g_job.b_noise_seed, K_DIM, R_RANK, g_job.pairs_b.data());

    uint32_t seed_label[16];
    seed_label_words(g_job.b_noise_seed, PEARL_SEED_LABEL_B, seed_label);
    uint64_t dsp_us = 0;
    const int err = cp_pearlx_set_b_gen(g_px, n, K_DIM, seed_label, g_job.pairs_b.data(), &dsp_us);
    if(err){
        cp_pearlx_free(g_px, cvs);
        fprintf(stderr, "[hexagon] pearlx set_b_gen failed (0x%x): n=%d may not fit in DSP "
                        "memory\n", err, n);
        return -2;
    }

    /* Signal A (unless a proof still holds it): its tree for this key, so attempt 0 is
     * incremental like the rest. */
    double t_a = 0.0;
    if(h_Ap_global && cvs){
        const double ta0 = cp_now_sec();
        if(!g_ahash) g_ahash = pearl_matrix_hash_create();
        uint8_t root_a[32];
        if(g_ahash &&
           dsp_matrix_hash(h_Ap_global, a_bytes, job_key, cvs, g_ahash, root_a) == 0){
            if(g_hash_check){
                uint8_t ref[32];
                pearl_keyed_digest_int8(h_Ap_global, a_bytes, job_key, ref);
                printf("[hexagon] hash check: signal A tree from the DSP %s the host's hash\n",
                       memcmp(ref, root_a, 32) == 0 ? "matches" : "DIFFERS FROM");
                if(memcmp(ref, root_a, 32) != 0) pearl_matrix_hash_invalidate(g_ahash);
            }
        } else if(g_ahash) {
            pearl_matrix_hash_invalidate(g_ahash);
        }
        t_a = cp_now_sec() - ta0;
    }
    cp_pearlx_free(g_px, cvs);

    int rows, cols;
    launch_shape(m, n, &rows, &cols);
    if(ensure_buffers(rows, cols) != 0)
        return -2;
    printf("[hexagon] job setup %.3fs: B seed %.3fs, noisy B generated on the DSP %.3fs, "
           "signal A hash tree %.3fs (salted=%d)\n", cp_now_sec() - t0, t_seed, dsp_us / 1e6,
           t_a, g_job.salted);
    fflush(stdout);
    g_job.ready = 1;
    return 0;
}

/* One DSP launch: rows [row0, row0 + rows) x columns [col0, col0 + cols). */
struct Launch {
    int panel;
    int row0, rows;
    int col0, cols;
    int first_in_panel;
    int index_in_panel, launches_in_panel;
};

struct Hit {
    int found = 0;
    int t_rows = -1;
    int t_cols = -1;
    uint32_t xor_words[kMilestones]{};
};

/* Jackpot over one launch's records: [cols/128 column tiles][rows/4 row tiles][kRecWords],
 * word ms * 2 + h = hash tile h of the register tile after milestone ms. */
Hit check_launch(const Launch& l, const uint32_t* xr, const uint32_t a_key8[8],
                 const uint32_t bound[8])
{
    const int mt = l.rows / kTileRows;
    const int records = (l.cols / kRegCols) * mt;
    std::atomic<int> best{-1};   /* lowest hit (record * 2 + half), for a deterministic pick */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int rec = 0; rec < records; rec++){
        /* The records are uncached: copy each one in with wide loads instead of reading its
         * words one by one (one memory transaction per load either way). */
        uint32_t w[kRecWords];
        memcpy(w, xr + (size_t)rec * kRecWords, sizeof(w));
        for(int h = 0; h < kSplit; h++){
            uint32_t ms_xor[kMilestones];
            for(int ms = 0; ms < kMilestones; ms++)
                ms_xor[ms] = w[ms * kSplit + h];
            if(cp_jackpot::tile_beats_target(ms_xor, kMilestones, a_key8, bound)){
                const int id = rec * kSplit + h;
                int cur = best.load();
                while((cur < 0 || id < cur) && !best.compare_exchange_weak(cur, id)){
                }
            }
        }
    }
    Hit hit;
    const int id = best.load();
    if(id >= 0){
        const int rec = id / kSplit, h = id % kSplit;
        const int nt = rec / mt, t = rec % mt;
        hit.found = 1;
        hit.t_rows = l.row0 + t * kTileRows;
        hit.t_cols = l.col0 + nt * kRegCols + h * kTileCols;
        const uint32_t* w = xr + (size_t)rec * kRecWords;
        for(int ms = 0; ms < kMilestones; ms++)
            hit.xor_words[ms] = w[ms * kSplit + h];
    }
    return hit;
}

/* Recompute a hit tile on the host: its A rows from signal A and B columns from the job
 * seed, with the host's noise code. A DSP fault (kernel or noise generation) can therefore
 * never turn into an invalid share. */
int verify_hit(const Hit& hit, const int8_t* a_sig, const uint8_t a_seed[32],
               const uint32_t* pairs_a, const uint32_t a_key8[8], const uint32_t bound[8])
{
    std::vector<int8_t> bcol((size_t)kTileCols * K_DIM), el((size_t)R_RANK), nr((size_t)K_DIM);
    std::vector<int8_t> arow((size_t)kTileRows * K_DIM);
    for(int i = 0; i < kTileRows; i++)
        pearl_fuse_noise_row_a_buf(hit.t_rows + i, K_DIM, R_RANK, a_seed, pairs_a,
                                   a_sig + (size_t)(hit.t_rows + i) * K_DIM,
                                   arow.data() + (size_t)i * K_DIM, el.data(), nr.data());
    for(int j = 0; j < kTileCols; j++)
        pearl_fuse_noise_row_b_buf(hit.t_cols + j, K_DIM, R_RANK, g_job.b_noise_seed,
                                   g_job.pairs_b.data(), NULL, bcol.data() + (size_t)j * K_DIM,
                                   el.data(), nr.data());
    uint32_t ref[kMilestones] = {};
    for(int i = 0; i < kTileRows; i++){
        const int8_t* a = arow.data() + (size_t)i * K_DIM;
        for(int j = 0; j < kTileCols; j++){
            const int8_t* b = bcol.data() + (size_t)j * K_DIM;
            int32_t acc = 0;
            for(int ms = 0; ms < kMilestones; ms++){
                for(int l = ms * R_RANK; l < (ms + 1) * R_RANK; l++)
                    acc += (int32_t)a[l] * (int32_t)b[l];
                ref[ms] ^= (uint32_t)acc;
            }
        }
    }
    if(memcmp(ref, hit.xor_words, sizeof(ref)) != 0){
        fprintf(stderr, "[hexagon] DSP tile XOR differs from the host recompute at t_rows=%d "
                        "t_cols=%d; share dropped\n", hit.t_rows, hit.t_cols);
        return -1;
    }
    return cp_jackpot::tile_beats_target(ref, kMilestones, a_key8, bound) ? 0 : -1;
}

} /* namespace */

extern "C" int8_t* cp_pearl_hexagon_worker_alloc_signal_a(size_t bytes)
{
    /* Shared with the DSP: set_a_gen reads each panel's signal rows in place. */
    if(!g_px) return nullptr;
    int8_t* p = (int8_t*)cp_pearlx_alloc(g_px, bytes);
    if(p) memset(p, 0, bytes);
    return p;
}

extern "C" void cp_pearl_hexagon_worker_free_signal_a(int8_t* p)
{
    if(p && g_px) cp_pearlx_free(g_px, p);
}

extern "C" void cp_pearl_hexagon_worker_set_launch(int row_macros, int col_macros)
{
    g_row_macros = row_macros > 0 ? row_macros : CP_HEXAGON_LAUNCH_MACROS_DEFAULT;
    g_col_macros = col_macros > 0 ? col_macros : CP_HEXAGON_LAUNCH_MACROS_DEFAULT;
}

extern "C" void cp_pearl_hexagon_worker_launch(int m, int n, int* rows, int* cols)
{
    launch_shape(m, n, rows, cols);
}

extern "C" void cp_pearl_hexagon_worker_init(void)
{
    if(g_px) return;
    g_px = cp_pearlx_open();
    if(!g_px) return;
    g_dsp = new DspThread();
    g_dsp->start([] {});   /* runs the thread's setup (priority) before we report it */
    g_dsp->wait();
    int hvx = 0, vote_err = 0, mhz = 0, vtcm_kb = 0;
    cp_pearlx_info(g_px, &hvx, &vote_err, &mhz, &vtcm_kb);
    printf("[hexagon] cDSP: %d HVX contexts, %d MHz, VTCM %d KB, max-clock vote %s\n", hvx, mhz,
           vtcm_kb, vote_err ? "FAILED" : "ok");
    printf("[hexagon] fused HVX GEMM + XOR (4x64 hash tiles, zero-B), host BLAKE3 jackpot\n");
    if(g_dsp->prio_ok_)
        printf("[hexagon] DSP launch thread: nice -10\n");
    else
        printf("[hexagon] DSP launch thread: default priority (raising it failed: %s)\n",
               strerror(g_dsp->prio_errno_));
    fflush(stdout);
}

extern "C" void cp_pearl_hexagon_worker_shutdown(void)
{
    delete g_dsp;
    g_dsp = nullptr;
    pearl_matrix_hash_free(g_ahash);
    g_ahash = nullptr;
    free_buffers();
    g_job.ready = 0;
    g_job.pairs_b.clear();
    cp_pearlx_close(g_px);
    g_px = nullptr;
}

extern "C" int cp_pearl_hexagon_worker_is_ready(void)
{
    return g_px != nullptr;
}

extern "C" int cp_pearl_hexagon_worker_list_devices(void)
{
    cp_pearl_hexagon_worker_init();
    return g_px ? 1 : 0;
}

extern "C" void cp_pearl_hexagon_worker_begin_job(const uint8_t job_key[32], int m, int n,
                                                  uint32_t cert_version)
{
    g_job.ready = 0;
    g_job.salted = cert_version >= 3 ? 1 : 0;
    if(g_px)
        prepare_job(job_key, m, n);
}

extern "C" int cp_pearl_hexagon_worker_mine_attempt(
        const uint8_t* ab_seed, int ab_seed_len, const uint8_t job_key[32],
        const uint32_t pool_tgt[8], int m, int n, int cpu_matrices,
        const int8_t* h_A_noisy, const int8_t* h_B_noisy, const uint8_t* a_key,
        int8_t* h_A_sig, int8_t* h_Bt_sig, int* out_t_rows, int* out_t_cols,
        uint64_t* out_tiles_scanned)
{
    (void)ab_seed;
    (void)ab_seed_len;
    (void)cpu_matrices;
    (void)h_A_noisy;
    (void)h_B_noisy;
    (void)a_key;
    (void)h_Bt_sig;
    if(out_tiles_scanned) *out_tiles_scanned = 0;
    if(out_t_rows) *out_t_rows = -1;
    if(out_t_cols) *out_t_cols = -1;
    if(!g_px){
        fprintf(stderr, "[hexagon] cDSP not available\n");
        return -2;
    }
    if(!h_A_sig){
        fprintf(stderr, "[hexagon] mine_attempt requires h_Ap_global\n");
        return -2;
    }
    const double attempt_t0 = cp_now_sec();
    if(!job_matches(job_key, m, n)){
        const int rc = prepare_job(job_key, m, n);
        if(rc != 0) return rc;
    }

    /* Signal A: ~K_DIM random writes per attempt (CSPRNG), then the attempt's noise seed. */
    uint8_t a_rng[32];
    if(cp_random_bytes(a_rng, sizeof(a_rng)) != 0){
        fprintf(stderr, "[hexagon] CSPRNG failed for random A\n");
        return -2;
    }
    std::vector<uint64_t> changed((size_t)K_DIM);
    if(pearl_perturb_random_a_one_per_col_ex(a_rng, (int)sizeof(a_rng), m, K_DIM, h_A_sig,
                                             changed.data()) != 0)
        return -1;
    /* hash_a over all of signal A, but only the chunks this attempt wrote (and their tree
     * ancestors) are hashed again; the first attempt of a job hashes everything. */
    const double t_hash = cp_now_sec();
    uint8_t hash_a[32];
    size_t rehashed = 0;
    if(!g_ahash) g_ahash = pearl_matrix_hash_create();
    if(pearl_matrix_hash_digest(g_ahash, h_A_sig, (size_t)m * K_DIM, job_key, changed.data(),
                                changed.size(), hash_a, &rehashed) != 0){
        fprintf(stderr, "[hexagon] incremental hash of signal A failed; hashing it all\n");
        pearl_matrix_hash_invalidate(g_ahash);
        pearl_keyed_digest_int8(h_A_sig, (size_t)m * K_DIM, job_key, hash_a);
        rehashed = (size_t)m * K_DIM / 1024;
    }
    const double hash_sec = cp_now_sec() - t_hash;
    if(g_hash_check){
        uint8_t full[32];
        pearl_keyed_digest_int8(h_A_sig, (size_t)m * K_DIM, job_key, full);
        printf("[hexagon] hash check: incremental hash_a %s the full hash\n",
               memcmp(full, hash_a, 32) == 0 ? "matches" : "DIFFERS FROM");
        if(memcmp(full, hash_a, 32) != 0){
            pearl_matrix_hash_invalidate(g_ahash);
            memcpy(hash_a, full, 32);
        }
    }
    uint8_t a_seed[32];
    pearl_a_noise_seed_from_hash(g_job.b_noise_seed, hash_a, (uint32_t)m, g_job.salted, a_seed);
    std::vector<uint32_t> pairs_a((size_t)K_DIM * 2);
    pearl_build_perm_pairs_a(a_seed, K_DIM, R_RANK, pairs_a.data());

    uint32_t bound[8];
    cp_scale_jackpot_target(pool_tgt, bound);
    uint32_t a_key8[8];
    memcpy(a_key8, a_seed, 32);

    /* Launch grid: row panels of at most g_buf.rows, column blocks of at most g_buf.cols
     * (m and n are multiples of 1024, so every piece is a multiple of 128). */
    std::vector<Launch> launches;
    std::vector<int> panel_rows;
    for(int p = 0, row0 = 0; row0 < m; p++, row0 += g_buf.rows){
        const int rows = m - row0 < g_buf.rows ? m - row0 : g_buf.rows;
        const int per_panel = (n + g_buf.cols - 1) / g_buf.cols;
        panel_rows.push_back(rows);
        for(int c = 0, col0 = 0; col0 < n; c++, col0 += g_buf.cols){
            const int cols = n - col0 < g_buf.cols ? n - col0 : g_buf.cols;
            launches.push_back({p, row0, rows, col0, cols, c == 0, c, per_panel});
        }
    }
    const int S = (int)launches.size();
    const int panels = (int)panel_rows.size();

    double check_sec = 0.0, wait_sec = 0.0;
    uint32_t a_seed_label[16];
    seed_label_words(a_seed, PEARL_SEED_LABEL_A, a_seed_label);

    const double prep_sec = cp_now_sec() - attempt_t0;
    const double scan_t0 = cp_now_sec();
    double last_report = scan_t0;
    uint64_t tiles_done = 0;
    Hit hit;
    int rc = 0;

    printf("[hexagon] plain_proof scan %d launches of %dx%d (%d panels), %llu hash tiles, "
           "difficulty scaled by %llu; hash_a %.3fs (%zu of %zu chunks)\n",
           S, g_buf.rows, g_buf.cols, panels,
           (unsigned long long)((uint64_t)(m / kTileRows) * (uint64_t)(n / kTileCols)),
           (unsigned long long)cp_jackpot_scale_factor(), hash_sec, rehashed,
           (size_t)m * K_DIM / 1024);
    fflush(stdout);

    auto launch_tiles = [](const Launch& l) {
        return (uint64_t)(l.rows / kTileRows) * (uint64_t)(l.cols / kTileCols);
    };
    auto check = [&](int s) {
        const double t = cp_now_sec();
        const Launch& l = launches[s];
        Hit h = check_launch(l, g_buf.xr + (size_t)(s % kRecSlots) * g_buf.xr_words, a_key8,
                             bound);
        if(h.found && verify_hit(h, h_A_sig, a_seed, pairs_a.data(), a_key8, bound) != 0)
            h.found = 0;
        tiles_done += launch_tiles(l);
        check_sec += cp_now_sec() - t;
        return h;
    };

    /* The DSP runs the launches itself (pearlx scan_run), generating each row panel's noisy
     * A at its start and waiting only when the record slot it is about to reuse is not
     * checked yet. The host follows through the control block, checking each launch, and
     * sets its stop word on a hit or a new job. One call covers whole row panels with at
     * most kSegmentBytes of signal A; the DSP thread makes the calls back to back. */
    struct Segment {
        int row0, rows, s0, launches;
    };
    std::vector<Segment> segments;
    const int per_panel = launches[0].launches_in_panel;
    const int seg_panels = std::max(1, (int)(kSegmentBytes / ((size_t)g_buf.rows * K_DIM)));
    for(int p = 0; p < panels; p += seg_panels){
        const int np = std::min(seg_panels, panels - p);
        const int row0 = p * g_buf.rows;
        int rows = 0;
        for(int q = p; q < p + np; q++) rows += panel_rows[q];
        segments.push_back({row0, rows, p * per_panel, np * per_panel});
    }

    struct Run {
        std::atomic<bool> stop{false};
        std::atomic<bool> done{false};
        int err = 0;
        uint64_t dsp_us = 0, wait_us = 0;
    } run;
    const uint32_t epoch = ++g_buf.epoch;
    ctl_store(CP_PEARLX_CTL_CHECKED, 0);
    ctl_store(CP_PEARLX_CTL_STOP, 0);
    ctl_store(CP_PEARLX_CTL_EPOCH, epoch);

    g_dsp->start([&]() {
        for(const Segment& g : segments){
            if(run.stop.load()) break;
            uint64_t dsp_us = 0, wait_us = 0;
            int done = 0;
            const int err = cp_pearlx_scan_run(
                g_px, h_A_sig + (size_t)g.row0 * K_DIM, g.row0, g.rows, K_DIM, g_buf.rows,
                g_buf.cols, g.s0, kRecSlots, 0, a_seed_label, pairs_a.data(), g_buf.ctl,
                g_buf.xr, kRecSlots * g_buf.xr_words, &dsp_us, &wait_us, &done);
            run.dsp_us += dsp_us;
            run.wait_us += wait_us;
            if(err){
                run.err = err;
                break;
            }
            if(done < g.launches) break;   /* stopped */
        }
        run.done.store(true);
    });
    /* Launches the DSP has finished in this attempt (0 until it has seen this epoch). */
    auto finished = [&]() {
        return ctl_load(CP_PEARLX_CTL_EPOCH_ACK) == epoch ? (int)ctl_load(CP_PEARLX_CTL_LAUNCHED)
                                                          : 0;
    };

    for(int s = 0; s < S; s++){
        const double t = cp_now_sec();
        bool cancelled = false;
        while(finished() <= s && !run.done.load()){
            if(cp_job_should_cancel()){
                cancelled = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        wait_sec += cp_now_sec() - t;
        if(cancelled)
            break;
        if(finished() <= s){
            if(run.err){
                fprintf(stderr, "[hexagon] pearlx scan_run failed (0x%x)\n", run.err);
                rc = -2;
            }
            break;
        }
        hit = check(s);
        ctl_store(CP_PEARLX_CTL_CHECKED, (uint32_t)(s + 1));
        if(hit.found || cp_job_should_cancel())
            break;

        const double now = cp_now_sec();
        if(now - last_report >= 5.0){
            char mac_buf[32];
            cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(tiles_done, now - scan_t0), mac_buf,
                               sizeof(mac_buf));
            printf("[hexagon] plain_proof progress: launch %d/%d (%.1f%%) %s\n", s + 1, S,
                   100.0 * (s + 1) / S, mac_buf);
            fflush(stdout);
            last_report = now;
        }
    }
    /* Stops the DSP mid-launch after a hit or a cancel; after the last launch a no-op. */
    run.stop.store(true);
    ctl_store(CP_PEARLX_CTL_STOP, 1);
    g_dsp->wait();

    const double scan_sec = cp_now_sec() - scan_t0;
    if(out_tiles_scanned) *out_tiles_scanned = tiles_done;
    cp_log_attempt_timing("hexagon", prep_sec, scan_sec, tiles_done, 0.0);
    printf("[hexagon] scan %.3fs: DSP busy %.3fs, waited for the host %.3fs; host jackpot "
           "%.3fs, waiting for the DSP %.3fs\n", scan_sec, run.dsp_us / 1e6, run.wait_us / 1e6,
           check_sec, wait_sec);
    fflush(stdout);
    if(rc != 0)
        return rc;
    if(hit.found){
        printf("[hexagon] plain_proof SHARE t_rows=%d t_cols=%d (host recompute matches DSP)\n",
               hit.t_rows, hit.t_cols);
        fflush(stdout);
        if(out_t_rows) *out_t_rows = hit.t_rows;
        if(out_t_cols) *out_t_cols = hit.t_cols;
        return 1;
    }
    if(cp_job_should_cancel())
        return -1;
    return 0;
}
