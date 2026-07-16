#include "cp_share_queue.h"

#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_pool.h"
#include "cp_proof.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace {

struct ShareSnapshot {
    uint64_t nonce = 0;
    int t_rows = -1;
    int t_cols = -1;
    uint64_t tiles_at_hit = 0;
    double hit_elapsed_sec = 0.0;

    char job_key[320]{};
    char job_id[128]{};
    char target_hex[80]{};
    uint8_t header[INCOMPLETE_HEADER_BYTES]{};
    int header_len = 0;

    int8_t *a_sig = nullptr;
    size_t sz_a = 0;
    int8_t *bt_sig = nullptr;
    size_t sz_bt = 0;
    int bt_owned = 0;
};

static int verify_proof_file(const char *hdr_path, const char *target_hex, const char *proof_path) {
    uint8_t header[INCOMPLETE_HEADER_BYTES];
    uint8_t target_be[32];
    char errbuf[4096];
    char *b64 = nullptr;
    size_t cap = 0;
    size_t n = 0;
    FILE *pf = nullptr;

    if (!target_hex || !target_hex[0]) {
        return -1;
    }

    FILE *hf = fopen(hdr_path, "rb");
    if (!hf) {
        perror("verify: header open");
        return -1;
    }
    if (fread(header, 1, sizeof(header), hf) != sizeof(header)) {
        fprintf(stderr, "verify: header must be %d bytes\n", INCOMPLETE_HEADER_BYTES);
        fclose(hf);
        return -1;
    }
    fclose(hf);

    if (cp_hex_to_bytes(target_hex, target_be, 32) != 32) {
        fprintf(stderr, "verify: invalid target hex\n");
        return -1;
    }

    pf = fopen(proof_path, "rb");
    if (!pf) {
        perror("verify: proof open");
        return -1;
    }
    fseek(pf, 0, SEEK_END);
    const long fsz = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    if (fsz <= 0 || fsz > (long)PLAIN_PROOF_B64_MAX) {
        fprintf(stderr, "verify: invalid proof size %ld\n", fsz);
        fclose(pf);
        return -1;
    }
    cap = (size_t)fsz + 1;
    b64 = (char *)malloc(cap);
    if (!b64) {
        fclose(pf);
        return -1;
    }
    n = fread(b64, 1, (size_t)fsz, pf);
    fclose(pf);
    if (n != (size_t)fsz) {
        free(b64);
        return -1;
    }
    while (n > 0 && (b64[n - 1] == '\n' || b64[n - 1] == '\r')) {
        b64[--n] = 0;
    }

    errbuf[0] = 0;
    if (cp_proof_verify(header, sizeof(header), (const uint8_t *)b64, n, target_be, errbuf,
                        sizeof(errbuf)) != 0) {
        fprintf(stderr, "verify FAIL: %s\n", errbuf[0] ? errbuf : "unknown");
        free(b64);
        return -1;
    }
    free(b64);
    return 0;
}

static void share_snapshot_free(ShareSnapshot *snap) {
    if (!snap) {
        return;
    }
    free(snap->a_sig);
    if (snap->bt_owned) {
        free(snap->bt_sig);
    }
    delete snap;
}

static const int8_t *share_bt_for_proof(const ShareSnapshot *snap) {
    if (snap->bt_owned && snap->bt_sig) {
        return snap->bt_sig;
    }
    return h_BpT_global;
}

struct CpShareQueueImpl {
    explicit CpShareQueueImpl(int max_depth_in) : max_depth(max_depth_in > 0 ? max_depth_in : 1) {}

    const int max_depth;
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<ShareSnapshot *> pending;
    std::thread worker;
    bool shutdown = false;
    bool job_active = false;
    int in_flight = 0;
    CpShareJobCtx job_ctx{};
    char job_key[320]{};

    void worker_main();
    void process_snapshot(ShareSnapshot *snap);
};

void CpShareQueueImpl::process_snapshot(ShareSnapshot *snap) {
    if (!snap) {
        return;
    }

    if (!cp_job_key_matches(snap->job_key)) {
        printf("[plain] stale share nonce=%llu dropped (job changed)\n",
               (unsigned long long)snap->nonce);
        fflush(stdout);
        share_snapshot_free(snap);
        return;
    }

    printf("[plain] %s hit nonce=%llu t_rows=%d t_cols=%d - building proof (async)...\n",
           cp_worker_backend_name(), (unsigned long long)snap->nonce, snap->t_rows, snap->t_cols);
    fflush(stdout);

    char *b64 = (char *)malloc(PLAIN_PROOF_B64_MAX);
    if (!b64) {
        fprintf(stderr, "[plain] OOM proof b64 buffer (nonce=%llu)\n",
                (unsigned long long)snap->nonce);
        share_snapshot_free(snap);
        return;
    }

    int tile_layout = cp_worker_default_tile_layout();
    if (g_cutlass_fused) {
        tile_layout = CP_TILE_LAYOUT_CUTLASS;
    }

    const uint8_t *mining_cfg = PEARL_SCATTERED_CONFIG;
    if (tile_layout == CP_TILE_LAYOUT_CUTLASS) {
        mining_cfg = PEARL_CUTLASS_CONFIG;
    } else if (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS) {
        mining_cfg = PEARL_CONTIGUOUS_CONFIG;
    }

    char errbuf[512];
    const int8_t *bt = share_bt_for_proof(snap);
    const int prc =
            cp_proof_build(snap->header, (size_t)snap->header_len, mining_cfg, 52, snap->a_sig, bt,
                           job_ctx.m, job_ctx.n, K_DIM, R_RANK, snap->t_rows, snap->t_cols,
                           tile_layout, b64, PLAIN_PROOF_B64_MAX, errbuf, sizeof(errbuf));
    if (prc != 0) {
        printf("[plain] proof build failed (nonce=%llu): %s\n", (unsigned long long)snap->nonce,
               errbuf[0] ? errbuf : "unknown");
        fflush(stdout);
        free(b64);
        share_snapshot_free(snap);
        return;
    }

    const int bn = (int)strlen(b64);
    if (bn < 32) {
        printf("[plain] proof too short (%d) nonce=%llu\n", bn, (unsigned long long)snap->nonce);
        fflush(stdout);
        free(b64);
        share_snapshot_free(snap);
        return;
    }

    if (g_dry_run || g_plain_verify) {
        FILE *pf = fopen(job_ctx.proof_path, "wb");
        if (pf) {
            fwrite(b64, 1, (size_t)bn, pf);
            fclose(pf);
        }
    }

    if (!cp_job_key_matches(snap->job_key)) {
        printf("[plain] stale share nonce=%llu dropped before verify/submit\n",
               (unsigned long long)snap->nonce);
        fflush(stdout);
        free(b64);
        share_snapshot_free(snap);
        return;
    }

    if (g_plain_verify && snap->target_hex[0]) {
        if (verify_proof_file(job_ctx.hdr_path, snap->target_hex, job_ctx.proof_path) != 0) {
            printf("[plain] verify failed (nonce=%llu)\n", (unsigned long long)snap->nonce);
            fflush(stdout);
            free(b64);
            share_snapshot_free(snap);
            return;
        }
        printf("[plain] verify OK (nonce=%llu)\n", (unsigned long long)snap->nonce);
        fflush(stdout);
    }

    double hs = cp_pp_mac_rate_from_tiles(snap->tiles_at_hit, snap->hit_elapsed_sec);
    if (snap->hit_elapsed_sec < 1e-3) {
        hs = cp_pp_mac_rate_from_tiles(snap->tiles_at_hit, 1e-3);
    }
    {
        char mac_buf[32];
        cp_pp_fmt_mac_rate(hs, mac_buf, sizeof(mac_buf));
        printf("[plain] proof ready (%d chars) nonce=%llu %s (hs=%.0f)\n", bn,
               (unsigned long long)snap->nonce, mac_buf, hs);
    }
    fflush(stdout);

    if (g_dry_run) {
        printf("[plain] dry-run: proof saved to %s (nonce=%llu)\n", job_ctx.proof_path,
               (unsigned long long)snap->nonce);
        fflush(stdout);
        free(b64);
        share_snapshot_free(snap);
        return;
    }

    if (!job_ctx.msg_id) {
        free(b64);
        share_snapshot_free(snap);
        return;
    }

    const int submit_id = (*job_ctx.msg_id)++;
    if (!cp_pool_send_plain_proof_submit(job_ctx.sock, submit_id, snap->job_id, b64, hs)) {
        printf("[plain] submit failed (nonce=%llu)\n", (unsigned long long)snap->nonce);
        fflush(stdout);
        free(b64);
        share_snapshot_free(snap);
        return;
    }

    cp_pool_set_submit_inflight(1);
    printf("[net] plain_proof submit sent (nonce=%llu)\n", (unsigned long long)snap->nonce);
    cp_pool_log_share_submit_outcome();
    fflush(stdout);

    free(b64);
    share_snapshot_free(snap);
}

void CpShareQueueImpl::worker_main() {
    for (;;) {
        ShareSnapshot *snap = nullptr;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return shutdown || !pending.empty(); });
            if (shutdown && pending.empty()) {
                break;
            }
            snap = pending.front();
            pending.pop_front();
            in_flight++;
        }

        process_snapshot(snap);

        {
            std::lock_guard<std::mutex> lock(mtx);
            in_flight--;
            cv.notify_all();
        }
    }
}

} /* namespace */

struct CpShareQueue {
    CpShareQueueImpl impl;
    explicit CpShareQueue(int max_depth) : impl(max_depth) {}
};

extern "C" CpShareQueue *cp_share_queue_create(int max_depth) {
    auto *q = new (std::nothrow) CpShareQueue(max_depth);
    if (!q) {
        return nullptr;
    }
    q->impl.worker = std::thread(&CpShareQueueImpl::worker_main, &q->impl);
    return q;
}

extern "C" void cp_share_queue_destroy(CpShareQueue *q) {
    if (!q) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(q->impl.mtx);
        q->impl.shutdown = true;
    }
    q->impl.cv.notify_all();
    if (q->impl.worker.joinable()) {
        q->impl.worker.join();
    }
    while (!q->impl.pending.empty()) {
        share_snapshot_free(q->impl.pending.front());
        q->impl.pending.pop_front();
    }
    delete q;
}

extern "C" void cp_share_queue_begin_job(CpShareQueue *q, const CpShareJobCtx *ctx,
                                         const char *job_key) {
    if (!q || !ctx || !job_key) {
        return;
    }
    std::lock_guard<std::mutex> lock(q->impl.mtx);
    q->impl.job_ctx = *ctx;
    strncpy(q->impl.job_key, job_key, sizeof(q->impl.job_key) - 1);
    q->impl.job_key[sizeof(q->impl.job_key) - 1] = '\0';
    q->impl.job_active = true;
}

extern "C" void cp_share_queue_end_job(CpShareQueue *q) {
    if (!q) {
        return;
    }
    std::unique_lock<std::mutex> lock(q->impl.mtx);
    q->impl.job_active = false;
    q->impl.cv.wait(lock, [&] { return q->impl.pending.empty() && q->impl.in_flight == 0; });
}

extern "C" int cp_share_queue_enqueue_hit(CpShareQueue *q, const CpShareHit *hit,
                                          const uint8_t *header, int hlen, const char *job_id,
                                          const char *target_hex, const int8_t *a_src, size_t sz_a,
                                          const int8_t *bt_src, size_t sz_bt) {
    if (!q || !hit || !header || hlen <= 0 || !job_id || !a_src || sz_a == 0) {
        return -1;
    }

    auto *snap = new (std::nothrow) ShareSnapshot();
    if (!snap) {
        return -1;
    }

    snap->nonce = hit->nonce;
    snap->t_rows = hit->t_rows;
    snap->t_cols = hit->t_cols;
    snap->tiles_at_hit = hit->tiles_at_hit;
    snap->hit_elapsed_sec = hit->hit_elapsed_sec;
    snap->sz_a = sz_a;
    snap->sz_bt = sz_bt;

    strncpy(snap->job_id, job_id, sizeof(snap->job_id) - 1);
    snap->job_id[sizeof(snap->job_id) - 1] = '\0';
    strncpy(snap->job_key, q->impl.job_key, sizeof(snap->job_key) - 1);
    snap->job_key[sizeof(snap->job_key) - 1] = '\0';
    if (target_hex) {
        strncpy(snap->target_hex, target_hex, sizeof(snap->target_hex) - 1);
        snap->target_hex[sizeof(snap->target_hex) - 1] = '\0';
    }

    const size_t hdr_copy = (size_t)hlen < sizeof(snap->header) ? (size_t)hlen : sizeof(snap->header);
    memcpy(snap->header, header, hdr_copy);
    snap->header_len = (int)hdr_copy;

    snap->a_sig = (int8_t *)malloc(sz_a);
    if (!snap->a_sig) {
        share_snapshot_free(snap);
        return -1;
    }
    memcpy(snap->a_sig, a_src, sz_a);

    if (hit->copy_bt && bt_src && sz_bt > 0) {
        snap->bt_sig = (int8_t *)malloc(sz_bt);
        if (!snap->bt_sig) {
            share_snapshot_free(snap);
            return -1;
        }
        memcpy(snap->bt_sig, bt_src, sz_bt);
        snap->bt_owned = 1;
    }

    {
        std::unique_lock<std::mutex> lock(q->impl.mtx);
        if (!q->impl.job_active || q->impl.shutdown) {
            share_snapshot_free(snap);
            return -1;
        }
        q->impl.cv.wait(lock, [&] {
            return q->impl.shutdown ||
                   (int)q->impl.pending.size() < q->impl.max_depth;
        });
        if (q->impl.shutdown || !q->impl.job_active) {
            share_snapshot_free(snap);
            return -1;
        }
        q->impl.pending.push_back(snap);
    }
    q->impl.cv.notify_one();
    return 0;
}
