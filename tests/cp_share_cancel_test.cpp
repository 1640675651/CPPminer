// Run the real share queue with deterministic pauses at the proof/verify boundary.
#include "cp_job_ctrl.h"
#include "cp_noise.h"
#include "cp_pool.h"
#include "cp_share_queue.h"
#include "cp_state.h"
#include "cp_worker.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>

const uint8_t PEARL_SCATTERED_CONFIG[52] = {};
const uint8_t PEARL_CONTIGUOUS_CONFIG[52] = {};
const uint8_t PEARL_CONTIGUOUS_8x8_CONFIG[52] = {};
const uint8_t PEARL_CONTIGUOUS_4x8_CONFIG[52] = {};
const uint8_t PEARL_CONTIGUOUS_16x16_CONFIG[52] = {};
const uint8_t PEARL_CUTLASS_CONFIG[52] = {};

static std::mutex gate;
static std::condition_variable cv;
static int pause_at = 0, entered = 0, submits = 0;
static bool released = false;
static int verify_result = 0, verify_calls = 0;

static void boundary(int stage)
{
    std::unique_lock<std::mutex> lock(gate);
    if(pause_at != stage) return;
    entered = stage;
    cv.notify_all();
    assert(cv.wait_for(lock, std::chrono::seconds(5), [] { return released; }));
}

extern "C" int cp_proof_build(const uint8_t*, size_t, const uint8_t*, size_t,
    const int8_t*, const int8_t*, int, int, int, int, int, int, int,
    char* out, size_t, char*, size_t)
{
    boundary(1);
    memset(out, 'A', 64);
    out[64] = 0;
    return 0;
}
extern "C" int cp_proof_build_witness(const uint8_t*, size_t, const uint8_t*, size_t,
    const CpMatrixWitness*, const CpMatrixWitness*, int, int, int, int, int, int, int,
    char*, size_t, char*, size_t) { assert(false); return -1; }
extern "C" int cp_proof_verify(const uint8_t* header, size_t header_len, const uint8_t* proof,
    size_t proof_len, const uint8_t* target, uint32_t cert, char*, size_t)
{
    ++verify_calls;
    assert(header_len == INCOMPLETE_HEADER_BYTES && header[0] == 7);
    assert(proof_len == 64 && proof[0] == 'A' && target[0] == 0xff && cert == 3);
    boundary(3);
    return verify_result;
}
extern "C" int cp_pool_conn_lost(void) { return 0; }
extern "C" int cp_worker_proof_tile_layout(void) { return CP_TILE_LAYOUT_SCATTERED; }
extern "C" int cp_pool_send_plain_proof_submit(int, int, const char*, const char* proof, double)
{
    assert(strlen(proof) == 64 && proof[0] == 'A');
    ++submits;
    return 1;
}
extern "C" void cp_pool_log_share_submit_outcome(void) {}

static void run_case(int stage)
{
    pause_at = stage;
    entered = submits = 0;
    released = false;
    int msg_id = 1;
    CpShareQueue* queue = cp_share_queue_create(1);
    assert(queue);
    CpShareJobCtx ctx = {1, &msg_id, 1, 1, 3, "unused-header", "unused-proof"};
    cp_job_mine_begin("test-job");
    cp_share_queue_begin_job(queue, &ctx, "test-job");
    uint8_t header[INCOMPLETE_HEADER_BYTES] = {};
    int8_t* matrix = (int8_t*)malloc(1);
    assert(matrix);
    CpShareHit hit = {0, 0, 0, 1, 1.0, 0};
    assert(cp_share_queue_enqueue_hit(queue, &hit, header, sizeof(header), "test-job",
                                      "", &matrix, 1, nullptr, 0) == 0);
    assert(!matrix); // The worker must return ownership even when cancellation drops a proof.
    if(stage){
        std::unique_lock<std::mutex> lock(gate);
        assert(cv.wait_for(lock, std::chrono::seconds(5), [stage] { return entered == stage; }));
        cp_job_request_cancel();
        released = true;
        cv.notify_all();
    }
    cp_share_queue_end_job(queue);
    assert(submits == (stage ? 0 : 1));
    assert(msg_id == (stage ? 1 : 2));
    assert(cp_share_queue_last_outcome(queue) ==
           (stage ? CP_SHARE_OUTCOME_DROPPED : CP_SHARE_OUTCOME_OK));
    cp_share_queue_reclaim_matrices(queue, &matrix, nullptr);
    assert(matrix);
    free(matrix);
    cp_share_queue_destroy(queue);
    cp_job_mine_end();
}

static void test_memory_verification(int fail, int cancel_stage = 0)
{
    pause_at = cancel_stage;
    entered = submits = verify_calls = 0;
    released = false;
    verify_result = fail;
    g_plain_verify = 1;
    int msg_id = 1;
    CpShareQueue* queue = cp_share_queue_create(1);
    CpShareJobCtx ctx = {1, &msg_id, 1, 1, 3, nullptr, nullptr}; // No filesystem dependency.
    cp_job_mine_begin("verified-job");
    cp_share_queue_begin_job(queue, &ctx, "verified-job");
    uint8_t header[INCOMPLETE_HEADER_BYTES] = {7};
    int8_t* matrix = (int8_t*)malloc(1);
    CpShareHit hit = {0, 0, 0, 1, 1.0, 0};
    const char* target = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    assert(cp_share_queue_enqueue_hit(queue, &hit, header, sizeof(header), "verified-job",
                                      target, &matrix, 1, nullptr, 0) == 0);
    if(cancel_stage){
        std::unique_lock<std::mutex> lock(gate);
        assert(cv.wait_for(lock, std::chrono::seconds(5), [] { return entered == 3; }));
        cp_job_request_cancel();
        released = true;
        cv.notify_all();
    }
    cp_share_queue_end_job(queue);
    assert(verify_calls == 1);
    assert(submits == (!fail && !cancel_stage ? 1 : 0));
    assert(msg_id == submits + 1);
    assert(cp_share_queue_last_outcome(queue) == (fail ? CP_SHARE_OUTCOME_VERIFY_FAIL :
           cancel_stage ? CP_SHARE_OUTCOME_DROPPED : CP_SHARE_OUTCOME_OK));
    cp_share_queue_reclaim_matrices(queue, &matrix, nullptr);
    assert(matrix);
    free(matrix);
    cp_share_queue_destroy(queue);
    cp_job_mine_end();
    g_plain_verify = 0;
}

static void test_diagnostic_write_failure()
{
    pause_at = 0;
    g_dry_run = 1;
    CpShareQueue* queue = cp_share_queue_create(1);
    CpShareJobCtx ctx = {-1, nullptr, 1, 1, 3, nullptr, nullptr};
    cp_job_mine_begin("dry-job");
    cp_share_queue_begin_job(queue, &ctx, "dry-job");
    uint8_t header[INCOMPLETE_HEADER_BYTES] = {};
    int8_t* matrix = (int8_t*)malloc(1);
    CpShareHit hit = {0, 0, 0, 1, 1.0, 0};
    assert(cp_share_queue_enqueue_hit(queue, &hit, header, sizeof(header), "dry-job", "",
                                      &matrix, 1, nullptr, 0) == 0);
    cp_share_queue_end_job(queue);
    assert(cp_share_queue_last_outcome(queue) == CP_SHARE_OUTCOME_PROOF_FAIL);
    cp_share_queue_reclaim_matrices(queue, &matrix, nullptr);
    assert(matrix);
    free(matrix);
    cp_share_queue_destroy(queue);
    cp_job_mine_end();
    g_dry_run = 0;
}

int main()
{
    run_case(0); // Control: unchanged work submits normally.
    run_case(1); // New notify while proof construction is running.
    test_memory_verification(0);
    test_memory_verification(-1);
    test_memory_verification(0, 3); // New notify while the proof is being verified.
    test_diagnostic_write_failure();
}
