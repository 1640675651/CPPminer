// Run the real share queue with a deterministic pause inside proof construction.
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
extern "C" int cp_proof_verify(const uint8_t*, size_t, const uint8_t*, size_t,
    const uint8_t*, uint32_t, char*, size_t) { assert(false); return -1; }
extern "C" int cp_pool_conn_lost(void) { return 0; }
extern "C" int cp_worker_proof_tile_layout(void) { return CP_TILE_LAYOUT_SCATTERED; }
extern "C" int cp_pool_send_plain_proof_submit(int, int, const char*, const char*, double)
{ ++submits; return 1; }
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

int main()
{
    run_case(0); // Control: unchanged work submits normally.
    run_case(1); // New notify while proof construction is running.
}
