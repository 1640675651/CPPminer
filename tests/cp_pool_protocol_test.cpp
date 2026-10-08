#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_util.h"
#include "cp_pool_session.hpp"
#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "cp_platform.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>

static void test_response_ids()
{
    int id = -1, accepted = 0;
    assert(cp_json_rpc_response("{\"id\":7,\"result\":true,\"error\":null}", &id, &accepted));
    assert(id == 7 && accepted);
    assert(cp_json_rpc_response("{\"result\":{\"id\":99},\"id\":7}", &id, &accepted));
    assert(id == 7 && accepted);
    assert(cp_json_rpc_response("{\"id\":0,\"result\":false}", &id, &accepted));
    assert(id == 0 && !accepted);
    assert(cp_json_rpc_response("{\"id\":7,\"error\":{\"code\":-1}}", &id, &accepted));
    assert(!accepted);
    assert(!cp_json_rpc_response("{\"result\":{\"id\":7}}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":null,\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":\"7\",\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":7.5,\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":2147483648,\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":7,\"method\":\"job\",\"result\":true}", &id, &accepted));
}

static void test_submit_deadlines()
{
    CpPoolSession session;
    assert(session.begin_submit(2, 100));
    assert(!session.finish_submit(999)); // Unrelated responses must not clear the request.
    assert(!session.begin_submit(2, 110)); // Nor may duplicate ids move its deadline.
    assert(session.begin_submit(3, 150));
    assert(session.expired(159.9) == CpPoolSession::Timeout::None);
    assert(session.expired(160) == CpPoolSession::Timeout::Submit);
    assert(session.finish_submit(3)); // An out-of-order ACK leaves id 2 outstanding.
    assert(session.expired(165) == CpPoolSession::Timeout::Submit);
    assert(session.finish_submit(2));
    assert(!session.has_pending_submits());
    assert(session.expired(200) == CpPoolSession::Timeout::None);

    // A fast ACK can arrive on the reader thread before send returns.
    assert(session.begin_submit(4, 200));
    std::thread reader([&] { assert(session.finish_submit(4)); });
    reader.join();
    assert(!session.has_pending_submits());
    assert(session.expired(300) == CpPoolSession::Timeout::None);

    assert(session.begin_submit(5, 300));
    assert(session.finish_submit(5)); // Send failure removes only that request.
    assert(!session.has_pending_submits());
    assert(session.begin_submit(6, 300));
    session.reset(); // Reconnect discards old ids/deadlines.
    assert(!session.finish_submit(6));
    assert(session.expired(400) == CpPoolSession::Timeout::None);
}

static void test_handshake_deadlines()
{
    CpPoolSession session;
    session.begin_authorize(1, 100);
    assert(!session.authorize_response(999, true, 110));
    assert(session.expired(129.9) == CpPoolSession::Timeout::None);
    assert(session.expired(130) == CpPoolSession::Timeout::Authorize);

    session.reset();
    session.begin_authorize(2, 200);
    assert(session.authorize_response(2, false, 201));
    assert(!session.authorized());

    session.reset();
    session.begin_authorize(3, 300);
    assert(session.authorize_response(3, true, 305));
    assert(session.authorized());
    assert(session.expired(334.9) == CpPoolSession::Timeout::None);
    assert(session.expired(335) == CpPoolSession::Timeout::FirstJob);
    session.received_job();
    assert(session.expired(400) == CpPoolSession::Timeout::None);

    // Some pools notify before the authorize ACK; keep the job but still wait for ACK.
    session.reset();
    session.begin_authorize(4, 400);
    session.received_job();
    assert(!session.authorized());
    assert(session.authorize_response(4, true, 410));
    assert(session.authorized());
    assert(session.expired(500) == CpPoolSession::Timeout::None);

    session.reset();
    session.begin_authorize(5, 500);
    session.received_job();
    assert(session.expired(530) == CpPoolSession::Timeout::Authorize);
    session.reset();
    assert(!session.authorized());
}

static void test_work_identity_and_cancellation()
{
    uint8_t header[INCOMPLETE_HEADER_BYTES] = {};
    uint32_t target[8] = {};
    char key[CP_JOB_KEY_CAP], changed[CP_JOB_KEY_CAP];
    char job_id[128];
    memset(job_id, 'j', sizeof(job_id) - 1);
    job_id[sizeof(job_id) - 1] = 0;
    assert(cp_pearl_job_key(key, sizeof(key), job_id, header, sizeof(header), target, 3));
    header[75] = 1; // Same ID and first eight bytes, different work.
    assert(cp_pearl_job_key(changed, sizeof(changed), job_id, header, sizeof(header), target, 3));
    assert(strcmp(key, changed));
    header[75] = 0;
    target[0] = 1;
    assert(cp_pearl_job_key(changed, sizeof(changed), job_id, header, sizeof(header), target, 3));
    assert(strcmp(key, changed));
    target[0] = 0;
    assert(cp_pearl_job_key(changed, sizeof(changed), job_id, header, sizeof(header), target, 2));
    assert(strcmp(key, changed));
    assert(!cp_pearl_job_key(changed, 8, job_id, header, sizeof(header), target, 3));
    assert(!changed[0]);

    cp_job_mine_begin(key);
    assert(cp_job_key_matches(key) && !cp_job_should_cancel());
    cp_job_request_cancel();
    assert(cp_job_should_cancel()); // A proof must stop even though the old key still matches.
    cp_job_mine_end();
    cp_job_mine_begin(key);
    assert(!cp_job_should_cancel()); // A new mining epoch clears the old cancellation.
    cp_job_mine_end();
}

int main()
{
    std::vector<std::thread> clocks;
    for(int i = 0; i < 4; ++i){
        clocks.emplace_back([] {
            double previous = cp_now_sec();
            for(int j = 0; j < 1000; ++j){
                const double before = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                const double now = cp_now_sec();
                const double after = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                assert(now >= previous && now >= before && now <= after);
                previous = now;
            }
        });
    }
    for(auto& clock : clocks) clock.join();
    test_response_ids();
    test_submit_deadlines();
    test_handshake_deadlines();
    test_work_identity_and_cancellation();
#ifdef __linux__
    int pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    close(pair[1]);
    assert(!cp_send_all(pair[0], "test", 4)); // Broken peer returns an error, not SIGPIPE exit.
    close(pair[0]);
#endif
    size_t len = 0;
    const char* combined = "{\"job_id\":\"a}b\\\"{c\",\"target\":[1,2]}{\"id\":2}";
    assert(cp_json_object_length(combined, strlen(combined), &len) == 1);
    assert(len == strlen("{\"job_id\":\"a}b\\\"{c\",\"target\":[1,2]}"));
    const char* partial = "{\"id\":\"unfinished";
    assert(cp_json_object_length(partial, strlen(partial), &len) == 0);
    assert(cp_json_object_length("{\"id\":[1}}", strlen("{\"id\":[1}}"), &len) == -1);

    char value[16];
    assert(cp_json_str("{\"note\":\"fake \\\"id\\\":\\\"x\\\"\",\"id\":\"a\\\"b\"}",
                       "id", value, sizeof(value)) == 1);
    assert(strcmp(value, "a\"b") == 0);
    assert(cp_json_str("{\"id\":\"12345678901234567\"}", "id", value, sizeof(value)) == 0);
    assert(value[0] == 0);
    assert(cp_json_str("{\"id\":\"unterminated}", "id", value, sizeof(value)) == 0);
    assert(cp_json_num("{\"note\":\"fake \\\"seq\\\":99\",\"seq\":7}", "seq") == 7);

    uint8_t bytes[2] = {};
    assert(cp_hex_to_bytes("0aFf", bytes, 2) == 2 && bytes[0] == 10 && bytes[1] == 255);
    assert(cp_hex_to_bytes("0g", bytes, 2) == 0);
    assert(cp_hex_to_bytes("g0", bytes, 2) == 0);
    assert(cp_json_escape("a\"\\\nb") == "a\\\"\\\\\\u000ab");

    /* stratum difficulty -> target: (0xFFFF << 208) / diff, example job from the pool */
    uint32_t tgt[8];
    char hex[65];
    cp_pool_target_from_difficulty(26000.0, tgt);
    cp_le_words_to_be_target_hex(tgt, hex);
    assert(strcmp(hex, "0000000000028544877baaede211544877baaede211544877baaede211544877") == 0);
    cp_pool_target_from_difficulty(1.0, tgt);
    cp_le_words_to_be_target_hex(tgt, hex);
    assert(strcmp(hex, "00000000ffff0000000000000000000000000000000000000000000000000000") == 0);
    /* divisors above 2^32 must stay exact */
    cp_pool_target_from_difficulty(5000000000.0, tgt);
    cp_le_words_to_be_target_hex(tgt, hex);
    assert(strcmp(hex, "0000000000000000dbe622e7bf1f17d0dfb48e9b1679f292e2a3d73dc7945c4b") == 0);
    cp_pool_target_from_difficulty(18446744073709549568.0, tgt); /* 2^64 - 2048, largest double < 2^64 */
    cp_le_words_to_be_target_hex(tgt, hex);
    assert(strcmp(hex, "000000000000000000000000ffff0000000007fff8000000003fffc000000001") == 0);
    /* mock: 2^(256 - D), independent of the hash tile */
    cp_mock_target_from_difficulty(44.0, tgt);
    cp_le_words_to_be_target_hex(tgt, hex);
    assert(strcmp(hex, "0000000000100000000000000000000000000000000000000000000000000000") == 0);
    return 0;
}
