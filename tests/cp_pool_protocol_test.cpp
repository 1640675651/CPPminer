/* Before any include: a header that pulls in <cassert> first would keep assert off in Release. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_submit_table.hpp"
#include "cp_util.h"

#include <cassert>
#include <cstring>

int main()
{
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

    /* Outermost members only: the Quantus login nests its session "id" in result. */
    const char* login = "{\"result\":{\"id\":\"sess\",\"status\":\"OK\"},\"jsonrpc\":\"2.0\",\"id\":5,\"error\":null}";
    int id = 0;
    assert(cp_json_rpc_response_id(login, &id) == 1 && id == 5);
    assert(!cp_json_rpc_rejected(login));
    assert(cp_json_member(login, "status") == NULL);
    assert(cp_json_member("{\"a\":{\"b\":1}}", "b") == NULL);
    assert(cp_json_member("{\"note\":\"\\\"id\\\":7\",\"x\":1}", "id") == NULL);
    const char* params = cp_json_member("{ \"params\" : [ 512 ] }", "params");
    assert(params && params[0] == '[');
    assert(!cp_json_rpc_response_id("{\"id\":3,\"method\":\"job\",\"params\":{}}", &id));
    assert(cp_json_rpc_response_id("{\"id\":\"3\",\"result\":true}", &id) && id == 3);
    assert(!cp_json_rpc_response_id("{\"id\":\"3x\",\"result\":true}", &id));
    assert(!cp_json_rpc_response_id("{\"id\":\"\",\"result\":true}", &id));
    assert(!cp_json_rpc_response_id("{\"id\":null,\"result\":true}", &id));
    assert(!cp_json_rpc_response_id("{\"id\":3.5,\"result\":true}", &id));

    /* Each submit waits for its own reply; the oldest one sets the deadline. */
    CpSubmitTable submits;
    double sent_at = 0.0;
    assert(!submits.oldest(&sent_at));
    submits.track(7, 100.0);
    submits.track(8, 130.0);
    assert(submits.oldest(&sent_at) && sent_at == 100.0);
    assert(!submits.finish(9) && submits.pending(7));
    assert(submits.finish(8) && !submits.pending(8));
    assert(submits.oldest(&sent_at) && sent_at == 100.0); /* a newer reply leaves 7 waiting */
    submits.track(9, 150.0);
    assert(submits.finish(7) && submits.oldest(&sent_at) && sent_at == 150.0);
    submits.clear();
    assert(!submits.pending(9) && !submits.oldest(&sent_at));
    assert(cp_json_rpc_rejected(
        "{\"error\":{\"code\":24,\"msg\":\"login is missing PRL wallet\"},\"id\":1,\"result\":null}"));
    assert(cp_json_rpc_rejected("{\"id\":1,\"result\":false}"));
    assert(!cp_json_rpc_rejected("{\"id\":1,\"result\":true,\"error\":null}"));
    assert(!cp_json_rpc_rejected("{\"id\":1,\"result\":{\"type\":\"v2\"}}"));

    const double t0 = cp_now_sec(), t1 = cp_now_sec();
    assert(t1 >= t0);

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
