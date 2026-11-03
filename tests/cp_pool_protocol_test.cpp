/* Before any include: a header that pulls in <cassert> first would keep assert off in Release. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_pool.h"
#include "cp_state.h"
#include "cp_submit_table.hpp"
#include "cp_util.h"

#include <cassert>
#include <cstring>
#include <string>

struct Notify {
    char job_id[16], header[320], target[80];
    uint32_t cert_version;
    int ok;
};

static Notify parse_notify(const std::string& json)
{
    Notify n;
    n.ok = cp_pool_parse_notify(json.c_str(), n.job_id, sizeof(n.job_id), n.header,
                                sizeof(n.header), n.target, sizeof(n.target), &n.cert_version);
    return n;
}

static void test_parse_notify()
{
    const std::string h = std::string(HEADER_HEX_LEN, 'a'), t = std::string(TARGET_HEX_LEN, 'f');
    const std::string obj = "{\"id\":null,\"method\":\"mining.notify\",\"params\":{";
    Notify n = parse_notify(obj + "\"job_id\":\"j1\",\"header\":\"" + h + "\",\"target\":\"" + t +
                            "\",\"cert_version\":2}}");
    assert(n.ok && !strcmp(n.job_id, "j1") && n.header == h && n.target == t && n.cert_version == 2);
    /* cert_version beside params; no target. */
    n = parse_notify("{\"cert_version\":3,\"params\": {\"job_id\":\"j1\",\"header\":\"" + h + "\"}}");
    assert(n.ok && n.cert_version == 3 && !n.target[0]);
    /* Fields only from params itself, not from nested objects. */
    n = parse_notify(obj + "\"extra\":{\"job_id\":\"decoy\",\"target\":\"" + t + "\"},\"header\":\"" +
                     h + "\",\"job_id\":\"real\"}}");
    assert(n.ok && !strcmp(n.job_id, "real") && !n.target[0]);
    /* No params: members of the message itself. */
    n = parse_notify("{\"method\":\"mining.notify\",\"job_id\":\"j1\",\"header\":\"" + h + "\"}");
    assert(n.ok && !strcmp(n.job_id, "j1"));
    /* The target is copied as sent; callers check it. */
    n = parse_notify(obj + "\"job_id\":\"j1\",\"header\":\"" + h + "\",\"target\":\"" +
                     std::string(TARGET_HEX_LEN, 'z') + "\"}}");
    assert(n.ok && n.target[0] == 'z');
    /* Rejected: no header, no / empty / over-long job id, unknown cert_version. */
    assert(!parse_notify(obj + "\"job_id\":\"j1\"}}").ok);
    assert(!parse_notify(obj + "\"header\":\"" + h + "\"}}").ok);
    assert(!parse_notify(obj + "\"job_id\":\"\",\"header\":\"" + h + "\"}}").ok);
    assert(!parse_notify(obj + "\"job_id\":\"0123456789abcdef\",\"header\":\"" + h + "\"}}").ok);
    const std::string cv4 = obj + "\"job_id\":\"j1\",\"header\":\"" + h + "\",\"cert_version\":4}}";
    assert(!parse_notify(cv4).ok);
    assert(!parse_notify(obj + "\"job_id\":\"j1\",\"header\":\"" + h + "\",\"cert_version\":2.5}}").ok);
    g_cert_version_forced = 1; /* --cert-version overrides whatever the pool sends */
    n = parse_notify(cv4);
    assert(n.ok && n.cert_version == 0);
    g_cert_version_forced = 0;

    /* Positional params: id first, then target and header by length; a trailing object. */
    n = parse_notify("{\"method\":\"mining.notify\",\"params\":[\"j2\", \"" + h + "\", \"" + t +
                     "\", {\"cert_version\":1}]}");
    assert(n.ok && !strcmp(n.job_id, "j2") && n.header == h && n.target == t && n.cert_version == 1);
    assert(!parse_notify("{\"params\":[\"0123456789abcdef\",\"" + h + "\"]}").ok);
    assert(!parse_notify("{\"params\":[\"" + h + "\"]}").ok); /* the only string is the id */
}

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

    test_parse_notify();

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
