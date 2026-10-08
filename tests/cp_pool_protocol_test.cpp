#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_util.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
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
    /* mock: 2^(256 - D), independent of the hash tile */
    cp_mock_target_from_difficulty(44.0, tgt);
    cp_le_words_to_be_target_hex(tgt, hex);
    assert(strcmp(hex, "0000000000100000000000000000000000000000000000000000000000000000") == 0);
    return 0;
}
