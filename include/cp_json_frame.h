#ifndef CP_JSON_FRAME_H
#define CP_JSON_FRAME_H

#include <stddef.h>
#include <stdint.h>
#include <string>

/* Scan a JSON object starting at '{'. 1 = complete, 0 = incomplete,
 * -1 = malformed or too deeply nested. Braces inside strings are ignored. */
int cp_json_object_length(const char* data, size_t len, size_t* object_len);
int cp_json_value_length(const char* data, size_t len, size_t* value_len);
int cp_json_valid(const char* json);
int cp_json_decode_string(const char* value, std::string& out);
const char* cp_json_find_member(const char* json, const char* key);
int cp_json_number_value(const char* value, double* out);
int cp_json_uint64_value(const char* value, uint64_t* out);

#endif
