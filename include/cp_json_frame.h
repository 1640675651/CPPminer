#ifndef CP_JSON_FRAME_H
#define CP_JSON_FRAME_H

#include <stddef.h>

/* Scan a JSON object starting at '{'. 1 = complete, 0 = incomplete,
 * -1 = malformed or too deeply nested. Braces inside strings are ignored. */
int cp_json_object_length(const char* data, size_t len, size_t* object_len);

#endif
