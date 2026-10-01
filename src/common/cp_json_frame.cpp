#include "cp_json_frame.h"

int cp_json_object_length(const char* data, size_t len, size_t* object_len)
{
    if(!data || !object_len || !len || data[0] != '{') return -1;
    char stack[128];
    size_t depth = 0;
    bool quoted = false;
    bool escaped = false;
    for(size_t i = 0; i < len; ++i){
        const char c = data[i];
        if(quoted){
            if(escaped) escaped = false;
            else if(c == '\\') escaped = true;
            else if(c == '"') quoted = false;
            continue;
        }
        if(c == '"') quoted = true;
        else if(c == '{' || c == '['){
            if(depth == sizeof(stack)) return -1;
            stack[depth++] = c;
        } else if(c == '}' || c == ']'){
            if(!depth || stack[depth - 1] != (c == '}' ? '{' : '[')) return -1;
            if(--depth == 0){
                *object_len = i + 1;
                return 1;
            }
        }
    }
    return 0;
}
