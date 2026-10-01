#ifndef CP_JSON_TEXT_HPP
#define CP_JSON_TEXT_HPP

#include <string>

/* Escape a string value before placing it in a pool JSON request. */
inline std::string cp_json_escape(const char* input)
{
    std::string out;
    if(!input) return out;
    static const char hex[] = "0123456789abcdef";
    for(const unsigned char* p = (const unsigned char*)input; *p; ++p){
        const unsigned char c = *p;
        if(c == '"' || c == '\\'){
            out += '\\';
            out += (char)c;
        } else if(c < 0x20){
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += (char)c;
        }
    }
    return out;
}

#endif
