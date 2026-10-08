#include "cp_json_frame.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

namespace {
bool space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
bool digit(char c) { return c >= '0' && c <= '9'; }
bool boundary(char c) { return !c || space(c) || c == ',' || c == '}' || c == ']'; }

void append_utf8(std::string& out, uint32_t c)
{
    if(c < 0x80) out += (char)c;
    else if(c < 0x800){ out += (char)(0xc0 | (c >> 6)); out += (char)(0x80 | (c & 63)); }
    else if(c < 0x10000){
        out += (char)(0xe0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 63));
        out += (char)(0x80 | (c & 63));
    }else{
        out += (char)(0xf0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 63));
        out += (char)(0x80 | ((c >> 6) & 63)); out += (char)(0x80 | (c & 63));
    }
}

/* 1=complete, 0=needs more bytes, -1=invalid. Reads only the supplied span. */
struct Parser {
    const char* data;
    size_t len, pos = 0;
    void whitespace() { while(pos < len && space(data[pos])) ++pos; }
    int expect(char c) {
        if(pos == len) return 0;
        return data[pos++] == c ? 1 : -1;
    }
    int hex4(uint32_t& out) {
        out = 0;
        for(int i = 0; i < 4; ++i){
            if(pos == len) return 0;
            const unsigned char c = data[pos++];
            unsigned n;
            if(c >= '0' && c <= '9') n = c - '0';
            else if(c >= 'a' && c <= 'f') n = c - 'a' + 10;
            else if(c >= 'A' && c <= 'F') n = c - 'A' + 10;
            else return -1;
            out = (out << 4) | n;
        }
        return 1;
    }
    int string(std::string* out = nullptr) {
        int st = expect('"');
        if(st != 1) return st;
        while(pos < len){
            uint32_t c = (unsigned char)data[pos++];
            if(c == '"') return 1;
            if(c < 0x20) return -1;
            if(c == '\\'){
                if(pos == len) return 0;
                c = (unsigned char)data[pos++];
                switch(c){
                    case '"': case '\\': case '/': break;
                    case 'b': c = '\b'; break;
                    case 'f': c = '\f'; break;
                    case 'n': c = '\n'; break;
                    case 'r': c = '\r'; break;
                    case 't': c = '\t'; break;
                    case 'u': {
                        st = hex4(c); if(st != 1) return st;
                        if(c >= 0xd800 && c <= 0xdbff){
                            st = expect('\\'); if(st != 1) return st;
                            st = expect('u'); if(st != 1) return st;
                            uint32_t low;
                            st = hex4(low); if(st != 1) return st;
                            if(low < 0xdc00 || low > 0xdfff) return -1;
                            c = 0x10000 + ((c - 0xd800) << 10) + low - 0xdc00;
                        }else if(c >= 0xdc00 && c <= 0xdfff) return -1;
                        break;
                    }
                    default: return -1;
                }
            }else if(c >= 0x80){
                int extra;
                uint32_t minimum;
                if(c >= 0xc2 && c <= 0xdf){ extra = 1; minimum = 0x80; c &= 31; }
                else if(c >= 0xe0 && c <= 0xef){ extra = 2; minimum = 0x800; c &= 15; }
                else if(c >= 0xf0 && c <= 0xf4){ extra = 3; minimum = 0x10000; c &= 7; }
                else return -1;
                while(extra--){
                    if(pos == len) return 0;
                    const unsigned char next = data[pos++];
                    if((next & 0xc0) != 0x80) return -1;
                    c = (c << 6) | (next & 63);
                }
                if(c < minimum || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff)) return -1;
            }
            if(out) append_utf8(*out, c);
        }
        return 0;
    }
    int number() {
        if(pos < len && data[pos] == '-') ++pos;
        if(pos == len) return 0;
        if(data[pos] == '0') ++pos;
        else {
            if(data[pos] < '1' || data[pos] > '9') return -1;
            while(pos < len && digit(data[pos])) ++pos;
        }
        if(pos < len && data[pos] == '.'){
            ++pos;
            if(pos == len) return 0;
            if(!digit(data[pos])) return -1;
            while(pos < len && digit(data[pos])) ++pos;
        }
        if(pos < len && (data[pos] == 'e' || data[pos] == 'E')){
            ++pos;
            if(pos < len && (data[pos] == '+' || data[pos] == '-')) ++pos;
            if(pos == len) return 0;
            if(!digit(data[pos])) return -1;
            while(pos < len && digit(data[pos])) ++pos;
        }
        return pos == len || boundary(data[pos]) ? 1 : -1;
    }
    int value(unsigned depth = 0) {
        whitespace();
        if(pos == len) return 0;
        const char c = data[pos];
        if(c == '"') return string();
        if(c == '-' || digit(c)) return number();
        if(c == '{' || c == '['){
            if(depth >= 128) return -1;
            const bool object = c == '{';
            const char close = object ? '}' : ']';
            ++pos;
            whitespace();
            if(pos < len && data[pos] == close){ ++pos; return 1; }
            std::unordered_set<std::string> keys;
            for(;;){
                int st;
                if(object){
                    std::string key;
                    st = string(&key); if(st != 1) return st;
                    if(!keys.insert(key).second) return -1;
                    whitespace();
                    st = expect(':'); if(st != 1) return st;
                }
                st = value(depth + 1); if(st != 1) return st;
                whitespace();
                if(pos == len) return 0;
                if(data[pos] == close){ ++pos; return 1; }
                st = expect(','); if(st != 1) return st;
                whitespace();
            }
        }
        const char* literal = c == 't' ? "true" : c == 'f' ? "false" : c == 'n' ? "null" : nullptr;
        if(!literal) return -1;
        while(*literal){
            if(pos == len) return 0;
            if(data[pos++] != *literal++) return -1;
        }
        return pos == len || boundary(data[pos]) ? 1 : -1;
    }
};
}

int cp_json_value_length(const char* data, size_t len, size_t* value_len)
{
    if(!data || !value_len) return -1;
    Parser p{data, len};
    const int st = p.value();
    if(st == 1) *value_len = p.pos;
    return st;
}

int cp_json_object_length(const char* data, size_t len, size_t* object_len)
{
    if(!data || !len || data[0] != '{') return -1;
    return cp_json_value_length(data, len, object_len);
}

int cp_json_valid(const char* json)
{
    if(!json) return 0;
    Parser p{json, strlen(json)};
    p.whitespace();
    if(p.pos == p.len || json[p.pos] != '{' || p.value() != 1) return 0;
    p.whitespace();
    return p.pos == p.len;
}

int cp_json_decode_string(const char* value, std::string& out)
{
    out.clear();
    if(!value) return 0;
    Parser p{value, strlen(value)};
    if(p.string(&out) == 1 && boundary(value[p.pos])) return 1;
    out.clear();
    return 0;
}

const char* cp_json_find_member(const char* json, const char* key)
{
    if(!json || !key) return nullptr;
    Parser p{json, strlen(json)};
    p.whitespace();
    if(p.expect('{') != 1) return nullptr;
    p.whitespace();
    while(p.pos < p.len && json[p.pos] != '}'){
        std::string name;
        if(p.string(&name) != 1) return nullptr;
        p.whitespace();
        if(p.expect(':') != 1) return nullptr;
        p.whitespace();
        const size_t begin = p.pos;
        if(p.value() != 1) return nullptr;
        if(name == key) return json + begin;
        p.whitespace();
        if(p.pos < p.len && json[p.pos] == '}') break;
        if(p.expect(',') != 1) return nullptr;
        p.whitespace();
    }
    return nullptr;
}

int cp_json_number_value(const char* value, double* out)
{
    if(!value || !out || (*value != '-' && !digit(*value))) return 0;
    Parser p{value, strlen(value)};
    if(p.number() != 1) return 0;
    char* end;
    const double number = strtod(value, &end);
    if(end != value + p.pos || !std::isfinite(number)) return 0;
    *out = number;
    return 1;
}

int cp_json_uint64_value(const char* value, uint64_t* out)
{
    if(!value || !out || !digit(*value) || (*value == '0' && digit(value[1]))) return 0;
    uint64_t number = 0;
    do {
        const unsigned n = *value++ - '0';
        if(number > (UINT64_MAX - n) / 10) return 0;
        number = number * 10 + n;
    }while(digit(*value));
    if(!boundary(*value)) return 0;
    *out = number;
    return 1;
}
