#ifndef CP_CLI_H
#define CP_CLI_H

/* Strict command-line value parsing. Every helper throws std::invalid_argument on a
 * missing, malformed, out-of-range or too long value; cp_options_parse reports it as
 * "invalid command line <option>: <reason>" and exits with status 1 before any backend
 * or pool work starts. */

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

/* "--name=value" -> "--name"; any other argument unchanged. */
inline std::string cp_cli_option_name(const char* arg)
{
    const char* equals = strchr(arg, '=');
    return equals ? std::string(arg, (size_t)(equals - arg)) : std::string(arg);
}

/* "--name" or "--name=value"; a longer option with the same prefix does not match. */
inline bool cp_cli_option(const char* arg, const char* name)
{
    const size_t n = strlen(name);
    return !strncmp(arg, name, n) && (arg[n] == '\0' || arg[n] == '=');
}

/* Value of "--name=value" or of the next argument. A separate value may not start with
 * "--" (it would be the next option), but "-rig01" or "-1" is a value. capacity (when
 * nonzero) is the buffer the value is copied into, terminator included. */
inline const char* cp_cli_value(int argc, char** argv, int& index, size_t capacity = 0)
{
    const char* equals = strchr(argv[index], '=');
    const char* value = equals ? equals + 1 : (index + 1 < argc ? argv[++index] : nullptr);
    if(!value || !*value || (!equals && !strncmp(value, "--", 2)))
        throw std::invalid_argument("requires a value");
    if(capacity && strlen(value) >= capacity)
        throw std::invalid_argument("value is longer than " + std::to_string(capacity - 1) +
                                    " characters");
    return value;
}

/* cp_cli_value copied into dst (dst_size bytes, terminator included). */
inline void cp_cli_string(int argc, char** argv, int& index, char* dst, size_t dst_size)
{
    const char* value = cp_cli_value(argc, argv, index, dst_size);
    memcpy(dst, value, strlen(value) + 1);
}

/* Decimal digits only, minimum..maximum. */
inline int cp_cli_int(const char* value, int minimum = 0, int maximum = INT_MAX)
{
    const std::string range = maximum == INT_MAX
        ? "an integer of at least " + std::to_string(minimum)
        : "an integer from " + std::to_string(minimum) + " to " + std::to_string(maximum);
    if(!value || !*value) throw std::invalid_argument("expected " + range);
    unsigned int result = 0;
    for(const unsigned char* p = (const unsigned char*)value; *p; ++p){
        if(*p < '0' || *p > '9')
            throw std::invalid_argument("expected " + range + ", got \"" + value + "\"");
        const unsigned int digit = *p - '0';
        if(result > (unsigned int)maximum / 10 ||
           (result == (unsigned int)maximum / 10 && digit > (unsigned int)maximum % 10))
            throw std::invalid_argument("expected " + range + ", got " + value);
        result = result * 10 + digit;
    }
    if(result < (unsigned int)minimum)
        throw std::invalid_argument("expected " + range + ", got " + value);
    return (int)result;
}

/* A finite decimal number (JSON number syntax: no hex, inf/nan, leading '+' or spaces)
 * that is at least minimum. */
inline double cp_cli_real(const char* value, double minimum)
{
    const char* p = value;
    bool ok = p && *p;
    if(ok && *p == '-') ++p;
    if(ok && *p == '0') ++p;
    else if(ok && *p >= '1' && *p <= '9') while(*p >= '0' && *p <= '9') ++p;
    else ok = false;
    if(ok && *p == '.'){
        ++p;
        ok = *p >= '0' && *p <= '9';
        while(*p >= '0' && *p <= '9') ++p;
    }
    if(ok && (*p == 'e' || *p == 'E')){
        ++p;
        if(*p == '+' || *p == '-') ++p;
        ok = *p >= '0' && *p <= '9';
        while(*p >= '0' && *p <= '9') ++p;
    }
    ok = ok && !*p;
    const double result = ok ? strtod(value, nullptr) : 0.0;
    if(!ok || !std::isfinite(result) || result < minimum){
        char text[64];
        snprintf(text, sizeof(text), "expected a decimal number of at least %g", minimum);
        throw std::invalid_argument(std::string(text) + ", got \"" + (value ? value : "") + "\"");
    }
    return result;
}

/* "MxN" (returns 2) or, when macro_m/macro_n are given, "MxN/MxN" (returns 4); each
 * dimension at least 1. */
inline int cp_cli_dimensions(const char* value, int* m, int* n, int* macro_m = nullptr,
                             int* macro_n = nullptr)
{
    const std::string text(value);
    const size_t x = text.find('x'), slash = text.find('/');
    if(x == std::string::npos || (slash != std::string::npos && x > slash))
        throw std::invalid_argument("expected MxN, got \"" + text + "\"");
    *m = cp_cli_int(text.substr(0, x).c_str(), 1);
    *n = cp_cli_int(text.substr(x + 1, slash == std::string::npos ? slash : slash - x - 1).c_str(), 1);
    if(slash == std::string::npos) return 2;
    if(!macro_m || !macro_n) throw std::invalid_argument("expected MxN, got \"" + text + "\"");
    const size_t mx = text.find('x', slash + 1);
    if(mx == std::string::npos)
        throw std::invalid_argument("expected MxN/MxN, got \"" + text + "\"");
    *macro_m = cp_cli_int(text.substr(slash + 1, mx - slash - 1).c_str(), 1);
    *macro_n = cp_cli_int(text.substr(mx + 1).c_str(), 1);
    return 4;
}

/* "scheme://host:port" or "scheme://[IPv6]:port", optionally followed by "/..." (ignored).
 * Copies the host (without brackets) into host (shorter than capacity) and returns the
 * port, 1..65535. */
inline int cp_cli_pool_uri(const char* value, char* host, size_t capacity)
{
    const std::string expected = "expected scheme://host:port, got \"" + std::string(value) + "\"";
    const char* h = strstr(value, "://");
    if(!h) throw std::invalid_argument(expected);
    h += 3;
    const char* end = h + strcspn(h, "/");
    const char* host_begin = h;
    const char* host_end = nullptr;
    if(*h == '['){
        const char* close = (const char*)memchr(h, ']', (size_t)(end - h));
        if(!close || close[1] != ':') throw std::invalid_argument(expected);
        host_begin = h + 1;
        host_end = close;
    } else {
        for(const char* p = h; p < end; ++p)
            if(*p == ':') host_end = p;
        if(!host_end) throw std::invalid_argument(expected);
    }
    if(host_end == host_begin) throw std::invalid_argument(expected);
    const char* port_begin = host_end + (*host_end == ']' ? 2 : 1);
    const size_t length = (size_t)(host_end - host_begin);
    if(length >= capacity)
        throw std::invalid_argument("host name is longer than " + std::to_string(capacity - 1) +
                                    " characters");
    const int port = cp_cli_int(std::string(port_begin, end).c_str(), 1, 65535);
    memcpy(host, host_begin, length);
    host[length] = 0;
    return port;
}

/* Comma-separated device ids appended to devices[count..]; empty entries and more than
 * capacity ids in total are errors. */
inline void cp_cli_devices(const char* value, int* devices, int& count, int capacity)
{
    const std::string text(value);
    size_t start = 0;
    for(;;){
        const size_t end = text.find(',', start);
        if(count >= capacity)
            throw std::invalid_argument("more than " + std::to_string(capacity) + " devices");
        if(end == start || start == text.size())
            throw std::invalid_argument("empty device id in \"" + text + "\"");
        devices[count++] = cp_cli_int(text.substr(start, end - start).c_str());
        if(end == std::string::npos) break;
        start = end + 1;
    }
}

#endif /* CP_CLI_H */
