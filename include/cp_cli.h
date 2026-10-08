#pragma once

/* Strict command-line value parsing. Every helper throws std::invalid_argument
 * on a missing, malformed, out-of-range or truncated value; main() reports it
 * and exits with status 1 before any backend or pool work starts. */

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

/* "--name" or "--name=value"; a longer option with the same prefix does not match. */
inline bool cp_cli_option(const char* arg, const char* name)
{
    const size_t n = strlen(name);
    return !strncmp(arg, name, n) && (arg[n] == '\0' || arg[n] == '=');
}

/* Value of "--name=value" or of the next argument. A separate value may not
 * start with '-' (it would be the next option); capacity bounds the length. */
inline const char* cp_cli_value(int argc, char** argv, int& index, size_t capacity = 0)
{
    const char* arg = argv[index];
    const char* equals = strchr(arg, '=');
    const std::string option(arg, equals ? (size_t)(equals - arg) : strlen(arg));
    const char* value = equals ? equals + 1 : (index + 1 < argc ? argv[++index] : nullptr);
    if(!value || !*value || (!equals && value[0] == '-'))
        throw std::invalid_argument(option + " requires a value");
    if(capacity && strlen(value) >= capacity)
        throw std::invalid_argument(option + " value is too long");
    return value;
}

/* Decimal digits only, minimum..maximum. */
inline int cp_cli_int(const char* value, int minimum = 0, int maximum = INT_MAX)
{
    unsigned int result = 0;
    if(!value || !*value) throw std::invalid_argument("empty integer value");
    for(const unsigned char* p = (const unsigned char*)value; *p; ++p){
        if(*p < '0' || *p > '9')
            throw std::invalid_argument("integer value is invalid or out of range");
        const unsigned int digit = *p - '0';
        if(result > (unsigned int)maximum / 10 ||
           (result == (unsigned int)maximum / 10 && digit > (unsigned int)maximum % 10))
            throw std::invalid_argument("integer value is out of range");
        result = result * 10 + digit;
    }
    if(result < (unsigned int)minimum)
        throw std::invalid_argument("integer value is out of range");
    return (int)result;
}

/* A finite decimal number (JSON number syntax: no hex, inf/nan, leading '+' or
 * spaces) that is at least minimum. */
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
    if(!ok || !std::isfinite(result) || result < minimum)
        throw std::invalid_argument("numeric value is invalid or out of range");
    return result;
}

/* "MxN" (returns 2) or, when macro_m/macro_n are given, "MxN/MxN" (returns 4). */
inline int cp_cli_dimensions(const char* value, int* m, int* n, int* macro_m = nullptr,
                             int* macro_n = nullptr)
{
    const std::string text(value);
    const size_t x = text.find('x'), slash = text.find('/');
    if(x == std::string::npos || (slash != std::string::npos && x > slash))
        throw std::invalid_argument("expected MxN dimensions");
    *m = cp_cli_int(text.substr(0, x).c_str(), 1);
    *n = cp_cli_int(text.substr(x + 1, slash == std::string::npos ? slash : slash - x - 1).c_str(), 1);
    if(slash == std::string::npos) return 2;
    if(!macro_m || !macro_n) throw std::invalid_argument("unexpected macro dimensions");
    const size_t mx = text.find('x', slash + 1);
    if(mx == std::string::npos) throw std::invalid_argument("expected macro MxN dimensions");
    *macro_m = cp_cli_int(text.substr(slash + 1, mx - slash - 1).c_str(), 1);
    *macro_n = cp_cli_int(text.substr(mx + 1).c_str(), 1);
    return 4;
}

/* Comma-separated device ids; empty entries and more than capacity ids are errors. */
inline void cp_cli_devices(const char* value, int* devices, int& count, int capacity)
{
    const std::string text(value);
    size_t start = 0;
    for(;;){
        const size_t end = text.find(',', start);
        if(count >= capacity) throw std::invalid_argument("too many devices");
        devices[count++] = cp_cli_int(text.substr(start, end - start).c_str());
        if(end == std::string::npos) break;
        start = end + 1;
    }
}
