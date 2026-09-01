/*******************************************************************************
 * Case5 nGEN config: half/bfloat stubs so gemmstone can compile without oneDNN.
 *******************************************************************************/
#ifndef NGEN_CONFIG_HPP
#define NGEN_CONFIG_HPP

#include <cstdint>
#include <cstring>

#ifndef NGEN_NAMESPACE
#define NGEN_NAMESPACE ngen
#endif

#ifndef NGEN_SAFE
#define NGEN_SAFE
#endif
#ifndef NGEN_SHORT_NAMES
#define NGEN_SHORT_NAMES
#endif
#define NGEN_NEO_INTERFACE
#define NGEN_WINDOWS_COMPAT
#ifndef NGEN_NO_OP_NAMES
#define NGEN_NO_OP_NAMES
#endif

namespace NGEN_NAMESPACE {

// Minimal IEEE-ish f16 wrapper (enough for Immediate / cast).
struct half {
    uint16_t raw = 0;
    half() = default;
    explicit half(float f) {
        // Truncating conversion is fine for compile-time paths; runtime f16 rare in Case5.
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        uint32_t sign = (u >> 16) & 0x8000u;
        int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x3FFu;
        if (exp <= 0) {
            raw = uint16_t(sign);
        } else if (exp >= 31) {
            raw = uint16_t(sign | 0x7C00u);
        } else {
            raw = uint16_t(sign | (uint32_t(exp) << 10) | mant);
        }
    }
    explicit operator float() const {
        uint32_t sign = uint32_t(raw & 0x8000u) << 16;
        uint32_t exp = (raw >> 10) & 0x1Fu;
        uint32_t mant = raw & 0x3FFu;
        uint32_t u;
        if (exp == 0) {
            u = sign;
        } else if (exp == 31) {
            u = sign | 0x7F800000u | (mant << 13);
        } else {
            u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
};

struct bfloat16 {
    uint16_t raw = 0;
    bfloat16() = default;
    explicit bfloat16(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        raw = uint16_t(u >> 16);
    }
    explicit operator float() const {
        uint32_t u = uint32_t(raw) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
};

} // namespace NGEN_NAMESPACE

#define NGEN_HALF_TYPE
#define NGEN_BFLOAT16_TYPE

#endif
