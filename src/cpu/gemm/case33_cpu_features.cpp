#include "case33_cpu_features.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

#if defined(_WIN32) && defined(_M_ARM64)
#include <windows.h>
#endif

namespace {

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
void cpuid_ex(int leaf, int subleaf, int out[4]) {
    __cpuidex(out, leaf, subleaf);
}
#elif defined(__i386__) || defined(__x86_64__)
void cpuid_ex(int leaf, int subleaf, int out[4]) {
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
}
#endif

} // namespace

bool case33_is_x86_build() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__i386__) || defined(__x86_64__)
    return true;
#else
    return false;
#endif
}

bool case33_is_aarch64_build() {
#if defined(_M_ARM64) || defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

Case33CpuFeatures case33_detect_cpu_features() {
    Case33CpuFeatures features;

#if defined(_M_X64) || defined(_M_IX86) || defined(__i386__) || defined(__x86_64__)
    int r[4] = {};
    cpuid_ex(0, 0, r);
    const int max_leaf = r[0];
    if (max_leaf >= 1) {
        cpuid_ex(1, 0, r);
        features.ssse3 = (r[2] & (1 << 9)) != 0;
        const bool osxsave = (r[2] & (1 << 27)) != 0;
        const bool avx = (r[2] & (1 << 28)) != 0;
        if (max_leaf >= 7 && osxsave && avx) {
#if defined(_MSC_VER)
            const unsigned long long xcr = _xgetbv(0);
#else
            unsigned int eax = 0, edx = 0;
            __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            const unsigned long long xcr =
                    (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
            if ((xcr & 0x6) == 0x6) {
                cpuid_ex(7, 0, r);
                features.avx2 = (r[1] & (1 << 5)) != 0;
            }
        }
    }
#elif defined(_M_ARM64) || defined(__aarch64__)
    /* Advanced SIMD is mandatory in the AArch64 architecture profile. */
    features.neon = true;
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
    features.dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    features.dotprod = sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size,
                                    nullptr, 0) == 0 && value != 0;
#elif defined(_WIN32) && defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
    features.dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#elif defined(__arm__)
#if defined(__linux__) && defined(HWCAP_NEON)
    features.neon = (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
#endif
#endif

    return features;
}
