#pragma once

/* Runtime capabilities are limited to kernels compiled into this binary. */
struct Case33CpuFeatures {
    bool ssse3 = false;
    bool avx2 = false;
    bool dotprod = false;
    bool neon = false;
};

Case33CpuFeatures case33_detect_cpu_features();
bool case33_is_x86_build();
bool case33_is_aarch64_build();
