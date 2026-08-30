#pragma once

// Case 5: gemmstone int8 GEMM → OpenCL binary "case5_igemm".
// HW: Gen12LP (XeLP) or XeHPG (Core Ultra / MTL/ARL), selected at runtime.
// Strategy: oneDNN kernel.db select() (gen_nocopy contract) + Case5 XOR filters + fallbacks.
#include <cstdint>
#include <string>

#include "gemmstone/driver_info.hpp"
#include "gemmstone/strategy.hpp"
#include "ngen_core.hpp"

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace case5_ngen {

struct DriverInfo {
    int unrollM = 32;
    int unrollN = 16;
    int xorSubM = 16;
    int xorSubN = 16;
    int xorSubGridM = 1;
    int xorSubGridN = 1;
    int xorOwnedM = 32;
    int xorOwnedN = 16;
    int unrollK = 32;
    int wgM = 4;
    int wgN = 4;
    int wgK = 1;
    int subgroupSize = 8;
    int slm = 0;
    int wgExpand = 1;
    bool fixedWG = true;
    bool fusedEUs = false;
    bool isNMK = false;
    const char *hwName = "unknown";
    const char *strategyName = "unknown";
    std::string strategyNameOwned;
    std::string selectionLog; // catalog rejection reasons when a fallback strategy wins
    ngen::HW hw = ngen::HW::Unknown;
    int euCount = 128;
    gemmstone::GEMMStrategy strategy;
    gemmstone::CommonDriverInfo commonDriver;
};

// oneDNN-style leading-dimension pad for int8 (optional host padding).
inline int pad_ld_int8(int inner) {
    size_t stride = static_cast<size_t>(inner);
    if (stride > 32) {
        stride = (stride + 63) & ~size_t{63};
        if (stride % 256 == 0) {
            stride += 64;
        }
        return static_cast<int>(stride);
    }
    if (stride > 2) {
        return static_cast<int>((stride + 3) & ~size_t{3});
    }
    int p = 1;
    while (p < inner) {
        p <<= 1;
    }
    return p;
}

// Gen12LP/XeLP and XeHPG (Core Ultra MTL/ARL). Xe2/LNL not supported yet.
bool is_supported_device(cl_context ctx, cl_device_id device, std::string *err = nullptr);

struct BuildParams {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
};

// Build OpenCL kernel via gemmstone. Caller owns with clReleaseKernel.
// dims may be null (defaults to 4096³). Pass M/N/K/lda/ldb/ldc for catalog selection.
// fused_jackpot: Case 5.6 fused GEMM+XOR+BLAKE3+jackpot in one kernel (no tile_xor global I/O).
cl_kernel build_igemm_kernel(cl_context ctx, cl_device_id device, const BuildParams *dims,
                             DriverInfo *info, std::string *err = nullptr, bool xor_nop = false,
                             bool fused_jackpot = false);

} // namespace case5_ngen
