#pragma once

#include "case5_ngen_gemm.hpp"

#include "gemmstone/driver_info.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace case5_ngen {

struct LaunchBuffers {
    cl_mem a = nullptr;
    cl_mem b = nullptr;
    cl_mem c = nullptr;
    cl_mem tile_xor = nullptr;
    int64_t offset_a = 0;
    int64_t offset_b = 0;
    int64_t offset_c = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    uint32_t flags = 0;
    int tile_count = 0;
    int tile_cols = 0;
    int xor_period = 1;
    cl_mem blake3_out = nullptr;   // digest[spatial*8+w] (non-fused blake3 dump only)
    cl_mem blake3_beats = nullptr; // u32[spatial]: in-kernel beat flag per tile (fused jackpot)
    uint32_t blake3_key_words[8] = {};
    uint32_t blake3_bound_words[8] = {};
    cl_mem blake3_bound = nullptr; // optional legacy global bound buffer (unused by kernel)
    cl_mem found_flag = nullptr;
    cl_mem out_t_rows = nullptr;
    cl_mem out_t_cols = nullptr;
    int tr_base = 0;
    int tc_base = 0;
};

struct LaunchDims {
    size_t gws[2] = {1, 1};
    size_t lws[2] = {1, 1};
};

// Mirrors oneDNN gen_kernel_t::init_interface() plus Case5 tile_xor args.
void init_case5_gemm_interface(ngen::InterfaceHandler &iface, ngen::HW hw,
        const gemmstone::GEMMProblem &problem, const gemmstone::GEMMStrategy &strategy,
        const char *kernel_name);

LaunchDims compute_case5_launch_dims(const DriverInfo &info, int m, int n);

void case5_launch_dims_for_walk_order(const LaunchDims &dims, int subgroup_size,
        size_t wg_gws[2], size_t wg_lws[2]);

void apply_linear_order_launch_dims(const DriverInfo &info, LaunchDims &dims, int m, int n, int k);

cl_int bind_case5_kernel_args(cl_kernel kernel, const DriverInfo &info,
        const gemmstone::GEMMProblem &problem, const LaunchBuffers &bufs,
        const LaunchDims &dims, int *out_arg_count = nullptr);

} // namespace case5_ngen
