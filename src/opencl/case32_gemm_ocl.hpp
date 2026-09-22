#pragma once

#include "opencl_context.hpp"

#include <cstdint>
#include <string>
#include <vector>

/* Concrete GEMM 4-wide MAC implementation compiled into the OpenCL kernel. */
enum class Case32OclDotBackend {
    Sudot4,      // __builtin_amdgcn_sudot4 (gfx11 / RDNA3)
    Sdot4,       // __builtin_amdgcn_sdot4 (gfx9 / RDNA1-2)
    AsmDot4c,    // inline asm v_dot4c_i32_i8 (experimental / opt-in)
    KhrDpi,      // cl_khr_integer_dot_product when advertised
    KhrDpiForce, // force -cl-ext=+cl_khr_integer_dot_product
    Scalar,      // NO_DPI: broadcast cpm or packed scalar (via issue_mode)
};

/* How to choose backends. Pin* tries that backend then Scalar. */
enum class Case32OclDotPolicy {
    Auto,     // vendor-ordered accelerated cascade → Scalar
    ForceKhr, // KhrDpiForce → Scalar
    Off,      // Scalar only
    PinSudot4,
    PinSdot4,
    PinAsm,
    PinKhr,
};

struct Case32GemmOcl {
    Case32GemmOcl() = default;
    ~Case32GemmOcl();

    void set_dot_policy(Case32OclDotPolicy mode) { dot_policy_ = mode; }
    Case32OclDotPolicy dot_policy() const { return dot_policy_; }
    bool using_integer_dot() const { return using_integer_dot_; }
    bool using_asm_dot() const { return using_asm_dot_; }
    bool using_builtin_dot() const { return using_builtin_dot_; }
    bool using_lds() const { return using_lds_; }
    bool using_coalesce() const { return using_coalesce_; }
    bool using_wi_rowmajor() const { return using_wi_rowmajor_; }
    bool device_reports_integer_dot() const { return device_reports_integer_dot_; }

    bool init(OpenClContext *ocl, int M, int N, int K, const int8_t *a, const int8_t *b,
              const char *kernel_cl_path, int device_index = -1);
    bool available() const { return available_; }

    void run_kernel();
    void read_c_host();
    void run();
    const int32_t *c_host() const { return c_host_.data(); }
    const char *backend() const { return backend_; }
    const char *device_name() const { return device_name_.c_str(); }
    const char *dpi_status() const { return dpi_status_; }

private:
    bool available_ = false;
    OpenClContext *ocl_ = nullptr;
    bool owns_ocl_ = false;
    OpenClContext ocl_owned_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
    int blocks_k_ = 0;
    int macro_rows_ = 0;
    int macro_cols_ = 0;
    int tile_cols_ = 0;
    Case32OclDotPolicy dot_policy_ = Case32OclDotPolicy::Auto;
    bool device_reports_integer_dot_ = false;
    bool using_integer_dot_ = false;
    bool using_asm_dot_ = false;
    bool using_builtin_dot_ = false;
    bool using_lds_ = false;
    bool using_coalesce_ = false;
    bool using_wi_rowmajor_ = false;

    cl_kernel kernel_ = nullptr;
    cl_mem a_buf_ = nullptr;
    cl_mem b_buf_ = nullptr;
    cl_mem c_buf_ = nullptr;
    cl_mem b_comp_buf_ = nullptr;

    std::vector<int8_t> a_pre_host_;
    std::vector<int8_t> b_pre_host_;
    std::vector<int32_t> b_comp_host_;
    std::vector<int32_t> c_host_;

    std::string device_name_;
    char backend_[192] = {};
    char dpi_status_[128] = {};
};
