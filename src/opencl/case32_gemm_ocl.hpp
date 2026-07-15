#pragma once

#include "opencl_context.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum class Case32OclDpiMode {
    Auto,    // use cl_khr_integer_dot_product only if advertised
    Force,   // try -cl-ext=+cl_khr_integer_dot_product even if not advertised
    Asm,     // try AMD inline asm v_dot4c_i32_i8 (gfx1011/1012)
    Builtin, // try __builtin_amdgcn_sdot4 (maps to v_dot4c)
    Off,     // always scalar 4x mul
};

struct Case32GemmOcl {
    Case32GemmOcl() = default;
    ~Case32GemmOcl();

    void set_dpi_mode(Case32OclDpiMode mode) { dpi_mode_ = mode; }
    Case32OclDpiMode dpi_mode() const { return dpi_mode_; }
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

    // Enqueue kernel + clFinish (no C readback).
    void run_kernel();
    // Download C to host (for correctness checks).
    void read_c_host();
    // run_kernel() then read_c_host().
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
    Case32OclDpiMode dpi_mode_ = Case32OclDpiMode::Auto;
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
