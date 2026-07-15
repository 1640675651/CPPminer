#include "case33_gemm_ocl.hpp"

#include "case32_layout.hpp"
#include "case32_prepack.hpp"
#include "cp_config.h"
#include "cp_jackpot.hpp"

#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

namespace {

std::string directory_of_exe() {
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        return ".";
    }
    std::string s(path, path + n);
    const size_t slash = s.find_last_of("\\/");
    return slash == std::string::npos ? "." : s.substr(0, slash);
#else
    char path[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) {
        return ".";
    }
    path[n] = '\0';
    std::string s(path);
    const size_t slash = s.find_last_of('/');
    return slash == std::string::npos ? "." : s.substr(0, slash);
#endif
}

} // namespace

std::string cp_ocl_resolve_kernel_path() {
    const std::string base = directory_of_exe();
#ifdef _WIN32
    const std::string rel = base + "\\kernels\\case33_gemm_xor.cl";
#else
    const std::string rel = base + "/kernels/case33_gemm_xor.cl";
#endif
    return rel;
}

namespace {

int clamp_macro_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_MACRO_BATCH_MAX) {
        batch = CP_MACRO_BATCH_MAX;
    }
    return batch;
}

} // namespace

void Case33GemmOcl::set_macro_batch(int batch) {
    macro_batch_ = clamp_macro_batch(batch);
}

Case33GemmOcl::~Case33GemmOcl() {
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    if (a_buf_) {
        clReleaseMemObject(a_buf_);
        a_buf_ = nullptr;
    }
    if (b_buf_) {
        clReleaseMemObject(b_buf_);
        b_buf_ = nullptr;
    }
    if (dummy_buf_) {
        clReleaseMemObject(dummy_buf_);
        dummy_buf_ = nullptr;
    }
    if (a_key_buf_) {
        clReleaseMemObject(a_key_buf_);
        a_key_buf_ = nullptr;
    }
    if (bound_buf_) {
        clReleaseMemObject(bound_buf_);
        bound_buf_ = nullptr;
    }
    if (found_buf_) {
        clReleaseMemObject(found_buf_);
        found_buf_ = nullptr;
    }
    if (out_rows_buf_) {
        clReleaseMemObject(out_rows_buf_);
        out_rows_buf_ = nullptr;
    }
    if (out_cols_buf_) {
        clReleaseMemObject(out_cols_buf_);
        out_cols_buf_ = nullptr;
    }
}

bool Case33GemmOcl::build_kernel_(const char *kernel_cl_path) {
    auto try_build = [&](bool use_dot, bool force_ext, bool use_asm, bool use_builtin,
                         const char *label) -> bool {
        std::string build_opts = "-cl-std=CL1.2";
        build_opts += " -DMR=" + std::to_string(case32::kMR);
        build_opts += " -DNR=" + std::to_string(case32::kNR);
        build_opts += " -DCASE32_COALESCE=1";
        build_opts += " -DCASE32_WI_ROWMAJOR=1";
        if (use_asm) {
            build_opts += " -DCASE32_USE_ASM_DOT=1";
        } else if (use_builtin) {
            build_opts += " -DCASE32_USE_BUILTIN_SDOT4=1";
        } else if (use_dot) {
            if (force_ext) {
                build_opts += " -DCASE32_FORCE_DPI=1";
            } else {
                build_opts += " -Dcl_khr_integer_dot_product";
            }
        }
        if (!ocl_.build_program_from_file(kernel_cl_path, build_opts.c_str())) {
            if (use_dot && force_ext && !use_asm && !use_builtin) {
                const std::string build_opts2 =
                        "-cl-std=CL1.2 -cl-ext=+cl_khr_integer_dot_product "
                        "-DCASE32_FORCE_DPI=1 -DMR=" +
                        std::to_string(case32::kMR) + " -DNR=" +
                        std::to_string(case32::kNR) +
                        " -DCASE32_COALESCE=1 -DCASE32_WI_ROWMAJOR=1";
                if (ocl_.build_program_from_file(kernel_cl_path, build_opts2.c_str())) {
                    if (kernel_) {
                        clReleaseKernel(kernel_);
                        kernel_ = nullptr;
                    }
                    kernel_ = ocl_.create_kernel("case33_macro_gemm_xor");
                    if (kernel_) {
                        using_integer_dot_ = true;
                        using_asm_dot_ = false;
                        using_builtin_dot_ = false;
                        std::snprintf(dpi_status_, sizeof(dpi_status_), "%s (-cl-ext): OK",
                                      label);
                        return true;
                    }
                }
            }
            std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: BUILD FAILED", label);
            return false;
        }
        if (kernel_) {
            clReleaseKernel(kernel_);
            kernel_ = nullptr;
        }
        kernel_ = ocl_.create_kernel("case33_macro_gemm_xor");
        if (!kernel_) {
            std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: kernel create FAILED", label);
            return false;
        }
        using_integer_dot_ = use_dot && !use_asm && !use_builtin;
        using_asm_dot_ = use_asm;
        using_builtin_dot_ = use_builtin;
        std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: OK", label);
        return true;
    };

    bool built = false;
    if (dpi_mode_ == Case32OclDpiMode::Off) {
        built = try_build(false, false, false, false, "scalar (forced off)");
    } else if (dpi_mode_ == Case32OclDpiMode::Asm) {
        built = try_build(false, false, true, false, "asm v_dot4c_i32_i8");
        if (!built) {
            built = try_build(false, false, false, false, "scalar (asm failed)");
        }
    } else if (dpi_mode_ == Case32OclDpiMode::Builtin) {
        built = try_build(false, false, false, true, "builtin __builtin_amdgcn_sdot4");
        if (!built) {
            built = try_build(true, false, false, false, "KHR dot_acc_sat (builtin failed)");
        }
        if (!built) {
            built = try_build(false, false, false, false, "scalar (builtin failed)");
        }
    } else if (dpi_mode_ == Case32OclDpiMode::Force) {
        built = try_build(true, true, false, false, "force CASE32_FORCE_DPI");
        if (!built) {
            built = try_build(false, false, false, false, "scalar (force failed)");
        }
    } else if (ocl_.has_integer_dot_product) {
        built = try_build(true, false, false, false, "auto cl_khr_integer_dot_product");
        if (!built) {
            built = try_build(false, false, false, false, "scalar (auto DPI failed)");
        }
    } else {
        built = try_build(false, false, false, false, "scalar (no DPI ext)");
    }
    return built;
}

bool Case33GemmOcl::init_context(const char *kernel_cl_path, int device_index) {
    context_ready_ = false;
    available_ = false;
    if (!ocl_.init(device_index)) {
        return false;
    }
    if (!build_kernel_(kernel_cl_path)) {
        return false;
    }
    if (!ensure_jackpot_bufs_()) {
        return false;
    }
    device_name_ = ocl_.device_name;
    std::snprintf(backend_, sizeof(backend_), "OpenCL Case3.3 context ready");
    context_ready_ = true;
    return true;
}

bool Case33GemmOcl::prepare_job(int M, int N, int K, const int8_t *b_colmajor) {
    available_ = false;
    if (!context_ready_ || !kernel_ || !b_colmajor) {
        return false;
    }

    M_ = M;
    N_ = N;
    K_ = K;
    num_milestones_ = case32::kNumMilestones;
    milestone_k_ = K / num_milestones_;
    blocks_k_ = K / case32::kKR;
    blocks_per_milestone_ = milestone_k_ / case32::kKR;
    macro_rows_ = M / case32::kMacroM;
    macro_cols_ = N / case32::kMacroN;
    tile_cols_ = N / case32::kNR;
    tile_count_ = static_cast<size_t>(M / case32::kMR) * static_cast<size_t>(tile_cols_);
    macro_blocks_ = macro_cols_ * macro_rows_;

    if (M % case32::kMR != 0 || N % case32::kNR != 0 || K % case32::kKR != 0) {
        return false;
    }
    if (M % case32::kMacroM != 0 || N % case32::kMacroN != 0) {
        return false;
    }
    if (K % num_milestones_ != 0 || milestone_k_ % case32::kKR != 0) {
        return false;
    }

    case32::prepack_b_coalesced(b_colmajor, N_, K_, blocks_k_, macro_cols_, &b_pre_host_);

    if (b_buf_) {
        clReleaseMemObject(b_buf_);
        b_buf_ = nullptr;
    }
    if (a_buf_) {
        clReleaseMemObject(a_buf_);
        a_buf_ = nullptr;
    }

    b_buf_ = ocl_.alloc_buffer(b_pre_host_.size(), CL_MEM_READ_ONLY);
    a_buf_ = ocl_.alloc_buffer(
            static_cast<size_t>(M_ / case32::kMR) * static_cast<size_t>(blocks_k_) *
                    static_cast<size_t>(case32::kPanelA),
            CL_MEM_READ_ONLY);
    if (!dummy_buf_) {
        dummy_buf_ = ocl_.alloc_buffer(sizeof(uint32_t), CL_MEM_READ_WRITE);
    }

    if (!b_buf_ || !a_buf_ || !dummy_buf_) {
        return false;
    }
    if (!ocl_.write_buffer(b_buf_, b_pre_host_.data(), b_pre_host_.size())) {
        return false;
    }

    const char *dot_kind = "scalar";
    if (using_asm_dot_) {
        dot_kind = "asm v_dot4c";
    } else if (using_builtin_dot_) {
        dot_kind = "builtin sdot4";
    } else if (using_integer_dot_) {
        dot_kind = "dot_acc_sat";
    }
    std::snprintf(backend_, sizeof(backend_),
                  "OpenCL Case3.3 %dx%d macro batch=%d fused GEMM+XOR+jackpot, %dx%d KR=%d %s s8s8",
                  case32::kMacroM, case32::kMacroN, macro_batch_, case32::kMR, case32::kNR,
                  case32::kKR, dot_kind);
    available_ = true;
    return true;
}

bool Case33GemmOcl::prepare_attempt_a(const int8_t *a_rowmajor) {
    if (!available_ || !a_rowmajor || !a_buf_) {
        return false;
    }
    case32::prepack_a_coalesced(a_rowmajor, M_, K_, blocks_k_, macro_rows_, false, &a_pre_host_);
    return ocl_.write_buffer(a_buf_, a_pre_host_.data(), a_pre_host_.size());
}

bool Case33GemmOcl::ensure_jackpot_bufs_() {
    if (!a_key_buf_) {
        a_key_buf_ = ocl_.alloc_buffer(8 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    }
    if (!bound_buf_) {
        bound_buf_ = ocl_.alloc_buffer(8 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    }
    if (!found_buf_) {
        found_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_READ_WRITE);
    }
    if (!out_rows_buf_) {
        out_rows_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_WRITE_ONLY);
    }
    if (!out_cols_buf_) {
        out_cols_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_WRITE_ONLY);
    }
    return a_key_buf_ && bound_buf_ && found_buf_ && out_rows_buf_ && out_cols_buf_;
}

bool Case33GemmOcl::run_macro_batch_(int mb_begin, int batch_count) {
    if (batch_count < 1) {
        return false;
    }

    const int xor_after = 1;
    const int compact_xor = 0;
    const int fuse_jackpot = 1;
    const int tile_count_i = static_cast<int>(tile_count_);
    const size_t global = static_cast<size_t>(batch_count) * case32::kMacroWorkItems;
    const size_t local = case32::kMacroWorkItems;

    cl_mem tile_xor_dummy = dummy_buf_;

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &a_buf_);
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &b_buf_);
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &tile_xor_dummy);
    err |= clSetKernelArg(kernel_, 3, sizeof(int), &N_);
    err |= clSetKernelArg(kernel_, 4, sizeof(int), &blocks_k_);
    err |= clSetKernelArg(kernel_, 5, sizeof(int), &blocks_per_milestone_);
    err |= clSetKernelArg(kernel_, 6, sizeof(int), &num_milestones_);
    err |= clSetKernelArg(kernel_, 7, sizeof(int), &tile_count_i);
    err |= clSetKernelArg(kernel_, 8, sizeof(int), &macro_rows_);
    err |= clSetKernelArg(kernel_, 9, sizeof(int), &macro_cols_);
    err |= clSetKernelArg(kernel_, 10, sizeof(int), &xor_after);
    err |= clSetKernelArg(kernel_, 11, sizeof(int), &mb_begin);
    err |= clSetKernelArg(kernel_, 12, sizeof(int), &compact_xor);
    err |= clSetKernelArg(kernel_, 13, sizeof(cl_mem), &a_key_buf_);
    err |= clSetKernelArg(kernel_, 14, sizeof(cl_mem), &bound_buf_);
    err |= clSetKernelArg(kernel_, 15, sizeof(cl_mem), &found_buf_);
    err |= clSetKernelArg(kernel_, 16, sizeof(cl_mem), &out_rows_buf_);
    err |= clSetKernelArg(kernel_, 17, sizeof(cl_mem), &out_cols_buf_);
    err |= clSetKernelArg(kernel_, 18, sizeof(int), &fuse_jackpot);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl] clSetKernelArg failed\n");
        return false;
    }

    err = clEnqueueNDRangeKernel(ocl_.queue, kernel_, 1, nullptr, &global, &local, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl] clEnqueueNDRangeKernel failed: %s\n",
                     OpenClContext::error_string(err).c_str());
        return false;
    }
    return true;
}

bool Case33GemmOcl::scan_for_share(const uint32_t a_key8[8], const uint32_t bound[8],
                                   int *out_found, int *out_t_rows, int *out_t_cols,
                                   uint64_t *out_tiles_scanned,
                                   const std::function<bool()> &should_cancel,
                                   const std::function<void(uint64_t)> &on_progress,
                                   const int8_t *host_a_rowmajor,
                                   const int8_t *host_b_colmajor) {
    if (!available_ || !a_key8 || !bound) {
        return false;
    }
    if (!ensure_jackpot_bufs_()) {
        return false;
    }

    if (out_found) {
        *out_found = 0;
    }
    if (out_t_rows) {
        *out_t_rows = -1;
    }
    if (out_t_cols) {
        *out_t_cols = -1;
    }
    if (out_tiles_scanned) {
        *out_tiles_scanned = 0;
    }

    if (!ocl_.write_buffer(a_key_buf_, a_key8, 8 * sizeof(uint32_t)) ||
        !ocl_.write_buffer(bound_buf_, bound, 8 * sizeof(uint32_t))) {
        return false;
    }

    const int zero = 0;
    if (!ocl_.write_buffer(found_buf_, &zero, sizeof(int))) {
        return false;
    }

    uint64_t tiles_scanned = 0;
    int found = 0;
    const bool host_validate = host_a_rowmajor != nullptr && host_b_colmajor != nullptr;

    for (int mb0 = 0; mb0 < macro_blocks_ && !found; mb0 += macro_batch_) {
        if (should_cancel && should_cancel()) {
            return false;
        }
        int batch_count = macro_batch_;
        if (mb0 + batch_count > macro_blocks_) {
            batch_count = macro_blocks_ - mb0;
        }
        if (!run_macro_batch_(mb0, batch_count)) {
            return false;
        }
        clFinish(ocl_.queue);

        int batch_found = 0;
        if (!ocl_.read_buffer(found_buf_, &batch_found, sizeof(int))) {
            return false;
        }
        tiles_scanned += static_cast<uint64_t>(batch_count) * case32::kMacroWorkItems;
        if (out_tiles_scanned) {
            *out_tiles_scanned = tiles_scanned;
        }
        if (on_progress) {
            on_progress(tiles_scanned);
        }

        if (!batch_found) {
            continue;
        }

        for (;;) {
            int t_rows = -1;
            int t_cols = -1;
            if (!ocl_.read_buffer(out_rows_buf_, &t_rows, sizeof(int)) ||
                !ocl_.read_buffer(out_cols_buf_, &t_cols, sizeof(int))) {
                return false;
            }

            bool accept = true;
            if (host_validate) {
                uint32_t milestone_xor[case32::kNumMilestones] = {};
                case32::compute_tile_milestone_xor_naive(
                        host_a_rowmajor, host_b_colmajor, M_, N_, K_, t_rows, t_cols,
                        num_milestones_, milestone_k_, milestone_xor);
                accept = cp_jackpot::tile_beats_target(milestone_xor, num_milestones_, a_key8,
                                                       bound);
                if (!accept) {
                    std::fprintf(stderr,
                                 "[ocl] device jackpot false positive at t_rows=%d t_cols=%d, "
                                 "rescanning batch\n",
                                 t_rows, t_cols);
                }
            }

            if (accept) {
                found = 1;
                if (out_found) {
                    *out_found = 1;
                }
                if (out_t_rows) {
                    *out_t_rows = t_rows;
                }
                if (out_t_cols) {
                    *out_t_cols = t_cols;
                }
                break;
            }

            if (!ocl_.write_buffer(found_buf_, &zero, sizeof(int))) {
                return false;
            }
            if (!run_macro_batch_(mb0, batch_count)) {
                return false;
            }
            clFinish(ocl_.queue);

            batch_found = 0;
            if (!ocl_.read_buffer(found_buf_, &batch_found, sizeof(int))) {
                return false;
            }
            if (!batch_found) {
                break;
            }
        }
    }

    if (!found) {
        if (out_found) {
            *out_found = 0;
        }
    }

    return true;
}
