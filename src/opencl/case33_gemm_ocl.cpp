#include "case33_gemm_ocl.hpp"

#include "case32_layout.hpp"
#include "case32_prepack.hpp"
#include "cp_config.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

const char *dot_backend_label(Case32OclDotBackend b, int issue_mode, bool cpm_int) {
    switch (b) {
    case Case32OclDotBackend::Sudot4:
        return "builtin __builtin_amdgcn_sudot4 (gfx11)";
    case Case32OclDotBackend::Sdot4:
        return "builtin __builtin_amdgcn_sdot4";
    case Case32OclDotBackend::AsmDot4c:
        return "asm v_dot4c_i32_i8";
    case Case32OclDotBackend::KhrDpi:
        return "cl_khr_integer_dot_product";
    case Case32OclDotBackend::KhrDpiForce:
        return "force cl_khr_integer_dot_product";
    case Case32OclDotBackend::Scalar:
        if (issue_mode == 1) {
            return cpm_int ? "broadcast int (cpm)" : "broadcast float (cpm)";
        }
        if (issue_mode == 2) {
            return "packed scalar";
        }
        return cpm_int ? "cpm int" : "cpm float";
    }
    return "unknown";
}

const char *dot_kind_short(Case32OclDotBackend b, int issue_mode, bool cpm_int) {
    switch (b) {
    case Case32OclDotBackend::Sudot4:
        return "builtin sudot4";
    case Case32OclDotBackend::Sdot4:
        return "builtin sdot4";
    case Case32OclDotBackend::AsmDot4c:
        return "asm v_dot4c";
    case Case32OclDotBackend::KhrDpi:
    case Case32OclDotBackend::KhrDpiForce:
        return "dot_acc_sat";
    case Case32OclDotBackend::Scalar:
        if (issue_mode == 2) {
            return "packed scalar";
        }
        return cpm_int ? "clblast cpm int" : "clblast cpm float";
    }
    return "clblast cpm";
}

/* Build ordered candidate list. issue_mode==1 (broadcast) is handled by caller. */
std::vector<Case32OclDotBackend> select_dot_backends(Case32OclDotPolicy policy, bool vendor_amd,
                                                     bool has_khr) {
    using B = Case32OclDotBackend;
    std::vector<B> out;

    auto push_unique = [&](B b) {
        for (B x : out) {
            if (x == b) {
                return;
            }
        }
        out.push_back(b);
    };
    auto finish_with_scalar = [&]() {
        push_unique(B::Scalar);
        return out;
    };

    switch (policy) {
    case Case32OclDotPolicy::Off:
        return finish_with_scalar();
    case Case32OclDotPolicy::ForceKhr:
        push_unique(B::KhrDpiForce);
        return finish_with_scalar();
    case Case32OclDotPolicy::PinSudot4:
        push_unique(B::Sudot4);
        return finish_with_scalar();
    case Case32OclDotPolicy::PinSdot4:
        push_unique(B::Sdot4);
        return finish_with_scalar();
    case Case32OclDotPolicy::PinAsm:
        push_unique(B::AsmDot4c);
        return finish_with_scalar();
    case Case32OclDotPolicy::PinKhr:
        if (has_khr) {
            push_unique(B::KhrDpi);
        }
        return finish_with_scalar();
    case Case32OclDotPolicy::Auto:
    default:
        /* AMD: sudot4 (RDNA3) → sdot4 (GFX9/RDNA2) → KHR if advertised → scalar.
         * Intel/NVIDIA/other: KHR if advertised → scalar.
         * Asm stays opt-in via PinAsm only. */
        if (vendor_amd) {
            push_unique(B::Sudot4);
            push_unique(B::Sdot4);
        }
        if (has_khr) {
            push_unique(B::KhrDpi);
        }
        return finish_with_scalar();
    }
}

} // namespace

void Case33GemmOcl::set_macro_batch(int batch) {
    macro_batch_ = clamp_macro_batch(batch);
}

void Case33GemmOcl::set_issue_mode(int mode) {
    if (mode < 0) {
        mode = 0;
    }
    if (mode > 2) {
        mode = 2;
    }
    issue_mode_ = mode;
}

void Case33GemmOcl::set_issue_broadcast(int on) {
    issue_mode_ = on ? 1 : 0;
}

void Case33GemmOcl::set_cpm_int(int on) {
    use_cpm_int_ = on != 0;
}

void Case33GemmOcl::set_use_lds(int on) {
    use_lds_ = on != 0;
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
    auto with_cl_std = [](std::string opts, const char *ver) {
        const std::string needle = "-cl-std=CL1.2";
        const size_t pos = opts.find(needle);
        const std::string repl = std::string("-cl-std=") + ver;
        if (pos != std::string::npos) {
            opts.replace(pos, needle.size(), repl);
        } else {
            opts = repl + " " + opts;
        }
        return opts;
    };

    auto try_build = [&](Case32OclDotBackend backend) -> bool {
        const bool use_sudot = backend == Case32OclDotBackend::Sudot4;
        const bool use_asm = backend == Case32OclDotBackend::AsmDot4c;
        const bool use_builtin = backend == Case32OclDotBackend::Sdot4;
        const bool use_dot = backend == Case32OclDotBackend::KhrDpi ||
                             backend == Case32OclDotBackend::KhrDpiForce;
        const bool force_ext = backend == Case32OclDotBackend::KhrDpiForce;
        const bool scalar = backend == Case32OclDotBackend::Scalar;
        const char *label = dot_backend_label(backend, issue_mode_, use_cpm_int_);

        std::string build_opts = "-cl-std=CL1.2";
        build_opts += " -DMR=" + std::to_string(case32::kMR);
        build_opts += " -DNR=" + std::to_string(case32::kNR);
        build_opts += " -DHASH_NR=" + std::to_string(case32::hash_tile_nr());
        build_opts += " -DKR=" + std::to_string(case32::kKR);
        build_opts += " -DMACRO_M=" + std::to_string(case32::kMacroM);
        build_opts += " -DMACRO_N=" + std::to_string(case32::kMacroN);
        build_opts += " -DR_RANK=" + std::to_string(R_RANK);
        build_opts += " -DPP_MAX_MILESTONES=" + std::to_string(case32::kNumMilestones);
        build_opts += " -DCASE32_COALESCE=1";
        build_opts += " -DCASE32_WI_ROWMAJOR=" +
                      std::to_string(case32::wi_row_major() ? 1 : 0);
        build_opts += use_lds_ ? " -DCASE32_USE_LDS=1" : " -DCASE32_USE_LDS=0";
        /* Scalar/cpm nest: never let the compiler auto-enable KHR DPI (case36 / beignet-fix). */
        if (scalar) {
            build_opts += " -DCASE32_NO_DPI=1";
            if (use_cpm_int_) {
                build_opts += " -DCASE32_CPM_INT=1";
            }
            if (issue_mode_ == 2) {
                build_opts += " -DCASE32_FORCE_PACKED=1";
            }
        } else if (issue_mode_ == 2) {
            build_opts += " -DCASE32_FORCE_PACKED=1";
        }
        if (use_sudot) {
            build_opts += " -DCASE32_USE_BUILTIN_SUDOT4=1";
        } else if (use_asm) {
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

        auto adopt_kernel = [&](const char *status_label) -> bool {
            if (kernel_) {
                clReleaseKernel(kernel_);
                kernel_ = nullptr;
            }
            kernel_ = ocl_.create_kernel("case33_macro_gemm_xor");
            if (!kernel_) {
                std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: kernel create FAILED",
                              label);
                return false;
            }
            adopted_backend_ = backend;
            using_integer_dot_ = use_dot;
            using_asm_dot_ = use_asm;
            using_builtin_dot_ = use_builtin || use_sudot;
            using_cpm_ = scalar && issue_mode_ != 2;
            std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: OK", status_label);
            return true;
        };

        if (use_dot) {
            /* Primary: CL1.2 + extension macro (NVIDIA, AMD, and other KHR DPI drivers). */
            if (ocl_.build_program_from_file(kernel_cl_path, build_opts.c_str(), true)) {
                return adopt_kernel(label);
            }
            if (force_ext) {
                std::string build_opts2 =
                        "-cl-std=CL1.2 -cl-ext=+cl_khr_integer_dot_product "
                        "-DCASE32_FORCE_DPI=1 -DMR=" +
                        std::to_string(case32::kMR) + " -DNR=" +
                        std::to_string(case32::kNR) + " -DKR=" +
                        std::to_string(case32::kKR) +
                        " -DMACRO_M=" + std::to_string(case32::kMacroM) +
                        " -DMACRO_N=" + std::to_string(case32::kMacroN) +
                        " -DHASH_NR=" + std::to_string(case32::hash_tile_nr()) +
                        " -DCASE32_COALESCE=1 -DCASE32_WI_ROWMAJOR=" +
                        std::to_string(case32::wi_row_major() ? 1 : 0) +
                        (use_lds_ ? " -DCASE32_USE_LDS=1" : " -DCASE32_USE_LDS=0");
                if (issue_mode_ == 2) {
                    build_opts2 += " -DCASE32_FORCE_PACKED=1";
                }
                if (ocl_.build_program_from_file(kernel_cl_path, build_opts2.c_str(), true)) {
                    return adopt_kernel(label);
                }
            }
            /* Intel NEO: CL1.2 advertises KHR DPI but dot_acc_sat(char4) needs CL3.0.
             * Kernel uses __opencl_c_integer_dot_product_input_4x8bit only (no packed API). */
            if (ocl_.has_integer_dot_product &&
                ocl_.vendor_name.find("Intel") != std::string::npos) {
                const std::string cl30_opts =
                        with_cl_std(build_opts, "CL3.0") + " -DCASE32_FORCE_DPI=1";
                if (ocl_.build_program_from_file(kernel_cl_path, cl30_opts.c_str(), true)) {
                    return adopt_kernel("Intel CL3.0 dot_acc_sat");
                }
            }
            std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: BUILD FAILED", label);
            return false;
        }

        if (!ocl_.safe_build_program_from_file(kernel_cl_path, build_opts.c_str())) {
            std::snprintf(dpi_status_, sizeof(dpi_status_), "%s: BUILD FAILED", label);
            return false;
        }
        return adopt_kernel(label);
    };

    const bool vendor_is_amd = ocl_.vendor_name.find("AMD") != std::string::npos ||
                               ocl_.vendor_name.find("Advanced Micro") != std::string::npos;

    std::vector<Case32OclDotBackend> candidates;
    if (issue_mode_ == 1) {
        /* --ocl-issue broadcast: force CLBlast cpm (beignet-fix scalar nest). */
        candidates = {Case32OclDotBackend::Scalar};
    } else {
        candidates = select_dot_backends(dot_policy_, vendor_is_amd, ocl_.has_integer_dot_product);
    }

    bool built = false;
    for (Case32OclDotBackend backend : candidates) {
        if (try_build(backend)) {
            built = true;
            break;
        }
    }

    if (built && kernel_) {
        cl_ulong local_b = 0;
        cl_ulong priv_b = 0;
        (void)clGetKernelWorkGroupInfo(kernel_, ocl_.device, CL_KERNEL_LOCAL_MEM_SIZE,
                                       sizeof(local_b), &local_b, nullptr);
        (void)clGetKernelWorkGroupInfo(kernel_, ocl_.device, CL_KERNEL_PRIVATE_MEM_SIZE,
                                       sizeof(priv_b), &priv_b, nullptr);
        std::printf("[ocl] kernel mem: local=%llu B/WG private=%llu B/WI\n",
                    (unsigned long long)local_b, (unsigned long long)priv_b);
        std::fflush(stdout);
    }
    return built;
}

bool Case33GemmOcl::init_context(const char *kernel_cl_path, int device_index,
                                 int platform_filter, bool gpu_prep) {
    context_ready_ = false;
    available_ = false;
    if (!ocl_.init(device_index, platform_filter)) {
        return false;
    }
    if (!build_kernel_(kernel_cl_path)) {
        return false;
    }
    if (!ensure_jackpot_bufs_()) {
        return false;
    }
    if (gpu_prep) {
        if (!prep_.init(&ocl_, cp_ocl_kernel_dir())) {
            std::fprintf(stderr, "[ocl] prep kernel init failed\n");
            return false;
        }
    } else {
        std::printf("[ocl] skipping GPU prep kernels (--cpu-gen)\n");
    }
    device_name_ = ocl_.device_name;
    platform_name_ = ocl_.platform_name;
    device_flat_index_ = ocl_.device_flat_index;
    discrete_gpu_ = ocl_.discrete_gpu;
    std::snprintf(backend_, sizeof(backend_), "OpenCL context ready");
    context_ready_ = true;
    return true;
}

bool Case33GemmOcl::setup_dims_(int M, int N, int K) {
    M_ = M;
    N_ = N;
    K_ = K;
    num_milestones_ = K / R_RANK;
    milestone_k_ = R_RANK;
    blocks_k_ = K / case32::kKR;
    blocks_per_milestone_ = 1;
    macro_rows_ = M / case32::kMacroM;
    macro_cols_ = N / case32::kMacroN;
    tile_cols_ = N / case32::hash_tile_nr();
    tile_count_ = static_cast<size_t>(M / case32::hash_tile_mr()) *
                  static_cast<size_t>(tile_cols_);
    macro_blocks_ = macro_cols_ * macro_rows_;

    if (M % case32::kMR != 0 || N % case32::kNR != 0 || K % case32::kKR != 0) {
        return false;
    }
    if (M % case32::kMacroM != 0 || N % case32::kMacroN != 0) {
        return false;
    }
    if (case32::kKR != R_RANK || K % R_RANK != 0 ||
        num_milestones_ != case32::kNumMilestones) {
        return false;
    }
    return true;
}

bool Case33GemmOcl::prepare_job(int M, int N, int K, const int8_t *b_colmajor) {
    available_ = false;
    if (!context_ready_ || !kernel_ || !b_colmajor) {
        return false;
    }

    if (!setup_dims_(M, N, K)) {
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

    const char *dot_kind = dot_kind_short(adopted_backend_, issue_mode_, use_cpm_int_);
    std::snprintf(backend_, sizeof(backend_),
                  "OpenCL %dx%d macro batch=%d fused GEMM+XOR+jackpot, hash tile %dx%d KR=%d %s s8s8",
                  case32::kMacroM, case32::kMacroN, macro_batch_, case32::kMR, case32::kNR,
                  case32::kKR, dot_kind);
    available_ = true;
    return true;
}

bool Case33GemmOcl::prepare_job_gpu(int M, int N, int K, const uint8_t b_noise_seed[32]) {
    available_ = false;
    if (!context_ready_ || !kernel_ || !b_noise_seed || !prep_.ready()) {
        return false;
    }
    if (!setup_dims_(M, N, K)) {
        return false;
    }
    if (!prep_.ensure_buffers(M, N, K)) {
        return false;
    }

    if (b_buf_) {
        clReleaseMemObject(b_buf_);
        b_buf_ = nullptr;
    }
    if (a_buf_) {
        clReleaseMemObject(a_buf_);
        a_buf_ = nullptr;
    }

    const size_t b_bytes = static_cast<size_t>(macro_cols_) * static_cast<size_t>(blocks_k_) *
                           static_cast<size_t>(case32::kMacroKbBlockB);
    const size_t a_bytes = static_cast<size_t>(macro_rows_) * static_cast<size_t>(blocks_k_) *
                           static_cast<size_t>(case32::kMacroKbBlockA);

    b_buf_ = ocl_.alloc_buffer(b_bytes, CL_MEM_READ_ONLY);
    a_buf_ = ocl_.alloc_buffer(a_bytes, CL_MEM_READ_ONLY);
    if (!dummy_buf_) {
        dummy_buf_ = ocl_.alloc_buffer(sizeof(uint32_t), CL_MEM_READ_WRITE);
    }
    if (!b_buf_ || !a_buf_ || !dummy_buf_) {
        return false;
    }

    if (!prep_.prepare_job_b(b_buf_, b_noise_seed, N_, K_, blocks_k_, macro_cols_)) {
        return false;
    }

    const char *dot_kind = dot_kind_short(adopted_backend_, issue_mode_, use_cpm_int_);
    std::snprintf(backend_, sizeof(backend_),
                  "OpenCL %dx%d macro batch=%d fused GEMM+XOR+jackpot, register tile %dx%d, hash tile %dx%d KR=%d %s s8s8 GPU-prep",
                  case32::kMacroM, case32::kMacroN, macro_batch_, case32::kMR, case32::kNR,
                  case32::hash_tile_mr(), case32::hash_tile_nr(), case32::kKR, dot_kind);
    available_ = true;
    return true;
}

bool Case33GemmOcl::prepare_attempt_gpu(const uint8_t *ab_seed, int ab_seed_len,
                                        const uint8_t job_key[32],
                                        const uint8_t b_noise_seed[32], int salted,
                                        uint8_t a_key_out[32]) {
    if (!available_ || !a_buf_ || !prep_.ready()) {
        return false;
    }
    return prep_.prepare_attempt_a(a_buf_, ab_seed, ab_seed_len, job_key, b_noise_seed, M_, K_,
                                   blocks_k_, macro_rows_, salted, a_key_out);
}

bool Case33GemmOcl::read_A_sig(int8_t *h_A_sig) {
    if (!h_A_sig || M_ <= 0 || K_ <= 0) {
        return false;
    }
    return prep_.read_A_sig(h_A_sig, static_cast<size_t>(M_) * static_cast<size_t>(K_));
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

    const int micro_m = case32::kHashPerMacroM;
    const int micro_n = case32::kHashPerMacroN;
    size_t max_wg = ocl_.max_work_group_size;
    size_t kernel_max_wg = 0;
    if (clGetKernelWorkGroupInfo(kernel_, ocl_.device, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(kernel_max_wg), &kernel_max_wg, nullptr) == CL_SUCCESS &&
        kernel_max_wg > 0 && kernel_max_wg < max_wg) {
        max_wg = kernel_max_wg;
    }

    int slice_m = micro_m;
    if (case32::kMacroWorkItems > max_wg && micro_n > 0) {
        slice_m = static_cast<int>(max_wg / static_cast<size_t>(micro_n));
        if (slice_m < 1) {
            slice_m = 1;
        }
        while (slice_m > 1 && (micro_m % slice_m) != 0) {
            --slice_m;
        }
    }

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

    for (int m0 = 0; m0 < micro_m; m0 += slice_m) {
        const int micro_m_begin = m0;
        const int micro_m_count = (m0 + slice_m <= micro_m) ? slice_m : (micro_m - m0);
        const size_t local =
                static_cast<size_t>(micro_m_count) * static_cast<size_t>(micro_n);
        const size_t global = static_cast<size_t>(batch_count) * local;

        err = CL_SUCCESS;
        err |= clSetKernelArg(kernel_, 19, sizeof(int), &micro_m_begin);
        err |= clSetKernelArg(kernel_, 20, sizeof(int), &micro_m_count);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[ocl] clSetKernelArg micro_m slice failed\n");
            return false;
        }

        err = clEnqueueNDRangeKernel(ocl_.queue, kernel_, 1, nullptr, &global, &local, 0,
                                     nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[ocl] clEnqueueNDRangeKernel failed: %s\n",
                         OpenClContext::error_string(err).c_str());
            return false;
        }
    }
    return true;
}

bool Case33GemmOcl::scan_for_share(const uint32_t a_key8[8], const uint32_t bound[8],
                                   int *out_found, int *out_t_rows, int *out_t_cols,
                                   uint64_t *out_tiles_scanned,
                                   const std::function<bool()> &should_cancel,
                                   const std::function<void(uint64_t)> &on_progress) {
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

        if (!ocl_.read_buffer(found_buf_, &found, sizeof(int))) {
            return false;
        }
        tiles_scanned += static_cast<uint64_t>(batch_count) * case32::hash_tiles_per_macro();
        if (out_tiles_scanned) {
            *out_tiles_scanned = tiles_scanned;
        }
        if (on_progress) {
            on_progress(tiles_scanned);
        }
    }

    if (found) {
        int t_rows = -1;
        int t_cols = -1;
        if (!ocl_.read_buffer(out_rows_buf_, &t_rows, sizeof(int)) ||
            !ocl_.read_buffer(out_cols_buf_, &t_cols, sizeof(int))) {
            return false;
        }
        if (out_found) {
            *out_found = 1;
        }
        if (out_t_rows) {
            *out_t_rows = t_rows;
        }
        if (out_t_cols) {
            *out_t_cols = t_cols;
        }
    } else if (out_found) {
        *out_found = 0;
    }

    return true;
}
