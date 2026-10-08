#include "case33_ocl_prep.hpp"

#include "case32_layout.hpp"
#include "cp_config.h"
#include "cp_noise.h"
#include "cp_util.h"

#include <cstdio>
#include <cstdlib>
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

/* CP_OCL_PREP_TIMING=1: per-stage A prep timings (adds queue syncs). */
bool prep_timing_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_PREP_TIMING");
        return v && v[0] == '1';
    }();
    return on;
}

/* CP_OCL_PREP_WG=0 keeps the per-row noise kernel. */
bool prep_wg_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_PREP_WG");
        return !(v && v[0] == '0');
    }();
    return on;
}

/* CP_OCL_FPA_WG=0 keeps the 8-lane fused A prepack (one work group per 8-row stripe). */
bool fpa_wg_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_FPA_WG");
        return !(v && v[0] == '0');
    }();
    return on;
}

/* CP_OCL_PREP_CHECK=1: compare the work-group noise path with the per-row kernel. */
bool prep_check_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_PREP_CHECK");
        return v && v[0] == '1';
    }();
    return on;
}

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

uint64_t pearl_seed_to_u64(const uint8_t *seed, int seed_len) {
    uint64_t s = 0;
    for (int i = 0; i < seed_len; i++) {
        s ^= static_cast<uint64_t>(seed[i]) << ((i & 7) * 8);
    }
    return s;
}

} // namespace

std::string cp_ocl_kernel_dir() {
    const std::string base = directory_of_exe();
#ifdef _WIN32
    return base + "\\kernels";
#else
    return base + "/kernels";
#endif
}

Case33OclPrep::~Case33OclPrep() {
    release_kernels_();
    if (d_A_sig_) {
        clReleaseMemObject(d_A_sig_);
        d_A_sig_ = nullptr;
    }
    if (d_job_key_) {
        clReleaseMemObject(d_job_key_);
        d_job_key_ = nullptr;
    }
    if (d_noise_seed_) {
        clReleaseMemObject(d_noise_seed_);
        d_noise_seed_ = nullptr;
    }
    if (d_pairs_) {
        clReleaseMemObject(d_pairs_);
        d_pairs_ = nullptr;
    }
    if (d_merkle_roots_) {
        clReleaseMemObject(d_merkle_roots_);
        d_merkle_roots_ = nullptr;
    }
    if (d_a_subroots_) {
        clReleaseMemObject(d_a_subroots_);
        d_a_subroots_ = nullptr;
    }
    if (d_noisy_scratch_) {
        clReleaseMemObject(d_noisy_scratch_);
        d_noisy_scratch_ = nullptr;
    }
    if (d_el_rows_) {
        clReleaseMemObject(d_el_rows_);
        d_el_rows_ = nullptr;
        el_rows_cap_ = 0;
    }
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    ready_ = false;
}

bool Case33OclPrep::build_program_(const std::string &kernel_dir) {
    const auto read_cl = [&](const char *name) -> std::string {
#ifdef _WIN32
        const std::string path = kernel_dir + "\\" + name;
#else
        const std::string path = kernel_dir + "/" + name;
#endif
        return read_text_file(path.c_str());
    };

    const std::string src =
            read_cl("cp_ocl_blake3.cl") + read_cl("cp_ocl_merkle.cl") + read_cl("cp_ocl_prep.cl");
    if (src.empty()) {
        std::fprintf(stderr, "[ocl-prep] failed to read prep kernel sources from %s\n",
                     kernel_dir.c_str());
        return false;
    }

    std::string opts = "-cl-std=CL1.2";
    opts += " -DMR=" + std::to_string(case32::kMR);
    opts += " -DNR=" + std::to_string(case32::kNR);
    opts += " -DKR=" + std::to_string(case32::kKR);
    opts += " -DMACRO_M=" + std::to_string(case32::kMacroM);
    opts += " -DMACRO_N=" + std::to_string(case32::kMacroN);
    opts += " -DR_RANK=" + std::to_string(R_RANK);
    {
        /* ocl_fused_prepack_a_wg stages 1/FPA_SPLIT of a block's rows; split until they fit
           the work group's local memory (AMD's Windows driver allows 32 KB, ROCm 64 KB). */
        cl_ulong lds = 0;
        clGetDeviceInfo(ocl_->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lds), &lds, nullptr);
        auto need = [](int rows) {
            return static_cast<cl_ulong>(rows) * (case32::kKR / 4 + 1 + R_RANK / 4 + 1) * 4 +
                   2 * static_cast<cl_ulong>(case32::kKR) + 1024;
        };
        fpa_split_ = 1;
        while (need(case32::kMacroM / fpa_split_) > lds && case32::kMacroM / fpa_split_ > 32) {
            fpa_split_ *= 2;
        }
        opts += " -DFPA_SPLIT=" + std::to_string(fpa_split_);
    }

    cl_int err = CL_SUCCESS;
    const char *srcs[] = {src.c_str()};
    const size_t lens[] = {src.size()};
    program_ = clCreateProgramWithSource(ocl_->context, 1, srcs, lens, &err);
    if (!program_ || err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] clCreateProgramWithSource failed\n");
        return false;
    }

    err = clBuildProgram(program_, 1, &ocl_->device, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, ocl_->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(std::max(log_size, size_t{1}));
        clGetProgramBuildInfo(program_, ocl_->device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(),
                              nullptr);
        std::fprintf(stderr, "[ocl-prep] build log:\n%s\n", log.data());
        return false;
    }
    return true;
}

bool Case33OclPrep::create_kernels_() {
    auto mk = [&](const char *name) -> cl_kernel {
        cl_int err = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program_, name, &err);
        if (!k || err != CL_SUCCESS) {
            std::fprintf(stderr, "[ocl-prep] kernel %s not found\n", name);
        }
        return k;
    };

    k_gen_random_ = mk("ocl_gen_random_matrix");
    k_gen_random16_ = mk("ocl_gen_random_matrix16");
    k_build_pairs_ = mk("ocl_build_perm_pairs");
    k_keyed_chunk_roots_ = mk("ocl_keyed_chunk_roots");
    k_compute_blake_mt_ = mk("ocl_compute_blake_mt");
    k_reduce_roots_ = mk("ocl_reduce_roots");
    k_fused_prepack_a_ = fused_prepack_ ? mk("ocl_fused_prepack_a") : nullptr;
    k_fused_prepack_a_wg_ = fused_prepack_ ? mk("ocl_fused_prepack_a_wg") : nullptr;
    if (k_fused_prepack_a_wg_) {
        /* the kernel needs exactly 256 work-items per group */
        size_t max_wg = 0;
        clGetKernelWorkGroupInfo(k_fused_prepack_a_wg_, ocl_->device, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(max_wg), &max_wg, nullptr);
        if (max_wg < 256) {
            std::fprintf(stderr, "[ocl-prep] fused_prepack_a_wg allows %zu work-items; using the "
                                 "8-lane kernel\n", max_wg);
            clReleaseKernel(k_fused_prepack_a_wg_);
            k_fused_prepack_a_wg_ = nullptr;
        }
    }
    k_fused_prepack_b_ = fused_prepack_ ? mk("ocl_fused_prepack_b") : nullptr;
    k_noisy_rowmajor_ = mk("ocl_noisy_matrix_rowmajor");
    k_uniform_rows_ = mk("ocl_uniform_rows");
    k_noisy_rowmajor_wg_ = mk("ocl_noisy_matrix_rowmajor_wg");
    k_noisy_a_colmajor_ = mk("ocl_noisy_matrix_a_colmajor");
    k_noisy_colmajor_ = mk("ocl_noisy_matrix_colmajor");
    k_noisy_b_rowmajor_ = mk("ocl_noisy_matrix_b_rowmajor");
    k_pad_row_tails_ = mk("ocl_pad_rowmajor_row_tails");
    k_pad_col_tails_ = mk("ocl_pad_colmajor_col_tails");
    k_test_random_hash_ = mk("ocl_test_get_random_hash");

    const bool core = k_gen_random_ && k_build_pairs_ && k_keyed_chunk_roots_ &&
                      k_compute_blake_mt_ && k_reduce_roots_ && k_noisy_rowmajor_ &&
                      k_noisy_a_colmajor_ && k_noisy_colmajor_ && k_noisy_b_rowmajor_ &&
                      k_pad_row_tails_ && k_pad_col_tails_;
    if (!core) {
        return false;
    }
    if (fused_prepack_ && (!k_fused_prepack_a_ || !k_fused_prepack_b_)) {
        return false;
    }
    return true;
}

void Case33OclPrep::release_kernels_() {
    if (k_gen_random_) {
        clReleaseKernel(k_gen_random_);
        k_gen_random_ = nullptr;
    }
    if (k_gen_random16_) {
        clReleaseKernel(k_gen_random16_);
        k_gen_random16_ = nullptr;
    }
    if (k_build_pairs_) {
        clReleaseKernel(k_build_pairs_);
        k_build_pairs_ = nullptr;
    }
    if (k_keyed_chunk_roots_) {
        clReleaseKernel(k_keyed_chunk_roots_);
        k_keyed_chunk_roots_ = nullptr;
    }
    if (k_compute_blake_mt_) {
        clReleaseKernel(k_compute_blake_mt_);
        k_compute_blake_mt_ = nullptr;
    }
    if (k_reduce_roots_) {
        clReleaseKernel(k_reduce_roots_);
        k_reduce_roots_ = nullptr;
    }
    if (k_fused_prepack_a_) {
        clReleaseKernel(k_fused_prepack_a_);
        k_fused_prepack_a_ = nullptr;
    }
    if (k_fused_prepack_a_wg_) {
        clReleaseKernel(k_fused_prepack_a_wg_);
        k_fused_prepack_a_wg_ = nullptr;
    }
    if (k_fused_prepack_b_) {
        clReleaseKernel(k_fused_prepack_b_);
        k_fused_prepack_b_ = nullptr;
    }
    if (k_noisy_rowmajor_) {
        clReleaseKernel(k_noisy_rowmajor_);
        k_noisy_rowmajor_ = nullptr;
    }
    if (k_uniform_rows_) {
        clReleaseKernel(k_uniform_rows_);
        k_uniform_rows_ = nullptr;
    }
    if (k_noisy_rowmajor_wg_) {
        clReleaseKernel(k_noisy_rowmajor_wg_);
        k_noisy_rowmajor_wg_ = nullptr;
    }
    if (k_noisy_a_colmajor_) {
        clReleaseKernel(k_noisy_a_colmajor_);
        k_noisy_a_colmajor_ = nullptr;
    }
    if (k_noisy_colmajor_) {
        clReleaseKernel(k_noisy_colmajor_);
        k_noisy_colmajor_ = nullptr;
    }
    if (k_noisy_b_rowmajor_) {
        clReleaseKernel(k_noisy_b_rowmajor_);
        k_noisy_b_rowmajor_ = nullptr;
    }
    if (k_pad_row_tails_) {
        clReleaseKernel(k_pad_row_tails_);
        k_pad_row_tails_ = nullptr;
    }
    if (k_pad_col_tails_) {
        clReleaseKernel(k_pad_col_tails_);
        k_pad_col_tails_ = nullptr;
    }
    if (k_test_random_hash_) {
        clReleaseKernel(k_test_random_hash_);
        k_test_random_hash_ = nullptr;
    }
}

bool Case33OclPrep::init(OpenClContext *ocl, const std::string &kernel_dir, bool fused_prepack) {
    ready_ = false;
    fused_prepack_ = fused_prepack;
    ocl_ = ocl;
    if (!ocl_ || !ocl_->context) {
        return false;
    }
    if (!build_program_(kernel_dir)) {
        return false;
    }
    if (!create_kernels_()) {
        return false;
    }
    if (!d_job_key_) {
        d_job_key_ = ocl_->alloc_buffer(32, CL_MEM_READ_ONLY);
    }
    if (!d_noise_seed_) {
        d_noise_seed_ = ocl_->alloc_buffer(32, CL_MEM_READ_ONLY);
    }
    if (!d_pairs_) {
        d_pairs_ = ocl_->alloc_buffer(static_cast<size_t>(K_DIM) * 2u * sizeof(uint32_t),
                                      CL_MEM_READ_ONLY);
    }
    if (!d_job_key_ || !d_noise_seed_ || !d_pairs_) {
        return false;
    }
    ready_ = true;
    return true;
}

bool Case33OclPrep::ensure_buffers(int m, int n, int k) {
    if (!ready_) {
        return false;
    }
    const size_t szA = static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t szB = static_cast<size_t>(n) * static_cast<size_t>(k);
    const size_t raw_max = szA > szB ? szA : szB;
    const size_t pad_max = (raw_max + 1023) / 1024 * 1024;
    const size_t chunks_max = pad_max / 1024;
    size_t merkle_need = ((chunks_max + 255) / 256) * 32;
    if (merkle_need < 32) {
        merkle_need = 32;
    }

    if (m > m_cap_ || !d_A_sig_) {
        if (d_A_sig_) {
            clReleaseMemObject(d_A_sig_);
        }
        d_A_sig_ = ocl_->alloc_buffer(szA, CL_MEM_READ_WRITE);
        m_cap_ = m;
    }
    if (merkle_need > merkle_cap_) {
        if (d_merkle_roots_) {
            clReleaseMemObject(d_merkle_roots_);
        }
        if (d_a_subroots_) {
            clReleaseMemObject(d_a_subroots_);
        }
        d_merkle_roots_ = ocl_->alloc_buffer(merkle_need, CL_MEM_READ_WRITE);
        d_a_subroots_ = ocl_->alloc_buffer(merkle_need, CL_MEM_READ_WRITE);
        merkle_cap_ = merkle_need;
        a_witness_valid_ = false;
    }
    n_cap_ = n;
    return d_A_sig_ && d_merkle_roots_ && d_a_subroots_;
}

bool Case33OclPrep::ensure_noisy_scratch(size_t bytes) {
    if (!ready_ || bytes == 0) {
        return false;
    }
    if (bytes <= noisy_scratch_cap_ && d_noisy_scratch_) {
        return true;
    }
    if (d_noisy_scratch_) {
        clReleaseMemObject(d_noisy_scratch_);
        d_noisy_scratch_ = nullptr;
    }
    d_noisy_scratch_ = ocl_->alloc_buffer(bytes, CL_MEM_READ_WRITE);
    noisy_scratch_cap_ = d_noisy_scratch_ ? bytes : 0;
    return d_noisy_scratch_ != nullptr;
}

bool Case33OclPrep::gen_random_a_(uint64_t rng_seed, int total) {
    const int matrix_tag = 0;
    const cl_ulong rng_ul = static_cast<cl_ulong>(rng_seed);
    const bool wide = k_gen_random16_ && (total % 16) == 0;
    cl_kernel k = wide ? k_gen_random16_ : k_gen_random_;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_ulong), &rng_ul);
    err |= clSetKernelArg(k, 1, sizeof(int), &matrix_tag);
    err |= clSetKernelArg(k, 2, sizeof(int), &total);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &d_A_sig_);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t items = wide ? static_cast<size_t>(total / 16) : static_cast<size_t>(total);
    const size_t l = 256;
    const size_t g = (items + l - 1) / l * l;
    if (clEnqueueNDRangeKernel(ocl_->queue, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)
            != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::build_perm_pairs_(int is_b, int K) {
    if (!ready_ || K <= 0) {
        return false;
    }
    const int rank = R_RANK;
    const int pair_groups = (K + 7) / 8;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_build_pairs_, 0, sizeof(int), &is_b);
    err |= clSetKernelArg(k_build_pairs_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_build_pairs_, 2, sizeof(int), &K);
    err |= clSetKernelArg(k_build_pairs_, 3, sizeof(int), &rank);
    err |= clSetKernelArg(k_build_pairs_, 4, sizeof(cl_mem), &d_pairs_);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = static_cast<size_t>(pair_groups);
    err = clEnqueueNDRangeKernel(ocl_->queue, k_build_pairs_, 1, nullptr, &g, nullptr, 0, nullptr,
                                 nullptr);
    return err == CL_SUCCESS;
}

bool Case33OclPrep::noisy_matrix_rowmajor_(cl_mem out, cl_mem signal, int rows, int K, int out_lda,
                                           int is_b, int has_signal) {
    if (!ready_ || !out || rows <= 0 || K <= 0 || out_lda < K) {
        return false;
    }
    if (has_signal && !signal) {
        return false;
    }
    if (!build_perm_pairs_(is_b, K)) {
        return false;
    }

    const int rank = R_RANK;
    cl_mem sig_arg = signal;
    if (noisy_matrix_rowmajor_wg_(out, signal, rows, K, out_lda, is_b, has_signal)) {
        return true;
    }
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_rowmajor_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_noisy_rowmajor_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_noisy_rowmajor_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_rowmajor_, 3, sizeof(int), &rows);
    err |= clSetKernelArg(k_noisy_rowmajor_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_rowmajor_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_noisy_rowmajor_, 6, sizeof(int), &is_b);
    err |= clSetKernelArg(k_noisy_rowmajor_, 7, sizeof(int), &has_signal);
    err |= clSetKernelArg(k_noisy_rowmajor_, 8, sizeof(cl_mem), &sig_arg);
    err |= clSetKernelArg(k_noisy_rowmajor_, 9, sizeof(int), &out_lda);
    if (err != CL_SUCCESS) {
        return false;
    }

    const size_t g = static_cast<size_t>((rows + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_noisy_rowmajor_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] noisy_matrix_rowmajor launch failed\n");
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::ensure_el_rows_(size_t el_bytes) {
    if (el_bytes <= el_rows_cap_ && d_el_rows_) {
        return true;
    }
    if (d_el_rows_) {
        clReleaseMemObject(d_el_rows_);
    }
    d_el_rows_ = ocl_->alloc_buffer(el_bytes, CL_MEM_READ_WRITE);
    el_rows_cap_ = d_el_rows_ ? el_bytes : 0;
    return d_el_rows_ != nullptr;
}

/* d_el_rows_[row * R_RANK ...] = the row's uniform noise values for d_noise_seed_. */
bool Case33OclPrep::uniform_rows_(int rows, int is_b) {
    if (!k_uniform_rows_) {
        return false;
    }
    int rank = R_RANK;
    if (!ensure_el_rows_(static_cast<size_t>(rows) * static_cast<size_t>(rank))) {
        return false;
    }
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_uniform_rows_, 0, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_uniform_rows_, 1, sizeof(int), &rows);
    err |= clSetKernelArg(k_uniform_rows_, 2, sizeof(int), &rank);
    err |= clSetKernelArg(k_uniform_rows_, 3, sizeof(int), &is_b);
    err |= clSetKernelArg(k_uniform_rows_, 4, sizeof(cl_mem), &d_el_rows_);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t lu = 64;
    const size_t gu = (static_cast<size_t>(rows) + lu - 1) / lu * lu;
    return clEnqueueNDRangeKernel(ocl_->queue, k_uniform_rows_, 1, nullptr, &gu, &lu, 0, nullptr,
                                  nullptr) == CL_SUCCESS;
}

bool Case33OclPrep::noisy_matrix_rowmajor_wg_(cl_mem out, cl_mem signal, int rows, int K,
                                              int out_lda, int is_b, int has_signal) {
    if (!k_uniform_rows_ || !k_noisy_rowmajor_wg_ || !prep_wg_enabled()) {
        return false;
    }
    const int rank = R_RANK;
    if (!uniform_rows_(rows, is_b)) {
        return false;
    }

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 1, sizeof(cl_mem), &d_el_rows_);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 3, sizeof(int), &rows);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 6, sizeof(int), &has_signal);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 7, sizeof(cl_mem), &signal);
    err |= clSetKernelArg(k_noisy_rowmajor_wg_, 8, sizeof(int), &out_lda);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t lw = 256;
    const size_t gw = static_cast<size_t>(rows) * lw;
    if (clEnqueueNDRangeKernel(ocl_->queue, k_noisy_rowmajor_wg_, 1, nullptr, &gw, &lw, 0,
                               nullptr, nullptr) != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] noisy_matrix_rowmajor_wg launch failed; using row path\n");
        return false;
    }
    clFinish(ocl_->queue);
    if (prep_check_enabled()) {
        return noisy_matrix_rowmajor_check_(out, signal, rows, K, out_lda, is_b, has_signal);
    }
    return true;
}

/* CP_OCL_PREP_CHECK=1: recompute with the per-row kernel and require identical bytes. */
bool Case33OclPrep::noisy_matrix_rowmajor_check_(cl_mem out, cl_mem signal, int rows, int K,
                                                 int out_lda, int is_b, int has_signal) {
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(out_lda);
    if (!ensure_noisy_scratch(bytes)) {
        return false;
    }
    const int rank = R_RANK;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_rowmajor_, 0, sizeof(cl_mem), &d_noisy_scratch_);
    err |= clSetKernelArg(k_noisy_rowmajor_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_noisy_rowmajor_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_rowmajor_, 3, sizeof(int), &rows);
    err |= clSetKernelArg(k_noisy_rowmajor_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_rowmajor_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_noisy_rowmajor_, 6, sizeof(int), &is_b);
    err |= clSetKernelArg(k_noisy_rowmajor_, 7, sizeof(int), &has_signal);
    err |= clSetKernelArg(k_noisy_rowmajor_, 8, sizeof(cl_mem), &signal);
    err |= clSetKernelArg(k_noisy_rowmajor_, 9, sizeof(int), &out_lda);
    const size_t l = 256;
    const size_t g = (static_cast<size_t>(rows) + l - 1) / l * l;
    if (err != CL_SUCCESS || clEnqueueNDRangeKernel(ocl_->queue, k_noisy_rowmajor_, 1, nullptr,
                                                    &g, &l, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    std::vector<char> a(bytes), b(bytes);
    if (!ocl_->read_buffer(out, a.data(), bytes) ||
        !ocl_->read_buffer(d_noisy_scratch_, b.data(), bytes)) {
        return false;
    }
    const bool same = a == b;
    std::printf("[ocl-prep] row-major noise check (%s, %dx%d): %s\n", is_b ? "B" : "A", rows,
                out_lda, same ? "identical" : "MISMATCH");
    std::fflush(stdout);
    return same;
}

bool Case33OclPrep::noisy_matrix_a_colmajor_(cl_mem out, cl_mem signal, int rows, int K, int lda) {
    if (!ready_ || !out || !signal || rows <= 0 || K <= 0 || lda < rows) {
        return false;
    }
    if (!build_perm_pairs_(0, K)) {
        return false;
    }

    const int rank = R_RANK;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_a_colmajor_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 3, sizeof(int), &rows);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 6, sizeof(cl_mem), &signal);
    err |= clSetKernelArg(k_noisy_a_colmajor_, 7, sizeof(int), &lda);
    if (err != CL_SUCCESS) {
        return false;
    }

    const size_t g = static_cast<size_t>((rows + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_noisy_a_colmajor_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] noisy_matrix_a_colmajor launch failed\n");
        return false;
    }
    if (!pad_colmajor_col_tails_(out, K, rows, lda)) {
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::pad_colmajor_col_tails_(cl_mem out, int cols, int valid_rows, int lda) {
    if (!ready_ || !out || cols <= 0 || lda <= valid_rows) {
        return true;
    }
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_pad_col_tails_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_pad_col_tails_, 1, sizeof(int), &cols);
    err |= clSetKernelArg(k_pad_col_tails_, 2, sizeof(int), &valid_rows);
    err |= clSetKernelArg(k_pad_col_tails_, 3, sizeof(int), &lda);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = static_cast<size_t>((cols + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_pad_col_tails_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    return err == CL_SUCCESS;
}

bool Case33OclPrep::noisy_matrix_colmajor_(cl_mem out, int cols, int K, int ldb, int is_b) {
    if (!ready_ || !out || cols <= 0 || K <= 0 || ldb < K) {
        return false;
    }
    if (!build_perm_pairs_(is_b, K)) {
        return false;
    }
    // Column j at out + j*ldb is the row-major noise layout without signal.
    const double t0 = prep_timing_enabled() ? cp_now_sec() : 0.0;
    if (noisy_matrix_rowmajor_wg_(out, nullptr, cols, K, ldb, is_b, 0)) {
        if (prep_timing_enabled()) {
            std::printf("[ocl-prep] %s column-major noise: %.1fms\n", is_b ? "B" : "A",
                        (cp_now_sec() - t0) * 1e3);
        }
        return true;
    }

    const int rank = R_RANK;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_colmajor_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_noisy_colmajor_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_noisy_colmajor_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_colmajor_, 3, sizeof(int), &cols);
    err |= clSetKernelArg(k_noisy_colmajor_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_colmajor_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_noisy_colmajor_, 6, sizeof(int), &is_b);
    err |= clSetKernelArg(k_noisy_colmajor_, 7, sizeof(int), &ldb);
    if (err != CL_SUCCESS) {
        return false;
    }

    const size_t g = static_cast<size_t>((cols + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_noisy_colmajor_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] noisy_matrix_colmajor launch failed\n");
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::pad_rowmajor_row_tails_(cl_mem out, int rows, int valid_cols, int ldb) {
    if (!ready_ || !out || rows <= 0 || ldb <= valid_cols) {
        return true;
    }
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_pad_row_tails_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_pad_row_tails_, 1, sizeof(int), &rows);
    err |= clSetKernelArg(k_pad_row_tails_, 2, sizeof(int), &valid_cols);
    err |= clSetKernelArg(k_pad_row_tails_, 3, sizeof(int), &ldb);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = static_cast<size_t>((rows + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_pad_row_tails_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::noisy_matrix_b_rowmajor_(cl_mem out, int K, int N, int ldb) {
    if (!ready_ || !out || K <= 0 || N <= 0 || ldb < N) {
        return false;
    }
    if (!build_perm_pairs_(1, K)) {
        return false;
    }

    const int rank = R_RANK;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 3, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 4, sizeof(int), &N);
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_noisy_b_rowmajor_, 6, sizeof(int), &ldb);
    if (err != CL_SUCCESS) {
        return false;
    }

    const size_t g = static_cast<size_t>((N + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_noisy_b_rowmajor_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] noisy_matrix_b_rowmajor launch failed\n");
        return false;
    }
    if (!pad_rowmajor_row_tails_(out, K, N, ldb)) {
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::merkle_finish_root_(int num_subroots) {
    cl_int err = CL_SUCCESS;
    const int num_mt_blocks = (num_subroots + 255) / 256;

    int num_leaves = num_subroots;
    int is_single = (num_mt_blocks == 1) ? 1 : 0;
    err |= clSetKernelArg(k_compute_blake_mt_, 0, sizeof(cl_mem), &d_job_key_);
    err |= clSetKernelArg(k_compute_blake_mt_, 1, sizeof(cl_mem), &d_merkle_roots_);
    err |= clSetKernelArg(k_compute_blake_mt_, 2, sizeof(int), &num_leaves);
    err |= clSetKernelArg(k_compute_blake_mt_, 3, sizeof(int), &is_single);
    if (err != CL_SUCCESS) {
        return false;
    }

    const size_t g = static_cast<size_t>(num_mt_blocks) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_compute_blake_mt_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }

    if (num_mt_blocks > 1) {
        num_leaves = num_mt_blocks;
        err |= clSetKernelArg(k_reduce_roots_, 0, sizeof(cl_mem), &d_job_key_);
        err |= clSetKernelArg(k_reduce_roots_, 1, sizeof(cl_mem), &d_merkle_roots_);
        err |= clSetKernelArg(k_reduce_roots_, 2, sizeof(int), &num_leaves);
        if (err != CL_SUCCESS) {
            return false;
        }
        const size_t g2 = 256;
        const size_t l2 = 256;
        err = clEnqueueNDRangeKernel(ocl_->queue, k_reduce_roots_, 1, nullptr, &g2, &l2, 0, nullptr,
                                     nullptr);
    }
    return err == CL_SUCCESS;
}

bool Case33OclPrep::zero_matrix_keyed_hash(int n, int K, const uint8_t job_key[32],
                                           uint8_t out[32]) {
    if (!ready_ || n <= 0 || K <= 0 || !job_key || !out) {
        return false;
    }
    /* size the Merkle scratch for n x K without shrinking the A buffer */
    if (!ensure_buffers(m_cap_ > 0 ? m_cap_ : n, n, K)) {
        return false;
    }
    const size_t raw = static_cast<size_t>(n) * static_cast<size_t>(K);
    const size_t pad = (raw + 1023) / 1024 * 1024;
    if (pad / 1024 <= 1) {
        return false; /* single-chunk case: leave it to the CPU path */
    }
    /* raw_len 0: every chunk takes the kernel's all-zero path, the buffer is never read */
    return matrix_keyed_hash_(d_A_sig_, 0, pad, job_key, out, nullptr, nullptr);
}

bool Case33OclPrep::hash_signal_a_(size_t raw_len, size_t pad_len, const uint8_t job_key[32],
                                   uint8_t out[32]) {
    a_witness_valid_ = false;
    if (!matrix_keyed_hash_(d_A_sig_, raw_len, pad_len, job_key, out, d_a_subroots_,
                            &a_num_subroots_)) {
        return false;
    }
    memcpy(a_root_, out, 32);
    a_witness_valid_ = true;
    return true;
}

bool Case33OclPrep::matrix_keyed_hash_(cl_mem d_mat, size_t raw_len, size_t pad_len,
                                       const uint8_t job_key[32], uint8_t out[32],
                                       cl_mem subroots_out, int *num_subroots_out) {
    const int num_chunks = static_cast<int>(pad_len / 1024);
    if (num_chunks <= 0) {
        return false;
    }
    if (num_subroots_out) {
        *num_subroots_out = 0;
    }

    if (num_chunks == 1) {
        std::vector<uint8_t> tmp(pad_len);
        if (!ocl_->read_buffer(d_mat, tmp.data(), raw_len)) {
            return false;
        }
        if (pad_len > raw_len) {
            memset(tmp.data() + raw_len, 0, pad_len - raw_len);
        }
        pearl_keyed_matrix_digest(tmp.data(), pad_len, job_key, out);
        return true;
    }

    if (!ocl_->write_buffer(d_job_key_, job_key, 32)) {
        return false;
    }

    const int num_subroots = (num_chunks + 255) / 256;
    cl_ulong raw_len_ul = static_cast<cl_ulong>(raw_len);
    cl_ulong pad_len_ul = static_cast<cl_ulong>(pad_len);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_keyed_chunk_roots_, 0, sizeof(cl_mem), &d_mat);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 1, sizeof(cl_ulong), &raw_len_ul);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 2, sizeof(cl_ulong), &pad_len_ul);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 3, sizeof(cl_mem), &d_job_key_);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 4, sizeof(cl_mem), &d_merkle_roots_);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 5, sizeof(int), &num_chunks);
    if (err != CL_SUCCESS) {
        return false;
    }

    const size_t g = static_cast<size_t>(num_subroots) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_keyed_chunk_roots_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    if (subroots_out) {
        err = clEnqueueCopyBuffer(ocl_->queue, d_merkle_roots_, subroots_out, 0, 0,
                                  static_cast<size_t>(num_subroots) * 32, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }
        if (num_subroots_out) {
            *num_subroots_out = num_subroots;
        }
    }
    if (!merkle_finish_root_(num_subroots)) {
        return false;
    }
    clFinish(ocl_->queue);
    return ocl_->read_buffer(d_merkle_roots_, out, 32);
}

bool Case33OclPrep::prepare_job_b(cl_mem b_buf, const uint8_t b_noise_seed[32], int n, int K,
                                  int blocks_k, int macro_cols) {
    if (!ready_ || !b_buf || !b_noise_seed) {
        return false;
    }
    if (!d_A_sig_) {
        return false;
    }
    if (!ocl_->write_buffer(d_noise_seed_, b_noise_seed, 32)) {
        return false;
    }

    const int is_b = 1;
    const int rank = R_RANK;
    const int pair_groups = (K + 7) / 8;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_build_pairs_, 0, sizeof(int), &is_b);
    err |= clSetKernelArg(k_build_pairs_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_build_pairs_, 2, sizeof(int), &K);
    err |= clSetKernelArg(k_build_pairs_, 3, sizeof(int), &rank);
    err |= clSetKernelArg(k_build_pairs_, 4, sizeof(cl_mem), &d_pairs_);
    if (err != CL_SUCCESS) {
        return false;
    }
    {
        const size_t g = static_cast<size_t>(pair_groups);
        err = clEnqueueNDRangeKernel(ocl_->queue, k_build_pairs_, 1, nullptr, &g, nullptr, 0,
                                     nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }
    }

    const int has_signal = 0;
    cl_mem dummy_sig = d_A_sig_;
    err = CL_SUCCESS;
    err |= clSetKernelArg(k_fused_prepack_b_, 0, sizeof(cl_mem), &b_buf);
    err |= clSetKernelArg(k_fused_prepack_b_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_fused_prepack_b_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_fused_prepack_b_, 3, sizeof(int), &n);
    err |= clSetKernelArg(k_fused_prepack_b_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_fused_prepack_b_, 5, sizeof(int), &rank);
    err |= clSetKernelArg(k_fused_prepack_b_, 6, sizeof(int), &blocks_k);
    err |= clSetKernelArg(k_fused_prepack_b_, 7, sizeof(int), &macro_cols);
    err |= clSetKernelArg(k_fused_prepack_b_, 8, sizeof(int), &has_signal);
    err |= clSetKernelArg(k_fused_prepack_b_, 9, sizeof(cl_mem), &dummy_sig);
    if (err != CL_SUCCESS) {
        return false;
    }

    const int micro_n = case32::kMacroN / case32::kNR;
    const size_t num_groups =
            static_cast<size_t>(macro_cols) * static_cast<size_t>(blocks_k) *
            static_cast<size_t>(micro_n);
    const size_t l2 = case32::kNR;
    const size_t g2 = num_groups * l2;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_fused_prepack_b_, 1, nullptr, &g2, &l2, 0,
                                 nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] fused_prepack_b launch failed\n");
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::prepare_attempt_a(cl_mem a_buf, const uint8_t *ab_seed, int ab_seed_len,
                                      const uint8_t job_key[32], const uint8_t b_noise_seed[32],
                                      int m, int K, int blocks_k, int macro_rows, int salted,
                                      uint8_t a_key_out[32]) {
    if (!ready_ || !a_buf || !ab_seed || !job_key || !b_noise_seed || !a_key_out) {
        return false;
    }
    if (!ensure_buffers(m, 1, K)) {
        return false;
    }

    if (!gen_random_a_(pearl_seed_to_u64(ab_seed, ab_seed_len), m * K)) {
        return false;
    }

    const size_t raw_a = static_cast<size_t>(m) * static_cast<size_t>(K);
    const size_t pad_a = (raw_a + 1023) / 1024 * 1024;
    uint8_t hash_a[32];
    if (!hash_signal_a_(raw_a, pad_a, job_key, hash_a)) {
        return false;
    }

    pearl_a_noise_seed_from_hash(b_noise_seed, hash_a, static_cast<uint32_t>(m), salted,
                                 a_key_out);

    if (!ocl_->write_buffer(d_noise_seed_, a_key_out, 32)) {
        return false;
    }

    return fused_prepack_a(a_buf, m, K, blocks_k, macro_rows);
}

bool Case33OclPrep::prepare_attempt_a_rowmajor(cl_mem a_noisy_out, const uint8_t *ab_seed,
                                               int ab_seed_len, const uint8_t job_key[32],
                                               const uint8_t b_noise_seed[32], int m, int K,
                                               int out_lda, int salted, uint8_t a_key_out[32]) {
    if (!ready_ || !a_noisy_out || !ab_seed || !job_key || !b_noise_seed || !a_key_out) {
        return false;
    }
    if (!ensure_buffers(m, 1, K)) {
        return false;
    }
    const double t_start = prep_timing_enabled() ? cp_now_sec() : 0.0;

    if (!gen_random_a_(pearl_seed_to_u64(ab_seed, ab_seed_len), m * K)) {
        return false;
    }
    const double t_gen = prep_timing_enabled() ? cp_now_sec() : 0.0;

    const size_t raw_a = static_cast<size_t>(m) * static_cast<size_t>(K);
    const size_t pad_a = (raw_a + 1023) / 1024 * 1024;
    uint8_t hash_a[32];
    if (!hash_signal_a_(raw_a, pad_a, job_key, hash_a)) {
        return false;
    }
    const double t_hash = prep_timing_enabled() ? cp_now_sec() : 0.0;

    pearl_a_noise_seed_from_hash(b_noise_seed, hash_a, static_cast<uint32_t>(m), salted,
                                 a_key_out);

    if (!ocl_->write_buffer(d_noise_seed_, a_key_out, 32)) {
        return false;
    }

    const bool ok = noisy_matrix_rowmajor_(a_noisy_out, d_A_sig_, m, K, out_lda, 0, 1);
    if (prep_timing_enabled()) {
        clFinish(ocl_->queue);
        std::printf("[ocl-prep] A row-major: gen=%.1fms hash=%.1fms noise=%.1fms\n",
                    (t_gen - t_start) * 1e3, (t_hash - t_gen) * 1e3,
                    (cp_now_sec() - t_hash) * 1e3);
    }
    return ok;
}

bool Case33OclPrep::prepare_job_b_rowmajor(cl_mem b_noisy_out, const uint8_t b_noise_seed[32],
                                           int n, int K) {
    if (!ready_ || !b_noisy_out || !b_noise_seed || n <= 0 || K <= 0) {
        return false;
    }
    if (!ocl_->write_buffer(d_noise_seed_, b_noise_seed, 32)) {
        return false;
    }
    return noisy_matrix_rowmajor_(b_noisy_out, nullptr, n, K, K, 1, 0);
}

bool Case33OclPrep::prepare_job_b_colmajor(cl_mem b_buf, const uint8_t b_noise_seed[32], int n,
                                           int K, int ldb) {
    if (!ready_ || !b_buf || !b_noise_seed || n <= 0 || K <= 0 || ldb < K) {
        return false;
    }
    if (!ocl_->write_buffer(d_noise_seed_, b_noise_seed, 32)) {
        return false;
    }
    return noisy_matrix_colmajor_(b_buf, n, K, ldb, 1);
}

bool Case33OclPrep::prepare_job_b_gpu(cl_mem b_buf, const uint8_t b_noise_seed[32], int n, int K,
                                      int ldb, bool b_row_major) {
    if (b_row_major) {
        if (!ready_ || !b_buf || !b_noise_seed || n <= 0 || K <= 0 || ldb < n) {
            return false;
        }
        if (!ocl_->write_buffer(d_noise_seed_, b_noise_seed, 32)) {
            return false;
        }
        return noisy_matrix_b_rowmajor_(b_buf, K, n, ldb);
    }
    return prepare_job_b_colmajor(b_buf, b_noise_seed, n, K, ldb);
}

bool Case33OclPrep::prepare_attempt_a_gpu(cl_mem a_buf, const uint8_t *ab_seed, int ab_seed_len,
                                          const uint8_t job_key[32], const uint8_t b_noise_seed[32],
                                          int m, int K, int lda, bool a_row_major, int salted,
                                          uint8_t a_key_out[32]) {
    if (a_row_major) {
        return prepare_attempt_a_rowmajor(a_buf, ab_seed, ab_seed_len, job_key, b_noise_seed, m, K,
                                          lda, salted, a_key_out);
    }
    if (!ready_ || !a_buf || !ab_seed || !job_key || !b_noise_seed || !a_key_out) {
        return false;
    }
    if (!ensure_buffers(m, 1, K)) {
        return false;
    }

    if (!gen_random_a_(pearl_seed_to_u64(ab_seed, ab_seed_len), m * K)) {
        return false;
    }

    const size_t raw_a = static_cast<size_t>(m) * static_cast<size_t>(K);
    const size_t pad_a = (raw_a + 1023) / 1024 * 1024;
    uint8_t hash_a[32];
    if (!hash_signal_a_(raw_a, pad_a, job_key, hash_a)) {
        return false;
    }

    pearl_a_noise_seed_from_hash(b_noise_seed, hash_a, static_cast<uint32_t>(m), salted,
                                 a_key_out);

    if (!ocl_->write_buffer(d_noise_seed_, a_key_out, 32)) {
        return false;
    }

    return noisy_matrix_a_colmajor_(a_buf, d_A_sig_, m, K, lda);
}

bool Case33OclPrep::fused_prepack_a_launch_(cl_kernel k, cl_mem a_buf, int m, int K,
                                            int blocks_k, int macro_rows) {
    int rank = R_RANK;
    const int micro_m = case32::kMacroM / case32::kMR;
    cl_int err = CL_SUCCESS;
    size_t g2 = 0, l2 = 0;
    if (k == k_fused_prepack_a_wg_) {
        err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a_buf);
        err |= clSetKernelArg(k, 1, sizeof(cl_mem), &d_el_rows_);
        err |= clSetKernelArg(k, 2, sizeof(cl_mem), &d_pairs_);
        err |= clSetKernelArg(k, 3, sizeof(cl_mem), &d_A_sig_);
        err |= clSetKernelArg(k, 4, sizeof(int), &K);
        err |= clSetKernelArg(k, 5, sizeof(int), &blocks_k);
        l2 = 256;
        g2 = static_cast<size_t>(macro_rows) * static_cast<size_t>(blocks_k) *
             static_cast<size_t>(fpa_split_) * l2;
    } else {
        err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a_buf);
        err |= clSetKernelArg(k, 1, sizeof(cl_mem), &d_noise_seed_);
        err |= clSetKernelArg(k, 2, sizeof(cl_mem), &d_pairs_);
        err |= clSetKernelArg(k, 3, sizeof(cl_mem), &d_A_sig_);
        err |= clSetKernelArg(k, 4, sizeof(int), &m);
        err |= clSetKernelArg(k, 5, sizeof(int), &K);
        err |= clSetKernelArg(k, 6, sizeof(int), &rank);
        err |= clSetKernelArg(k, 7, sizeof(int), &blocks_k);
        err |= clSetKernelArg(k, 8, sizeof(int), &macro_rows);
        l2 = case32::kMR;
        g2 = static_cast<size_t>(macro_rows) * static_cast<size_t>(blocks_k) *
             static_cast<size_t>(micro_m) * l2;
    }
    if (err != CL_SUCCESS) {
        return false;
    }
    err = clEnqueueNDRangeKernel(ocl_->queue, k, 1, nullptr, &g2, &l2, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] fused_prepack_a%s launch failed (%d)\n",
                     k == k_fused_prepack_a_wg_ ? "_wg" : "", (int)err);
        return false;
    }
    return true;
}

bool Case33OclPrep::fused_prepack_a(cl_mem a_buf, int m, int K, int blocks_k, int macro_rows) {
    if (!ready_ || !a_buf || m <= 0 || K <= 0 || blocks_k <= 0 || macro_rows <= 0) {
        return false;
    }

    if (!build_perm_pairs_(0, K)) {
        return false;
    }

    /* The 256-wide kernel stages R_RANK-byte noise rows (from ocl_uniform_rows) in local
       memory and keeps the pair indices in bytes. */
    const int rows = macro_rows * case32::kMacroM;
    const bool wg = k_fused_prepack_a_wg_ && fpa_wg_enabled() && R_RANK <= 256 &&
                    K % case32::kKR == 0 && uniform_rows_(rows, 0);
    if (wg && !fused_prepack_a_launch_(k_fused_prepack_a_wg_, a_buf, m, K, blocks_k, macro_rows)) {
        /* a driver that refuses the 256-wide kernel keeps mining on the 8-lane one */
        std::fprintf(stderr, "[ocl-prep] disabling fused_prepack_a_wg; using the 8-lane kernel\n");
        clReleaseKernel(k_fused_prepack_a_wg_);
        k_fused_prepack_a_wg_ = nullptr;
        return fused_prepack_a(a_buf, m, K, blocks_k, macro_rows);
    }
    if (!wg && !fused_prepack_a_launch_(k_fused_prepack_a_, a_buf, m, K, blocks_k, macro_rows)) {
        return false;
    }
    clFinish(ocl_->queue);

    if (wg && prep_check_enabled()) {
        const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(K);
        if (!ensure_noisy_scratch(bytes) ||
            !fused_prepack_a_launch_(k_fused_prepack_a_, d_noisy_scratch_, m, K, blocks_k,
                                     macro_rows)) {
            return false;
        }
        clFinish(ocl_->queue);
        std::vector<uint8_t> x(bytes), y(bytes);
        if (!ocl_->read_buffer(a_buf, x.data(), bytes) ||
            !ocl_->read_buffer(d_noisy_scratch_, y.data(), bytes)) {
            return false;
        }
        size_t diff = 0, first = bytes;
        for (size_t i = 0; i < bytes; ++i) {
            if (x[i] != y[i]) {
                if (!diff) {
                    first = i;
                }
                ++diff;
            }
        }
        if (diff) {
            std::fprintf(stderr,
                         "[ocl-prep] fused_prepack_a_wg DIFFERS from the 8-lane kernel: %zu bytes, "
                         "first at %zu; using the 8-lane result\n",
                         diff, first);
            return ocl_->write_buffer(a_buf, y.data(), bytes);
        }
        std::fprintf(stderr, "[ocl-prep] fused_prepack_a_wg matches the 8-lane kernel\n");
    }
    return true;
}

bool Case33OclPrep::write_noise_seed(const uint8_t seed[32]) {
    if (!ready_ || !seed) {
        return false;
    }
    return ocl_->write_buffer(d_noise_seed_, seed, 32);
}

bool Case33OclPrep::read_A_sig(int8_t *h_A_sig, size_t bytes) const {
    if (!d_A_sig_ || !h_A_sig) {
        return false;
    }
    return ocl_->read_buffer(d_A_sig_, h_A_sig, bytes);
}

bool Case33OclPrep::read_A_witness(size_t sz_a, const uint32_t *block_idx, int num_blocks,
                                   size_t block_bytes, uint8_t *blocks_out,
                                   uint8_t *subroots_out, uint8_t root_out[32]) const {
    if (!a_witness_valid_ || !d_A_sig_ || (num_blocks > 0 && (!block_idx || !blocks_out))) {
        return false;
    }
    for (int i = 0; i < num_blocks; ++i) {
        const size_t off = static_cast<size_t>(block_idx[i]) * block_bytes;
        if (off >= sz_a) {
            continue;
        }
        const size_t len = (sz_a - off < block_bytes) ? sz_a - off : block_bytes;
        if (!ocl_->read_buffer(d_A_sig_, blocks_out + static_cast<size_t>(i) * block_bytes, len,
                               off)) {
            return false;
        }
    }
    if (a_num_subroots_ > 0) {
        if (!subroots_out ||
            !ocl_->read_buffer(d_a_subroots_, subroots_out,
                               static_cast<size_t>(a_num_subroots_) * 32)) {
            return false;
        }
    }
    memcpy(root_out, a_root_, 32);
    return true;
}

bool Case33OclPrep::align_gen_random(cl_mem dst, uint64_t rng_seed, int matrix_tag,
                                     int total_elems) {
    if (!ready_ || !dst || total_elems <= 0) {
        return false;
    }
    cl_ulong rng_ul = static_cast<cl_ulong>(rng_seed);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_gen_random_, 0, sizeof(cl_ulong), &rng_ul);
    err |= clSetKernelArg(k_gen_random_, 1, sizeof(int), &matrix_tag);
    err |= clSetKernelArg(k_gen_random_, 2, sizeof(int), &total_elems);
    err |= clSetKernelArg(k_gen_random_, 3, sizeof(cl_mem), &dst);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = static_cast<size_t>((total_elems + 255) / 256) * 256;
    const size_t l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_gen_random_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::align_keyed_hash(cl_mem d_mat, size_t raw_len, size_t pad_len,
                                     const uint8_t job_key[32], uint8_t out[32]) {
    return matrix_keyed_hash_(d_mat, raw_len, pad_len, job_key, out);
}

bool Case33OclPrep::align_run_get_random_hash(int index, const uint8_t seed[32], int is_b,
                                             int prepend_index, uint8_t out[32]) {
    if (!ready_ || !seed || !out || !k_test_random_hash_ || !d_noise_seed_) {
        return false;
    }
    if (!ocl_->write_buffer(d_noise_seed_, seed, 32)) {
        return false;
    }
    cl_mem d_out = ocl_->alloc_buffer(32, CL_MEM_WRITE_ONLY);
    if (!d_out) {
        return false;
    }
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_test_random_hash_, 0, sizeof(int), &index);
    err |= clSetKernelArg(k_test_random_hash_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_test_random_hash_, 2, sizeof(int), &is_b);
    err |= clSetKernelArg(k_test_random_hash_, 3, sizeof(int), &prepend_index);
    err |= clSetKernelArg(k_test_random_hash_, 4, sizeof(cl_mem), &d_out);
    bool ok = false;
    if (err == CL_SUCCESS) {
        const size_t g = 1;
        err = clEnqueueNDRangeKernel(ocl_->queue, k_test_random_hash_, 1, nullptr, &g, nullptr, 0,
                                     nullptr, nullptr);
        if (err == CL_SUCCESS) {
            clFinish(ocl_->queue);
            ok = ocl_->read_buffer(d_out, out, 32);
        }
    }
    clReleaseMemObject(d_out);
    return ok;
}

bool Case33OclPrep::align_build_perm_pairs(int is_b, const uint8_t noise_seed[32], int k) {
    if (!ready_ || !noise_seed || k <= 0) {
        return false;
    }
    if (!ocl_->write_buffer(d_noise_seed_, noise_seed, 32)) {
        return false;
    }
    const int rank = R_RANK;
    const int pair_groups = (k + 7) / 8;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_build_pairs_, 0, sizeof(int), &is_b);
    err |= clSetKernelArg(k_build_pairs_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_build_pairs_, 2, sizeof(int), &k);
    err |= clSetKernelArg(k_build_pairs_, 3, sizeof(int), &rank);
    err |= clSetKernelArg(k_build_pairs_, 4, sizeof(cl_mem), &d_pairs_);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = static_cast<size_t>(pair_groups);
    err = clEnqueueNDRangeKernel(ocl_->queue, k_build_pairs_, 1, nullptr, &g, nullptr, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    return true;
}

bool Case33OclPrep::align_read_perm_pairs(uint32_t *out, size_t word_count) {
    if (!d_pairs_ || !out || word_count == 0) {
        return false;
    }
    return ocl_->read_buffer(d_pairs_, out, word_count * sizeof(uint32_t));
}
