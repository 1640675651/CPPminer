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

/* CP_ONEDNN_PANEL_TRACE=1 (with CP_OCL_QUEUE_PROFILING=1): the per-attempt A prep records an event per kernel
   and prints [prep-trace] / [prep-host] lines (debug; see Case33OclPrep::trace_ev_). */
bool prep_trace_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_ONEDNN_PANEL_TRACE");
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

/* CP_OCL_PREP_ROWPOKE=0: regenerate and rehash the whole signal matrix every nonce instead of
   updating one random row and its Merkle group. */
bool prep_rowpoke_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_PREP_ROWPOKE");
        return !(v && v[0] == '0');
    }();
    return on;
}

static inline uint64_t prep_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* CP_OCL_PREP_UROWS_B3=0 keeps the byte-array uniform-rows kernel (ocl_uniform_rows). */
bool prep_urows_b3_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_PREP_UROWS_B3");
        return !(v && v[0] == '0');
    }();
    return on;
}

/* CP_OCL_PREP_SG=0 keeps the local-memory gather kernel (ocl_noisy_matrix_rowmajor_wg) instead of
   the sub-group shuffle one. */
bool prep_sg_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CP_OCL_PREP_SG");
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
    if (attempt_done_ev_) {
        clWaitForEvents(1, &attempt_done_ev_);   // its callback may still run
        clReleaseEvent(attempt_done_ev_);
        attempt_done_ev_ = nullptr;
    }
    release_kernels_();
    if (d_dummy_) {
        clReleaseMemObject(d_dummy_);
        d_dummy_ = nullptr;
    }
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
    if (d_a_key_) {
        clReleaseMemObject(d_a_key_);
        d_a_key_ = nullptr;
    }
    if (d_a_root_) {
        clReleaseMemObject(d_a_root_);
        d_a_root_ = nullptr;
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
    /* CP_OCL_PREP_DEFS="-DNSG_KPL=16 ...": extra build options for prep kernel experiments
       (the NSG_* defines select ocl_noisy_matrix_rowmajor_sg's decomposition). */
    if (const char *d = std::getenv("CP_OCL_PREP_DEFS"); d && d[0]) {
        opts += std::string(" ") + d;
    }
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
    k_keyed_chunk_roots_fill_ = mk("ocl_keyed_chunk_roots_fill");
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
    k_noisy_rowmajor_sg_ = mk("ocl_noisy_matrix_rowmajor_sg");
    k_uniform_rows_b3_ = mk("ocl_uniform_rows_b3");
    k_noisy_a_packed_ = mk("ocl_noisy_matrix_a_packed16");
    k_noisy_a_packed_any_ = mk("ocl_noisy_matrix_a_packed");
    k_noisy_b_packed_ = mk("ocl_noisy_matrix_b_packed");
    k_noisy_a_colmajor_ = mk("ocl_noisy_matrix_a_colmajor");
    k_noisy_colmajor_ = mk("ocl_noisy_matrix_colmajor");
    k_noisy_b_rowmajor_ = mk("ocl_noisy_matrix_b_rowmajor");
    k_pad_row_tails_ = mk("ocl_pad_rowmajor_row_tails");
    k_pad_col_tails_ = mk("ocl_pad_colmajor_col_tails");
    k_test_random_hash_ = mk("ocl_test_get_random_hash");
    k_a_noise_seed_ = mk("ocl_a_noise_seed");

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
    if (k_keyed_chunk_roots_fill_) {
        clReleaseKernel(k_keyed_chunk_roots_fill_);
        k_keyed_chunk_roots_fill_ = nullptr;
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
    if (k_noisy_rowmajor_sg_) {
        clReleaseKernel(k_noisy_rowmajor_sg_);
        k_noisy_rowmajor_sg_ = nullptr;
    }
    if (k_uniform_rows_b3_) {
        clReleaseKernel(k_uniform_rows_b3_);
        k_uniform_rows_b3_ = nullptr;
    }
    if (k_noisy_a_packed_) {
        clReleaseKernel(k_noisy_a_packed_);
        k_noisy_a_packed_ = nullptr;
    }
    if (k_noisy_a_packed_any_) {
        clReleaseKernel(k_noisy_a_packed_any_);
        k_noisy_a_packed_any_ = nullptr;
    }
    if (k_noisy_b_packed_) {
        clReleaseKernel(k_noisy_b_packed_);
        k_noisy_b_packed_ = nullptr;
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
    if (k_a_noise_seed_) {
        clReleaseKernel(k_a_noise_seed_);
        k_a_noise_seed_ = nullptr;
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
        // Host-written every attempt: READ_WRITE, not READ_ONLY (see OpenClContext::write_buffer).
        d_job_key_ = ocl_->alloc_buffer(32, CL_MEM_READ_WRITE);
    }
    if (!d_noise_seed_) {
        d_noise_seed_ = ocl_->alloc_buffer(32, CL_MEM_READ_WRITE);
    }
    if (!d_a_key_) {
        d_a_key_ = ocl_->alloc_buffer(32, CL_MEM_READ_WRITE);
    }
    if (!d_a_root_) {
        d_a_root_ = ocl_->alloc_buffer(32, CL_MEM_READ_WRITE);
    }
    if (!d_pairs_) {
        d_pairs_ = ocl_->alloc_buffer(static_cast<size_t>(K_DIM) * 2u * sizeof(uint32_t),
                                      CL_MEM_READ_ONLY);
    }
    if (!d_job_key_ || !d_noise_seed_ || !d_pairs_ || !d_a_key_ || !d_a_root_) {
        return false;
    }
    ready_ = true;
    return true;
}

bool Case33OclPrep::ensure_buffers(int m, int n, int k) {
    if (!ready_) {
        return false;
    }
    /* constant signal: the committed A has one more row (the dummy row) */
    const size_t szA = static_cast<size_t>(m + (const_sig_ ? 1 : 0)) * static_cast<size_t>(k);
    const size_t szB = static_cast<size_t>(n) * static_cast<size_t>(k);
    const size_t raw_max = szA > szB ? szA : szB;
    const size_t pad_max = (raw_max + 1023) / 1024 * 1024;
    const size_t chunks_max = pad_max / 1024;
    size_t merkle_need = ((chunks_max + 255) / 256) * 32;
    if (merkle_need < 32) {
        merkle_need = 32;
    }

    if (const_sig_) {
        /* no signal A buffer: only the dummy row */
        if (!d_dummy_ || dummy_cap_ < static_cast<size_t>(k)) {
            if (d_dummy_) {
                clReleaseMemObject(d_dummy_);
            }
            d_dummy_ = ocl_->alloc_buffer(static_cast<size_t>(k), CL_MEM_READ_WRITE);
            dummy_cap_ = d_dummy_ ? static_cast<size_t>(k) : 0;
            const_a_init_ = false;
        }
        if (m > m_cap_) {
            m_cap_ = m;
        }
    } else if (m > m_cap_ || !d_A_sig_) {
        if (d_A_sig_) {
            clReleaseMemObject(d_A_sig_);
        }
        d_A_sig_ = ocl_->alloc_buffer(szA, CL_MEM_READ_WRITE);
        a_sig_init_ = false;
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
        const_a_init_ = false;
    }
    n_cap_ = n;
    return (const_sig_ ? d_dummy_ != nullptr : d_A_sig_ != nullptr) && d_merkle_roots_ && d_a_subroots_;
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
    return gen_random_(d_A_sig_, rng_seed, total);
}

bool Case33OclPrep::gen_random_(cl_mem dst, uint64_t rng_seed, int total) {
    const int matrix_tag = 0;
    const cl_ulong rng_ul = static_cast<cl_ulong>(rng_seed);
    const bool wide = k_gen_random16_ && (total % 16) == 0;
    cl_kernel k = wide ? k_gen_random16_ : k_gen_random_;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_ulong), &rng_ul);
    err |= clSetKernelArg(k, 1, sizeof(int), &matrix_tag);
    err |= clSetKernelArg(k, 2, sizeof(int), &total);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &dst);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t items = wide ? static_cast<size_t>(total / 16) : static_cast<size_t>(total);
    const size_t l = 256;
    const size_t g = (items + l - 1) / l * l;
    if (clEnqueueNDRangeKernel(ocl_->queue, k, 1, nullptr, &g, &l, 0, nullptr, trace_ev_("gen_random"))
            != CL_SUCCESS) {
        return false;
    }
    if (!attempt_nofinish_) {   // async attempt prep: the next kernels are on the same in-order queue
        clFinish(ocl_->queue);
    }
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
                                 trace_ev_("build_pairs"));
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
    if (k_uniform_rows_b3_ && rank == 128 && prep_urows_b3_enabled()) {
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(k_uniform_rows_b3_, 0, sizeof(cl_mem), &d_noise_seed_);
        err |= clSetKernelArg(k_uniform_rows_b3_, 1, sizeof(int), &rows);
        err |= clSetKernelArg(k_uniform_rows_b3_, 2, sizeof(int), &is_b);
        err |= clSetKernelArg(k_uniform_rows_b3_, 3, sizeof(cl_mem), &d_el_rows_);
        const size_t lb = 256;
        const size_t gb = (static_cast<size_t>(rows) * 4 + lb - 1) / lb * lb;
        if (err == CL_SUCCESS &&
            clEnqueueNDRangeKernel(ocl_->queue, k_uniform_rows_b3_, 1, nullptr, &gb, &lb, 0,
                                   nullptr, trace_ev_("uniform_rows_b3")) == CL_SUCCESS) {
            if (prep_check_enabled()) {
                return uniform_rows_check_(rows, is_b);
            }
            return true;
        }
        std::fprintf(stderr, "[ocl-prep] uniform_rows_b3 launch failed; using the byte-array kernel\n");
        clReleaseKernel(k_uniform_rows_b3_);
        k_uniform_rows_b3_ = nullptr;
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
                                  trace_ev_("uniform_rows")) == CL_SUCCESS;
}

/* Noisy A in gemmstone's packed layout, panels of a_pack_rows_ rows with 4 k per dword
   (ocl_noisy_matrix_a_packed16 for 16 rows, ocl_noisy_matrix_a_packed for other heights up to 64),
   optionally tiled a_tile_r_ x a_tile_c_ inside the panel (gemmstone A4#r,c; the kernels take the
   untiled layout as tile P x 64). There is no fallback: the GEMM kernel was built for this layout, so a
   failure fails the attempt. */
// The packed A launch. In an async attempt prep its event also carries the completion callback that timestamps
// the end of the prep (watch_attempt_done_).
bool Case33OclPrep::enqueue_noisy_a_packed_(cl_kernel kern, size_t g, size_t l) {
    cl_event *trace = trace_ev_("noisy_a_packed");
    cl_event own = nullptr;
    cl_event *ev = trace ? trace : (attempt_nofinish_ ? &own : nullptr);
    if (clEnqueueNDRangeKernel(ocl_->queue, kern, 1, nullptr, &g, &l, 0, nullptr, ev) != CL_SUCCESS) {
        return false;
    }
    if (attempt_nofinish_ && ev && *ev) {
        watch_attempt_done_(*ev, /*owned=*/ev == &own);
    }
    return true;
}

static void CL_CALLBACK attempt_done_cb(cl_event, cl_int, void *user) {
    static_cast<std::atomic<double> *>(user)->store(cp_now_sec(), std::memory_order_release);
}

void Case33OclPrep::watch_attempt_done_(cl_event ev, bool owned) {
    if (!owned) {
        clRetainEvent(ev);   // the trace list releases its own reference
    }
    attempt_done_ev_ = ev;
    if (clSetEventCallback(ev, CL_COMPLETE, attempt_done_cb, &attempt_done_sec_) != CL_SUCCESS) {
        attempt_done_sec_.store(0.0, std::memory_order_release);
    }
}

bool Case33OclPrep::noisy_matrix_a_packed_(cl_mem out, cl_mem signal, int rows, int K, int out_lda) {
    const int P = a_pack_rows_;
    const bool p16 = P == 16;
    const int unit = p16 ? 64 * 4 : 64; /* NPA_UNROLL default for the 16-row kernel */
    cl_kernel kern = p16 ? k_noisy_a_packed_ : k_noisy_a_packed_any_;
    const int tR = (a_tile_r_ > 0) ? a_tile_r_ : P;
    const int tC = (a_tile_c_ > 0) ? a_tile_c_ : 64;
    if (!ready_ || !out || !signal || !kern || P <= 0 || P > 64 || rows <= 0 || rows % P != 0 ||
        K % unit != 0 || out_lda < K || out_lda % 4 != 0 || R_RANK != 128 || P % tR != 0 ||
        tC % 4 != 0 || 64 % tC != 0) {
        std::fprintf(stderr, "[ocl-prep] packed A: unsupported shape (rows %d, K %d, lda %d, panel %d, tile %dx%d)\n",
                     rows, K, out_lda, P, tR, tC);
        return false;
    }
    if (!build_perm_pairs_(0, K)) {
        return false;
    }
    const double t0 = prep_timing_enabled() ? cp_now_sec() : 0.0;
    if (!uniform_rows_(rows, 0)) {
        return false;
    }
    if (prep_timing_enabled()) {
        clFinish(ocl_->queue);
    }
    const double t1 = prep_timing_enabled() ? cp_now_sec() : 0.0;
    const int panel_stride_dw = P * (out_lda / 4);
    cl_int err = CL_SUCCESS;
    int arg = 0;
    err |= clSetKernelArg(kern, arg++, sizeof(cl_mem), &out);
    err |= clSetKernelArg(kern, arg++, sizeof(cl_mem), &d_el_rows_);
    err |= clSetKernelArg(kern, arg++, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(kern, arg++, sizeof(int), &rows);
    err |= clSetKernelArg(kern, arg++, sizeof(int), &K);
    err |= clSetKernelArg(kern, arg++, sizeof(cl_mem), &signal);
    if (!p16) {
        err |= clSetKernelArg(kern, arg++, sizeof(int), &P);
    }
    err |= clSetKernelArg(kern, arg++, sizeof(int), &panel_stride_dw);
    err |= clSetKernelArg(kern, arg++, sizeof(int), &tR);
    err |= clSetKernelArg(kern, arg++, sizeof(int), &tC);
    const cl_int sig_from_buf = const_sig_ ? 0 : 1;   /* constant signal: `signal` is not read */
    const cl_uint sig_word = 0x01010101u * static_cast<uint8_t>(const_sig_);
    err |= clSetKernelArg(kern, arg++, sizeof(cl_int), &sig_from_buf);
    err |= clSetKernelArg(kern, arg++, sizeof(cl_uint), &sig_word);
    const size_t l = 256;
    const size_t g = static_cast<size_t>(rows / P) * l;
    if (err != CL_SUCCESS ||
        !enqueue_noisy_a_packed_(kern, g, l)) {
        std::fprintf(stderr, "[ocl-prep] packed A (%d-row panels) launch failed\n", P);
        return false;
    }
    if (!attempt_nofinish_ || prep_timing_enabled()) {   // async attempt prep: the scan queues behind it
        clFinish(ocl_->queue);
    }
    if (prep_timing_enabled()) {
        std::printf("[ocl-prep]   uniform rows %.1fms, noise (packed A) %.1fms\n", (t1 - t0) * 1e3,
                    (cp_now_sec() - t1) * 1e3);
    }
    if (!prep_check_enabled()) {
        return true;
    }
    /* CP_OCL_PREP_CHECK=1: the per-row reference kernel writes row-major A; compare through the
       packed index map. */
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(out_lda);
    if (!ensure_noisy_scratch(bytes)) {
        return false;
    }
    const int rank = R_RANK;
    const int is_b = 0, has_signal = const_sig_ ? 0 : 1;   /* constant signal: added below */
    err = CL_SUCCESS;
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
    const size_t lr = 256;
    const size_t gr = (static_cast<size_t>(rows) + lr - 1) / lr * lr;
    if (err != CL_SUCCESS || clEnqueueNDRangeKernel(ocl_->queue, k_noisy_rowmajor_, 1, nullptr, &gr,
                                                    &lr, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    std::vector<int8_t> packed(bytes), ref(bytes);
    if (!ocl_->read_buffer(out, packed.data(), bytes) ||
        !ocl_->read_buffer(d_noisy_scratch_, ref.data(), bytes)) {
        return false;
    }
    size_t bad = 0;
    for (int m = 0; m < rows; ++m) {
        const int8_t *pan = packed.data() + static_cast<size_t>(m / P) * static_cast<size_t>(P) *
                                                    static_cast<size_t>(out_lda);
        const int8_t *rr = ref.data() + static_cast<size_t>(m) * static_cast<size_t>(out_lda);
        const size_t i = static_cast<size_t>(m % P);
        for (int k = 0; k < K; ++k) {
            // gemmstone's tiled Pc offset (crosspack 4): untiled when tR = P and tC covers K.
            const size_t kk = static_cast<size_t>(k), r = static_cast<size_t>(tR), c = static_cast<size_t>(tC);
            const size_t off = (i - i % r) * c + (kk - kk % c) * static_cast<size_t>(P) + (i % r) * 4u +
                               (kk % c - kk % 4u) * r + kk % 4u;
            if (pan[off] != static_cast<int8_t>(rr[k] + const_sig_)) {
                ++bad;
            }
        }
    }
    std::printf("[ocl-prep] packed A noise check (%dx%d, %d-row panels, tile %dx%d): %s\n", rows, K, P, tR, tC,
                bad == 0 ? "identical" : "MISMATCH");
    std::fflush(stdout);
    return bad == 0;
}

/* CP_OCL_PREP_CHECK=1: recompute the uniform rows with the byte-array kernel, require identical bytes. */
bool Case33OclPrep::uniform_rows_check_(int rows, int is_b) {
    const int rank = R_RANK;
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(rank);
    if (!ensure_noisy_scratch(bytes)) {
        return false;
    }
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_uniform_rows_, 0, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_uniform_rows_, 1, sizeof(int), &rows);
    err |= clSetKernelArg(k_uniform_rows_, 2, sizeof(int), &rank);
    err |= clSetKernelArg(k_uniform_rows_, 3, sizeof(int), &is_b);
    err |= clSetKernelArg(k_uniform_rows_, 4, sizeof(cl_mem), &d_noisy_scratch_);
    const size_t lu = 64;
    const size_t gu = (static_cast<size_t>(rows) + lu - 1) / lu * lu;
    if (err != CL_SUCCESS || clEnqueueNDRangeKernel(ocl_->queue, k_uniform_rows_, 1, nullptr, &gu,
                                                    &lu, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }
    clFinish(ocl_->queue);
    std::vector<uint8_t> a(bytes), b(bytes);
    if (!ocl_->read_buffer(d_el_rows_, a.data(), bytes) ||
        !ocl_->read_buffer(d_noisy_scratch_, b.data(), bytes)) {
        return false;
    }
    const bool same = a == b;
    std::printf("[ocl-prep] uniform rows check (%s, %d rows): %s\n", is_b ? "B" : "A", rows,
                same ? "identical" : "MISMATCH");
    std::fflush(stdout);
    return same;
}

bool Case33OclPrep::noisy_matrix_rowmajor_wg_(cl_mem out, cl_mem signal, int rows, int K,
                                              int out_lda, int is_b, int has_signal) {
    if (!k_uniform_rows_ || !k_noisy_rowmajor_wg_ || !prep_wg_enabled()) {
        return false;
    }
    const int rank = R_RANK;
    const double t0 = prep_timing_enabled() ? cp_now_sec() : 0.0;
    if (!uniform_rows_(rows, is_b)) {
        return false;
    }
    if (prep_timing_enabled()) {
        clFinish(ocl_->queue);
    }
    const double t1 = prep_timing_enabled() ? cp_now_sec() : 0.0;
    /* 1024 = NSG_STEP at the kernel's default defines (32 lanes x 4 k x 8 units in flight). */
    const bool sg = k_noisy_rowmajor_sg_ && prep_sg_enabled() && rank == 128 && (K % 1024) == 0;
    if (sg) {
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 0, sizeof(cl_mem), &out);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 1, sizeof(cl_mem), &d_el_rows_);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 2, sizeof(cl_mem), &d_pairs_);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 3, sizeof(int), &rows);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 4, sizeof(int), &K);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 5, sizeof(int), &has_signal);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 6, sizeof(cl_mem), &signal);
        err |= clSetKernelArg(k_noisy_rowmajor_sg_, 7, sizeof(int), &out_lda);
        const size_t ls = 256;
        const size_t per_wg = 8u; /* 8 sub-groups of 32 lanes, one row each */
        const size_t gs = (static_cast<size_t>(rows) + per_wg - 1) / per_wg * ls;
        if (err != CL_SUCCESS ||
            clEnqueueNDRangeKernel(ocl_->queue, k_noisy_rowmajor_sg_, 1, nullptr, &gs, &ls, 0,
                                   nullptr, nullptr) != CL_SUCCESS) {
            std::fprintf(stderr, "[ocl-prep] noisy_matrix_rowmajor_sg launch failed; using wg path\n");
            clReleaseKernel(k_noisy_rowmajor_sg_);
            k_noisy_rowmajor_sg_ = nullptr;
            return noisy_matrix_rowmajor_wg_(out, signal, rows, K, out_lda, is_b, has_signal);
        }
        clFinish(ocl_->queue);
        if (prep_timing_enabled()) {
            std::printf("[ocl-prep]   uniform rows %.1fms, noise (coalesced kernel) %.1fms\n",
                        (t1 - t0) * 1e3, (cp_now_sec() - t1) * 1e3);
        }
        if (prep_check_enabled()) {
            return noisy_matrix_rowmajor_check_(out, signal, rows, K, out_lda, is_b, has_signal);
        }
        return true;
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
    if (prep_timing_enabled()) {
        std::printf("[ocl-prep]   uniform rows %.1fms, noise (local-memory gather) %.1fms\n",
                    (t1 - t0) * 1e3, (cp_now_sec() - t1) * 1e3);
    }
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

bool Case33OclPrep::set_job_key_(const uint8_t job_key[32]) {
    if (dev_job_key_valid_ && memcmp(dev_job_key_, job_key, 32) == 0) {
        return true;
    }
    dev_job_key_valid_ = false;
    if (!ocl_->write_buffer(d_job_key_, job_key, 32)) {
        return false;
    }
    memcpy(dev_job_key_, job_key, 32);
    dev_job_key_valid_ = true;
    return true;
}

bool Case33OclPrep::derive_a_noise_seed_(const uint8_t b_noise_seed[32], int m, int salted,
                                         uint8_t *a_key_out) {
    if (seed_mode_ != 0 && k_a_noise_seed_ && d_a_key_ && d_a_root_ && d_merkle_roots_) {
        cl_uint8 bw;
        memcpy(&bw, b_noise_seed, 32);
        const cl_uint mu = static_cast<cl_uint>(m);
        const cl_int s = salted ? 1 : 0;
        cl_int err = clSetKernelArg(k_a_noise_seed_, 0, sizeof(cl_mem), &d_merkle_roots_);
        err |= clSetKernelArg(k_a_noise_seed_, 1, sizeof(cl_uint), &mu);
        err |= clSetKernelArg(k_a_noise_seed_, 2, sizeof(cl_int), &s);
        err |= clSetKernelArg(k_a_noise_seed_, 3, sizeof(cl_uint8), &bw);
        err |= clSetKernelArg(k_a_noise_seed_, 4, sizeof(cl_mem), &d_noise_seed_);
        err |= clSetKernelArg(k_a_noise_seed_, 5, sizeof(cl_mem), &d_a_key_);
        err |= clSetKernelArg(k_a_noise_seed_, 6, sizeof(cl_mem), &d_a_root_);
        const size_t g = 1, l = 1;
        if (err == CL_SUCCESS)
            err = clEnqueueNDRangeKernel(ocl_->queue, k_a_noise_seed_, 1, nullptr, &g, &l, 0, nullptr,
                                         trace_ev_("a_noise_seed"));
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[ocl-prep] noise seed kernel failed (%d); deriving it on the host\n", err);
            seed_mode_ = 0;
        } else if (seed_mode_ < 0 || prep_check_enabled()) {
            /* First use (and CP_OCL_PREP_CHECK): compare with the host derivation once. */
            uint8_t root[32], dev[32], host[32];
            if (!ocl_->read_buffer(d_a_root_, root, 32) || !ocl_->read_buffer(d_a_key_, dev, 32)) {
                return false;
            }
            pearl_a_noise_seed_from_hash(b_noise_seed, root, static_cast<uint32_t>(m), salted, host);
            const bool same = memcmp(dev, host, 32) == 0;
            if (!same || prep_check_enabled()) {
                std::printf("[ocl-prep] device A noise seed check: %s\n",
                            same ? "identical" : "MISMATCH, deriving it on the host from now on");
                std::fflush(stdout);
            }
            if (same) {
                seed_mode_ = 1;
                if (a_key_out) memcpy(a_key_out, dev, 32);
                return true;
            }
            seed_mode_ = 0;
        } else {
            return !a_key_out || ocl_->read_buffer(d_a_key_, a_key_out, 32);
        }
    }
    /* Host derivation: root from the device, seed back to the device. */
    uint8_t root[32], seed[32];
    if (!ocl_->read_buffer(d_merkle_roots_, root, 32)) {
        return false;
    }
    pearl_a_noise_seed_from_hash(b_noise_seed, root, static_cast<uint32_t>(m), salted, seed);
    if (!ocl_->write_buffer(d_noise_seed_, seed, 32) ||
        (d_a_key_ && !ocl_->write_buffer(d_a_key_, seed, 32)) ||
        (d_a_root_ && !ocl_->write_buffer(d_a_root_, root, 32))) {
        return false;
    }
    if (a_key_out) memcpy(a_key_out, seed, 32);
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
                                 trace_ev_("merkle_mt"));
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
                                     trace_ev_("reduce_roots"));
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

bool Case33OclPrep::update_signal_a_(const uint8_t *ab_seed, int ab_seed_len,
                                     const uint8_t job_key[32], int m, int K, uint8_t hash_a[32],
                                     double *t_after_gen) {
    const size_t raw_a = static_cast<size_t>(m) * static_cast<size_t>(K);
    const size_t pad_a = (raw_a + 1023) / 1024 * 1024;
    const uint64_t rng_seed = pearl_seed_to_u64(ab_seed, ab_seed_len);
    const bool incremental = prep_rowpoke_enabled() && m > 0 && K > 0 && (K % 1024) == 0 &&
            pad_a == raw_a;
    uint8_t root_local[32];
    bool full_hash = false;
    if (incremental && (!a_sig_init_ || a_sig_m_ != m || a_sig_k_ != K)) {
        /* First attempt (or new dimensions): start from a zero signal, then hash it all once. */
        const cl_char zero = 0;
        if (clEnqueueFillBuffer(ocl_->queue, d_A_sig_, &zero, sizeof(zero), 0, raw_a, 0, nullptr,
                                nullptr) != CL_SUCCESS &&
            !gen_random_a_(rng_seed, m * K)) {
            return false;
        }
        a_sig_init_ = true;
        a_sig_m_ = m;
        a_sig_k_ = K;
        full_hash = true;
    }
    if (!incremental) {
        /* Rows not whole chunks, or CP_OCL_PREP_ROWPOKE=0: the full random signal and its full
           commitment. */
        if (!gen_random_a_(rng_seed, m * K)) {
            return false;
        }
        if (t_after_gen) {
            *t_after_gen = prep_timing_enabled() ? cp_now_sec() : 0.0;
        }
        if (!hash_signal_a_(raw_a, pad_a, job_key, hash_a ? hash_a : root_local)) {
            return false;
        }
        a_sig_init_ = true;
        a_sig_m_ = m;
        a_sig_k_ = K;
        memcpy(a_hash_key_, job_key, 32);
        return true;
    }

    /* One random full row, generated on the GPU by the full generator (ocl_gen_random_matrix, tag 0)
       over just that row (global work offset); the rest of the signal keeps earlier nonces' rows,
       zero until written. */
    const uint32_t row = static_cast<uint32_t>(prep_splitmix64(rng_seed ^ 0xD1B54A32D192ED03ULL) %
                                               static_cast<uint64_t>(m));
    {
        const bool wide = k_gen_random16_ && (K % 16) == 0;
        cl_kernel k = wide ? k_gen_random16_ : k_gen_random_;
        const cl_ulong rng_ul = static_cast<cl_ulong>(rng_seed);
        const int tag = 0;
        const int total = m * K;
        cl_int err = clSetKernelArg(k, 0, sizeof(cl_ulong), &rng_ul);
        err |= clSetKernelArg(k, 1, sizeof(int), &tag);
        err |= clSetKernelArg(k, 2, sizeof(int), &total);
        err |= clSetKernelArg(k, 3, sizeof(cl_mem), &d_A_sig_);
        const size_t per = wide ? 16 : 1;
        const size_t off = static_cast<size_t>(row) * static_cast<size_t>(K) / per;
        const size_t g = static_cast<size_t>(K) / per;
        const size_t l = (g % 256 == 0) ? 256 : ((g % 64 == 0) ? 64 : 1);
        if (err == CL_SUCCESS)
            err = clEnqueueNDRangeKernel(ocl_->queue, k, 1, &off, &g, &l, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[ocl-prep] row generation launch failed (%d)\n", err);
            return false;
        }
    }
    if (prep_check_enabled()) {
        /* Compare the GPU row with the host formula the CPU worker uses. */
        std::vector<int8_t> got(static_cast<size_t>(K));
        if (!ocl_->read_buffer(d_A_sig_, got.data(), got.size(),
                               static_cast<size_t>(row) * static_cast<size_t>(K))) {
            return false;
        }
        int bad = 0;
        for (int j = 0; j < K; ++j) {
            const uint64_t idx = static_cast<uint64_t>(row) * static_cast<uint64_t>(K) + static_cast<uint64_t>(j);
            const uint64_t sv = prep_splitmix64(rng_seed ^ (idx * 0x9E3779B97F4A7C15ULL));
            bad += got[static_cast<size_t>(j)] != static_cast<int8_t>(static_cast<int>((sv >> 32) % 128u) - 64);
        }
        std::printf("[ocl-prep] GPU row %u check: %s\n", row, bad ? "MISMATCH" : "identical");
        std::fflush(stdout);
        if (bad) {
            return false;
        }
    }
    if (t_after_gen) {
        *t_after_gen = prep_timing_enabled() ? cp_now_sec() : 0.0;
    }

    /* The root stays on the device (d_merkle_roots_[0..31]) unless hash_a asks for a copy. */
    uint8_t *root_out = hash_a ? hash_a : (prep_check_enabled() ? root_local : nullptr);
    if (full_hash || !a_witness_valid_ || memcmp(a_hash_key_, job_key, 32) != 0) {
        /* First attempt or new job key (every subroot changes): hash the whole signal once. */
        if (!hash_signal_a_(raw_a, pad_a, job_key, root_out ? root_out : root_local)) {
            return false;
        }
        memcpy(a_hash_key_, job_key, 32);
    } else {
        const int group = static_cast<int>((static_cast<size_t>(row) * static_cast<size_t>(K) / 1024) / 256);
        if (!rehash_signal_a_group_(raw_a, pad_a, job_key, group, root_out)) {
            return false;
        }
    }
    if (prep_check_enabled()) {
        /* Recompute the whole commitment and compare roots (also refreshes the stored subroots). */
        uint8_t full[32];
        if (!hash_signal_a_(raw_a, pad_a, job_key, full)) {
            return false;
        }
        const bool same = memcmp(full, root_out, 32) == 0;
        std::printf("[ocl-prep] incremental commitment check (row %u): %s\n", row,
                    same ? "identical" : "MISMATCH");
        std::fflush(stdout);
        if (!same) {
            return false;
        }
    }
    return true;
}

bool Case33OclPrep::rehash_signal_a_group_(size_t raw_len, size_t pad_len, const uint8_t job_key[32],
                                           int group, uint8_t out[32]) {
    const int num_chunks = static_cast<int>(pad_len / 1024);
    const int num_subroots = (num_chunks + 255) / 256;
    if (num_chunks <= 0 || group < 0 || group >= num_subroots || !d_a_subroots_ || !d_merkle_roots_) {
        return false;
    }
    if (!set_job_key_(job_key)) {
        return false;
    }
    cl_ulong raw_len_ul = static_cast<cl_ulong>(raw_len);
    cl_ulong pad_len_ul = static_cast<cl_ulong>(pad_len);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_keyed_chunk_roots_, 0, sizeof(cl_mem), &d_A_sig_);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 1, sizeof(cl_ulong), &raw_len_ul);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 2, sizeof(cl_ulong), &pad_len_ul);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 3, sizeof(cl_mem), &d_job_key_);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 4, sizeof(cl_mem), &d_a_subroots_);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 5, sizeof(int), &num_chunks);
    err |= clSetKernelArg(k_keyed_chunk_roots_, 6, sizeof(int), &group);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = 256, l = 256;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_keyed_chunk_roots_, 1, nullptr, &g, &l, 0, nullptr,
                                 nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    /* The reduction works in place: run it on a copy of the stored subroots (a copy kernel on the
       compute queue, not a copy-engine transfer). */
    if (!ocl_->copy_buffer(d_a_subroots_, 0, d_merkle_roots_, 0, static_cast<size_t>(num_subroots) * 32)) {
        return false;
    }
    a_witness_valid_ = false;
    if (!merkle_finish_root_(num_subroots)) {
        return false;
    }
    if (out) {   /* the root stays in d_merkle_roots_[0..31]; copy it out only on request */
        if (!ocl_->read_buffer(d_merkle_roots_, out, 32)) {
            return false;
        }
        memcpy(a_root_, out, 32);
    }
    a_num_subroots_ = num_subroots;
    a_witness_valid_ = true;
    return true;
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
        /* keep the device root consistent with the multi-chunk path (A's noise seed reads it there) */
        return !d_merkle_roots_ || ocl_->write_buffer(d_merkle_roots_, out, 32);
    }

    if (!set_job_key_(job_key)) {
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
    const int group_base0 = 0;
    err |= clSetKernelArg(k_keyed_chunk_roots_, 6, sizeof(int), &group_base0);
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
        if (!ocl_->copy_buffer(d_merkle_roots_, 0, subroots_out, 0, static_cast<size_t>(num_subroots) * 32)) {
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
    if (!(const_sig_ ? d_dummy_ : d_A_sig_)) {
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
    const int sig_const = const_sig_;   /* 0: zero B */
    cl_mem dummy_sig = const_sig_ ? d_dummy_ : d_A_sig_;
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
    err |= clSetKernelArg(k_fused_prepack_b_, 10, sizeof(int), &sig_const);
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
    if (!ready_ || !a_buf || !ab_seed || !job_key || !b_noise_seed) {
        return false;
    }
    if (const_sig_) {
        const double t0 = prep_timing_enabled() ? cp_now_sec() : 0.0;
        if (!const_attempt_signal_(ab_seed, ab_seed_len, job_key, b_noise_seed, m, K, salted, a_key_out)) {
            return false;
        }
        const double t1 = prep_timing_enabled() ? cp_now_sec() : 0.0;
        const bool ok = fused_prepack_a(a_buf, m, K, blocks_k, macro_rows);
        if (prep_timing_enabled()) {
            std::printf("[ocl-prep] A constant signal: dummy row + commitment %.1fms, prepack %.1fms\n",
                        (t1 - t0) * 1e3, (cp_now_sec() - t1) * 1e3);
        }
        return ok;
    }
    if (!ensure_buffers(m, 1, K)) {
        return false;
    }

    if (!update_signal_a_(ab_seed, ab_seed_len, job_key, m, K, nullptr, nullptr)) {
        return false;
    }

    if (!derive_a_noise_seed_(b_noise_seed, m, salted, a_key_out)) {
        return false;
    }

    return fused_prepack_a(a_buf, m, K, blocks_k, macro_rows);
}

bool Case33OclPrep::prepare_attempt_a_rowmajor(cl_mem a_noisy_out, const uint8_t *ab_seed,
                                               int ab_seed_len, const uint8_t job_key[32],
                                               const uint8_t b_noise_seed[32], int m, int K,
                                               int out_lda, int salted, uint8_t a_key_out[32]) {
    if (!ready_ || !a_noisy_out || !ab_seed || !job_key || !b_noise_seed) {
        return false;
    }
    if (!ensure_buffers(m, 1, K)) {
        return false;
    }
    const double t_start = prep_timing_enabled() ? cp_now_sec() : 0.0;

    double t_gen = 0.0;
    if (!update_signal_a_(ab_seed, ab_seed_len, job_key, m, K, nullptr, &t_gen)) {
        return false;
    }
    const double t_hash = prep_timing_enabled() ? cp_now_sec() : 0.0;

    if (!derive_a_noise_seed_(b_noise_seed, m, salted, a_key_out)) {
        return false;
    }

    const bool ok = (a_pack_rows_ > 0) ? noisy_matrix_a_packed_(a_noisy_out, d_A_sig_, m, K, out_lda)
                                         : noisy_matrix_rowmajor_(a_noisy_out, d_A_sig_, m, K, out_lda, 0, 1);
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
    if (b_pack_cols_ > 0) {   // no fallback: the GEMM kernel was built for this layout
        return prepare_job_b_packed_(b_buf, n, K, ldb);
    }
    return noisy_matrix_colmajor_(b_buf, n, K, ldb, 1);
}

/* Per-job noisy B in gemmstone's packed-rows layout, written directly by ocl_noisy_matrix_b_packed:
   panels of b_pack_cols_ columns with b_pack_cp_ k per column (A770: 64 x 32, UHD 770: 32 x 4). No
   column-major scratch, so the job needs only the packed B buffer (CP_OCL_PREP_CHECK builds a
   column-major reference in scratch and compares). No fallback: the GEMM kernel was built for this
   layout. */
bool Case33OclPrep::prepare_job_b_packed_(cl_mem b_buf, int n, int K, int ldb) {
    const int Q = b_pack_cols_, cp = b_pack_cp_;
    if (!k_noisy_b_packed_ || Q <= 0 || Q > 64 || cp <= 0 || cp % 4 != 0 || 64 % cp != 0 || n % Q != 0 ||
        K % 64 != 0 || ldb < K || ldb % 4 != 0 || R_RANK != 128) {
        std::fprintf(stderr, "[ocl-prep] packed B: unsupported shape (n %d, K %d, ldb %d, %d x %d)\n", n, K,
                     ldb, Q, cp);
        return false;
    }
    const double t0 = prep_timing_enabled() ? cp_now_sec() : 0.0;
    if (!build_perm_pairs_(1, K) || !uniform_rows_(n, 1)) {
        return false;
    }
    const int panel_stride_dw = Q * (ldb / 4);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_noisy_b_packed_, 0, sizeof(cl_mem), &b_buf);
    err |= clSetKernelArg(k_noisy_b_packed_, 1, sizeof(cl_mem), &d_el_rows_);
    err |= clSetKernelArg(k_noisy_b_packed_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_noisy_b_packed_, 3, sizeof(int), &n);
    err |= clSetKernelArg(k_noisy_b_packed_, 4, sizeof(int), &K);
    err |= clSetKernelArg(k_noisy_b_packed_, 5, sizeof(int), &Q);
    err |= clSetKernelArg(k_noisy_b_packed_, 6, sizeof(int), &cp);
    err |= clSetKernelArg(k_noisy_b_packed_, 7, sizeof(int), &panel_stride_dw);
    const cl_uint sig_word = 0x01010101u * static_cast<uint8_t>(const_sig_);   /* 0: zero B */
    err |= clSetKernelArg(k_noisy_b_packed_, 8, sizeof(cl_uint), &sig_word);
    const size_t l = 256;
    const size_t g = static_cast<size_t>(n / Q) * l;
    if (err != CL_SUCCESS ||
        clEnqueueNDRangeKernel(ocl_->queue, k_noisy_b_packed_, 1, nullptr, &g, &l, 0, nullptr, nullptr) !=
                CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] ocl_noisy_matrix_b_packed launch failed\n");
        return false;
    }
    if (prep_timing_enabled()) {
        clFinish(ocl_->queue);
        std::printf("[ocl-prep] B packed (%d x %d) per job: %.1fms\n", Q, cp, (cp_now_sec() - t0) * 1e3);
    }
    if (!prep_check_enabled()) {
        return true;
    }
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(ldb);
    if (!ensure_noisy_scratch(bytes) || !noisy_matrix_colmajor_(d_noisy_scratch_, n, K, ldb, 1)) {
        return false;
    }
    std::vector<int8_t> packed(bytes), ref(bytes);
    if (!ocl_->read_buffer(b_buf, packed.data(), bytes) ||
        !ocl_->read_buffer(d_noisy_scratch_, ref.data(), bytes)) {
        return false;
    }
    size_t bad = 0;
    for (int c = 0; c < n; ++c) {
        const int8_t *pan = packed.data() + static_cast<size_t>(c / Q) * static_cast<size_t>(Q) *
                                                    static_cast<size_t>(ldb);
        const int8_t *col = ref.data() + static_cast<size_t>(c) * static_cast<size_t>(ldb);
        for (int k = 0; k < K; ++k) {
            if (pan[static_cast<size_t>(k / cp) * static_cast<size_t>(Q * cp) +
                    static_cast<size_t>(c % Q) * static_cast<size_t>(cp) + static_cast<size_t>(k % cp)] !=
                static_cast<int8_t>(col[k] + const_sig_)) {
                ++bad;
            }
        }
    }
    std::printf("[ocl-prep] packed B check (%dx%d): %s\n", K, n, bad == 0 ? "identical" : "MISMATCH");
    std::fflush(stdout);
    clReleaseMemObject(d_noisy_scratch_);   /* 512 MB at production size; check mode only */
    d_noisy_scratch_ = nullptr;
    noisy_scratch_cap_ = 0;
    return bad == 0;
}

bool Case33OclPrep::prepare_job_b_gpu(cl_mem b_buf, const uint8_t b_noise_seed[32], int n, int K,
                                      int ldb, bool b_row_major) {
    if (const_sig_ && (b_row_major || b_pack_cols_ <= 0)) {
        std::fprintf(stderr, "[ocl-prep] constant signal needs the packed B layout\n");
        return false;
    }
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
    if (const_sig_) {
        if (!a_row_major || a_pack_rows_ <= 0) {
            std::fprintf(stderr, "[ocl-prep] constant signal needs the packed A layout\n");
            return false;
        }
        return prepare_attempt_a_const_(a_buf, ab_seed, ab_seed_len, job_key, b_noise_seed, m, K, lda,
                                        salted, a_key_out);
    }
    if (a_row_major) {
        return prepare_attempt_a_rowmajor(a_buf, ab_seed, ab_seed_len, job_key, b_noise_seed, m, K,
                                          lda, salted, a_key_out);
    }
    if (!ready_ || !a_buf || !ab_seed || !job_key || !b_noise_seed) {
        return false;
    }
    if (!ensure_buffers(m, 1, K)) {
        return false;
    }

    if (!update_signal_a_(ab_seed, ab_seed_len, job_key, m, K, nullptr, nullptr)) {
        return false;
    }

    if (!derive_a_noise_seed_(b_noise_seed, m, salted, a_key_out)) {
        return false;
    }

    return noisy_matrix_a_colmajor_(a_buf, d_A_sig_, m, K, lda);
}

bool Case33OclPrep::fused_prepack_a_launch_(cl_kernel k, cl_mem a_buf, int m, int K,
                                            int blocks_k, int macro_rows) {
    int rank = R_RANK;
    const int micro_m = case32::kMacroM / case32::kMR;
    /* constant signal: no signal buffer is read (d_dummy_ only fills the argument) */
    cl_mem sig_buf = const_sig_ ? d_dummy_ : d_A_sig_;
    const cl_int sig_from_buf = const_sig_ ? 0 : 1;
    const cl_int sig_const = const_sig_;
    const cl_uint sig_word = 0x01010101u * static_cast<uint8_t>(const_sig_);
    cl_int err = CL_SUCCESS;
    size_t g2 = 0, l2 = 0;
    if (k == k_fused_prepack_a_wg_) {
        err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a_buf);
        err |= clSetKernelArg(k, 1, sizeof(cl_mem), &d_el_rows_);
        err |= clSetKernelArg(k, 2, sizeof(cl_mem), &d_pairs_);
        err |= clSetKernelArg(k, 3, sizeof(cl_mem), &sig_buf);
        err |= clSetKernelArg(k, 4, sizeof(int), &K);
        err |= clSetKernelArg(k, 5, sizeof(int), &blocks_k);
        err |= clSetKernelArg(k, 6, sizeof(cl_int), &sig_from_buf);
        err |= clSetKernelArg(k, 7, sizeof(cl_uint), &sig_word);
        l2 = 256;
        g2 = static_cast<size_t>(macro_rows) * static_cast<size_t>(blocks_k) *
             static_cast<size_t>(fpa_split_) * l2;
    } else {
        err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a_buf);
        err |= clSetKernelArg(k, 1, sizeof(cl_mem), &d_noise_seed_);
        err |= clSetKernelArg(k, 2, sizeof(cl_mem), &d_pairs_);
        err |= clSetKernelArg(k, 3, sizeof(cl_mem), &sig_buf);
        err |= clSetKernelArg(k, 4, sizeof(int), &m);
        err |= clSetKernelArg(k, 5, sizeof(int), &K);
        err |= clSetKernelArg(k, 6, sizeof(int), &rank);
        err |= clSetKernelArg(k, 7, sizeof(int), &blocks_k);
        err |= clSetKernelArg(k, 8, sizeof(int), &macro_rows);
        err |= clSetKernelArg(k, 9, sizeof(cl_int), &sig_from_buf);
        err |= clSetKernelArg(k, 10, sizeof(cl_int), &sig_const);
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
    if (!a_witness_valid_ || (!d_A_sig_ && !const_sig_) ||
        (num_blocks > 0 && (!block_idx || !blocks_out))) {
        return false;
    }
    if (const_sig_) {
        /* Rows below the dummy row are the constant; the dummy row (the last K bytes of sz_a) comes from
           d_dummy_; bytes past sz_a are zero padding. */
        const size_t K = static_cast<size_t>(const_a_k_);
        if (K == 0 || sz_a < K || !d_dummy_) {
            return false;
        }
        const size_t fill_end = sz_a - K;
        std::vector<uint8_t> dummy;
        for (int i = 0; i < num_blocks; ++i) {
            const size_t off = static_cast<size_t>(block_idx[i]) * block_bytes;
            uint8_t *dst = blocks_out + static_cast<size_t>(i) * block_bytes;
            for (size_t j = 0; j < block_bytes; ++j) {
                const size_t g = off + j;
                if (g < fill_end) {
                    dst[j] = static_cast<uint8_t>(const_sig_);
                } else if (g < sz_a) {
                    if (dummy.empty()) {
                        dummy.resize(K);
                        if (!ocl_->read_buffer(d_dummy_, dummy.data(), K)) {
                            return false;
                        }
                    }
                    dst[j] = dummy[g - fill_end];
                } else {
                    dst[j] = 0;
                }
            }
        }
        num_blocks = 0;   /* nothing to read from a signal A buffer below */
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
    if (d_a_root_) {   /* written by derive_a_noise_seed_ for the attempt being proven */
        if (!ocl_->read_buffer(d_a_root_, root_out, 32)) {
            return false;
        }
    } else {
        memcpy(root_out, a_root_, 32);
    }
    return true;
}

/* ---------------------------------------------------------------------------------------------
 * Constant-signal mining (set_const_signal): signal B is the constant everywhere, signal A is the
 * constant in its m GEMM rows plus one random dummy row (row m) per attempt. The commitments hash the
 * constant without a buffer (ocl_keyed_chunk_roots_fill); A's is incremental: the first attempt of a
 * job hashes all groups, later attempts only the dummy row's group (K % 1024 == 0 puts the dummy row in
 * groups of its own when m * K / 1024 is a multiple of 256, else it shares the last group).
 * --------------------------------------------------------------------------------------------- */
bool Case33OclPrep::fill_chunk_roots_(cl_mem tail, size_t raw_len, size_t fill_end, int group_first,
                                      cl_mem roots_out) {
    const size_t pad_len = (raw_len + 1023) / 1024 * 1024;
    const int num_chunks = static_cast<int>(pad_len / 1024);
    const int num_subroots = (num_chunks + 255) / 256;
    if (!k_keyed_chunk_roots_fill_ || !tail || !roots_out || fill_end % 1024 != 0 || fill_end > raw_len ||
        group_first < 0 || group_first >= num_subroots) {
        return false;
    }
    const cl_ulong raw_ul = static_cast<cl_ulong>(raw_len), pad_ul = static_cast<cl_ulong>(pad_len);
    const cl_ulong fill_ul = static_cast<cl_ulong>(fill_end);
    const cl_uint word = 0x01010101u * static_cast<uint8_t>(const_sig_);
    cl_kernel k = k_keyed_chunk_roots_fill_;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &tail);
    err |= clSetKernelArg(k, 1, sizeof(cl_ulong), &raw_ul);
    err |= clSetKernelArg(k, 2, sizeof(cl_ulong), &pad_ul);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &d_job_key_);
    err |= clSetKernelArg(k, 4, sizeof(cl_mem), &roots_out);
    err |= clSetKernelArg(k, 5, sizeof(int), &num_chunks);
    err |= clSetKernelArg(k, 6, sizeof(int), &group_first);
    err |= clSetKernelArg(k, 7, sizeof(cl_ulong), &fill_ul);
    err |= clSetKernelArg(k, 8, sizeof(cl_uint), &word);
    if (err != CL_SUCCESS) {
        return false;
    }
    const size_t g = static_cast<size_t>(num_subroots - group_first) * 256, l = 256;
    return clEnqueueNDRangeKernel(ocl_->queue, k, 1, nullptr, &g, &l, 0, nullptr, trace_ev_("chunk_roots"))
           == CL_SUCCESS;
}

bool Case33OclPrep::const_matrix_keyed_hash(int n, int K, const uint8_t job_key[32], uint8_t root[32],
                                            std::vector<uint8_t> *subroots) {
    if (!ready_ || !const_sig_ || n <= 0 || K <= 0 || !job_key || !root) {
        return false;
    }
    if (!ensure_buffers(m_cap_ > 0 ? m_cap_ : 1, n, K) || !set_job_key_(job_key) ||
        !const_hash_selftest_(job_key, K)) {
        return false;
    }
    const size_t raw = static_cast<size_t>(n) * static_cast<size_t>(K);
    const size_t fill_end = raw / 1024 * 1024;
    const int num_chunks = static_cast<int>((raw + 1023) / 1024);
    const int num_subroots = (num_chunks + 255) / 256;
    if (num_chunks <= 1 || fill_end != raw) {
        std::fprintf(stderr, "[ocl-prep] constant matrix hash: unsupported size %dx%d\n", n, K);
        return false;
    }
    if (!fill_chunk_roots_(d_dummy_, raw, fill_end, 0, d_merkle_roots_)) {
        return false;
    }
    if (subroots) {
        subroots->resize(static_cast<size_t>(num_subroots) * 32);
        if (!ocl_->read_buffer(d_merkle_roots_, subroots->data(), subroots->size())) {
            return false;
        }
    }
    if (!merkle_finish_root_(num_subroots)) {
        return false;
    }
    clFinish(ocl_->queue);
    return ocl_->read_buffer(d_merkle_roots_, root, 32);
}

bool Case33OclPrep::commit_const_a_(const uint8_t job_key[32], int m, int K) {
    const size_t raw = static_cast<size_t>(m + 1) * static_cast<size_t>(K);
    const size_t fill_end = static_cast<size_t>(m) * static_cast<size_t>(K);
    const int num_chunks = static_cast<int>((raw + 1023) / 1024);
    const int num_subroots = (num_chunks + 255) / 256;
    if (num_chunks <= 1 || fill_end % 1024 != 0) {
        std::fprintf(stderr, "[ocl-prep] constant signal: unsupported A size %dx%d\n", m + 1, K);
        return false;
    }
    if (!set_job_key_(job_key)) {
        return false;
    }
    const bool full = !const_a_init_ || const_a_m_ != m || const_a_k_ != K || !a_witness_valid_ ||
                      memcmp(a_hash_key_, job_key, 32) != 0;
    const int first = full ? 0 : static_cast<int>((fill_end / 1024) / 256);
    a_witness_valid_ = false;
    /* the stored sub-roots are updated in place; the reduction runs on a copy */
    if (!fill_chunk_roots_(d_dummy_, raw, fill_end, first, d_a_subroots_) ||
        !ocl_->copy_buffer(d_a_subroots_, 0, d_merkle_roots_, 0, static_cast<size_t>(num_subroots) * 32) ||
        !merkle_finish_root_(num_subroots)) {
        const_a_init_ = false;
        return false;
    }
    const_a_init_ = true;
    const_a_m_ = m;
    const_a_k_ = K;
    memcpy(a_hash_key_, job_key, 32);
    a_num_subroots_ = num_subroots;
    a_witness_valid_ = true;
    if (prep_check_enabled() && !full) {
        /* CP_OCL_PREP_CHECK=1: the incremental root against a full recomputation (into the reduction
           scratch only; the stored sub-roots stay). */
        uint8_t inc[32], all[32];
        if (!ocl_->read_buffer(d_merkle_roots_, inc, 32) ||
            !fill_chunk_roots_(d_dummy_, raw, fill_end, 0, d_merkle_roots_) ||
            !merkle_finish_root_(num_subroots) || !ocl_->read_buffer(d_merkle_roots_, all, 32)) {
            return false;
        }
        const bool same = memcmp(inc, all, 32) == 0;
        std::printf("[ocl-prep] constant-signal incremental commitment check: %s\n",
                    same ? "identical" : "MISMATCH");
        std::fflush(stdout);
        if (!same) {
            return false;
        }
    }
    return true;
}

bool Case33OclPrep::const_hash_selftest_(const uint8_t job_key[32], int K) {
    if (const_selftest_ != 0) {
        return const_selftest_ > 0;
    }
    const_selftest_ = -1;
    /* three full groups of the constant, then one row of random bytes from d_dummy_: the fill path, the
       tail path and a short last group, against the host digest of the same bytes */
    const size_t fill_end = 3u * 256u * 1024u;
    const size_t raw = fill_end + static_cast<size_t>(K);
    const int num_subroots = static_cast<int>(((raw + 1023) / 1024 + 255) / 256);
    if (!d_dummy_ || dummy_cap_ < static_cast<size_t>(K) || merkle_cap_ < static_cast<size_t>(num_subroots) * 32 ||
        !set_job_key_(job_key) || !gen_random_(d_dummy_, 0x5EEDC0DE64ull, K)) {
        std::fprintf(stderr, "[ocl-prep] constant-signal hash self-test: setup failed\n");
        return false;
    }
    uint8_t dev[32], host[32];
    if (!fill_chunk_roots_(d_dummy_, raw, fill_end, 0, d_merkle_roots_) ||
        !merkle_finish_root_(num_subroots) || clFinish(ocl_->queue) != CL_SUCCESS ||
        !ocl_->read_buffer(d_merkle_roots_, dev, 32)) {
        std::fprintf(stderr, "[ocl-prep] constant-signal hash self-test: kernel failed\n");
        return false;
    }
    std::vector<uint8_t> bytes((raw + 1023) / 1024 * 1024, 0);
    memset(bytes.data(), const_sig_, fill_end);
    if (!ocl_->read_buffer(d_dummy_, bytes.data() + fill_end, static_cast<size_t>(K))) {
        return false;
    }
    pearl_keyed_matrix_digest(bytes.data(), bytes.size(), job_key, host);
    const bool same = memcmp(dev, host, 32) == 0;
    std::printf("[ocl-prep] constant-signal hash self-test: %s\n", same ? "identical" : "MISMATCH");
    std::fflush(stdout);
    const_selftest_ = same ? 1 : -1;
    const_a_init_ = false;   /* d_merkle_roots_ was used as scratch */
    return same;
}

bool Case33OclPrep::const_attempt_signal_(const uint8_t *ab_seed, int ab_seed_len,
                                          const uint8_t job_key[32], const uint8_t b_noise_seed[32],
                                          int m, int K, int salted, uint8_t a_key_out[32]) {
    if (!ready_ || !const_sig_ || !ab_seed || !job_key || !b_noise_seed || m <= 0 || K <= 0) {
        return false;
    }
    if (!ensure_buffers(m, 1, K) || !const_hash_selftest_(job_key, K)) {
        return false;
    }
    /* The dummy row: K signal bytes in [-64, 63] from the attempt seed (ocl_gen_random_matrix, tag 0). */
    const uint64_t rng_seed = pearl_seed_to_u64(ab_seed, ab_seed_len);
    if (!gen_random_(d_dummy_, rng_seed, K)) {
        std::fprintf(stderr, "[ocl-prep] dummy row generation failed\n");
        return false;
    }
    if (!commit_const_a_(job_key, m, K)) {
        return false;
    }
    /* the protocol matrix has m + 1 rows */
    return derive_a_noise_seed_(b_noise_seed, m + 1, salted, a_key_out);
}

bool Case33OclPrep::prepare_attempt_a_const_(cl_mem a_buf, const uint8_t *ab_seed, int ab_seed_len,
                                             const uint8_t job_key[32], const uint8_t b_noise_seed[32],
                                             int m, int K, int lda, int salted, uint8_t a_key_out[32]) {
    if (!a_buf) {
        return false;
    }
    const double t0 = (prep_timing_enabled() || prep_trace_enabled()) ? cp_now_sec() : 0.0;
    trace_attempt_ = prep_trace_enabled();
    attempt_nofinish_ = attempt_async_;
    if (attempt_done_ev_) {   // the previous attempt's prep finished long ago (its scan followed it)
        clWaitForEvents(1, &attempt_done_ev_);
        clReleaseEvent(attempt_done_ev_);
        attempt_done_ev_ = nullptr;
    }
    attempt_done_sec_.store(0.0, std::memory_order_release);
    if (!const_attempt_signal_(ab_seed, ab_seed_len, job_key, b_noise_seed, m, K, salted, a_key_out)) {
        trace_attempt_ = false;
        attempt_nofinish_ = false;
        return false;
    }
    const double t1 = (prep_timing_enabled() || prep_trace_enabled()) ? cp_now_sec() : 0.0;
    const bool ok = noisy_matrix_a_packed_(a_buf, d_dummy_, m, K, lda);
    const bool async = attempt_nofinish_;
    attempt_nofinish_ = false;
    if (trace_attempt_) {
        trace_attempt_ = false;
        // noise_done = the host returns (async: the kernels are only enqueued; their events print at the
        // first panel of the scan, see flush_trace).
        std::fprintf(stderr, "[prep-host] start=%.6f signal_done=%.6f noise_done=%.6f\n", t0, t1, cp_now_sec());
        if (!async) {
            flush_trace();
        }
    }
    if (prep_timing_enabled()) {
        clFinish(ocl_->queue);
        std::printf("[ocl-prep] A constant signal: dummy row + commitment %.1fms, noise %.1fms\n",
                    (t1 - t0) * 1e3, (cp_now_sec() - t1) * 1e3);
    }
    return ok;
}

cl_event *Case33OclPrep::trace_ev_(const char *name) {
    if (!trace_attempt_) {
        return nullptr;
    }
    trace_events_.emplace_back(name, nullptr);
    return &trace_events_.back().second;
}

void Case33OclPrep::flush_trace() {
    for (auto &e : trace_events_) {
        if (!e.second) {
            continue;
        }
        clWaitForEvents(1, &e.second);   // not clFinish: that would also wait for queued panels
        cl_ulong st = 0, en = 0;
        clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_START, sizeof(st), &st, nullptr);
        clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr);
        std::fprintf(stderr, "[prep-trace] %s gpu_start=%llu gpu_end=%llu\n", e.first,
                     static_cast<unsigned long long>(st), static_cast<unsigned long long>(en));
        clReleaseEvent(e.second);
    }
    trace_events_.clear();
    std::fflush(stderr);
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
    cl_mem d_out = ocl_->alloc_buffer(32, CL_MEM_READ_WRITE);
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
