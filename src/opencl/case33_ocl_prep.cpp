#include "case33_ocl_prep.hpp"

#include "case32_layout.hpp"
#include "cp_config.h"
#include "cp_noise.h"

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
    opts += " -DCOLS_PER_GROUP=" + std::to_string(case32::kColsPerGroup);
    opts += " -DR_RANK=" + std::to_string(R_RANK);

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
    k_build_pairs_ = mk("ocl_build_perm_pairs");
    k_keyed_chunk_roots_ = mk("ocl_keyed_chunk_roots");
    k_compute_blake_mt_ = mk("ocl_compute_blake_mt");
    k_reduce_roots_ = mk("ocl_reduce_roots");
    k_fused_prepack_a_ = mk("ocl_fused_prepack_a");
    k_fused_prepack_b_ = mk("ocl_fused_prepack_b");
    k_test_random_hash_ = mk("ocl_test_get_random_hash");

    return k_gen_random_ && k_build_pairs_ && k_keyed_chunk_roots_ && k_compute_blake_mt_ &&
           k_reduce_roots_ && k_fused_prepack_a_ && k_fused_prepack_b_ && k_test_random_hash_;
}

void Case33OclPrep::release_kernels_() {
    if (k_gen_random_) {
        clReleaseKernel(k_gen_random_);
        k_gen_random_ = nullptr;
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
    if (k_fused_prepack_b_) {
        clReleaseKernel(k_fused_prepack_b_);
        k_fused_prepack_b_ = nullptr;
    }
    if (k_test_random_hash_) {
        clReleaseKernel(k_test_random_hash_);
        k_test_random_hash_ = nullptr;
    }
}

bool Case33OclPrep::init(OpenClContext *ocl, const std::string &kernel_dir) {
    ready_ = false;
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
        d_merkle_roots_ = ocl_->alloc_buffer(merkle_need, CL_MEM_READ_WRITE);
        merkle_cap_ = merkle_need;
    }
    n_cap_ = n;
    return d_A_sig_ && d_merkle_roots_;
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

bool Case33OclPrep::matrix_keyed_hash_(cl_mem d_mat, size_t raw_len, size_t pad_len,
                                       const uint8_t job_key[32], uint8_t out[32]) {
    const int num_chunks = static_cast<int>(pad_len / 1024);
    if (num_chunks <= 0) {
        return false;
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

    const uint64_t rng_seed = pearl_seed_to_u64(ab_seed, ab_seed_len);
    const int total_a = m * K;
    const int matrix_tag = 0;
    cl_ulong rng_ul = static_cast<cl_ulong>(rng_seed);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k_gen_random_, 0, sizeof(cl_ulong), &rng_ul);
    err |= clSetKernelArg(k_gen_random_, 1, sizeof(int), &matrix_tag);
    err |= clSetKernelArg(k_gen_random_, 2, sizeof(int), &total_a);
    err |= clSetKernelArg(k_gen_random_, 3, sizeof(cl_mem), &d_A_sig_);
    if (err != CL_SUCCESS) {
        return false;
    }
    {
        const size_t g = static_cast<size_t>((total_a + 255) / 256) * 256;
        const size_t l = 256;
        err = clEnqueueNDRangeKernel(ocl_->queue, k_gen_random_, 1, nullptr, &g, &l, 0, nullptr,
                                     nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }
    }
    clFinish(ocl_->queue);

    const size_t raw_a = static_cast<size_t>(m) * static_cast<size_t>(K);
    const size_t pad_a = (raw_a + 1023) / 1024 * 1024;
    uint8_t hash_a[32];
    if (!matrix_keyed_hash_(d_A_sig_, raw_a, pad_a, job_key, hash_a)) {
        return false;
    }

    pearl_a_noise_seed_from_hash(b_noise_seed, hash_a, static_cast<uint32_t>(m), salted,
                                 a_key_out);

    if (!ocl_->write_buffer(d_noise_seed_, a_key_out, 32)) {
        return false;
    }

    return fused_prepack_a(a_buf, m, K, blocks_k, macro_rows);
}

bool Case33OclPrep::fused_prepack_a(cl_mem a_buf, int m, int K, int blocks_k, int macro_rows) {
    if (!ready_ || !a_buf || m <= 0 || K <= 0 || blocks_k <= 0 || macro_rows <= 0) {
        return false;
    }

    const int is_b = 0;
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

    err = CL_SUCCESS;
    err |= clSetKernelArg(k_fused_prepack_a_, 0, sizeof(cl_mem), &a_buf);
    err |= clSetKernelArg(k_fused_prepack_a_, 1, sizeof(cl_mem), &d_noise_seed_);
    err |= clSetKernelArg(k_fused_prepack_a_, 2, sizeof(cl_mem), &d_pairs_);
    err |= clSetKernelArg(k_fused_prepack_a_, 3, sizeof(cl_mem), &d_A_sig_);
    err |= clSetKernelArg(k_fused_prepack_a_, 4, sizeof(int), &m);
    err |= clSetKernelArg(k_fused_prepack_a_, 5, sizeof(int), &K);
    err |= clSetKernelArg(k_fused_prepack_a_, 6, sizeof(int), &rank);
    err |= clSetKernelArg(k_fused_prepack_a_, 7, sizeof(int), &blocks_k);
    err |= clSetKernelArg(k_fused_prepack_a_, 8, sizeof(int), &macro_rows);
    if (err != CL_SUCCESS) {
        return false;
    }

    const int micro_m = case32::kMacroM / case32::kMR;
    const size_t num_groups = static_cast<size_t>(macro_rows) * static_cast<size_t>(blocks_k) *
                              static_cast<size_t>(micro_m);
    const size_t l2 = case32::kMR;
    const size_t g2 = num_groups * l2;
    err = clEnqueueNDRangeKernel(ocl_->queue, k_fused_prepack_a_, 1, nullptr, &g2, &l2, 0,
                                 nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[ocl-prep] fused_prepack_a launch failed\n");
        return false;
    }
    clFinish(ocl_->queue);
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
