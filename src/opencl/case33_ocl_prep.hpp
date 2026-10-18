#pragma once

#include "opencl_context.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

struct Case33OclPrep {
    Case33OclPrep() = default;
    ~Case33OclPrep();

    bool init(OpenClContext *ocl, const std::string &kernel_dir, bool fused_prepack = true);
    bool ensure_buffers(int m, int n, int k);
    bool ensure_noisy_scratch(size_t bytes);

    bool prepare_job_b(cl_mem b_buf, const uint8_t b_noise_seed[32], int n, int K, int blocks_k,
                       int macro_cols);

    bool prepare_attempt_a(cl_mem a_buf, const uint8_t *ab_seed, int ab_seed_len,
                           const uint8_t job_key[32], const uint8_t b_noise_seed[32], int m,
                           int K, int blocks_k, int macro_rows, int salted,
                           uint8_t a_key_out[32]);

    /* Keyed digest of an all-zero n x K matrix (zero-B mining) on the GPU. */
    bool zero_matrix_keyed_hash(int n, int K, const uint8_t job_key[32], uint8_t out[32]);

    /* GPU random A + keyed hash + row-major noisy A (signal from d_A_sig_). */
    bool prepare_attempt_a_rowmajor(cl_mem a_noisy_out, const uint8_t *ab_seed, int ab_seed_len,
                                    const uint8_t job_key[32], const uint8_t b_noise_seed[32],
                                    int m, int K, int out_lda, int salted, uint8_t a_key_out[32]);

    /* Row-major B^T noisy matrix (zero signal). */
    bool prepare_job_b_rowmajor(cl_mem b_noisy_out, const uint8_t b_noise_seed[32], int n, int K);

    /* gemmstone column-major noisy B (zero signal) directly into GEMM buffer. */
    bool prepare_job_b_colmajor(cl_mem b_buf, const uint8_t b_noise_seed[32], int n, int K,
                                int ldb);

    /* Layout-dispatching GPU prep for gemmstone device buffers. */
    bool prepare_job_b_gpu(cl_mem b_buf, const uint8_t b_noise_seed[32], int n, int K, int ldb,
                           bool b_row_major);
    bool prepare_attempt_a_gpu(cl_mem a_buf, const uint8_t *ab_seed, int ab_seed_len,
                               const uint8_t job_key[32], const uint8_t b_noise_seed[32], int m,
                               int K, int lda, bool a_row_major, int salted,
                               uint8_t a_key_out[32]);

    /* Noise perm pairs + fused coalesced A prepack (d_noise_seed_ and d_A_sig_ must be set). */
    bool fused_prepack_a(cl_mem a_buf, int m, int K, int blocks_k, int macro_rows);

    bool write_noise_seed(const uint8_t seed[32]);

    bool read_A_sig(int8_t *h_A_sig, size_t bytes) const;

    /* Share witness for the last GPU-hashed signal A: sub-roots captured during the keyed hash
     * (a_witness_subroots() of them; 0 when A fits in one 256-chunk subtree, -1 when no GPU hash
     * has run) plus the requested blocks, zero-padded past sz_a. */
    int a_witness_subroots() const { return a_witness_valid_ ? a_num_subroots_ : -1; }
    bool read_A_witness(size_t sz_a, const uint32_t *block_idx, int num_blocks,
                        size_t block_bytes, uint8_t *blocks_out, uint8_t *subroots_out,
                        uint8_t root_out[32]) const;

    cl_mem noisy_scratch_buffer() const { return d_noisy_scratch_; }

    /* OpenCL --align-test helpers (GPU vs CPU reference). */
    bool align_gen_random(cl_mem dst, uint64_t rng_seed, int matrix_tag, int total_elems);
    bool align_keyed_hash(cl_mem d_mat, size_t raw_len, size_t pad_len,
                          const uint8_t job_key[32], uint8_t out[32]);
    bool align_run_get_random_hash(int index, const uint8_t seed[32], int is_b,
                                   int prepend_index, uint8_t out[32]);
    bool align_build_perm_pairs(int is_b, const uint8_t noise_seed[32], int k);
    bool align_read_perm_pairs(uint32_t *out, size_t word_count);

    cl_mem signal_A_buffer() const { return d_A_sig_; }

    bool ready() const { return ready_; }

private:
    bool build_program_(const std::string &kernel_dir);
    bool create_kernels_();
    void release_kernels_();
    bool matrix_keyed_hash_(cl_mem d_mat, size_t raw_len, size_t pad_len,
                            const uint8_t job_key[32], uint8_t out[32],
                            cl_mem subroots_out = nullptr, int *num_subroots_out = nullptr);
    bool hash_signal_a_(size_t raw_len, size_t pad_len, const uint8_t job_key[32],
                        uint8_t out[32]);
    bool merkle_finish_root_(int num_subroots);
    bool build_perm_pairs_(int is_b, int K);
    bool gen_random_a_(uint64_t rng_seed, int total); /* signal A into d_A_sig_ */
    bool noisy_matrix_rowmajor_(cl_mem out, cl_mem signal, int rows, int K, int out_lda, int is_b,
                                int has_signal);
    /* Work-group row-major path; false when unavailable so the caller falls back. */
    bool noisy_matrix_rowmajor_wg_(cl_mem out, cl_mem signal, int rows, int K, int out_lda,
                                   int is_b, int has_signal);
    bool noisy_matrix_rowmajor_check_(cl_mem out, cl_mem signal, int rows, int K, int out_lda,
                                      int is_b, int has_signal);
    bool noisy_matrix_a_colmajor_(cl_mem out, cl_mem signal, int rows, int K, int lda);
    bool noisy_matrix_colmajor_(cl_mem out, int cols, int K, int ldb, int is_b);
    bool noisy_matrix_b_rowmajor_(cl_mem out, int K, int N, int ldb);
    bool pad_rowmajor_row_tails_(cl_mem out, int rows, int valid_cols, int ldb);
    bool pad_colmajor_col_tails_(cl_mem out, int cols, int valid_rows, int lda);

    OpenClContext *ocl_ = nullptr;
    cl_program program_ = nullptr;
    bool ready_ = false;
    bool fused_prepack_ = true;

    int m_cap_ = 0;
    int n_cap_ = 0;
    size_t merkle_cap_ = 0;

    cl_kernel k_gen_random_ = nullptr;
    cl_kernel k_gen_random16_ = nullptr; // optional: 16 elements per work item
    cl_kernel k_build_pairs_ = nullptr;
    cl_kernel k_keyed_chunk_roots_ = nullptr;
    cl_kernel k_compute_blake_mt_ = nullptr;
    cl_kernel k_reduce_roots_ = nullptr;
    cl_kernel k_fused_prepack_a_ = nullptr;
    cl_kernel k_fused_prepack_a_wg_ = nullptr; // optional: 256-wide blocks, needs el rows
    int fpa_split_ = 1; // work groups per block in k_fused_prepack_a_wg_ (local memory)
    cl_kernel k_fused_prepack_b_ = nullptr;
    cl_kernel k_noisy_rowmajor_ = nullptr;
    cl_kernel k_uniform_rows_ = nullptr;      // optional: work-group row-major noise path
    cl_kernel k_noisy_rowmajor_wg_ = nullptr; // optional, with k_uniform_rows_
    cl_kernel k_noisy_a_colmajor_ = nullptr;
    cl_kernel k_noisy_colmajor_ = nullptr;
    cl_kernel k_noisy_b_rowmajor_ = nullptr;
    cl_kernel k_pad_row_tails_ = nullptr;
    cl_kernel k_pad_col_tails_ = nullptr;
    cl_kernel k_test_random_hash_ = nullptr;

    cl_mem d_A_sig_ = nullptr;
    cl_mem d_job_key_ = nullptr;
    cl_mem d_noise_seed_ = nullptr;
    cl_mem d_pairs_ = nullptr;
    cl_mem d_merkle_roots_ = nullptr;
    cl_mem d_a_subroots_ = nullptr;
    int a_num_subroots_ = 0;
    uint8_t a_root_[32] = {};
    bool a_witness_valid_ = false;
    cl_mem d_noisy_scratch_ = nullptr;
    size_t noisy_scratch_cap_ = 0;
    cl_mem d_el_rows_ = nullptr; // rows x R_RANK uniform noise for the work-group path
    size_t el_rows_cap_ = 0;
    bool ensure_el_rows_(size_t bytes);
    bool uniform_rows_(int rows, int is_b);
    bool fused_prepack_a_launch_(cl_kernel k, cl_mem a_buf, int m, int K, int blocks_k,
                                 int macro_rows);
};

std::string cp_ocl_kernel_dir();
