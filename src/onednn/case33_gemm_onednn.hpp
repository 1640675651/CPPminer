#pragma once

#include "case5_ngen_gemm.hpp"
#include "case33_ocl_prep.hpp"
#include "cp_config.h"
#include "opencl_context.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/* Case 3.3 Intel GPU: gemmstone / oneDNN catalog int8 GEMM + milestoned tile XOR + GPU jackpot. */
struct Case33GemmOnednn {
    Case33GemmOnednn() = default;
    ~Case33GemmOnednn();

    void set_row_period_batch(int batch);
    void set_col_period_batch(int batch);
    void set_fused_jackpot(bool fused);
    int row_period_batch() const { return row_period_batch_; }
    int col_period_batch() const { return col_period_batch_; }

    bool init_context(int device_index = 0, int platform_filter = -1);
    bool prepare_job(int M, int N, int K, const int8_t *b_rowmajor);
    bool prepare_job_gpu(int M, int N, int K, const uint8_t b_noise_seed[32]);
    bool prepare_attempt_a(const int8_t *a_rowmajor);
    bool prepare_attempt_gpu(const uint8_t *ab_seed, int ab_seed_len, const uint8_t job_key[32],
                             const uint8_t b_noise_seed[32], int salted, uint8_t a_key_out[32]);
    bool read_A_sig(int8_t *h_A_sig);

    bool available() const { return available_; }

    bool scan_for_share(const uint32_t a_key8[8], const uint32_t bound[8], int *out_found,
                        int *out_t_rows, int *out_t_cols, uint64_t *out_tiles_scanned,
                        const std::function<bool()> &should_cancel = {},
                        const std::function<void(uint64_t)> &on_progress = {});

    const char *backend() const { return backend_; }
    const char *device_name() const { return device_name_.c_str(); }
    const char *platform_name() const { return platform_name_.c_str(); }
    int device_index() const { return device_flat_index_; }
    const case5_ngen::DriverInfo &driver_info() const { return info_; }
    bool gpu_prep_ready() const { return prep_ready_; }

private:
    bool setup_dims_(int M, int N, int K);
    void compute_tile_grid_(int m, int n, int &out_tile_rows, int &out_tile_cols,
                            int &out_tile_count) const;
    bool ensure_matrix_bufs_();
    bool ensure_panel_tile_xor_buf_(int panel_tile_count);
    bool build_jackpot_kernel_();
    bool ensure_jackpot_bufs_();
    bool upload_a_rowmajor_(const int8_t *a_rowmajor);
    bool upload_b_colmajor_(const int8_t *b_rowmajor);
    bool run_gemm_panel_(int m_panel, int n_panel, int64_t offset_a_rows, int64_t offset_b_cols,
                         int panel_tile_count, int panel_tile_cols, bool finish_queue);
    bool run_gpu_jackpot_panel_(int panel_tile_count, int panel_tile_cols, int tr_base,
                                int tc_base, int *out_found, bool finish_queue);
    bool run_gemm_jackpot_panel_(int m_panel, int n_panel, int64_t offset_a_rows,
                                 int64_t offset_b_cols, int panel_tile_count, int panel_tile_cols,
                                 int tr_base, int tc_base, int *out_found);
    bool scan_tile_xor_panel_host_(const uint32_t a_key8[8], const uint32_t bound[8],
                                   int panel_tile_rows, int panel_tile_cols, int panel_tile_count,
                                   int tr_base, int tc_base, int *out_found, int *out_t_rows,
                                   int *out_t_cols, uint64_t *out_tiles_scanned,
                                   const std::function<bool()> &should_cancel,
                                   const std::function<void(uint64_t)> &on_progress);

    bool context_ready_ = false;
    bool available_ = false;
    bool prep_ready_ = false;
    OpenClContext ocl_;
    Case33OclPrep prep_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
    int lda_ = 0;
    int ldb_ = 0;
    int ldc_ = 0;

    int tile_cols_ = 0;
    int tile_rows_ = 0;
    int tile_count_ = 0;
    int num_milestones_ = 0;
    int folded_msg_words_ = 0;
    int milestone_k_ = 128;
    bool fused_jackpot_ = false;
    int xor_period_ = 1;
    int row_period_batch_ = CP_ROW_PERIOD_BATCH_DEFAULT;
    int col_period_batch_ = CP_PERIOD_BATCH_DEFAULT;
    int hash_tile_rows_ = 0;
    int hash_tile_cols_ = 0;

    case5_ngen::DriverInfo info_;
    cl_kernel kernel_ = nullptr;
    cl_mem a_buf_ = nullptr;
    cl_mem b_buf_ = nullptr;
    cl_mem c_buf_ = nullptr;
    cl_mem tile_xor_buf_ = nullptr;
    cl_program jackpot_program_ = nullptr;
    cl_kernel jackpot_kernel_ = nullptr;
    cl_mem a_key_buf_ = nullptr;
    cl_mem bound_buf_ = nullptr;
    cl_mem found_buf_ = nullptr;
    cl_mem out_rows_buf_ = nullptr;
    cl_mem out_cols_buf_ = nullptr;

    size_t a_buf_bytes_ = 0;
    size_t b_buf_bytes_ = 0;
    int panel_tile_xor_cap_ = 0;
    bool jackpot_ready_ = false;

    std::vector<uint32_t> tile_xor_host_;
    std::vector<int8_t> a_host_;
    std::vector<int8_t> b_host_;
    std::vector<int8_t> pack_scratch_;
    std::string device_name_;
    std::string platform_name_;
    int device_flat_index_ = -1;
    char backend_[320] = {};
};
