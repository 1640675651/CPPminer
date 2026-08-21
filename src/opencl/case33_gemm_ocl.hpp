#pragma once



#include "case32_gemm_ocl.hpp"

#include "case33_ocl_prep.hpp"
#include "cp_config.h"

#include "opencl_context.hpp"



#include <cstdint>

#include <functional>

#include <string>

#include <vector>



struct Case33GemmOcl {

    Case33GemmOcl() = default;

    ~Case33GemmOcl();



    void set_dpi_mode(Case32OclDpiMode mode) { dpi_mode_ = mode; }

    void set_macro_batch(int batch);

    int macro_batch() const { return macro_batch_; }

    /* 0 = auto (DPI then cpm; beignet-fix), 1 = broadcast/cpm, 2 = packed per-C dot4. */
    void set_issue_mode(int mode);
    /* Legacy: on → broadcast (1), off → auto (0). */
    void set_issue_broadcast(int on);
    /* Broadcast/cpm element type: 0 = float4 mad (default), 1 = int32. Ignored if packed. */
    void set_cpm_int(int on);
    /* Stage A/B panels in __local (CASE32_USE_LDS). Default off. */
    void set_use_lds(int on);
    bool use_lds() const { return use_lds_; }



    bool init_context(const char *kernel_cl_path, int device_index = 0,
                      int platform_filter = -1, bool gpu_prep = true);

    bool prepare_job(int M, int N, int K, const int8_t *b_colmajor);

    /* GPU prep path: noise + coalesced prepack directly into device GEMM buffers. */
    bool prepare_job_gpu(int M, int N, int K, const uint8_t b_noise_seed[32]);
    bool prepare_attempt_gpu(const uint8_t *ab_seed, int ab_seed_len,
                             const uint8_t job_key[32], const uint8_t b_noise_seed[32],
                             int salted, uint8_t a_key_out[32]);
    bool read_A_sig(int8_t *h_A_sig);

    bool prepare_attempt_a(const int8_t *a_rowmajor);

    bool available() const { return available_; }



    /* Device-side jackpot scan: one batched kernel launch per macro batch; host readback is

     * found_flag (+ t_rows/t_cols only on hit). */

    bool scan_for_share(const uint32_t a_key8[8], const uint32_t bound[8], int *out_found,
                        int *out_t_rows, int *out_t_cols, uint64_t *out_tiles_scanned,
                        const std::function<bool()> &should_cancel = {},
                        const std::function<void(uint64_t)> &on_progress = {});



    const char *backend() const { return backend_; }

    const char *device_name() const { return device_name_.c_str(); }

    const char *platform_name() const { return platform_name_.c_str(); }

    int device_index() const { return device_flat_index_; }

    bool discrete_gpu() const { return discrete_gpu_; }

    const char *dpi_status() const { return dpi_status_; }

    size_t max_work_group_size() const { return ocl_.max_work_group_size; }



private:

    bool build_kernel_(const char *kernel_cl_path);

    bool setup_dims_(int M, int N, int K);

    bool ensure_jackpot_bufs_();

    bool run_macro_batch_(int mb_begin, int batch_count);



    bool context_ready_ = false;

    bool available_ = false;

    OpenClContext ocl_;



    int M_ = 0;

    int N_ = 0;

    int K_ = 0;

    int milestone_k_ = 0;

    int blocks_k_ = 0;

    int blocks_per_milestone_ = 0;

    int num_milestones_ = 0;

    int macro_rows_ = 0;

    int macro_cols_ = 0;

    int tile_cols_ = 0;

    size_t tile_count_ = 0;

    int macro_blocks_ = 0;

    int macro_batch_ = CP_MACRO_BATCH_DEFAULT;

    Case32OclDpiMode dpi_mode_ = Case32OclDpiMode::Builtin;

    int issue_mode_ = 0; /* 0=auto, 1=broadcast/cpm, 2=packed */

    bool use_cpm_int_ = false;

    bool use_lds_ = false;



    bool using_integer_dot_ = false;

    bool using_asm_dot_ = false;

    bool using_builtin_dot_ = false;

    bool using_cpm_ = false;



    cl_kernel kernel_ = nullptr;

    cl_mem a_buf_ = nullptr;

    cl_mem b_buf_ = nullptr;

    cl_mem dummy_buf_ = nullptr;

    cl_mem a_key_buf_ = nullptr;

    cl_mem bound_buf_ = nullptr;

    cl_mem found_buf_ = nullptr;

    cl_mem out_rows_buf_ = nullptr;

    cl_mem out_cols_buf_ = nullptr;



    std::vector<int8_t> a_pre_host_;

    std::vector<int8_t> b_pre_host_;

    Case33OclPrep prep_;

    std::string device_name_;

    std::string platform_name_;

    int device_flat_index_ = -1;

    bool discrete_gpu_ = false;

    char backend_[192] = {};

    char dpi_status_[128] = {};

};



std::string cp_ocl_resolve_kernel_path();


