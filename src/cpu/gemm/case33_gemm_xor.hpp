#pragma once

#include "case32_gemm.hpp"

#include <cstdint>
#include <functional>
#include <vector>

enum class Case33PrepackMode {
    Separate,     /* row-major noisy + persistent a_pre_/b_pre_ */
    ReuseScanBuf, /* row-major noisy + prepack swap into scan buffer */
    Fused,        /* noise injection directly into scan/prepack layout */
};

// Case 3.3: Case 3.2 8x16 AVX2 ukernel + milestoned 8x16 tile XOR.
// tile_xor[milestone * tile_count + spatial_tile_id]
struct Case33GemmXor {
    static constexpr int kNumMilestones = 16;
    static constexpr int kMR = Case32Gemm::kMR;
    static constexpr int kNR = Case32Gemm::kNR;
    static constexpr int kKR = Case32Gemm::kKR;
    static constexpr int kTileRows = kMR;
    static constexpr int kTileCols = kNR;
    static constexpr int kMacroM = Case32Gemm::kMacroM;
    static constexpr int kMacroN = Case32Gemm::kMacroN;

    void set_int8_mode(Case32Int8Mode mode) { int8_mode_ = mode; }
    void set_prepack_mode(Case33PrepackMode mode) { prepack_mode_ = mode; }
    Case33PrepackMode prepack_mode() const { return prepack_mode_; }
    void set_inplace_prepack(bool on) {
        prepack_mode_ = on ? Case33PrepackMode::ReuseScanBuf : Case33PrepackMode::Separate;
    }
    bool inplace_prepack() const {
        return prepack_mode_ != Case33PrepackMode::Separate;
    }

    bool init(int M, int N, int K, const int8_t *a, const int8_t *b);
    /* Zero-B CPU: B once per job, A each attempt. */
    bool prepare_job_b(int M, int N, int K, std::vector<int8_t> *b_buf,
                       const int8_t *b_signal, const uint8_t *b_noise_seed, int rank);
    bool prepare_attempt_a(std::vector<int8_t> *a_buf, const int8_t *a_signal,
                           const uint8_t *a_noise_seed, int rank);
    void reset();

    bool available() const { return available_; }

    void run();
    void run_gemm_only();

    bool scan_tiles(const std::function<bool(const uint32_t *milestone_xor, int t_rows,
                                             int t_cols)> &on_tile,
                    const std::function<bool()> &should_cancel = {});

    const std::vector<uint32_t> &tile_xor() const { return tile_xor_; }
    const char *backend() const { return backend_; }
    int num_threads() const { return num_threads_; }
    size_t tile_count() const { return tile_count_; }
    int tile_cols() const { return tile_cols_; }

private:
    bool setup_dims_(int M, int N, int K);
    void update_backend_label_();
    bool fused_noisy_prepack_a_(const int8_t *a_signal, const uint8_t *a_noise_seed,
                                int rank, std::vector<int8_t> *scan);
    bool fused_noisy_prepack_b_(const int8_t *b_signal, const uint8_t *b_noise_seed,
                                int rank, std::vector<int8_t> *scan);

    bool available_ = false;
    bool b_job_ready_ = false;
    Case33PrepackMode prepack_mode_ = Case33PrepackMode::Separate;
    const char *backend_ = "unavailable";
    int num_threads_ = 1;
    Case32Int8Mode int8_mode_ = Case32Int8Mode::FastU8S8;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
    int milestone_k_ = 0;
    int blocks_per_milestone_ = 0;
    int tile_cols_ = 0;
    int blocks_k_ = 0;
    int macro_rows_ = 0;
    int macro_cols_ = 0;
    size_t tile_count_ = 0;

    int8_t *a_scan_ = nullptr;
    int8_t *b_scan_ = nullptr;
    std::vector<int8_t> a_pre_;
    std::vector<int8_t> b_pre_;
    std::vector<int32_t> b_comp_ms_;
    std::vector<uint32_t> tile_xor_;
    char backend_buf_[192] = {};
};

int case33_test_inplace_prepack(int M, int N, int K);
int case33_test_fused_prepack(int M, int N, int K, int rank);
