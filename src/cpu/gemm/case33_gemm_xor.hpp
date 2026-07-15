#pragma once

#include "case32_gemm.hpp"

#include <cstdint>
#include <functional>
#include <vector>

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

    bool init(int M, int N, int K, const int8_t *a, const int8_t *b);
    bool available() const { return available_; }

    void run();
    void run_gemm_only();

    /* Online jackpot scan: no milestone×tile buffer. Contiguous 8×16 tile origins.
     * on_tile(milestone_xor[16], t_rows, t_cols) → true to keep scanning, false to stop.
     * Returns false if unavailable / cancelled before any tile. */
    bool scan_tiles(const std::function<bool(const uint32_t *milestone_xor, int t_rows,
                                             int t_cols)> &on_tile,
                    const std::function<bool()> &should_cancel = {});

    const std::vector<uint32_t> &tile_xor() const { return tile_xor_; }
    const char *backend() const { return backend_; }
    int num_threads() const { return num_threads_; }
    size_t tile_count() const { return tile_count_; }
    int tile_cols() const { return tile_cols_; }

private:
    bool available_ = false;
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

    std::vector<int8_t> a_pre_;
    std::vector<int8_t> b_pre_;
    std::vector<int32_t> b_comp_ms_;
    std::vector<uint32_t> tile_xor_;
    char backend_buf_[160] = {};
};
