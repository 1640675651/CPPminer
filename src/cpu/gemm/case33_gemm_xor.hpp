#pragma once

#include "case32_gemm.hpp"
#include "cp_config.h"

#include <cstdint>
#include <functional>
#include <vector>

enum class Case33PrepackMode {
    Separate,     /* row-major noisy + persistent a_pre_/b_pre_ */
    ReuseScanBuf, /* row-major noisy + prepack swap into scan buffer */
    Fused,        /* noise injection directly into scan/prepack layout */
};

/* ISA preference for micro-kernel dispatch. */
enum class Case33Isa {
    Auto,   /* architecture-specific best ISA, else scalar */
    Avx2,   /* prefer AVX2; fall back if unavailable */
    Sse,    /* force SSSE3 path (disable AVX2); tile via Case33SseTile */
    Neon,   /* force AArch64 Advanced SIMD path */
    Scalar, /* force scalar reference ukernel */
};

/* SSSE3 register-tile shape inside the fixed 8x16 semantic/pack tile. */
enum class Case33SseTile {
    R4C8,  /* 4x8 (default): two row-halves x two col-groups */
    R8C8,  /* 8x8: both row-halves live across K for one col-group */
    R4C16, /* 4x16: one row-half x all 16 cols (better A reuse) */
};

// Case 3.3: Case 3.2 8x16 ukernel + milestoned 8x16 tile XOR.
// tile_xor[milestone * tile_count + spatial_tile_id]
// Pearl jackpot milestones are every R_RANK along K (GEMM KR may be larger).
struct Case33GemmXor {
    static constexpr int kNumMilestones = K_DIM / R_RANK;
    static constexpr int kMR = Case32Gemm::kMR;
    static constexpr int kNR = Case32Gemm::kNR;
    static constexpr int kKR = Case32Gemm::kKR;
    static constexpr int kTileRows = kMR;
    static constexpr int kTileCols = kNR;
    static constexpr int kMacroM = Case32Gemm::kMacroM;
    static constexpr int kMacroN = Case32Gemm::kMacroN;
    static constexpr int kPackRank = 4;

    static_assert(K_DIM % R_RANK == 0, "K must be a multiple of noise rank");
    static_assert(R_RANK % kPackRank == 0, "noise rank must be a multiple of pack rank");
    static_assert(kKR == R_RANK, "GEMM KR must equal pearl noise rank");
    static_assert(kNumMilestones == K_DIM / kKR, "one milestone per KR panel");

    void set_int8_mode(Case32Int8Mode mode) { int8_mode_ = mode; }
    void set_prepack_mode(Case33PrepackMode mode) { prepack_mode_ = mode; }
    Case33PrepackMode prepack_mode() const { return prepack_mode_; }
    void set_inplace_prepack(bool on) {
        prepack_mode_ = on ? Case33PrepackMode::ReuseScanBuf : Case33PrepackMode::Separate;
    }
    bool inplace_prepack() const {
        return prepack_mode_ != Case33PrepackMode::Separate;
    }

    void set_isa(Case33Isa isa) { isa_pref_ = isa; }
    void set_sse_tile(Case33SseTile tile) { sse_tile_ = tile; }
    Case33Isa isa() const { return isa_pref_; }
    Case33Isa isa_used() const { return isa_used_; }
    Case33SseTile sse_tile() const { return sse_tile_; }
    /* Resolve the requested ISA against compiled and runtime capabilities. */
    bool resolve_runtime_isa();
    const char *simd_error() const { return simd_error_; }

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
    bool use_fast_u8s8_() const {
        return int8_mode_ == Case32Int8Mode::FastU8S8 && isa_used_ != Case33Isa::Scalar;
    }
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
    Case33Isa isa_pref_ = Case33Isa::Auto;
    Case33Isa isa_used_ = Case33Isa::Scalar;
    Case33SseTile sse_tile_ = Case33SseTile::R4C8;

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
    char simd_error_[160] = {};
};

int case33_test_inplace_prepack(int M, int N, int K);
int case33_test_fused_prepack(int M, int N, int K, int rank);
/* Compare every compiled/runtime-supported SIMD implementation with scalar. */
int case33_test_simd_parity();
