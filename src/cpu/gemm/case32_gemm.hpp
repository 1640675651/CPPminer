#pragma once

#include <cstdint>
#include <vector>

// Case 3.2: hard-coded BLIS-style int8 GEMM (AVX2 ukernel + fixed blocking).
// Mirrors oneDNN's algorithm: prepack B/A once, jc-parallel macro-kernel, K-blocked micro-GEMM.
enum class Case32Int8Mode {
    // oneDNN-style: A+128 at prepack, u8×s8 dot, subtract 128·ΣB per column at store.
    // Algebraically exact for signed int8 A,B (full range including B=-128). Default.
    // See INT8_MODES.md (repo root).
    FastU8S8,
    // sign_epi8 + maddubs: literal signed dot; AVX2 disabled when B contains INT8_MIN.
    ExactS8S8,
};

struct Case32Gemm {
    // Micro-kernel (MR x NR x KR): 8x16 register tile (one A stream covers all NR).
    static constexpr int kMR = 8;
    static constexpr int kNR = 16;
    static constexpr int kKR = 256;
    // Macro blocking: 128x128 is the best balanced fixed shape so far.
    static constexpr int kMacroM = 128;
    static constexpr int kMacroN = 128;

    void set_int8_mode(Case32Int8Mode mode) { int8_mode_ = mode; }
    Case32Int8Mode int8_mode() const { return int8_mode_; }

    bool init(int M, int N, int K, const int8_t *a, const int8_t *b);
    bool available() const { return available_; }

    void run();

    const int32_t *c() const { return c_.data(); }
    const char *backend() const { return backend_; }
    int num_threads() const { return num_threads_; }

private:
    bool available_ = false;
    const char *backend_ = "unavailable";
    int num_threads_ = 1;
    Case32Int8Mode int8_mode_ = Case32Int8Mode::FastU8S8;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
    int tile_cols_ = 0;
    int blocks_k_ = 0;
    int macro_rows_ = 0;
    int macro_cols_ = 0;
    bool b_has_min_int8_ = false;

    std::vector<int8_t> a_pre_;
    std::vector<int8_t> b_pre_;
    std::vector<int32_t> b_comp_;
    std::vector<int32_t> c_;
    char backend_buf_[128] = {};
};
