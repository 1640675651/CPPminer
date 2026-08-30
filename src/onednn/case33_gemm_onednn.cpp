#include "case33_gemm_onednn.hpp"

#include "case5_gemm_launch.hpp"
#include "case5_xor_tile.hpp"
#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_state.h"
#include "cp_util.h"
#include "onednn_intel_devices.hpp"

#include "gemmstone/problem.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace {

constexpr size_t kJackpotCmpDumpDigestOff = 0u;
constexpr size_t kJackpotCmpDumpBoundOff = 8u;
constexpr size_t kJackpotCmpDumpCodeOff = 16u;
constexpr size_t kJackpotCmpDumpFlagGtOff = 24u;
constexpr size_t kJackpotCmpDumpFlagLtOff = 32u;
constexpr size_t kJackpotCmpDumpFallGtOff = 40u;
constexpr size_t kJackpotCmpDumpFallLtOff = 48u;
constexpr size_t kJackpotCmpDumpFallStoreOff = 56u;
constexpr size_t kJackpotCmpDumpReachStoreBeatOff = 64u;
constexpr size_t kJackpotCmpDumpWordsPerTile = 65u;
constexpr uint32_t kGpuCmpUnevaluated = 0xFFFFFFFFu;
constexpr size_t kJackpotFoundFlagOff = 8u * sizeof(uint32_t);
constexpr size_t kJackpotFoundBufBytes = kJackpotFoundFlagOff + sizeof(uint32_t);

int div_up(int a, int b) { return (a + b - 1) / b; }

int rnd_up(int a, int b) { return div_up(a, b) * b; }

// Row-major MxK with leading dimension lda (>= K); zero-pad each row if lda > K.
void pack_a_rowmajor(const int8_t *a_rm, int M, int K, int lda, int8_t *out) {
    const size_t row_bytes = static_cast<size_t>(lda);
    for (int i = 0; i < M; ++i) {
        int8_t *row = out + static_cast<size_t>(i) * row_bytes;
        std::memcpy(row, a_rm + static_cast<size_t>(i) * K, K);
        if (lda > K) {
            std::memset(row + K, 0, static_cast<size_t>(lda - K));
        }
    }
}

// pearl_build_noisy_b / CPU prepack_b_panel: B^T row-major, element (col j, k) at j*K+k.
void pack_b_colmajor(const int8_t *b_bt_rm, int K, int N, int ldb, int8_t *out) {
    std::memset(out, 0, static_cast<size_t>(ldb) * static_cast<size_t>(N));
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            out[static_cast<size_t>(j) * ldb + k] = b_bt_rm[static_cast<size_t>(j) * K + k];
        }
    }
}

int clamp_row_period_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_ROW_PERIOD_BATCH_MAX) {
        batch = CP_ROW_PERIOD_BATCH_MAX;
    }
    return batch;
}

int clamp_col_period_batch(int batch) {
    if (batch < 1) {
        batch = 1;
    }
    if (batch > CP_PERIOD_BATCH_MAX) {
        batch = CP_PERIOD_BATCH_MAX;
    }
    return batch;
}

} // namespace

void Case33GemmOnednn::set_row_period_batch(int batch) {
    row_period_batch_ = clamp_row_period_batch(batch);
}

void Case33GemmOnednn::set_col_period_batch(int batch) {
    col_period_batch_ = clamp_col_period_batch(batch);
}

void Case33GemmOnednn::set_fused_jackpot(bool fused) {
    if (context_ready_) {
        std::fprintf(stderr, "[onednn] set_fused_jackpot ignored after init_context\n");
        return;
    }
    fused_jackpot_ = fused;
}

Case33GemmOnednn::~Case33GemmOnednn() {
    if (jackpot_kernel_) {
        clReleaseKernel(jackpot_kernel_);
        jackpot_kernel_ = nullptr;
    }
    if (blake3_kernel_) {
        clReleaseKernel(blake3_kernel_);
        blake3_kernel_ = nullptr;
    }
    if (jackpot_program_) {
        clReleaseProgram(jackpot_program_);
        jackpot_program_ = nullptr;
    }
    if (a_key_buf_) {
        clReleaseMemObject(a_key_buf_);
        a_key_buf_ = nullptr;
    }
    if (bound_buf_) {
        clReleaseMemObject(bound_buf_);
        bound_buf_ = nullptr;
    }
    if (found_buf_) {
        clReleaseMemObject(found_buf_);
        found_buf_ = nullptr;
    }
    if (out_rows_buf_) {
        clReleaseMemObject(out_rows_buf_);
        out_rows_buf_ = nullptr;
    }
    if (out_cols_buf_) {
        clReleaseMemObject(out_cols_buf_);
        out_cols_buf_ = nullptr;
    }
    if (digest_buf_) {
        clReleaseMemObject(digest_buf_);
        digest_buf_ = nullptr;
    }
    if (beats_buf_) {
        clReleaseMemObject(beats_buf_);
        beats_buf_ = nullptr;
    }
    if (cmp_dump_buf_) {
        clReleaseMemObject(cmp_dump_buf_);
        cmp_dump_buf_ = nullptr;
    }
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    if (a_buf_) {
        clReleaseMemObject(a_buf_);
        a_buf_ = nullptr;
    }
    if (b_buf_) {
        clReleaseMemObject(b_buf_);
        b_buf_ = nullptr;
    }
    if (c_buf_) {
        clReleaseMemObject(c_buf_);
        c_buf_ = nullptr;
    }
    if (tile_xor_buf_) {
        clReleaseMemObject(tile_xor_buf_);
        tile_xor_buf_ = nullptr;
    }
}

bool Case33GemmOnednn::setup_dims_(int M, int N, int K) {
    M_ = M;
    N_ = N;
    K_ = K;
    // A row-major (lda>=K); B column-major (ldb>=K). ldc kept for ABI only.
    lda_ = case5_ngen::pad_ld_int8(K_);
    ldb_ = case5_ngen::pad_ld_int8(K_);
    ldc_ = case5_ngen::pad_ld_int8(M_);

    constexpr int milestone_k = 128;
    if (info_.unrollK <= 0 || (K_ % info_.unrollK) != 0) {
        std::fprintf(stderr, "[onednn] K %% unrollK != 0 (K=%d unrollK=%d)\n", K_, info_.unrollK);
        return false;
    }
    if ((milestone_k % info_.unrollK) != 0 || (K_ % milestone_k) != 0) {
        std::fprintf(stderr,
                     "[onednn] milestone_k=%d must divide unrollK=%d and K=%d\n", milestone_k,
                     info_.unrollK, K_);
        return false;
    }
    xor_period_ = milestone_k / info_.unrollK;
    num_milestones_ = K_ / milestone_k;
    folded_msg_words_ = cp_jackpot::kJackpotWords;
    milestone_k_ = milestone_k;

    int threads_m = div_up(M_, info_.unrollM);
    int threads_n = div_up(N_, info_.unrollN);
    if (info_.isNMK) {
        std::swap(threads_m, threads_n);
    }
    if (info_.fusedEUs && threads_m > 1) {
        threads_m = rnd_up(threads_m, 2);
    }
    if (info_.fixedWG || threads_m > info_.wgM) {
        threads_m = rnd_up(threads_m, info_.wgM);
    }
    if (info_.fixedWG || threads_n > info_.wgN) {
        threads_n = rnd_up(threads_n, info_.wgN);
    }
    threads_n *= info_.wgExpand > 0 ? info_.wgExpand : 1;
    if (info_.isNMK) {
        tile_rows_ = threads_n;
        tile_cols_ = threads_m;
    } else {
        tile_rows_ = threads_m;
        tile_cols_ = threads_n;
    }
    tile_rows_ *= (info_.xorSubGridM > 1) ? info_.xorSubGridM : 1;
    tile_cols_ *= (info_.xorSubGridN > 1) ? info_.xorSubGridN : 1;
    tile_count_ = tile_rows_ * tile_cols_;

    const int hash_mr = info_.xorSubM;
    const int hash_nr = info_.xorSubN;
    if ((M_ % info_.unrollM) != 0 || (N_ % info_.unrollN) != 0) {
        std::fprintf(stderr,
                     "[onednn] M,N must be multiples of gemmstone unroll %dx%d (got %dx%d)\n",
                     info_.unrollM, info_.unrollN, M_, N_);
        return false;
    }
    if ((M_ % hash_mr) != 0 || (N_ % hash_nr) != 0) {
        std::fprintf(stderr,
                     "[onednn] M,N must be multiples of logical hash tile %dx%d (got %dx%d)\n",
                     hash_mr, hash_nr, M_, N_);
        return false;
    }
    if ((PP_ROW_PERIOD % hash_mr) != 0 || (PP_COL_PERIOD % hash_nr) != 0) {
        std::fprintf(stderr,
                     "[onednn] period %dx%d must be multiples of logical hash tile %dx%d\n",
                     PP_ROW_PERIOD, PP_COL_PERIOD, hash_mr, hash_nr);
        return false;
    }

    hash_tile_rows_ = M_ / hash_mr;
    hash_tile_cols_ = N_ / hash_nr;

    if ((M_ % PP_ROW_PERIOD) != 0 || (N_ % PP_COL_PERIOD) != 0) {
        std::fprintf(stderr,
                     "[onednn] M,N must be multiples of %d,%d (got %dx%d)\n", PP_ROW_PERIOD,
                     PP_COL_PERIOD, M_, N_);
        return false;
    }

    return true;
}

void Case33GemmOnednn::compute_tile_grid_(int m, int n, int &out_tile_rows, int &out_tile_cols,
                                            int &out_tile_count) const {
    int threads_m = div_up(m, info_.unrollM);
    int threads_n = div_up(n, info_.unrollN);
    if (info_.isNMK) {
        std::swap(threads_m, threads_n);
    }
    if (info_.fusedEUs && threads_m > 1) {
        threads_m = rnd_up(threads_m, 2);
    }
    if (info_.fixedWG || threads_m > info_.wgM) {
        threads_m = rnd_up(threads_m, info_.wgM);
    }
    if (info_.fixedWG || threads_n > info_.wgN) {
        threads_n = rnd_up(threads_n, info_.wgN);
    }
    threads_n *= info_.wgExpand > 0 ? info_.wgExpand : 1;
    if (info_.isNMK) {
        out_tile_rows = threads_n;
        out_tile_cols = threads_m;
    } else {
        out_tile_rows = threads_m;
        out_tile_cols = threads_n;
    }
    out_tile_rows *= (info_.xorSubGridM > 1) ? info_.xorSubGridM : 1;
    out_tile_cols *= (info_.xorSubGridN > 1) ? info_.xorSubGridN : 1;
    out_tile_count = out_tile_rows * out_tile_cols;
}

bool Case33GemmOnednn::ensure_matrix_bufs_() {
    const size_t a_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(M_);
    const size_t b_bytes = static_cast<size_t>(ldb_) * static_cast<size_t>(N_);

    if (a_buf_ && a_buf_bytes_ != a_bytes) {
        clReleaseMemObject(a_buf_);
        a_buf_ = nullptr;
    }
    if (b_buf_ && b_buf_bytes_ != b_bytes) {
        clReleaseMemObject(b_buf_);
        b_buf_ = nullptr;
    }

    if (!a_buf_) {
        a_buf_ = ocl_.alloc_buffer(a_bytes, CL_MEM_READ_ONLY);
        if (!a_buf_) {
            std::fprintf(stderr, "[onednn] failed to allocate A buffer (%zu bytes)\n", a_bytes);
            return false;
        }
        a_buf_bytes_ = a_bytes;
    }
    if (!b_buf_) {
        b_buf_ = ocl_.alloc_buffer(b_bytes, CL_MEM_READ_ONLY);
        if (!b_buf_) {
            std::fprintf(stderr, "[onednn] failed to allocate B buffer (%zu bytes)\n", b_bytes);
            return false;
        }
        b_buf_bytes_ = b_bytes;
    }
    return true;
}

bool Case33GemmOnednn::ensure_panel_tile_xor_buf_(int panel_tile_count) {
    if (panel_tile_count <= 0) {
        return false;
    }
    if (tile_xor_buf_ && panel_tile_xor_cap_ >= panel_tile_count) {
        return true;
    }
    if (tile_xor_buf_) {
        clReleaseMemObject(tile_xor_buf_);
        tile_xor_buf_ = nullptr;
    }
    const int tile_xor_words = fused_jackpot_ ? folded_msg_words_ : num_milestones_;
    const size_t bytes =
            static_cast<size_t>(tile_xor_words) * static_cast<size_t>(panel_tile_count) *
            sizeof(uint32_t);
    tile_xor_buf_ = ocl_.alloc_buffer(bytes, CL_MEM_READ_WRITE);
    if (!tile_xor_buf_) {
        std::fprintf(stderr,
                     "[onednn] failed to allocate tile_xor panel buffer (%zu bytes, %d tiles)\n",
                     bytes, panel_tile_count);
        return false;
    }
    panel_tile_xor_cap_ = panel_tile_count;
    return true;
}

bool Case33GemmOnednn::build_jackpot_kernel_() {
    jackpot_ready_ = false;
    if (jackpot_kernel_) {
        clReleaseKernel(jackpot_kernel_);
        jackpot_kernel_ = nullptr;
    }
    if (blake3_kernel_) {
        clReleaseKernel(blake3_kernel_);
        blake3_kernel_ = nullptr;
    }
    if (jackpot_program_) {
        clReleaseProgram(jackpot_program_);
        jackpot_program_ = nullptr;
    }

    const std::string kernel_dir = cp_ocl_kernel_dir();
#ifdef _WIN32
    const std::string kernel_path = kernel_dir + "\\cp_onednn_jackpot.cl";
#else
    const std::string kernel_path = kernel_dir + "/cp_onednn_jackpot.cl";
#endif
    std::ifstream in(kernel_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "[onednn] failed to open jackpot kernel: %s\n", kernel_path.c_str());
        return false;
    }
    const std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!ocl_.safe_build_program_from_source(source.c_str(), "-cl-std=CL1.2")) {
        std::fprintf(stderr, "[onednn] jackpot OpenCL build failed\n");
        return false;
    }
    jackpot_program_ = ocl_.program;
    ocl_.program = nullptr;

    jackpot_kernel_ = clCreateKernel(jackpot_program_, "cp_onednn_jackpot_scan", nullptr);
    if (!jackpot_kernel_) {
        std::fprintf(stderr, "[onednn] failed to create cp_onednn_jackpot_scan kernel\n");
        return false;
    }
    blake3_kernel_ = clCreateKernel(jackpot_program_, "cp_onednn_blake3_panel", nullptr);
    if (!blake3_kernel_) {
        std::fprintf(stderr, "[onednn] failed to create cp_onednn_blake3_panel kernel\n");
        return false;
    }
    jackpot_ready_ = true;
    return true;
}

bool Case33GemmOnednn::ensure_jackpot_bufs_() {
    if (!a_key_buf_) {
        a_key_buf_ = ocl_.alloc_buffer(8 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    }
    if (!bound_buf_) {
        bound_buf_ = ocl_.alloc_buffer(8 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    }
    if (!found_buf_) {
        found_buf_ = ocl_.alloc_buffer(kJackpotFoundBufBytes, CL_MEM_READ_WRITE);
    }
    if (!out_rows_buf_) {
        out_rows_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_WRITE_ONLY);
    }
    if (!out_cols_buf_) {
        out_cols_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_WRITE_ONLY);
    }
    return a_key_buf_ && bound_buf_ && found_buf_ && out_rows_buf_ && out_cols_buf_;
}

bool Case33GemmOnednn::init_context(int device_index, int platform_filter) {
    available_ = false;
    context_ready_ = false;
    std::snprintf(backend_, sizeof(backend_), "unavailable");

    const std::vector<OclDeviceInfo> intel_gpus =
            onednn_intel::enumerate_intel_gpus(platform_filter);
    if (intel_gpus.empty()) {
        std::fprintf(stderr,
                     "[onednn] no Intel GPU found (Case 5 needs XeLP/Gen12LP or XeHPG)\n");
        std::snprintf(backend_, sizeof(backend_), "no Intel GPU");
        return false;
    }

    if (device_index < 0 || device_index >= static_cast<int>(intel_gpus.size())) {
        std::fprintf(stderr,
                     "[onednn] invalid --devices %d (valid: 0..%d Intel GPUs). Available:\n",
                     device_index, static_cast<int>(intel_gpus.size()) - 1);
        onednn_intel::list_intel_gpus(platform_filter);
        std::snprintf(backend_, sizeof(backend_), "invalid device index");
        return false;
    }

    const OclDeviceInfo &pick = intel_gpus[static_cast<size_t>(device_index)];
    if (!ocl_.init(pick)) {
        std::snprintf(backend_, sizeof(backend_), "OpenCL init failed");
        return false;
    }

    std::string err;
    if (!case5_ngen::is_supported_device(ocl_.context, ocl_.device, &err)) {
        std::fprintf(stderr, "[onednn] %s\n", err.c_str());
        std::snprintf(backend_, sizeof(backend_), "%s", err.c_str());
        return false;
    }

    const int init_m = g_m_active > 0 ? g_m_active : M_DIM;
    const int init_n = g_n_active > 0 ? g_n_active : N_DIM;

    /* Kernel JIT/catalog ranking uses modest probe dims (gemm_xor default 256³); the
     * OpenCL kernel is size-independent. Production M/N only affect host tile grids. */
    constexpr int kKernelSelectM = 256;
    constexpr int kKernelSelectN = 256;
    case5_ngen::BuildParams build_dims{kKernelSelectM, kKernelSelectN, K_DIM, 0, 0, 0};
    build_dims.lda = case5_ngen::pad_ld_int8(build_dims.k);
    build_dims.ldb = case5_ngen::pad_ld_int8(build_dims.k);
    build_dims.ldc = case5_ngen::pad_ld_int8(build_dims.m);

    std::string ngen_err;
    kernel_ = case5_ngen::build_igemm_kernel(ocl_.context, ocl_.device, &build_dims, &info_,
                                             &ngen_err, false, fused_jackpot_);
    if (!kernel_) {
        std::fprintf(stderr, "[onednn] gemmstone kernel build failed: %s\n", ngen_err.c_str());
        std::snprintf(backend_, sizeof(backend_), "gemmstone build failed: %s", ngen_err.c_str());
        return false;
    }
    if (!info_.selectionLog.empty()) {
        std::fprintf(stderr, "[onednn] %s", info_.selectionLog.c_str());
    }

    if (!setup_dims_(init_m, init_n, K_DIM)) {
        return false;
    }

    c_buf_ = ocl_.alloc_buffer(sizeof(int32_t), CL_MEM_READ_WRITE);
    if (!c_buf_) {
        return false;
    }

    int32_t c_stub = 0;
    if (!ocl_.write_buffer(c_buf_, &c_stub, sizeof(c_stub))) {
        return false;
    }

    device_name_ = ocl_.device_name;
    platform_name_ = ocl_.platform_name;
    device_flat_index_ = ocl_.device_flat_index;
    const char *scan_mode =
            fused_jackpot_ ? "igemm+wrapGRF+case56Blake3+gpuJudge" : "igemm+tileXOR+GPUjackpot";
    std::snprintf(backend_, sizeof(backend_),
                  "oneDNN gemmstone %s/%s %s unroll %dx%d xor %dx%d wg %dx%d "
                  "sg %d ms=%d fold=%d tiles=%dx%d hash=%dx%d row_batch=%d col_batch=%d",
                  info_.hwName, info_.strategyName, scan_mode, info_.unrollM, info_.unrollN,
                  info_.xorSubM, info_.xorSubN, info_.wgM, info_.wgN, info_.subgroupSize,
                  num_milestones_, fused_jackpot_ ? folded_msg_words_ : 0, tile_rows_, tile_cols_,
                  hash_tile_rows_, hash_tile_cols_, row_period_batch_, col_period_batch_);
    context_ready_ = true;
    prep_ready_ = prep_.init(&ocl_, cp_ocl_kernel_dir(), false);
    if (!prep_ready_) {
        std::fprintf(stderr, "[onednn] GPU matrix prep init failed; using CPU fallback\n");
    }
    if (!build_jackpot_kernel_()) {
        std::fprintf(stderr, "[onednn] GPU jackpot kernel init failed\n");
        return false;
    }
    available_ = false;
    return true;
}

bool Case33GemmOnednn::upload_a_rowmajor_(const int8_t *a_rowmajor) {
    if (!a_rowmajor) {
        return false;
    }
    const size_t raw_bytes = static_cast<size_t>(M_) * static_cast<size_t>(K_);
    const size_t bytes = static_cast<size_t>(lda_) * static_cast<size_t>(M_);
    if (lda_ == K_) {
        a_host_.assign(a_rowmajor, a_rowmajor + raw_bytes);
        return ocl_.write_buffer(a_buf_, a_rowmajor, bytes);
    }
    a_host_.resize(bytes);
    pack_a_rowmajor(a_rowmajor, M_, K_, lda_, a_host_.data());
    return ocl_.write_buffer(a_buf_, a_host_.data(), bytes);
}

bool Case33GemmOnednn::upload_b_colmajor_(const int8_t *b_rowmajor) {
    if (!b_rowmajor) {
        return false;
    }
    const size_t bytes = static_cast<size_t>(ldb_) * static_cast<size_t>(N_);
    b_host_.resize(bytes);
    pack_b_colmajor(b_rowmajor, K_, N_, ldb_, b_host_.data());
    return ocl_.write_buffer(b_buf_, b_host_.data(), bytes);
}

bool Case33GemmOnednn::prepare_job(int M, int N, int K, const int8_t *b_rowmajor) {
    available_ = false;
    if (!context_ready_ || !kernel_ || !b_rowmajor) {
        return false;
    }
    if (!setup_dims_(M, N, K)) {
        return false;
    }
    if (!ensure_matrix_bufs_()) {
        return false;
    }
    if (!upload_b_colmajor_(b_rowmajor)) {
        return false;
    }
    available_ = true;
    return true;
}

bool Case33GemmOnednn::prepare_job_gpu(int M, int N, int K, const uint8_t b_noise_seed[32]) {
    available_ = false;
    if (!context_ready_ || !kernel_ || !b_noise_seed || !prep_ready_) {
        return false;
    }
    if (!setup_dims_(M, N, K)) {
        return false;
    }
    if (!ensure_matrix_bufs_()) {
        return false;
    }
    if (!prep_.prepare_job_b_colmajor(b_buf_, b_noise_seed, N_, K_, ldb_)) {
        return false;
    }
    available_ = true;
    return true;
}

bool Case33GemmOnednn::prepare_attempt_a(const int8_t *a_rowmajor) {
    if (!available_ || !a_rowmajor) {
        return false;
    }
    return upload_a_rowmajor_(a_rowmajor);
}

bool Case33GemmOnednn::prepare_attempt_gpu(const uint8_t *ab_seed, int ab_seed_len,
                                           const uint8_t job_key[32],
                                           const uint8_t b_noise_seed[32], int salted,
                                           uint8_t a_key_out[32]) {
    if (!available_ || !ab_seed || !job_key || !b_noise_seed || !a_key_out || !prep_ready_) {
        return false;
    }
    if (!ensure_matrix_bufs_() || !prep_.ensure_buffers(M_, N_, K_)) {
        return false;
    }
    return prep_.prepare_attempt_a_rowmajor(a_buf_, ab_seed, ab_seed_len, job_key, b_noise_seed,
                                            M_, K_, lda_, salted, a_key_out);
}

bool Case33GemmOnednn::read_A_sig(int8_t *h_A_sig) {
    if (!h_A_sig || M_ <= 0 || K_ <= 0) {
        return false;
    }
    if (prep_ready_) {
        return prep_.read_A_sig(h_A_sig, static_cast<size_t>(M_) * static_cast<size_t>(K_));
    }
    if (a_host_.empty()) {
        return false;
    }
    const size_t need = static_cast<size_t>(M_) * static_cast<size_t>(K_);
    if (a_host_.size() < need) {
        return false;
    }
    if (lda_ == K_) {
        std::memcpy(h_A_sig, a_host_.data(), need);
        return true;
    }
    for (int i = 0; i < M_; ++i) {
        std::memcpy(h_A_sig + static_cast<size_t>(i) * K_, a_host_.data() + static_cast<size_t>(i) * lda_,
                    K_);
    }
    return true;
}

bool Case33GemmOnednn::run_gemm_panel_(int m_panel, int n_panel, int64_t offset_a_rows,
                                       int64_t offset_b_cols, int panel_tile_count,
                                       int panel_tile_cols, int tr_base, int tc_base,
                                       bool finish_queue, int *out_found) {
    if (!available_ || !kernel_) {
        return false;
    }
    if (fused_jackpot_) {
        if (!digest_buf_ || !beats_buf_ || !cmp_dump_buf_ || !found_buf_ || !out_rows_buf_ || !out_cols_buf_
                || !bound_buf_) {
            return false;
        }
        // Sentinel so unwritten beats/cmp show up in the dump.
        if (beats_host_.size() < static_cast<size_t>(panel_tile_count)) {
            beats_host_.assign(static_cast<size_t>(panel_tile_count), 0xFFFFFFFFu);
        } else {
            std::fill(beats_host_.begin(), beats_host_.begin() + panel_tile_count, 0xFFFFFFFFu);
        }
        const size_t cmp_dump_words =
                static_cast<size_t>(panel_tile_count) * kJackpotCmpDumpWordsPerTile;
        if (cmp_dump_host_.size() < cmp_dump_words) {
            cmp_dump_host_.assign(cmp_dump_words, kGpuCmpUnevaluated);
        } else {
            std::fill(cmp_dump_host_.begin(), cmp_dump_host_.begin() + cmp_dump_words,
                      kGpuCmpUnevaluated);
        }
        if (!ocl_.write_buffer(found_buf_, scan_jackpot_bound_, 8u * sizeof(uint32_t), 0)) {
            return false;
        }
        const int zero = 0;
        if (!ocl_.write_buffer(found_buf_, &zero, sizeof(zero), kJackpotFoundFlagOff)) {
            return false;
        }
        if (!ocl_.write_buffer(beats_buf_, beats_host_.data(),
                               static_cast<size_t>(panel_tile_count) * sizeof(uint32_t))) {
            return false;
        }
        if (!ocl_.write_buffer(cmp_dump_buf_, cmp_dump_host_.data(),
                               cmp_dump_words * sizeof(uint32_t))) {
            return false;
        }
    } else if (!tile_xor_buf_) {
        return false;
    }

    case5_ngen::LaunchBuffers bufs;
    bufs.a = a_buf_;
    bufs.b = b_buf_;
    bufs.c = c_buf_;
    bufs.tile_xor = fused_jackpot_ ? nullptr : tile_xor_buf_;
    bufs.offset_a = offset_a_rows * static_cast<int64_t>(lda_);
    bufs.offset_b = offset_b_cols;
    bufs.offset_c = 0;
    bufs.lda = lda_;
    bufs.ldb = ldb_;
    bufs.ldc = ldc_;
    bufs.m = m_panel;
    bufs.n = n_panel;
    bufs.k = K_;
    bufs.tile_count = panel_tile_count;
    bufs.tile_cols = panel_tile_cols;
    bufs.xor_period = xor_period_;
    if (fused_jackpot_) {
        std::memcpy(bufs.blake3_key_words, scan_jackpot_key_, sizeof(bufs.blake3_key_words));
        std::memcpy(bufs.blake3_bound_words, scan_jackpot_bound_, sizeof(bufs.blake3_bound_words));
        bufs.blake3_out = digest_buf_;
        bufs.blake3_beats = beats_buf_;
        bufs.blake3_cmp_dump = cmp_dump_buf_;
        bufs.found_flag = found_buf_;
        bufs.out_t_rows = out_rows_buf_;
        bufs.out_t_cols = out_cols_buf_;
        bufs.tr_base = tr_base;
        bufs.tc_base = tc_base;
    }

    gemmstone::GEMMProblem problem;
    problem.Ta = problem.Ta_ext = gemmstone::Type::s8;
    problem.Tb = problem.Tb_ext = gemmstone::Type::s8;
    problem.Tc = problem.Tc_ext = gemmstone::Type::s32;
    problem.Ts = gemmstone::Type::f32;
    problem.alpha = 1;
    problem.beta = 0;
    problem.case5TileXor = true;
    problem.case5TileXorNop = false;
    problem.case5FuseJackpot = fused_jackpot_;
    problem.case5TileXorBlake3 = fused_jackpot_;
    problem.A.layout = gemmstone::MatrixLayout::T;
    problem.B.layout = gemmstone::MatrixLayout::N;
    problem.C.layout = gemmstone::MatrixLayout::N;

    const case5_ngen::LaunchDims dims =
            case5_ngen::compute_case5_launch_dims(info_, m_panel, n_panel);
    cl_int err = case5_ngen::bind_case5_kernel_args(kernel_, info_, problem, bufs, dims);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] clSetKernelArg failed (%d)\n", err);
        return false;
    }

    const size_t gws[2] = {dims.gws[0], dims.gws[1]};
    const size_t lws[2] = {dims.lws[0], dims.lws[1]};
    err = clEnqueueNDRangeKernel(ocl_.queue, kernel_, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] enqueue failed (%d): %s\n", err,
                     OpenClContext::error_string(err).c_str());
        return false;
    }
    if (finish_queue) {
        err = clFinish(ocl_.queue);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[onednn] clFinish failed (%d): %s\n", err,
                         OpenClContext::error_string(err).c_str());
            return false;
        }
    }

    (void)out_found;
    return true;
}

bool Case33GemmOnednn::run_gpu_jackpot_panel_(int panel_tile_count, int panel_tile_cols,
                                              int tr_base, int tc_base, int *out_found,
                                              bool finish_queue) {
    if (!jackpot_ready_ || !jackpot_kernel_ || !tile_xor_buf_ || panel_tile_count <= 0) {
        return false;
    }
    if (!ensure_jackpot_bufs_()) {
        return false;
    }

    const int hash_mr = info_.xorSubM;
    const int hash_nr = info_.xorSubN;
    const int use_folded_msg = fused_jackpot_ ? 1 : 0;
    const int jackpot_words = use_folded_msg ? folded_msg_words_ : num_milestones_;
    cl_int err = CL_SUCCESS;
    int arg = 0;
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &tile_xor_buf_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &jackpot_words);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &panel_tile_count);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &panel_tile_cols);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &tr_base);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &tc_base);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &hash_mr);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &hash_nr);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &use_folded_msg);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &a_key_buf_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &bound_buf_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &found_buf_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &out_rows_buf_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &out_cols_buf_);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] jackpot clSetKernelArg failed (%d)\n", err);
        return false;
    }

    size_t local = 256;
    if (static_cast<size_t>(panel_tile_count) < local) {
        local = static_cast<size_t>(panel_tile_count);
    }
    while (local > 1 && (static_cast<size_t>(panel_tile_count) % local) != 0) {
        local >>= 1;
    }
    const size_t global = static_cast<size_t>(panel_tile_count);
    err = clEnqueueNDRangeKernel(ocl_.queue, jackpot_kernel_, 1, nullptr, &global,
                                 local > 1 ? &local : nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] jackpot enqueue failed (%d): %s\n", err,
                     OpenClContext::error_string(err).c_str());
        return false;
    }
    if (finish_queue) {
        err = clFinish(ocl_.queue);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[onednn] jackpot clFinish failed (%d)\n", err);
            return false;
        }

        int found = 0;
        if (!ocl_.read_buffer(found_buf_, &found, sizeof(found), kJackpotFoundFlagOff)) {
            return false;
        }
        if (out_found) {
            *out_found = found;
        }
    }

    return true;
}

bool Case33GemmOnednn::run_gemm_jackpot_panel_(int m_panel, int n_panel, int64_t offset_a_rows,
                                               int64_t offset_b_cols, int panel_tile_count,
                                               int panel_tile_cols, int tr_base, int tc_base,
                                               int *out_found, int *out_t_rows, int *out_t_cols) {
    if (!run_gemm_panel_(m_panel, n_panel, offset_a_rows, offset_b_cols, panel_tile_count,
                         panel_tile_cols, tr_base, tc_base, /*finish_queue=*/true, nullptr)) {
        return false;
    }
    if (fused_jackpot_) {
        static const bool dump_compare = [] {
            const char *v = std::getenv("ONEDNN_DUMP_COMPARE");
            return v && v[0] != '0';
        }();
        if (dump_compare && !compare_dump_done_) {
            dump_compare_panel_(panel_tile_count);
            compare_dump_done_ = true;
        }

        int found = 0;
        if (!ocl_.read_buffer(found_buf_, &found, sizeof(found), kJackpotFoundFlagOff)) {
            return false;
        }
        if (out_found) {
            *out_found = found ? 1 : 0;
        }
        if (found) {
            int t_rows = -1;
            int t_cols = -1;
            int spat = -1;
            if (!find_fused_panel_hit_(panel_tile_count, panel_tile_cols, tr_base, tc_base,
                                       &t_rows, &t_cols, &spat)) {
                std::fprintf(stderr,
                             "[onednn] ignoring GPU jackpot claim (found_flag set, no CPU-valid "
                             "beat in beats buffer)\n");
                std::fflush(stderr);
                found = 0;
                if (out_found) {
                    *out_found = 0;
                }
                return true;
            }
            const bool cpu_beat =
                    log_tile_jackpot_compare_(spat, t_rows, t_cols, "jackpot hit compare");
            if (!cpu_beat) {
                std::fprintf(stderr,
                             "[onednn] ignoring GPU jackpot claim (cpu_beat=0 on GPU digest)\n");
                std::fflush(stderr);
                found = 0;
                if (out_found) {
                    *out_found = 0;
                }
                return true;
            }
            if (dump_compare) {
                dump_compare_panel_(panel_tile_count);
            }
            if (out_t_rows) {
                *out_t_rows = t_rows;
            }
            if (out_t_cols) {
                *out_t_cols = t_cols;
            }
        }
        return true;
    }
    if (!run_gpu_jackpot_panel_(panel_tile_count, panel_tile_cols, tr_base, tc_base, out_found,
                                true)) {
        return false;
    }
    if (out_found && *out_found) {
        if (out_t_rows && !ocl_.read_buffer(out_rows_buf_, out_t_rows, sizeof(int))) {
            return false;
        }
        if (out_t_cols && !ocl_.read_buffer(out_cols_buf_, out_t_cols, sizeof(int))) {
            return false;
        }
    }
    return true;
}

bool Case33GemmOnednn::ensure_panel_digest_bufs_(int panel_tile_count) {
    if (panel_tile_count <= 0) {
        return false;
    }
    if (panel_digest_cap_ >= panel_tile_count && digest_buf_ && beats_buf_ && cmp_dump_buf_) {
        return true;
    }
    if (digest_buf_) {
        clReleaseMemObject(digest_buf_);
        digest_buf_ = nullptr;
    }
    if (beats_buf_) {
        clReleaseMemObject(beats_buf_);
        beats_buf_ = nullptr;
    }
    if (cmp_dump_buf_) {
        clReleaseMemObject(cmp_dump_buf_);
        cmp_dump_buf_ = nullptr;
    }
    const size_t digest_bytes =
            static_cast<size_t>(panel_tile_count) * 8u * sizeof(uint32_t);
    const size_t beats_bytes = static_cast<size_t>(panel_tile_count) * sizeof(uint32_t);
    const size_t cmp_dump_bytes =
            static_cast<size_t>(panel_tile_count) * kJackpotCmpDumpWordsPerTile * sizeof(uint32_t);
    digest_buf_ = ocl_.alloc_buffer(digest_bytes, CL_MEM_READ_WRITE);
    beats_buf_ = ocl_.alloc_buffer(beats_bytes, CL_MEM_READ_WRITE);
    cmp_dump_buf_ = ocl_.alloc_buffer(cmp_dump_bytes, CL_MEM_READ_WRITE);
    if (!digest_buf_ || !beats_buf_ || !cmp_dump_buf_) {
        return false;
    }
    panel_digest_cap_ = panel_tile_count;
    digest_host_.assign(static_cast<size_t>(panel_tile_count) * 8u, 0u);
    beats_host_.assign(static_cast<size_t>(panel_tile_count), 0u);
    cmp_dump_host_.assign(static_cast<size_t>(panel_tile_count) * kJackpotCmpDumpWordsPerTile, 0u);
    return true;
}

namespace {

void fmt_blake3_msb_hex(const uint32_t digest[8], char *out, size_t out_cap) {
    size_t pos = 0;
    for (int w = 7; w >= 0 && pos + 8 < out_cap; --w) {
        pos += static_cast<size_t>(std::snprintf(out + pos, out_cap - pos, "%08x", digest[w]));
    }
    if (pos < out_cap) {
        out[pos] = '\0';
    } else if (out_cap > 0) {
        out[out_cap - 1] = '\0';
    }
}

const char *unsigned_word_cmp(uint32_t d, uint32_t b) {
    if (d < b) {
        return "lt";
    }
    if (d > b) {
        return "gt";
    }
    return "eq";
}

const char *compare_action(const char *cmp) {
    if (std::strcmp(cmp, "lt") == 0) {
        return "BEAT";
    }
    if (std::strcmp(cmp, "gt") == 0) {
        return "lose";
    }
    return "next";
}


bool simulate_gpu_cmp_beat(const uint32_t gpu_cmp[8]) {
    for (int w = 7; w >= 0; --w) {
        const uint32_t code = gpu_cmp[w];
        if (code == kGpuCmpUnevaluated) {
            continue;
        }
        if (code == 1) {
            return false;
        }
        if (code == 2) {
            return true;
        }
        if (code == 3) {
            continue;
        }
    }
    return true;
}

const char *gpu_cmp_code_label(uint32_t code) {
    switch (code) {
    case 1:
        return "gt";
    case 2:
        return "lt";
    case 3:
        return "eq";
    default:
        return "--";
    }
}

const char *gpu_cmp_action_label(uint32_t code) {
    switch (code) {
    case 1:
        return "lose";
    case 2:
        return "BEAT";
    case 3:
        return "next";
    default:
        return "----";
    }
}

// GPU compare trace vs CPU digest_beats_target; see case5_gen12_jackpot_cmp.md.
void dump_tile_gpu_cpu_word_compare(const char *label, const uint32_t digest[8],
                                    const uint32_t cpu_bound[8],
                                    const uint32_t gpu_digest_at_cmp[8],
                                    const uint32_t gpu_bound[8], const uint32_t gpu_cmp[8],
                                    const uint32_t gpu_f1_gt[8], const uint32_t gpu_f1_lt[8],
                                    const uint32_t gpu_fall_gt[8], const uint32_t gpu_fall_lt[8],
                                    const uint32_t gpu_fall_store[8], uint32_t gpu_reach_store) {
    std::fprintf(stderr, "[onednn] %s per-word GPU vs CPU compare (MSB-first w=7..0):\n", label);
    std::fprintf(stderr,
                 "  w   gpu_dig     gpu_bound   f1_gt f1_lt !gt_j !lt_j !store gpu cmp gpu act  "
                 "digest      cpu_bound   cpu cmp cpu act  dig_ok bound_ok cmp_ok\n");

    int bound_mismatch = 0;
    int digest_mismatch = 0;
    int cmp_mismatch = 0;
    for (int w = 7; w >= 0; --w) {
        const uint32_t gd = gpu_digest_at_cmp[w];
        const uint32_t gb = gpu_bound[w];
        const uint32_t code = gpu_cmp[w];
        const uint32_t f1_gt = gpu_f1_gt ? gpu_f1_gt[w] : kGpuCmpUnevaluated;
        const uint32_t f1_lt = gpu_f1_lt ? gpu_f1_lt[w] : kGpuCmpUnevaluated;
        const uint32_t fall_gt = gpu_fall_gt ? gpu_fall_gt[w] : kGpuCmpUnevaluated;
        const uint32_t fall_lt = gpu_fall_lt ? gpu_fall_lt[w] : kGpuCmpUnevaluated;
        const uint32_t fall_store = gpu_fall_store ? gpu_fall_store[w] : kGpuCmpUnevaluated;
        const uint32_t d = digest[w];
        const uint32_t cb = cpu_bound[w];
        const char *cpu_cmp = unsigned_word_cmp(d, cb);
        const char *gpu_cmp_str = gpu_cmp_code_label(code);
        const bool evaluated = code != kGpuCmpUnevaluated;
        const bool dig_ok = !evaluated || (gd == d);
        const bool bound_ok = !evaluated || (gb == cb);
        bool cmp_ok = true;
        if (evaluated) {
            const char *gpu_op_cmp = unsigned_word_cmp(gd, gb);
            if (std::strcmp(gpu_cmp_str, gpu_op_cmp) != 0) {
                cmp_ok = false;
            }
            if (std::strcmp(gpu_cmp_str, cpu_cmp) != 0) {
                cmp_ok = false;
            }
        }
        if (!dig_ok) {
            ++digest_mismatch;
        }
        if (!bound_ok) {
            ++bound_mismatch;
        }
        if (evaluated && !cmp_ok) {
            ++cmp_mismatch;
        }
        std::fprintf(stderr,
                     "  %d  %08x  %08x  %4s  %4s  %4s  %4s  %5s  %3s   %-5s  %08x  %08x  %3s   "
                     "%-5s  %3s    %3s      %3s\n",
                     w, gd, gb,
                     f1_gt == kGpuCmpUnevaluated ? "--"
                                                 : (f1_gt ? "1" : "0"),
                     f1_lt == kGpuCmpUnevaluated ? "--"
                                                 : (f1_lt ? "1" : "0"),
                     fall_gt == kGpuCmpUnevaluated ? "--" : (fall_gt ? "YES" : "no"),
                     fall_lt == kGpuCmpUnevaluated ? "--" : (fall_lt ? "YES" : "no"),
                     fall_store == kGpuCmpUnevaluated ? "--" : (fall_store ? "YES" : "no"),
                     gpu_cmp_str, gpu_cmp_action_label(code), d, cb, cpu_cmp,
                     compare_action(cpu_cmp), dig_ok ? (evaluated ? "yes" : "--") : "NO",
                     bound_ok ? (evaluated ? "yes" : "--") : "NO",
                     !evaluated ? "--" : (cmp_ok ? "yes" : "NO"));
    }

    const bool cpu_beat = cp_jackpot::digest_beats_target(digest, cpu_bound);
    const bool gpu_trace_beat = simulate_gpu_cmp_beat(gpu_cmp);
    std::fprintf(stderr,
                 "[onednn] %s overall: cpu_beat=%d gpu_trace_beat=%d reach_store_beat=%s "
                 "digest_mismatch_words=%d bound_mismatch_words=%d cmp_mismatch_words=%d\n",
                 label, cpu_beat ? 1 : 0, gpu_trace_beat ? 1 : 0,
                 gpu_reach_store == kGpuCmpUnevaluated
                         ? "--"
                         : (gpu_reach_store ? "YES" : "no"),
                 digest_mismatch, bound_mismatch, cmp_mismatch);
    std::fflush(stderr);
}

} // namespace

bool Case33GemmOnednn::read_tile_jackpot_(int spat, uint32_t digest[8], uint32_t *gpu_beat_raw,
                                          uint32_t gpu_digest_at_cmp[8], uint32_t gpu_bound[8],
                                          uint32_t gpu_cmp[8], uint32_t gpu_f1_gt[8],
                                          uint32_t gpu_f1_lt[8], uint32_t gpu_fall_gt[8],
                                          uint32_t gpu_fall_lt[8], uint32_t gpu_fall_store[8],
                                          uint32_t *gpu_reach_store) const {
    if (!digest_buf_ || !beats_buf_ || spat < 0 || !digest || !gpu_beat_raw) {
        return false;
    }
    const size_t dig_off = static_cast<size_t>(spat) * 8u * sizeof(uint32_t);
    const size_t beat_off = static_cast<size_t>(spat) * sizeof(uint32_t);
    const size_t cmp_base = static_cast<size_t>(spat) * kJackpotCmpDumpWordsPerTile;
    const size_t cmp_digest_off = (cmp_base + kJackpotCmpDumpDigestOff) * sizeof(uint32_t);
    const size_t cmp_bound_off = (cmp_base + kJackpotCmpDumpBoundOff) * sizeof(uint32_t);
    const size_t cmp_code_off = (cmp_base + kJackpotCmpDumpCodeOff) * sizeof(uint32_t);
    const size_t cmp_f1_gt_off = (cmp_base + kJackpotCmpDumpFlagGtOff) * sizeof(uint32_t);
    const size_t cmp_f1_lt_off = (cmp_base + kJackpotCmpDumpFlagLtOff) * sizeof(uint32_t);
    const size_t cmp_fall_gt_off = (cmp_base + kJackpotCmpDumpFallGtOff) * sizeof(uint32_t);
    const size_t cmp_fall_lt_off = (cmp_base + kJackpotCmpDumpFallLtOff) * sizeof(uint32_t);
    const size_t cmp_fall_store_off = (cmp_base + kJackpotCmpDumpFallStoreOff) * sizeof(uint32_t);
    const size_t cmp_reach_store_off =
            (cmp_base + kJackpotCmpDumpReachStoreBeatOff) * sizeof(uint32_t);
    if (!ocl_.read_buffer(digest_buf_, digest, 8u * sizeof(uint32_t), dig_off) ||
        !ocl_.read_buffer(beats_buf_, gpu_beat_raw, sizeof(uint32_t), beat_off)) {
        return false;
    }
    if (gpu_digest_at_cmp && gpu_bound && gpu_cmp && cmp_dump_buf_) {
        if (!ocl_.read_buffer(cmp_dump_buf_, gpu_digest_at_cmp, 8u * sizeof(uint32_t),
                              cmp_digest_off) ||
            !ocl_.read_buffer(cmp_dump_buf_, gpu_bound, 8u * sizeof(uint32_t), cmp_bound_off) ||
            !ocl_.read_buffer(cmp_dump_buf_, gpu_cmp, 8u * sizeof(uint32_t), cmp_code_off)) {
            return false;
        }
        if (gpu_f1_gt) {
            if (!ocl_.read_buffer(cmp_dump_buf_, gpu_f1_gt, 8u * sizeof(uint32_t), cmp_f1_gt_off)) {
                return false;
            }
        }
        if (gpu_f1_lt) {
            if (!ocl_.read_buffer(cmp_dump_buf_, gpu_f1_lt, 8u * sizeof(uint32_t), cmp_f1_lt_off)) {
                return false;
            }
        }
        if (gpu_fall_gt) {
            if (!ocl_.read_buffer(cmp_dump_buf_, gpu_fall_gt, 8u * sizeof(uint32_t),
                                  cmp_fall_gt_off)) {
                return false;
            }
        }
        if (gpu_fall_lt) {
            if (!ocl_.read_buffer(cmp_dump_buf_, gpu_fall_lt, 8u * sizeof(uint32_t),
                                  cmp_fall_lt_off)) {
                return false;
            }
        }
        if (gpu_fall_store) {
            if (!ocl_.read_buffer(cmp_dump_buf_, gpu_fall_store, 8u * sizeof(uint32_t),
                                  cmp_fall_store_off)) {
                return false;
            }
        }
        if (gpu_reach_store) {
            if (!ocl_.read_buffer(cmp_dump_buf_, gpu_reach_store, sizeof(uint32_t),
                                  cmp_reach_store_off)) {
                return false;
            }
        }
        return true;
    }
    return true;
}

bool Case33GemmOnednn::find_fused_panel_hit_(int panel_tile_count, int panel_tile_cols,
                                             int tr_base, int tc_base, int *out_t_rows,
                                             int *out_t_cols, int *out_spat) {
    if (!digest_buf_ || !beats_buf_ || panel_tile_count <= 0) {
        return false;
    }
    const int hash_mr = info_.xorSubM;
    const int hash_nr = info_.xorSubN;
    if (hash_mr <= 0 || hash_nr <= 0) {
        return false;
    }

    const size_t digest_words = static_cast<size_t>(panel_tile_count) * 8u;
    const size_t beats_words = static_cast<size_t>(panel_tile_count);
    if (digest_host_.size() < digest_words) {
        digest_host_.resize(digest_words);
    }
    if (beats_host_.size() < beats_words) {
        beats_host_.resize(beats_words);
    }
    if (!ocl_.read_buffer(digest_buf_, digest_host_.data(), digest_words * sizeof(uint32_t)) ||
        !ocl_.read_buffer(beats_buf_, beats_host_.data(), beats_words * sizeof(uint32_t))) {
        return false;
    }

    for (int spat = 0; spat < panel_tile_count; ++spat) {
        const uint32_t graw = beats_host_[static_cast<size_t>(spat)];
        if (graw == 0u || graw == 0xFFFFFFFFu) {
            continue;
        }
        uint32_t digest[8];
        for (int w = 0; w < 8; ++w) {
            digest[w] = digest_host_[static_cast<size_t>(spat) * 8u + static_cast<size_t>(w)];
        }
        if (!cp_jackpot::digest_beats_target(digest, scan_jackpot_bound_)) {
            continue;
        }
        const int tr = spat / panel_tile_cols;
        const int tc = spat % panel_tile_cols;
        const int lr = tr_base + tr;
        const int lc = tc_base + tc;
        if (out_t_rows) {
            *out_t_rows = lr * hash_mr;
        }
        if (out_t_cols) {
            *out_t_cols = lc * hash_nr;
        }
        if (out_spat) {
            *out_spat = spat;
        }
        return true;
    }
    return false;
}

bool Case33GemmOnednn::log_tile_jackpot_compare_(int spat, int t_rows, int t_cols,
                                                const char *tag) {
    uint32_t digest[8] = {};
    uint32_t gpu_digest_at_cmp[8] = {};
    uint32_t gpu_bound[8] = {};
    uint32_t gpu_cmp[8] = {};
    uint32_t gpu_f1_gt[8] = {};
    uint32_t gpu_f1_lt[8] = {};
    uint32_t gpu_fall_gt[8] = {};
    uint32_t gpu_fall_lt[8] = {};
    uint32_t gpu_fall_store[8] = {};
    uint32_t gpu_reach_store = kGpuCmpUnevaluated;
    uint32_t gpu_beat_raw = 0xFFFFFFFFu;
    if (!read_tile_jackpot_(spat, digest, &gpu_beat_raw, gpu_digest_at_cmp, gpu_bound, gpu_cmp,
                            gpu_f1_gt, gpu_f1_lt, gpu_fall_gt, gpu_fall_lt, gpu_fall_store,
                            &gpu_reach_store)) {
        std::fprintf(stderr, "[onednn] %s t_rows=%d t_cols=%d spat=%d: failed to read GPU digest/beats\n",
                     tag ? tag : "jackpot compare", t_rows, t_cols, spat);
        std::fflush(stderr);
        return false;
    }

    const bool gpu = gpu_beat_raw != 0u && gpu_beat_raw != 0xFFFFFFFFu;
    const bool cpu = cp_jackpot::digest_beats_target(digest, scan_jackpot_bound_);
    const bool gpu_trace = simulate_gpu_cmp_beat(gpu_cmp);
    char blake_hex[65];
    fmt_blake3_msb_hex(digest, blake_hex, sizeof(blake_hex));

    std::fprintf(stderr,
                 "[onednn] %s t_rows=%d t_cols=%d spat=%d gpu_beat=%d gpu_trace_beat=%d "
                 "cpu_beat=%d gpu_beat_raw=0x%08x blake3=%s bound=",
                 tag ? tag : "jackpot compare", t_rows, t_cols, spat, gpu ? 1 : 0,
                 gpu_trace ? 1 : 0, cpu ? 1 : 0, gpu_beat_raw, blake_hex);
    for (int w = 7; w >= 0; --w) {
        std::fprintf(stderr, "%08x", scan_jackpot_bound_[w]);
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);

    dump_tile_gpu_cpu_word_compare("gpu digest vs bound", digest, scan_jackpot_bound_,
                                   gpu_digest_at_cmp, gpu_bound, gpu_cmp, gpu_f1_gt, gpu_f1_lt,
                                   gpu_fall_gt, gpu_fall_lt, gpu_fall_store, gpu_reach_store);
    if (gpu != gpu_trace) {
        std::fprintf(stderr,
                     "[onednn] WARNING: gpu_beat flag (%d) disagrees with GPU trace replay (%d)\n",
                     gpu ? 1 : 0, gpu_trace ? 1 : 0);
        std::fflush(stderr);
    }

    log_cpu_recompute_hit_(t_rows, t_cols, digest, gpu_digest_at_cmp, gpu_bound, gpu_cmp, gpu_f1_gt,
                           gpu_f1_lt, gpu_fall_gt, gpu_fall_lt, gpu_fall_store, gpu_reach_store);
    return cpu;
}

bool Case33GemmOnednn::read_device_ab_for_recompute_() {
    if (!a_buf_ || !b_buf_ || M_ <= 0 || N_ <= 0 || K_ <= 0) {
        return false;
    }
    const size_t a_bytes = static_cast<size_t>(M_) * static_cast<size_t>(lda_);
    const size_t b_bytes = static_cast<size_t>(N_) * static_cast<size_t>(ldb_);
    if (recompute_a_scratch_.size() < a_bytes) {
        recompute_a_scratch_.resize(a_bytes);
    }
    if (recompute_b_scratch_.size() < b_bytes) {
        recompute_b_scratch_.resize(b_bytes);
    }
    return ocl_.read_buffer(a_buf_, recompute_a_scratch_.data(), a_bytes) &&
           ocl_.read_buffer(b_buf_, recompute_b_scratch_.data(), b_bytes);
}

void Case33GemmOnednn::log_cpu_recompute_hit_(int t_rows, int t_cols, const uint32_t gpu_digest[8],
                                              const uint32_t gpu_digest_at_cmp[8],
                                              const uint32_t gpu_bound[8],
                                              const uint32_t gpu_cmp[8],
                                              const uint32_t gpu_f1_gt[8],
                                              const uint32_t gpu_f1_lt[8],
                                              const uint32_t gpu_fall_gt[8],
                                              const uint32_t gpu_fall_lt[8],
                                              const uint32_t gpu_fall_store[8],
                                              uint32_t gpu_reach_store) {
    const int hash_mr = info_.xorSubM;
    const int hash_nr = info_.xorSubN;
    if (hash_mr <= 0 || hash_nr <= 0 || (t_rows % hash_mr) != 0 || (t_cols % hash_nr) != 0) {
        std::fprintf(stderr,
                     "[onednn] cpu recompute: t_rows=%d t_cols=%d not aligned to hash tile %dx%d\n",
                     t_rows, t_cols, hash_mr, hash_nr);
        std::fflush(stderr);
        return;
    }
    const int lr = t_rows / hash_mr;
    const int lc = t_cols / hash_nr;

    std::fprintf(stderr,
                 "[onednn] cpu recompute: lr=%d lc=%d - reading device A(%dx%d) B(%dx%d)...\n", lr,
                 lc, M_, K_, N_, K_);
    std::fflush(stderr);

    if (!read_device_ab_for_recompute_()) {
        std::fprintf(stderr, "[onednn] cpu recompute: failed to read A/B from GPU\n");
        std::fflush(stderr);
        return;
    }

    uint32_t milestones[64] = {};
    uint32_t msg[cp_jackpot::kJackpotWords] = {};
    uint32_t cpu_digest[8] = {};
    case5_ngen::compute_single_hash_tile_jackpot(
            recompute_a_scratch_.data(), lda_, recompute_b_scratch_.data(), ldb_, M_, N_, K_,
            num_milestones_, milestone_k_, info_.unrollM, info_.unrollN, info_.xorSubM,
            info_.xorSubN, info_.xorSubGridM, info_.xorSubGridN, lr, lc, scan_jackpot_key_,
            milestones, msg, cpu_digest);

    const bool cpu_beat = cp_jackpot::digest_beats_target(cpu_digest, scan_jackpot_bound_);
    char gpu_hex[65];
    char cpu_hex[65];
    fmt_blake3_msb_hex(gpu_digest, gpu_hex, sizeof(gpu_hex));
    fmt_blake3_msb_hex(cpu_digest, cpu_hex, sizeof(cpu_hex));

    bool digest_match = true;
    for (int w = 0; w < 8; ++w) {
        if (gpu_digest[w] != cpu_digest[w]) {
            digest_match = false;
            break;
        }
    }

    std::fprintf(stderr, "[onednn] cpu recompute digest gpu=%s\n", gpu_hex);
    std::fprintf(stderr, "[onednn] cpu recompute digest cpu=%s digest_match=%s\n", cpu_hex,
                 digest_match ? "yes" : "NO");
    if (!digest_match) {
        dump_tile_gpu_cpu_word_compare("cpu-recomputed digest vs bound", cpu_digest,
                                       scan_jackpot_bound_, gpu_digest_at_cmp, gpu_bound, gpu_cmp,
                                       gpu_f1_gt, gpu_f1_lt, gpu_fall_gt, gpu_fall_lt,
                                       gpu_fall_store, gpu_reach_store);
        dump_tile_gpu_cpu_word_compare("gpu digest vs bound (repeat)", gpu_digest,
                                       scan_jackpot_bound_, gpu_digest_at_cmp, gpu_bound, gpu_cmp,
                                       gpu_f1_gt, gpu_f1_lt, gpu_fall_gt, gpu_fall_lt,
                                       gpu_fall_store, gpu_reach_store);
    }
    std::fprintf(stderr,
                 "[onednn] cpu recompute beat cpu_beat=%d gpu_beat flag/gpu_trace from line above\n",
                 cpu_beat ? 1 : 0);
    std::fprintf(stderr, "[onednn] cpu recompute msg=");
    for (int w = 0; w < cp_jackpot::kJackpotWords; ++w) {
        std::fprintf(stderr, "%08x", msg[w]);
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "[onednn] cpu recompute milestones (ms0..ms%d):", num_milestones_ - 1);
    for (int ms = 0; ms < num_milestones_; ++ms) {
        std::fprintf(stderr, " %08x", milestones[ms]);
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

void Case33GemmOnednn::dump_compare_panel_(int panel_tile_count) {
    if (!digest_buf_ || !beats_buf_ || panel_tile_count <= 0) {
        return;
    }
    const size_t digest_words = static_cast<size_t>(panel_tile_count) * 8u;
    const size_t beats_words = static_cast<size_t>(panel_tile_count);
    if (digest_host_.size() < digest_words) {
        digest_host_.resize(digest_words);
    }
    if (beats_host_.size() < beats_words) {
        beats_host_.resize(beats_words);
    }
    if (!ocl_.read_buffer(digest_buf_, digest_host_.data(), digest_words * sizeof(uint32_t)) ||
        !ocl_.read_buffer(beats_buf_, beats_host_.data(), beats_words * sizeof(uint32_t))) {
        std::fprintf(stderr, "[onednn] compare dump: failed to read digest/beats buffers\n");
        return;
    }

    int gpu_beats = 0;
    int cpu_beats = 0;
    int mismatch = 0;
    int false_pos = 0; // GPU yes, CPU no
    int false_neg = 0; // GPU no, CPU yes
    int unwritten = 0;
    int compared = 0;
    constexpr int kPrintMax = 32;
    int printed = 0;

    std::fprintf(stderr, "[onednn] compare dump: tiles=%d bound MSB-first:", panel_tile_count);
    for (int w = 7; w >= 0; --w) {
        std::fprintf(stderr, " %08x", scan_jackpot_bound_[w]);
    }
    std::fprintf(stderr, "\n");

    for (int s = 0; s < panel_tile_count; ++s) {
        uint32_t digest[8];
        for (int w = 0; w < 8; ++w) {
            digest[w] = digest_host_[static_cast<size_t>(s) * 8u + static_cast<size_t>(w)];
        }
        const uint32_t graw = beats_host_[static_cast<size_t>(s)];
        if (graw == 0xFFFFFFFFu) {
            ++unwritten;
            if (printed < kPrintMax && unwritten <= 8) {
                std::fprintf(stderr,
                             "[onednn] UNWRITTEN beat spatial=%d (early-exit or OOB thread)\n", s);
                ++printed;
            }
            continue;
        }
        ++compared;
        const bool cpu = cp_jackpot::digest_beats_target(digest, scan_jackpot_bound_);
        const bool gpu = graw != 0;
        if (gpu) {
            ++gpu_beats;
        }
        if (cpu) {
            ++cpu_beats;
        }
        if (gpu != cpu) {
            ++mismatch;
            if (gpu && !cpu) {
                ++false_pos;
            }
            if (!gpu && cpu) {
                ++false_neg;
            }
        }
        if ((gpu || cpu || gpu != cpu) && printed < kPrintMax) {
            char blake_hex[65];
            fmt_blake3_msb_hex(digest, blake_hex, sizeof(blake_hex));
            std::fprintf(stderr,
                         "[onednn] compare tile spat=%d gpu_beat=%d cpu_beat=%d "
                         "gpu_beat_raw=0x%08x blake3=%s\n",
                         s, gpu ? 1 : 0, cpu ? 1 : 0, graw, blake_hex);
            ++printed;
        }
    }

    std::fprintf(stderr,
                 "[onednn] compare summary: compared=%d unwritten=%d gpu_beats=%d cpu_beats=%d "
                 "mismatch=%d (false_pos=%d false_neg=%d)\n",
                 compared, unwritten, gpu_beats, cpu_beats, mismatch, false_pos, false_neg);
    if (mismatch == 0 && compared > 0) {
        std::fprintf(stderr, "[onednn] compare: GPU matches CPU digest_beats_target on written tiles\n");
    } else if (mismatch > 0) {
        std::fprintf(stderr,
                     "[onednn] compare: GPU judge disagrees with CPU — difficulty compare is broken\n");
    }
    std::fflush(stderr);
}

bool Case33GemmOnednn::scan_tile_xor_panel_host_(const uint32_t a_key8[8], const uint32_t bound[8],
                                            int panel_tile_rows, int panel_tile_cols,
                                            int panel_tile_count, int tr_base, int tc_base,
                                            int *out_found, int *out_t_rows, int *out_t_cols,
                                            uint64_t *out_tiles_scanned,
                                            const std::function<bool()> &should_cancel,
                                            const std::function<void(uint64_t)> &on_progress) {
    uint64_t tiles_scanned = out_tiles_scanned ? *out_tiles_scanned : 0;
    int found = 0;
    int hit_rows = -1;
    int hit_cols = -1;

    for (int tr = 0; tr < panel_tile_rows && !found; ++tr) {
        if (should_cancel && should_cancel()) {
            return false;
        }
        for (int tc = 0; tc < panel_tile_cols && !found; ++tc) {
            if (should_cancel && should_cancel()) {
                return false;
            }
            const size_t spatial_id =
                    static_cast<size_t>(tr) * static_cast<size_t>(panel_tile_cols) +
                    static_cast<size_t>(tc);

            uint32_t msg[cp_jackpot::kJackpotWords];
            if (fused_jackpot_) {
                for (int w = 0; w < folded_msg_words_; ++w) {
                    msg[w] =
                            tile_xor_host_[static_cast<size_t>(w) *
                                                   static_cast<size_t>(panel_tile_count) +
                                           spatial_id];
                }
            } else {
                uint32_t milestone_xor[K_DIM / R_RANK];
                for (int ms = 0; ms < num_milestones_; ++ms) {
                    milestone_xor[ms] =
                            tile_xor_host_[static_cast<size_t>(ms) *
                                                   static_cast<size_t>(panel_tile_count) +
                                           spatial_id];
                }
                cp_jackpot::fold_milestones(milestone_xor, num_milestones_, msg);
            }

            ++tiles_scanned;
            if (on_progress) {
                on_progress(tiles_scanned);
            }
            uint32_t digest[8];
            cp_jackpot::b3_compress64(a_key8, msg, digest);
            if (!cp_jackpot::digest_beats_target(digest, bound)) {
                continue;
            }

            const int lr = tr_base + tr;
            const int lc = tc_base + tc;

            found = 1;
            hit_rows = lr * info_.xorSubM;
            hit_cols = lc * info_.xorSubN;
        }
    }

    if (out_tiles_scanned) {
        *out_tiles_scanned = tiles_scanned;
    }
    if (out_found && found) {
        *out_found = 1;
    }
    if (found) {
        if (out_t_rows) {
            *out_t_rows = hit_rows;
        }
        if (out_t_cols) {
            *out_t_cols = hit_cols;
        }
    }
    return true;
}

bool Case33GemmOnednn::scan_for_share(const uint32_t a_key8[8], const uint32_t bound[8],
                                      int *out_found, int *out_t_rows, int *out_t_cols,
                                      uint64_t *out_tiles_scanned,
                                      const std::function<bool()> &should_cancel,
                                      const std::function<void(uint64_t)> &on_progress) {
    if (!available_ || !a_key8 || !bound) {
        return false;
    }
    if (!jackpot_ready_) {
        return false;
    }
    if (!ensure_jackpot_bufs_()) {
        return false;
    }

    if (out_found) {
        *out_found = 0;
    }
    if (out_t_rows) {
        *out_t_rows = -1;
    }
    if (out_t_cols) {
        *out_t_cols = -1;
    }
    if (out_tiles_scanned) {
        *out_tiles_scanned = 0;
    }

    std::memcpy(scan_jackpot_key_, a_key8, sizeof(scan_jackpot_key_));
    std::memcpy(scan_jackpot_bound_, bound, sizeof(scan_jackpot_bound_));

    if (!ocl_.write_buffer(a_key_buf_, a_key8, 8 * sizeof(uint32_t)) ||
        !ocl_.write_buffer(bound_buf_, bound, 8 * sizeof(uint32_t)) ||
        !ocl_.write_buffer(found_buf_, bound, 8 * sizeof(uint32_t), 0)) {
        return false;
    }
    const int zero = 0;
    if (!ocl_.write_buffer(found_buf_, &zero, sizeof(zero), kJackpotFoundFlagOff)) {
        return false;
    }
    compare_dump_done_ = false;

    const int row_periods = hash_tile_rows_;
    const int col_periods = hash_tile_cols_;
    const int total_tiles = hash_tile_rows_ * hash_tile_cols_;
    int found = 0;
    uint64_t tiles_scanned = 0;

    for (int rpi0 = 0; rpi0 < row_periods && !found; rpi0 += row_period_batch_) {
        if (should_cancel && should_cancel()) {
            return false;
        }

        int row_batch = row_period_batch_;
        if (rpi0 + row_batch > row_periods) {
            row_batch = row_periods - rpi0;
        }

        for (int cpi0 = 0; cpi0 < col_periods && !found; cpi0 += col_period_batch_) {
            if (should_cancel && should_cancel()) {
                return false;
            }

            int col_batch = col_period_batch_;
            if (cpi0 + col_batch > col_periods) {
                col_batch = col_periods - cpi0;
            }

            const int hash_mr = info_.xorSubM;
            const int hash_nr = info_.xorSubN;
            const int m_panel = row_batch * hash_mr;
            const int n_panel = col_batch * hash_nr;
            const int64_t offset_a_rows = static_cast<int64_t>(rpi0) * hash_mr;
            const int64_t offset_b_cols =
                    static_cast<int64_t>(cpi0) * hash_nr * static_cast<int64_t>(ldb_);

            int panel_tile_rows = 0;
            int panel_tile_cols = 0;
            int panel_tile_count = 0;
            compute_tile_grid_(m_panel, n_panel, panel_tile_rows, panel_tile_cols, panel_tile_count);

            if (fused_jackpot_) {
                if (!ensure_panel_digest_bufs_(panel_tile_count) || !ensure_jackpot_bufs_()) {
                    return false;
                }
            } else if (!ensure_panel_tile_xor_buf_(panel_tile_count)) {
                return false;
            }

            const int tr_base = rpi0;
            const int tc_base = cpi0;
            int panel_found = 0;
            int panel_t_rows = -1;
            int panel_t_cols = -1;
            if (!run_gemm_jackpot_panel_(m_panel, n_panel, offset_a_rows, offset_b_cols,
                                         panel_tile_count, panel_tile_cols, tr_base, tc_base,
                                         &panel_found, &panel_t_rows, &panel_t_cols)) {
                return false;
            }

            tiles_scanned += static_cast<uint64_t>(panel_tile_count);
            if (on_progress) {
                on_progress(tiles_scanned);
            }

            if (panel_found) {
                found = 1;
                if (out_t_rows) {
                    *out_t_rows = panel_t_rows;
                }
                if (out_t_cols) {
                    *out_t_cols = panel_t_cols;
                }
            }
        }
    }

    if (out_tiles_scanned) {
        *out_tiles_scanned = tiles_scanned;
    }
    if (out_found) {
        *out_found = found ? 1 : 0;
    }
    if (!found && tiles_scanned != static_cast<uint64_t>(total_tiles)) {
        std::fprintf(stderr,
                     "[onednn] incomplete scan: tiles %llu/%d (row_batch=%d col_batch=%d)\n",
                     static_cast<unsigned long long>(tiles_scanned), total_tiles, row_period_batch_,
                     col_period_batch_);
        return false;
    }
    return true;
}
