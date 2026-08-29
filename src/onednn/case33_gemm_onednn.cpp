#include "case33_gemm_onednn.hpp"

#include "case5_gemm_launch.hpp"
#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_state.h"
#include "cp_util.h"
#include "onednn_intel_devices.hpp"

#include "gemmstone/problem.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace {

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

Case33GemmOnednn::~Case33GemmOnednn() {
    if (jackpot_kernel_) {
        clReleaseKernel(jackpot_kernel_);
        jackpot_kernel_ = nullptr;
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
    const size_t bytes =
            static_cast<size_t>(num_milestones_) * static_cast<size_t>(panel_tile_count) *
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
        found_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_READ_WRITE);
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
                                             &ngen_err, false);
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
    std::snprintf(backend_, sizeof(backend_),
                  "oneDNN gemmstone %s/%s igemm+tileXOR+GPUjackpot unroll %dx%d xor %dx%d wg %dx%d "
                  "sg %d ms=%d tiles=%dx%d hash=%dx%d row_batch=%d col_batch=%d",
                  info_.hwName, info_.strategyName, info_.unrollM, info_.unrollN, info_.xorSubM,
                  info_.xorSubN, info_.wgM, info_.wgN, info_.subgroupSize, num_milestones_,
                  tile_rows_, tile_cols_, hash_tile_rows_, hash_tile_cols_, row_period_batch_,
                  col_period_batch_);
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
                                       int panel_tile_cols) {
    if (!available_ || !kernel_ || !tile_xor_buf_) {
        return false;
    }

    case5_ngen::LaunchBuffers bufs;
    bufs.a = a_buf_;
    bufs.b = b_buf_;
    bufs.c = c_buf_;
    bufs.tile_xor = tile_xor_buf_;
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

    gemmstone::GEMMProblem problem;
    problem.Ta = problem.Ta_ext = gemmstone::Type::s8;
    problem.Tb = problem.Tb_ext = gemmstone::Type::s8;
    problem.Tc = problem.Tc_ext = gemmstone::Type::s32;
    problem.Ts = gemmstone::Type::f32;
    problem.alpha = 1;
    problem.beta = 0;
    problem.case5TileXor = true;
    problem.case5TileXorNop = false;
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
    err = clFinish(ocl_.queue);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] clFinish failed (%d)\n", err);
        return false;
    }

    return true;
}

bool Case33GemmOnednn::run_gpu_jackpot_panel_(int panel_tile_count, int panel_tile_cols,
                                              int tr_base, int tc_base, int *out_found) {
    if (!jackpot_ready_ || !jackpot_kernel_ || !tile_xor_buf_ || panel_tile_count <= 0) {
        return false;
    }
    if (!ensure_jackpot_bufs_()) {
        return false;
    }

    const int hash_mr = info_.xorSubM;
    const int hash_nr = info_.xorSubN;
    cl_int err = CL_SUCCESS;
    int arg = 0;
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &tile_xor_buf_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &num_milestones_);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &panel_tile_count);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &panel_tile_cols);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &tr_base);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &tc_base);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &hash_mr);
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(int), &hash_nr);
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
    err = clFinish(ocl_.queue);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] jackpot clFinish failed (%d)\n", err);
        return false;
    }

    int found = 0;
    if (!ocl_.read_buffer(found_buf_, &found, sizeof(found))) {
        return false;
    }
    if (out_found) {
        *out_found = found;
    }
    return true;
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

            uint32_t milestone_xor[K_DIM / R_RANK];
            for (int ms = 0; ms < num_milestones_; ++ms) {
                milestone_xor[ms] =
                        tile_xor_host_[static_cast<size_t>(ms) * static_cast<size_t>(panel_tile_count) +
                                       spatial_id];
            }

            ++tiles_scanned;
            if (on_progress) {
                on_progress(tiles_scanned);
            }
            if (!cp_jackpot::tile_beats_target(milestone_xor, num_milestones_, a_key8, bound)) {
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
    if (!available_ || !a_key8 || !bound || !jackpot_ready_) {
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

    if (!ocl_.write_buffer(a_key_buf_, a_key8, 8 * sizeof(uint32_t)) ||
        !ocl_.write_buffer(bound_buf_, bound, 8 * sizeof(uint32_t))) {
        return false;
    }
    const int zero = 0;
    if (!ocl_.write_buffer(found_buf_, &zero, sizeof(zero))) {
        return false;
    }

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

            if (!ensure_panel_tile_xor_buf_(panel_tile_count)) {
                return false;
            }

            if (!run_gemm_panel_(m_panel, n_panel, offset_a_rows, offset_b_cols, panel_tile_count,
                                 panel_tile_cols)) {
                return false;
            }

            const int tr_base = rpi0;
            const int tc_base = cpi0;
            int panel_found = 0;
            if (!run_gpu_jackpot_panel_(panel_tile_count, panel_tile_cols, tr_base, tc_base,
                                        &panel_found)) {
                return false;
            }

            tiles_scanned += static_cast<uint64_t>(panel_tile_count);
            if (on_progress) {
                on_progress(tiles_scanned);
            }

            if (panel_found) {
                found = 1;
                if (out_t_rows && !ocl_.read_buffer(out_rows_buf_, out_t_rows, sizeof(int))) {
                    return false;
                }
                if (out_t_cols && !ocl_.read_buffer(out_cols_buf_, out_t_cols, sizeof(int))) {
                    return false;
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
