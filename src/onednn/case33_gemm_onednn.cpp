#ifdef _WIN32
/* Before any header: the OpenCL/oneDNN headers may include <windows.h>. */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "case33_gemm_onednn.hpp"

#include "case5_gemm_launch.hpp"
#include "case5_xor_tile.hpp"
#include "cp_config.h"
#include "cp_jackpot.hpp"
#include "cp_state.h"
#include "cp_util.h"
#include "onednn_intel_devices.hpp"

#include "gemmstone/problem.hpp"

#include "../esimd/cp_esimd_scan.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

/* CP_INTEL_GEMM: auto (default: ESIMD when libcp_esimd loads and the GPU has
 * XMX), esimd (required), gemmstone. */
const char *intel_gemm_mode() {
    const char *v = std::getenv("CP_INTEL_GEMM");
    return (v && *v) ? v : "auto";
}

void *esimd_dlopen(const std::string &path) {
#ifdef _WIN32
    return reinterpret_cast<void *>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void *esimd_dlsym(void *lib, const char *name) {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void esimd_dlclose(void *lib) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(lib));
#else
    dlclose(lib);
#endif
}

/* CASE5_PIPELINE=0 keeps the synchronous GEMM -> jackpot -> read loop. */
bool pipeline_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("CASE5_PIPELINE");
        return !(v && v[0] == '0');
    }();
    return on;
}

constexpr size_t kJackpotFoundFlagOffFused = 8u * sizeof(uint32_t);
constexpr size_t kJackpotFoundCoordsOff =
        (kJackpotFoundFlagOffFused + sizeof(uint32_t) + 7u) & ~size_t(7u);
constexpr size_t kJackpotFoundBufBytes = kJackpotFoundCoordsOff + sizeof(uint64_t);

// Fused gemmstone kernel packs bound[+0] found[+32] coords[+40] in one buffer.
// Non-fused cp_onednn_jackpot_scan uses found_flag at buffer base (offset 0).
constexpr size_t jackpot_found_flag_off(bool fused) {
    return fused ? kJackpotFoundFlagOffFused : 0u;
}

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

// Row-major K×N with leading dimension ldb (>= N); host B^T is N×K row-major.
void pack_b_rowmajor_from_bt(const int8_t *b_bt_rm, int K, int N, int ldb, int8_t *out) {
    const size_t row_bytes = static_cast<size_t>(ldb);
    std::memset(out, 0, row_bytes * static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        int8_t *row = out + static_cast<size_t>(k) * row_bytes;
        for (int n = 0; n < N; ++n) {
            row[n] = b_bt_rm[static_cast<size_t>(n) * K + k];
        }
        if (ldb > N) {
            std::memset(row + N, 0, static_cast<size_t>(ldb - N));
        }
    }
}

// Column-major K×M with leading dimension lda (>= M); host A is row-major M×K.
void pack_a_colmajor(const int8_t *a_rm, int M, int K, int lda, int8_t *out) {
    const size_t col_stride = static_cast<size_t>(lda);
    std::memset(out, 0, col_stride * static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        int8_t *col = out + static_cast<size_t>(k) * col_stride;
        for (int i = 0; i < M; ++i) {
            col[i] = a_rm[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)];
        }
        if (lda > M) {
            std::memset(col + M, 0, static_cast<size_t>(lda - M));
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

void Case33GemmOnednn::set_gemm_layout(bool a_row_major, bool b_row_major) {
    if (context_ready_) {
        std::fprintf(stderr, "[onednn] set_gemm_layout ignored after init_context\n");
        return;
    }
    a_row_major_ = a_row_major;
    b_row_major_ = b_row_major;
}

int64_t Case33GemmOnednn::panel_offset_a_(int64_t row_offset) const {
    return a_row_major_ ? row_offset * static_cast<int64_t>(lda_) : row_offset;
}

int64_t Case33GemmOnednn::panel_offset_b_(int64_t col_offset) const {
    return b_row_major_ ? col_offset : col_offset * static_cast<int64_t>(ldb_);
}

Case33GemmOnednn::~Case33GemmOnednn() {
    release_esimd_();
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
    for (cl_mem &buf : pipe_tile_xor_) {
        if (buf) {
            clReleaseMemObject(buf);
            buf = nullptr;
        }
    }
    if (pipe_queue_) {
        clReleaseCommandQueue(pipe_queue_);
        pipe_queue_ = nullptr;
    }
}

bool Case33GemmOnednn::setup_dims_(int M, int N, int K) {
    M_ = M;
    N_ = N;
    K_ = K;
    lda_ = a_row_major_ ? case5_ngen::pad_ld_int8(K_) : case5_ngen::pad_ld_int8(M_);
    ldb_ = b_row_major_ ? case5_ngen::pad_ld_int8(N_) : case5_ngen::pad_ld_int8(K_);
    ldc_ = case5_ngen::pad_ld_int8(M_);

    constexpr int milestone_k = 128;
    if (info_.unrollK <= 0 || (K_ % info_.unrollK) != 0) {
        std::fprintf(stderr, "[onednn] K %% unrollK != 0 (K=%d unrollK=%d)\n", K_, info_.unrollK);
        return false;
    }
    /* unrollK <= milestone_k: XOR every xor_period unrolled panels. Larger
     * (systolic) unrolls schedule the XOR every milestone_k inside the panel. */
    if (((milestone_k % info_.unrollK) != 0 && (info_.unrollK % milestone_k) != 0)
        || (K_ % milestone_k) != 0) {
        std::fprintf(stderr,
                     "[onednn] milestone_k=%d must divide or be a multiple of unrollK=%d, "
                     "and divide K=%d\n", milestone_k, info_.unrollK, K_);
        return false;
    }
    xor_period_ = info_.unrollK <= milestone_k ? milestone_k / info_.unrollK : 1;
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
    // Active unroll panels only (panels are exact multiples of unroll). Do not rnd_up to WG
    // here — launch padding does not produce tile_xor stores inside the panel bounds.
    int threads_m = div_up(m, info_.unrollM);
    int threads_n = div_up(n, info_.unrollN);
    if (info_.isNMK) {
        std::swap(threads_m, threads_n);
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
    const size_t a_bytes = static_cast<size_t>(lda_) *
                           static_cast<size_t>(a_row_major_ ? M_ : K_);
    const size_t b_bytes = static_cast<size_t>(ldb_) *
                           static_cast<size_t>(b_row_major_ ? K_ : N_);

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
        out_rows_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_READ_WRITE);
    }
    if (!out_cols_buf_) {
        out_cols_buf_ = ocl_.alloc_buffer(sizeof(int), CL_MEM_READ_WRITE);
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

    /* Kernel JIT/catalog ranking uses modest probe dims (gemm_xor default 256鲁); the
     * OpenCL kernel is size-independent. Production M/N only affect host tile grids. */
    constexpr int kKernelSelectM = 256;
    constexpr int kKernelSelectN = 256;
    case5_ngen::BuildParams build_dims{kKernelSelectM, kKernelSelectN, K_DIM, 0, 0, 0,
                                       a_row_major_, b_row_major_};
    build_dims.lda = a_row_major_ ? case5_ngen::pad_ld_int8(build_dims.k)
                                  : case5_ngen::pad_ld_int8(build_dims.m);
    build_dims.ldb = b_row_major_ ? case5_ngen::pad_ld_int8(build_dims.n)
                                  : case5_ngen::pad_ld_int8(build_dims.k);
    build_dims.ldc = case5_ngen::pad_ld_int8(build_dims.m);

    std::string ngen_err;
    kernel_ = case5_ngen::build_igemm_kernel(ocl_.context, ocl_.device, &build_dims, &info_,
                                             &ngen_err, false, fused_jackpot_, a_row_major_,
                                             b_row_major_);
    if (!kernel_) {
        std::fprintf(stderr, "[onednn] gemmstone kernel build failed: %s\n", ngen_err.c_str());
        std::snprintf(backend_, sizeof(backend_), "gemmstone build failed: %s", ngen_err.c_str());
        return false;
    }
    if (!info_.selectionLog.empty()) {
        std::fprintf(stderr, "[onednn] %s", info_.selectionLog.c_str());
    }
    if (try_init_esimd_()) {
        /* The ESIMD kernel hashes 16x16 contiguous tiles. */
        info_.xorSubM = CP_ESIMD_HASH_TILE;
        info_.xorSubN = CP_ESIMD_HASH_TILE;
    } else if (!std::strcmp(intel_gemm_mode(), "esimd")) {
        std::snprintf(backend_, sizeof(backend_), "CP_INTEL_GEMM=esimd unavailable");
        return false;
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
    char layout_name[8] = {};
    case5_ngen::case5_gemm_layout_name(a_row_major_, b_row_major_, layout_name, sizeof(layout_name));
    std::snprintf(backend_, sizeof(backend_),
                  "oneDNN gemmstone %s/%s %s layout=%s unroll %dx%d xor %dx%d wg %dx%d "
                  "sg %d ms=%d fold=%d tiles=%dx%d hash=%dx%d",
                  info_.hwName, info_.strategyName, scan_mode, layout_name, info_.unrollM,
                  info_.unrollN, info_.xorSubM, info_.xorSubN, info_.wgM, info_.wgN,
                  info_.subgroupSize, num_milestones_, fused_jackpot_ ? folded_msg_words_ : 0,
                  tile_rows_, tile_cols_, hash_tile_rows_, hash_tile_cols_);
    if (esimd_) {
        std::snprintf(backend_, sizeof(backend_),
                      "ESIMD XMX scan (dpas%s es=%d, %dx%d work-group tile, in-kernel jackpot) "
                      "hash=%dx%d ms=%d",
                      esimd_es_ == 8 ? "w" : "", esimd_es_, esimd_tile_m_, esimd_tile_n_,
                      hash_tile_rows_, hash_tile_cols_, num_milestones_);
    }
    context_ready_ = true;
    prep_ready_ = prep_.init(&ocl_, cp_ocl_kernel_dir(), false);
    if (!prep_ready_) {
        std::fprintf(stderr, "[onednn] GPU matrix prep init failed; using CPU fallback\n");
        if (esimd_) {
            /* Only the GPU prep writes the blocked operand layouts, and the hash
             * tile was already set for the ESIMD kernel. */
            std::fprintf(stderr, "[onednn] ESIMD scan needs GPU prep; retry with "
                                 "CP_INTEL_GEMM=gemmstone\n");
            return false;
        }
    } else if (esimd_) {
        prep_.set_esimd_layout(true, esimd_es_);
    }
    if (!build_jackpot_kernel_()) {
        std::fprintf(stderr, "[onednn] GPU jackpot kernel init failed\n");
        return false;
    }
    available_ = false;
    return true;
}

bool Case33GemmOnednn::upload_a_(const int8_t *a_rowmajor) {
    if (!a_rowmajor) {
        return false;
    }
    const size_t raw_bytes = static_cast<size_t>(M_) * static_cast<size_t>(K_);
    const size_t bytes = static_cast<size_t>(lda_) *
                         static_cast<size_t>(a_row_major_ ? M_ : K_);
    if (a_row_major_ && lda_ == K_) {
        a_host_.assign(a_rowmajor, a_rowmajor + raw_bytes);
        return ocl_.write_buffer(a_buf_, a_rowmajor, bytes);
    }
    a_host_.resize(bytes);
    if (a_row_major_) {
        pack_a_rowmajor(a_rowmajor, M_, K_, lda_, a_host_.data());
    } else {
        pack_a_colmajor(a_rowmajor, M_, K_, lda_, a_host_.data());
    }
    return ocl_.write_buffer(a_buf_, a_host_.data(), bytes);
}

bool Case33GemmOnednn::upload_b_(const int8_t *b_bt_rowmajor) {
    if (!b_bt_rowmajor) {
        return false;
    }
    const size_t bytes = static_cast<size_t>(ldb_) *
                         static_cast<size_t>(b_row_major_ ? K_ : N_);
    b_host_.resize(bytes);
    if (b_row_major_) {
        pack_b_rowmajor_from_bt(b_bt_rowmajor, K_, N_, ldb_, b_host_.data());
    } else {
        pack_b_colmajor(b_bt_rowmajor, K_, N_, ldb_, b_host_.data());
    }
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
    if (!upload_b_(b_rowmajor)) {
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
    if (!prep_.prepare_job_b_gpu(b_buf_, b_noise_seed, N_, K_, ldb_, b_row_major_)) {
        return false;
    }
    available_ = true;
    return true;
}

bool Case33GemmOnednn::prepare_attempt_a(const int8_t *a_rowmajor) {
    if (!available_ || !a_rowmajor) {
        return false;
    }
    return upload_a_(a_rowmajor);
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
    return prep_.prepare_attempt_a_gpu(a_buf_, ab_seed, ab_seed_len, job_key, b_noise_seed, M_, K_,
                                       lda_, a_row_major_, salted, a_key_out);
}

bool Case33GemmOnednn::read_A_witness(const uint32_t *block_idx, int num_blocks,
                                      size_t block_bytes, uint8_t *blocks_out,
                                      uint8_t *subroots_out, uint8_t root_out[32]) const {
    if (!prep_ready_ || M_ <= 0 || K_ <= 0) {
        return false;
    }
    return prep_.read_A_witness(static_cast<size_t>(M_) * static_cast<size_t>(K_), block_idx,
                                num_blocks, block_bytes, blocks_out, subroots_out, root_out);
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
    if (lda_ == K_ && a_row_major_) {
        std::memcpy(h_A_sig, a_host_.data(), need);
        return true;
    }
    if (a_row_major_) {
        for (int i = 0; i < M_; ++i) {
            std::memcpy(h_A_sig + static_cast<size_t>(i) * K_,
                        a_host_.data() + static_cast<size_t>(i) * lda_, K_);
        }
        return true;
    }
    for (int k = 0; k < K_; ++k) {
        for (int i = 0; i < M_; ++i) {
            h_A_sig[static_cast<size_t>(i) * K_ + k] =
                    a_host_[static_cast<size_t>(k) * lda_ + i];
        }
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
        if (!found_buf_ || !bound_buf_) {
            return false;
        }
        if (!ocl_.write_buffer(found_buf_, scan_jackpot_bound_, 8u * sizeof(uint32_t), 0)) {
            return false;
        }
        const int zero = 0;
        if (!ocl_.write_buffer(found_buf_, &zero, sizeof(zero),
                               jackpot_found_flag_off(/*fused=*/true))) {
            return false;
        }
        const uint64_t zero_coords = 0;
        if (!ocl_.write_buffer(found_buf_, &zero_coords, sizeof(zero_coords),
                                    kJackpotFoundCoordsOff)) {
            return false;
        }
    } else if (!tile_xor_buf_) {
        return false;
    }

    if (!enqueue_gemm_panel_(ocl_.queue, fused_jackpot_ ? nullptr : tile_xor_buf_, m_panel, n_panel,
                             offset_a_rows, offset_b_cols, panel_tile_count, panel_tile_cols,
                             tr_base, tc_base, 0, nullptr, nullptr)) {
        return false;
    }
    if (finish_queue) {
        const cl_int err = clFinish(ocl_.queue);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[onednn] clFinish failed (%d): %s\n", err,
                         OpenClContext::error_string(err).c_str());
            return false;
        }
    }

    (void)out_found;
    return true;
}

bool Case33GemmOnednn::enqueue_gemm_panel_(cl_command_queue queue, cl_mem tile_xor, int m_panel,
                                           int n_panel, int64_t offset_a_rows,
                                           int64_t offset_b_cols, int panel_tile_count,
                                           int panel_tile_cols, int tr_base, int tc_base,
                                           cl_uint num_wait, const cl_event *wait,
                                           cl_event *done) {
    case5_ngen::LaunchBuffers bufs;
    bufs.a = a_buf_;
    bufs.b = b_buf_;
    bufs.c = c_buf_;
    bufs.tile_xor = tile_xor;
    bufs.offset_a = panel_offset_a_(offset_a_rows);
    bufs.offset_b = panel_offset_b_(offset_b_cols);
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
        bufs.found_flag = found_buf_;
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
    problem.A.layout = a_row_major_ ? gemmstone::MatrixLayout::T : gemmstone::MatrixLayout::N;
    problem.B.layout = b_row_major_ ? gemmstone::MatrixLayout::T : gemmstone::MatrixLayout::N;
    problem.C.layout = gemmstone::MatrixLayout::N;

    case5_ngen::LaunchDims dims =
            case5_ngen::compute_case5_launch_dims(info_, m_panel, n_panel);
    cl_int err = case5_ngen::bind_case5_kernel_args(kernel_, info_, problem, bufs, dims);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] clSetKernelArg failed (%d)\n", err);
        return false;
    }
    case5_ngen::apply_linear_order_launch_dims(info_, dims, m_panel, n_panel, K_);

    const size_t gws[2] = {dims.gws[0], dims.gws[1]};
    const size_t lws[2] = {dims.lws[0], dims.lws[1]};
    err = clEnqueueNDRangeKernel(queue, kernel_, 2, nullptr, gws, lws, num_wait, wait, done);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] enqueue failed (%d): %s\n", err,
                     OpenClContext::error_string(err).c_str());
        return false;
    }
    return true;
}

bool Case33GemmOnednn::debug_audit_tile_xor_panel_(int panel_tile_count, int panel_tile_cols,
                                                   int panel_tile_rows, int tr_base, int tc_base,
                                                   int row_batch, int col_batch) {
    if (!case5_ngen::case5_debug_tile_xor_zeros_enabled() || fused_jackpot_ || !tile_xor_buf_ ||
        panel_tile_count <= 0 || num_milestones_ <= 0) {
        return true;
    }

    static bool announced = false;
    if (!announced) {
        std::fprintf(stderr,
                     "[onednn] CASE5_DEBUG_TILE_XOR_ZEROS: auditing tile_xor before jackpot "
                     "(warn >= %.2f%% all-zero tiles per panel)\n",
                     case5_ngen::case5_debug_tile_xor_zero_warn_pct());
        std::fflush(stderr);
        announced = true;
    }

    const size_t words =
            static_cast<size_t>(num_milestones_) * static_cast<size_t>(panel_tile_count);
    tile_xor_host_.resize(words);
    if (!ocl_.read_buffer(tile_xor_buf_, tile_xor_host_.data(), words * sizeof(uint32_t))) {
        std::fprintf(stderr, "[onednn] tile_xor zero audit: GPU readback failed\n");
        return false;
    }

    const case5_ngen::Case5TileXorZeroAudit audit = case5_ngen::audit_case5_tile_xor_zeros(
            tile_xor_host_.data(), tile_xor_host_.size(), num_milestones_, panel_tile_count);
    case5_ngen::report_case5_tile_xor_zero_audit(audit, tr_base, tc_base, panel_tile_rows,
                                                 panel_tile_cols, row_batch, col_batch);
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
    if (!enqueue_jackpot_panel_(ocl_.queue, tile_xor_buf_, panel_tile_count, panel_tile_cols,
                                tr_base, tc_base, 0, nullptr, nullptr)) {
        return false;
    }
    if (finish_queue) {
        const cl_int err = clFinish(ocl_.queue);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[onednn] jackpot clFinish failed (%d)\n", err);
            return false;
        }

        int found = 0;
        if (!ocl_.read_buffer(found_buf_, &found, sizeof(found),
                               jackpot_found_flag_off(/*fused=*/false))) {
            return false;
        }
        if (out_found) {
            *out_found = found;
        }
    }

    return true;
}

bool Case33GemmOnednn::enqueue_jackpot_panel_(cl_command_queue queue, cl_mem tile_xor,
                                              int panel_tile_count, int panel_tile_cols,
                                              int tr_base, int tc_base, cl_uint num_wait,
                                              const cl_event *wait, cl_event *done) {
    const int hash_mr = info_.xorSubM;
    const int hash_nr = info_.xorSubN;
    const int use_folded_msg = fused_jackpot_ ? 1 : 0;
    const int jackpot_words = use_folded_msg ? folded_msg_words_ : num_milestones_;
    cl_int err = CL_SUCCESS;
    int arg = 0;
    err |= clSetKernelArg(jackpot_kernel_, arg++, sizeof(cl_mem), &tile_xor);
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
    err = clEnqueueNDRangeKernel(queue, jackpot_kernel_, 1, nullptr, &global,
                                 local > 1 ? &local : nullptr, num_wait, wait, done);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[onednn] jackpot enqueue failed (%d): %s\n", err,
                     OpenClContext::error_string(err).c_str());
        return false;
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
    if (!fused_jackpot_) {
        const int panel_tile_rows =
                panel_tile_cols > 0 ? panel_tile_count / panel_tile_cols : 0;
        const int row_batch =
                info_.xorSubM > 0 ? m_panel / info_.xorSubM : 0;
        const int col_batch =
                info_.xorSubN > 0 ? n_panel / info_.xorSubN : 0;
        if (!debug_audit_tile_xor_panel_(panel_tile_count, panel_tile_cols, panel_tile_rows,
                                         tr_base, tc_base, row_batch, col_batch)) {
            return false;
        }
    }
    if (fused_jackpot_) {
        int found = 0;
        if (!ocl_.read_buffer(found_buf_, &found, sizeof(found),
                               jackpot_found_flag_off(/*fused=*/true))) {
            return false;
        }
        if (out_found) {
            *out_found = found ? 1 : 0;
        }
        if (found) {
            uint64_t packed_coords = 0;
            if (!ocl_.read_buffer(found_buf_, &packed_coords, sizeof(packed_coords),
                                   kJackpotFoundCoordsOff)) {
                return false;
            }
            const int t_rows = static_cast<int>(static_cast<uint32_t>(packed_coords));
            const int t_cols =
                    static_cast<int>(static_cast<uint32_t>(packed_coords >> 32));
            if (out_t_rows) {
                *out_t_rows = t_rows;
            }
            if (out_t_cols) {
                *out_t_cols = t_cols;
            }
        }
        return true;
    } else if (!run_gpu_jackpot_panel_(panel_tile_count, panel_tile_cols, tr_base, tc_base,
                                       out_found, true)) {
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

bool Case33GemmOnednn::try_init_esimd_() {
    const char *mode = intel_gemm_mode();
    const bool required = !std::strcmp(mode, "esimd");
    if (!std::strcmp(mode, "gemmstone")) {
        return false;
    }
    if (fused_jackpot_ || !a_row_major_ || b_row_major_) {
        if (required) {
            std::fprintf(stderr, "[onednn] ESIMD scan needs layout TN without --fused-jackpot\n");
        }
        return false;
    }
#ifdef _WIN32
    const std::string path = cp_ocl_kernel_dir() + "\\cp_esimd.dll";
#else
    const std::string path = cp_ocl_kernel_dir() + "/libcp_esimd.so";
#endif
    void *lib = esimd_dlopen(path);
    if (!lib) {
        if (required) {
            std::fprintf(stderr, "[onednn] cannot load %s\n", path.c_str());
        }
        return false;
    }
    auto version = reinterpret_cast<cp_esimd_abi_version_fn>(esimd_dlsym(lib, "cp_esimd_abi_version"));
    auto create = reinterpret_cast<cp_esimd_create_fn>(esimd_dlsym(lib, "cp_esimd_create"));
    void *panel = esimd_dlsym(lib, "cp_esimd_scan_panel");
    void *wait = esimd_dlsym(lib, "cp_esimd_wait");
    void *destroy = esimd_dlsym(lib, "cp_esimd_destroy");
    if (!version || !create || !panel || !wait || !destroy || version() != CP_ESIMD_ABI_VERSION) {
        std::fprintf(stderr, "[onednn] %s: missing symbols or ABI mismatch\n", path.c_str());
        esimd_dlclose(lib);
        return false;
    }
    CpEsimdInfo info{};
    char err[256] = {};
    CpEsimdScan *scan = create(ocl_.context, ocl_.device, ocl_.queue, &info, err, sizeof(err));
    if (!scan) {
        if (required) {
            std::fprintf(stderr, "[onednn] ESIMD scan unavailable: %s\n", err);
        }
        esimd_dlclose(lib);
        return false;
    }
    esimd_found_ = ocl_.alloc_buffer(4 * sizeof(int), CL_MEM_READ_WRITE);
    if (!esimd_found_) {
        reinterpret_cast<cp_esimd_destroy_fn>(destroy)(scan);
        esimd_dlclose(lib);
        return false;
    }
    esimd_lib_ = lib;
    esimd_scan_ = scan;
    esimd_panel_fn_ = panel;
    esimd_wait_fn_ = wait;
    esimd_destroy_fn_ = destroy;
    esimd_es_ = info.exec_size;
    esimd_tile_m_ = info.tile_m;
    esimd_tile_n_ = info.tile_n;
    esimd_ = true;
    return true;
}

void Case33GemmOnednn::release_esimd_() {
    if (esimd_scan_) {
        reinterpret_cast<cp_esimd_destroy_fn>(esimd_destroy_fn_)(esimd_scan_);
        esimd_scan_ = nullptr;
    }
    if (esimd_found_) {
        clReleaseMemObject(esimd_found_);
        esimd_found_ = nullptr;
    }
    if (esimd_lib_) {
        esimd_dlclose(esimd_lib_);
        esimd_lib_ = nullptr;
    }
    esimd_ = false;
}

/* One ESIMD launch per panel on the miner's queue; the host keeps one panel in
 * flight and reads the previous panel's found flag after its kernel event. */
int Case33GemmOnednn::scan_esimd_(int *out_found, int *out_t_rows, int *out_t_cols,
                                  uint64_t *out_tiles_scanned,
                                  const std::function<bool()> &should_cancel,
                                  const std::function<void(uint64_t)> &on_progress) {
    struct Panel {
        int m0, n0, m, n;
    };
    std::vector<Panel> panels;
    const int hm = info_.xorSubM, hn = info_.xorSubN;
    for (int rpi0 = 0; rpi0 < hash_tile_rows_; rpi0 += row_period_batch_) {
        const int rows = std::min(row_period_batch_, hash_tile_rows_ - rpi0) * hm;
        for (int cpi0 = 0; cpi0 < hash_tile_cols_; cpi0 += col_period_batch_) {
            const int cols = std::min(col_period_batch_, hash_tile_cols_ - cpi0) * hn;
            if (rows % esimd_tile_m_ || cols % esimd_tile_n_) {
                std::fprintf(stderr, "[onednn] ESIMD panel %dx%d is not a multiple of %dx%d\n", rows,
                             cols, esimd_tile_m_, esimd_tile_n_);
                return 0;
            }
            panels.push_back(Panel{rpi0 * hm, cpi0 * hn, rows, cols});
        }
    }
    const int zero4[4] = {0, 0, 0, 0};
    if (panels.empty() || !ocl_.write_buffer(esimd_found_, zero4, sizeof(zero4))) {
        return 0;
    }
    std::memset(esimd_found_host_, 0, sizeof(esimd_found_host_));

    auto scan_panel = reinterpret_cast<cp_esimd_scan_panel_fn>(esimd_panel_fn_);
    cl_event rd_ev[2] = {};
    auto release = [](cl_event &e) {
        if (e) {
            clReleaseEvent(e);
            e = nullptr;
        }
    };
    uint64_t tiles_scanned = 0;
    int found = 0, found_slot = -1;
    bool ok = true;
    for (size_t i = 0; i < panels.size() && !found && ok; ++i) {
        if (should_cancel && should_cancel()) {
            ok = false;
            break;
        }
        const int s = static_cast<int>(i & 1);
        const Panel &p = panels[i];
        void *kernel_done = nullptr;
        if (scan_panel(esimd_scan_, a_buf_, b_buf_, esimd_found_, p.m0, p.n0, p.m, p.n,
                       scan_jackpot_key_, scan_jackpot_bound_, &kernel_done) != 0 ||
            !kernel_done) {
            ok = false;
            break;
        }
        cl_event kev = static_cast<cl_event>(kernel_done);
        release(rd_ev[s]);
        const cl_int err = clEnqueueReadBuffer(ocl_.queue, esimd_found_, CL_FALSE, 0,
                                               sizeof(esimd_found_host_[s]), esimd_found_host_[s],
                                               1, &kev, &rd_ev[s]);
        clReleaseEvent(kev);
        if (err != CL_SUCCESS) {
            ok = false;
            break;
        }
        clFlush(ocl_.queue);
        if (i >= 1) {
            const int ps = s ^ 1;
            if (clWaitForEvents(1, &rd_ev[ps]) != CL_SUCCESS) {
                ok = false;
                break;
            }
            tiles_scanned += static_cast<uint64_t>(panels[i - 1].m / hm) * (panels[i - 1].n / hn);
            if (on_progress) {
                on_progress(tiles_scanned);
            }
            if (esimd_found_host_[ps][0]) {
                found = 1;
                found_slot = ps;
            }
        }
    }
    reinterpret_cast<cp_esimd_wait_fn>(esimd_wait_fn_)(esimd_scan_);
    clFinish(ocl_.queue);
    if (ok && !found) {
        const int last = static_cast<int>((panels.size() - 1) & 1);
        tiles_scanned += static_cast<uint64_t>(panels.back().m / hm) * (panels.back().n / hn);
        if (on_progress) {
            on_progress(tiles_scanned);
        }
        if (esimd_found_host_[last][0]) {
            found = 1;
            found_slot = last;
        }
    }
    release(rd_ev[0]);
    release(rd_ev[1]);
    if (!ok) {
        return 0;
    }
    if (found) {
        /* A later panel may have overwritten the hit; read the settled copy. */
        int hit[4] = {};
        if (!ocl_.read_buffer(esimd_found_, hit, sizeof(hit))) {
            return 0;
        }
        (void)found_slot;
        if (out_t_rows) {
            *out_t_rows = hit[1];
        }
        if (out_t_cols) {
            *out_t_cols = hit[2];
        }
    }
    if (out_tiles_scanned) {
        *out_tiles_scanned = tiles_scanned;
    }
    if (out_found) {
        *out_found = found;
    }
    return 1;
}

bool Case33GemmOnednn::ensure_pipeline_(int panel_tile_count) {
    if (!pipe_queue_) {
        cl_int err = CL_SUCCESS;
        pipe_queue_ = clCreateCommandQueue(ocl_.context, ocl_.device,
                                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        if (!pipe_queue_ || err != CL_SUCCESS) {
            pipe_queue_ = nullptr;
            return false;
        }
    }
    if (pipe_tile_xor_[0] && pipe_tile_xor_[1] && pipe_tile_xor_cap_ >= panel_tile_count) {
        return true;
    }
    const size_t bytes = static_cast<size_t>(num_milestones_) *
                         static_cast<size_t>(panel_tile_count) * sizeof(uint32_t);
    for (cl_mem &buf : pipe_tile_xor_) {
        if (buf) {
            clReleaseMemObject(buf);
        }
        buf = ocl_.alloc_buffer(bytes, CL_MEM_READ_WRITE);
        if (!buf) {
            return false;
        }
    }
    pipe_tile_xor_cap_ = panel_tile_count;
    return true;
}

/* GEMM panel i and the jackpot of panel i-1 overlap on an out-of-order queue:
 * jackpot i waits for GEMM i, GEMM i waits for the jackpot that last read its
 * tile_xor buffer (i-2), and the host only waits for panel i-1's found flag.
 * Returns 1 done, 0 error/cancel, -1 pipeline unavailable. */
int Case33GemmOnednn::scan_pipelined_(int *out_found, int *out_t_rows, int *out_t_cols,
                                      uint64_t *out_tiles_scanned,
                                      const std::function<bool()> &should_cancel,
                                      const std::function<void(uint64_t)> &on_progress) {
    struct Panel {
        int m, n, tile_count, tile_cols, tr_base, tc_base;
        int64_t off_a, off_b;
    };
    std::vector<Panel> panels;
    int max_tiles = 0;
    for (int rpi0 = 0; rpi0 < hash_tile_rows_; rpi0 += row_period_batch_) {
        const int row_batch = std::min(row_period_batch_, hash_tile_rows_ - rpi0);
        for (int cpi0 = 0; cpi0 < hash_tile_cols_; cpi0 += col_period_batch_) {
            const int col_batch = std::min(col_period_batch_, hash_tile_cols_ - cpi0);
            Panel p{};
            p.m = row_batch * info_.xorSubM;
            p.n = col_batch * info_.xorSubN;
            int tile_rows = 0;
            compute_tile_grid_(p.m, p.n, tile_rows, p.tile_cols, p.tile_count);
            p.tr_base = rpi0;
            p.tc_base = cpi0;
            p.off_a = static_cast<int64_t>(rpi0) * info_.xorSubM;
            p.off_b = static_cast<int64_t>(cpi0) * info_.xorSubN;
            max_tiles = std::max(max_tiles, p.tile_count);
            panels.push_back(p);
        }
    }
    if (panels.empty() || !ensure_pipeline_(max_tiles)) {
        return -1;
    }
    pipe_found_host_[0] = pipe_found_host_[1] = 0;

    auto release = [](cl_event &e) {
        if (e) {
            clReleaseEvent(e);
            e = nullptr;
        }
    };
    cl_event gemm_ev[2] = {}, jp_ev[2] = {}, rd_ev[2] = {};
    const size_t flag_off = jackpot_found_flag_off(/*fused=*/false);
    uint64_t tiles_scanned = 0;
    int found = 0;
    bool ok = true;
    for (size_t i = 0; i < panels.size() && !found && ok; ++i) {
        if (should_cancel && should_cancel()) {
            ok = false;
            break;
        }
        const int s = static_cast<int>(i & 1);
        const Panel &p = panels[i];
        cl_event prev_reader = jp_ev[s];
        cl_event gemm_done = nullptr;
        if (!enqueue_gemm_panel_(pipe_queue_, pipe_tile_xor_[s], p.m, p.n, p.off_a, p.off_b,
                                 p.tile_count, p.tile_cols, p.tr_base, p.tc_base,
                                 prev_reader ? 1 : 0, prev_reader ? &prev_reader : nullptr,
                                 &gemm_done)) {
            ok = false;
            break;
        }
        release(jp_ev[s]);
        release(gemm_ev[s]);
        gemm_ev[s] = gemm_done;
        release(rd_ev[s]);
        if (!enqueue_jackpot_panel_(pipe_queue_, pipe_tile_xor_[s], p.tile_count, p.tile_cols,
                                    p.tr_base, p.tc_base, 1, &gemm_ev[s], &jp_ev[s]) ||
            clEnqueueReadBuffer(pipe_queue_, found_buf_, CL_FALSE, flag_off, sizeof(int),
                                &pipe_found_host_[s], 1, &jp_ev[s], &rd_ev[s]) != CL_SUCCESS) {
            ok = false;
            break;
        }
        clFlush(pipe_queue_);
        if (i >= 1) {
            const int ps = s ^ 1;
            if (clWaitForEvents(1, &rd_ev[ps]) != CL_SUCCESS) {
                ok = false;
                break;
            }
            tiles_scanned += static_cast<uint64_t>(panels[i - 1].tile_count);
            if (on_progress) {
                on_progress(tiles_scanned);
            }
            found = pipe_found_host_[ps] != 0;
        }
    }
    clFinish(pipe_queue_);
    if (ok && !found) {
        const int last = static_cast<int>((panels.size() - 1) & 1);
        tiles_scanned += static_cast<uint64_t>(panels.back().tile_count);
        if (on_progress) {
            on_progress(tiles_scanned);
        }
        found = pipe_found_host_[last] != 0;
    }
    for (int s = 0; s < 2; ++s) {
        release(gemm_ev[s]);
        release(jp_ev[s]);
        release(rd_ev[s]);
    }
    if (!ok) {
        return 0;
    }
    if (found) {
        if (out_t_rows && !ocl_.read_buffer(out_rows_buf_, out_t_rows, sizeof(int))) {
            return 0;
        }
        if (out_t_cols && !ocl_.read_buffer(out_cols_buf_, out_t_cols, sizeof(int))) {
            return 0;
        }
    }
    if (out_tiles_scanned) {
        *out_tiles_scanned = tiles_scanned;
    }
    if (out_found) {
        *out_found = found;
    }
    return 1;
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
        !ocl_.write_buffer(bound_buf_, bound, 8 * sizeof(uint32_t))) {
        return false;
    }
    if (fused_jackpot_) {
        if (!ocl_.write_buffer(found_buf_, bound, 8 * sizeof(uint32_t), 0)) {
            return false;
        }
    }
    const int zero = 0;
    if (!ocl_.write_buffer(found_buf_, &zero, sizeof(zero),
                           jackpot_found_flag_off(fused_jackpot_))) {
        return false;
    }

    if (esimd_) {
        return scan_esimd_(out_found, out_t_rows, out_t_cols, out_tiles_scanned, should_cancel,
                           on_progress) != 0;
    }
    if (!fused_jackpot_ && pipeline_enabled() &&
        !case5_ngen::case5_debug_tile_xor_zeros_enabled()) {
        const int rc = scan_pipelined_(out_found, out_t_rows, out_t_cols, out_tiles_scanned,
                                       should_cancel, on_progress);
        if (rc >= 0) {
            return rc != 0;
        }
        /* rc < 0: pipeline resources unavailable, use the synchronous loop. */
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
            const int64_t offset_b_cols = static_cast<int64_t>(cpi0) * hash_nr;

            int panel_tile_rows = 0;
            int panel_tile_cols = 0;
            int panel_tile_count = 0;
            compute_tile_grid_(m_panel, n_panel, panel_tile_rows, panel_tile_cols, panel_tile_count);

            if (fused_jackpot_) {
                if (!ensure_jackpot_bufs_()) {
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
