#ifndef CP_WORKER_H
#define CP_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CP_BACKEND_NONE   = 0,
    CP_BACKEND_CPU    = 1,
    CP_BACKEND_CUDA   = 2,
    CP_BACKEND_OPENCL = 3
} CpBackendId;

/* Compile-time availability (1 if linked). */
int cp_worker_has_cpu(void);
int cp_worker_has_cuda(void);
int cp_worker_has_opencl(void);

const char* cp_worker_backend_name(void);
CpBackendId cp_worker_backend_id(void);

/* Select backend before init when several are compiled. Returns 0 on ok. */
int cp_worker_select(CpBackendId id);

void cp_worker_init(int* devices, int ndev);
void cp_worker_shutdown(void);

/* Print devices for the selected backend (OpenCL/CUDA). Returns count, or 0. */
int cp_worker_list_devices(void);
/* OpenCL-only: restrict device enumeration to platform index (-1 = all). */
void cp_worker_set_ocl_platform(int platform_index);
/* OpenCL-only: hash tile MR x NR (4x8, 8x8, or 8x16). mr<=0 restores auto detection. */
void cp_worker_set_ocl_tile(int mr, int nr);
/* OpenCL-only: GEMM issue. 0 = auto (DPI then cpm), 1 = broadcast/cpm, 2 = packed. */
void cp_worker_set_ocl_issue_mode(int mode);
/* Legacy: on → broadcast, off → auto. */
void cp_worker_set_ocl_issue_broadcast(int on);
/* OpenCL-only: broadcast cpm type. 0 = float (default), 1 = int32. */
void cp_worker_set_ocl_cpm_int(int on);
/* OpenCL-only: resolve tile size for device before init or align tests. */
void cp_worker_configure_ocl_tile(int device_index);

void cp_worker_apply_backend_defaults(void);
int cp_worker_uses_contiguous_tiles(void);
void cp_worker_set_period_gemm(int on);
void cp_worker_set_period_batch(int batch);
void cp_worker_set_row_period_batch(int batch);
void cp_worker_set_col_period_batch(int batch);
void cp_worker_set_step_major_ap(int on);
void cp_worker_set_cutlass_fused(int on);

typedef enum {
    CP_PREPACK_SEPARATE = 0, /* row-major noisy + persistent a_pre_/b_pre_ */
    CP_PREPACK_REUSE    = 1, /* row-major noisy + prepack swap into scan buf */
    CP_PREPACK_FUSED    = 2, /* noise injection directly into scan/prepack layout */
} CpPrepackMode;

void cp_worker_set_prepack_mode(CpPrepackMode mode);
/* Legacy alias for CP_PREPACK_REUSE. */
void cp_worker_set_inplace_prepack(int on);

/* CPU SIMD ISA preference (ignored on CUDA/OpenCL). */
typedef enum {
    CP_SIMD_AUTO   = 0, /* AVX2 → SSSE3 → scalar */
    CP_SIMD_AVX2   = 1,
    CP_SIMD_SSE    = 2, /* force SSSE3 path */
    CP_SIMD_SCALAR = 3,
} CpSimdIsa;

void cp_worker_set_simd_isa(CpSimdIsa isa);

/* Prefer host matrix path when non-zero (CPU backend always uses host matrices). */
int cp_worker_prefers_host_matrices(void);

/* Worker generates noisy matrices internally (CPU zero-B). */
int cp_worker_worker_handles_matrix_prep(void);
void cp_worker_begin_job(const uint8_t job_key[32], int m, int n);

/* Default tile layout for proof build (matches CP_TILE_LAYOUT_* in cp_proof.h). */
int cp_worker_default_tile_layout(void);

/*
 * One matrix attempt: prepare noisy A/B (host or device), scan for jackpot.
 * Returns 1 on share, 0 on miss, -1 on cancel/error.
 * On share with device-generated matrices, signal download may be deferred via
 * cp_worker_fetch_share_signals when h_A_sig was NULL (buffer loaned to proof).
 */
int cp_worker_mine_attempt(
    const uint8_t* ab_seed, int ab_seed_len,
    const uint8_t job_key[32],
    const uint32_t pool_tgt[8],
    int m, int n,
    int cpu_matrices,
    const int8_t* h_A_noisy, const int8_t* h_B_noisy,
    const uint8_t* a_key,
    int8_t* h_A_sig, int8_t* h_Bt_sig,
    int* out_t_rows, int* out_t_cols,
    uint64_t* out_tiles_scanned);

/* Device → host signal matrices after a share (no-op for CPU / already-host paths). */
int cp_worker_fetch_share_signals(int8_t* h_A_sig, int8_t* h_Bt_sig);

#ifdef __cplusplus
}
#endif

#endif /* CP_WORKER_H */
