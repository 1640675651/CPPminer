#include "cp_cutlass.h"
#include "cp_config.h"

#include "cp_cutlass_gemm_types.h"
#include "cp_cutlass_layout.h"

#include <cuda_runtime.h>
#include <stdio.h>

#define CP_CUTLASS_CHECK(status)                                               \
  do {                                                                         \
    cutlass::Status _st = (status);                                            \
    if (_st != cutlass::Status::kSuccess) {                                    \
      fprintf(stderr, "[cutlass] %s:%d status %d\n", __FILE__, __LINE__,       \
              static_cast<int>(_st));                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x128StepMajor>
    g_fused_step_major;
static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x128RowMajor>
    g_fused_row_major;
static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x128TensorOp>
    g_fused_tensorop;

static int g_mma_mode = CP_CUTLASS_MMA_AUTO;

void cp_cutlass_set_mma_mode(int mode)
{
  g_mma_mode = mode;
}

int cp_cutlass_mma_mode(void)
{
  return g_mma_mode;
}

static int device_has_imma(const cudaDeviceProp& prop)
{
  /* mma.sync.m8n8k16 s8 exists on sm_75 and every later architecture. */
  return prop.major > 7 || (prop.major == 7 && prop.minor >= 5);
}

int cp_cutlass_device_ok(int dev)
{
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    return 0;
  }
  if (g_mma_mode == CP_CUTLASS_MMA_TENSOROP)
    return device_has_imma(prop);
  /* SIMT dp4a needs sm_61+; the sm_75 PTX in the binary JIT-compiles on newer
   * parts. Tensor-op auto-selects on sm_75+. */
  return prop.major > 6 || (prop.major == 6 && prop.minor >= 1);
}

int cp_cutlass_mma_kind(int dev)
{
  if (g_mma_mode == CP_CUTLASS_MMA_SIMT)
    return CP_CUTLASS_MMA_SIMT;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    return CP_CUTLASS_MMA_SIMT;
  if (g_mma_mode == CP_CUTLASS_MMA_TENSOROP)
    return CP_CUTLASS_MMA_TENSOROP;
  return device_has_imma(prop) ? CP_CUTLASS_MMA_TENSOROP : CP_CUTLASS_MMA_SIMT;
}

const char* cp_cutlass_mma_kind_name(int kind)
{
  return kind == CP_CUTLASS_MMA_TENSOROP
             ? "tensorop 128x128x64 / 32x64x64 / mma.m8n8k16.s8"
             : "simt dp4a 128x128x32 / 32x64x32";
}

size_t cp_cutlass_tiles_per_batch(int row_batch_count, int col_batch_count)
{
  return static_cast<size_t>(row_batch_count) *
         static_cast<size_t>(col_batch_count) *
         static_cast<size_t>(MmaLaneTile128x128::kThreadsPerCta);
}

size_t cp_cutlass_tile_xor_bytes(int row_batch_count, int col_batch_count)
{
  const int num_steps = K_DIM / R_RANK;
  return cp_cutlass_tiles_per_batch(row_batch_count, col_batch_count) *
         static_cast<size_t>(num_steps) * sizeof(uint32_t);
}

int cp_cutlass_period_batch(
    int dev, const int8_t* d_Ap, const int8_t* d_BpT, int m, int n,
    int row_period0, int col_period0, int row_batch_count, int col_batch_count,
    int step_major, uint32_t* d_tile_xor, size_t tiles_per_batch,
    const CpCutlassJackpotLaunch* jackpot)
{
  if (cudaSetDevice(dev) != cudaSuccess) {
    return -1;
  }

  const int M = row_batch_count * CP_CUTLASS_CTA_M;
  const int N_fat = col_batch_count * CP_CUTLASS_CTA_N;
  const int K = K_DIM;
  const int cta_cols = col_batch_count;
  const size_t tile_count = tiles_per_batch;

  const int8_t* d_A = d_Ap;
  const int8_t* d_B = d_BpT;
  if (step_major) {
    d_A = d_Ap + (size_t)row_period0 * CP_CUTLASS_CTA_M * R_RANK;
    d_B = d_BpT + (size_t)col_period0 * CP_CUTLASS_CTA_N * R_RANK;
  } else {
    d_A = d_Ap + (size_t)row_period0 * CP_CUTLASS_CTA_M * K_DIM;
    d_B = d_BpT + (size_t)col_period0 * CP_CUTLASS_CTA_N * K_DIM;
  }

  /* Resolve simt/tensorop once per device (cudaGetDeviceProperties is slow). */
  static int kind_cache[64];
  static int kind_cache_mode[64];
  static bool kind_cached[64];
  int kind = CP_CUTLASS_MMA_SIMT;
  if (dev >= 0 && dev < 64) {
    if (!kind_cached[dev] || kind_cache_mode[dev] != g_mma_mode) {
      kind_cache[dev] = cp_cutlass_mma_kind(dev);
      kind_cache_mode[dev] = g_mma_mode;
      kind_cached[dev] = true;
    }
    kind = kind_cache[dev];
  } else {
    kind = cp_cutlass_mma_kind(dev);
  }

  cutlass::Status st = cutlass::Status::kErrorInternal;
  if (step_major) {
    CP_CUTLASS_CHECK(g_fused_step_major.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count, jackpot));
    st = g_fused_step_major();
  } else if (kind == CP_CUTLASS_MMA_TENSOROP) {
    CP_CUTLASS_CHECK(g_fused_tensorop.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count, jackpot));
    st = g_fused_tensorop();
  } else {
    CP_CUTLASS_CHECK(g_fused_row_major.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count, jackpot));
    st = g_fused_row_major();
  }
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "[cutlass] kernel launch failed status %d\n",
            static_cast<int>(st));
    return -1;
  }
  return 0;
}
