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

static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x256StepMajor>
    g_fused_step_major;
static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x256RowMajor>
    g_fused_row_major;

int cp_cutlass_device_ok(int dev)
{
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    return 0;
  }
  /* SIMT int8 DP4A path targets Pascal (sm_61) and similar pre-TensorCore GPUs. */
  return prop.major < 7 || (prop.major == 7 && prop.minor <= 5);
}

size_t cp_cutlass_tiles_per_batch(int batch_count)
{
  return static_cast<size_t>(batch_count) *
         static_cast<size_t>(cp_cutlass::CutlassScatteredTile128x256::
                                 kThreadsPerCta);
}

size_t cp_cutlass_tile_xor_bytes(int batch_count)
{
  const int num_steps = K_DIM / R_RANK;
  return cp_cutlass_tiles_per_batch(batch_count) * static_cast<size_t>(num_steps) *
         sizeof(uint32_t);
}

int cp_cutlass_period_batch(
    int dev, const int8_t* d_Ap, const int8_t* d_BpT, int m, int n,
    int row_period, int col_period0, int batch_count, int step_major,
    uint32_t* d_tile_xor, size_t tiles_per_batch)
{
  if (cudaSetDevice(dev) != cudaSuccess) {
    return -1;
  }

  const int M = PP_ROW_PERIOD;
  const int N_fat = batch_count * PP_COL_PERIOD;
  const int K = K_DIM;
  const int cta_cols = batch_count;
  const size_t tile_count = tiles_per_batch;

  const int8_t* d_A = d_Ap;
  const int8_t* d_B = d_BpT;
  if (step_major) {
    d_A = d_Ap + (size_t)row_period * PP_ROW_PERIOD * R_RANK;
    d_B = d_BpT + (size_t)col_period0 * PP_COL_PERIOD * R_RANK;
  } else {
    d_A = d_Ap + (size_t)row_period * PP_ROW_PERIOD * K_DIM;
    d_B = d_BpT + (size_t)col_period0 * PP_COL_PERIOD * K_DIM;
  }

  cutlass::Status st = cutlass::Status::kErrorInternal;
  if (step_major) {
    CP_CUTLASS_CHECK(g_fused_step_major.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count));
    st = g_fused_step_major();
  } else {
    CP_CUTLASS_CHECK(g_fused_row_major.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count));
    st = g_fused_row_major();
  }
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "[cutlass] kernel launch failed status %d\n",
            static_cast<int>(st));
    return -1;
  }
  return 0;
}
