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
static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x128TensorOp80>
    g_fused_tensorop80;
static cp_cutlass::FusedMilestoneGemmOp<cp_cutlass::Gemm128x128TensorOpMs>
    g_fused_tensorop_ms;

static int g_mma_mode = CP_CUTLASS_MMA_AUTO;

void cp_cutlass_set_mma_mode(int mode)
{
  g_mma_mode = mode;
}

int cp_cutlass_mma_mode(void)
{
  return g_mma_mode;
}

const char* cp_cutlass_mma_mode_name(int mode)
{
  switch (mode) {
  case CP_CUTLASS_MMA_AUTO: return "auto";
  case CP_CUTLASS_MMA_SIMT: return "simt";
  case CP_CUTLASS_MMA_TENSOROP: return "tensorop";
  case CP_CUTLASS_MMA_TENSOROP80: return "tensorop80";
  case CP_CUTLASS_MMA_TENSOROP_MS: return "tensoropms";
  default: return "?";
  }
}

static int device_sm(int dev)
{
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    return 0;
  return prop.major * 10 + prop.minor;
}

/* PTX ISA version of the image of the kernel that is loaded on dev (e.g. 86
 * for the sm_86 cubin, 75 when only compute_75 PTX matches and is JITed), so
 * a kernel whose body is guarded by __CUDA_ARCH__ >= 800 is real only if the
 * version is >= 80. */
template <typename GemmTypesT>
static int kernel_ptx_version(int dev)
{
  int prev = -1;
  cudaGetDevice(&prev);
  if (prev != dev && cudaSetDevice(dev) != cudaSuccess)
    return 0;
  cudaFuncAttributes attr;
  cudaError_t e =
      cudaFuncGetAttributes(&attr, cp_cutlass::FusedKernelEntry<GemmTypesT>);
  if (prev >= 0 && prev != dev)
    cudaSetDevice(prev);
  if (e != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return attr.ptxVersion;
}

int cp_cutlass_kind_supported(int dev, int kind)
{
  const int sm = device_sm(dev);
  switch (kind) {
  case CP_CUTLASS_MMA_SIMT:
    return sm >= 61;
  case CP_CUTLASS_MMA_TENSOROP:
  case CP_CUTLASS_MMA_TENSOROP_MS:
    /* mma.sync.m8n8k16 s8 exists on sm_75 and every later architecture. */
    return sm >= 75;
  case CP_CUTLASS_MMA_TENSOROP80: {
    using T = cp_cutlass::Gemm128x128TensorOp80;
    if (sm < T::kMinCudaArch / 10)
      return 0;
    static int cache[64];
    static bool cached[64];
    if (dev >= 0 && dev < 64) {
      if (!cached[dev]) {
        cache[dev] = kernel_ptx_version<T>(dev) >= T::kMinCudaArch / 10;
        cached[dev] = true;
      }
      return cache[dev];
    }
    return kernel_ptx_version<T>(dev) >= T::kMinCudaArch / 10;
  }
  default:
    return 0;
  }
}

int cp_cutlass_mma_kind(int dev)
{
  if (g_mma_mode != CP_CUTLASS_MMA_AUTO)
    return g_mma_mode;
  if (cp_cutlass_kind_supported(dev, CP_CUTLASS_MMA_TENSOROP80))
    return CP_CUTLASS_MMA_TENSOROP80;
  if (cp_cutlass_kind_supported(dev, CP_CUTLASS_MMA_TENSOROP))
    return CP_CUTLASS_MMA_TENSOROP;
  return CP_CUTLASS_MMA_SIMT;
}

int cp_cutlass_device_ok(int dev)
{
  /* SIMT dp4a needs sm_61+; the sm_75 PTX in the binary JIT-compiles on newer
   * parts. Tensor-op needs sm_75+, tensorop80 sm_80+ and an sm_80+ image. */
  return cp_cutlass_kind_supported(dev, cp_cutlass_mma_kind(dev));
}

template <typename T>
static void format_tensorop_name(char* buf, size_t len, const char* tag,
                                 const char* instr)
{
  using K = typename T::GemmKernel;
  snprintf(buf, len,
           "%s %dx%dx%d / %dx%dx%d / %s / %d-stage%s (%d thr, %d KiB smem)", tag,
           K::ThreadblockShape::kM, K::ThreadblockShape::kN,
           K::ThreadblockShape::kK, T::WarpShapeT::kM, T::WarpShapeT::kN,
           T::WarpShapeT::kK, instr, T::kStages,
           T::kMultistage ? " cp.async" : "", K::kThreadCount,
           static_cast<int>(sizeof(typename K::SharedStorage) / 1024));
}

const char* cp_cutlass_mma_kind_name(int kind)
{
  static char name[CP_CUTLASS_MMA_KINDS][128];
  if (kind < 0 || kind >= CP_CUTLASS_MMA_KINDS)
    kind = CP_CUTLASS_MMA_SIMT;
  char* buf = name[kind];
  if (buf[0] != '\0')
    return buf;
  switch (kind) {
  case CP_CUTLASS_MMA_TENSOROP:
    format_tensorop_name<cp_cutlass::Gemm128x128TensorOp>(
        buf, sizeof(name[0]), "tensorop", "mma.m8n8k16.s8");
    break;
  case CP_CUTLASS_MMA_TENSOROP80:
    format_tensorop_name<cp_cutlass::Gemm128x128TensorOp80>(
        buf, sizeof(name[0]), "tensorop80", "mma.m16n8k32.s8");
    break;
  case CP_CUTLASS_MMA_TENSOROP_MS:
    format_tensorop_name<cp_cutlass::Gemm128x128TensorOpMs>(
        buf, sizeof(name[0]), "tensoropms", "mma.m8n8k16.s8");
    break;
  default: {
    using T = cp_cutlass::Gemm128x128RowMajor;
    snprintf(buf, sizeof(name[0]), "simt dp4a %dx%dx%d / %dx%dx%d (%d thr)",
             T::GemmKernel::ThreadblockShape::kM,
             T::GemmKernel::ThreadblockShape::kN,
             T::GemmKernel::ThreadblockShape::kK,
             T::DefaultMma::Operator::Shape::kM,
             T::DefaultMma::Operator::Shape::kN,
             T::DefaultMma::Operator::Shape::kK, T::GemmKernel::kThreadCount);
    break;
  }
  }
  return buf;
}

/* ---- hash-tile policy self-test (see cp_cutlass.h) ---------------------- */

__device__ __forceinline__ uint32_t selftest_mix(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

template <typename GemmTypesT>
__global__ void HashPolicySelftestKernel(int32_t* d_c, uint32_t* d_words,
                                         int* d_count, uint32_t seed)
{
  using GemmKernel = typename GemmTypesT::GemmKernel;
  using Mma = typename GemmKernel::Mma;
  using Operator = typename Mma::Operator;
  using FragmentC = typename Mma::FragmentC;
  using Policy = typename GemmKernel::HashTilePolicy;
  using WarpCount = typename Mma::WarpCount;
  using WarpShape = typename Operator::Shape;
  constexpr int kLdc = GemmKernel::ThreadblockShape::kN;

  const int warp_idx = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  FragmentC accum;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < FragmentC::kElements; ++i)
    accum[i] = static_cast<int32_t>(
        selftest_mix(seed + threadIdx.x * FragmentC::kElements + i));

  /* Same warp placement as the mainloops' warp tile iterators. */
  const int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
  const int warp_m = warp_idx_mn % WarpCount::kM;
  const int warp_n = warp_idx_mn / WarpCount::kM;
  typename Operator::IteratorC iter_c(
      {d_c + warp_m * WarpShape::kM * kLdc + warp_n * WarpShape::kN,
       cutlass::layout::RowMajor(kLdc)},
      lane);
  iter_c.store(accum);

  Policy::milestone_xor(accum, lane, [&](int t, uint32_t xv) {
    const int vt = Policy::virtual_thread(warp_idx, lane, t);
    d_words[vt] = xv;
    atomicAdd(&d_count[vt], 1);
  });
}

template <typename GemmTypesT>
static int hash_policy_selftest_t(int dev)
{
  constexpr int kM = GemmTypesT::GemmKernel::ThreadblockShape::kM;
  constexpr int kN = GemmTypesT::GemmKernel::ThreadblockShape::kN;
  constexpr int kTiles = MmaLaneTile128x128::kThreadsPerCta;
  static_assert(kM == 128 && kN == 128, "self-test assumes 128x128 CTAs");
  static int32_t h_c[kM * kN];
  uint32_t h_words[kTiles];
  int h_count[kTiles];
  int32_t* d_c = nullptr;
  uint32_t* d_words = nullptr;
  int* d_count = nullptr;
  int rc = -1;
  if (cudaSetDevice(dev) != cudaSuccess)
    return -1;
  if (cudaMalloc(&d_c, sizeof(h_c)) == cudaSuccess &&
      cudaMalloc(&d_words, sizeof(h_words)) == cudaSuccess &&
      cudaMalloc(&d_count, sizeof(h_count)) == cudaSuccess &&
      /* 0x5a fill: a cell the iterator does not write shows up as a hole. */
      cudaMemset(d_c, 0x5a, sizeof(h_c)) == cudaSuccess &&
      cudaMemset(d_words, 0, sizeof(h_words)) == cudaSuccess &&
      cudaMemset(d_count, 0, sizeof(h_count)) == cudaSuccess) {
    HashPolicySelftestKernel<GemmTypesT>
        <<<1, GemmTypesT::GemmKernel::kThreadCount>>>(d_c, d_words, d_count,
                                                       0x9e3779b9u);
    if (cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(h_c, d_c, sizeof(h_c), cudaMemcpyDeviceToHost) ==
            cudaSuccess &&
        cudaMemcpy(h_words, d_words, sizeof(h_words),
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(h_count, d_count, sizeof(h_count),
                   cudaMemcpyDeviceToHost) == cudaSuccess) {
      int bad = 0;
      for (int i = 0; i < kM * kN; ++i)
        if (static_cast<uint32_t>(h_c[i]) == 0x5a5a5a5au) {
          bad = 1;
          break;
        }
      static const int roffs[8] = {0, 1, 2, 3, 16, 17, 18, 19};
      static const int coffs[8] = {0, 1, 2, 3, 32, 33, 34, 35};
      for (int vt = 0; vt < kTiles; ++vt) {
        int row0 = 0, col0 = 0;
        MmaLaneTile128x128::thread_block_origin(vt, row0, col0);
        uint32_t ref = 0;
        for (int r = 0; r < 8; ++r)
          for (int c = 0; c < 8; ++c)
            ref ^= static_cast<uint32_t>(
                h_c[(row0 + roffs[r]) * kN + col0 + coffs[c]]);
        if (h_count[vt] != 1 || h_words[vt] != ref)
          bad++;
      }
      rc = bad;
    }
  }
  if (rc < 0)
    cudaGetLastError();
  cudaFree(d_c);
  cudaFree(d_words);
  cudaFree(d_count);
  return rc;
}

int cp_cutlass_hash_policy_selftest(int dev, int kind)
{
  switch (kind) {
  case CP_CUTLASS_MMA_TENSOROP:
    return hash_policy_selftest_t<cp_cutlass::Gemm128x128TensorOp>(dev);
  case CP_CUTLASS_MMA_TENSOROP80:
    return hash_policy_selftest_t<cp_cutlass::Gemm128x128TensorOp80>(dev);
  case CP_CUTLASS_MMA_TENSOROP_MS:
    return hash_policy_selftest_t<cp_cutlass::Gemm128x128TensorOpMs>(dev);
  default:
    return 0; /* SIMT: the accumulator fragment is the hash tile itself */
  }
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

  /* Resolve the kernel kind once per device (cudaGetDeviceProperties is slow). */
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
  } else if (kind == CP_CUTLASS_MMA_TENSOROP80) {
    CP_CUTLASS_CHECK(g_fused_tensorop80.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count, jackpot));
    st = g_fused_tensorop80();
  } else if (kind == CP_CUTLASS_MMA_TENSOROP_MS) {
    CP_CUTLASS_CHECK(g_fused_tensorop_ms.initialize(
        M, N_fat, K, m, n, const_cast<int8_t*>(d_A), const_cast<int8_t*>(d_B),
        d_tile_xor, cta_cols, tile_count, jackpot));
    st = g_fused_tensorop_ms();
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
