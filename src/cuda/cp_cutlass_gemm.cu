#include "cp_cutlass.h"
#include "cp_config.h"

#include "cp_cutlass_gemm_types.h"
#include "cp_cutlass_layout.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CP_CUTLASS_CHECK(status)                                               \
  do {                                                                         \
    cutlass::Status _st = (status);                                            \
    if (_st != cutlass::Status::kSuccess) {                                    \
      fprintf(stderr, "[cutlass] %s:%d status %d\n", __FILE__, __LINE__,       \
              static_cast<int>(_st));                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

/* ---- kind x threadblock tile -> GEMM instantiation ------------------------
 * The only place that maps the (CP_CUTLASS_MMA_*, CP_CUTLASS_TB_*) enums onto
 * the Case-10 GEMM types: names, kernel-image checks, the hash-policy
 * self-test and the launch all go through with_gemm_types(). A tile the kind
 * does not have (cp_cutlass_kind_has_tb) maps to its 128x128 kernel. */
template <typename T> struct GemmTag {
  using type = T;
};

template <typename Fn>
static auto with_gemm_types(int kind, int tb, Fn&& fn)
{
  using namespace cp_cutlass;
  switch (kind) {
  case CP_CUTLASS_MMA_TENSOROP:
    if (tb == CP_CUTLASS_TB_256x128)
      return fn(GemmTag<Gemm256x128TensorOp>{});
    if (tb == CP_CUTLASS_TB_128x256)
      return fn(GemmTag<Gemm128x256TensorOp>{});
    return fn(GemmTag<Gemm128x128TensorOp>{});
  case CP_CUTLASS_MMA_TENSOROP80:
    if (tb == CP_CUTLASS_TB_256x128)
      return fn(GemmTag<Gemm256x128TensorOp80>{});
    if (tb == CP_CUTLASS_TB_128x256)
      return fn(GemmTag<Gemm128x256TensorOp80>{});
    return fn(GemmTag<Gemm128x128TensorOp80>{});
  case CP_CUTLASS_MMA_TENSOROP_MS:
    return fn(GemmTag<Gemm128x128TensorOpMs>{});
  default:
    return fn(GemmTag<Gemm128x128RowMajor>{});
  }
}

template <typename T>
static constexpr bool kIsSimt =
    std::is_same<T, cp_cutlass::Gemm128x128RowMajor>::value;

/* Per-device cache of small ints (kernel images, defaults); devices past
 * kMaxCachedDevices are simply recomputed on every call. */
constexpr int kMaxCachedDevices = 64;

static int g_mma_mode = CP_CUTLASS_MMA_AUTO;
/* Explicit threadblock tile (--cuda-tb / CP_CUDA_TB): -2 = CP_CUDA_TB not read
 * yet, -1 = none (per-device default), else a CP_CUTLASS_TB_* value. */
static int g_tb_override = -2;

int cp_cutlass_tb_parse(const char* s)
{
  if (s == nullptr)
    return -1;
  if (!strcmp(s, "128x128"))
    return CP_CUTLASS_TB_128x128;
  if (!strcmp(s, "256x128"))
    return CP_CUTLASS_TB_256x128;
  if (!strcmp(s, "128x256"))
    return CP_CUTLASS_TB_128x256;
  return -1;
}

const char* cp_cutlass_tb_name(int tb)
{
  switch (tb) {
  case CP_CUTLASS_TB_256x128: return "256x128";
  case CP_CUTLASS_TB_128x256: return "128x256";
  default: return "128x128";
  }
}

void cp_cutlass_set_tb(int tb)
{
  g_tb_override = (tb >= 0 && tb < CP_CUTLASS_TB_COUNT) ? tb : -1;
}

int cp_cutlass_tb_override(void)
{
  if (g_tb_override == -2) {
    g_tb_override = -1;
    const char* env = getenv("CP_CUDA_TB");
    if (env != nullptr && env[0] != '\0') {
      g_tb_override = cp_cutlass_tb_parse(env);
      if (g_tb_override < 0)
        fprintf(stderr,
                "[cutlass] CP_CUDA_TB=%s invalid (128x128|256x128|128x256), "
                "using the per-GPU default\n",
                env);
    }
  }
  return g_tb_override;
}

int cp_cutlass_tb_default(int dev)
{
  /* Default by architecture, measured at 131072^2:
   *  sm_75 (CMP 50HX, 225 W): 256x128 61.2 vs 128x128 56.9 TMAC/s — the
   *    2-stage m8n8k16 kernel is bandwidth/power bound, the bigger tile cuts
   *    global->shared traffic per MAC by 25%;
   *  sm_86 (RTX 3090, 350 W): 128x128 97 vs 256x128 94-96 — with only one
   *    256-thread CTA per SM latency hiding loses more than the traffic saves.
   * sm_80+/89 follow sm_86 until measured otherwise. Per device, so a mixed
   * rig gets each card's own default. */
  static int cache[kMaxCachedDevices];
  static bool cached[kMaxCachedDevices];
  const bool cacheable = dev >= 0 && dev < kMaxCachedDevices;
  if (cacheable && cached[dev])
    return cache[dev];
  int major = 0;
  const int tb =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) ==
                  cudaSuccess &&
              major == 7
          ? CP_CUTLASS_TB_256x128
          : CP_CUTLASS_TB_128x128;
  if (cacheable) {
    cache[dev] = tb;
    cached[dev] = true;
  }
  return tb;
}

int cp_cutlass_tb(int dev)
{
  const int o = cp_cutlass_tb_override();
  return o >= 0 ? o : cp_cutlass_tb_default(dev);
}

int cp_cutlass_kind_has_tb(int kind, int tb)
{
  if (tb == CP_CUTLASS_TB_128x128)
    return 1;
  if (tb != CP_CUTLASS_TB_256x128 && tb != CP_CUTLASS_TB_128x256)
    return 0;
  return kind == CP_CUTLASS_MMA_TENSOROP || kind == CP_CUTLASS_MMA_TENSOROP80;
}

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

/* A kind runs on dev if the GPU has its instruction (dp4a sm_61, mma.m8n8k16
 * sm_75, mma.m16n8k32 sm_80) AND the image the driver loads for dev was
 * compiled for at least that arch. FusedKernelEntry's body is compiled out
 * below kMinCudaArch, so e.g. a --cuda-arch 61 binary on a Turing card, or a
 * --cuda-arch 75 binary on an Ampere card (JIT from compute_75 PTX), would
 * otherwise launch an empty kernel that never reports a share. All tiles of a
 * kind share the guard and the compiled arch list, so the 128x128 kernel
 * stands for the kind. */
static int kind_supported_uncached(int dev, int kind)
{
  return with_gemm_types(kind, CP_CUTLASS_TB_128x128, [&](auto tag) {
    using T = typename decltype(tag)::type;
    constexpr int kMinSm = T::kMinCudaArch / 10;
    return device_sm(dev) >= kMinSm && kernel_ptx_version<T>(dev) >= kMinSm
               ? 1
               : 0;
  });
}

int cp_cutlass_kind_supported(int dev, int kind)
{
  if (kind < CP_CUTLASS_MMA_SIMT || kind >= CP_CUTLASS_MMA_KINDS)
    return 0;
  static signed char cache[kMaxCachedDevices][CP_CUTLASS_MMA_KINDS];
  static bool cached[kMaxCachedDevices][CP_CUTLASS_MMA_KINDS];
  if (dev < 0 || dev >= kMaxCachedDevices)
    return kind_supported_uncached(dev, kind);
  if (!cached[dev][kind]) {
    cache[dev][kind] = static_cast<signed char>(kind_supported_uncached(dev, kind));
    cached[dev][kind] = true;
  }
  return cache[dev][kind];
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

const char* cp_cutlass_variant_name(int kind, int tb)
{
  static char name[CP_CUTLASS_MMA_KINDS][CP_CUTLASS_TB_COUNT][128];
  if (kind < CP_CUTLASS_MMA_SIMT || kind >= CP_CUTLASS_MMA_KINDS)
    kind = CP_CUTLASS_MMA_SIMT;
  if (tb < 0 || tb >= CP_CUTLASS_TB_COUNT || !cp_cutlass_kind_has_tb(kind, tb))
    tb = CP_CUTLASS_TB_128x128;
  char* buf = name[kind][tb];
  const size_t len = sizeof(name[0][0]);
  if (buf[0] != '\0')
    return buf;
  with_gemm_types(kind, tb, [&](auto tag) {
    using T = typename decltype(tag)::type;
    using K = typename T::GemmKernel;
    if constexpr (kIsSimt<T>) {
      snprintf(buf, len, "simt dp4a %dx%dx%d / %dx%dx%d (%d thr)",
               K::ThreadblockShape::kM, K::ThreadblockShape::kN,
               K::ThreadblockShape::kK, T::DefaultMma::Operator::Shape::kM,
               T::DefaultMma::Operator::Shape::kN,
               T::DefaultMma::Operator::Shape::kK, K::kThreadCount);
    } else {
      snprintf(buf, len,
               "%s %dx%dx%d / %dx%dx%d / %s / %d-stage%s (%d thr, %d KiB smem)",
               cp_cutlass_mma_mode_name(kind), K::ThreadblockShape::kM,
               K::ThreadblockShape::kN, K::ThreadblockShape::kK,
               T::WarpShapeT::kM, T::WarpShapeT::kN, T::WarpShapeT::kK,
               T::InstructionShapeT::kM == 16 ? "mma.m16n8k32.s8"
                                              : "mma.m8n8k16.s8",
               T::kStages, T::kMultistage ? " cp.async" : "", K::kThreadCount,
               static_cast<int>(sizeof(typename K::SharedStorage) / 1024));
    }
  });
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
  constexpr int kTilesPerCta = MmaLaneTile128x128::kThreadsPerCta;

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

  /* Words land at [virtual CTA vm*kVirtN+vn][virtual SIMT thread]. */
  int vm = 0, vn = 0;
  Policy::virtual_cta(warp_idx, vm, vn);
  const int vcta = vm * GemmKernel::kVirtN + vn;
  Policy::milestone_xor(accum, lane, [&](int t, uint32_t xv) {
    const int vt = vcta * kTilesPerCta + Policy::virtual_thread(warp_idx, lane, t);
    d_words[vt] = xv;
    atomicAdd(&d_count[vt], 1);
  });
}

template <typename GemmTypesT>
static int hash_policy_selftest_t(int dev, int* out_tiles)
{
  using GemmKernel = typename GemmTypesT::GemmKernel;
  constexpr int kM = GemmKernel::ThreadblockShape::kM;
  constexpr int kN = GemmKernel::ThreadblockShape::kN;
  constexpr int kVirtM = GemmKernel::kVirtM;
  constexpr int kVirtN = GemmKernel::kVirtN;
  constexpr int kTilesPerCta = MmaLaneTile128x128::kThreadsPerCta;
  constexpr int kTiles = kTilesPerCta * kVirtM * kVirtN;
  static_assert(kM == 128 * kVirtM && kN == 128 * kVirtN,
                "self-test: threadblock must be virtual 128x128 CTAs");
  if (out_tiles)
    *out_tiles = kTiles;
  const size_t c_bytes = sizeof(int32_t) * kM * kN;
  int32_t* h_c = static_cast<int32_t*>(malloc(c_bytes));
  uint32_t* h_words = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * kTiles));
  int* h_count = static_cast<int*>(malloc(sizeof(int) * kTiles));
  int32_t* d_c = nullptr;
  uint32_t* d_words = nullptr;
  int* d_count = nullptr;
  int rc = -1;
  if (h_c && h_words && h_count && cudaSetDevice(dev) == cudaSuccess &&
      cudaMalloc(&d_c, c_bytes) == cudaSuccess &&
      cudaMalloc(&d_words, sizeof(uint32_t) * kTiles) == cudaSuccess &&
      cudaMalloc(&d_count, sizeof(int) * kTiles) == cudaSuccess &&
      /* 0x5a fill: a cell the iterator does not write shows up as a hole. */
      cudaMemset(d_c, 0x5a, c_bytes) == cudaSuccess &&
      cudaMemset(d_words, 0, sizeof(uint32_t) * kTiles) == cudaSuccess &&
      cudaMemset(d_count, 0, sizeof(int) * kTiles) == cudaSuccess) {
    HashPolicySelftestKernel<GemmTypesT>
        <<<1, GemmKernel::kThreadCount>>>(d_c, d_words, d_count, 0x9e3779b9u);
    if (cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(h_words, d_words, sizeof(uint32_t) * kTiles,
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(h_count, d_count, sizeof(int) * kTiles,
                   cudaMemcpyDeviceToHost) == cudaSuccess) {
      int bad = 0;
      for (int i = 0; i < kM * kN; ++i)
        if (static_cast<uint32_t>(h_c[i]) == 0x5a5a5a5au) {
          bad = 1;
          break;
        }
      static const int roffs[8] = {0, 1, 2, 3, 16, 17, 18, 19};
      static const int coffs[8] = {0, 1, 2, 3, 32, 33, 34, 35};
      /* Reference: the 128x128 SIMT lane map applied to every virtual CTA. */
      for (int vm = 0; vm < kVirtM; ++vm)
        for (int vn = 0; vn < kVirtN; ++vn)
          for (int vt = 0; vt < kTilesPerCta; ++vt) {
            const int idx = (vm * kVirtN + vn) * kTilesPerCta + vt;
            int row0 = 0, col0 = 0;
            MmaLaneTile128x128::thread_block_origin(vt, row0, col0);
            row0 += vm * 128;
            col0 += vn * 128;
            uint32_t ref = 0;
            for (int r = 0; r < 8; ++r)
              for (int c = 0; c < 8; ++c)
                ref ^= static_cast<uint32_t>(
                    h_c[(row0 + roffs[r]) * kN + col0 + coffs[c]]);
            if (h_count[idx] != 1 || h_words[idx] != ref)
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
  free(h_c);
  free(h_words);
  free(h_count);
  return rc;
}

int cp_cutlass_hash_policy_selftest(int dev, int kind, int tb, int* out_tiles)
{
  if (out_tiles)
    *out_tiles = MmaLaneTile128x128::kThreadsPerCta;
  if (!cp_cutlass_kind_has_tb(kind, tb))
    tb = CP_CUTLASS_TB_128x128;
  return with_gemm_types(kind, tb, [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (kIsSimt<T>)
      return 0; /* SIMT: the accumulator fragment is the hash tile itself */
    else
      return hash_policy_selftest_t<T>(dev, out_tiles);
  });
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
  static int kind_cache[kMaxCachedDevices];
  static int kind_cache_mode[kMaxCachedDevices];
  static bool kind_cached[kMaxCachedDevices];
  int kind = CP_CUTLASS_MMA_SIMT;
  if (dev >= 0 && dev < kMaxCachedDevices) {
    if (!kind_cached[dev] || kind_cache_mode[dev] != g_mma_mode) {
      kind_cache[dev] = cp_cutlass_mma_kind(dev);
      kind_cache_mode[dev] = g_mma_mode;
      kind_cached[dev] = true;
    }
    kind = kind_cache[dev];
  } else {
    kind = cp_cutlass_mma_kind(dev);
  }

  /* Threadblock tile: 256x128 needs an even number of 128-row periods in the
   * batch, 128x256 an even number of 128-col periods; otherwise this launch
   * uses the 128x128 kernel of the same kind (identical hash tiles). */
  int tb = cp_cutlass_tb(dev);
  if (!cp_cutlass_kind_has_tb(kind, tb) || step_major)
    tb = CP_CUTLASS_TB_128x128;
  if ((tb == CP_CUTLASS_TB_256x128 && (row_batch_count % 2) != 0) ||
      (tb == CP_CUTLASS_TB_128x256 && (col_batch_count % 2) != 0)) {
    static bool warned = false;
    if (!warned) {
      fprintf(stderr,
              "[cutlass] %s tile needs an even %s period batch (got %dx%d); "
              "using 128x128 for such batches\n",
              cp_cutlass_tb_name(tb),
              tb == CP_CUTLASS_TB_256x128 ? "row" : "col", row_batch_count,
              col_batch_count);
      warned = true;
    }
    tb = CP_CUTLASS_TB_128x128;
  }

  int8_t* A = const_cast<int8_t*>(d_A);
  int8_t* B = const_cast<int8_t*>(d_B);
  /* The op only holds the kernel Params, so each launch builds its own on the
   * stack (no shared state between GPUs driven from different threads). */
  auto run = [&](auto tag) -> int {
    using T = typename decltype(tag)::type;
    cp_cutlass::FusedMilestoneGemmOp<T> op;
    CP_CUTLASS_CHECK(op.initialize(M, N_fat, K, m, n, A, B, d_tile_xor,
                                   cta_cols, tile_count, jackpot));
    const cutlass::Status st = op();
    if (st != cutlass::Status::kSuccess) {
      fprintf(stderr, "[cutlass] kernel launch failed status %d\n",
              static_cast<int>(st));
      return -1;
    }
    return 0;
  };
  if (step_major)
    return run(GemmTag<cp_cutlass::Gemm128x128StepMajor>{});
  return with_gemm_types(kind, tb, run);
}
