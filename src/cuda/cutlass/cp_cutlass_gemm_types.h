/* CUTLASS fused int8 GEMM + in-register milestone XOR.
 * Case 10 (default / row-major): continuous in-order K pipeline, XOR every
 *   R_RANK/32 K-tiles (skip residue-first). No per-milestone wind_down.
 * Case 9 (step-major panels): reuse Mma + wind_down per milestone. */
#pragma once

#include "cp_config.h"
#include "cp_cutlass.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/layout/matrix.h"

#include <type_traits>

#include "epilogue_visitor_store_c.h"
#include "epilogue_with_visitor_visit_batch.h"
#include "gemm_inline_xor_kernel.h"
#include "gemm_with_milestone_mainloop.h"
#include "hash_tile_policy.h"
#include "mma_milestone.h"

namespace cp_cutlass {

constexpr int kCtaM = 128;
constexpr int kCtaN = 128;
constexpr int kCtaK = 32; /* SIMT (dp4a) K-tile */

static_assert(R_RANK % kCtaK == 0, "R_RANK must be a multiple of CTA K-tile");
constexpr int kItersPerMs = R_RANK / kCtaK; /* 128/32 = 4 (SIMT) */

using ElementInput = int8_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

/* Alignment = A/B access granularity in elements (int8): Ap/BpT rows are
 * K_DIM = 4096 bytes and CTA panels start at multiples of 128 rows, so any
 * power of two up to 16 (one 128-bit access) holds. */
/* EpilogueVectorLength: output-op vector width. The fused kernels never run
 * the epilogue (only its types/SharedStorage are used), but the tensor-op
 * DefaultEpilogueTensorOp only instantiates int32 -> int32 with width 4. */
template <typename ArchTag, typename OpClassTag, typename ThreadblockShape,
          typename WarpShape, typename InstructionShape, int Stages,
          int Alignment = 1, int EpilogueVectorLength = 1>
struct GemmTypesCommon {
  using EpilogueOpT = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, EpilogueVectorLength, ElementAccumulator, ElementCompute>;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInput, LayoutA, Alignment, ElementInput, LayoutB, Alignment,
      ElementOutput, LayoutC, ElementAccumulator, OpClassTag, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOpT,
      ThreadblockSwizzle, Stages, false, cutlass::arch::OpMultiplyAdd,
      cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

  using EpilogueVisitor =
      cutlass::epilogue::threadblock::EpilogueVisitorStoreC<
          typename DefaultGemmKernel::Mma::Shape,
          DefaultGemmKernel::kThreadCount,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator,
          ElementAccumulator, EpilogueOpT>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogueSelect<
          EpilogueVisitor, typename DefaultGemmKernel::Epilogue, 1>::Epilogue;
};

/* Case 10: continuous pipeline, contiguous K (row-major Ap/BpT).
 * Milestone every R_RANK / ThreadblockShape::kK K-tiles. */
template <typename ArchTag, typename OpClassTag, typename ThreadblockShape,
          typename WarpShape, typename InstructionShape, int Stages,
          int Alignment = 1, typename HashTilePolicy = HashTileSimt,
          int EpilogueVectorLength = 1>
struct GemmTypesCase10
    : GemmTypesCommon<ArchTag, OpClassTag, ThreadblockShape, WarpShape,
                      InstructionShape, Stages, Alignment, EpilogueVectorLength> {
  using Base = GemmTypesCommon<ArchTag, OpClassTag, ThreadblockShape, WarpShape,
                               InstructionShape, Stages, Alignment,
                               EpilogueVectorLength>;
  static_assert(R_RANK % ThreadblockShape::kK == 0,
                "R_RANK must be a multiple of the CTA K-tile");
  static constexpr int kItersPerMilestone = R_RANK / ThreadblockShape::kK;
  /* SIMT dp4a uses PredicatedTileIterator2dThreadTile (residue = last K-tile,
   * visited first); tensor-op uses PredicatedTileIterator (residue first, at
   * k = 0). See MmaMilestone. */
  static constexpr bool kResidueTileIsLast =
      std::is_same<OpClassTag, cutlass::arch::OpClassSimt>::value;
  using MmaPipelined = typename Base::DefaultGemmKernel::Mma;
  using Mma = cutlass::gemm::threadblock::MmaMilestone<
      MmaPipelined, kItersPerMilestone, kResidueTileIsLast>;
  using GemmKernel = cutlass::gemm::kernel::InlineXorKernel<
      Mma, typename Base::Epilogue, typename Base::ThreadblockSwizzle,
      HashTilePolicy>;
};

/* Tensor-op hash-tile policy for a given warp shape (threadblock 128x128). */
template <typename WarpShape>
using HashTileTensorOpFor =
    HashTileTensorOp<WarpShape, cutlass::gemm::GemmShape<128 / WarpShape::kM,
                                                         128 / WarpShape::kN, 1>,
                     WarpShape::kM / 8, WarpShape::kN / 8>;

/* Case 9: wind_down per milestone — required for step-major (non-contiguous K). */
template <typename ArchTag, typename OpClassTag, typename ThreadblockShape,
          typename WarpShape, typename InstructionShape, int Stages,
          bool PersistentAccumAcrossMilestones, bool UseMilestoneMajorStorage>
struct GemmTypesCase9
    : GemmTypesCommon<ArchTag, OpClassTag, ThreadblockShape, WarpShape,
                      InstructionShape, Stages> {
  using Base = GemmTypesCommon<ArchTag, OpClassTag, ThreadblockShape, WarpShape,
                               InstructionShape, Stages>;
  using GemmKernel = cutlass::gemm::kernel::GemmWithMilestoneMainloop<
      typename Base::DefaultGemmKernel::Mma, typename Base::Epilogue,
      typename Base::ThreadblockSwizzle, PersistentAccumAcrossMilestones,
      UseMilestoneMajorStorage,
      /*kInlineXor=*/true, /*kReuseMmaAcrossMilestones=*/true>;
};

/* SIMT dp4a (Pascal and anything without usable int8 tensor cores).
 * Alignment 4 = one dp4a int8x4 access. Note that CUTLASS's DP4A DefaultMma
 * specialization never forwards kAlignmentA/B (PredicatedTileIterator2dThreadTile
 * has no alignment parameter), so this documents the real access width but
 * does not change code generation (verified: identical SASS). */
using Gemm128x128RowMajor = GemmTypesCase10<
    cutlass::arch::Sm61, cutlass::arch::OpClassSimt,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<1, 1, 4>, 2,
    /*Alignment=*/4>;

/* Turing+ int8 tensor cores (IMMA, mma.sync.m8n8k16.s8): 128x128x64 CTA,
 * 32x64x64 warps (8 warps = 256 threads, one hash tile per thread after the
 * reduce-scatter), 2-stage MmaPipelined, 16-byte A/B accesses, milestone every
 * 2 K-tiles. */
using TensorOpWarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
using Gemm128x128TensorOp = GemmTypesCase10<
    cutlass::arch::Sm75, cutlass::arch::OpClassTensorOp,
    cutlass::gemm::GemmShape<128, 128, 64>, TensorOpWarpShape,
    cutlass::gemm::GemmShape<8, 8, 16>, 2, 16,
    HashTileTensorOpFor<TensorOpWarpShape>, 4>;

using Gemm128x128StepMajor = GemmTypesCase9<
    cutlass::arch::Sm61, cutlass::arch::OpClassSimt,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<1, 1, 4>, 2,
    true, true>;

template <typename GemmTypesT>
struct FusedMilestoneGemmOp {
  typename GemmTypesT::GemmKernel::Params params;

  cutlass::Status initialize(int M, int N, int K, int full_M, int full_N,
                             ElementInput *d_A, ElementInput *d_B,
                             uint32_t *d_tile_xor, int tile_cols,
                             size_t tile_count,
                             const CpCutlassJackpotLaunch *jackpot) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename GemmTypesT::EpilogueOpT::Params linear{ElementCompute(1),
                                                    ElementCompute(0)};
    auto layout_c = LayoutC::packed({M, N});
    constexpr bool kMsMajor =
        GemmTypesT::GemmKernel::kMilestoneMajorStorage;
    auto layout_a = kMsMajor ? LayoutA::packed({M, R_RANK})
                             : LayoutA::packed({M, K});
    auto layout_b = kMsMajor ? LayoutB::packed({R_RANK, N})
                             : LayoutB::packed({K, N});
    int64_t batch_stride_a = 0;
    int64_t batch_stride_b = 0;
    if (kMsMajor) {
      batch_stride_a =
          static_cast<int64_t>(full_M) * static_cast<int64_t>(R_RANK);
      batch_stride_b =
          static_cast<int64_t>(full_N) * static_cast<int64_t>(R_RANK);
    }
    typename GemmTypesT::GemmKernel::JackpotParams jp;
    if (jackpot != nullptr) {
      jp.enabled = true;
      jp.ptr_a_key8 = jackpot->d_a_key8;
      jp.ptr_found = jackpot->d_found;
      jp.ptr_out_t_rows = jackpot->d_out_t_rows;
      jp.ptr_out_t_cols = jackpot->d_out_t_cols;
      jp.row_period0 = jackpot->row_period0;
      jp.col_period0 = jackpot->col_period0;
      for (int i = 0; i < 8; ++i)
        jp.bound[i] = jackpot->bound[i];
    }
    typename GemmTypesT::GemmKernel::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, 1,
        cutlass::TensorRef<ElementInput, LayoutA>(d_A, layout_a),
        cutlass::TensorRef<ElementInput, LayoutB>(d_B, layout_b),
        cutlass::TensorRef<ElementOutput, LayoutC>(nullptr, layout_c),
        cutlass::TensorRef<ElementOutput, LayoutC>(nullptr, layout_c),
        nullptr, d_tile_xor, batch_stride_a, batch_stride_b, R_RANK,
        typename GemmTypesT::EpilogueVisitor::Arguments(
            linear, tile_cols, static_cast<int>(tile_count), false, false),
        jp);
    cutlass::Status st = GemmTypesT::GemmKernel::can_implement(args);
    if (st != cutlass::Status::kSuccess)
      return st;
    params = typename GemmTypesT::GemmKernel::Params(args);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status operator()() {
    using GemmKernel = typename GemmTypesT::GemmKernel;
    using ThreadblockSwizzle = typename GemmTypesT::ThreadblockSwizzle;
    dim3 grid =
        ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    int smem = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    cutlass::Kernel<GemmKernel><<<grid, block, smem>>>(params);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? cutlass::Status::kSuccess
                                : cutlass::Status::kErrorInternal;
  }
};

} // namespace cp_cutlass
