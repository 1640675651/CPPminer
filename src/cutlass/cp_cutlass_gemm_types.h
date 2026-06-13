/* CUTLASS fused int8 GEMM + per-thread scattered tile XOR (Case 7.1 / 7.2 style). */
#pragma once

#include "cp_cutlass.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/layout/matrix.h"

#include "epilogue_visitor_tile_xor_reg_per_thread.h"
#include "epilogue_with_visitor_visit_batch.h"
#include "gemm_with_milestone_mainloop.h"

namespace cp_cutlass {

constexpr int kMilestoneK = 256;

using ElementInput = int8_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template <typename ArchTag, typename OpClassTag, typename ThreadblockShape,
          typename WarpShape, typename InstructionShape, int Stages,
          bool PersistentAccumAcrossMilestones,
          bool UseMilestoneMajorStorage, bool UsePerThreadTileXor>
struct GemmTypesFused {
  using EpilogueOpT = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 1, ElementAccumulator, ElementCompute>;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInput, LayoutA, 1, ElementInput, LayoutB, 1, ElementOutput,
      LayoutC, ElementAccumulator, OpClassTag, ArchTag, ThreadblockShape,
      WarpShape, InstructionShape, EpilogueOpT, ThreadblockSwizzle, Stages,
      false, cutlass::arch::OpMultiplyAdd,
      cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

  using EpilogueVisitor =
      cutlass::epilogue::threadblock::EpilogueVisitorTileXorRegPerThread<
          typename DefaultGemmKernel::Mma::Shape,
          DefaultGemmKernel::kThreadCount,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator,
          ElementAccumulator, EpilogueOpT>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogueSelect<
          EpilogueVisitor, typename DefaultGemmKernel::Epilogue, 1>::Epilogue;

  using GemmKernel = cutlass::gemm::kernel::GemmWithMilestoneMainloop<
      typename DefaultGemmKernel::Mma, Epilogue, ThreadblockSwizzle,
      PersistentAccumAcrossMilestones, UseMilestoneMajorStorage>;
};

/* 128x256 CTA, persistent accum, per-thread 128-cell XOR tiles. */
using Gemm128x256StepMajor = GemmTypesFused<
    cutlass::arch::Sm61, cutlass::arch::OpClassSimt,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<32, 128, 32>, cutlass::gemm::GemmShape<1, 1, 4>,
    2, true, true, true>;

using Gemm128x256RowMajor = GemmTypesFused<
    cutlass::arch::Sm61, cutlass::arch::OpClassSimt,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<32, 128, 32>, cutlass::gemm::GemmShape<1, 1, 4>,
    2, true, false, true>;

template <typename GemmTypesT>
struct FusedMilestoneGemmOp {
  typename GemmTypesT::GemmKernel::Params params;

  cutlass::Status initialize(int M, int N, int K, int full_M, int full_N,
                               ElementInput* d_A, ElementInput* d_B,
                               uint32_t* d_tile_xor, int tile_cols,
                               size_t tile_count,
                               const CpCutlassJackpotLaunch* jackpot) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename GemmTypesT::EpilogueOpT::Params linear{ElementCompute(1),
                                                    ElementCompute(0)};
    auto layout_c = LayoutC::packed({M, N});
    auto layout_a = GemmTypesT::GemmKernel::kMilestoneMajorStorage
                        ? LayoutA::packed({M, kMilestoneK})
                        : LayoutA::packed({M, K});
    auto layout_b = GemmTypesT::GemmKernel::kMilestoneMajorStorage
                        ? LayoutB::packed({kMilestoneK, N})
                        : LayoutB::packed({K, N});
    int64_t batch_stride_a = 0;
    int64_t batch_stride_b = 0;
    if (GemmTypesT::GemmKernel::kMilestoneMajorStorage) {
      batch_stride_a =
          static_cast<int64_t>(full_M) * static_cast<int64_t>(kMilestoneK);
      batch_stride_b =
          static_cast<int64_t>(full_N) * static_cast<int64_t>(kMilestoneK);
    }
    const bool fuse_jackpot = jackpot != nullptr;
    typename GemmTypesT::GemmKernel::JackpotParams jp;
    if (fuse_jackpot) {
      jp.enabled = true;
      jp.ptr_a_key8 = jackpot->d_a_key8;
      jp.ptr_found = jackpot->d_found;
      jp.ptr_out_t_rows = jackpot->d_out_t_rows;
      jp.ptr_out_t_cols = jackpot->d_out_t_cols;
      jp.row_period = jackpot->row_period;
      jp.col_period0 = jackpot->col_period0;
      for (int i = 0; i < 8; ++i) {
        jp.bound[i] = jackpot->bound[i];
      }
    }
    typename GemmTypesT::GemmKernel::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, 1,
        cutlass::TensorRef<ElementInput, LayoutA>(d_A, layout_a),
        cutlass::TensorRef<ElementInput, LayoutB>(d_B, layout_b),
        cutlass::TensorRef<ElementOutput, LayoutC>(nullptr, layout_c),
        cutlass::TensorRef<ElementOutput, LayoutC>(nullptr, layout_c),
        nullptr, d_tile_xor, batch_stride_a, batch_stride_b, kMilestoneK,
        typename GemmTypesT::EpilogueVisitor::Arguments(
            linear, tile_cols, static_cast<int>(tile_count), false, false,
            fuse_jackpot),
        jp);
    cutlass::Status st = GemmTypesT::GemmKernel::can_implement(args);
    if (st != cutlass::Status::kSuccess) {
      return st;
    }
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

}  // namespace cp_cutlass
