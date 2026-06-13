// CUTLASS GEMM kernel: K-serial milestone mainloop with epilogue visitor per slice.
// One launch over MxN. Case 2: per-milestone accum + global |C| prefix (beta=1).
// Case 3 (kPersistentAccumAcrossMilestones): register-held prefix, no global |C|.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/trace.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          bool kPersistentAccumAcrossMilestones_ = false,
          bool kMilestoneMajorStorage_ = false>
struct GemmWithMilestoneMainloop {
public:
  static bool const kPersistentAccumAcrossMilestones =
      kPersistentAccumAcrossMilestones_;
  static bool const kMilestoneMajorStorage = kMilestoneMajorStorage_;

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  using ThreadblockShape = typename Mma::Shape;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ElementNorm = typename EpilogueVisitor::ElementNorm;
  using ElementSum = typename EpilogueVisitor::ElementSum;

  struct Arguments {
    GemmUniversalMode mode;
    GemmCoord problem_size;
    int batch_count;

    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;

    ElementNorm *ptr_Max;
    ElementSum *ptr_Sum;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    int milestone_k;

    typename EpilogueVisitor::Arguments epilogue_visitor;

    Arguments()
        : mode(GemmUniversalMode::kGemm), batch_count(1), ptr_Max(nullptr),
          ptr_Sum(nullptr), batch_stride_A(0), batch_stride_B(0),
          milestone_k(0) {}

    Arguments(GemmUniversalMode mode_, GemmCoord problem_size_, int batch_count_,
              TensorRefA ref_A_, TensorRefB ref_B_, TensorRefC ref_C_,
              TensorRefC ref_D_, ElementNorm *ptr_Max_, ElementSum *ptr_Sum_,
              int64_t batch_stride_A_, int64_t batch_stride_B_, int milestone_k_,
              typename EpilogueVisitor::Arguments epilogue_visitor_)
        : mode(mode_), problem_size(problem_size_), batch_count(batch_count_),
          ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_),
          ptr_Max(ptr_Max_), ptr_Sum(ptr_Sum_), batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_), milestone_k(milestone_k_),
          epilogue_visitor(epilogue_visitor_) {}
  };

  struct Params {
    GemmCoord problem_size;
    GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;

    void *ptr_A;
    void *ptr_B;
    ElementC *ptr_C;
    ElementC *ptr_D;

    ElementNorm *ptr_Max;
    ElementSum *ptr_Sum;

    int milestone_k;
    int64_t batch_stride_A;
    int64_t batch_stride_B;

    typename EpilogueVisitor::Params epilogue_visitor;

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0), ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr),
          ptr_D(nullptr), ptr_Max(nullptr), ptr_Sum(nullptr), milestone_k(0),
          batch_stride_A(0), batch_stride_B(0) {}

    Params(Arguments const &args)
        : problem_size(args.problem_size), swizzle_log_tile(0),
          params_A(args.ref_A.layout()), params_B(args.ref_B.layout()),
          params_C(args.ref_C.layout()), params_D(args.ref_D.layout()),
          ptr_A(args.ref_A.data()), ptr_B(args.ref_B.data()),
          ptr_C(args.ref_C.data()), ptr_D(args.ref_D.data()),
          ptr_Max(args.ptr_Max), ptr_Sum(args.ptr_Sum),
          milestone_k(args.milestone_k),
          batch_stride_A(args.batch_stride_A),
          batch_stride_B(args.batch_stride_B),
          epilogue_visitor(args.epilogue_visitor) {
      ThreadblockSwizzle threadblock_swizzle;
      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          args.problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, 1);
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epilogue;
  };

  CUTLASS_DEVICE
  GemmWithMilestoneMainloop() {}

  static Status can_implement(GemmCoord const &problem_size, int milestone_k) {
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

    if (milestone_k <= 0 || problem_size.k() % milestone_k != 0) {
      return Status::kErrorInvalidProblem;
    }
    if (milestone_k % Mma::Shape::kK != 0) {
      return Status::kErrorInvalidProblem;
    }

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      int const k_align_a =
          kMilestoneMajorStorage ? milestone_k : problem_size.k();
      if (k_align_a % kAlignmentA) {
        return Status::kErrorMisalignedOperand;
      }
    }
    if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      int const k_align_b =
          kMilestoneMajorStorage ? milestone_k : problem_size.k();
      if (k_align_b % kAlignmentB) {
        return Status::kErrorMisalignedOperand;
      }
    }
    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      if (problem_size.n() % kAlignmentC) {
        return Status::kErrorMisalignedOperand;
      }
    }
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size, args.milestone_k);
  }

  // Case 2/3 GPU kernel entry: one CTA runs K milestones (MMA + epilogue each).
  // Case 2: clear accum each m, beta=1 prefix in global C. Case 3: persistent accum.
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    int const thread_idx = threadIdx.x;
    int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int const lane_idx = threadIdx.x % 32;

    MatrixCoord const threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN);

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    using ElementCompute = typename EpilogueVisitor::ElementwiseFunctor::ElementCompute;

    int const milestone_k = params.milestone_k;
    int const num_milestones = params.problem_size.k() / milestone_k;

    typename Mma::FragmentC accumulators;
    if (kPersistentAccumAcrossMilestones) {
      accumulators.clear();
    }

    int const problem_m = params.problem_size.m();
    int const problem_n = params.problem_size.n();
    /* Step-major Ap/BpT panels are spaced by full matrix height/width, not the
     * local 128x256 CTA slice. batch_stride_* is set from full m/n in initialize(). */
    size_t const stride_A_milestone =
        params.batch_stride_A != 0
            ? static_cast<size_t>(params.batch_stride_A)
            : static_cast<size_t>(problem_m) * static_cast<size_t>(milestone_k);
    size_t const stride_B_milestone =
        params.batch_stride_B != 0
            ? static_cast<size_t>(params.batch_stride_B)
            : static_cast<size_t>(milestone_k) * static_cast<size_t>(problem_n);

    for (int m = 0; m < num_milestones; ++m) {
      if (!kPersistentAccumAcrossMilestones) {
        accumulators.clear();
      }

      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      ElementA *ptr_A_m = ptr_A;
      ElementB *ptr_B_m = ptr_B;
      int gemm_k_iterations = 0;
      MatrixCoord tb_offset_A;
      MatrixCoord tb_offset_B;
      GemmCoord problem_size_ab;

      if (kMilestoneMajorStorage) {
        ptr_A_m = ptr_A + m * stride_A_milestone;
        ptr_B_m = ptr_B + m * stride_B_milestone;
        gemm_k_iterations =
            (milestone_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
        tb_offset_A = MatrixCoord(
            threadblock_tile_offset.m() * Mma::Shape::kM, 0);
        tb_offset_B = MatrixCoord(
            0, threadblock_tile_offset.n() * Mma::Shape::kN);
        problem_size_ab = GemmCoord(problem_m, problem_n, milestone_k);
      } else {
        int const offset_k = m * milestone_k;
        int const problem_size_k = offset_k + milestone_k;
        gemm_k_iterations =
            (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
        tb_offset_A = MatrixCoord(
            threadblock_tile_offset.m() * Mma::Shape::kM, offset_k);
        tb_offset_B = MatrixCoord(
            offset_k, threadblock_tile_offset.n() * Mma::Shape::kN);
        problem_size_ab =
            GemmCoord(problem_m, problem_n, problem_size_k);
      }

      typename Mma::IteratorA iterator_A(
          params.params_A, ptr_A_m,
          {problem_size_ab.m(), problem_size_ab.k()}, thread_idx, tb_offset_A);

      typename Mma::IteratorB iterator_B(
          params.params_B, ptr_B_m,
          {problem_size_ab.k(), problem_size_ab.n()}, thread_idx, tb_offset_B);

      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

      __syncthreads();

      Epilogue epilogue(shared_storage.epilogue.epilogue, thread_idx, warp_idx,
                        lane_idx);

      ElementSum *ptr_tile_xor_m =
          params.ptr_Sum +
          static_cast<size_t>(m) *
              static_cast<size_t>(params.epilogue_visitor.milestone_stride);

      typename EpilogueVisitor::Params visitor_params = params.epilogue_visitor;
      ElementC *ptr_C = nullptr;
      ElementC *ptr_D = nullptr;
      if (kPersistentAccumAcrossMilestones) {
        visitor_params.elementwise =
            typename EpilogueVisitor::ElementwiseFunctor::Params(
                ElementCompute(1), ElementCompute(0));
        visitor_params.serial_split_k = false;
      } else {
        visitor_params.elementwise =
            typename EpilogueVisitor::ElementwiseFunctor::Params(
                ElementCompute(1), ElementCompute(m == 0 ? 0 : 1));
        visitor_params.serial_split_k = true;
        ptr_C = (m > 0) ? params.ptr_D : params.ptr_C;
        ptr_D = params.ptr_D;
      }

      EpilogueVisitor epilogue_visitor(
          visitor_params, shared_storage.epilogue.visitor,
          params.problem_size.mn(), thread_idx, warp_idx, lane_idx,
          params.params_C, params.params_D, ptr_C, ptr_D, params.ptr_Max,
          ptr_tile_xor_m, threadblock_offset,
          blockIdx.y * params.problem_size.m());

      epilogue(epilogue_visitor, accumulators);

      __syncthreads();
    }
  }
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
