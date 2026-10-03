// Case 10: ONE continuous main loop with XOR at milestone boundaries.
// Mma is MmaMilestone (2-stage MmaPipelined, mma_milestone.h) or
// MmaMilestoneMultistage (cp.async, mma_milestone_multistage.h); both expose
// inline_operator(total_iters, accum, iterA, iterB, src, cb) with in-order K.
// CPminer: jackpot fold in the XOR callback; optional tile-xor store.
// HashTilePolicy_ (hash_tile_policy.h) maps the accumulator fragment onto the
// 256 virtual SIMT hash tiles of the 128x128 CTA so SIMT and tensor-op
// instantiations emit identical per-tile words (proof layout unchanged).
// Threadblocks larger than 128x128 run as kVirtM x kVirtN virtual 128x128
// CTAs (HashTilePolicy::virtual_cta).
#pragma once

#include "cp_cutlass_jackpot.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "hash_tile_policy.h"
#include "mma_milestone.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename MmaMilestone_, typename Epilogue_,
          typename ThreadblockSwizzle_,
          typename HashTilePolicy_ = cp_cutlass::HashTileSimt>
struct InlineXorKernel {
public:
  using Mma = MmaMilestone_;
  /* Global-memory iterator types (MmaMilestone forwards MmaPipelined's,
   * MmaMilestoneMultistage MmaMultistage's). */
  using IteratorA = typename Mma::IteratorA;
  using IteratorB = typename Mma::IteratorB;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using HashTilePolicy = HashTilePolicy_;
  static int const kTilesPerThread = HashTilePolicy::kTilesPerThread;

  /* Traits expected by FusedMilestoneGemmOp / layout asserts. */
  static bool const kPersistentAccumAcrossMilestones = true;
  static bool const kMilestoneMajorStorage = false;
  static bool const kInlineXor = true;
  static bool const kReuseMmaAcrossMilestones = true; /* continuous pipeline */
  static bool const kCase10Continuous = true;

  using ElementA = typename IteratorA::Element;
  using LayoutA = typename IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename IteratorB::Element;
  using LayoutB = typename IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  using ThreadblockShape = typename Mma::Shape;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;
  /* A threadblock is kVirtM x kVirtN virtual 128x128 CTAs of the proof
   * format (1 x 1 for the 128x128 kernels). All hash-tile indexing below --
   * tile-xor dump offsets, jackpot row/col period, t_rows/t_cols -- is in
   * virtual-CTA units, so a 256x128 or 128x256 threadblock emits exactly the
   * words and hits the 128x128 kernel would for the same C cells. */
  static int const kVirtM = HashTilePolicy::kVirtM;
  static int const kVirtN = HashTilePolicy::kVirtN;
  static_assert(ThreadblockShape::kM == 128 * kVirtM &&
                    ThreadblockShape::kN == 128 * kVirtN,
                "threadblock must be kVirtM x kVirtN virtual 128x128 CTAs");
  /* Hash tiles per (virtual) CTA: always the 256 of the SIMT lane map. */
  static int const kHashTilesPerCta = MmaLaneTile128x128::kThreadsPerCta;
  static_assert(kThreadCount * kTilesPerThread ==
                    kHashTilesPerCta * kVirtM * kVirtN,
                "hash-tile policy must cover the 256 SIMT lane tiles of every "
                "virtual CTA");

  using ElementAccumulator = typename Mma::ElementC;
  using ElementSum = typename EpilogueVisitor::ElementSum;
  using ElementNorm = typename EpilogueVisitor::ElementNorm;

  struct JackpotParams {
    const uint32_t *ptr_a_key8;
    uint32_t bound[8];
    int *ptr_found;
    int *ptr_out_t_rows;
    int *ptr_out_t_cols;
    int row_period0;
    int col_period0;
    bool enabled;

    CUTLASS_HOST_DEVICE
    JackpotParams()
        : ptr_a_key8(nullptr), ptr_found(nullptr), ptr_out_t_rows(nullptr),
          ptr_out_t_cols(nullptr), row_period0(0), col_period0(0),
          enabled(false) {
      for (int i = 0; i < 8; ++i)
        bound[i] = 0u;
    }
  };

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
    JackpotParams jackpot;

    Arguments()
        : mode(GemmUniversalMode::kGemm), batch_count(1), ptr_Max(nullptr),
          ptr_Sum(nullptr), batch_stride_A(0), batch_stride_B(0),
          milestone_k(0) {}

    Arguments(GemmUniversalMode mode_, GemmCoord problem_size_, int batch_count_,
              TensorRefA ref_A_, TensorRefB ref_B_, TensorRefC ref_C_,
              TensorRefC ref_D_, ElementNorm *ptr_Max_, ElementSum *ptr_Sum_,
              int64_t batch_stride_A_, int64_t batch_stride_B_, int milestone_k_,
              typename EpilogueVisitor::Arguments epilogue_visitor_,
              JackpotParams jackpot_ = JackpotParams())
        : mode(mode_), problem_size(problem_size_), batch_count(batch_count_),
          ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_),
          ptr_Max(ptr_Max_), ptr_Sum(ptr_Sum_), batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_), milestone_k(milestone_k_),
          epilogue_visitor(epilogue_visitor_), jackpot(jackpot_) {}
  };

  struct Params {
    GemmCoord problem_size;
    GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename IteratorA::Params params_A;
    typename IteratorB::Params params_B;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;
    void *ptr_A;
    void *ptr_B;
    ElementC *ptr_C;
    ElementC *ptr_D;
    ElementNorm *ptr_Max;
    ElementSum *ptr_Sum;
    int milestone_k;
    int gemm_k_iterations;
    int tile_cols;
    size_t milestone_stride;
    typename EpilogueVisitor::Params epilogue_visitor;
    JackpotParams jackpot;

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0), ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr),
          ptr_D(nullptr), ptr_Max(nullptr), ptr_Sum(nullptr), milestone_k(0),
          gemm_k_iterations(0), tile_cols(0), milestone_stride(0) {}

    Params(Arguments const &args)
        : problem_size(args.problem_size), swizzle_log_tile(0),
          params_A(args.ref_A.layout()), params_B(args.ref_B.layout()),
          params_C(args.ref_C.layout()), params_D(args.ref_D.layout()),
          ptr_A(args.ref_A.data()), ptr_B(args.ref_B.data()),
          ptr_C(args.ref_C.data()), ptr_D(args.ref_D.data()),
          ptr_Max(args.ptr_Max), ptr_Sum(args.ptr_Sum),
          milestone_k(args.milestone_k),
          epilogue_visitor(args.epilogue_visitor), jackpot(args.jackpot) {
      ThreadblockSwizzle threadblock_swizzle;
      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          args.problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          1);
      grid_tiled_shape = GemmCoord(grid_tiled_shape.m(), grid_tiled_shape.n(), 1);
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);

      int const M = args.problem_size.m();
      int const N = args.problem_size.n();
      int const K = args.problem_size.k();
      /* Virtual 128x128 CTA columns / words per milestone. */
      tile_cols = N / (ThreadblockShape::kN / kVirtN);
      milestone_stride =
          static_cast<size_t>(M / (ThreadblockShape::kM / kVirtM)) *
          static_cast<size_t>(tile_cols) * kHashTilesPerCta;
      if (args.epilogue_visitor.milestone_stride > 0)
        milestone_stride =
            static_cast<size_t>(args.epilogue_visitor.milestone_stride);
      gemm_k_iterations =
          (K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epi;
  };

  CUTLASS_DEVICE
  InlineXorKernel() {}

  static Status can_implement(GemmCoord const &problem_size, int milestone_k) {
    if (milestone_k <= 0 || problem_size.k() % milestone_k != 0)
      return Status::kErrorInvalidProblem;
    if (milestone_k % ThreadblockShape::kK != 0)
      return Status::kErrorInvalidProblem;
    if (milestone_k / ThreadblockShape::kK != Mma::kItersPerMs)
      return Status::kErrorInvalidProblem;
    if (problem_size.m() % ThreadblockShape::kM ||
        problem_size.n() % ThreadblockShape::kN ||
        problem_size.k() % ThreadblockShape::kK)
      return Status::kErrorInvalidProblem;
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size, args.milestone_k);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    GemmCoord tbo =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= tbo.m() ||
        params.grid_tiled_shape.n() <= tbo.n())
      return;

    if (params.jackpot.enabled && params.jackpot.ptr_found != nullptr &&
        *params.jackpot.ptr_found != 0)
      return;

    int const thread_idx = threadIdx.x;
    int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int const lane_idx = threadIdx.x % 32;

    MatrixCoord const threadblock_offset(tbo.m() * Mma::Shape::kM,
                                         tbo.n() * Mma::Shape::kN);

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    int const M = params.problem_size.m();
    int const N = params.problem_size.n();
    int const K = params.problem_size.k();
    /* Virtual 128x128 CTA of this warp (warp-uniform). */
    int vm = 0;
    int vn = 0;
    HashTilePolicy::virtual_cta(warp_idx, vm, vn);
    int const cta_r = tbo.m() * kVirtM + vm;
    int const cta_c = tbo.n() * kVirtN + vn;
    int const tile_cols = params.tile_cols;

    MatrixCoord offA(threadblock_offset.row(), 0);
    MatrixCoord offB(0, threadblock_offset.column());

    IteratorA iterA(params.params_A, ptr_A, {M, K},
                                      thread_idx, offA);
    IteratorB iterB(params.params_B, ptr_B, {K, N},
                                      thread_idx, offB);

    typename Mma::FragmentC accum;
    accum.clear();

    /* One fold state per hash tile this thread ends up owning. */
    uint32_t jackpot_words[kTilesPerThread][CP_CUTLASS_JACKPOT_WORDS];
    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < kTilesPerThread; ++t)
      for (int i = 0; i < CP_CUTLASS_JACKPOT_WORDS; ++i)
        jackpot_words[t][i] = 0u;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    if (params.ptr_Sum == nullptr) {
      /* Mining: the milestone callback is only the hash-tile XOR + jackpot
       * fold. Folding unconditionally (the words are simply unused when the
       * jackpot is disabled) keeps per-milestone predicate set-up and the
       * dump address arithmetic out of the hot loop. */
      mma.inline_operator(
          params.gemm_k_iterations, accum, iterA, iterB, accum,
          [&](int ms_idx, typename Mma::FragmentC const &acc) {
            HashTilePolicy::milestone_xor(
                acc, lane_idx, [&](int t, uint32_t xv) {
                  cp_cutlass_jackpot_fold_step(jackpot_words[t], ms_idx, xv);
                });
          });
    } else {
      /* Tile-xor dump (--align-test-prod): [milestone][virtual CTA][virtual
       * SIMT thread]; the milestone-invariant part of the offset is hoisted. */
      size_t dump_base[kTilesPerThread];
      CUTLASS_PRAGMA_UNROLL
      for (int t = 0; t < kTilesPerThread; ++t)
        dump_base[t] =
            (static_cast<size_t>(cta_r) * tile_cols + cta_c) * kHashTilesPerCta +
            HashTilePolicy::virtual_thread(warp_idx, lane_idx, t);
      mma.inline_operator(
          params.gemm_k_iterations, accum, iterA, iterB, accum,
          [&](int ms_idx, typename Mma::FragmentC const &acc) {
            HashTilePolicy::milestone_xor(
                acc, lane_idx, [&](int t, uint32_t xv) {
                  if (params.jackpot.enabled)
                    cp_cutlass_jackpot_fold_step(jackpot_words[t], ms_idx, xv);
                  params.ptr_Sum[static_cast<size_t>(ms_idx) *
                                     params.milestone_stride +
                                 dump_base[t]] = xv;
                });
          });
    }

    if (params.jackpot.enabled && params.jackpot.ptr_found != nullptr &&
        params.jackpot.ptr_a_key8 != nullptr &&
        *params.jackpot.ptr_found == 0) {
      const int row_period_eff = params.jackpot.row_period0 + cta_r;
      const int col_period_eff = params.jackpot.col_period0 + cta_c;
      CUTLASS_PRAGMA_UNROLL
      for (int t = 0; t < kTilesPerThread; ++t) {
        int const vt = HashTilePolicy::virtual_thread(warp_idx, lane_idx, t);
        cp_cutlass_jackpot_try(
            jackpot_words[t], params.jackpot.ptr_a_key8, params.jackpot.bound,
            row_period_eff, col_period_eff, vt, params.jackpot.ptr_found,
            params.jackpot.ptr_out_t_rows, params.jackpot.ptr_out_t_cols);
      }
    }

    (void)N;
    (void)shared_storage;
  }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
