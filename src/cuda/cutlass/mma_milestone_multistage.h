// MmaMilestoneMultistage: Ampere-style multistage (cp.async) threadblock
// mainloop with the Case-10 milestone callback.
//
// Modelled on CUTLASS 2.11 cutlass::gemm::threadblock::MmaMultistage (same
// prologue, same mac_loop_iter: warp-tile double buffering, cp.async copies
// spread over the warp-level k-groups, one commit group per K-tile,
// cp.async.wait_group<kStages-2> + __syncthreads before a stage is read).
// MmaMultistage keeps its pipeline state private, so the loop is re-stated
// here instead of derived from; the only functional addition is the callback
// cb(ms, accum) after every kMilestoneIters K-tiles, which the fused kernel
// turns into per-hash-tile XOR
// words while the next stages' cp.async copies are still in flight.
//
// K order / residue tile: the multistage global iterators
// (PredicatedTileAccessIterator) start with the "residue" tile at k = 0 (size
// K % kK, or a full kK when K is a multiple of kK — always the case here,
// can_implement enforces it), and the first add_tile_offset({0, 1}) advances
// by the residue extent, so K-tiles are produced strictly in order
// 0, kK, 2kK, ... exactly like the Sm75 tensor-op PredicatedTileIterator.
// Nothing is skipped (contrast: the SIMT dp4a iterator visits the LAST tile
// first, see MmaMilestone::kSkipResidueTile). The --align-test-prod CPU prefix
// GEMM check verifies this on hardware (tensoropms runs this loop on sm_75).
//
// Below sm_80 CUTLASS's cp_async/cp_async_zfill fall back to synchronous
// 16-byte copies and fence/wait are no-ops; the __syncthreads in gmem_wait()
// still orders the stage ring, so the same mainloop is functional (not fast)
// on Turing. That is what lets the sm_75 box validate the K schedule.
#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

/// True for CUTLASS MmaMultistage instantiations (the Sm80 DefaultMma with
/// Stages >= 3); used to pick the milestone mainloop.
template <typename T> struct IsMmaMultistage {
  static constexpr bool value = false;
};

template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          arch::CacheOperation::Kind CacheOpA, typename IteratorB_,
          typename SmemIteratorB_, arch::CacheOperation::Kind CacheOpB,
          typename ElementC_, typename LayoutC_, typename Policy_, int Stages,
          SharedMemoryClearOption SharedMemoryClear, typename Enable>
struct IsMmaMultistage<
    MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_,
                  SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_,
                  Stages, SharedMemoryClear, Enable>> {
  static constexpr bool value = true;
};

template <typename MmaMultistage_, int kMilestoneIters>
class MmaMilestoneMultistage;

template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          arch::CacheOperation::Kind CacheOpA, typename IteratorB_,
          typename SmemIteratorB_, arch::CacheOperation::Kind CacheOpB,
          typename ElementC_, typename LayoutC_, typename Policy_, int Stages,
          SharedMemoryClearOption SharedMemoryClear, typename Enable,
          int kMilestoneIters>
class MmaMilestoneMultistage<
    MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_,
                  SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_,
                  Stages, SharedMemoryClear, Enable>,
    kMilestoneIters> : public MmaBase<Shape_, Policy_, Stages> {
public:
  using Base = MmaBase<Shape_, Policy_, Stages>;
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using Operator = typename Policy::Operator;
  using FragmentC = typename Operator::FragmentC;
  using WarpCount = typename Base::WarpCount;
  using SharedStorage = typename Base::SharedStorage;
  using ArchTag = arch::Sm80;

  static constexpr int kItersPerMs = kMilestoneIters;
  static constexpr int kStages = Stages;

  static_assert(Stages >= 3, "multistage mainloop needs >= 3 stages");
  static_assert(SharedMemoryClear != SharedMemoryClearOption::kClearLastStage,
                "kClearLastStage is not supported by the milestone mainloop");
  static_assert(kMilestoneIters >= 1, "milestone must span >= 1 K-tile");

  struct Detail {
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) /
        Base::kWarpGemmIterations;
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) /
        Base::kWarpGemmIterations;
  };

  static_assert(
      !platform::is_same<typename Operator::MathOperator,
                         arch::OpMultiplyAddFastF32>::value &&
          !platform::is_same<typename Operator::MathOperator,
                             arch::OpMultiplyAddComplexFastF32>::value,
      "staged accumulation is not supported (int8 never needs it)");

private:
  struct PipeState {
    typename Operator::FragmentA warp_loaded_frag_A_[2];
    typename Operator::TransformedFragmentA warp_transformed_frag_A_[2];
    typename Operator::FragmentB warp_loaded_frag_B_[2];
    typename Operator::TransformedFragmentB warp_transformed_frag_B_[2];
  };

  Operator warp_mma_;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  int smem_write_stage_idx_;
  int smem_read_stage_idx_;

public:
  CUTLASS_DEVICE
  MmaMilestoneMultistage(SharedStorage &shared_storage, int thread_idx,
                         int warp_idx, int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_write_stage_idx_(0), smem_read_stage_idx_(0) {
    int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int warp_idx_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int warp_idx_m = warp_idx_mn % WarpCount::kM;
    int warp_idx_n = warp_idx_mn / WarpCount::kM;
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

private:
  CUTLASS_DEVICE
  void advance_smem_read_stage() {
    ++smem_read_stage_idx_;
    if (smem_read_stage_idx_ == Base::kStages) {
      this->warp_tile_iterator_A_.add_tile_offset(
          {0, -Base::kStages * Policy::kPartitionsK *
                  Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
          {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
           0});
      smem_read_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage(IteratorA &iterator_A,
                                IteratorB &iterator_B) {
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});
    ++smem_write_stage_idx_;
    if (smem_write_stage_idx_ == Base::kStages) {
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  template <bool kZfillAll>
  CUTLASS_DEVICE void copy_group(IteratorA &iterator_A, IteratorB &iterator_B,
                                 int group_start_A, int group_start_B,
                                 int count_A, int count_B) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      if (j < count_A &&
          group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());
        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if (kZfillAll ||
              SharedMemoryClear == SharedMemoryClearOption::kZfill)
            arch::cp_async_zfill<kSrcBytes, CacheOpA>(
                dst_ptr + v, iterator_A.get(), iterator_A.valid());
          else
            arch::cp_async<kSrcBytes, CacheOpA>(dst_ptr + v, iterator_A.get(),
                                                iterator_A.valid());
          ++iterator_A;
        }
        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      if (j < count_B &&
          group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());
        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (kZfillAll ||
              SharedMemoryClear == SharedMemoryClearOption::kZfill)
            arch::cp_async_zfill<kSrcBytes, CacheOpB>(
                dst_ptr + v, iterator_B.get(), iterator_B.valid());
          else
            arch::cp_async<kSrcBytes, CacheOpB>(dst_ptr + v, iterator_B.get(),
                                                iterator_B.valid());
          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// MmaMultistage::prologue: fill kStages-1 stages (cp.async zfill), one
  /// commit group per stage.
  CUTLASS_DEVICE
  void prologue(IteratorA &iterator_A, IteratorB &iterator_B,
                int &gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);
      copy_group<true>(iterator_A, iterator_B, 0, 0,
                       Detail::AsyncCopyIterationsPerStageA,
                       Detail::AsyncCopyIterationsPerStageB);
      advance_smem_write_stage(iterator_A, iterator_B);
      arch::cp_async_fence();
    }
  }

  /// At least one committed stage is resident and visible to all warps.
  CUTLASS_DEVICE
  void gmem_wait() {
    arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }

  /// MmaMultistage::mac_loop_iter: one K-tile of warp MMAs from the read
  /// stage while the copies for K-tile (current + kStages - 1) are issued.
  CUTLASS_DEVICE
  void mac_loop_iter(PipeState &pipe_state, FragmentC &accum,
                     IteratorA &iterator_A, IteratorB &iterator_B,
                     int &gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
         ++warp_mma_k) {
      this->warp_tile_iterator_A_.set_kgroup_index(
          (warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      this->warp_tile_iterator_B_.set_kgroup_index(
          (warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      if (warp_mma_k > 0) {
        warp_mma_.transform(pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                            pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                            pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                            pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      warp_mma_(accum, pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2], accum);

      if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        copy_group<false>(iterator_A, iterator_B,
                          warp_mma_k * Detail::kAccessesPerGroupA,
                          warp_mma_k * Detail::kAccessesPerGroupB,
                          Detail::kAccessesPerGroupA,
                          Detail::kAccessesPerGroupB);
      }

      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
        copy_group<false>(iterator_A, iterator_B,
                          (warp_mma_k + 1) * Detail::kAccessesPerGroupA,
                          (warp_mma_k + 1) * Detail::kAccessesPerGroupB,
                          Detail::kAccessesPerGroupA,
                          Detail::kAccessesPerGroupB);
        arch::cp_async_fence();
        gmem_wait();
        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
      }

      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
        warp_mma_.transform(
            pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }
    }
  }

public:
  /// Continuous in-order K pipeline over total_iters K-tiles; after each
  /// milestone (kMilestoneIters K-tiles) calls cb(ms, accum) with the live
  /// accumulator fragment. Same contract as MmaMilestone::inline_operator
  /// (total_iters must be a multiple of kMilestoneIters).
  template <typename Callback>
  CUTLASS_DEVICE void inline_operator(int total_iters, FragmentC &accum,
                                      IteratorA iterator_A,
                                      IteratorB iterator_B,
                                      FragmentC const &src_accum,
                                      Callback &&cb) {
    int gemm_k_iterations = total_iters;
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();
    accum = src_accum;

    PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;
    warp_mma_.transform(pipe_state.warp_transformed_frag_A_[0],
                        pipe_state.warp_transformed_frag_B_[0],
                        pipe_state.warp_loaded_frag_A_[0],
                        pipe_state.warp_loaded_frag_B_[0]);

    /* MmaMultistage::gemm_iters runs while gemm_k_iterations > 1 - kStages,
     * i.e. exactly total_iters mac_loop_iter calls, one K-tile each, in K
     * order; here they are grouped per milestone. InlineXorKernel::
     * can_implement guarantees total_iters % kMilestoneIters == 0 (K is a
     * multiple of milestone_k == kMilestoneIters * kK), so there is no
     * partial last milestone. */
    int const num_ms = total_iters / kMilestoneIters;
    CUTLASS_GEMM_LOOP
    for (int ms_idx = 0; ms_idx < num_ms; ++ms_idx) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kMilestoneIters; ++i)
        mac_loop_iter(pipe_state, accum, iterator_A, iterator_B,
                      gemm_k_iterations);
      cb(ms_idx, accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      arch::cp_async_fence();
      arch::cp_async_wait<0>();
      __syncthreads();
    }
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
