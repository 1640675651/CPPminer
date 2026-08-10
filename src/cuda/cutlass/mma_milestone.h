// MmaMilestone: continuous GEMM pipeline with XOR at milestone boundaries.
//
// Case 10 — one prologue, continuous gemm_iters-shaped loop, XOR every
// kMilestoneIters K-tiles. Pre-advance iterators past residue-first so tiles
// run in order (prefix partial-GEMM XOR). Hot path mirrors MmaPipelined::
// gemm_iters (countdown + clear_mask).
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename MmaPipelined_, int kMilestoneIters>
class MmaMilestone : public MmaPipelined_ {
public:
  using Base = MmaPipelined_;
  static constexpr int kItersPerMs = kMilestoneIters;

  CUTLASS_DEVICE
  MmaMilestone(typename Base::SharedStorage &shared_storage, int thread_idx,
               int warp_idx, int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx) {}

  using Shape = typename Base::Shape;
  using FragmentC = typename Base::FragmentC;
  using IteratorA = typename Base::IteratorA;
  using IteratorB = typename Base::IteratorB;

  CUTLASS_DEVICE
  static void skip_residue_tile(IteratorA &iterator_A, IteratorB &iterator_B) {
    ++iterator_A;
    ++iterator_B;
  }

  /// Continuous in-order K pipeline; XOR after each milestone via callback(ms, xv).
  template <typename Callback>
  CUTLASS_DEVICE
  void inline_operator(int total_iters, FragmentC &accum, IteratorA iterator_A,
                       IteratorB iterator_B, FragmentC const &src_accum,
                       Callback &&cb) {
    using WarpFragmentA = typename Base::Operator::FragmentA;
    using WarpFragmentB = typename Base::Operator::FragmentB;
    using FragmentA = typename Base::FragmentA;
    using FragmentB = typename Base::FragmentB;

    skip_residue_tile(iterator_A, iterator_B);

    Base::prologue(iterator_A, iterator_B, total_iters);
    Base::gmem_wait();
    accum = src_accum;

    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    ++this->warp_tile_iterator_A_;

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    ++this->warp_tile_iterator_B_;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    int gemm_k_iterations = total_iters;
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    int ms_idx = 0;
    int since_ms = 0;

    // Same control shape as MmaPipelined::gemm_iters (countdown).
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          this->smem_iterator_A_.store(this->transform_A_(tb_frag_A));
          this->smem_iterator_B_.store(this->transform_B_(tb_frag_B));
          Base::gmem_wait();
          this->advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {
          tb_frag_A.clear();
          iterator_A.load(tb_frag_A);
          ++iterator_A;
          tb_frag_B.clear();
          iterator_B.load(tb_frag_B);
          ++iterator_B;

          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        this->warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                       warp_frag_B[warp_mma_k % 2], accum);
      }

      ++since_ms;

      if (since_ms == kMilestoneIters) {
        uint32_t xv = 0u;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentC::kElements; ++i)
          xv ^= static_cast<uint32_t>(accum[i]);
        cb(ms_idx++, xv);
        since_ms = 0;
      }
    }

    if (since_ms > 0) {
      uint32_t xv = 0u;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentC::kElements; ++i)
        xv ^= static_cast<uint32_t>(accum[i]);
      cb(ms_idx, xv);
    }

    Base::wind_down();
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
