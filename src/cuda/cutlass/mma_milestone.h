// MmaMilestone: continuous GEMM pipeline with XOR at milestone boundaries.
//
// Case 10 — one prologue, continuous gemm_iters-shaped loop, XOR every
// kMilestoneIters K-tiles. Hot path mirrors MmaPipelined::gemm_iters
// (countdown + clear_mask).
//
// kResidueTileIsLast: the milestone XOR is a prefix over K, so K-tiles must be
// consumed in order. CUTLASS's SIMT dp4a iterator
// (PredicatedTileIterator2dThreadTile) visits the LAST K-tile first ("residue
// tile") and then restarts from k = 0, so that first tile is skipped
// (gemm_k_iterations already counts K/kK in-order tiles). The tensor-op
// PredicatedTileIterator places its residue tile at k = 0 and runs in order,
// so nothing may be skipped there.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename MmaPipelined_, int kMilestoneIters,
          bool kResidueTileIsLast = true>
class MmaMilestone : public MmaPipelined_ {
public:
  using Base = MmaPipelined_;
  static constexpr int kItersPerMs = kMilestoneIters;
  static constexpr bool kSkipResidueTile = kResidueTileIsLast;

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

  /// Continuous in-order K pipeline; after each milestone calls cb(ms, accum)
  /// with the live accumulator fragment (the kernel's hash-tile policy turns
  /// it into per-hash-tile XOR words).
  template <typename Callback>
  CUTLASS_DEVICE
  void inline_operator(int total_iters, FragmentC &accum, IteratorA iterator_A,
                       IteratorB iterator_B, FragmentC const &src_accum,
                       Callback &&cb) {
    using WarpFragmentA = typename Base::Operator::FragmentA;
    using WarpFragmentB = typename Base::Operator::FragmentB;
    using FragmentA = typename Base::FragmentA;
    using FragmentB = typename Base::FragmentB;

    if (kSkipResidueTile)
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
        cb(ms_idx++, accum);
        since_ms = 0;
      }
    }

    if (since_ms > 0)
      cb(ms_idx, accum);

    Base::wind_down();
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
