// EpilogueWithVisitor with batched visit(): passes kVisitVectorWidth scalar
// accumulators per visit() while OutputTileIterator::kElementsPerAccess may stay 1.
#pragma once

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
  typename Visitor_, typename Shape_, typename WarpMmaOperator_, int PartitionsK,
  typename AccumulatorFragmentIterator_, typename WarpTileIterator_,
  typename SharedLoadIterator_, typename Padding_, int FragmentsPerPartition = 1,
  int IterationsUnroll =
      (true || !IsEpilogueFunctorHeavy<Visitor_>::value)>
class EpilogueWithVisitorVisitBatch
    : public EpilogueBase<
          Shape_, typename WarpMmaOperator_::Shape, PartitionsK,
          AccumulatorFragmentIterator_, WarpTileIterator_, Padding_,
          FragmentsPerPartition> {
public:
  using Visitor = Visitor_;
  using Base = EpilogueBase<Shape_, typename WarpMmaOperator_::Shape, PartitionsK,
                            AccumulatorFragmentIterator_, WarpTileIterator_,
                            Padding_, FragmentsPerPartition>;
  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using Padding = Padding_;
  using AccumulatorTile = typename Base::AccumulatorTile;
  using ElementAccumulator = typename WarpTileIterator::Element;
  static int const kVisitVectorWidth = Visitor::kVisitVectorWidth;
  using VisitAccumulatorAccessType =
      Array<ElementAccumulator, kVisitVectorWidth>;
  using WarpCount = typename Base::WarpCount;
  static int constexpr kSmemTiles =
      Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration
                                         : kPartitionsK;
  static int constexpr kSmemPointerOffset =
      Base::SharedStorage::StorageShape::kCount / kSmemTiles;
  using SharedStorage = typename Base::SharedStorage;

  static_assert(kVisitVectorWidth > 1,
                "EpilogueWithVisitorVisitBatch requires kVisitVectorWidth > 1");
  static_assert(
      (AccumulatorTile::kElements %
       (Visitor::kIterations * kVisitVectorWidth)) == 0,
      "Accumulator tile must divide by kIterations * kVisitVectorWidth");
  static_assert(
      (SharedLoadIterator::ThreadMap::Iterations::kColumn %
       kVisitVectorWidth) == 0,
      "Epilogue column iterations must divide by kVisitVectorWidth");

private:
  SharedLoadIterator shared_load_iterator_;

public:
  CUTLASS_DEVICE
  EpilogueWithVisitorVisitBatch(SharedStorage &shared_storage, int thread_idx,
                                int warp_idx, int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {}

  CUTLASS_DEVICE
  void operator()(Visitor &visitor, AccumulatorTile const &accumulators) {
    visitor.begin_epilogue();

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

#pragma unroll(IterationsUnroll ? Visitor::kIterations : 1)
    for (int iter_idx = 0; iter_idx < Visitor::kIterations; ++iter_idx) {
      visitor.begin_step(iter_idx);

      __syncthreads();

      acc2smem_source_needed<
          cutlass::make_index_sequence<Visitor::kIterations>>::
          push(iter_idx, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      if (kPartitionsK > 1) {
        plus<typename SharedLoadIterator::Fragment> add_fragments;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] =
              add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }
        shared_load_iterator_.add_pointer_offset(
            (1 - kPartitionsK) * kSmemPointerOffset);
      }

      VisitAccumulatorAccessType const *visit_accum_ptr =
          reinterpret_cast<VisitAccumulatorAccessType const *>(
              &aligned_accum_fragment[0]);

      int const kVisitBatchCount = AccumulatorTile::kElements /
                                   (Visitor::kIterations * kVisitVectorWidth);

      CUTLASS_PRAGMA_UNROLL
      for (int batch_idx = 0; batch_idx < kVisitBatchCount; ++batch_idx) {
        int const scalar_idx = batch_idx * kVisitVectorWidth;
        int row_idx =
            scalar_idx / SharedLoadIterator::ThreadMap::Iterations::kColumn;
        int col_idx =
            scalar_idx % SharedLoadIterator::ThreadMap::Iterations::kColumn;

        if (!col_idx) {
          visitor.begin_row(row_idx);
        }

        visitor.visit(iter_idx, row_idx, col_idx, batch_idx,
                      visit_accum_ptr[batch_idx]);

        if (col_idx + kVisitVectorWidth ==
            SharedLoadIterator::ThreadMap::Iterations::kColumn) {
          visitor.end_row(row_idx);
        }
      }

      visitor.end_step(iter_idx);
    }

    visitor.end_epilogue();
  }

private:
  template <class Seq>
  struct acc2smem_source_needed;

  template <size_t... Seq>
  struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE
    static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                       WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }
      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {
          (pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
      (void)dummy;
    }
  };
};

template <typename Visitor_, typename Existing_, int VisitVectorWidth_,
          bool IterationsUnroll = true>
struct EpilogueWithVisitorFromExistingEpilogueSelect {
  using Epilogue = typename platform::conditional<
      (VisitVectorWidth_ > 1),
      EpilogueWithVisitorVisitBatch<
          Visitor_, typename Existing_::Shape,
          typename Existing_::WarpMmaOperator, Existing_::kPartitionsK,
          typename Existing_::AccumulatorFragmentIterator,
          typename Existing_::WarpTileIterator,
          typename Existing_::SharedLoadIterator, typename Existing_::Padding,
          Existing_::kFragmentsPerIteration, IterationsUnroll>,
      EpilogueWithVisitor<
          Visitor_, typename Existing_::Shape,
          typename Existing_::WarpMmaOperator, Existing_::kPartitionsK,
          typename Existing_::AccumulatorFragmentIterator,
          typename Existing_::WarpTileIterator,
          typename Existing_::SharedLoadIterator, typename Existing_::Padding,
          Existing_::kFragmentsPerIteration, IterationsUnroll>>::type;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
