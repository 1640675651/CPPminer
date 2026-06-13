// Case 7 / 7.1 / 7.2: one scattered epilogue tile per thread (all cells that
// thread visits in the epilogue). Register partial XOR only; each thread stores
// its own tile (one global store per milestone slice). No warp shuffle, no smem.

#pragma once

#include <type_traits>

#include "cp_cutlass_jackpot.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/numeric_conversion.h"
namespace cutlass {
namespace epilogue {
namespace threadblock {

template <typename ThreadblockShape_, int ThreadCount,
          typename OutputTileIterator_, typename ElementAccumulator_,
          typename ElementwiseFunctor_>
class EpilogueVisitorTileXorRegPerThread {
public:
  using ThreadblockShape = ThreadblockShape_;
  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kThreadCount = ThreadCount;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kFragsPerStep =
      OutputTileIterator::Fragment::kElements / kElementsPerAccess;

  static_assert(kThreadCount == OutputTileIterator::kThreads,
                "thread count must match epilogue OutputTileIterator");
  static_assert(ThreadblockShape::kM * ThreadblockShape::kN ==
                    kThreadCount * kIterations * kFragsPerStep,
                "one scattered epilogue tile per thread");

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = typename ElementwiseFunctor::ElementCompute;
  using ElementNorm = float;
  using ElementSum = uint32_t;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using ComputeFragment = Array<ElementCompute, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;

  struct Arguments {
    typename ElementwiseFunctor::Params elementwise;
    int tile_cols;
    int milestone_stride;
    bool store_output;
    bool serial_split_k;
    bool fuse_jackpot;

    CUTLASS_HOST_DEVICE
    Arguments()
        : tile_cols(0), milestone_stride(0), store_output(false),
          serial_split_k(false), fuse_jackpot(false) {}

    CUTLASS_HOST_DEVICE
    Arguments(typename ElementwiseFunctor::Params elementwise_, int tile_cols_,
              int milestone_stride_ = 0, bool store_output_ = false,
              bool serial_split_k_ = false, bool fuse_jackpot_ = false)
        : elementwise(elementwise_), tile_cols(tile_cols_),
          milestone_stride(milestone_stride_), store_output(store_output_),
          serial_split_k(serial_split_k_), fuse_jackpot(fuse_jackpot_) {}
  };

  struct Params {
    typename ElementwiseFunctor::Params elementwise;
    int tile_cols;
    int milestone_stride;
    bool store_output;
    bool serial_split_k;
    bool fuse_jackpot;
    uint32_t *jackpot_words;
    int milestone_index;

    CUTLASS_HOST_DEVICE
    Params()
        : tile_cols(0), milestone_stride(0), store_output(false),
          serial_split_k(false), fuse_jackpot(false), jackpot_words(nullptr),
          milestone_index(0) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args)
        : elementwise(args.elementwise), tile_cols(args.tile_cols),
          milestone_stride(args.milestone_stride), store_output(args.store_output),
          serial_split_k(args.serial_split_k), fuse_jackpot(args.fuse_jackpot),
          jackpot_words(nullptr), milestone_index(0) {}
  };

  struct SharedStorage {};

private:
  Params const &params_;
  SharedStorage &shared_storage_;
  MatrixCoord threadblock_offset_;
  ElementwiseFunctor elementwise_;

  OutputTileIterator iterator_C_;
  OutputTileIterator iterator_D_;

  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  bool store_output_;
  bool serial_split_k_;
  int k_partition_;
  uint32_t *ptr_tile_xor_;
  int tile_cols_;
  int thread_idx_;
  uint32_t partial_xor_;

public:
  CUTLASS_DEVICE
  EpilogueVisitorTileXorRegPerThread(
      Params const &params, SharedStorage &shared_storage,
      MatrixCoord const &problem_size, int thread_idx, int warp_idx,
      int lane_idx, typename OutputTileIterator::Params params_C,
      typename OutputTileIterator::Params params_D,
      typename OutputTileIterator::Element *ptr_C,
      typename OutputTileIterator::Element *ptr_D, float * /*ptr_unused*/,
      uint32_t *ptr_tile_xor,
      MatrixCoord const &threadblock_offset = MatrixCoord(0, 0),
      int /*column_offset*/ = 0,
      MatrixCoord const & /*problem_size_real*/ = MatrixCoord(0, 0))
      : params_(params), shared_storage_(shared_storage),
        threadblock_offset_(threadblock_offset), elementwise_(params.elementwise),
        iterator_C_(params_C, ptr_C, problem_size, thread_idx, threadblock_offset),
        iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
        store_output_(params.store_output), serial_split_k_(params.serial_split_k),
        ptr_tile_xor_(ptr_tile_xor), tile_cols_(params.tile_cols),
        k_partition_(0), thread_idx_(thread_idx), partial_xor_(0) {
    (void)warp_idx;
    (void)lane_idx;
    (void)shared_storage_;
  }

  CUTLASS_DEVICE
  void set_k_partition(int split_k_index, int split_k_slices) {
    k_partition_ = split_k_index;
    elementwise_.set_k_partition(split_k_index, split_k_slices);
  }

  CUTLASS_DEVICE
  void set_batch_index(int /*batch_idx*/) {}

  CUTLASS_DEVICE
  void begin_epilogue() { partial_xor_ = 0u; }

  CUTLASS_DEVICE
  void begin_step(int /*step_idx*/) {
    fragment_D_.clear();
    fragment_C_.clear();

    if (k_partition_ > 0 ||
        elementwise_.kScale !=
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      iterator_C_.load(fragment_C_);
      ++iterator_C_;
    }
  }

  CUTLASS_DEVICE
  void begin_row(int /*row_idx*/) {}

  CUTLASS_DEVICE
  void visit(int /*iter_idx*/, int /*row_idx*/, int /*column_idx*/, int frag_idx,
             AccumulatorFragment const &accum) {
    NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>
        acc_converter;

    ComputeFragment result;
    OutputVector &source_vector =
        reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    if (k_partition_ == 0 &&
        elementwise_.kScale ==
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      result = acc_converter(elementwise_(accum));
    } else {
      result = acc_converter(elementwise_(accum, source_vector));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      // Match zk-pow compute_jackpot: XOR int32 MAC accumulators (not clamped output).
      partial_xor_ ^= static_cast<uint32_t>(accum[i]);
    }

    if (store_output_ || serial_split_k_) {
      NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
          out_converter;
      OutputVector &output =
          reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
      output = out_converter(result);
    }
  }

  CUTLASS_DEVICE
  void end_row(int /*row_idx*/) {}

  CUTLASS_DEVICE
  void end_step(int /*step_idx*/) {
    if (store_output_ || serial_split_k_) {
      iterator_D_.store(fragment_D_);
    }
    ++iterator_D_;
  }

  CUTLASS_DEVICE
  void end_epilogue() {
    if (params_.fuse_jackpot && params_.jackpot_words != nullptr) {
      cp_cutlass_jackpot_fold_step(params_.jackpot_words, params_.milestone_index,
                                   partial_xor_);
      return;
    }

    if (ptr_tile_xor_ == nullptr) {
      return;
    }

    int const cta_tr = threadblock_offset_.row() / ThreadblockShape::kM;
    int const cta_tc = threadblock_offset_.column() / ThreadblockShape::kN;
    size_t const tile_id =
        (static_cast<size_t>(cta_tr) * static_cast<size_t>(tile_cols_) +
         static_cast<size_t>(cta_tc)) *
            static_cast<size_t>(kThreadCount) +
        static_cast<size_t>(thread_idx_);

    ptr_tile_xor_[tile_id] = partial_xor_;
  }
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
