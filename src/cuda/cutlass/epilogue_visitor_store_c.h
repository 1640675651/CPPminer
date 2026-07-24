// Minimal C-output-only visitor. No XOR logic — just writes accumulator
// fragments to the output matrix D via the standard epilogue pipeline.
// Used by Case 8 for the final C output after all milestones.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <typename ThreadblockShape_, int ThreadCount_,
          typename OutputTileIterator_, typename ElementAccumulator_,
          typename ElementwiseFunctor_>
class EpilogueVisitorStoreC {
public:
  using ThreadblockShape = ThreadblockShape_;
  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kThreadCount = ThreadCount_;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = typename ElementwiseFunctor::ElementCompute;
  using ElementNorm = float;
  using ElementSum = uint32_t;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using OutputVector = cutlass::Array<ElementOutput, kElementsPerAccess>;

  struct Arguments {
    typename ElementwiseFunctor::Params elementwise;
    int tile_cols;
    int milestone_stride;
    bool store_output;
    bool serial_split_k;
    CUTLASS_HOST_DEVICE Arguments()
        : tile_cols(0), milestone_stride(0), store_output(false),
          serial_split_k(false) {}
    CUTLASS_HOST_DEVICE Arguments(typename ElementwiseFunctor::Params e, int tc,
                                   int ms, bool so, bool ssk)
        : elementwise(e), tile_cols(tc), milestone_stride(ms),
          store_output(so), serial_split_k(ssk) {}
  };

  struct Params {
    typename ElementwiseFunctor::Params elementwise;
    int tile_cols;
    int milestone_stride;
    bool store_output;
    bool serial_split_k;
    CUTLASS_HOST_DEVICE Params()
        : tile_cols(0), milestone_stride(0), store_output(false),
          serial_split_k(false) {}
    CUTLASS_HOST_DEVICE Params(Arguments const &a)
        : elementwise(a.elementwise), tile_cols(a.tile_cols),
          milestone_stride(a.milestone_stride), store_output(a.store_output),
          serial_split_k(a.serial_split_k) {}
  };

  struct SharedStorage {};

private:
  Params const &params_;
  SharedStorage &shared_storage_;
  ElementwiseFunctor elementwise_;
  OutputTileIterator iterator_D_;
  typename OutputTileIterator::Fragment fragment_D_;
  bool store_output_;
  int k_partition_;

public:
  CUTLASS_DEVICE
  EpilogueVisitorStoreC(
      Params const &params, SharedStorage &shared_storage,
      MatrixCoord const &problem_size, int thread_idx, int warp_idx,
      int lane_idx, typename OutputTileIterator::Params params_C,
      typename OutputTileIterator::Params params_D,
      typename OutputTileIterator::Element *ptr_C,
      typename OutputTileIterator::Element *ptr_D, float *,
      uint32_t *, MatrixCoord const &threadblock_offset = MatrixCoord(0, 0),
      int = 0, MatrixCoord const & = MatrixCoord(0, 0))
      : params_(params), shared_storage_(shared_storage),
        elementwise_(params.elementwise),
        iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
        store_output_(params.store_output), k_partition_(0) {
    (void)warp_idx; (void)lane_idx; (void)shared_storage_;
    (void)params_C; (void)ptr_C;
  }

  CUTLASS_DEVICE void set_k_partition(int sk, int sks) {
    k_partition_ = sk;
    elementwise_.set_k_partition(sk, sks);
  }
  CUTLASS_DEVICE void set_batch_index(int) {}
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) { fragment_D_.clear(); }
  CUTLASS_DEVICE void begin_row(int) {}
  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void end_step(int) {
    if (store_output_) {
      iterator_D_.store(fragment_D_);
      ++iterator_D_;
    }
  }

  CUTLASS_DEVICE
  void visit(int, int, int, int frag_idx, AccumulatorFragment const &accum) {
    NumericArrayConverter<ElementOutput, ElementAccumulator, kElementsPerAccess> conv;
    auto &out = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    out = conv(accum);
  }

  CUTLASS_DEVICE void end_epilogue() {}
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass
