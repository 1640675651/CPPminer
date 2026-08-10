#pragma once

#include "cp_config.h"
#include "cp_cutlass_gemm_types.h"
#include "mma_lane_tile.h"

namespace cp_cutlass {

static_assert(Gemm128x128RowMajor::GemmKernel::kThreadCount == 256,
              "CUTLASS Case 10 requires 256 threads per CTA");
static_assert(Gemm128x128RowMajor::GemmKernel::kInlineXor,
              "CUTLASS Case 10 requires in-register XOR");
static_assert(Gemm128x128RowMajor::GemmKernel::kCase10Continuous,
              "row-major fused path must use Case 10 continuous pipeline");
static_assert(Gemm128x128RowMajor::GemmKernel::kMilestoneMajorStorage == false,
              "Case 10 requires contiguous K (row-major Ap/BpT)");
static_assert(kItersPerMs == R_RANK / kCtaK,
              "milestone K-tile count must match R_RANK");

static_assert(Gemm128x128StepMajor::GemmKernel::kThreadCount == 256,
              "CUTLASS Case 9 step-major requires 256 threads per CTA");
static_assert(Gemm128x128StepMajor::GemmKernel::kMilestoneMajorStorage,
              "step-major fused path uses milestone-major storage");

static_assert(MmaLaneTile128x128::kHashH == CP_CUTLASS_HASH_H &&
                  MmaLaneTile128x128::kHashW == CP_CUTLASS_HASH_W,
              "hash tile must be 8x8 contiguous MMA lane block");

} // namespace cp_cutlass
