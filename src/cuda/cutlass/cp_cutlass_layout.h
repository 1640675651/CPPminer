#pragma once

#include "cp_config.h"
#include "cp_cutlass_gemm_types.h"
#include "mma_lane_tile.h"

namespace cp_cutlass {

static_assert(Gemm128x128StepMajor::GemmKernel::kThreadCount == 256,
              "CUTLASS Case 9 requires 256 threads per CTA");
static_assert(Gemm128x128StepMajor::GemmKernel::kInlineXor,
              "CUTLASS Case 9 requires in-register XOR");
static_assert(Gemm128x128StepMajor::GemmKernel::kReuseMmaAcrossMilestones,
              "CUTLASS Case 9 requires Mma reuse + wind_down");
static_assert(MmaLaneTile128x128::kHashH == CP_CUTLASS_HASH_H &&
                  MmaLaneTile128x128::kHashW == CP_CUTLASS_HASH_W,
              "hash tile must be 8x8 contiguous MMA lane block");

}  // namespace cp_cutlass
