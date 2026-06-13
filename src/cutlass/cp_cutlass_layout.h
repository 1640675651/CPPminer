#pragma once

#include "cp_cutlass_gemm_types.h"
#include "scattered_thread_tile.h"

namespace cp_cutlass {

using Case71OutIter = typename Gemm128x256StepMajor::GemmKernel::EpilogueVisitor::
    OutputTileIterator;

static_assert(Gemm128x256StepMajor::GemmKernel::kThreadCount == 256,
              "CUTLASS fused path requires 256 threads per CTA");

using CutlassScatteredTile128x256 = ScatteredThreadTile<
    128, 256, Case71OutIter::kIterations,
    Case71OutIter::Fragment::kElements / Case71OutIter::kElementsPerAccess,
    (Case71OutIter::Fragment::kElements / Case71OutIter::kElementsPerAccess) / 2>;

static_assert(128 * 256 == CutlassScatteredTile128x256::kThreadsPerCta *
                                CutlassScatteredTile128x256::kCellsPerThread,
              "one scattered epilogue tile per thread");

/* Proof row/col offsets relative to tile anchor (step 0, frag 0). Keep in sync with
 * rust/cp-proof-ffi CUTLASS_ROWS / CUTLASS_COLS and plain_proof_host.py. */
static_assert(CutlassScatteredTile128x256::kEpilogueSteps == 8,
              "CUTLASS proof patterns assume 8 epilogue steps");
static_assert(CutlassScatteredTile128x256::kFragsPerStep == 16,
              "CUTLASS proof patterns assume 16 frags/step (128 cells/thread)");

}  // namespace cp_cutlass
