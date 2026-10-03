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

static_assert(Gemm128x128TensorOp::GemmKernel::kHashTilesPerCta == 256,
              "tensor-op Case 10 must emit the 256 SIMT hash tiles per CTA");
static_assert(Gemm128x128TensorOp::GemmKernel::kInlineXor &&
                  Gemm128x128TensorOp::GemmKernel::kCase10Continuous,
              "tensor-op fused path must use Case 10 continuous pipeline");
static_assert(Gemm128x128TensorOp::kItersPerMilestone * 64 == R_RANK,
              "tensor-op milestone must span R_RANK (2 K-tiles of 64)");
static_assert(Gemm128x128TensorOp::GemmKernel::ThreadblockShape::kM ==
                      CP_CUTLASS_CTA_M &&
                  Gemm128x128TensorOp::GemmKernel::ThreadblockShape::kN ==
                      CP_CUTLASS_CTA_N,
              "tensor-op CTA must match the proof period size");

/* Larger threadblocks = two virtual 128x128 CTAs of 256 hash tiles each. */
static_assert(Gemm256x128TensorOp80::GemmKernel::kVirtM == 2 &&
                  Gemm256x128TensorOp80::GemmKernel::kVirtN == 1 &&
                  Gemm128x256TensorOp80::GemmKernel::kVirtM == 1 &&
                  Gemm128x256TensorOp80::GemmKernel::kVirtN == 2 &&
                  Gemm256x128TensorOp::GemmKernel::kVirtM == 2 &&
                  Gemm128x256TensorOp::GemmKernel::kVirtN == 2,
              "256-wide threadblocks must map onto 2 virtual 128x128 CTAs");
static_assert(Gemm256x128TensorOp80::GemmKernel::kHashTilesPerCta == 256 &&
                  Gemm128x256TensorOp::GemmKernel::kHashTilesPerCta == 256,
              "virtual CTAs keep the 256 SIMT hash tiles");
static_assert(Gemm256x128TensorOp80::kItersPerMilestone * 64 == R_RANK &&
                  Gemm256x128TensorOp::kItersPerMilestone * 64 == R_RANK,
              "milestone must span R_RANK");

static_assert(Gemm128x128StepMajor::GemmKernel::kThreadCount == 256,
              "CUTLASS Case 9 step-major requires 256 threads per CTA");
static_assert(Gemm128x128StepMajor::GemmKernel::kMilestoneMajorStorage,
              "step-major fused path uses milestone-major storage");

static_assert(MmaLaneTile128x128::kHashH == CP_CUTLASS_HASH_H &&
                  MmaLaneTile128x128::kHashW == CP_CUTLASS_HASH_W,
              "hash tile must be 8x8 contiguous MMA lane block");

} // namespace cp_cutlass
