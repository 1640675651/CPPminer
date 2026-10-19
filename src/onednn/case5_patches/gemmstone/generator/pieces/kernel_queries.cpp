/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "gemmstone/driver_info.hpp"
#include "hw_utils.hpp"

#include <cstdlib>
#include <cstring>

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;

static int case5XorOutputMilestoneCount(const GEMMProblem &problem) {
    if (!problem.case5TileXorWrap) {
        return std::max(1, problem.case5XorMaxMilestones);
    }
    if (problem.case5XorOutputMilestones > 0) {
        return problem.case5XorOutputMilestones;
    }
    return std::max(1, problem.case5XorMaxMilestones / 2);
}

bool case5WrapAtFlush(const GEMMProblem &problem)
{
    static const bool atFlush = [] {
        const char *v = std::getenv("CASE5_XOR_WRAP_AT");
        return !(v && v[0] && !std::strcmp(v, "loop"));
    }();
    return atFlush && problem.case5TileXorWrap && problem.case5TileXorSlm && !problem.case5TileXorWrapGrf;
}

int case5XorSlmSlotMs(const GEMMProblem &problem)
{
    return case5WrapAtFlush(problem) ? std::max(1, problem.case5XorMaxMilestones)
                                     : case5XorOutputMilestoneCount(problem);
}

static size_t case5TileXorSlmBytesPerWG(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (!problem.case5TileXor || !problem.case5TileXorSlm)
        return 0;
    const int nFolds = std::max(1, problem.case5XorSubGridM) * std::max(1, problem.case5XorSubGridN);
    const int maxMs = case5XorSlmSlotMs(problem);
    const int slotsPerThread = maxMs * nFolds;
    const int wgThreads = strategy.wg[LoopM] * strategy.wg[LoopN];
    return static_cast<size_t>(wgThreads) * static_cast<size_t>(slotsPerThread) * sizeof(uint32_t);
}

size_t case5TileXorSlmByteOffset(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy,
                                bool computeMax)
{
    GEMMProblem base = problem;
    base.case5TileXorSlm = false;
    return gemmSLMSize(hw, base, strategy, computeMax);
}

size_t gemmSLMSize(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool computeMax)
{
    size_t slmSize = 0;

    // Space needed by SLM copies.
    slmSize = strategy.slmABufSize(problem) + strategy.slmBBufSize(problem);
    if (strategy.kParallelLocal && !computeMax)
        slmSize /= strategy.wg[LoopK];

    // Space needed for row/column sum reduction/sharing.
    if ((problem.needsASums() && strategy.slmA) || (problem.needsBSums() && strategy.slmB)) {
        slmSize = std::max<size_t>(slmSize, (strategy.unroll[LoopM] * strategy.wg[LoopM]
                                           + strategy.unroll[LoopN] * strategy.wg[LoopN]) * problem.Tc);
    }

    // Beta/post-op fusing needs SLM to transmit a single flag.
    if ((strategy.fuseBeta && !strategy.altFusedBeta) || strategy.fusePostOps)
        slmSize = std::max<size_t>(slmSize, 8);

    slmSize += case5TileXorSlmBytesPerWG(problem, strategy);
    // Cooperative jackpot tables live in the k loop's (then dead) SLM buffers: above 16 KB the
    // A770 loses a quarter of its throughput, so they only ever grow the allocation when needed.
    if (problem.case5JackpotCoop)
        slmSize = std::max(slmSize, case5JackpotCoopSlmBytes(problem, strategy));
    // CASE5_SLM_PAD=<bytes> (timing experiments only): pad or shrink the SLM allocation.
    if (const char *pad = std::getenv("CASE5_SLM_PAD"); pad && pad[0]) {
        const long long p = std::atoll(pad);
        slmSize = size_t(std::max<long long>(0, static_cast<long long>(slmSize) + p));
    }

    return slmSize;
}

int case5JackpotCoopTiles(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    return strategy.wg[LoopM] * strategy.wg[LoopN] * subGridM * subGridN;
}

size_t case5JackpotCoopSlmBytes(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (!problem.case5JackpotCoop) return 0;
    const int tiles = std::max(0, case5JackpotCoopTiles(problem, strategy));
    return size_t(tiles) * (64 + 4 + 8);          // per tile: message, spatial id, packed (t_rows, t_cols)
}

size_t case5JackpotCoopSlmByteOffset(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool computeMax)
{
    GEMMProblem base = problem;
    base.case5JackpotCoop = false;
    return (gemmSLMSize(hw, base, strategy, computeMax) + 63) & ~size_t(63);
}

size_t gemmPerKSLMSize(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    size_t slmSize = 0;

    // Space needed for local k reduction (as much as possible).
    if (strategy.kParallelLocal) {
        // Calculate max SLM usage that doesn't reduce thread count.
        int mnThreads = strategy.wg[LoopM] * strategy.wg[LoopN];
        if (mnThreads <= 0) stub();
        int concurrentK = std::max(1, threadsPerEU(hw, strategy) * eusPerSubslice(hw) / mnThreads);
        slmSize = rounddown_pow2(maxSLMPerWG(problem.product, strategy.GRFs) / concurrentK);
        if (!problem.sumA && !problem.sumB) {
            auto singleTile = strategy.wg[LoopM] * strategy.wg[LoopN]
                            * align_up(strategy.unroll[LoopM] * strategy.unroll[LoopN] * problem.Tc, GRF::bytes(hw));
            slmSize = std::min<size_t>(slmSize, singleTile);
        }

        // Calculate space needed by SLM copies.
        size_t totalSLM = strategy.slmABufSize(problem) + strategy.slmBBufSize(problem);
        slmSize = std::max<size_t>(slmSize, totalSLM / strategy.wg[LoopK]);
    }

    return slmSize;
}

void getCRemainders(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool &remM_C, bool &remN_C)
{
    bool remainderM = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remainderN = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);

    int C_mgran, C_ngran;
    getGranularities(problem.C, C_mgran, C_ngran);

    bool noStdCRem = strategy.C.padded
                  || strategy.altCRemainder;

    remM_C = remainderM && !noStdCRem && (C_mgran < strategy.unroll[LoopM]);
    remN_C = remainderN && !noStdCRem && (C_ngran < strategy.unroll[LoopN]);
}

bool keepIJ0(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (problem.hasBinaryPostOp()) return true;
    if (problem.aoPtrDims > 0 || problem.boPtrDims > 0) return true;
    if (problem.aScale2D() || problem.bScale2D()) return true;
    if (problem.earlyDequantizeA() ||  problem.earlyDequantizeB()) return true;
    return false;
}

bool keepH0(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (problem.quantized2DA() || problem.quantized2DB()) return true;
    return strategy.kParallelVariable && strategy.fuseBeta;
}

GEMMSTONE_NAMESPACE_END
