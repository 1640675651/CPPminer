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


#ifndef GEMMSTONE_GENERATOR_PIECES_KERNEL_QUERIES_HPP
#define GEMMSTONE_GENERATOR_PIECES_KERNEL_QUERIES_HPP

#include "internal/ngen_includes.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "state.hpp"

GEMMSTONE_NAMESPACE_START


// Return amount of SLM needed by a GEMM kernel.
size_t gemmSLMSize(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool computeMax = false);

// Byte offset in SLM where Case5 tile_xor staging begins (after A/B SLM).
size_t case5TileXorSlmByteOffset(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy,
                                 bool computeMax = false);

// Cooperative fused jackpot (case5JackpotCoop): tiles per work group, bytes of the SLM region that
// holds every tile's 16-word message, spatial id and packed (t_rows, t_cols), and its byte offset
// (after everything else gemmstone and the Case5 staging use, 64-byte aligned).
int case5JackpotCoopTiles(const GEMMProblem &problem, const GEMMStrategy &strategy);
size_t case5JackpotCoopSlmBytes(const GEMMProblem &problem, const GEMMStrategy &strategy);
size_t case5JackpotCoopSlmByteOffset(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy,
                                     bool computeMax = false);

// Case5.4 with SLM staging: stage every K milestone in SLM and fold slot ms with ms + outputMs
// at the epilogue flush (default), instead of a load-xor-store in the K loop
// (CASE5_XOR_WRAP_AT=loop restores that).
bool case5WrapAtFlush(const GEMMProblem &problem);

// Milestone slots per fold staged in SLM: all K milestones when wrapping at the flush,
// otherwise the output milestone count.
int case5XorSlmSlotMs(const GEMMProblem &problem);

// Return amount of per-k SLM needed by a GEMM kernel.
size_t gemmPerKSLMSize(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy);

// Decide whether C layout needs m/n remainder handling.
void getCRemainders(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool &remM_C, bool &remN_C);

// Check if i0/j0 need to be saved across the k loop.
bool keepIJ0(const GEMMProblem &problem, const GEMMStrategy &strategy);

// Check if h0 needs to be saved across the k loop.
bool keepH0(const GEMMProblem &problem, const GEMMStrategy &strategy);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
