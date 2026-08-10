#pragma once

// Case 3.2 blocking constants (shared by host prepack and OpenCL kernels).
// Hash/register tile is 8x8 or 8x16 (selected via --ocl-tile / AMD auto-detect).

#include "cp_config.h"

#ifndef CASE32_USE_LDS
#define CASE32_USE_LDS 0
#endif
#ifndef CASE32_COALESCE
#define CASE32_COALESCE 1
#endif
#ifndef CASE32_WI_ROWMAJOR
#define CASE32_WI_ROWMAJOR 1
#endif

namespace case32 {

constexpr int kMacroM = 128;
constexpr int kMacroN = 128;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;

/* Runtime tile shape and derived layout (set via configure() before OpenCL init). */
extern int kMR;
extern int kNR;
extern int kKR;
extern int kMicroPerMacroM;
extern int kMicroPerMacroN;
extern int kPanelA;
extern int kPanelB;
extern int kKGroups;
extern int kMacroWorkItems;
extern int kNumMilestones;
extern int kKgBytesA;
extern int kKgSliceB;
extern int kMacroKgStripA;
extern int kMacroKgStripB;
extern int kMacroKbBlockA;
extern int kMacroKbBlockB;

/* Configure hash-tile MR x NR (8x8 or 8x16). */
bool configure(int mr, int nr);

/* Hash tiles covered by one 128x128 macro block. */
int hash_tiles_per_macro();

} // namespace case32
