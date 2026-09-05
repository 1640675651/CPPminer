#pragma once

// Case 3.2 blocking constants (shared by host prepack and OpenCL kernels).
// Register tile is 4x4, 4x8, 8x8, or 8x16 (selected via --ocl-tile / AMD auto-detect).
// Pearl hash tiles contain at least 32 cells, so one WI walks two 4x4 halves of a 4x8 hash.
// Fused GEMM private memory: see docs/memory.md (Beignet 8×8 ≈384 B/WI, 8×16 ≈1152 B/WI).
//
// Macro block: 8x* tiles use 128×128; 4x* tiles use 64×64 (Case 3.4 4x8/64x64 → 128 WI/WG).

#include "cp_config.h"

#ifndef CASE32_USE_LDS
#define CASE32_USE_LDS 0
#endif
#ifndef CASE32_COALESCE
#define CASE32_COALESCE 1
#endif
#ifndef CASE32_WI_ROWMAJOR
#define CASE32_WI_ROWMAJOR 0
#endif

namespace case32 {

constexpr int kRank = 4;
/* Prefer ≤256 WIs/WG (Case 3.4 / many Intel & AMD devices). */
constexpr int kMacroWorkItemsMax = 256;
constexpr int kMacroLarge = 128; /* 8x8 / 8x16 */
constexpr int kMacroSmall = 64;  /* 4x4 / 4x8 — Case 3.4 alignment */

/* Runtime tile + macro shape (set via configure() before OpenCL init). */
extern int kMR;
extern int kNR;
extern int kKR;
extern int kMacroM;
extern int kMacroN;
extern int kMicroPerMacroM;
extern int kMicroPerMacroN;
extern int kHashPerMacroM;
extern int kHashPerMacroN;
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

/* Configure register-tile MR x NR (4x4, 4x8, 8x8, or 8x16). Sets macro size. */
bool configure(int mr, int nr);

int hash_tile_mr();
int hash_tile_nr();

/* All supported tiles use column-major work-item traversal. */
bool wi_row_major();

/* Hash tiles covered by one macro block. */
int hash_tiles_per_macro();

} // namespace case32
