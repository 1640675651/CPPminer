#pragma once

// Case 3.2 blocking constants (shared by host prepack and OpenCL kernels).
// Override at CMake configure time: -DOPENCL_CASE32_MR=4 -DOPENCL_CASE32_NR=8

#ifndef CASE32_MR
#define CASE32_MR 8
#endif
#ifndef CASE32_NR
#define CASE32_NR 16
#endif
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

constexpr int kMR = CASE32_MR;
constexpr int kNR = CASE32_NR;
constexpr int kKR = 256;
constexpr int kMacroM = 128;
constexpr int kMacroN = 128;
constexpr int kMicroPerMacroM = kMacroM / kMR;
constexpr int kMicroPerMacroN = kMacroN / kNR;
constexpr int kPanelA = kKR * kMR;
constexpr int kPanelB = kKR * kNR;
constexpr int kColsPerGroup = 8;
constexpr int kRank = 4;
constexpr int kKGroups = kKR / kRank;
constexpr int kMacroWorkItems = kMicroPerMacroM * kMicroPerMacroN;
constexpr int kNumMilestones = 16;
constexpr int kKgBytesA = kMR * kRank;
constexpr int kKgSliceB = (kNR / kColsPerGroup) * 32;
constexpr int kMacroKgStripA = kMicroPerMacroM * kKgBytesA;
constexpr int kMacroKgStripB = kMicroPerMacroN * kKgSliceB;
constexpr int kMacroKbBlockA = kKGroups * kMacroKgStripA;
constexpr int kMacroKbBlockB = kKGroups * kMacroKgStripB;

static_assert(kMR > 0 && kNR > 0, "MR and NR must be positive");
static_assert(kMacroM % kMR == 0 && kMacroN % kNR == 0, "macro must divide micro tile");
static_assert(kNR % kColsPerGroup == 0, "NR must be a multiple of COLS_PER_GROUP (8)");
static_assert(kKR % kRank == 0, "KR must divide rank");
static_assert(kKgBytesA % kMicroPerMacroN == 0,
              "A panel bytes must divide evenly across column micro-tiles for LDS");
static_assert(kPanelB % kMicroPerMacroM == 0,
              "B panel bytes must divide evenly across row micro-tiles for LDS");
static_assert(kMacroWorkItems <= 256, "macro work-items must fit typical AMD max work-group");

} // namespace case32
