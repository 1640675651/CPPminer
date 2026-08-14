#include "case32_layout.hpp"

#include "cp_config.h"

#include <cstdio>

namespace case32 {

int kMR = PP_HASH_H;
int kNR = PP_HASH_W;
int kKR = R_RANK;
int kMicroPerMacroM = kMacroM / PP_HASH_H;
int kMicroPerMacroN = kMacroN / PP_HASH_W;
int kPanelA = kKR * PP_HASH_H;
int kPanelB = kKR * PP_HASH_W;
int kKGroups = kKR / kRank;
int kMacroWorkItems = kMicroPerMacroM * kMicroPerMacroN;
int kNumMilestones = K_DIM / R_RANK;
int kKgBytesA = PP_HASH_H * kRank;
int kKgSliceB = (PP_HASH_W / kColsPerGroup) * 32;
int kMacroKgStripA = kMicroPerMacroM * kKgBytesA;
int kMacroKgStripB = kMicroPerMacroN * kKgSliceB;
int kMacroKbBlockA = kKGroups * kMacroKgStripA;
int kMacroKbBlockB = kKGroups * kMacroKgStripB;

namespace {

void update_derived() {
    kKR = R_RANK;
    kMicroPerMacroM = kMacroM / kMR;
    kMicroPerMacroN = kMacroN / kNR;
    kPanelA = kKR * kMR;
    kPanelB = kKR * kNR;
    kKGroups = kKR / kRank;
    kMacroWorkItems = kMicroPerMacroM * kMicroPerMacroN;
    kNumMilestones = K_DIM / kKR;
    kKgBytesA = kMR * kRank;
    kKgSliceB = (kNR / kColsPerGroup) * 32;
    kMacroKgStripA = kMicroPerMacroM * kKgBytesA;
    kMacroKgStripB = kMicroPerMacroN * kKgSliceB;
    kMacroKbBlockA = kKGroups * kMacroKgStripA;
    kMacroKbBlockB = kKGroups * kMacroKgStripB;
}

bool is_supported_tile(int mr, int nr) {
    if (mr == 4 && nr == 8) {
        return true;
    }
    if (mr == PP_HASH_H && (nr == 8 || nr == 16)) {
        return true;
    }
    return false;
}

} // namespace

bool configure(int mr, int nr) {
    if (!is_supported_tile(mr, nr)) {
        std::fprintf(stderr, "[ocl] hash tile must be 4x8, 8x8, or 8x16 (got %dx%d)\n", mr, nr);
        return false;
    }
    if (kMacroM % mr != 0 || kMacroN % nr != 0) {
        std::fprintf(stderr,
                     "[ocl] tile %dx%d must divide macro block %dx%d\n", mr, nr, kMacroM,
                     kMacroN);
        return false;
    }
    if (nr % kColsPerGroup != 0) {
        std::fprintf(stderr, "[ocl] tile NR=%d must be a multiple of %d\n", nr, kColsPerGroup);
        return false;
    }
    if (kKR % kRank != 0 || K_DIM % R_RANK != 0) {
        std::fprintf(stderr, "[ocl] KR/rank layout mismatch\n");
        return false;
    }
    const int work_items = (kMacroM / mr) * (kMacroN / nr);
    if (work_items > kMacroWorkItemsMax) {
        std::fprintf(stderr,
                     "[ocl] tile %dx%d needs %d work-items per macro block (max %d)\n", mr, nr,
                     work_items, kMacroWorkItemsMax);
        return false;
    }
    kMR = mr;
    kNR = nr;
    update_derived();
    return true;
}

int hash_tiles_per_macro() { return (kMacroM / kMR) * (kMacroN / kNR); }

} // namespace case32
