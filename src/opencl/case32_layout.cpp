#include "case32_layout.hpp"

#include "cp_config.h"

#include <cstdio>

namespace case32 {

int kMR = PP_HASH_H;
int kNR = PP_HASH_W;
int kKR = R_RANK;
int kMacroM = kMacroLarge;
int kMacroN = kMacroLarge;
int kMicroPerMacroM = kMacroLarge / PP_HASH_H;
int kMicroPerMacroN = kMacroLarge / PP_HASH_W;
int kHashPerMacroM = kMicroPerMacroM;
int kHashPerMacroN = kMicroPerMacroN;
int kPanelA = kKR * PP_HASH_H;
int kPanelB = kKR * PP_HASH_W;
int kKGroups = kKR / kRank;
int kMacroWorkItems = kMicroPerMacroM * kMicroPerMacroN;
int kNumMilestones = K_DIM / R_RANK;
int kKgBytesA = PP_HASH_H * kRank;
int kKgSliceB = PP_HASH_W * kRank;
int kMacroKgStripA = kMicroPerMacroM * kKgBytesA;
int kMacroKgStripB = kMicroPerMacroN * kKgSliceB;
int kMacroKbBlockA = kKGroups * kMacroKgStripA;
int kMacroKbBlockB = kKGroups * kMacroKgStripB;

namespace {

void update_derived() {
    kKR = R_RANK;
    kMicroPerMacroM = kMacroM / kMR;
    kMicroPerMacroN = kMacroN / kNR;
    kHashPerMacroM = kMacroM / hash_tile_mr();
    kHashPerMacroN = kMacroN / hash_tile_nr();
    kPanelA = kKR * kMR;
    kPanelB = kKR * kNR;
    kKGroups = kKR / kRank;
    kMacroWorkItems = kHashPerMacroM * kHashPerMacroN;
    kNumMilestones = K_DIM / kKR;
    kKgBytesA = kMR * kRank;
    kKgSliceB = kNR * kRank;
    kMacroKgStripA = kMicroPerMacroM * kKgBytesA;
    kMacroKgStripB = kMicroPerMacroN * kKgSliceB;
    kMacroKbBlockA = kKGroups * kMacroKgStripA;
    kMacroKbBlockB = kKGroups * kMacroKgStripB;
}

bool is_supported_tile(int mr, int nr) {
    if (mr == 4 && nr == 4) {
        return true;
    }
    if (mr == 4 && nr == 8) {
        return true;
    }
    if (mr == PP_HASH_H && (nr == 8 || nr == 16)) {
        return true;
    }
    return false;
}

bool is_supported_macro(int macro_m, int macro_n) {
    if (macro_m == kMacroSmall && macro_n == kMacroSmall) {
        return true;
    }
    if (macro_m == kMacroLarge && macro_n == kMacroLarge) {
        return true;
    }
    return false;
}

/* Soft default when --ocl-macro omitted: always 128×128. */
void default_macro(int *macro_m, int *macro_n) {
    *macro_m = kMacroLarge;
    *macro_n = kMacroLarge;
}

} // namespace

bool configure(int mr, int nr, int macro_m, int macro_n) {
    if (!is_supported_tile(mr, nr)) {
        std::fprintf(stderr, "[ocl] register tile must be 4x4, 4x8, 8x8, or 8x16 (got %dx%d)\n",
                     mr, nr);
        return false;
    }

    if (macro_m <= 0 || macro_n <= 0) {
        default_macro(&macro_m, &macro_n);
    } else if (!is_supported_macro(macro_m, macro_n)) {
        std::fprintf(stderr,
                     "[ocl] macro must be 64x64 or 128x128 (got %dx%d)\n", macro_m, macro_n);
        return false;
    }

    if (macro_m % mr != 0 || macro_n % nr != 0) {
        std::fprintf(stderr,
                     "[ocl] tile %dx%d must divide macro block %dx%d\n", mr, nr, macro_m,
                     macro_n);
        return false;
    }
    if (kKR % kRank != 0 || K_DIM % R_RANK != 0) {
        std::fprintf(stderr, "[ocl] KR/rank layout mismatch\n");
        return false;
    }
    const int hash_nr = (mr == 4 && nr == 4) ? 8 : nr;
    const int work_items = (macro_m / mr) * (macro_n / hash_nr);
    if (work_items > kMacroWorkItemsMax) {
        std::fprintf(stderr,
                     "[ocl] tile %dx%d / macro %dx%d needs %d work-items per macro (max %d)\n",
                     mr, nr, macro_m, macro_n, work_items, kMacroWorkItemsMax);
        return false;
    }
    kMR = mr;
    kNR = nr;
    kMacroM = macro_m;
    kMacroN = macro_n;
    update_derived();
    return true;
}

int hash_tile_mr() { return kMR; }

int hash_tile_nr() { return (kMR == 4 && kNR == 4) ? 8 : kNR; }

bool wi_row_major() { return false; }

int hash_tiles_per_macro() {
    return (kMacroM / hash_tile_mr()) * (kMacroN / hash_tile_nr());
}

} // namespace case32
