#pragma once

#include <cstdlib>
#include "case5_ngen_gemm.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/problem.hpp"

#include "ngen_core.hpp"

#include <string>
#include <vector>

namespace gemmstone {
namespace kcatalog {
struct Entry;
}
} // namespace gemmstone

namespace case5_ngen {

struct CatalogCandidate {
    const gemmstone::kcatalog::Entry *entry = nullptr;
    std::string label;
    std::string strategy;
    bool seed_gen12_unroll = false;
    bool from_catalog = false;
    int unroll_m = 0; // non-catalog candidates: explicit unroll (0 = parser default)
    int unroll_n = 0;
    bool preferred = false; // built-in measured-best strategy; keeps the problem's real alignment
    bool pack_a = false;    // A in gemmstone's packed layout (16-row panels, 4 k per dword; prep writes it)
    bool pack_b = false;    // B in gemmstone's packed layout (64-column panels, 32 k per column; prep writes it per job)
};

// CASE5_A_PACKED: unset = packed A for the XeHPG preferred kernel (default), 0 = off,
// 1 = also force it onto every other candidate (experiments).
// CASE5_B_PACKED: unset = packed B for the XeHPG preferred kernel (default), 0 = off,
// 1 = also force it onto every other candidate (experiments; CASE5_B_CP / CASE5_B_TILE).
inline int case5_b_packed_env() {
    const char *v = std::getenv("CASE5_B_PACKED");
    if (!v || !v[0]) return -1;
    return v[0] == '1' ? 1 : 0;
}

inline int case5_a_packed_env() {
    const char *v = std::getenv("CASE5_A_PACKED");
    if (!v || !v[0]) return -1;
    return v[0] == '1' ? 1 : 0;
}

struct Case5CandidateList {
    std::vector<CatalogCandidate> candidates;
    std::string match_tags;
    gemmstone::EvaluateParams eval_params{};
    bool systolic_hw = false;
    int stepping = 0;
};

Case5CandidateList select_case5_candidates(ngen::HW hw, const ngen::Product &product,
                                          cl_device_id device, const gemmstone::GEMMProblem &problem,
                                          const BuildParams &dims);

bool prepare_case5_strategy(ngen::HW hw, const ngen::Product &product, gemmstone::GEMMProblem &problem,
                            const CatalogCandidate &candidate, const char *tags,
                            const gemmstone::EvaluateAuxOutput &aux, bool systolic_hw, int k,
                            gemmstone::GEMMStrategy &strategy, std::string *err);

} // namespace case5_ngen
