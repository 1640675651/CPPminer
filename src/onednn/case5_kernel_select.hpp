#pragma once

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
};

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
