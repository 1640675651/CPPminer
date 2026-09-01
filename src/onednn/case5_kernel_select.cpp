#include "case5_kernel_select.hpp"

#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gemmstone/strategy_parser.hpp"

#include "ngen_core.hpp"

#include <exception>
#include <string>
#include <vector>

namespace gemmstone {
const kcatalog::Catalog &case5_catalog();
}

namespace case5_ngen {
namespace {

using namespace ngen;
using namespace gemmstone;
using namespace kcatalog;

#ifndef CL_DEVICE_FEATURE_CAPABILITIES_INTEL
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL 0x4908
#endif
#ifndef CL_DEVICE_FEATURE_FLAG_DPAS_INTEL
#define CL_DEVICE_FEATURE_FLAG_DPAS_INTEL (1 << 1)
#endif
#ifndef CL_DEVICE_STEPPING_ID_INTEL
#define CL_DEVICE_STEPPING_ID_INTEL 0x10B6
#endif

constexpr int kMilestoneK = 128;

constexpr char kFallbackGen12LP[] = "sb4 sb8 sb l4 int k32 cab1 wg 4x4 ek";
constexpr char kFallbackXeHPG[] = "sb4 sb8 sb l4 int k32 cab1 wg 4x4 ek grf256";

bool device_has_systolic(cl_device_id device, const Product &product) {
    cl_bitfield caps = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_FEATURE_CAPABILITIES_INTEL, sizeof(caps), &caps,
                nullptr)
            == CL_SUCCESS) {
        return (caps & CL_DEVICE_FEATURE_FLAG_DPAS_INTEL) != 0;
    }
    return hasSystolic(product.family);
}

int device_eu_count(cl_device_id device) {
    cl_uint units = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr)
            == CL_SUCCESS) {
        return static_cast<int>(units);
    }
    return 128;
}

int device_stepping(cl_device_id device) {
    cl_uint stepping = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_STEPPING_ID_INTEL, sizeof(stepping), &stepping,
                nullptr)
            == CL_SUCCESS) {
        return static_cast<int>(stepping);
    }
    return 0;
}

void apply_systolic_caps(bool systolic_hw, GEMMStrategy &strategy) {
    strategy.systolicAvailable = systolic_hw;
    if (!systolic_hw) {
        strategy.systolic = false;
        strategy.dpasw = false;
        strategy.fixedSystolic = false;
    }
}

bool entry_case5_compatible(const Entry &entry) {
    const auto &di = entry.driverInfo;
    if (di.kParallel() || di.kParallelLocal() || di.kParallelVariable()) {
        return false;
    }
    const int unroll_k = di.unroll[LoopK];
    if (unroll_k <= 0 || (kMilestoneK % unroll_k) != 0) {
        return false;
    }
    return true;
}

bool strategy_postparse_compatible(const GEMMStrategy &strategy) {
    if (strategy.kParallel || strategy.kParallelLocal || strategy.kParallelVariable) {
        return false;
    }
    if (strategy.fuseBeta || strategy.fusePostOps) {
        return false;
    }
    if (strategy.registerOutput()) {
        return false;
    }
    const int unroll_k = strategy.unroll[LoopK];
    if (unroll_k <= 0 || (kMilestoneK % unroll_k) != 0) {
        return false;
    }
    return true;
}

struct OnednnSelectInput {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t batch = 0;
    int64_t lda = 0;
    int64_t ldb = 0;
    int64_t ldc = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
};

// Mirrors gen_nocopy_desc_t::select_kernel() match/eval setup (gen_kernel.cpp).
struct OnednnSelectSetup {
    std::vector<MatchParams> match_params;
    EvaluateParams eval_params{};
    std::string match_tags;
};

OnednnSelectSetup build_onednn_select_setup(HW hw, bool systolic_hw, const Product &product,
        const GEMMProblem &problem, int stepping, const OnednnSelectInput &in) {
    OnednnSelectSetup setup;

    MatchParams base(hw, systolic_hw, product, problem);
    // oneDNN relaxes gemmstone's default Tc_ext >= Tc restriction.
    base.precisionCExt = '\0';

    base.sizes.m = in.m;
    base.sizes.n = in.n;
    base.sizes.k = in.k;
    base.sizes.batch = in.batch;
    base.stepping = stepping;
    base.ignoreCase = true;

    const bool can_2d_a = (in.lda * problem.Ta_ext) <= 16777216;
    const bool can_2d_b = (in.ldb * problem.Tb_ext) <= 16777216;
    const bool can_2d_c = (in.ldc * problem.Tc_ext) <= 16777216;

    auto tags = const_cast<char *>(base.tags);
    while (*tags) {
        ++tags;
    }
    if (problem.A.needA64 || problem.B.needA64 || problem.C.needA64) {
        *tags++ = ReqBatchN;
    }
    if (can_2d_a) {
        *tags++ = ReqBlock2DA;
    }
    if (can_2d_b) {
        *tags++ = ReqBlock2DB;
    }
    if (can_2d_c) {
        *tags++ = ReqBlock2DC;
    }

    setup.match_params.push_back(base);

    // Same int-acc → float-acc pattern expansion as oneDNN (no-op for pure O,O,I).
    if (problem.Tc == Type::s32) {
        const size_t npatterns = setup.match_params.size();
        std::vector<MatchParams> float_strats;
        float_strats.reserve(npatterns);
        for (size_t i = 0; i < npatterns; ++i) {
            auto start = setup.match_params[i];
            if (!std::string("I").compare(start.selector.precisions[2])) {
                float_strats.push_back(start);
                float_strats.back().selector.precisions[2] = "S";
            }
        }
        setup.match_params.insert(
                setup.match_params.begin(), float_strats.begin(), float_strats.end());
    }

    setup.eval_params.sizes = base.sizes;
    setup.eval_params.alpha = in.alpha;
    setup.eval_params.beta = in.beta;
    setup.eval_params.postOps = !problem.postOps.empty();
    setup.eval_params.cConvert = (problem.Tc != problem.Tc_ext);
    setup.eval_params.batch = (problem.batchDims > 0);
    setup.eval_params.deterministic = false;
    setup.eval_params.Tc_ext = problem.Tc_ext;

    setup.match_tags = setup.match_params[0].tags;
    return setup;
}

OnednnSelectInput onednn_select_input_from_dims(const BuildParams &dims) {
    OnednnSelectInput in;
    in.m = dims.m;
    in.n = dims.n;
    in.k = dims.k;
    in.batch = 0;
    in.alpha = 1.0f;
    in.beta = 0.0f;
    // Row-major A: lda>=K. Column-major A: lda>=M. Column-major B: ldb>=K. Row-major B: ldb>=N.
    in.lda = dims.lda > 0 ? dims.lda : (dims.a_row_major ? dims.k : dims.m);
    in.ldb = dims.ldb > 0 ? dims.ldb : (dims.b_row_major ? dims.n : dims.k);
    in.ldc = dims.ldc > 0 ? dims.ldc : dims.m;
    return in;
}

std::vector<CatalogCandidate> fallback_candidates(HW hw) {
    std::vector<CatalogCandidate> out;
    if (hw == HW::XeHPG) {
        out.push_back(CatalogCandidate{nullptr, "fallback-grf256", kFallbackXeHPG, true, true});
    } else {
        out.push_back(CatalogCandidate{nullptr, "fallback-Gen12LP", kFallbackGen12LP, true, true});
    }
    return out;
}

void append_catalog_candidate(std::vector<CatalogCandidate> &out, const Entry *entry) {
    if (!entry) {
        return;
    }
    CatalogCandidate candidate;
    candidate.entry = entry;
    candidate.from_catalog = true;
    candidate.label = "catalog-" + std::to_string(entry - case5_catalog().entries);
    candidate.strategy = entry->strategy;
    out.push_back(std::move(candidate));
}

} // namespace

Case5CandidateList select_case5_candidates(HW hw, const Product &product, cl_device_id device,
                                           const GEMMProblem &problem, const BuildParams &dims) {
    Case5CandidateList result;
    result.systolic_hw = device_has_systolic(device, product);
    result.stepping = device_stepping(device);

    const OnednnSelectInput select_in = onednn_select_input_from_dims(dims);
    OnednnSelectSetup setup = build_onednn_select_setup(
            hw, result.systolic_hw, product, problem, result.stepping, select_in);

    result.match_tags = setup.match_tags;
    result.eval_params = setup.eval_params;
    result.eval_params.euCount = device_eu_count(device);

    EvaluateAuxOutput aux;
    const auto ranked = select(case5_catalog(), static_cast<int>(setup.match_params.size()),
            setup.match_params.data(), result.eval_params, aux);

    const auto fallbacks = fallback_candidates(hw);
    result.candidates.reserve(ranked.size() + fallbacks.size());

    // Case 5 filters apply after oneDNN's ordered catalog list (same contract as entries[0..N]).
    for (const Entry *entry : ranked) {
        if (!entry || !entry_case5_compatible(*entry)) {
            continue;
        }
        append_catalog_candidate(result.candidates, entry);
    }

    for (auto &fallback : fallbacks) {
        result.candidates.push_back(std::move(fallback));
    }
    return result;
}

bool prepare_case5_strategy(HW hw, const Product &product, GEMMProblem &problem,
                            const CatalogCandidate &candidate,
                            const char *tags, const EvaluateAuxOutput &aux, bool systolic_hw,
                            int k, GEMMStrategy &strategy, std::string *err) {
    try {
        if (candidate.from_catalog) {
            const Entry &entry = *candidate.entry;
            if (!isPacked(problem.A.layout)
                    && problem.Ta_ext.paddedSize() >= problem.Ta.paddedSize()) {
                problem.A.setAlignment(
                        std::max(problem.Ta_ext.paddedSize(), entry.driverInfo.alignment[0]));
            }
            if (!isPacked(problem.B.layout)
                    && problem.Tb_ext.paddedSize() >= problem.Tb.paddedSize()) {
                problem.B.setAlignment(
                        std::max(problem.Tb_ext.paddedSize(), entry.driverInfo.alignment[1]));
            }
            if (!isPacked(problem.C.layout)) {
                problem.C.setAlignment(std::max(problem.Tc_ext.paddedSize(),
                        entry.restrictions.alignment[2]));
            }

            strategy = GEMMStrategy(hw, product.stepping);
            strategy.unroll[LoopM] = entry.driverInfo.unroll[LoopM];
            strategy.unroll[LoopN] = entry.driverInfo.unroll[LoopN];
            parseStrategy(entry.strategy, hw, problem, strategy);
        } else {
            strategy = GEMMStrategy(hw, product.stepping);
            if (candidate.seed_gen12_unroll) {
                strategy.unroll[LoopM] = 32;
                strategy.unroll[LoopN] = 16;
            }
            parseStrategy(candidate.strategy.c_str(), hw, problem, strategy);
        }

        apply_systolic_caps(systolic_hw, strategy);
        modifyStrategy(strategy, aux);
        strategy.panelCheck |= (isPacked(problem.A.layout) || isPacked(problem.B.layout));

        if (strategy.kParallel && k > 0) {
            auto k_min = aux.k0 * aux.wgK;
            if (k <= k_min) {
                strategy.kParallel = false;
                strategy.C.atomic = false;
                strategy.CO.atomic = false;
            }
        }
        if (strategy.fixedSystolic) {
            strategy.GRFs = 256;
        }

        adjustStrategy(hw, problem, strategy, tags);
        if (!strategy_postparse_compatible(strategy)) {
            throw std::runtime_error("strategy incompatible with Case5 tile XOR");
        }
        strategy.preflight(hw, problem);
        return true;
    } catch (const std::exception &e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

} // namespace case5_ngen
