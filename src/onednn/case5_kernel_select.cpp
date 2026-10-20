#include "case5_kernel_select.hpp"
#include "case5_xor_tile.hpp"

#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gemmstone/strategy_parser.hpp"

#include "ngen_core.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Measured-best XeHPG int8 strategy (kernel.db entry "sS32x2 sB32 ... wg 16x2", unroll 16x64).
// On an Arc A770 it reaches 197 TOPS at 16384x16384x8192 and 216 TOPS at 32768x16384x4096
// (GEMM-only), versus 121-145 TOPS for the entries the catalog model ranks first; it also wins
// at 4096^3 and 8192^3. Requires 64-byte aligned A rows and 4-byte aligned B (entry restriction).
// CASE5_PREFER=0 disables it and falls back to the catalog ranking.
// k128 makes the 128-K milestone one loop body (no per-step gate; GEMM speed unchanged). The real
// fold interleaves at dpas chain boundaries, which exist per row pass only with plain af (xaf
// chains the whole step and is ~2% faster for pure GEMM, so it stays for 5.1 / no-store runs).
constexpr char kPreferredXeHPGInt8Xaf[] =
        "sS32x2 sB32 aB wg 16x2 cb4x2 ks32 k128 xaf dw vav bo sr bk0 sn dm grf256 sys pab";
constexpr char kPreferredXeHPGInt8Af[] =
        "sS32x2 sB32 aB wg 16x2 cb4x2 ks32 k128 af dw vav bo sr bk0 sn dm grf256 sys pab";
// Packed A (the default on XeHPG, see case5_a_packed_env): A arrives in 16-row panels with 4 consecutive
// k per dword, so it is block-loaded (sB32) instead of gathered (sS32x2); the gathers needed two buffers
// to hide their latency, block loads need one, and the scan is ~7% faster (0.638 vs 0.688 s per
// 131072^2 attempt on the A770).
constexpr char kPreferredXeHPGInt8XafPackedA[] =
        "sB32 sB32 aB wg 16x2 cb4x2 ks32 k128 xaf dw vav bo sr bk0 sn dm grf256 sys pab";
constexpr char kPreferredXeHPGInt8AfPackedA[] =
        "sB32 sB32 aB wg 16x2 cb4x2 ks32 k128 af dw vav bo sr bk0 sn dm grf256 sys pab";
constexpr int kPreferredXeHPGUnrollM = 16;
constexpr int kPreferredXeHPGUnrollN = 64;

// ---- Kernel selection layers ------------------------------------------------------------------
// 1. per-device overrides (measured exceptions to the family default; empty for now)
// 2. per-family defaults (gemmstone HW family: the measured-best strategy on a representative device)
// 3. gemmstone's catalog ranking, then generic fallbacks
// gemmstone's own selector cannot choose these: its catalog has no packed int8 entries, and for the
// miner's layout it matches a single entry per family (Gen12LP: entry 30 only).
//
// A family kernel names a tile (unroll), the strategies for plain and packed A, and the packing it
// wants: A in panels of unroll_m rows (the prep writes 16-row panels, 4 k per dword), B in panels of
// unroll_n columns with pack_b_cp consecutive k per column (written once per job). A combination
// whose strategy is nullptr does not build and is skipped (the catalog takes over).
struct FamilyKernel {
    HW hw;
    const char *label;
    const char *measured_on;     // documentation
    bool needs_systolic;
    int unroll_m, unroll_n;
    const char *plain;           // A row-major (not packed)
    const char *plain_alt;       // XMX real fold with CASE5_XOR_XAF=0 (af instead of xaf); nullptr = plain
    const char *packed;          // A packed
    const char *packed_alt;
    int pack_a_cp;               // 0 = never pack A
    int pack_a_max_k;            // pack A only while K <= this (0 = any K)
    int pack_b_cp;               // 0 = never pack B
    int min_align_a, min_align_b;
};

// UHD 770 (Gen12LP, 32 EUs): gemm_xor pure GEMM 8192^2 x 4096 (note.md 2026-10-20): 177.8 ms = 97.4% of
// the dp4a peak, against 192.6 ms (89.9%) for catalog entry 30 plain; verified with packed A and B.
// Builds only with packed operands (plain runs out of registers), so plain = nullptr.
constexpr char kGen12LPInt8Packed[] = "sb16 sb16 sb l4 nse k16 cab1 wg 4x4";

const FamilyKernel kFamilyKernels[] = {
    {HW::XeHPG, "xehpg-preferred", "Arc A770", true, kPreferredXeHPGUnrollM, kPreferredXeHPGUnrollN,
     kPreferredXeHPGInt8Xaf, kPreferredXeHPGInt8Af, kPreferredXeHPGInt8XafPackedA,
     kPreferredXeHPGInt8AfPackedA, 4, 4096, 32, 64, 4},
    {HW::Gen12LP, "gen12lp-preferred", "UHD 770", false, 16, 32,
     nullptr, nullptr, kGen12LPInt8Packed, nullptr, 4, 0, 4, 1, 1},
};

const FamilyKernel *family_kernel(HW hw) {
    for (const auto &f : kFamilyKernels)
        if (f.hw == hw) return &f;
    return nullptr;
}

#ifndef CL_DEVICE_ID_INTEL
#define CL_DEVICE_ID_INTEL 0x4251
#endif

// Per-device overrides: {PCI device id, kernel}. Placeholder: no measured exception yet.
// Candidate, not enabled: Arc A380 (8086:56a5). PR #36 measured catalog entry 333 ("sB64 sB32 aB wg 8x4
// cab4 ks64 dw vav bo bk0 sm sn grf256 sys pab l4 sr", 16x32, `af` dropped: +1.06%) at 15.0 TMAC/s against
// 9.7 for the catalog's top pick, before the XeHPG family kernel existed; it has not been compared with
// the family kernel on an A380.
struct DeviceOverride {
    uint32_t pci_device_id;
    FamilyKernel kernel;
};
const std::vector<DeviceOverride> &device_overrides() {
    static const std::vector<DeviceOverride> kOverrides = {};
    return kOverrides;
}

const FamilyKernel *device_override(cl_device_id device, HW hw) {
    const auto &overrides = device_overrides();
    if (overrides.empty() || !device) return nullptr;
    cl_uint id = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_ID_INTEL, sizeof(id), &id, nullptr) != CL_SUCCESS) return nullptr;
    for (const auto &o : overrides)
        if (o.pci_device_id == id && o.kernel.hw == hw) return &o.kernel;
    return nullptr;
}

bool case5_prefer_measured_default() {
    const char *v = std::getenv("CASE5_PREFER");
    return !(v && v[0] == '0');
}

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
    if (unroll_k <= 0 || ((kMilestoneK % unroll_k) != 0 && (unroll_k % kMilestoneK) != 0)) {
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
    if (unroll_k <= 0 || ((kMilestoneK % unroll_k) != 0 && (unroll_k % kMilestoneK) != 0)) {
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
        out.push_back(CatalogCandidate{nullptr, "fallback-grf256", kFallbackXeHPG, true, false});
    } else {
        out.push_back(CatalogCandidate{nullptr, "fallback-Gen12LP", kFallbackGen12LP, true, false});
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
    // CASE5_DEBUG_RANK=1: print gemmstone's own ranking (model score = predicted time, lower is better).
    if (const char *dr = std::getenv("CASE5_DEBUG_RANK"); dr && dr[0] == '1') {
        int shown = 0;
        for (const Entry *e : ranked) {
            if (!e || shown >= 15) break;
            EvaluateAuxOutput a2;
            const double score = evaluate(*e, result.eval_params, a2);
            std::fprintf(stderr, "[rank %2d] catalog-%d score=%.4g :: %s\n", shown + 1,
                         int(e - case5_catalog().entries), score, static_cast<const char *>(e->strategy));
            ++shown;
        }
        std::fprintf(stderr, "[rank] %zu entries matched\n", ranked.size());
    }

    // Tuning override: CASE5_CATALOG_INDEX=<n> forces kernel.db entry n (sweeps).
    if (const char *idx = std::getenv("CASE5_CATALOG_INDEX")) {
        const int i = std::atoi(idx);
        if (idx[0] && i >= 0 && i < case5_catalog().entryCount) {
            append_catalog_candidate(result.candidates, &case5_catalog().entries[i]);
            return result;
        }
        std::fprintf(stderr, "CASE5_CATALOG_INDEX=%s out of range (0..%d)\n", idx,
                case5_catalog().entryCount - 1);
    }

    const auto fallbacks = fallback_candidates(hw);
    result.candidates.reserve(ranked.size() + fallbacks.size() + 2);

    // Tuning hook: CASE5_STRATEGY="<gemmstone strategy>" with CASE5_UNROLL=MxN is
    // tried first, as label "custom" (CASE5_ALIGN=AxBxC sets its assumed alignment).
    if (const char *s = std::getenv("CASE5_STRATEGY"); s && *s) {
        CatalogCandidate custom;
        custom.label = "custom";
        custom.strategy = s;
        for (char &c : custom.strategy) // '_' allowed as a separator for env-safe values
            if (c == '_')
                c = ' ';
        if (const char *u = std::getenv("CASE5_UNROLL"); u && *u)
            std::sscanf(u, "%dx%d", &custom.unroll_m, &custom.unroll_n);
        result.candidates.push_back(std::move(custom));
    }

    // Measured kernels first (device override, else the family default; CASE5_PREFER=0 skips both),
    // then the catalog.
    const FamilyKernel *fk = device_override(device, hw);
    if (!fk) fk = family_kernel(hw);
    if (fk && case5_prefer_measured_default() && (!fk->needs_systolic || result.systolic_hw)
            && problem.Ta == Type::s8 && problem.Tb == Type::s8
            && problem.A.alignment >= fk->min_align_a && problem.B.alignment >= fk->min_align_b) {
        CatalogCandidate c;
        c.label = fk->label;
        const bool realFold = !problem.case5TileXorNop && !case5_tile_xor_nostore_env()
                && case5_tile_xor_interleave_enabled();
        // XMX real fold: xaf with per-pass chains only in the first/last step of a milestone (default,
        // CASE5_XOR_XAF=hybrid). CASE5_XOR_XAF=0 uses the af strategy everywhere.
        const char *xafEnv = std::getenv("CASE5_XOR_XAF");
        const bool useAlt = realFold && xafEnv && xafEnv[0] == '0';
        // Packed A on XeHPG is single-buffered: at K > 4096 (A panels beyond the 16 MB L2) its block loads
        // cannot hide the misses and the double-buffered gathers win (A770 pure GEMM, 16384^2: K=8192 194.6
        // plain vs 176.0 packed A; K=16384 157.4 vs 175.6, but B-only packing gives 203.7 / 169.7), hence
        // pack_a_max_k. The miner uses K = 4096.
        c.pack_a = problem.A.layout == gemmstone::MatrixLayout::T && fk->pack_a_cp > 0
                && (case5_a_packed_env() == 1
                    || (case5_a_packed_env() != 0 && dims.k > 0
                        && (fk->pack_a_max_k == 0 || dims.k <= fk->pack_a_max_k)));
        // Packed B: one contiguous piece per thread and k step (A770: +3.8% per mining attempt, saves
        // energy under the power cap; UHD 770: part of the 97.4% kernel). Written once per job.
        c.pack_b = case5_b_packed_env() != 0 && fk->pack_b_cp > 0
                && problem.B.layout == gemmstone::MatrixLayout::N;
        c.pack_a_cp = fk->pack_a_cp;
        c.pack_b_cp = fk->pack_b_cp;
        const char *s = c.pack_a ? (useAlt && fk->packed_alt ? fk->packed_alt : fk->packed)
                                 : (useAlt && fk->plain_alt ? fk->plain_alt : fk->plain);
        // Gen12LP's kernel needs both operands packed; anything else falls through to the catalog.
        const bool usable = s && (fk->hw != HW::Gen12LP || (c.pack_a && c.pack_b));
        if (usable) {
            c.strategy = s;
            c.unroll_m = fk->unroll_m;
            c.unroll_n = fk->unroll_n;
            c.preferred = true;
            result.candidates.push_back(std::move(c));
        }
    }

    // Catalog layer: oneDNN's ordered catalog list as is (same contract as entries[0..N]), minus the
    // entries Case 5 cannot use. Measured choices belong in the device / family layers above.
    for (const Entry *entry : ranked) {
        if (entry && entry_case5_compatible(*entry)) {
            append_catalog_candidate(result.candidates, entry);
        }
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
            // candidate.strategy is the entry's string unless a preferred entry is tuned.
            parseStrategy(candidate.strategy.c_str(), hw, problem, strategy);
        } else {
            strategy = GEMMStrategy(hw, product.stepping);
            if (candidate.unroll_m > 0 && candidate.unroll_n > 0) {
                strategy.unroll[LoopM] = candidate.unroll_m;
                strategy.unroll[LoopN] = candidate.unroll_n;
            } else if (candidate.seed_gen12_unroll) {
                strategy.unroll[LoopM] = 32;
                strategy.unroll[LoopN] = 16;
            }
            if (!candidate.from_catalog && !candidate.seed_gen12_unroll && !candidate.preferred) {
                // Env strategies: assume catalog-like operand alignment (our buffers use padded
                // leading dimensions that are multiples of 64 bytes). CASE5_ALIGN=AxBxC overrides.
                int al_a = 64, al_b = 64, al_c = 64;
                if (const char *al = std::getenv("CASE5_ALIGN")) {
                    int a = 0, b = 0, c = 0;
                    if (std::sscanf(al, "%dx%dx%d", &a, &b, &c) == 3 && a > 0 && b > 0 && c > 0) {
                        al_a = a;
                        al_b = b;
                        al_c = c;
                    }
                }
                problem.A.setAlignment(std::max(problem.Ta_ext.paddedSize(), al_a));
                problem.B.setAlignment(std::max(problem.Tb_ext.paddedSize(), al_b));
                problem.C.setAlignment(std::max(problem.Tc_ext.paddedSize(), al_c));
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
