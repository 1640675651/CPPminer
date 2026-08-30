// Case 5: gemmstone int8 GEMM → OpenCL binary "case5_igemm" (Gen12LP or XeHPG).

#include "case5_ngen_gemm.hpp"
#include "case5_gemm_launch.hpp"
#include "case5_kernel_select.hpp"
#include "case5_xor_tile.hpp"

#include "gemmstone/generator.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"

#include "ngen_opencl.hpp"

#include <cstdio>
#include <exception>
#include <cstdlib>
#include <string>
#include <vector>
namespace gemmstone {
BinaryOp PostOpsProblem::toBinaryOp(const PostOps::entry_t &) { return BinaryOp::Add; }

namespace microkernel {
Package::Status Package::finalize(const ClobberSet &) {
    // Case5 only uses Generator::gemm, not microkernels. Stub for full
    // Generator<Gen12LP> template instantiation.
    status = Status::Success;
    return status;
}
} // namespace microkernel
} // namespace gemmstone

namespace case5_ngen {
namespace {

using namespace ngen;
using namespace gemmstone;

Product detect_product(cl_context ctx, cl_device_id device) {
    Product product{};
    try {
        product = OpenCLCodeGenerator<HW::Unknown>::detectHWInfo(ctx, device);
    } catch (...) {
        product.family = ProductFamily::GenericXeLP;
        product.type = PlatformType::Integrated;
    }
    if (product.family == ProductFamily::Unknown) {
        product.family = ProductFamily::GenericXeLP;
        product.type = PlatformType::Integrated;
    }
    return product;
}

const char *product_family_name(ProductFamily family) {
    switch (family) {
    case ProductFamily::GenericXeLP:
        return "XeLP/Gen12LP";
    case ProductFamily::GenericXeHP:
        return "XeHP";
    case ProductFamily::GenericXeHPG:
        return "XeHPG";
    case ProductFamily::DG2:
        return "DG2";
    case ProductFamily::MTL:
        return "MTL (Core Ultra / Xe-LPG)";
    case ProductFamily::ARL:
        return "ARL (Arrow Lake)";
    case ProductFamily::GenericXeHPC:
        return "XeHPC";
    case ProductFamily::PVC:
        return "PVC";
    case ProductFamily::PVCVG:
        return "PVCVG";
    case ProductFamily::GenericXe2:
        return "Xe2";
    case ProductFamily::BMG:
        return "BMG";
    case ProductFamily::LNL:
        return "LNL (Lunar Lake)";
    case ProductFamily::GenericXe3:
        return "Xe3";
    case ProductFamily::GenericXe3p:
        return "Xe3p";
    case ProductFamily::NVLP:
        return "NVLP";
    case ProductFamily::CRI:
        return "CRI";
    default:
        return "unknown";
    }
}

bool is_supported_product(const Product &product) {
    const Core core = getCore(product.family);
    return core == Core::XeLP || core == Core::XeHPG;
}

HW select_hw(const Product &product) {
    switch (getCore(product.family)) {
    case Core::XeLP:
        return HW::Gen12LP;
    case Core::XeHPG:
        return HW::XeHPG;
    default:
        return HW::Unknown;
    }
}

const char *hw_name(HW hw) {
    switch (hw) {
    case HW::Gen12LP:
        return "Gen12LP";
    case HW::XeHPG:
        return "XeHPG";
    default:
        return "unknown";
    }
}

void set_binary_packaging(HW hw, cl_device_id device) {
    // XeHPG+ drivers expect zebin-first; XeLP used patch-token on older stacks.
    detail::tryZebinFirst(device, true, hw >= HW::XeHPG);
}

constexpr char kKernelName[] = "case5_igemm";
constexpr char kKernelNameNop[] = "case51_igemm";
constexpr int kMilestoneK = 128;

bool case5_debug_select() {
    static const bool on = [] {
        const char *v = std::getenv("CASE5_DEBUG_SELECT");
        return v && v[0] != '0';
    }();
    return on;
}

void fill_xor_subtile_driver_info(DriverInfo &out, const XorSubtileDims &xsd) {
    out.xorSubM = xsd.subM;
    out.xorSubN = xsd.subN;
    out.xorSubGridM = xsd.subGridM;
    out.xorSubGridN = xsd.subGridN;
    out.xorOwnedM = xsd.ownedM;
    out.xorOwnedN = xsd.ownedN;
}

GEMMProblem makeProblem(const Product &product, bool xor_nop, bool fused_jackpot) {
    GEMMProblem problem;
    problem.Ta = problem.Ta_ext = Type::s8;
    problem.Tb = problem.Tb_ext = Type::s8;
    problem.Tc = problem.Tc_ext = Type::s32;
    problem.Ts = Type::f32;
    problem.alpha = 1;
    problem.beta = 0;
    problem.A.layout = MatrixLayout::T;
    problem.B.layout = MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;
    problem.A.crosspack = problem.B.crosspack = problem.C.crosspack = 1;
    problem.A.packSize = problem.B.packSize = problem.C.packSize = 0;
    problem.A.setAlignment(problem.A.defaultAlignment(problem.Ta_ext));
    problem.B.setAlignment(problem.B.defaultAlignment(problem.Tb_ext));
    problem.C.setAlignment(problem.C.defaultAlignment(problem.Tc_ext));
    problem.product = product;
    problem.case5TileXor = true;
    problem.case5TileXorNop = xor_nop;
    problem.case5TileXorWrapGrf = fused_jackpot && !xor_nop;
    problem.case5TileXorWrap = fused_jackpot && !xor_nop;
    problem.case5TileXorBlake3 = fused_jackpot && !xor_nop;
    problem.case5FuseJackpot = fused_jackpot && !xor_nop;
    if (problem.case5TileXorWrapGrf) {
        // 0 = keep fold in wrap-GRF only (BLAKE3 path writes digests, not tile_xor).
        // Non-Blake wrap-GRF uses storeMode 2 (epilogue flush) elsewhere.
        problem.case5TileXorWrapGrfStoreMode = problem.case5TileXorBlake3 ? 0 : 2;
        problem.case5TileXorWrapGrfStageUnified = true;
    }
    return problem;
}

int device_eu_count(cl_device_id device) {
    cl_uint units = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr)
            == CL_SUCCESS) {
        return static_cast<int>(units);
    }
    return 128;
}

DriverInfo toDriverInfo(HW hw, cl_device_id device, const GEMMStrategy &strategy,
        const CommonDriverInfo &di, const XorSubtileDims &xsd) {
    DriverInfo out;
    out.unrollM = di.unroll[LoopM];
    out.unrollN = di.unroll[LoopN];
    fill_xor_subtile_driver_info(out, xsd);
    out.unrollK = di.unroll[LoopK];
    out.wgM = di.wg[LoopM];
    out.wgN = di.wg[LoopN];
    out.wgK = di.wg[LoopK];
    out.subgroupSize = di.subgroupSize;
    out.slm = di.slm;
    out.wgExpand = di.wgExpand;
    out.fixedWG = di.fixedWG();
    out.fusedEUs = di.fusedEUs();
    out.isNMK = di.isNMK();
    out.hw = hw;
    out.euCount = device_eu_count(device);
    out.strategy = strategy;
    out.commonDriver = di;
    return out;
}

void log_candidate_rejection(const CatalogCandidate &candidate, const std::string &reason,
                             std::vector<std::string> *rejected) {
    std::string line = candidate.label;
    if (!candidate.strategy.empty()) {
        line += " (";
        line += candidate.strategy;
        line += ')';
    }
    line += ": ";
    line += reason;
    if (rejected) {
        rejected->push_back(std::move(line));
    }
}

void log_strategy_fallback(const char *chosen, const std::vector<std::string> &rejected,
                           std::string *selection_log) {
    if (rejected.empty()) {
        return;
    }
    std::string log = "fallback ";
    log += chosen;
    log += " after catalog failures:\n";
    for (const auto &line : rejected) {
        log += "  ";
        log += line;
        log += '\n';
    }
    if (selection_log) {
        *selection_log = log;
    }
    std::fprintf(stderr, "Case5 strategy %s", log.c_str());
}

template <HW hw>
cl_kernel build_igemm_kernel_impl(cl_context ctx, cl_device_id device, Product product,
                                  const BuildParams &dims, DriverInfo *info, std::string *err,
                                  bool xor_nop, bool fused_jackpot) {
    set_binary_packaging(hw, device);

    auto problem = makeProblem(product, xor_nop, fused_jackpot);
    const auto selection = select_case5_candidates(hw, product, device, problem, dims);
    product.stepping = selection.stepping;

    std::vector<CatalogCandidate> candidates = selection.candidates;

    std::string last_err;
    std::vector<std::string> rejected;
    for (const auto &candidate : candidates) {
        if (case5_debug_select()) {
            std::fprintf(stderr, "Case5 trying %s\n", candidate.label.c_str());
            std::fflush(stderr);
        }
        GEMMStrategy strategy(hw);
        EvaluateAuxOutput aux;
        if (candidate.from_catalog) {
            evaluate(*candidate.entry, selection.eval_params, aux);
        }

        std::string prep_err;
        if (!prepare_case5_strategy(hw, product, problem, candidate, selection.match_tags.c_str(),
                    aux, selection.systolic_hw, dims.k, strategy, &prep_err)) {
            last_err = candidate.label + ": " + prep_err;
            if (candidate.from_catalog) {
                log_candidate_rejection(candidate, prep_err, &rejected);
            }
            continue;
        }

        try {
            if (strategy.unroll[LoopK] <= 0 || (kMilestoneK % strategy.unroll[LoopK]) != 0) {
                throw std::runtime_error("Case5 milestone_k must be a multiple of unrollK");
            }
            problem.case5XorPeriod = kMilestoneK / strategy.unroll[LoopK];
            const auto xsd =
                    compute_xor_subtile_dims(strategy.unroll[LoopM], strategy.unroll[LoopN]);
            problem.case5XorSubM = xsd.subM;
            problem.case5XorSubN = xsd.subN;
            problem.case5XorSubGridM = xsd.subGridM;
            problem.case5XorSubGridN = xsd.subGridN;
            if (dims.k > 0) {
                problem.case5XorMaxMilestones = std::max(1, dims.k / kMilestoneK);
            }
            if (problem.case5TileXorWrap) {
                problem.case5XorOutputMilestones =
                        std::max(1, problem.case5XorMaxMilestones / 2);
            }

            InterfaceHandler iface(hw);
            init_case5_gemm_interface(iface, hw, problem, strategy,
                    problem.case5TileXorNop ? kKernelNameNop : kKernelName);

            const auto common_driver = Generator<hw>::driverInfo(problem, strategy);
            if (info) {
                *info = toDriverInfo(hw, device, strategy, common_driver, xsd);
                info->hwName = hw_name(hw);
                info->strategyNameOwned = candidate.label;
                info->strategyName = info->strategyNameOwned.c_str();
                if (!candidate.from_catalog) {
                    log_strategy_fallback(candidate.label.c_str(), rejected, &info->selectionLog);
                } else {
                    info->selectionLog.clear();
                }
            } else if (!candidate.from_catalog) {
                log_strategy_fallback(candidate.label.c_str(), rejected, nullptr);
            }

            Generator<hw> gen(product);
            gen.gemm(problem, strategy, iface);
            cl_kernel k = gen.getKernel(ctx, device);
            if (!k) {
                throw std::runtime_error("getKernel returned null");
            }
            return k;
        } catch (const std::exception &e) {
            last_err = candidate.label + ": " + e.what();
            if (candidate.from_catalog) {
                log_candidate_rejection(candidate, e.what(), &rejected);
            }
        }
    }

    if (err) {
        *err = last_err.empty() ? "no Case5 strategy candidates succeeded" : last_err;
    }
    return nullptr;
}
} // namespace

bool is_supported_device(cl_context ctx, cl_device_id device, std::string *err) {
    const Product product = detect_product(ctx, device);
    if (is_supported_product(product)) {
        return true;
    }
    if (err) {
        *err = std::string("Case 5 requires XeLP/Gen12LP or XeHPG (Core Ultra MTL/ARL). Detected ")
               + product_family_name(product.family)
               + " — unsupported ISA for Case 5.";
    }
    return false;
}

cl_kernel build_igemm_kernel(cl_context ctx, cl_device_id device, const BuildParams *dims,
                             DriverInfo *info, std::string *err, bool xor_nop,
                             bool fused_jackpot) {
    BuildParams build_dims;
    if (dims) {
        build_dims = *dims;
    }
    try {
        const Product product = detect_product(ctx, device);
        if (!is_supported_product(product)) {
            throw std::runtime_error(
                std::string("unsupported GPU: ") + product_family_name(product.family)
                + " (Case 5 supports Gen12LP/XeLP and XeHPG only)");
        }

        const HW hw = select_hw(product);
        switch (hw) {
        case HW::Gen12LP:
            return build_igemm_kernel_impl<HW::Gen12LP>(ctx, device, product, build_dims, info,
                    err, xor_nop, fused_jackpot);
        case HW::XeHPG:
            return build_igemm_kernel_impl<HW::XeHPG>(ctx, device, product, build_dims, info, err,
                    xor_nop, fused_jackpot);
        default:
            throw std::runtime_error("internal error: unsupported HW selection");
        }
    } catch (const std::exception &e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    } catch (...) {
        if (err) {
            *err = "unknown gemmstone exception";
        }
        return nullptr;
    }
}
} // namespace case5_ngen
