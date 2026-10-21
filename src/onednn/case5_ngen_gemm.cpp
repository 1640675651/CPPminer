// Case 5: gemmstone int8 GEMM -> OpenCL binary "case5_igemm" (Gen12LP or XeHPG).

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

GEMMProblem makeProblem(const Product &product, bool xor_nop, bool fused_jackpot,
                        bool a_row_major, bool b_row_major) {
    GEMMProblem problem;
    problem.Ta = problem.Ta_ext = Type::s8;
    problem.Tb = problem.Tb_ext = Type::s8;
    problem.Tc = problem.Tc_ext = Type::s32;
    problem.Ts = Type::f32;
    problem.alpha = 1;
    problem.beta = 0;
    problem.A.layout = a_row_major ? MatrixLayout::T : MatrixLayout::N;
    problem.B.layout = b_row_major ? MatrixLayout::T : MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;
    problem.A.crosspack = problem.B.crosspack = problem.C.crosspack = 1;
    problem.A.packSize = problem.B.packSize = problem.C.packSize = 0;
    problem.A.setAlignment(problem.A.defaultAlignment(problem.Ta_ext));
    problem.B.setAlignment(problem.B.defaultAlignment(problem.Tb_ext));
    problem.C.setAlignment(problem.C.defaultAlignment(problem.Tc_ext));
    problem.product = product;
    problem.case5TileXor = true;
    problem.case5TileXorNop = xor_nop;
    // Fused jackpot: in-reg fold + Case5.6 BLAKE3 + in-kernel compare (diagnostic dumps digests/beats).
    problem.case5TileXorWrapGrf = fused_jackpot && !xor_nop;
    problem.case5TileXorWrap = fused_jackpot && !xor_nop;
    problem.case5TileXorBlake3 = fused_jackpot && !xor_nop;
    problem.case5FuseJackpot = fused_jackpot && !xor_nop;
    if (problem.case5TileXorWrapGrf) {
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

HashTileFilter g_hash_tile_filter = nullptr;

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
    if (case5_debug_select()) {
        std::fprintf(stderr, "Case5 rejected %s\n", line.c_str());
        std::fflush(stderr);
    }
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

    auto problem = makeProblem(product, xor_nop, fused_jackpot, dims.a_row_major, dims.b_row_major);
    const bool a_row_major = dims.a_row_major, b_row_major = dims.b_row_major;
    // Operand alignment from the actual leading dimensions (buffers are page aligned, offsets 0),
    // as oneDNN does before catalog matching. Without this the catalog entries that require
    // 64-byte aligned A (e.g. the fastest XeHPG int8 entries) never match.
    {
        auto ld_align = [](int64_t ld_bytes) {
            int al = 64;
            while (al > 1 && (ld_bytes % al) != 0) {
                al >>= 1;
            }
            return al;
        };
        const int64_t lda = dims.lda > 0 ? dims.lda : (a_row_major ? dims.k : dims.m);
        const int64_t ldb = dims.ldb > 0 ? dims.ldb : (b_row_major ? dims.n : dims.k);
        const int64_t ldc = dims.ldc > 0 ? dims.ldc : dims.m;
        problem.A.setAlignment(std::max<int>(problem.Ta_ext.paddedSize(), ld_align(lda * problem.Ta_ext)));
        problem.B.setAlignment(std::max<int>(problem.Tb_ext.paddedSize(), ld_align(ldb * problem.Tb_ext)));
        problem.C.setAlignment(std::max<int>(problem.Tc_ext.paddedSize(), ld_align(ldc * problem.Tc_ext)));
    }
    const auto selection = select_case5_candidates(hw, product, device, problem, dims);
    product.stepping = selection.stepping;

    std::vector<CatalogCandidate> candidates = selection.candidates;
    // CASE5_KERNEL=<label> (e.g. catalog-337) keeps only that candidate, for tuning sweeps.
    if (const char *only = std::getenv("CASE5_KERNEL"); only && *only) {
        std::vector<CatalogCandidate> kept;
        for (auto &candidate : candidates)
            if (candidate.label == only)
                kept.push_back(candidate);
        if (kept.empty()) {
            if (err)
                *err = std::string("CASE5_KERNEL=") + only + " is not among the selected candidates";
            return nullptr;
        }
        candidates = std::move(kept);
    }

    std::string last_err;
    std::vector<std::string> rejected;
    const auto case5BaseA = problem.A;   // packed A/B are per candidate; restore them for the next one
    const auto case5BaseB = problem.B;
    for (const auto &candidate : candidates) {
        problem.A = case5BaseA;
        problem.B = case5BaseB;
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
            } else if (case5_debug_select()) {
                std::fprintf(stderr, "Case5 rejected %s: %s\n", candidate.label.c_str(), prep_err.c_str());
            }
            continue;
        }

        try {
            const int milestone_k = kMilestoneK;
            const int unroll_k = strategy.unroll[LoopK];
            if (unroll_k <= 0
                    || ((kMilestoneK % unroll_k) != 0 && (unroll_k % kMilestoneK) != 0)
                    || (dims.k > 0 && (dims.k % unroll_k) != 0)) {
                throw std::runtime_error("Case5 milestone_k and unrollK must divide one another "
                        "and K (unrollK=" + std::to_string(unroll_k) + ", ka_load="
                        + std::to_string(strategy.ka_load) + ", kb_load="
                        + std::to_string(strategy.kb_load) + ")");
            }
            if (unroll_k <= kMilestoneK) {
                problem.case5XorPeriod = kMilestoneK / unroll_k;
                problem.case5XorEveryK = 0;
            } else {
                // Systolic kernels unroll K past one milestone: XOR inside the panel.
                problem.case5XorPeriod = 1;
                problem.case5XorEveryK = kMilestoneK;
            }
            const auto xsd =
                    compute_xor_subtile_dims(strategy.unroll[LoopM], strategy.unroll[LoopN]);
            if (g_hash_tile_filter && !g_hash_tile_filter(xsd.subM, xsd.subN)) {
                throw std::runtime_error("Case5 hash tile " + std::to_string(xsd.subM) + "x"
                        + std::to_string(xsd.subN) + " has no proof layout (hash-tile filter)");
            }
            problem.case5XorSubM = xsd.subM;
            problem.case5XorSubN = xsd.subN;
            problem.case5XorSubGridM = xsd.subGridM;
            problem.case5XorSubGridN = xsd.subGridN;
            if (dims.k > 0) {
                problem.case5XorMaxMilestones = std::max(1, dims.k / milestone_k);
            }
            if (problem.case5TileXorWrap) {
                problem.case5XorOutputMilestones =
                        std::max(1, problem.case5XorMaxMilestones / 2);
            }
            // Packed A: gemmstone packed columns (16-row panels = unrollM, 4 consecutive k per dword,
            // the systolic layout), written by the GPU prep (ocl_noisy_matrix_a_packed16) or packed on
            // the host. Default for the XeHPG preferred kernel; CASE5_A_PACKED=1 forces it everywhere.
            // Timing experiments (prep only writes 16-row packed A; other shapes compute on wrong data):
            // CASE5_A_PACKED=1 packs A for any candidate (panel = unrollM), CASE5_A_CP=<crosspack>
            // (default 4), CASE5_A_TILE=r,c; CASE5_B_PACKED=1 packs B (panel = unrollN),
            // CASE5_B_CP=<crosspack> (default 32), CASE5_B_TILE=r,c.
            auto envInt = [](const char *name, int dflt) {
                const char *v = std::getenv(name);
                return (v && v[0]) ? std::atoi(v) : dflt;
            };
            auto envTile = [](const char *name, uint16_t &r, uint16_t &c) {
                r = c = 0;
                if (const char *v = std::getenv(name); v && v[0]) {
                    int a = 0, b = 0;
                    if (std::sscanf(v, "%d,%d", &a, &b) == 2) { r = uint16_t(a); c = uint16_t(b); }
                }
            };
            // Packing comes with the candidate (family / device kernels, case5_kernel_select.cpp): panels
            // of the strategy's unroll, crosspack from the candidate. CASE5_A_PACKED=1 / CASE5_B_PACKED=1
            // force it onto every candidate (timing experiments, CASE5_*_CP / CASE5_*_TILE).
            const bool forceA = case5_a_packed_env() == 1;
            if ((candidate.pack_a || forceA)
                    && problem.A.layout == MatrixLayout::T && problem.Ta == Type::s8) {
                problem.A.layout = MatrixLayout::Pc;
                problem.A.packSize = strategy.unroll[LoopM];
                problem.A.crosspack = uint8_t(forceA ? envInt("CASE5_A_CP", 4) : candidate.pack_a_cp);
                problem.A.tileR = problem.A.tileC = 0;
                if (forceA) envTile("CASE5_A_TILE", problem.A.tileR, problem.A.tileC);
                problem.A.panelLength = 0;
                problem.A.setAlignment(problem.A.defaultAlignment(problem.Ta_ext));
            }
            const bool forceB = case5_b_packed_env() == 1;
            if ((candidate.pack_b || forceB)
                    && problem.B.layout == MatrixLayout::N && problem.Tb == Type::s8) {
                problem.B.layout = MatrixLayout::Pr;
                problem.B.packSize = strategy.unroll[LoopN];
                problem.B.crosspack = uint8_t(forceB ? envInt("CASE5_B_CP", 32) : candidate.pack_b_cp);
                problem.B.tileR = problem.B.tileC = 0;
                if (forceB) envTile("CASE5_B_TILE", problem.B.tileR, problem.B.tileC);
                problem.B.panelLength = 0;
                problem.B.setAlignment(problem.B.defaultAlignment(problem.Tb_ext));
            }
            // XMX: split the milestone fold across dpas chain boundaries. Needs one milestone per
            // k-unroll (period 1, no in-panel milestones) and a 1x4 / 1x8 sub-tile grid for the
            // single-message store; other strategies keep the generic fold.
            const bool fast_store_grid = xsd.split && xsd.subGridM == 1
                    && (xsd.subGridN == 4 || xsd.subGridN == 8);
            // Cooperative fused jackpot (CASE5_JACKPOT_COOP=0 disables): on the XMX fast-store path
            // the fold16 message stays in registers and, after one barrier, a quarter of the threads
            // hash 16 tiles per SIMD16 instruction and judge them in the kernel. It replaces the
            // per-tile SIMD1 Case 5.6 path (wrap-GRF fold) whenever the strategy qualifies.
            {
                const char *coopEnv = std::getenv("CASE5_JACKPOT_COOP");
                const char *f16 = std::getenv("CASE5_XOR_FOLD16");
                const bool coop_ok = problem.case5FuseJackpot && !(coopEnv && coopEnv[0] == '0')
                        && hw == HW::XeHPG && case5_tile_xor_interleave_enabled() && fast_store_grid
                        && problem.case5XorPeriod == 1 && problem.case5XorEveryK == 0 && !xor_nop
                        && problem.case5XorMaxMilestones % 16 == 0 && !(f16 && f16[0] == '0')
                        && strategy.wg[LoopM] > 0 && strategy.wg[LoopN] > 0 && !problem.case5TileXorSlm;
                problem.case5JackpotCoop = coop_ok;
                {
                    // Default: every thread hashes its own tiles (SIMD-N, no SLM). CASE5_COOP_OWN=0:
                    // SLM transpose, 8 tiles per hashing thread.
                    const char *ownEnv = std::getenv("CASE5_COOP_OWN");
                    problem.case5JackpotCoopSlm = coop_ok && ownEnv && ownEnv[0] == '0';
                }
                {
                    const char *auditEnv = std::getenv("CASE5_COOP_AUDIT");
                    problem.case5JackpotCoopAudit = coop_ok && auditEnv && auditEnv[0] == '1';
                }
                if (coop_ok) {
                    problem.case5TileXorWrapGrf = false;
                    problem.case5TileXorWrap = false;
                }
            }
            problem.case5TileXorInterleaved = case5_tile_xor_interleave_enabled() && hw == HW::XeHPG
                    && fast_store_grid && problem.case5XorPeriod == 1 && problem.case5XorEveryK == 0
                    && !xor_nop && !problem.case5TileXorWrap && !problem.case5TileXorWrapGrf
                    && (!problem.case5TileXorBlake3 || problem.case5JackpotCoop) && !problem.case5TileXorSlm;
            // tile_xor is tile-major ([tile][ms], index sid * num_ms + ms) on every path; the
            // jackpot kernel reads it that way.
            problem.case5XorLayout = 2;
            // Interleaved XMX fold: fold the milestones into the 16-word jackpot message in the
            // kernel (rotl 13 / xor per 16-milestone block) and write it once per tile, so tile_xor
            // and the jackpot pass move 16 words per tile instead of num_ms. CASE5_XOR_FOLD16=0
            // keeps the raw milestones.
            {
                const char *f16 = std::getenv("CASE5_XOR_FOLD16");
                problem.case5TileXorFold16 = problem.case5TileXorInterleaved
                        && problem.case5XorMaxMilestones % 16 == 0 && !(f16 && f16[0] == '0');
            }
            // Separate jackpot without the XMX interleaved fold (Gen12LP, Xe-LPG): fold each milestone into
            // the 16-word jackpot message in registers (the fused path's wrap-GRF fold: rotl 13, then xor
            // into word ms % 16) and write the 16 words per tile once at the end, instead of a global store
            // per hash tile and milestone inside the k loop. The jackpot kernel then only hashes (fold16
            // contract). In gemm_xor the same fold (Case 5.5) beats per-milestone stores (Case 5) once a
            // thread holds two or more hash tiles: UHD 770 entry 24 185.8 vs 186.4 ms, Xe-LPG 7132 vs 7096
            // GOPS. CASE5_SEPARATE_WRAP=0 keeps the per-milestone stores.
            bool wrapMsg = false;
            {
                const char *sw = std::getenv("CASE5_SEPARATE_WRAP");
                wrapMsg = !problem.case5FuseJackpot && !problem.case5TileXorInterleaved && !xor_nop
                        && !problem.case5TileXorSlm && !(sw && sw[0] == '0')
                        && problem.case5XorMaxMilestones >= 16 && problem.case5XorMaxMilestones % 16 == 0;
                if (wrapMsg) {
                    problem.case5TileXorWrap = true;
                    problem.case5TileXorWrapGrf = true;
                    problem.case5TileXorWrapGrfStoreMode = 2;          // flush at the end of the thread
                    problem.case5TileXorWrapGrfStageUnified = true;
                    problem.case5XorOutputMilestones = 16;             // word ms % 16 of the message
                }
            }

            InterfaceHandler iface(hw);
            init_case5_gemm_interface(iface, hw, problem, strategy,
                    problem.case5TileXorNop ? kKernelNameNop : kKernelName);

            const auto common_driver = Generator<hw>::driverInfo(problem, strategy);
            if (info) {
                *info = toDriverInfo(hw, device, strategy, common_driver, xsd);
                info->hwName = hw_name(hw);
                info->milestoneK = milestone_k;
                info->xorLayout = problem.case5XorLayout;
                info->fold16 = problem.case5TileXorFold16 || wrapMsg;
                info->wrapMsg = wrapMsg;
                info->jackpotCoop = problem.case5JackpotCoop;
                info->jackpotCoopAudit = problem.case5JackpotCoopAudit;
                info->jackpotCoopSlm = problem.case5JackpotCoopSlm;
                info->aPackRows = (problem.A.layout == MatrixLayout::Pc) ? int(problem.A.packSize) : 0;
                info->aPackCp = (problem.A.layout == MatrixLayout::Pc) ? int(problem.A.crosspack) : 0;
                info->bPackCols = (problem.B.layout == MatrixLayout::Pr) ? int(problem.B.packSize) : 0;
                info->bPackCp = (problem.B.layout == MatrixLayout::Pr) ? int(problem.B.crosspack) : 0;
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
            // CASE5_DUMP_BINARY=<file>: save the device binary (for `ocloc disasm`).
            if (const char *dump = std::getenv("CASE5_DUMP_BINARY")) {
                if (dump[0]) {
                    auto bin = gen.getBinary(ctx, device);
                    if (FILE *f = std::fopen(dump, "wb")) {
                        std::fwrite(bin.data(), 1, bin.size(), f);
                        std::fclose(f);
                    }
                }
            }
            cl_kernel k = gen.getKernel(ctx, device);
            if (!k) {
                throw std::runtime_error("getKernel returned null");
            }
            return k;
        } catch (const std::exception &e) {
            last_err = candidate.label + ": " + e.what();
            if (candidate.from_catalog) {
                log_candidate_rejection(candidate, e.what(), &rejected);
            } else if (case5_debug_select()) {
                std::fprintf(stderr, "Case5 rejected %s: %s\n", candidate.label.c_str(), e.what());
            }
        }
    }

    if (err) {
        *err = last_err.empty() ? "no Case5 strategy candidates succeeded" : last_err;
    }
    return nullptr;
}
} // namespace

void set_hash_tile_filter(HashTileFilter filter) { g_hash_tile_filter = filter; }

bool is_supported_device(cl_context ctx, cl_device_id device, std::string *err) {
    const Product product = detect_product(ctx, device);
    if (is_supported_product(product)) {
        return true;
    }
    if (err) {
        *err = std::string("Case 5 requires XeLP/Gen12LP or XeHPG (Core Ultra MTL/ARL). Detected ")
               + product_family_name(product.family)
               + " - unsupported ISA for Case 5.";
    }
    return false;
}

cl_kernel build_igemm_kernel(cl_context ctx, cl_device_id device, const BuildParams *dims,
                             DriverInfo *info, std::string *err, bool xor_nop,
                             bool fused_jackpot, bool a_row_major, bool b_row_major) {
    BuildParams build_dims;
    if (dims) {
        build_dims = *dims;
    }
    // CASE5_XOR_NOP=1: keep the milestone schedule but skip the C fold. Diagnostic
    // ceiling only -- the resulting tile hashes are wrong and shares will not verify.
    if (const char *nop = std::getenv("CASE5_XOR_NOP"); nop && nop[0] == '1') {
        xor_nop = true;
    }
    build_dims.a_row_major = a_row_major;
    build_dims.b_row_major = b_row_major;
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
