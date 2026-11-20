#include "case5_gemm_launch.hpp"
#include "case5_walk.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace case5_ngen {
namespace {

using namespace ngen;
using namespace gemmstone;

int div_up(int a, int b) { return (a + b - 1) / b; }

int rnd_up(int a, int b) { return div_up(a, b) * b; }

uint32_t div_up_u64(uint64_t a, uint32_t b) {
    return static_cast<uint32_t>((a + b - 1) / b);
}

// Reciprocal for gemmstone's runtime divDown(src, divisor, recip): the kernel computes
// (src * recip) >> 32 >> floor(log2 divisor), so recip = ceil(2^(32 + floor(log2 x)) / x) (gemmstone's own
// constant divDown and oneDNN use bsr(x)). The previous ceil(log2 x) shift gave a wrong, 33-bit value for
// every divisor that is not a power of two; the Hilbert walk divides by groupCountM * vd, the linear walk by
// groupCountM/N, so on group grids where that is not a power of two (not the 64x64 grid of 16384^2 with
// 32x64 tiles, but e.g. 3x8 or 6x16) tiles were skipped or computed twice and some group ids decoded
// outside the grid (stray tile_xor stores, GPU faults). Same fix as hexagon_test/gemm_xor/onednn.
uint32_t uint32_reciprocal(uint32_t x) {
    if (x == 0) {
        return 0;
    }
    int lg = 0;   // floor(log2 x)
    while ((uint64_t{2} << lg) <= x) {
        ++lg;
    }
    return static_cast<uint32_t>(div_up_u64(uint64_t{0x100000000ULL} << lg, x));
}

int max_eus_per_wg(HW hw) {
    switch (hw) {
    case HW::Gen12LP:
    case HW::XeHP:
    case HW::XeHPG:
        return 16;
    default:
        return 8;
    }
}

int grf_per_eu(HW hw) {
    switch (hw) {
    case HW::XeLP:
        return 896;
    case HW::XeHP:
    case HW::XeHPG:
    case HW::XeHPC:
    case HW::Xe2:
    case HW::Xe3:
    case HW::Xe3p:
        return 1024;
    default:
        return 0;
    }
}

int threads_per_eu(HW hw, int grfs) {
    if (hw <= HW::XeLP) {
        return 7;
    }
    return grf_per_eu(hw) / grfs;
}

bool case5_debug_launch() {
    static const bool on = [] {
        const char *v = std::getenv("CASE5_DEBUG_LAUNCH");
        return v && v[0] != '0';
    }();
    return on;
}

cl_int set_arg(cl_kernel kernel, int &arg, size_t size, const void *value) {
    const cl_int err = clSetKernelArg(kernel, arg++, size, value);
    return err;
}

struct WalkOrderArgs {
    bool has_k0 = false;
    uint32_t k0 = 0;
    bool has_group_count_k = false;
    uint32_t group_count_k = 0;
    bool has_group_count_m = false;
    uint32_t group_count_m = 0;
    bool has_group_count_n = false;
    uint32_t group_count_n = 0;
    bool has_group_count_recip = false;
    uint32_t group_count_recip = 0;
    bool has_hilbert_vd = false;
    uint32_t hilbert_vd = 0;
    bool has_hilbert_uvd_recip = false;
    uint32_t hilbert_uvd_recip = 0;
    bool has_hilbert_bail = false;
    uint32_t hilbert_bail = 0;
    bool has_bslice = false;
    int32_t bslice = 0;
    bool has_bthresh = false;
    int32_t bthresh = 0;
    const char *bsource = "";   // where the boustrophedon pair came from (CASE5_DEBUG_LAUNCH)
    bool has_kv_config = false;
    uint32_t kv_config = 0;
    bool has_k_recip = false;
    uint32_t k_recip = 0;
    bool has_group_stride = false;
    uint32_t group_stride = 0;
};

// ---- Boustrophedon walk (arithmetic in case5_walk.hpp) --------------------------------------------

using case5_walk::BoustrophedonPair;
using case5_walk::derive_boustrophedon;

// Scale of the m-slice target (same as gemm_xor's case5_gemm_launch.cpp). Taller slices save power on the
// XMX cards and the power cap turns that into clock (gemm_xor 2026-11-19, default limits, constant-64 u8,
// sustained): A770 at 65536^2 slices of 10 instead of 6 +1.8% Case 5.5 / +1.7% Case 5.6, B580 at 16384^2
// slices of 8 instead of Hilbert +1.0% / +1.2%. 1.75 gives 10 on the A770 and 8 on the B580. Only the m-slice target is
// scaled: grids sliced along n (many more m-groups than n-groups) keep the gemmstone default, nothing was
// measured there. CASE5_BSLICE_SCALE=<x> overrides it (1 = the gemmstone default); dp4a devices keep 1.
double case5_bslice_scale(HW hw, const GEMMStrategy &strategy) {
    if (const char *v = std::getenv("CASE5_BSLICE_SCALE")) {
        const double x = std::atof(v);
        if (x > 0) return x;
    }
    return (strategy.systolic && (hw == HW::XeHPG || hw == HW::Xe2)) ? 1.75 : 1.0;
}

// case5_walk::boustrophedon_map_ok, cached per grid and pair (a 256 x 512 grid takes about a millisecond;
// launches repeat the same few).
bool boustrophedon_pair_ok(uint32_t groups_m, uint32_t groups_n, const BoustrophedonPair &p) {
    static std::map<std::tuple<uint32_t, uint32_t, int32_t, int32_t>, bool> cache;
    const auto key = std::make_tuple(groups_m, groups_n, p.slice, p.thresh);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const bool ok = case5_walk::boustrophedon_map_ok(groups_m, groups_n, p);
    cache.emplace(key, ok);
    return ok;
}

void warn_bad_boustrophedon_pair(uint32_t groups_m, uint32_t groups_n, const BoustrophedonPair &p,
        const char *what) {
    static std::map<std::tuple<uint32_t, uint32_t, int32_t, int32_t>, bool> warned;
    if (!warned.emplace(std::make_tuple(groups_m, groups_n, p.slice, p.thresh), true).second) return;
    std::fprintf(stderr,
            "Case5 walk: %s pair bslice=%d bthresh=%d does not map the %ux%u group grid one to one "
            "(groups repeated or outside the grid); not used\n",
            what, p.slice, p.thresh, groups_m, groups_n);
    std::fflush(stderr);
}

WalkOrderArgs compute_walk_order_args(const GEMMStrategy &strategy, const CommonDriverInfo &info,
        int eu_count, HW hw, int m, int n, int k, const size_t gws[2], const size_t lws[2]) {
    WalkOrderArgs out;
    const int grf_count = strategy.GRFs > 0 ? strategy.GRFs : info.grfCount;

    if (strategy.kParallel && strategy.fuseBeta) {
        if (gws[2] > 0 && lws[2] > 0) {
            out.has_group_count_k = true;
            out.group_count_k = static_cast<uint32_t>(gws[2] / lws[2]);
        }
    }

    if (!info.isLinearOrder()) {
        return out;
    }

    const int m_index = info.isNMK() ? 1 : 0;
    const int n_index = info.isNMK() ? 0 : 1;
    const uint32_t groups_m = static_cast<uint32_t>(gws[m_index] / lws[m_index]);
    const uint32_t groups_n = static_cast<uint32_t>(gws[n_index] / lws[n_index]);
    uint32_t group_count = groups_m * groups_n;

    const uint32_t ss_count = static_cast<uint32_t>(eu_count / max_eus_per_wg(hw));
    const uint32_t thread_per_ss = static_cast<uint32_t>(
            (eu_count * threads_per_eu(hw, grf_count)) / std::max(1, static_cast<int>(ss_count)));
    const uint32_t thread_per_tg = static_cast<uint32_t>(lws[0] * lws[1]);
    const uint32_t tg_per_ss = thread_per_tg > 0 ? thread_per_ss / thread_per_tg : 1;
    const uint32_t concurrent_tg = tg_per_ss * std::max<uint32_t>(1, ss_count);

    out.has_group_count_m = true;
    out.group_count_m = groups_m;
    out.has_group_count_n = true;
    out.group_count_n = groups_n;

    if (info.isSimpleLinear()) {
        out.has_group_count_recip = true;
        out.group_count_recip = uint32_reciprocal(info.isMNK() ? groups_m : groups_n);
    } else if (info.isHilbert()) {
        uint32_t vd = 0;
        uint32_t uvd = 0;
        const double ratio = groups_n > 0 ? double(groups_n) / double(groups_m) : 1.0;
        if (ratio >= 1) {
            vd = static_cast<uint32_t>(std::ceil(groups_n / std::round(2 * ratio)));
            uvd = groups_m * vd;
        } else {
            vd = static_cast<uint32_t>(std::ceil(groups_m / std::round(2 / ratio)));
            uvd = groups_n * vd;
            vd |= 0xFFFF0000u;
        }
        out.has_hilbert_vd = true;
        out.hilbert_vd = vd;
        out.has_hilbert_uvd_recip = true;
        out.hilbert_uvd_recip = uint32_reciprocal(uvd);
        out.has_hilbert_bail = true;
        out.hilbert_bail = 1;
    } else if (info.isBoustrophedon()) {
        const double bias = double(info.wg[0] * info.unroll[0]) / double(info.wg[1] * info.unroll[1]);
        const double scale = case5_bslice_scale(hw, strategy);

        // Candidates in order of preference: the CASE5_BSLICE / CASE5_BTHRESH override (tuning), the
        // default with the scaled m-slice target, the unscaled default (gemmstone / oneDNN). The first
        // pair that maps every work group to its own tile is launched; the decode is replayed on the host
        // (boustrophedon_pair_ok) because a bad pair writes stray tiles and can fault the GPU.
        BoustrophedonPair scaled = derive_boustrophedon(groups_m, groups_n, concurrent_tg, bias, scale);
        const BoustrophedonPair plain = derive_boustrophedon(groups_m, groups_n, concurrent_tg, bias, 1.0);
        BoustrophedonPair chosen = scaled;
        const char *source = (scaled.slice != plain.slice || scaled.thresh != plain.thresh) ? "scaled default"
                                                                                             : "default";

        // CASE5_BSLICE / CASE5_BTHRESH: override the slice / threshold (tuning). CASE5_BSLICE replaces the
        // slice only and keeps the threshold derived for the default slice, the way the 2026-11 tuning runs
        // used it; on grids sliced along n (many more m-groups than n-groups) a positive slice with that
        // threshold maps groups twice, and the replay below rejects it.
        BoustrophedonPair env = scaled;
        bool has_env = false;
        if (const char *bs = std::getenv("CASE5_BSLICE")) {
            if (bs[0]) { env.slice = std::atoi(bs); has_env = true; }
        }
        if (const char *bt = std::getenv("CASE5_BTHRESH")) {
            if (bt[0]) { env.thresh = std::atoi(bt); has_env = true; }
        }

        if (has_env && boustrophedon_pair_ok(groups_m, groups_n, env)) {
            chosen = env;
            source = "CASE5_BSLICE/CASE5_BTHRESH";
        } else {
            if (has_env) warn_bad_boustrophedon_pair(groups_m, groups_n, env, "CASE5_BSLICE/CASE5_BTHRESH");
            if (!boustrophedon_pair_ok(groups_m, groups_n, scaled)) {
                warn_bad_boustrophedon_pair(groups_m, groups_n, scaled, "scaled default");
                if (!boustrophedon_pair_ok(groups_m, groups_n, plain)) {
                    char msg[160];
                    std::snprintf(msg, sizeof(msg),
                            "Case5 walk: no valid boustrophedon pair for the %ux%u group grid (default %d,%d)",
                            groups_m, groups_n, plain.slice, plain.thresh);
                    throw std::runtime_error(msg);
                }
                chosen = plain;
                source = "default";
            }
        }

        out.has_bslice = true;
        out.bslice = chosen.slice;
        out.has_bthresh = true;
        out.bthresh = chosen.thresh;
        out.bsource = source;
    }

    if (info.kParallelVariable()) {
        uint32_t k_parallel_start = (group_count / concurrent_tg) * concurrent_tg;
        if (k_parallel_start > 0 && k_parallel_start != group_count) {
            k_parallel_start -= concurrent_tg;
        }
        const uint32_t k_sliced_tiles = group_count - k_parallel_start;
        const uint32_t k_sliced_phases = 1;
        const uint32_t tiles_per_phase = div_up(k_sliced_tiles, k_sliced_phases);

        int k_padding = info.kPadding();
        int old_k_padding = k_padding;
        int k_padded = k;
        int64_t k_total = 0;
        uint32_t k0 = static_cast<uint32_t>(k);

        do {
            k_padded = rnd_up(k + k_padding, info.unroll[LoopK]);
            k_total = int64_t(k_padded) * tiles_per_phase;
            if (k_total == 0) {
                break;
            }

            k0 = static_cast<uint32_t>(div_up(k_total, concurrent_tg));
            k0 = static_cast<uint32_t>(rnd_up(int(k0), info.unroll[LoopK]));

            old_k_padding = k_padding;
            k_padding = std::min(k_padding, 2 * int(k0));
        } while (k_padding != old_k_padding);

        group_count = k_parallel_start;
        uint32_t k_parallel_groups = 0;
        uint32_t k_sync_slabs = 0;

        if (k0 > 0) {
            k_parallel_groups = static_cast<uint32_t>(div_up(k_total, k0));
            if (k_sliced_phases > 1) {
                k_parallel_groups = concurrent_tg;
            }
            group_count += k_parallel_groups * k_sliced_phases;

            if (tiles_per_phase > 0) {
                k_sync_slabs = (k_parallel_groups + (tiles_per_phase >> 1)) / tiles_per_phase;
                if (k_sync_slabs > 0) {
                    k_sync_slabs--;
                }
                k_sync_slabs = std::min(k_sync_slabs, uint32_t((k_padded - 1) / int(k0)));
            }
        }

        const uint32_t k_unsynced_padded = static_cast<uint32_t>(k_padded) - k_sync_slabs * k0;
        const uint32_t k_recip = uint32_reciprocal(k_unsynced_padded);
        uint32_t kv_config = k_sliced_tiles | (k_sync_slabs << 16);
        if (k_sliced_phases > 1) {
            kv_config |= 0x80000000u;
        }

        out.has_k0 = true;
        out.k0 = k0;
        out.has_kv_config = true;
        out.kv_config = kv_config;
        out.has_k_recip = true;
        out.k_recip = k_recip;
    }

    if (info.isPersistent()) {
        group_count = std::min(group_count, concurrent_tg);
        // CASE5_PERSISTENT_GROUPS=<n>: override the number of persistent work groups.
        if (const char *pg = std::getenv("CASE5_PERSISTENT_GROUPS")) {
            const int n = std::atoi(pg);
            if (n > 0) group_count = static_cast<uint32_t>(n);
        }
        out.has_group_stride = true;
        out.group_stride = group_count;
    }

    return out;
}

} // namespace

void init_case5_gemm_interface(InterfaceHandler &iface, HW hw, const GEMMProblem &problem,
        const GEMMStrategy &strategy, const char *kernel_name) {
    const auto s_type = problem.Ts.ngen();
    const auto a_access = strategy.A.getGlobalAccessType();
    const auto b_access = strategy.B.getGlobalAccessType();
    const auto c_access = strategy.C.getGlobalAccessType();
    const auto ao_access = strategy.AO.getGlobalAccessType();
    const auto bo_access = strategy.BO.getGlobalAccessType();
    const auto co_access = strategy.CO.getGlobalAccessType();
    const auto as_access = strategy.A_scale.getGlobalAccessType();
    const auto bs_access = strategy.B_scale.getGlobalAccessType();
    const auto ag_access = strategy.Ag.getGlobalAccessType();
    const auto bg_access = strategy.Bg.getGlobalAccessType();

    iface.newArgument("A", ExternalArgumentType::GlobalPtr, a_access);
    iface.newArgument("B", ExternalArgumentType::GlobalPtr, b_access);
    iface.newArgument("C", ExternalArgumentType::GlobalPtr, c_access);
    iface.newArgument("offset_A", DataType::q);
    iface.newArgument("offset_B", DataType::q);
    iface.newArgument("offset_C", DataType::q);
    iface.newArgument("lda", DataType::d);
    iface.newArgument("ldb", DataType::d);
    iface.newArgument("ldc", DataType::d);
    iface.newArgument("m", DataType::d);
    iface.newArgument("n", DataType::d);
    iface.newArgument("k", DataType::d);
    iface.newArgument("alpha_real", s_type);
    iface.newArgument("beta_real", s_type);

    if (problem.aoPtrDims >= 0) {
        iface.newArgument("ao_ptr", ExternalArgumentType::GlobalPtr, ao_access);
    }
    if (problem.boPtrDims >= 0) {
        iface.newArgument("bo_ptr", ExternalArgumentType::GlobalPtr, bo_access);
    }
    if (problem.aOffsetHostScalar()) {
        iface.newArgument("ao", DataType::w);
    }
    if (problem.bOffsetHostScalar()) {
        iface.newArgument("bo", DataType::w);
    }
    if (problem.aScale2D()) {
        iface.newArgument("a_scale_ptr", ExternalArgumentType::GlobalPtr, as_access);
    }
    if (problem.bScale2D()) {
        iface.newArgument("b_scale_ptr", ExternalArgumentType::GlobalPtr, bs_access);
    }
    if (problem.hasCMXScale()) {
        iface.newArgument("c_scale_ptr", ExternalArgumentType::GlobalPtr, c_access);
    }
    if (problem.needsAGroupSums()) {
        iface.newArgument("ag_ptr", ExternalArgumentType::GlobalPtr, ag_access);
    }
    if (problem.needsBGroupSums()) {
        iface.newArgument("bg_ptr", ExternalArgumentType::GlobalPtr, bg_access);
    }
    if (problem.aOffset2D() || problem.aScale2D() || problem.needsAGroupSums()) {
        iface.newArgument("ldaq", DataType::d);
    }
    if (problem.bOffset2D() || problem.bScale2D() || problem.needsBGroupSums()) {
        iface.newArgument("ldbq", DataType::d);
    }
    if (problem.hasCMXScale()) {
        iface.newArgument("ldcq", DataType::d);
    }
    if (problem.usesCOPtr()) {
        iface.newArgument("co_ptr", ExternalArgumentType::GlobalPtr, co_access);
        iface.newArgument("offset_CO", DataType::q);
        if (problem.cOffset == COffset::Pre) {
            iface.newArgument("ldco", DataType::d);
        }
    } else if (problem.cOffsetHostScalar()) {
        iface.newArgument("co", DataType::w);
    }
    if (problem.postOps.cStochasticRound) {
        iface.newArgument("sround_seed", ExternalArgumentType::GlobalPtr);
    }

    if (strategy.needsTempC(problem)) {
        iface.newArgument("temp_C", ExternalArgumentType::GlobalPtr, c_access);
    }
    iface.newArgument("flags", DataType::ud);
    if ((strategy.kParallel || strategy.kParallelLocal) && !strategy.kParallelVariable) {
        iface.newArgument("k0", DataType::d);
    }

    if (strategy.fuseBeta || strategy.fusePostOps) {
        iface.newArgument("status", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
    }
    if (strategy.fuseBeta && strategy.kParallel) {
        iface.newArgument("group_count_k", DataType::ud);
    }
    if (strategy.linearOrder()) {
        iface.newArgument("group_count_m", DataType::ud);
        iface.newArgument("group_count_n", DataType::ud);
    }
    if (strategy.cWalkOrder == WalkOrder::SimpleLinear) {
        iface.newArgument("group_count_recip", DataType::ud);
    } else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        iface.newArgument("hilbert_vd", DataType::ud);
        iface.newArgument("hilbert_uvd_recip", DataType::ud);
        iface.newArgument("hilbert_bail", DataType::ud);
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        iface.newArgument("bslice", DataType::d);
        iface.newArgument("bthresh", DataType::d);
    }
    if (strategy.kParallelVariable) {
        iface.newArgument("k0", DataType::ud);
        iface.newArgument("kv_config", DataType::ud);
        iface.newArgument("k_recip", DataType::ud);
    }
    if (strategy.persistent) {
        iface.newArgument("group_stride", DataType::ud);
    }
    if (strategy.variableSLM()) {
        iface.newArgument("local_mem", ExternalArgumentType::LocalPtr);
    }
    if (problem.aoPtrDims >= 1 || problem.aScale2D()) {
        iface.newArgument("offset_Aq", DataType::q);
    }
    if (problem.boPtrDims >= 1 || problem.bScale2D()) {
        iface.newArgument("offset_Bq", DataType::q);
    }

    // Fused Blake3: fold in wrap-GRF; in-kernel judge writes found + per-tile beats.
    // The cooperative fused jackpot keeps the (unused) tile_xor argument: the interleaved
    // fast-store path it builds on allocates against it (the host binds a null buffer).
    if (!problem.case5TileXorBlake3 || problem.case5JackpotCoop) {
        iface.newArgument("tile_xor", ExternalArgumentType::GlobalPtr, c_access);
    }
    iface.newArgument("tile_count", DataType::d);
    iface.newArgument("tile_cols", DataType::d);
    iface.newArgument("xor_period", DataType::d);
    if (problem.case5TileXorBlake3) {
        if (!problem.case5FuseJackpot) {
            iface.newArgument("blake3_out", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
        }
        for (int i = 0; i < 8; i++) {
            iface.newArgument("blake3_k" + std::to_string(i), DataType::ud);
        }
        if (problem.case5FuseJackpot) {
            iface.newArgument("found_flag", ExternalArgumentType::GlobalPtr,
                              GlobalAccessType::Stateless);
            iface.newArgument("tr_base", DataType::ud);
            iface.newArgument("tc_base", DataType::ud);
            for (int i = 0; i < 8; i++) {
                iface.newArgument("blake3_b" + std::to_string(i), DataType::ud);
            }
        }
    }

    // Host bind_case5_kernel_args sets args in declaration order. Rearrangement on
    // Gen12LP scrambled blake3_bound/found_flag and caused false jackpot claims.
    iface.allowArgumentRearrangement(false);
    iface.externalName(kernel_name);
}

LaunchDims compute_case5_launch_dims(const DriverInfo &info, int m, int n) {
    LaunchDims dims;
    dims.gws[0] = static_cast<size_t>(div_up(m, info.unrollM));
    dims.gws[1] = static_cast<size_t>(div_up(n, info.unrollN));
    dims.lws[0] = static_cast<size_t>(info.wgM);
    dims.lws[1] = static_cast<size_t>(info.wgN);

    if (info.isNMK) {
        std::swap(dims.gws[0], dims.gws[1]);
        std::swap(dims.lws[0], dims.lws[1]);
    }

    if (info.fusedEUs && dims.lws[0] > 1) {
        dims.gws[0] = static_cast<size_t>(rnd_up(static_cast<int>(dims.gws[0]), 2));
    }

    for (int d = 0; d < 2; ++d) {
        if (info.fixedWG || dims.gws[d] > dims.lws[d]) {
            dims.gws[d] =
                    static_cast<size_t>(rnd_up(static_cast<int>(dims.gws[d]), static_cast<int>(dims.lws[d])));
        } else {
            dims.lws[d] = dims.gws[d];
        }
    }

    dims.lws[1] *= static_cast<size_t>(std::max(1, info.wgExpand));
    dims.gws[1] *= static_cast<size_t>(std::max(1, info.wgExpand));

    dims.lws[0] *= static_cast<size_t>(info.subgroupSize);
    dims.gws[0] *= static_cast<size_t>(info.subgroupSize);

    return dims;
}

void case5_launch_dims_for_walk_order(const LaunchDims &dims, int subgroup_size,
        size_t wg_gws[2], size_t wg_lws[2]) {
    const size_t sg = static_cast<size_t>(std::max(1, subgroup_size));
    wg_gws[0] = dims.gws[0] / sg;
    wg_gws[1] = dims.gws[1];
    wg_lws[0] = dims.lws[0] / sg;
    wg_lws[1] = dims.lws[1];
}

void apply_linear_order_launch_dims(const DriverInfo &info, LaunchDims &dims, int m, int n, int k) {
    if (!info.commonDriver.isLinearOrder()) {
        return;
    }

    size_t wg_gws[2] = {};
    size_t wg_lws[2] = {};
    case5_launch_dims_for_walk_order(dims, info.subgroupSize, wg_gws, wg_lws);

    const WalkOrderArgs walk = compute_walk_order_args(
            info.strategy, info.commonDriver, info.euCount, info.hw, m, n, k, wg_gws, wg_lws);

    uint32_t group_count = walk.group_count_m * walk.group_count_n;
    if (walk.has_group_stride) {
        group_count = walk.group_stride;
    }

    const size_t extra = static_cast<size_t>(info.commonDriver.extraWGs());
    dims.gws[0] = dims.lws[0] * (static_cast<size_t>(group_count) + extra);
    dims.gws[1] = dims.lws[1];
}

cl_int bind_case5_kernel_args(cl_kernel kernel, const DriverInfo &info,
        const GEMMProblem &problem, const LaunchBuffers &bufs, const LaunchDims &dims,
        int *out_arg_count) {
    size_t wg_gws[2] = {};
    size_t wg_lws[2] = {};
    case5_launch_dims_for_walk_order(dims, info.subgroupSize, wg_gws, wg_lws);

    const WalkOrderArgs walk = compute_walk_order_args(
            info.strategy, info.commonDriver, info.euCount, info.hw, bufs.m, bufs.n, bufs.k,
            wg_gws, wg_lws);

    if (case5_debug_launch()) {
        std::fprintf(stderr,
                "Case5 launch walk-order: wg_gws=(%zu,%zu) wg_lws=(%zu,%zu) "
                "gcm=%u gcn=%u bslice=%d bthresh=%d%s%s\n",
                wg_gws[0], wg_gws[1], wg_lws[0], wg_lws[1], walk.group_count_m, walk.group_count_n,
                walk.bslice, walk.bthresh, walk.bsource[0] ? " from " : "", walk.bsource);
        std::fflush(stderr);
    }

    int arg = 0;
    cl_int err = CL_SUCCESS;
    err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.a);
    err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.b);
    err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.c);
    err |= set_arg(kernel, arg, sizeof(int64_t), &bufs.offset_a);
    err |= set_arg(kernel, arg, sizeof(int64_t), &bufs.offset_b);
    err |= set_arg(kernel, arg, sizeof(int64_t), &bufs.offset_c);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.lda);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.ldb);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.ldc);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.m);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.n);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.k);
    err |= set_arg(kernel, arg, sizeof(float), &bufs.alpha);
    err |= set_arg(kernel, arg, sizeof(float), &bufs.beta);

    if (info.strategy.needsTempC(problem)) {
        cl_mem temp_c = bufs.c;
        err |= set_arg(kernel, arg, sizeof(cl_mem), &temp_c);
    }
    err |= set_arg(kernel, arg, sizeof(uint32_t), &bufs.flags);

    if ((info.strategy.kParallel || info.strategy.kParallelLocal) && !info.strategy.kParallelVariable) {
        const int32_t k0 = 0;
        err |= set_arg(kernel, arg, sizeof(int32_t), &k0);
    }
    if (info.strategy.fuseBeta || info.strategy.fusePostOps) {
        cl_mem status = nullptr;
        err |= set_arg(kernel, arg, sizeof(cl_mem), &status);
    }
    if (info.strategy.fuseBeta && info.strategy.kParallel && walk.has_group_count_k) {
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.group_count_k);
    }
    if (info.strategy.linearOrder()) {
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.group_count_m);
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.group_count_n);
    }
    if (info.strategy.cWalkOrder == WalkOrder::SimpleLinear && walk.has_group_count_recip) {
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.group_count_recip);
    } else if (info.strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.hilbert_vd);
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.hilbert_uvd_recip);
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.hilbert_bail);
    } else if (info.strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        err |= set_arg(kernel, arg, sizeof(int32_t), &walk.bslice);
        err |= set_arg(kernel, arg, sizeof(int32_t), &walk.bthresh);
    }
    if (info.strategy.kParallelVariable) {
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.k0);
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.kv_config);
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.k_recip);
    }
    if (info.strategy.persistent && walk.has_group_stride) {
        err |= set_arg(kernel, arg, sizeof(uint32_t), &walk.group_stride);
    }
    if (info.strategy.variableSLM()) {
        err |= CL_INVALID_ARG_VALUE;
    }

    if (!problem.case5TileXorBlake3 || problem.case5JackpotCoop) {
        err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.tile_xor);
    }
    err |= set_arg(kernel, arg, sizeof(int), &bufs.tile_count);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.tile_cols);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.xor_period);
    if (problem.case5TileXorBlake3) {
        if (!problem.case5FuseJackpot) {
            err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.blake3_out);
        }
        for (int i = 0; i < 8; i++) {
            err |= set_arg(kernel, arg, sizeof(uint32_t), &bufs.blake3_key_words[i]);
        }
        if (problem.case5FuseJackpot) {
            err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.found_flag);
            err |= set_arg(kernel, arg, sizeof(int), &bufs.tr_base);
            err |= set_arg(kernel, arg, sizeof(int), &bufs.tc_base);
            for (int i = 0; i < 8; i++) {
                err |= set_arg(kernel, arg, sizeof(uint32_t), &bufs.blake3_bound_words[i]);
            }
        }
    }

    if (out_arg_count) {
        *out_arg_count = arg;
    }

    if (err == CL_SUCCESS) {
        cl_uint kernel_args = 0;
        if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_args), &kernel_args, nullptr)
                == CL_SUCCESS) {
            if (kernel_args != static_cast<cl_uint>(arg)) {
                std::fprintf(stderr,
                        "Case5 kernel arg mismatch: bound %d, kernel expects %u (strategy %s)\n",
                        arg, kernel_args, info.strategyName);
                std::fflush(stderr);
                return CL_INVALID_ARG_VALUE;
            }
        }
    }

    return err;
}

} // namespace case5_ngen
