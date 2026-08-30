#include "case5_gemm_launch.hpp"

#include <algorithm>
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

uint32_t uint32_reciprocal(uint32_t x) {
    if (x == 0) {
        return 0;
    }
    int lg = 0;
    while ((uint64_t{1} << lg) < x) {
        ++lg;
    }
    return div_up_u64(uint64_t{0x100000000ULL} << lg, x);
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
    bool has_kv_config = false;
    uint32_t kv_config = 0;
    bool has_k_recip = false;
    uint32_t k_recip = 0;
    bool has_group_stride = false;
    uint32_t group_stride = 0;
};

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
        const double sm = std::sqrt(double(concurrent_tg) / bias);
        const double sn = std::sqrt(double(concurrent_tg) * bias);

        int32_t slice = 0;
        int32_t thresh = 0;
        bool ok = false;

        for (bool nslice : {groups_m > groups_n, groups_m <= groups_n}) {
            double s = nslice ? sn : sm;
            auto sf = int(std::floor(s));
            auto sc = int(std::ceil(s));
            if (concurrent_tg % sc == 0) {
                s = sf = sc;
            }
            if (concurrent_tg % (sc + 1) == 0) {
                s = sf = sc = sc + 1;
            }

            const int gc = nslice ? int(groups_n) : int(groups_m);
            const int gco = nslice ? int(groups_m) : int(groups_n);

            for (int srange = 0; srange <= 2 && !ok; ++srange) {
                int s0 = (srange < 2) ? sc : sf;
                const bool up = (srange == 1);
                const int s1 = s0 + (up ? 1 : -1);
                if (s1 <= 0) {
                    continue;
                }

                const auto rem = gc % s0;
                if (!rem || up) {
                    thresh = gc / s0 - rem;
                } else {
                    thresh = div_up(gc, s0) - (s0 - rem);
                }

                ok = (thresh >= 0) && (gco >= 2 * std::max(s0, s1));
                slice = s0;
                if (!up) {
                    if (thresh > 0) {
                        thresh = -thresh;
                    } else {
                        slice--;
                        thresh = gc;
                    }
                }
                if (nslice) {
                    slice *= -1;
                }
            }

            if (ok) {
                break;
            }
        }

        if (!ok) {
            const bool nslice = (groups_m > groups_n);
            const double s = nslice ? sn : sm;
            const int gc = nslice ? int(groups_n) : int(groups_m);

            if (gc < s * 1.5) {
                slice = gc;
            } else {
                slice = gc / div_up(gc, int(std::round(s)));
            }

            thresh = std::max(0, (gc / slice) - (gc % slice));
            if (nslice) {
                slice *= -1;
            }
        }

        if (slice == 0) {
            slice = 1;
            thresh = int(groups_m);
        }

        out.has_bslice = true;
        out.bslice = slice;
        out.has_bthresh = true;
        out.bthresh = thresh;
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

    // Fused Blake3: fold stays in wrap-GRF; only blake3_out is a global result buffer.
    if (!problem.case5TileXorBlake3) {
        iface.newArgument("tile_xor", ExternalArgumentType::GlobalPtr, c_access);
    }
    iface.newArgument("tile_count", DataType::d);
    iface.newArgument("tile_cols", DataType::d);
    iface.newArgument("xor_period", DataType::d);
    if (problem.case5TileXorBlake3) {
        iface.newArgument("blake3_out", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
        for (int i = 0; i < 8; i++) {
            iface.newArgument("blake3_k" + std::to_string(i), DataType::ud);
        }
    }

    if (hw >= HW::XeHPG) {
        iface.allowArgumentRearrangement(false);
    }
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
                "gcm=%u gcn=%u bslice=%d bthresh=%d\n",
                wg_gws[0], wg_gws[1], wg_lws[0], wg_lws[1], walk.group_count_m, walk.group_count_n,
                walk.bslice, walk.bthresh);
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

    if (!problem.case5TileXorBlake3) {
        err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.tile_xor);
    }
    err |= set_arg(kernel, arg, sizeof(int), &bufs.tile_count);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.tile_cols);
    err |= set_arg(kernel, arg, sizeof(int), &bufs.xor_period);
    if (problem.case5TileXorBlake3) {
        err |= set_arg(kernel, arg, sizeof(cl_mem), &bufs.blake3_out);
        for (int i = 0; i < 8; i++) {
            err |= set_arg(kernel, arg, sizeof(uint32_t), &bufs.blake3_key_words[i]);
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
