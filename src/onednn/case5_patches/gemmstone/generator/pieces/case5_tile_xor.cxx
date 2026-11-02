/*******************************************************************************
 * Case5 milestoned tile XOR: once per case5XorPeriod unrollK panels
 * (Case 3.4 KR=128 → period=4 when unrollK=32), XOR-reduce live C_regs to one
 * uint32 per logical XOR sub-tile and store to tile_xor[ms * tile_count + logical_id].
 * Each thread folds its full unrollM x unrollN panel; if subGrid > 1, halve to
 * subM x subN pieces (see CASE5_XOR_TILE.md).
 *
 * The K-loop invokes store every unrollK panel; fold+store run only when
 * panel % xorPeriod == 0. Sequencer period must stay unrollK (ls.getUnroll() == unrollK).
 ******************************************************************************/

#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "gemmstone/generator.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

GEMMSTONE_NAMESPACE_START

using namespace ngen;

size_t case5TileXorSlmByteOffset(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy,
                                 bool computeMax);

constexpr int kFoldTreeFanin = 8;

// Interleaved-fold instruction recording (CASE5_XOR_RR, default on): case5FoldSubtileRows pushes its
// ops here instead of emitting them, and case5FoldEmitPending emits them round-robin over the
// sub-tile chains, so consecutive instructions target different accumulators (a single chain is
// 8 dependent bfn, each waiting on the previous one: {@1} on every instruction).
// kind: 0 mov dst<-a, 1 xor dst<-a^b, 2 bfn dst<-dst^a^b, 3 xor dst<-dst^a, 4 mov dst<-0.
namespace {
struct Case5FoldOp { int kind, exec, chain; ngen::RegData dst, a, b; };
bool g_c5FoldRecord = false;
int g_c5FoldChain = 0, g_c5FoldChainNext = 0;
std::vector<Case5FoldOp> g_c5FoldOps;
bool case5EnvOn(const char *name, bool def) { const char *v = std::getenv(name); return v ? v[0] != '0' : def; }
}

template <HW hw>
void Generator<hw>::case5ZeroFoldGrfs(const GEMMStrategy &strategy, GEMMState &state, int nFolds) {
    const int simd = elementsPerGRF(hw, DataType::ud);
    if (nFolds <= 0)
        return;
    if (nFolds == 1 && state.tileXorFold.isValid()) {
        mov(simd, state.tileXorFold.ud(), uint16_t(0));
        return;
    }
    for (int i = 0; i < nFolds; i++)
        mov(simd, state.tileXorFolds[i].ud(), uint16_t(0));
}

template <HW hw>
void Generator<hw>::case5InitTileXorSlmMatrix(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                              GEMMState &state) {
    if (!problem.case5TileXorSlm || state.tileXorSlmLayout.valid())
        return;

    MatrixAddressing slmA;
    MatrixAddressingStrategy slmS;

    slmA.layout = MatrixLayout::Pc;
    slmA.packSize = 1;
    slmA.crosspack = 1;
    slmA.setAlignment(sizeof(uint32_t));

    slmS.base = AddressBase::createSLM();
    slmS.accessType = AccessType::Scattered;
    slmS.padded = true;
    slmS.newDP = (hw >= HW::XeHPG);

    const int maxRBlock = (hw == HW::Gen12LP) ? 8 : 0;
    state.tileXorSlmLayout = RegisterLayout::tryCreate(hw, Type::u32, 1, 1, slmA, slmS, false, false, true,
                                                       AvoidFragment, maxRBlock, 0);
    if (!state.tileXorSlmLayout.valid())
        stub("case5 tile_xor SLM matrix layout");

    allocAddrRegs(state.tileXorSlmAddrs, state.tileXorSlmLayout, state);
}

namespace {

constexpr int kFoldRotl = 13;
constexpr int kFoldRotlShr = 32 - kFoldRotl;
constexpr int kWrapGrfPerFold = 2;

inline int case5XorOutputMilestoneCount(const GEMMProblem &problem) {
    if (!problem.case5TileXorWrap) {
        return std::max(1, problem.case5XorMaxMilestones);
    }
    if (problem.case5XorOutputMilestones > 0) {
        return problem.case5XorOutputMilestones;
    }
    return std::max(1, problem.case5XorMaxMilestones / 2);
}

inline bool case5TileXorSlmFastPath() {
    static const bool on = [] {
        const char *v = std::getenv("CASE5_XOR_SLM_FAST");
        if (!v || v[0] == '\0') {
            return true;
        }
        return v[0] != '0';
    }();
    return on;
}

// Per-milestone code variants. The defaults are the measured-fastest forms (UHD 770, 4096^3:
// -1% vs the original); the env values below switch back to the original for ablation.
// Each is read once.
inline const char *case5Env(const char *name) {
    const char *v = std::getenv(name);
    return (v && v[0]) ? v : nullptr;
}

// CASE5_XOR_FOLD: simd16 (default: one SIMD16 chain over GRF pairs) | tree (4 chains, no
//                 zero-init) | chain (original: zero + one chained SIMD8 xor per GRF).
// simd16 and tree fall back to chain if their temporaries can't be allocated.
inline int case5FoldMode() {
    static const int mode = [] {
        const char *v = case5Env("CASE5_XOR_FOLD");
        if (!v) return 2;
        if (!std::strcmp(v, "chain")) return 0;
        if (!std::strcmp(v, "tree")) return 1;
        return 2;
    }();
    return mode;
}

// CASE5_XOR_GATE: count (default: the panel register counts down from xorPeriod, add.ze + jmpi
//                 per panel) | panel (original: add, and, cmp, jmpi per panel).
inline bool case5GateCount() {
    static const bool on = [] {
        const char *v = case5Env("CASE5_XOR_GATE");
        return !(v && !std::strcmp(v, "panel"));
    }();
    return on;
}

// CASE5_XOR_ADDR: inc (default: a running SLM byte offset per thread, +4*nFolds per milestone;
//                 SLM, no wrap, 1 fold, else the original) | full (original: recompute the
//                 slot/offset from (ms, localId) every milestone).
inline bool case5AddrInc() {
    static const bool on = [] {
        const char *v = case5Env("CASE5_XOR_ADDR");
        return !(v && !std::strcmp(v, "full"));
    }();
    return on;
}

// CASE5_XOR_NOSTORE=1: emit no milestone code at all (and no epilogue flush): GEMM-only timing reference.
inline bool case5NoStore() {
    static const bool on = [] {
        const char *v = case5Env("CASE5_XOR_NOSTORE");
        return v && v[0] != '0';
    }();
    return on;
}
} // namespace

// Single-dword A64 messages below (exec 1, lane 0 address in hdr[0].uq(0)): Gen12LP..XeHPC use the
// legacy scattered dword message; Xe2 has no legacy data port (nGEN rejects those messages there), so
// it uses the LSC scattered dword. The headers get two GRFs: that is what the A64 descriptor declares.

template <HW hw>
void Generator<hw>::case5SlmDwordStore(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                       GEMMState &state, const Subregister &byteOff, const GRF &data) {
    case5InitTileXorSlmMatrix(problem, strategy, state);
    setupAddr(state.tileXorSlmAddrs, byteOff, state.tileXorSlmLayout, Subregister(), strategy, state);
    const auto &block = state.tileXorSlmLayout[0];
    const auto &atype = state.tileXorSlmLayout.addressing();
    const auto &astrategy = state.tileXorSlmLayout.addressingStrategy();
    if (astrategy.newDP)
        storeMatrixBlock(data, block, atype, astrategy, state.tileXorSlmAddrs[0], strategy, state);
    else
        store(block.simdSize, surface_dword(ChannelMask::r), astrategy.base, state.tileXorSlmAddrs[0], data);
}

template <HW hw>
void Generator<hw>::case5SlmDwordLoad(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                      GEMMState &state, const Subregister &byteOff, const GRF &data) {
    case5InitTileXorSlmMatrix(problem, strategy, state);
    setupAddr(state.tileXorSlmAddrs, byteOff, state.tileXorSlmLayout, Subregister(), strategy, state);
    const auto &block = state.tileXorSlmLayout[0];
    const auto &atype = state.tileXorSlmLayout.addressing();
    const auto &astrategy = state.tileXorSlmLayout.addressingStrategy();
    if (astrategy.newDP)
        loadMatrixBlock(data, block, atype, astrategy, state.tileXorSlmAddrs[0], strategy, state);
    else
        load(block.simdSize, data, surface_dword(ChannelMask::r), astrategy.base, state.tileXorSlmAddrs[0]);
}

template <HW hw>
void Generator<hw>::case5SlmMatrixStore(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                        GEMMState &state, const Subregister &ms, int fi, const GRF &data) {
    auto byteOff = state.ra.alloc_sub<uint32_t>();
    case5EmitTileXorSlmByteOff(problem, strategy, state, byteOff, ms, fi);
    if (case5TileXorSlmFastPath())
        case5SlmDwordStore(problem, strategy, state, byteOff, data);
    else {
        case5InitTileXorSlmMatrix(problem, strategy, state);
        setupAddr(state.tileXorSlmAddrs, byteOff, state.tileXorSlmLayout, Subregister(), strategy, state);
        storeMatrixBlock(data, state.tileXorSlmLayout[0], state.tileXorSlmLayout.addressing(),
                         state.tileXorSlmLayout.addressingStrategy(), state.tileXorSlmAddrs[0], strategy, state);
    }
    state.ra.safeRelease(byteOff);
}

template <HW hw>
void Generator<hw>::case5SlmMatrixLoad(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                       int ms, int fi, const GRF &data) {
    auto byteOff = state.ra.alloc_sub<uint32_t>();
    case5EmitTileXorSlmByteOffConstMs(problem, strategy, state, byteOff, ms, fi);
    if (case5TileXorSlmFastPath())
        case5SlmDwordLoad(problem, strategy, state, byteOff, data);
    else {
        case5InitTileXorSlmMatrix(problem, strategy, state);
        setupAddr(state.tileXorSlmAddrs, byteOff, state.tileXorSlmLayout, Subregister(), strategy, state);
        loadMatrixBlock(data, state.tileXorSlmLayout[0], state.tileXorSlmLayout.addressing(),
                        state.tileXorSlmLayout.addressingStrategy(), state.tileXorSlmAddrs[0], strategy, state);
    }
    state.ra.safeRelease(byteOff);
}

template <HW hw>
void Generator<hw>::case5EmitTileXorSlmByteOff(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                              GEMMState &state, const Subregister &out,
                                              const Subregister &ms, int fi) {
    const int nFolds = std::max(1, problem.case5XorSubGridM) * std::max(1, problem.case5XorSubGridN);
    const int slotsPerThread = case5XorSlmSlotMs(problem) * nFolds;
    const uint32_t slmBase =
            static_cast<uint32_t>(case5TileXorSlmByteOffset(hw, problem, strategy));

    auto slot = state.ra.alloc_sub<uint32_t>();
    emad(1, slot, Immediate::w(fi), ms, nFolds, strategy, state);
    emad(1, slot, slot, state.tileXorLocalId, slotsPerThread, strategy, state);
    mulConstant(1, out, slot, 4);
    if (slmBase != 0)
        add(1, out, out, slmBase);
    if (state.inputs.slmBase.isValid())
        add(1, out, out, state.inputs.slmBase);
    state.ra.safeRelease(slot);
}

template <HW hw>
void Generator<hw>::case5EmitTileXorSlmByteOffConstMs(const GEMMProblem &problem,
                                                      const GEMMStrategy &strategy, GEMMState &state,
                                                      const Subregister &out, int ms, int fi) {
    const int nFolds = std::max(1, problem.case5XorSubGridM) * std::max(1, problem.case5XorSubGridN);
    const int slotsPerThread = case5XorSlmSlotMs(problem) * nFolds;
    const uint32_t slmBase =
            static_cast<uint32_t>(case5TileXorSlmByteOffset(hw, problem, strategy));

    auto slot = state.ra.alloc_sub<uint32_t>();
    mov(1, slot, uint32_t(ms * nFolds + fi));
    emad(1, slot, slot, state.tileXorLocalId, slotsPerThread, strategy, state);
    mulConstant(1, out, slot, 4);
    if (slmBase != 0)
        add(1, out, out, slmBase);
    if (state.inputs.slmBase.isValid())
        add(1, out, out, state.inputs.slmBase);
    state.ra.safeRelease(slot);
}

template <HW hw>
void Generator<hw>::case5UpdateTileXorSpatialIdFromIds(const GEMMProblem &problem,
                                                         const GEMMStrategy &strategy, GEMMState &state,
                                                         const Subregister &idM, const Subregister &idN)
{
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);

    // Always keep gemm tile (row, col) for jackpot coords — avoid float divMod(spatial).
    mov(1, state.tileXorRow, idM);
    mov(1, state.tileXorCol, idN);
    emad(1, state.spatialId, idN, idM, state.inputs.tileCols, strategy, state);
    if (split && state.tileXorStoreHdr.isValid()) {
        auto lr = state.ra.alloc_sub<uint32_t>();
        auto lc = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, lr, state.tileXorRow, subGridM);
        mulConstant(1, lc, state.tileXorCol, subGridN);
        emad(1, lc, lc, lr, state.inputs.tileCols, strategy, state);
        shl(1, lc, lc, uint16_t(2));                       // sid0 * 4 bytes
        const int numMs = std::max(1, problem.case5XorMaxMilestones);
        if (problem.case5XorLayout == 1) {
            // [thread][ms][sub]: thread block at sid0 * num_ms * 4 bytes, 16 bytes per milestone.
            mulConstant(1, lc, lc, numMs);
            eadd(1, state.tileXorStoreHdr[0].uq(0), state.inputs.tileXor, lc, strategy, state);
            mov(1, state.tileXorCountBytes, uint16_t(subGridN * 4));
        } else if (problem.case5XorLayout == 2) {
            // [tile][ms]: sub-tile sc of this thread starts at (sid0 + sc) * words * 4 bytes, words =
            // num_ms (raw milestones) or 16 (Case 5.5a folded message); one header per sub-tile line,
            // advanced by B*4 bytes at each flush (5.5a stores once at the end).
            // Words per sub-tile line in tile_xor: the milestones, or the 16 folded (5.5a) / wrapped
            // (5.4, 5.5 interleaved: ms ^ ms+16) slots.
            const int words = (problem.case5TileXorFold16 || problem.case5TileXorWrap) ? 16 : numMs;
            mulConstant(1, lc, lc, words);
            // Sub-tile (sr, sc) of this thread has spatial id sid0 + sr * tile_cols + sc: consecutive
            // lines along a sub-tile row, then a jump of (tile_cols - subGridN + 1) lines to the next row.
            const int nLines = state.tileXorLineHdrPacked ? subGridM * subGridN : int(state.tileXorLineHdr.size());
            Subregister rowAdv;
            if (nLines > subGridN) {
                rowAdv = state.ra.alloc_sub<uint32_t>();
                add(1, rowAdv, state.inputs.tileCols, -int32_t(subGridN - 1));
                mulConstant(1, rowAdv, rowAdv, words * 4);
            }
            for (int fi = 0; fi < nLines; fi++) {
                auto hdrQ = state.tileXorLineHdrPacked ? state.tileXorLineHdr[0][0].uq(fi)
                                                       : state.tileXorLineHdr[fi][0].uq(0);
                eadd(1, hdrQ, state.inputs.tileXor, lc, strategy, state);
                if (fi + 1 >= nLines) break;
                if ((fi + 1) % subGridN == 0)
                    add(1, lc, lc, rowAdv);
                else
                    add(1, lc, lc, uint32_t(words * 4));
            }
            if (rowAdv.isValid()) state.ra.safeRelease(rowAdv);
            mov(1, state.tileXorCountBytes, uint16_t(state.tileXorLineBlock * 4));
        } else {
            eadd(1, state.tileXorStoreHdr[0].uq(0), state.inputs.tileXor, lc, strategy, state);
            shl(1, state.tileXorCountBytes, state.inputs.tileCount, uint16_t(2));
        }
        if (state.tileXorStoreBase.isValid() && problem.case5XorLayout != 2)
            mov(2, state.tileXorStoreBase.ud(0)(1), state.tileXorStoreHdr[0].ud(0)(1));
        state.ra.safeRelease(lr);
        state.ra.safeRelease(lc);
    }
    if (problem.case5TileXorInterleaved)
        case5InitTileXorPhase(problem, strategy, state, idM, idN);

    if ((problem.case5TileXorSlm || problem.case5TileXorAccSlm) && state.tileXorLocalId.isValid()) {
        const int wgM = strategy.wg[LoopM];
        const int wgN = strategy.wg[LoopN];
        auto localM = state.ra.alloc_sub<uint32_t>();
        auto localN = state.ra.alloc_sub<uint32_t>();
        mod(localM, idM, wgM, strategy, state);
        mod(localN, idN, wgN, strategy, state);
        emad(1, state.tileXorLocalId, localM, localN, wgM, strategy, state);
        state.ra.safeRelease(localM);
        state.ra.safeRelease(localN);
        if (problem.case5TileXorAccSlm && state.tileXorAccSlmOff.isValid()) {
            // Case 5.7: this thread's line blocks start at base + lid * (blocks * sub-tiles * 64 bytes).
            const int nFolds = std::max(1, problem.case5XorSubGridM) * std::max(1, problem.case5XorSubGridN);
            const int nBlocks = std::max(1, problem.case5XorMaxMilestones / 16);
            mulConstant(1, state.tileXorAccSlmOff, state.tileXorLocalId, nBlocks * nFolds * 64);
            const uint32_t base = static_cast<uint32_t>(case5TileXorAccSlmByteOffset(hw, problem, strategy));
            if (base)
                add(1, state.tileXorAccSlmOff, state.tileXorAccSlmOff, base);
            if (state.inputs.slmBase.isValid())
                add(1, state.tileXorAccSlmOff, state.tileXorAccSlmOff, state.inputs.slmBase);
        }
    }
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorAlloc(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                          GEMMState &state)
{
    if (!problem.case5TileXor || state.spatialId.isValid())
        return;

    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);

    state.tileXorMs = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.tileXorPanel = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    // Full GRFs: alloc_sub parents get reused by later ra.alloc() and clobber spat/row/col.
    state.spatialId = state.ra.alloc(getHint(HintType::LongTerm, strategy)).ud(0);
    if (problem.case5TileXorSlm || problem.case5TileXorAccSlm)
        state.tileXorLocalId = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.tileXorRow = state.ra.alloc(getHint(HintType::LongTerm, strategy)).ud(0);
    state.tileXorCol = state.ra.alloc(getHint(HintType::LongTerm, strategy)).ud(0);
    if (problem.case5JackpotCoop) {
        // Flat local thread id (lidN * wgM + lidM): names this thread's tiles in the SLM tables of
        // the cooperative jackpot epilogue (the saved local ids are released before then).
        // No extra GRF: the 16x64 strategy is at the register limit, so the flat local id lives in the
        // second dword of the (full) tileXorRow GRF, whose other dwords are never written.
        state.tileXorLidFlat = GRF(state.tileXorRow.getBase()).ud(1);
        mul(1, state.tileXorLidFlat, state.lidN, uint16_t(strategy.wg[LoopM]));
        add(1, state.tileXorLidFlat, state.tileXorLidFlat, state.lidM);
    }
    if (problem.case5TileXorSlm && (!problem.case5TileXorWrap || case5WrapAtFlush(problem)) && !split
            && case5AddrInc())
        state.tileXorSlmOff = state.ra.try_alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        // (invalid if registers are tight: the store then recomputes the offset as before)
    const int nFolds = split ? (subGridM * subGridN) : 1;
    // Interleaved XMX fold: one 16-lane accumulator per sub-tile (2 GRFs on 32-byte GRFs, 1 on Xe2) so
    // the SIMD16 bfn chain accumulates directly (no per-chunk collapse) and the scalars come out of one
    // packed reduction. The 1-GRF fold handles alias the first GRF of each range. CASE5_XOR_WIDE=0 disables.
    state.tileXorFoldWide.clear();
    {
        const char *wideEnv = std::getenv("CASE5_XOR_WIDE");
        if (problem.case5TileXorInterleaved && split && !(wideEnv && wideEnv[0] == '0')) {
            // 64-byte GRFs: a second GRF per accumulator for CASE5_XOR_SIMD32 / CASE5_XOR_ACC2.
            const bool twoGrf = (GRF::bytes(hw) == 64)
                    && (case5EnvOn("CASE5_XOR_SIMD32", false) || case5EnvOn("CASE5_XOR_ACC2", false));
            const int accGrfs = (GRF::bytes(hw) == 32 || twoGrf) ? 2 : 1;
            state.tileXorFoldWide.resize(nFolds);
            bool ok = true;
            for (int i = 0; i < nFolds; i++) {
                state.tileXorFoldWide[i] = state.ra.try_alloc_range(accGrfs, getHint(HintType::LongTerm, strategy));
                ok = ok && state.tileXorFoldWide[i].isValid();
            }
            if (!ok) {
                for (auto &r : state.tileXorFoldWide) state.ra.safeRelease(r);
                state.tileXorFoldWide.clear();
                std::fprintf(stderr, "Case5: wide fold accumulators (%d x %d GRF) do not fit; generic fold\n",
                             nFolds, accGrfs);
            }
        }
    }
    if (!state.tileXorFoldWide.empty()) {
        state.tileXorFolds.resize(nFolds);
        for (int i = 0; i < nFolds; i++)
            state.tileXorFolds[i] = state.tileXorFoldWide[i][0];
        state.tileXorFold = state.tileXorFolds[0];
    } else {
        state.tileXorFold = state.ra.alloc(getHint(HintType::LongTerm, strategy));
        if (split) {
            state.tileXorFolds.resize(nFolds);
            state.tileXorFolds[0] = state.tileXorFold;
            for (int i = 1; i < nFolds; i++)
                state.tileXorFolds[i] = state.ra.alloc(getHint(HintType::LongTerm, strategy));
        } else
            state.tileXorFolds.clear();
    }
    if (problem.case5XorPeriod > 1 || problem.case5TileXorInterleaved)
        state.flagCase5Xor = state.raVFlag.alloc();
    // Fast milestone store with A64 tile_xor: any sub-tile grid on the tile-major layout (one line
    // message per sub-tile every B milestones: 2x4, 3x2, ...), one row of 4 or 8 sub-tiles on the
    // [ms][tile] layout (one 16/32-byte message per milestone).
    // The wrap cases take it only interleaved (accumulated 16-milestone blocks); Case 5.6 has no tile_xor
    // pointer, nothing is stored there, so it needs no line headers.
    const bool wrapAcc = problem.case5TileXorWrap && problem.case5TileXorInterleaved;
    if (split && (problem.case5XorLayout == 2 || (subGridM == 1 && (nFolds == 4 || nFolds == 8)))
            && (wrapAcc || (!problem.case5TileXorWrap && !problem.case5TileXorWrapGrf)) && !problem.case5TileXorSlm
            && state.inputs.surfaceTileXor == InterfaceHandler::noSurface
            && (state.inputs.tileXor.isValid() || problem.case5TileXorBlake3)) {
        state.tileXorStoreHdr = state.ra.try_alloc_range(1, getHint(HintType::LongTerm, strategy));
        state.tileXorStoreData = state.ra.try_alloc(getHint(HintType::LongTerm, strategy));
        state.tileXorCountBytes = state.ra.try_alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (!state.tileXorStoreHdr.isValid() || !state.tileXorStoreData.isValid()
                || !state.tileXorCountBytes.isValid()) {
            state.ra.safeRelease(state.tileXorStoreHdr);
            state.ra.safeRelease(state.tileXorStoreData);
            state.ra.safeRelease(state.tileXorCountBytes);
        } else {
            // The send reads its header asynchronously; advancing the header right after the send
            // stalls on that read. Keep the running pointer in a separate GRF and copy it into the
            // header before each send instead. CASE5_XOR_HDRCOPY=0 disables.
            const char *hc = std::getenv("CASE5_XOR_HDRCOPY");
            if (!(hc && hc[0] == '0') && problem.case5XorLayout != 2)
                state.tileXorStoreBase = state.ra.try_alloc(getHint(HintType::LongTerm, strategy));
        }
    }
    // Tile-major output ([tile][ms]) on the fast-store path: each sub-tile's milestones are
    // contiguous, so a thread keeps B milestones per sub-tile in registers (rotating line buffer)
    // and writes one B*4-byte block per sub-tile every B milestones (B=16: 4x fewer messages than
    // one 16-byte store per milestone, full 64-byte lines completed by one thread). B is the
    // largest power of two <= 16 (CASE5_XOR_TILEBLOCK caps it) that divides the milestone count.
    // Without the fast-store path (other sub-tile grids, SLM staging, wrap) the generic path
    // stores one dword per sub-tile per milestone at the tile-major index instead.
    state.tileXorLineBlock = 0;
    state.tileXorLineHdr.clear();
    if (problem.case5XorLayout == 2 && state.tileXorStoreHdr.isValid() && hw < HW::XeHPG) {
        // The line store uses LSC (XeHPG+); older GPUs take the generic tile-major path.
        state.ra.safeRelease(state.tileXorStoreHdr);
        state.ra.safeRelease(state.tileXorStoreData);
        state.ra.safeRelease(state.tileXorCountBytes);
        state.ra.safeRelease(state.tileXorStoreBase);
    }
    if (problem.case5XorLayout == 2 && state.tileXorStoreHdr.isValid()) {
        const char *tb = std::getenv("CASE5_XOR_TILEBLOCK");
        int cap = tb ? std::atoi(tb) : 16;
        if (cap < 1 || cap > 16) cap = 16;
        const int numMs = std::max(1, problem.case5XorMaxMilestones);
        int B = 1;
        while (B * 2 <= cap && numMs % (B * 2) == 0) B *= 2;
        const int grfPerLine = std::max(1, B * 4 / GRF::bytes(hw));
        state.tileXorLineBlock = B;
        // The rotating line update reads one dword past the last line (pad GRF); the indirect slot
        // write (CASE5_XOR_LINE_IDX, 64-byte GRFs, default on in the miner: the Xe2 32x64 kernel is one
        // GRF short with the pad) does not. Same conditions as the gather below.
        const bool lineIdx = GRF::bytes(hw) == 64 && case5EnvOn("CASE5_XOR_LINE_IDX", true) && B > 1
                && grfPerLine == 1 && nFolds <= 8;
        state.tileXorLineBuf = state.ra.try_alloc_range(grfPerLine * nFolds + (lineIdx ? 0 : 1),
                                                        getHint(HintType::LongTerm, strategy));
        // The wrap cases never store a line from the loop (blocks are accumulated; 5.4 / 5.5 store the
        // accumulators once at the end, when the final flush computes the addresses, 5.6 hashes them),
        // so they keep no per-sub-tile headers: 8 GRFs the 32x64 kernel does not have.
        const int nHdr = (problem.case5TileXorBlake3 || wrapAcc || problem.case5TileXorFold16) ? 0 : nFolds;
        // 64-byte GRFs: up to 8 line addresses share one GRF (qword fi) and the send header is
        // tileXorStoreHdr, loaded before each store (one mov per line per flush); saves nHdr - 1 GRFs.
        // Opt-in (CASE5_XOR_HDR_PACK=1): nothing needs the GRFs yet.
        state.tileXorLineHdrPacked = nHdr > 1 && nHdr <= 8 && GRF::bytes(hw) == 64
                && case5EnvOn("CASE5_XOR_HDR_PACK", false);
        const int nHdrGrf = state.tileXorLineHdrPacked ? 1 : nHdr;
        state.tileXorLineHdr.resize(nHdrGrf);
        bool ok = state.tileXorLineBuf.isValid();
        for (int i = 0; i < nHdrGrf; i++) {
            state.tileXorLineHdr[i] = state.ra.try_alloc_range(1, getHint(HintType::LongTerm, strategy));
            ok = ok && state.tileXorLineHdr[i].isValid();
        }
        // The flush test needs a flag (set transiently right before its jmpi); period-1 paths
        // without the interleaved fold do not allocate one otherwise.
        if (ok && !state.flagCase5Xor.isValid())
            state.flagCase5Xor = state.raVFlag.alloc();
        // Case 5.5a and the interleaved wrap cases: one 16-word accumulator per sub-tile, laid out like
        // the lines (5.5a: rotl 13 and xor per block; wrap: xor, so the block count must be 2).
        if (ok && (problem.case5TileXorFold16 || wrapAcc)) {
            if (B != 16)
                throw std::runtime_error("Case5 wrap/fold16 accumulators need 16-milestone line blocks (K/128 % 16 == 0)");
            if (problem.case5TileXorAccSlm) {
                // Case 5.7: each completed line block goes to SLM (case5TileXorSend) and the message is
                // folded after the k loop (case5TileXorLoadAccFromSlm), in registers the finished C tile
                // no longer needs: the accumulators cost nothing during the loop.
                state.tileXorAccSlmOff = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            } else {
                state.tileXorFoldAcc = state.ra.try_alloc_range(grfPerLine * nFolds, getHint(HintType::LongTerm, strategy));
                if (!state.tileXorFoldAcc.isValid())
                    throw std::runtime_error("Case5 wrap/fold16: insufficient registers for the fold accumulators");
            }
        }
        if (!ok) {
            state.ra.safeRelease(state.tileXorLineBuf);
            for (auto &h : state.tileXorLineHdr) state.ra.safeRelease(h);
            state.tileXorLineHdr.clear();
            state.tileXorLineBlock = 0;
            if (problem.case5TileXorInterleaved)
                throw std::runtime_error("Case5 tile-major store: insufficient registers for the line buffer");
            // Generic path instead: drop the fast store (it writes [ms][tile] 16-byte groups).
            state.ra.safeRelease(state.tileXorStoreHdr);
            state.ra.safeRelease(state.tileXorStoreData);
            state.ra.safeRelease(state.tileXorCountBytes);
        }
    }
    if (problem.case5TileXorInterleaved) {
        state.tileXorPhase = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (state.tileXorFoldWide.empty())
            state.tileXorPairTmp = state.ra.try_alloc_range(2, getHint(HintType::LongTerm, strategy));
    }
    if (problem.case5TileXorInterleaved && !state.tileXorStoreHdr.isValid())
        throw std::runtime_error("Case5 interleaved XOR needs the fast store path (split sub-tiles on the "
                                 "tile-major layout or a 1x4/1x8 row on [ms][tile], A64)");
    if (problem.case5TileXorWrapGrf && !wrapAcc) {
        state.tileXorWrapGrf.resize(kWrapGrfPerFold * nFolds);
        for (int i = 0; i < kWrapGrfPerFold * nFolds; i++)
            state.tileXorWrapGrf[i] = state.ra.alloc(getHint(HintType::LongTerm, strategy));
    }
    if (problem.case5TileXorSlm)
        case5InitTileXorSlmMatrix(problem, strategy, state);
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorIterationInit(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                                  GEMMState &state)
{
    if (!problem.case5TileXor)
        return;

    gemmCase5TileXorAlloc(problem, strategy, state);

    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);
    const int nFolds = split ? (subGridM * subGridN) : 1;

    mov(1, state.tileXorMs, uint16_t(0));
    // CASE5_XOR_LINE_IDX (64-byte GRFs): a0.8..15 = line byte offsets 0, 64, .., 448 for the indirect
    // slot write in case5TileXorGather (a0 is otherwise unused inside the k loop).
    if (GRF::bytes(hw) == 64 && case5EnvOn("CASE5_XOR_LINE_IDX", true) && problem.case5XorLayout == 2
            && state.tileXorLineBuf.isValid() && state.tileXorLineBlock > 1) {
        mov(8, a0[8](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        shl(8, a0[8](1), a0[8](8, 8, 1), uint16_t(6));
    }
    // Case 5.5a: message accumulators start at zero (rotl(0) ^ block0 = block0).
    if (state.tileXorFoldAcc.isValid()) {
        const int dw = elementsPerGRF(hw, DataType::ud);
        for (int g = 0; g < state.tileXorFoldAcc.getLen(); g++)
            mov(dw, state.tileXorFoldAcc[g].ud(), uint16_t(0));
    }
    // Count mode (CASE5_XOR_GATE=count): panels left until the first milestone store.
    const bool countGate = (problem.case5XorPeriod > 1) && !problem.case5TileXorIncremental && case5GateCount();
    mov(1, state.tileXorPanel, uint16_t(countGate ? problem.case5XorPeriod : 0));
    if (state.tileXorSlmOff.isValid())
        case5EmitTileXorSlmByteOffConstMs(problem, strategy, state, state.tileXorSlmOff, 0, 0);
    case5ZeroFoldGrfs(strategy, state, nFolds);
    {
        const int simd = elementsPerGRF(hw, DataType::ud);
        for (auto &g : state.tileXorWrapGrf)   // (empty when the wrap runs on the interleaved accumulators)
            mov(simd, g.ud(), uint16_t(0));
    }
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorFoldChunk(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                              GEMMState &state, int chunkIdx, int numChunks)
{
    if (!problem.case5TileXor || !problem.case5TileXorIncremental)
        return;
    if (problem.case5TileXorNop)
        return;
    if (state.C_regs.empty() || state.C_regs[0].empty())
        return;
    if (numChunks <= 0 || chunkIdx < 0 || chunkIdx >= numChunks)
        return;

    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subN = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);

    const int simd = elementsPerGRF(hw, DataType::ud);
    const int unrollM = strategy.unroll[LoopM];
    const int unrollN = strategy.unroll[LoopN];

    auto xorSimdIntoFold = [&](GRF foldReg, const Subregister &C) {
        xor_(simd, foldReg.ud(), foldReg.ud(), C.d(0)(1));
    };

    if (!split) {
        const int nGrf = state.C_regs[0].getLen();
        const int grfsPerChunk = (nGrf + numChunks - 1) / numChunks;
        const int r0 = chunkIdx * grfsPerChunk;
        const int r1 = std::min(r0 + grfsPerChunk, nGrf);
        for (int r = r0; r < r1; r++)
            xor_(simd, state.tileXorFold.ud(), state.tileXorFold.ud(), state.C_regs[0][r].ud());
    } else {
        const auto &folds = state.tileXorFolds;
        auto foldAt = [&](int sr, int sc) { return folds[sr * subGridN + sc]; };
        const int colsPerChunk = unrollN / numChunks;
        const int col0 = chunkIdx * colsPerChunk;
        const int col1 = col0 + colsPerChunk;

        if (state.C_layout.colMajor()) {
            for (int col = col0; col < col1; col++) {
                const int sc = col / subN;
                for (int row = 0; row < unrollM; row += simd) {
                    const int sr = row / subM;
                    Subregister C0 = state.C_layout.find(row, col, state.C_regs[0]);
                    xorSimdIntoFold(foldAt(sr, sc), C0);
                }
            }
        } else {
            const int rowsPerChunk = unrollM / numChunks;
            const int row0 = chunkIdx * rowsPerChunk;
            const int row1 = row0 + rowsPerChunk;
            for (int row = row0; row < row1; row++) {
                const int sr = row / subM;
                for (int col = 0; col < unrollN; col += simd) {
                    const int sc = col / subN;
                    Subregister C0 = state.C_layout.find(row, col, state.C_regs[0]);
                    xorSimdIntoFold(foldAt(sr, sc), C0);
                }
            }
        }
    }
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorStore(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                          GEMMState &state)
{
    if (!problem.case5TileXor)
        return;
    if (state.C_regs.empty() || state.C_regs[0].empty())
        return;

    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subN = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);
    const bool incremental = problem.case5TileXorIncremental;
    const int xorPeriod = (problem.case5XorPeriod > 0) ? problem.case5XorPeriod : 1;
    const bool gateStore = (xorPeriod > 1) && !incremental;
    // Case5.4 with SLM staging folds ms with ms + outputMs at the flush: in the loop every
    // milestone is stored to its own slot, as without wrapping.
    const bool wrapAtFlush = case5WrapAtFlush(problem);
    const bool wrap = problem.case5TileXorWrap && !wrapAtFlush;
    const bool wrapGrf = problem.case5TileXorWrapGrf;
    const int wrapGrfStoreMode = problem.case5TileXorWrapGrfStoreMode;
    const bool wrapGrfStoreInLoop = wrapGrf && (wrapGrfStoreMode == 1);
    const bool wrapGrfStageUnified = problem.case5TileXorWrapGrfStageUnified;
    const int outputMs = case5XorOutputMilestoneCount(problem);
    const uint16_t outputMsMask = static_cast<uint16_t>(outputMs - 1);

    if (case5NoStore())
        return;

    if (problem.case5TileXorInterleaved && !problem.case5TileXorNop) {
        // Interleaved XMX mode: phase-A folds/stores come from the dpas chain hooks and the final
        // flush. Phase-B threads fold and store here (end-of-body sequencer position) unless
        // CASE5_XOR_BPOS=chain puts them right after the last dpas chain (hangs the GPU).
        const char *bpos = std::getenv("CASE5_XOR_BPOS");
        const bool bAtChain = bpos && !std::strcmp(bpos, "chain");
        const int osysB = systolicParams(problem).osys;
        if (!bAtChain && state.tileXorPhase.isValid() && strategy.unroll[LoopM] > osysB) {
            // Phase B: fold everything now (rows final), gather into the store GRF; the send itself
            // goes out at the next pass-0 boundary together with phase A (CASE5_XOR_BSEND=here to
            // store immediately).
            Label lSkipB;
            case5JumpIfPhase(state, false, lSkipB);
            for (int sr = 0; sr < subGridM; sr++)
                for (int sc = 0; sc < subGridN; sc++)
                    case5FoldSubtileFull(problem, strategy, state, state.tileXorFolds[sr * subGridN + sc], sr, sc);
            case5TileXorGather(problem, strategy, state, true, false);
            const char *bsend = std::getenv("CASE5_XOR_BSEND");
            if (bsend && !std::strcmp(bsend, "here"))
                case5TileXorSend(problem, strategy, state, false);
            mark(lSkipB);
        }
        add(1, state.tileXorMs, state.tileXorMs, uint16_t(1));
        return;
    }

    const bool gateCount = gateStore && case5GateCount();
    if (!gateCount)
        add(1, state.tileXorPanel, state.tileXorPanel, uint16_t(1));

    Label lSkipXor;
    Subregister rem;
    if (gateCount) {
        // Panels left in this milestone (initialised to xorPeriod; reset below after the store).
        add(1 | ze | state.flagCase5Xor, state.tileXorPanel, state.tileXorPanel, -1);
        jmpi(1 | ~state.flagCase5Xor, lSkipXor);
    } else if (gateStore) {
        rem = state.ra.alloc_sub<uint32_t>();
        and_(1, rem, state.tileXorPanel, uint16_t(xorPeriod - 1));
        cmp(1 | nz | state.flagCase5Xor, rem, 0);
        jmpi(1 | state.flagCase5Xor, lSkipXor);
    }

    const int simd = elementsPerGRF(hw, DataType::ud);
    const int unrollM = strategy.unroll[LoopM];
    const int unrollN = strategy.unroll[LoopN];
    GRF fold = state.tileXorFold.isValid() ? state.tileXorFold : state.ra.alloc();
    const bool releaseFold = !state.tileXorFold.isValid();
    const bool useSurface = (state.inputs.surfaceTileXor != InterfaceHandler::noSurface);
    const bool treeFold = problem.case5TileXorTree && !incremental && !problem.case5TileXorNop;
    const bool partialTile = (subM < unrollM) || (subN < unrollN);

    auto reduceFoldGrfToScalar = [&](GRF foldReg) {
        for (int n = simd >> 1; n >= 1; n >>= 1)
            xor_(n, foldReg.ud(0)(1), foldReg.ud(0)(1), foldReg.ud(n)(1));
    };

    auto xorSimdIntoFold = [&](GRF foldReg, const Subregister &C) {
        xor_(simd, foldReg.ud(), foldReg.ud(), C.d(0)(1));
    };

    // Xe-HP and later have bfn: 0x96 (s0 ^ s1 ^ s2) folds two accumulator chunks per instruction, half
    // the ALU cycles of the 2-input chain. That matters on dp4a kernels (Xe-LPG), where the fold shares
    // the ALU pipe with dp4a: 128 -> 64 fold instructions per milestone for a 32x32 tile. Only
    // GRF-aligned chunks are paired (3-source operands starting mid-GRF misbehaved on XeHPG).
    // CASE5_XOR_BFN=0 keeps the 2-input xor.
    const bool foldBfn = [&] {
        const char *v = std::getenv("CASE5_XOR_BFN");
        return hw >= HW::XeHP && !(v && v[0] == '0');
    }();
    // folds[f] = XOR of chunks[f] (simd lanes each). The first pair initialises the fold, so no zeroing.
    // The folds' chains are emitted round-robin, so consecutive instructions do not depend on each other.
    auto foldChunksInto = [&](const std::vector<GRF> &foldRegs, const std::vector<std::vector<Subregister>> &chunks) {
        struct Op { int kind; Subregister a, b; };   // 0 mov a, 1 xor a b (init), 2 bfn a b, 3 xor a (accumulate)
        std::vector<std::vector<Op>> ops(foldRegs.size());
        size_t longest = 0;
        for (size_t f = 0; f < foldRegs.size(); f++) {
            const auto &c = chunks[f];
            auto &o = ops[f];
            size_t i = 0;
            if (c.size() >= 2) { o.push_back({1, c[0], c[1]}); i = 2; }
            else if (c.size() == 1) { o.push_back({0, c[0], c[0]}); i = 1; }
            while (i < c.size()) {
                if (foldBfn && i + 1 < c.size() && c[i].getOffset() == 0 && c[i + 1].getOffset() == 0) {
                    o.push_back({2, c[i], c[i + 1]});
                    i += 2;
                } else {
                    o.push_back({3, c[i], c[i]});
                    i += 1;
                }
            }
            longest = std::max(longest, o.size());
        }
        for (size_t f = 0; f < foldRegs.size(); f++)
            if (ops[f].empty()) mov(simd, foldRegs[f].ud(), uint16_t(0));
        for (size_t k = 0; k < longest; k++) {
            for (size_t f = 0; f < foldRegs.size(); f++) {
                if (k >= ops[f].size()) continue;
                const Op &op = ops[f][k];
                const GRF &dst = foldRegs[f];
                switch (op.kind) {
                    case 0: mov(simd, dst.ud(), op.a.d(0)(1)); break;
                    case 1: xor_(simd, dst.ud(), op.a.d(0)(1), op.b.d(0)(1)); break;
                    case 2: bfn(simd, 0x96, dst.ud(), dst.ud(), op.a.ud(0)(1), op.b.ud(0)(1)); break;
                    default: xorSimdIntoFold(dst, op.a); break;
                }
            }
        }
    };

    auto treeMergeGrfPartials = [&](const std::vector<GRF> &partials) {
        for (int n = kFoldTreeFanin >> 1; n >= 1; n >>= 1)
            for (int j = 0; j < n; j++)
                xor_(simd, partials[j].ud(), partials[j].ud(), partials[j + n].ud());
    };

    auto foldGrfsTreeGrf = [&](GRF foldReg, auto emitGrfChunks) {
        std::vector<GRF> partials(kFoldTreeFanin);
        partials[0] = foldReg;
        for (int i = 1; i < kFoldTreeFanin; i++)
            partials[i] = state.ra.alloc(getHint(HintType::TempComp0, strategy));
        for (int i = 0; i < kFoldTreeFanin; i++)
            mov(simd, partials[i].ud(), uint16_t(0));

        int chunk = 0;
        emitGrfChunks([&](const GRF &Cgrf) {
            xor_(simd, partials[chunk % kFoldTreeFanin].ud(), partials[chunk % kFoldTreeFanin].ud(),
                 Cgrf.ud());
            chunk++;
        });

        treeMergeGrfPartials(partials);
        reduceFoldGrfToScalar(partials[0]);
        for (int i = 1; i < kFoldTreeFanin; i++)
            state.ra.safeRelease(partials[i]);
    };

    auto foldGrfsTreeSimd = [&](GRF foldReg, auto emitSimdChunks) {
        std::vector<GRF> partials(kFoldTreeFanin);
        partials[0] = foldReg;
        for (int i = 1; i < kFoldTreeFanin; i++)
            partials[i] = state.ra.alloc(getHint(HintType::TempComp0, strategy));
        for (int i = 0; i < kFoldTreeFanin; i++)
            mov(simd, partials[i].ud(), uint16_t(0));

        int chunk = 0;
        emitSimdChunks([&](const Subregister &C0) {
            xor_(simd, partials[chunk % kFoldTreeFanin].ud(), partials[chunk % kFoldTreeFanin].ud(),
                 C0.d(0)(1));
            chunk++;
        });

        treeMergeGrfPartials(partials);
        reduceFoldGrfToScalar(partials[0]);
        for (int i = 1; i < kFoldTreeFanin; i++)
            state.ra.safeRelease(partials[i]);
    };

    auto foldSubtile = [&](GRF foldReg, int sr, int sc) {
        const int row0 = sr * subM;
        const int row1 = row0 + subM;
        const int col0 = sc * subN;
        const int col1 = col0 + subN;

        auto emitSimdChunks = [&](auto emit) {
            if (state.C_layout.colMajor()) {
                for (int col = col0; col < col1; col++) {
                    for (int row = row0; row < row1; row += simd) {
                        emit(state.C_layout.find(row, col, state.C_regs[0]));
                    }
                }
            } else {
                for (int row = row0; row < row1; row++) {
                    for (int col = col0; col < col1; col += simd) {
                        emit(state.C_layout.find(row, col, state.C_regs[0]));
                    }
                }
            }
        };

        if (treeFold) {
            foldGrfsTreeSimd(foldReg, [&](auto emit) { emitSimdChunks(emit); });
            return;
        }

        mov(simd, foldReg.ud(), uint16_t(0));
        emitSimdChunks([&](const Subregister &C0) { xorSimdIntoFold(foldReg, C0); });
        reduceFoldGrfToScalar(foldReg);
    };

    auto foldToScalar = [&]() {
        if (problem.case5TileXorNop) {
            mov(1, fold.ud(0), uint32_t(0x51CACE51u));
            return;
        }
        if (incremental) {
            reduceFoldGrfToScalar(fold);
            return;
        }
        if (treeFold && !partialTile) {
            foldGrfsTreeGrf(fold, [&](auto emit) {
                for (int r = 0; r < state.C_regs[0].getLen(); r++)
                    emit(state.C_regs[0][r]);
            });
            return;
        }
        if (partialTile) {
            foldSubtile(fold, 0, 0);
            return;
        }
        const int nGrf = state.C_regs[0].getLen();
        const int foldMode = case5FoldMode();
        if (foldMode == 1 && nGrf >= 8) {
            // Four independent chains, each started by xor(C[a], C[b]) so no zero-init,
            // then merged: 31 xor for 32 GRFs instead of mov + 32.
            const int nChains = 4;
            std::vector<GRF> t(nChains);
            t[0] = fold;
            bool ok = true;
            for (int c = 1; c < nChains; c++) {
                t[c] = state.ra.try_alloc(getHint(HintType::TempComp0, strategy));
                ok = ok && t[c].isValid();
            }
            if (ok) {
                for (int c = 0; c < nChains; c++) {
                    const int r0 = nGrf * c / nChains, r1 = nGrf * (c + 1) / nChains;
                    if (r1 - r0 >= 2)
                        xor_(simd, t[c].ud(), state.C_regs[0][r0].ud(), state.C_regs[0][r0 + 1].ud());
                    else
                        mov(simd, t[c].ud(), state.C_regs[0][r0].ud());
                    for (int r = r0 + 2; r < r1; r++)
                        xor_(simd, t[c].ud(), t[c].ud(), state.C_regs[0][r].ud());
                }
                xor_(simd, t[0].ud(), t[0].ud(), t[2].ud());
                xor_(simd, t[1].ud(), t[1].ud(), t[3].ud());
                xor_(simd, t[0].ud(), t[0].ud(), t[1].ud());
            }
            for (int c = 1; c < nChains; c++)
                if (t[c].isValid())
                    state.ra.safeRelease(t[c]);
            if (ok) {
                reduceFoldGrfToScalar(fold);
                return;
            }
        }
        if (foldMode == 2 && nGrf >= 4 && (nGrf % 2) == 0 && simd == 8) {
            // One SIMD16 chain over GRF pairs (C_regs are contiguous; 32-byte GRFs, i.e. 8 dwords
            // each, so an exec-16 operand is exactly a pair), then fold the two halves.
            // SIMD16 int32 takes two cycles on Xe-LP, so this only saves issue slots (-0.4%).
            auto F = state.ra.try_alloc_range(2, getHint(HintType::TempComp0, strategy));
            if (F.isValid()) {
                xor_(16, F[0].ud(), state.C_regs[0][0].ud(), state.C_regs[0][2].ud());
                int r = 4;
                for (; foldBfn && r + 2 < nGrf; r += 4)
                    bfn(16, 0x96, F[0].ud(), F[0].ud(), state.C_regs[0][r].ud(), state.C_regs[0][r + 2].ud());
                for (; r < nGrf; r += 2)
                    xor_(16, F[0].ud(), F[0].ud(), state.C_regs[0][r].ud());
                xor_(simd, fold.ud(), F[0].ud(), F[1].ud());
                state.ra.safeRelease(F);
                reduceFoldGrfToScalar(fold);
                return;
            }
        }
        mov(simd, fold.ud(), uint16_t(0));
        for (int r = 0; r < nGrf; r++)
            xor_(simd, fold.ud(), fold.ud(), state.C_regs[0][r].ud());
        reduceFoldGrfToScalar(fold);
    };

    auto foldAllSubtilesOnePass = [&]() {
        const int nFolds = subGridM * subGridN;
        const auto &folds = state.tileXorFolds;

        if (problem.case5TileXorNop) {
            for (int i = 0; i < nFolds; i++)
                mov(1, folds[i].ud(0), uint32_t(0x51CACE51u));
            return;
        }
        if (incremental) {
            for (int i = 0; i < nFolds; i++)
                reduceFoldGrfToScalar(folds[i]);
            return;
        }

        if (treeFold) {
            for (int sr = 0; sr < subGridM; sr++)
                for (int sc = 0; sc < subGridN; sc++)
                    foldSubtile(folds[sr * subGridN + sc], sr, sc);
            return;
        }

        std::vector<std::vector<Subregister>> chunks(nFolds);
        if (state.C_layout.colMajor()) {
            for (int col = 0; col < unrollN; col++) {
                const int sc = col / subN;
                for (int row = 0; row < unrollM; row += simd) {
                    const int sr = row / subM;
                    chunks[sr * subGridN + sc].push_back(state.C_layout.find(row, col, state.C_regs[0]));
                }
            }
        } else {
            for (int row = 0; row < unrollM; row++) {
                const int sr = row / subM;
                for (int col = 0; col < unrollN; col += simd) {
                    const int sc = col / subN;
                    chunks[sr * subGridN + sc].push_back(state.C_layout.find(row, col, state.C_regs[0]));
                }
            }
        }
        std::vector<GRF> foldRegs(folds.begin(), folds.begin() + nFolds);
        foldChunksInto(foldRegs, chunks);

        for (int i = 0; i < nFolds; i++)
            reduceFoldGrfToScalar(folds[i]);
    };

    auto loadFoldAtIdx = [&](const Subregister &idx, const GRF &dst) {
        if (useSurface) {
            auto hdr = state.ra.alloc().ud();
            shl(1, hdr, idx, uint16_t(2));
            load(1, dst, surface_dword(ChannelMask::r), Surface(state.inputs.surfaceTileXor), hdr);
            state.ra.safeRelease(hdr);
        } else if (state.inputs.tileXor.isValid()) {
            auto hdr = state.ra.alloc_range(2);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
            if (hw >= HW::Xe2)
                load(1, dst, D32, A64, hdr);
            else
                load(1, dst, scattered_dword(), A64, hdr);
            state.ra.safeRelease(byteOff);
            state.ra.safeRelease(hdr);
        }
    };

    auto storeFoldAtIdx = [&](const Subregister &idx, const Subregister &slotMs, GRF foldReg, int fi = 0,
                              bool globalOnly = false) {
        if (problem.case5TileXorSlm && !globalOnly) {
            GRF storeVal = foldReg;
            if (std::getenv("CASE5_XOR_SLM_DEBUG_SPATIAL")) {
                storeVal = state.ra.alloc();
                mov(1, storeVal.ud(0), state.spatialId);
            }
            case5SlmMatrixStore(problem, strategy, state, slotMs, fi, storeVal);
            if (storeVal != foldReg)
                state.ra.safeRelease(storeVal);
            return;
        }
        if (useSurface) {
            auto hdr = state.ra.alloc().ud();
            shl(1, hdr, idx, uint16_t(2));
            store(1, surface_dword(ChannelMask::r), Surface(state.inputs.surfaceTileXor), hdr, foldReg);
            state.ra.safeRelease(hdr);
        } else if (state.inputs.tileXor.isValid()) {
            auto hdr = state.ra.alloc_range(2);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
            if (hw >= HW::Xe2)
                store(1, D32, A64, hdr, foldReg);
            else
                store(1, scattered_dword(), A64, hdr, foldReg);
            state.ra.safeRelease(byteOff);
            state.ra.safeRelease(hdr);
        }
    };

    auto storeWithWrap = [&](const Subregister &idx, const Subregister &slotMs, GRF foldReg, int fi = 0) {
        if (!wrap) {
            storeFoldAtIdx(idx, slotMs, foldReg, fi);
            return;
        }
        if (wrapGrf) {
            const GRF &wrapBase = state.tileXorWrapGrf[fi * kWrapGrfPerFold];
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            mov(1, a0[0], wrapBase.getBase() * GRF::bytes(hw));
            shl(1, byteOff, slotMs, uint16_t(2));
            add(1, a0[0], a0[0], byteOff);

            auto xorContributionIntoWrap = [&]() {
                if (problem.case5FuseJackpot || problem.case5TileXorBlake3
                        || problem.case5TileXorWrapGrf) {
                    // fold_milestones semantics (cp_jackpot / cp_onednn_jackpot): rotl slot then xor x.
                    auto slotVal = state.ra.alloc_sub<uint32_t>();
                    mov(1, slotVal, indirect[a0].ud(0));
                    auto rotHi = state.ra.alloc_sub<uint32_t>();
                    shr(1, rotHi, slotVal, uint16_t(kFoldRotlShr));
                    shl(1, slotVal, slotVal, uint16_t(kFoldRotl));
                    or_(1, slotVal, slotVal, rotHi);
                    xor_(1, slotVal, slotVal, foldReg.ud(0));
                    mov(1, indirect[a0].ud(0), slotVal);
                    state.ra.safeRelease(rotHi);
                    state.ra.safeRelease(slotVal);
                    return;
                }
                GRF contribution = foldReg;
                bool releaseContribution = false;
                xor_(1, indirect[a0].ud(0), indirect[a0].ud(0), contribution.ud(0));
                if (releaseContribution)
                    state.ra.safeRelease(contribution);
            };

            if (wrapGrfStageUnified) {
                xorContributionIntoWrap();
                if (wrapGrfStoreInLoop) {
                    Label lSkipWrapGrfStore;
                    cmp(1 | lt | state.flagCase5Xor, state.tileXorMs, uint16_t(outputMs));
                    jmpi(1 | state.flagCase5Xor, lSkipWrapGrfStore);
                    mov(1, foldReg.ud(0), indirect[a0].ud(0));
                    storeFoldAtIdx(idx, slotMs, foldReg, fi);
                    mark(lSkipWrapGrfStore);
                }
            } else {
                Label lWrapGrfHybridDone, lWrapGrfHybridSecondHalf;
                cmp(1 | lt | state.flagCase5Xor, state.tileXorMs, uint16_t(outputMs));
                jmpi(1 | ~state.flagCase5Xor, lWrapGrfHybridSecondHalf);
                mov(1, indirect[a0].ud(0), foldReg.ud(0));
                jmpi(1, lWrapGrfHybridDone);
                mark(lWrapGrfHybridSecondHalf);
                xorContributionIntoWrap();
                if (wrapGrfStoreInLoop)
                    storeFoldAtIdx(idx, slotMs, foldReg, fi);
                mark(lWrapGrfHybridDone);
            }
            state.ra.safeRelease(byteOff);
            return;
        }
        Label lSkipWrapLoad;
        cmp(1 | lt | state.flagCase5Xor, state.tileXorMs, uint16_t(outputMs));
        jmpi(1 | state.flagCase5Xor, lSkipWrapLoad);
        GRF prev = state.ra.alloc();
        if (problem.case5TileXorSlm) {
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            case5EmitTileXorSlmByteOff(problem, strategy, state, byteOff, slotMs, fi);
            case5SlmDwordLoad(problem, strategy, state, byteOff, prev);
            state.ra.safeRelease(byteOff);
        } else {
            loadFoldAtIdx(idx, prev);
        }
        xor_(1, foldReg.ud(0), foldReg.ud(0), prev.ud(0));
        state.ra.safeRelease(prev);
        mark(lSkipWrapLoad);
        storeFoldAtIdx(idx, slotMs, foldReg, fi);
    };

    const int nFolds = split ? (subGridM * subGridN) : 1;

    if (!split) {
        foldToScalar();
        if (!wrap && !wrapGrf && problem.case5TileXorSlm && case5AddrInc()
                && state.tileXorSlmOff.isValid()) {
            // Running SLM byte offset: this thread's slot for (ms, fold 0), advanced per milestone.
            case5SlmDwordStore(problem, strategy, state, state.tileXorSlmOff, fold);
            add(1, state.tileXorSlmOff, state.tileXorSlmOff, uint16_t(4 * nFolds));
        } else if (wrapGrf) {
            auto slotMs = state.ra.alloc_sub<uint32_t>();
            and_(1, slotMs, state.tileXorMs, outputMsMask);
            if (wrapGrfStoreInLoop) {
                auto idx = state.ra.alloc_sub<uint32_t>();
                case5TileXorIndex(problem, strategy, state, idx, state.spatialId, slotMs, outputMs);
                storeWithWrap(idx, slotMs, fold);
                state.ra.safeRelease(idx);
            } else {
                storeWithWrap(Subregister(), slotMs, fold);
            }
            state.ra.safeRelease(slotMs);
        } else {
            auto idx = state.ra.alloc_sub<uint32_t>();
            if (!wrap && !problem.case5TileXorSlm) {
                case5TileXorIndex(problem, strategy, state, idx, state.spatialId, state.tileXorMs, outputMs);
                storeFoldAtIdx(idx, state.tileXorMs, fold);
            } else {
                auto slotMs = state.ra.alloc_sub<uint32_t>();
                if (wrap)
                    and_(1, slotMs, state.tileXorMs, outputMsMask);
                else
                    mov(1, slotMs, state.tileXorMs);
                case5TileXorIndex(problem, strategy, state, idx, state.spatialId, slotMs, outputMs);
                if (wrap)
                    storeWithWrap(idx, slotMs, fold);
                else
                    storeFoldAtIdx(idx, slotMs, fold);
                state.ra.safeRelease(slotMs);
            }
            state.ra.safeRelease(idx);
        }
    } else {
        foldAllSubtilesOnePass();

        if (case5TileXorFastStoreOk(problem, strategy, state)) {
            case5TileXorFastStore(problem, strategy, state, false, false);
        } else if (wrapGrf) {
            auto slotMs = state.ra.alloc_sub<uint32_t>();
            and_(1, slotMs, state.tileXorMs, outputMsMask);
            if (wrapGrfStoreInLoop) {
                std::vector<Subregister> idxs(nFolds);
                for (int sr = 0; sr < subGridM; sr++) {
                    for (int sc = 0; sc < subGridN; sc++) {
                        const int fi = sr * subGridN + sc;
                        auto logicalRow = state.ra.alloc_sub<uint32_t>();
                        auto logicalCol = state.ra.alloc_sub<uint32_t>();
                        idxs[fi] = state.ra.alloc_sub<uint32_t>();

                        mulConstant(1, logicalRow, state.tileXorRow, subGridM);
                        add(1, logicalRow, logicalRow, uint16_t(sr));
                        mulConstant(1, logicalCol, state.tileXorCol, subGridN);
                        add(1, logicalCol, logicalCol, uint16_t(sc));
                        emad(1, idxs[fi], logicalCol, logicalRow, state.inputs.tileCols, strategy, state);
                        case5TileXorIndex(problem, strategy, state, idxs[fi], idxs[fi], slotMs, outputMs);

                        state.ra.safeRelease(logicalRow);
                        state.ra.safeRelease(logicalCol);
                    }
                }
                for (int fi = 0; fi < nFolds; fi++)
                    storeWithWrap(idxs[fi], slotMs, state.tileXorFolds[fi], fi);
                for (int fi = 0; fi < nFolds; fi++)
                    state.ra.safeRelease(idxs[fi]);
            } else {
                for (int fi = 0; fi < nFolds; fi++)
                    storeWithWrap(Subregister(), slotMs, state.tileXorFolds[fi], fi);
            }
            state.ra.safeRelease(slotMs);
        } else {
        Subregister slotMs;
        bool releaseSlotMs = false;
        if (wrap || problem.case5TileXorSlm) {
            slotMs = state.ra.alloc_sub<uint32_t>();
            releaseSlotMs = true;
            if (wrap)
                and_(1, slotMs, state.tileXorMs, outputMsMask);
            else
                mov(1, slotMs, state.tileXorMs);
        }

        std::vector<Subregister> idxs(nFolds);
        for (int sr = 0; sr < subGridM; sr++) {
            for (int sc = 0; sc < subGridN; sc++) {
                const int fi = sr * subGridN + sc;
                auto logicalRow = state.ra.alloc_sub<uint32_t>();
                auto logicalCol = state.ra.alloc_sub<uint32_t>();
                idxs[fi] = state.ra.alloc_sub<uint32_t>();

                mulConstant(1, logicalRow, state.tileXorRow, subGridM);
                add(1, logicalRow, logicalRow, uint16_t(sr));
                mulConstant(1, logicalCol, state.tileXorCol, subGridN);
                add(1, logicalCol, logicalCol, uint16_t(sc));
                emad(1, idxs[fi], logicalCol, logicalRow, state.inputs.tileCols, strategy, state);
                if (wrap || problem.case5TileXorSlm)
                    case5TileXorIndex(problem, strategy, state, idxs[fi], idxs[fi], slotMs, outputMs);
                else
                    case5TileXorIndex(problem, strategy, state, idxs[fi], idxs[fi], state.tileXorMs, outputMs);

                state.ra.safeRelease(logicalRow);
                state.ra.safeRelease(logicalCol);
            }
        }

        for (int fi = 0; fi < nFolds; fi++) {
            if (wrap || problem.case5TileXorSlm)
                storeWithWrap(idxs[fi], slotMs, state.tileXorFolds[fi], fi);
            else
                storeFoldAtIdx(idxs[fi], state.tileXorMs, state.tileXorFolds[fi], fi);
        }
        if (releaseSlotMs)
            state.ra.safeRelease(slotMs);
        for (int fi = 0; fi < nFolds; fi++)
            state.ra.safeRelease(idxs[fi]);
        }
    }

    if (incremental)
        case5ZeroFoldGrfs(strategy, state, nFolds);

    add(1, state.tileXorMs, state.tileXorMs, uint16_t(1));
    if (gateCount)
        mov(1, state.tileXorPanel, uint16_t(xorPeriod));

    if (gateStore) {
        mark(lSkipXor);
        state.ra.safeRelease(rem);
    }

    if (releaseFold)
        state.ra.safeRelease(fold);
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorFlushFromSlm(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                                 GEMMState &state)
{
    if (!problem.case5TileXor || !problem.case5TileXorSlm)
        return;
    if (!state.inputs.tileXor.isValid() && state.inputs.surfaceTileXor == InterfaceHandler::noSurface)
        return;
    if (case5NoStore())
        return;

    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);
    const int nFolds = split ? (subGridM * subGridN) : 1;
    const int maxMs = case5XorOutputMilestoneCount(problem);     // output slots
    const int slotMs = case5XorSlmSlotMs(problem);               // staged slots (>= maxMs)
    const bool wrapAtFlush = case5WrapAtFlush(problem);
    const bool useSurface = (state.inputs.surfaceTileXor != InterfaceHandler::noSurface);
    GRF wrapTmp = wrapAtFlush ? state.ra.alloc() : GRF();

    // Output slot ms = XOR of staged slots ms, ms + maxMs, ms + 2*maxMs, ...
    auto foldWrappedSlots = [&](int ms, int fi, const GRF &dst) {
        if (!wrapAtFlush)
            return;
        for (int s = ms + maxMs; s < slotMs; s += maxMs) {
            case5SlmMatrixLoad(problem, strategy, state, s, fi, wrapTmp);
            xor_(1, dst.ud(0), dst.ud(0), wrapTmp.ud(0));
        }
    };

    useTempAndR0(state, [&](GRF temp, GRF r0_info) {
        slmfence(temp, r0_info);
        fencewait();
        slmBarrier(temp, r0_info, strategy);
    });

    GRF fold = state.tileXorFold.isValid() ? state.tileXorFold : state.ra.alloc();
    const bool releaseFold = !state.tileXorFold.isValid();

    auto storeGlobalAtIdx = [&](const Subregister &idx, GRF foldReg) {
        if (useSurface) {
            auto hdr = state.ra.alloc().ud();
            shl(1, hdr, idx, uint16_t(2));
            store(1, surface_dword(ChannelMask::r), Surface(state.inputs.surfaceTileXor), hdr, foldReg);
            state.ra.safeRelease(hdr);
        } else if (state.inputs.tileXor.isValid()) {
            auto hdr = state.ra.alloc_range(2);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
            if (hw >= HW::Xe2)
                store(1, D32, A64, hdr, foldReg);
            else
                store(1, scattered_dword(), A64, hdr, foldReg);
            state.ra.safeRelease(byteOff);
            state.ra.safeRelease(hdr);
        }
    };

    if (!split) {
        auto idx = state.ra.alloc_sub<uint32_t>();
        auto msImm = state.ra.alloc_sub<uint32_t>();
        for (int ms = 0; ms < maxMs; ms++) {
            case5SlmMatrixLoad(problem, strategy, state, ms, 0, fold);
            foldWrappedSlots(ms, 0, fold);
            mov(1, msImm, uint32_t(ms));
            case5TileXorIndex(problem, strategy, state, idx, state.spatialId, msImm, maxMs);
            storeGlobalAtIdx(idx, fold);
        }
        state.ra.safeRelease(msImm);
        state.ra.safeRelease(idx);
    } else {
        std::vector<Subregister> idxs(nFolds);
        for (int sr = 0; sr < subGridM; sr++) {
            for (int sc = 0; sc < subGridN; sc++) {
                const int fi = sr * subGridN + sc;
                auto logicalRow = state.ra.alloc_sub<uint32_t>();
                auto logicalCol = state.ra.alloc_sub<uint32_t>();
                idxs[fi] = state.ra.alloc_sub<uint32_t>();

                mulConstant(1, logicalRow, state.tileXorRow, subGridM);
                add(1, logicalRow, logicalRow, uint16_t(sr));
                mulConstant(1, logicalCol, state.tileXorCol, subGridN);
                add(1, logicalCol, logicalCol, uint16_t(sc));
                emad(1, idxs[fi], logicalCol, logicalRow, state.inputs.tileCols, strategy, state);

                state.ra.safeRelease(logicalRow);
                state.ra.safeRelease(logicalCol);
            }
        }

        auto msImm = state.ra.alloc_sub<uint32_t>();
        for (int ms = 0; ms < maxMs; ms++) {
            mov(1, msImm, uint32_t(ms));
            for (int fi = 0; fi < nFolds; fi++) {
                case5SlmMatrixLoad(problem, strategy, state, ms, fi, state.tileXorFolds[fi]);
                foldWrappedSlots(ms, fi, state.tileXorFolds[fi]);
                auto idx = state.ra.alloc_sub<uint32_t>();
                case5TileXorIndex(problem, strategy, state, idx, idxs[fi], msImm, maxMs);
                storeGlobalAtIdx(idx, state.tileXorFolds[fi]);
                state.ra.safeRelease(idx);
            }
        }
        state.ra.safeRelease(msImm);

        for (int fi = 0; fi < nFolds; fi++)
            state.ra.safeRelease(idxs[fi]);
    }

    if (wrapTmp.isValid())
        state.ra.safeRelease(wrapTmp);
    if (releaseFold)
        state.ra.safeRelease(fold);
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorFlushFromGrf(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                                 GEMMState &state)
{
    if (!problem.case5TileXor || !problem.case5TileXorWrapGrf
        || problem.case5TileXorWrapGrfStoreMode != 2)
        return;
    if (!state.inputs.tileXor.isValid() && state.inputs.surfaceTileXor == InterfaceHandler::noSurface)
        return;

    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);
    const int nFolds = split ? (subGridM * subGridN) : 1;
    const int maxMs = case5XorOutputMilestoneCount(problem);
    const bool useSurface = (state.inputs.surfaceTileXor != InterfaceHandler::noSurface);

    GRF fold = state.tileXorFold.isValid() ? state.tileXorFold : state.ra.alloc();
    const bool releaseFold = !state.tileXorFold.isValid();

    Label labelSkipFlush;
    if (state.inputs.tileCount.isValid() && state.spatialId.isValid()) {
        cmp(1 | ge | f0[0], state.spatialId, state.inputs.tileCount);
        jmpi(1 | f0[0], labelSkipFlush);
    }

    // The slot is a compile-time constant here: read the subregister directly. (Going through
    // a0 cost an address-register write-to-use stall per slot.)
    const int dwPerGrf = elementsPerGRF(hw, DataType::ud);
    auto loadWrapGrfAtSlot = [&](int slot, int fi, const GRF &dst) {
        const GRF &w = state.tileXorWrapGrf[fi * kWrapGrfPerFold + slot / dwPerGrf];
        mov(1, dst.ud(0), w.ud(slot % dwPerGrf));
    };

    auto storeGlobalAtIdx = [&](const Subregister &idx, GRF foldReg) {
        if (useSurface) {
            auto hdr = state.ra.alloc().ud();
            shl(1, hdr, idx, uint16_t(2));
            store(1, surface_dword(ChannelMask::r), Surface(state.inputs.surfaceTileXor), hdr, foldReg);
            state.ra.safeRelease(hdr);
        } else if (state.inputs.tileXor.isValid()) {
            auto hdr = state.ra.alloc_range(2);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
            if (hw >= HW::Xe2)
                store(1, D32, A64, hdr, foldReg);
            else
                store(1, scattered_dword(), A64, hdr, foldReg);
            state.ra.safeRelease(byteOff);
            state.ra.safeRelease(hdr);
        }
    };

    if (!split) {
        auto idx = state.ra.alloc_sub<uint32_t>();
        auto msImm = state.ra.alloc_sub<uint32_t>();
        for (int ms = 0; ms < maxMs; ms++) {
            loadWrapGrfAtSlot(ms, 0, fold);
            mov(1, msImm, uint32_t(ms));
            case5TileXorIndex(problem, strategy, state, idx, state.spatialId, msImm, maxMs);
            storeGlobalAtIdx(idx, fold);
        }
        state.ra.safeRelease(msImm);
        state.ra.safeRelease(idx);
    } else {
        std::vector<Subregister> idxs(nFolds);
        for (int sr = 0; sr < subGridM; sr++) {
            for (int sc = 0; sc < subGridN; sc++) {
                const int fi = sr * subGridN + sc;
                auto logicalRow = state.ra.alloc_sub<uint32_t>();
                auto logicalCol = state.ra.alloc_sub<uint32_t>();
                idxs[fi] = state.ra.alloc_sub<uint32_t>();

                mulConstant(1, logicalRow, state.tileXorRow, subGridM);
                add(1, logicalRow, logicalRow, uint16_t(sr));
                mulConstant(1, logicalCol, state.tileXorCol, subGridN);
                add(1, logicalCol, logicalCol, uint16_t(sc));
                emad(1, idxs[fi], logicalCol, logicalRow, state.inputs.tileCols, strategy, state);

                state.ra.safeRelease(logicalRow);
                state.ra.safeRelease(logicalCol);
            }
        }

        auto msImm = state.ra.alloc_sub<uint32_t>();
        for (int ms = 0; ms < maxMs; ms++) {
            mov(1, msImm, uint32_t(ms));
            for (int fi = 0; fi < nFolds; fi++) {
                loadWrapGrfAtSlot(ms, fi, state.tileXorFolds[fi]);
                auto idx = state.ra.alloc_sub<uint32_t>();
                case5TileXorIndex(problem, strategy, state, idx, idxs[fi], msImm, maxMs);
                storeGlobalAtIdx(idx, state.tileXorFolds[fi]);
                state.ra.safeRelease(idx);
            }
        }
        state.ra.safeRelease(msImm);

        for (int fi = 0; fi < nFolds; fi++)
            state.ra.safeRelease(idxs[fi]);
    }

    mark(labelSkipFlush);
    if (releaseFold)
        state.ra.safeRelease(fold);
}

// ---- Interleaved XMX fold (problem.case5TileXorInterleaved) ------------------------------------
// On XMX all threads of an EU reach the milestone together, so a fold emitted after the last dpas
// of the milestone step idles the systolic pipe for its whole duration. With k-unroll == milestone
// (period 1) the milestone step is static, and the row passes of outerProductSystolic give a chain
// boundary after pass 0 (rows [0,osys) final). Those rows are folded there while the pass-1 dpas
// execute; the remaining rows stay intact until pass 1 of the NEXT step, so they are folded after
// pass 0 of that step and the milestone is stored with one 16-byte message. The last milestone is
// flushed after the k loop.

template <HW hw>
void Generator<hw>::case5FoldSubtileRows(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                         GEMMState &state, const GRF &foldReg, int sr, int sc,
                                         int rowBegin, int rowEnd, bool init)
{
    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subN = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];
    const int simd = elementsPerGRF(hw, DataType::ud);
    const int row0 = sr * subM + rowBegin, row1 = sr * subM + rowEnd;
    const int col0 = sc * subN, col1 = col0 + subN;
    bool first = init;

    // Fast path (colMajor C): the rows [row, row + simd) of one column are one GRF. On 32-byte GRFs
    // (Gen12LP..XeHPG) two adjacent columns of a dpas block sit in consecutive GRFs, so a SIMD16
    // operand covers both; on 64-byte GRFs (Xe2: 16 dwords) one column is already a 16-lane operand
    // and the units stay single GRFs. bfn 0x96 (s0^s1^s2) folds two equal-width units per
    // instruction into a 16-lane accumulator (two GRFs on XeHPG, one on Xe2), which is then folded
    // into foldReg. Roughly 4x fewer instructions than the SIMD8 chain. CASE5_XOR_SIMD16=0 /
    // CASE5_XOR_BFN=0 disable the parts.
    const char *s16Env = std::getenv("CASE5_XOR_SIMD16");
    const char *bfnEnv = std::getenv("CASE5_XOR_BFN");
    const bool grf32 = (GRF::bytes(hw) == 32);
    const int accLanes = grf32 ? 2 * simd : simd;   // 16 on both: the accumulator lanes in use
    int wideIdx = -1;
    for (size_t i = 0; i < state.tileXorFoldWide.size(); i++)
        if (state.tileXorFoldWide[i].isValid() && state.tileXorFoldWide[i][0].getBase() == foldReg.getBase())
            wideIdx = int(i);
    const bool useWide = (wideIdx >= 0) && state.C_layout.colMajor() && !(s16Env && s16Env[0] == '0');
    const bool useS16 = state.C_layout.colMajor() && state.tileXorPairTmp.isValid()
            && !(s16Env && s16Env[0] == '0');
    const bool useBfn = !(bfnEnv && bfnEnv[0] == '0');

    struct Unit { Subregister reg; int exec; };
    // The column operands of one row chunk, in column order.
    auto rowUnits = [&](int row) {
        std::vector<Unit> units;
        for (int col = col0; col < col1; ) {
            Subregister c0 = state.C_layout.find(row, col, state.C_regs[0]);
            bool pair = false;
            if (grf32 && col + 1 < col1) {
                Subregister c1 = state.C_layout.find(row, col + 1, state.C_regs[0]);
                pair = (c1.getBase() == c0.getBase() + 1) && c0.getOffset() == 0 && c1.getOffset() == 0;
            }
            units.push_back({c0, pair ? 2 * simd : simd});
            col += pair ? 2 : 1;
        }
        return units;
    };
    // Xe2 (64-byte GRFs) wide-accumulator variants, both needing a 2-GRF accumulator per sub-tile:
    //  CASE5_XOR_SIMD32=1: four consecutive C GRFs per bfn (exec 32, 32-lane accumulator), half the
    //                      fold instructions;
    //  CASE5_XOR_ACC2=1:   two independent 16-lane chains per sub-tile (even / odd GRF pairs), twice
    //                      the independent chains for the round-robin emission.
    // case5TileXorGather folds the second GRF into the first before the packed reduction.
    const int accGrfs = useWide ? state.tileXorFoldWide[wideIdx].getLen() : 1;
    const bool s32 = useWide && !grf32 && accGrfs == 2 && useBfn && case5EnvOn("CASE5_XOR_SIMD32", false);
    const bool acc2 = useWide && !grf32 && accGrfs == 2 && !s32 && case5EnvOn("CASE5_XOR_ACC2", false);
    // Emission goes through op(): recorded while g_c5FoldRecord (round-robin with the other chains in
    // case5FoldEmitPending), else emitted in place.
    auto op = [&](int kind, int e, const RegData &dst, const RegData &a, const RegData &b) {
        if (g_c5FoldRecord)
            g_c5FoldOps.push_back({kind, e, g_c5FoldChain, dst, a, b});
        else
            case5FoldEmitOp(kind, e, dst, a, b);
    };
    const RegData none;
    // XOR the units into lanes [0, accLanes) of acc. init: acc holds nothing yet; a unit narrower
    // than the accumulator zeroes it first. Equal-width GRF-aligned neighbours go two per bfn.
    auto accumulate = [&](const std::vector<Unit> &units, const GRF &acc, bool &init) {
        const int lanes = s32 ? 2 * accLanes : accLanes;
        size_t i = 0;
        while (i < units.size()) {
            const int e = units[i].exec;
            const bool sameNext = (i + 1 < units.size()) && units[i + 1].exec == e;
            const bool aligned = units[i].reg.getOffset() == 0
                    && (!sameNext || units[i + 1].reg.getOffset() == 0);
            if (s32 && e == simd && i + 3 < units.size()) {
                bool quad = true;
                for (int q = 0; q < 4; q++)
                    quad = quad && units[i + q].exec == simd && units[i + q].reg.getOffset() == 0
                            && units[i + q].reg.getBase() == units[i].reg.getBase() + q;
                if (quad) {
                    RegData d = acc.ud(0)(1), a = units[i].reg.ud(0)(1), b = units[i + 2].reg.ud(0)(1);
                    if (init) {
                        op(1, 2 * simd, d, a, b);
                        init = false;
                    } else
                        op(2, 2 * simd, d, a, b);
                    i += 4;
                    continue;
                }
            }
            if (init && e < lanes) {
                op(4, lanes, acc.ud(0)(1), none, none);
                init = false;
            }
            if (init) {
                if (sameNext) {
                    op(1, e, acc.ud(), units[i].reg.ud(0)(1), units[i + 1].reg.ud(0)(1));
                    i += 2;
                } else {
                    op(0, e, acc.ud(), units[i].reg.ud(0)(1), none);
                    i += 1;
                }
                init = false;
            } else if (useBfn && sameNext && aligned) {
                op(2, e, acc.ud(), units[i].reg.ud(0)(1), units[i + 1].reg.ud(0)(1));
                i += 2;
            } else {
                op(3, e, acc.ud(), units[i].reg.ud(0)(1), none);
                i += 1;
            }
        }
    };

    if (useWide) {
        // Wide path: the accumulator A takes the units directly (lanes [0, 16): both GRFs of A on
        // XeHPG, A[0] on Xe2); the reduction to a scalar happens once per milestone in
        // case5TileXorGather.
        auto A = state.tileXorFoldWide[wideIdx];
        const int chainA = g_c5FoldChainNext++;
        const int chainB = acc2 ? g_c5FoldChainNext++ : chainA;
        bool init = first, initB = first;
        for (int row = row0; row < row1; row += simd) {
            auto units = rowUnits(row);
            if (acc2) {
                // Even GRF pairs (0,1 / 4,5 / ...) to A[0], odd pairs (2,3 / 6,7 / ...) to A[1].
                std::vector<Unit> ua, ub;
                for (size_t i = 0; i < units.size(); i++)
                    (((i / 2) % 2) ? ub : ua).push_back(units[i]);
                g_c5FoldChain = chainA;
                accumulate(ua, A[0], init);
                g_c5FoldChain = chainB;
                accumulate(ub, A[1], initB);
            } else {
                g_c5FoldChain = chainA;
                accumulate(units, A[0], init);
            }
        }
        g_c5FoldChain = chainA;
        if (init) op(4, s32 ? 2 * accLanes : accLanes, A[0].ud(0)(1), none, none);
        if (acc2 && initB) {
            g_c5FoldChain = chainB;
            op(4, accLanes, A[1].ud(0)(1), none, none);
        }
        return;
    }
    if (useS16) {
        auto F = state.tileXorPairTmp;
        for (int row = row0; row < row1; row += simd) {
            // Accumulate the row chunk into the 16 lanes of F (F[0] and F[1] on XeHPG, F[0] on Xe2),
            // then fold them into foldReg (simd lanes).
            bool fInit = true;
            accumulate(rowUnits(row), F[0], fInit);
            if (fInit) continue;
            if (first) {
                if (grf32)
                    xor_(simd, foldReg.ud(), F[0].ud(), F[1].ud());
                else
                    mov(simd, foldReg.ud(), F[0].ud());
                first = false;
            } else if (grf32 && useBfn) {
                bfn(simd, 0x96, foldReg.ud(), foldReg.ud(), F[0].ud(), F[1].ud());
            } else {
                xor_(simd, foldReg.ud(), foldReg.ud(), F[0].ud());
                if (grf32) xor_(simd, foldReg.ud(), foldReg.ud(), F[1].ud());
            }
        }
        if (first) mov(simd, foldReg.ud(), uint16_t(0));
        return;
    }

    auto emit = [&](const Subregister &C0) {
        if (first) {
            mov(simd, foldReg.ud(), C0.ud(0)(1));
            first = false;
        } else
            xor_(simd, foldReg.ud(), foldReg.ud(), C0.ud(0)(1));
    };
    if (state.C_layout.colMajor()) {
        for (int col = col0; col < col1; col++)
            for (int row = row0; row < row1; row += simd)
                emit(state.C_layout.find(row, col, state.C_regs[0]));
    } else {
        for (int row = row0; row < row1; row++)
            for (int col = col0; col < col1; col += simd)
                emit(state.C_layout.find(row, col, state.C_regs[0]));
    }
}

template <HW hw>
void Generator<hw>::case5FoldEmitOp(int kind, int exec, const RegData &dst, const RegData &a, const RegData &b)
{
    switch (kind) {
        case 0: mov(exec, dst, a); break;
        case 1: xor_(exec, dst, a, b); break;
        case 2: bfn(exec, 0x96, dst, dst, a, b); break;
        case 3: xor_(exec, dst, dst, a); break;
        default: mov(exec, dst, uint16_t(0)); break;
    }
}

// Emit the recorded fold ops round-robin over their chains (op k of every chain, then op k+1, ...),
// keeping each chain's own order. Clears the recording.
template <HW hw>
void Generator<hw>::case5FoldEmitPending()
{
    g_c5FoldRecord = false;
    std::vector<int> ids;
    for (auto &o : g_c5FoldOps)
        if (std::find(ids.begin(), ids.end(), o.chain) == ids.end()) ids.push_back(o.chain);
    std::vector<std::vector<const Case5FoldOp *>> chains(ids.size());
    for (auto &o : g_c5FoldOps)
        chains[std::find(ids.begin(), ids.end(), o.chain) - ids.begin()].push_back(&o);
    size_t maxLen = 0;
    for (auto &c : chains) maxLen = std::max(maxLen, c.size());
    for (size_t k = 0; k < maxLen; k++)
        for (auto &c : chains)
            if (k < c.size()) case5FoldEmitOp(c[k]->kind, c[k]->exec, c[k]->dst, c[k]->a, c[k]->b);
    g_c5FoldOps.clear();
    g_c5FoldChainNext = 0;
}

template <HW hw>
bool Generator<hw>::case5TileXorFastStoreOk(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                            GEMMState &state)
{
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const int nFolds = subGridM * subGridN;
    if (nFolds < 2) return false;
    if (problem.case5XorLayout != 2 && (subGridM != 1 || (nFolds != 4 && nFolds != 8))) return false;
    if ((problem.case5TileXorWrap || problem.case5TileXorWrapGrf) && !problem.case5TileXorInterleaved) return false;
    if (problem.case5TileXorSlm) return false;
    if (state.inputs.surfaceTileXor != InterfaceHandler::noSurface) return false;
    if (!state.inputs.tileXor.isValid() && !problem.case5TileXorBlake3) return false;
    return state.tileXorStoreHdr.isValid() && state.tileXorStoreData.isValid()
            && state.tileXorCountBytes.isValid();
}

// Gather the per-sub-tile fold scalars (dword 0 of each fold GRF) and store them with one
// oword block message at the running milestone pointer, then advance it by tile_count * 4.
template <HW hw>
void Generator<hw>::case5TileXorFastStore(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                          GEMMState &state, bool reduce, bool msIncremented)
{
    case5TileXorGather(problem, strategy, state, reduce, msIncremented);
    case5TileXorSend(problem, strategy, state, msIncremented);
}

template <HW hw>
void Generator<hw>::case5TileXorGather(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                       GEMMState &state, bool reduce, bool msIncremented)
{
    (void)msIncremented;   // the line buffer rotates once per milestone regardless of the counter state
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const int nFolds = subGridM * subGridN;   // scalars in grid order (sr * subGridN + sc)
    const int simd = elementsPerGRF(hw, DataType::ud);
    auto &data = state.tileXorStoreData;
    // Packed reduction: 4 or 8 folds on 32-byte GRFs (two GRFs per accumulator), any 2..8 on Xe2.
    const bool packed = reduce && !state.tileXorFoldWide.empty() && int(state.tileXorFoldWide.size()) == nFolds
            && ((simd == 8 && (nFolds == 4 || nFolds == 8)) || (simd == 16 && nFolds >= 2 && nFolds <= 8));
    if (packed && simd == 8) {
        // Packed reduction of n 16-lane accumulators W[i] to n scalars in data.ud(0..n-1), reusing
        // the accumulator GRFs as scratch (every GRF written was already consumed):
        //   16 -> 8 lanes: slot i = W[i][0] ^ W[i][1]            (n SIMD8, lands in W[0..n/2))
        //    8 -> 4:       per range r: (8;4,1) ^ (8;4,1)+4      (n/2 SIMD8, lands in W[n/2..3n/4))
        //    4 -> 2:       per range q: (4;2,1) ^ (4;2,1)+2      (n/4 SIMD8, lands in W[3n/4..))
        //    2 -> 1:       (2;1,0) ^ (2;1,0)+1 -> data           (1 SIMDn)
        auto &W = state.tileXorFoldWide;
        const int n = nFolds;
        for (int i = 0; i < n; i++)
            xor_(8, W[i / 2][i % 2].ud(), W[i][0].ud(), W[i][1].ud());
        for (int r = 0; r < n / 2; r++)
            xor_(8, W[n / 2 + r / 2][r % 2].ud(), W[r][0].ud(0)(8, 4, 1), W[r][0].ud(4)(8, 4, 1));
        for (int q = 0; q < n / 4; q++)
            xor_(8, W[n / 2 + n / 4 + q / 2][q % 2].ud(), W[n / 2 + q][0].ud(0)(4, 2, 1),
                 W[n / 2 + q][0].ud(2)(4, 2, 1));
        xor_(n, data.ud(0)(1), W[n / 2 + n / 4][0].ud(0)(2, 1, 0), W[n / 2 + n / 4][0].ud(1)(2, 1, 0));
    } else if (packed && simd == 16) {
        // 64-byte GRFs (Xe2): each accumulator is the one 16-lane GRF W[i][0]. The same scheme, the
        // first step folding the two halves of a GRF, each step packing two folds per GRF; any n in
        // 2..8 (c1 = ceil(n/2), c2 = ceil(n/4) GRFs after steps 1 and 2; every write lands in a GRF the
        // previous step consumed; lanes past n hold garbage that nothing stores):
        //   16 -> 8: W[i/2] half i%2 = W[i] lanes [0,8) ^ [8,16)   (n SIMD8, lands in W[0..c1))
        //    8 -> 4: per GRF r: (8;4,1) ^ (8;4,1)+4                 (c1 SIMD8, lands in W[c1..c1+c2))
        //    4 -> 2: per GRF q: (4;2,1) ^ (4;2,1)+2                 (c2 SIMD8, lands in W[0])
        //    2 -> 1: (2;1,0) ^ (2;1,0)+1 -> data                    (1 SIMD np, np = pow2 >= n)
        auto &W = state.tileXorFoldWide;
        const int n = nFolds;
        const int c1 = (n + 1) / 2, c2 = (n + 3) / 4;
        int np = 2;
        while (np < n) np *= 2;
        if (W[0].getLen() == 2)   // 2-GRF accumulators (CASE5_XOR_SIMD32 / CASE5_XOR_ACC2): fold the halves
            for (int i = 0; i < n; i++)
                xor_(16, W[i][0].ud(), W[i][0].ud(), W[i][1].ud());
        for (int i = 0; i < n; i++)
            xor_(8, W[i / 2][0].ud(8 * (i % 2))(1), W[i][0].ud(0)(1), W[i][0].ud(8)(1));
        for (int r = 0; r < c1; r++)
            xor_(8, W[c1 + r / 2][0].ud(8 * (r % 2))(1), W[r][0].ud(0)(8, 4, 1), W[r][0].ud(4)(8, 4, 1));
        for (int q = 0; q < c2; q++)
            xor_(8, W[q / 2][0].ud(8 * (q % 2))(1), W[c1 + q][0].ud(0)(4, 2, 1), W[c1 + q][0].ud(2)(4, 2, 1));
        xor_(np, data.ud(0)(1), W[0][0].ud(0)(2, 1, 0), W[0][0].ud(1)(2, 1, 0));
    } else {
        for (int fi = 0; fi < nFolds; fi++) {
            GRF f = state.tileXorFolds[fi];
            if (reduce)
                for (int n = simd >> 1; n >= 1; n >>= 1)
                    xor_(n, f.ud(0)(1), f.ud(0)(1), f.ud(n)(1));
            mov(1, data.ud(fi), f.ud(0));
        }
    }
    if (problem.case5XorLayout == 2 && state.tileXorLineBuf.isValid()) {
        // Rotate each sub-tile line down by one dword and append this milestone's scalar. The
        // SIMD8 moves read one dword past their GRF (the next line, or the pad GRF for the last).
        // The new scalar goes to dword B-1, so after B milestones dwords 0..B-1 hold them in order.
        const int B = state.tileXorLineBlock;
        const int grfPerLine = std::max(1, B * 4 / GRF::bytes(hw));
        if (GRF::bytes(hw) == 64 && case5EnvOn("CASE5_XOR_LINE_IDX", true) && B > 1 && grfPerLine == 1
                && nFolds <= 8) {
            // Indirect slot write instead of rotating every line by one dword: a0.8..15 hold the line
            // byte offsets 0, 64, .., 448 (set per tile in gemmCase5TileXorInit), a0.fi = a0.(8+fi) +
            // line base + slot*4, one mov per sub-tile. Slot = index of the gathered milestone within
            // its block: tileXorMs - 1 when the counter was already incremented, else tileXorMs.
            auto t = state.ra.alloc_sub<uint32_t>();
            if (msIncremented) {
                add(1, t, state.tileXorMs, uint16_t(B - 1));
                and_(1, t, t, uint16_t(B - 1));
            } else
                and_(1, t, state.tileXorMs, uint16_t(B - 1));
            shl(1, t, t, uint16_t(2));
            add(1, t, t, uint32_t(state.tileXorLineBuf.getBase() * GRF::bytes(hw)));
            add(8, a0[0](1), a0[8](8, 8, 1), t.uw(0)(0, 1, 0));
            state.ra.safeRelease(t);
            for (int fi = 0; fi < nFolds; fi++)
                mov(1, indirect[a0[fi]].ud(0), data.ud(fi));
            return;
        }
        for (int fi = 0; fi < nFolds; fi++) {
            GRF line = state.tileXorLineBuf[fi * grfPerLine];
            if (B > 1)
                for (int g = 0; g < grfPerLine; g++)
                    mov(simd, GRF(line.getBase() + g).ud(0)(1), GRF(line.getBase() + g).ud(1)(1));
            mov(1, GRF(line.getBase() + (B - 1) / simd).ud((B - 1) % simd), data.ud(fi));
        }
    }
}

template <HW hw>
void Generator<hw>::case5TileXorSend(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                     GEMMState &state, bool msIncremented)
{
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const int nFolds = subGridM * subGridN;
    const char *sendEnv = std::getenv("CASE5_XOR_SEND");
    if (sendEnv && sendEnv[0] == 48) return;   // debug: measure fold cost without the store
    if (problem.case5XorLayout == 2 && state.tileXorLineBuf.isValid()) {
        // Flush the line buffers every B milestones. The stored milestone is tileXorMs when the
        // counter has not been incremented yet (end-of-body action), else tileXorMs - 1.
        const int B = state.tileXorLineBlock;
        const int grfPerLine = std::max(1, B * 4 / GRF::bytes(hw));
        Label lSkip;
        auto t = state.ra.alloc_sub<uint32_t>();
        and_(1, t, state.tileXorMs, uint16_t(B - 1));
        cmp(1 | ne | state.flagCase5Xor, t, uint16_t(msIncremented ? 0 : (B - 1)));
        state.ra.safeRelease(t);
        jmpi(1 | state.flagCase5Xor, lSkip);
        if (problem.case5TileXorAccSlm && state.tileXorAccSlmOff.isValid()) {
            // Case 5.7: the completed block (milestones 16b..16b+15 of every sub-tile) goes to SLM as it
            // is; the fold into the message happens after the k loop. The fast-store header carries the
            // SLM offset: nothing else uses it inside the loop on the fold16 / wrap paths.
            auto blk = state.ra.alloc_sub<uint32_t>();
            if (msIncremented)
                add(1, blk, state.tileXorMs, -int32_t(1));
            else
                mov(1, blk, state.tileXorMs);
            shr(1, blk, blk, uint16_t(4));                      // block index (B == 16)
            mulConstant(1, blk, blk, nFolds * 64);
            add(1, blk, blk, state.tileXorAccSlmOff);
            GRFRange hdr(state.tileXorStoreHdr.getBase(), 1);
            for (int fi = 0; fi < nFolds; fi++) {
                if (fi == 0)
                    mov(1, hdr[0].ud(0), blk);
                else
                    add(1, hdr[0].ud(0), blk, uint32_t(fi * 64));
                store(1, block(D32, 16), SLM, hdr,
                      GRFRange(state.tileXorLineBuf.getBase() + fi * grfPerLine, grfPerLine));
            }
            state.ra.safeRelease(blk);
            mark(lSkip);
            return;
        }
        if (state.tileXorFoldAcc.isValid()) {
            // The line holds milestones 16b..16b+15 in slot order. Case 5.5a: the protocol fold
            // msg[w] = rotl(msg[w], 13) ^ x[16b + w], one rotate and one xor per 16 lanes. Wrap cases:
            // slot[w] = x[w] ^ x[w + 16], the xor alone (two blocks).
            for (int fi = 0; fi < nFolds; fi++) {
                GRF acc = state.tileXorFoldAcc[fi * grfPerLine];
                GRF line = state.tileXorLineBuf[fi * grfPerLine];
                if (problem.case5TileXorFold16)
                    rol(16, acc.ud(), acc.ud(), uint16_t(13));
                xor_(16, acc.ud(), acc.ud(), line.ud());
            }
            mark(lSkip);
            return;
        }
        const char *cc = std::getenv("CASE5_XOR_STORE_CACHE");
        for (int fi = 0; fi < nFolds; fi++) {
            GRFRange line(state.tileXorLineBuf.getBase() + fi * grfPerLine, grfPerLine);
            auto spec = block(D32, B);
            if (cc && !std::strcmp(cc, "s")) spec = spec | CacheSettingsLSC::L1S_L3UC;
            else if (cc && !std::strcmp(cc, "uc")) spec = spec | CacheSettingsLSC::L1UC_L3UC;
            else if (cc && !std::strcmp(cc, "wt")) spec = spec | CacheSettingsLSC::L1WT_L3UC;
            else if (cc && !std::strcmp(cc, "wb")) spec = spec | CacheSettingsLSC::L1WT_L3WB;
            if (state.tileXorLineHdrPacked) {
                auto q = state.tileXorLineHdr[0][0].uq(fi);
                mov(1, state.tileXorStoreHdr[0].uq(0), q);
                store(1, spec, A64, state.tileXorStoreHdr, line);
                eadd(1, q, q, state.tileXorCountBytes, strategy, state);
            } else {
                store(1, spec, A64, state.tileXorLineHdr[fi], line);
                eadd(1, state.tileXorLineHdr[fi][0].uq(0), state.tileXorLineHdr[fi][0].uq(0),
                     state.tileXorCountBytes, strategy, state);
            }
        }
        mark(lSkip);
        return;
    }
    // debug: CASE5_XOR_SENDEVERY=n issues the store only when (ms % n) == n-1 (upper bound for batching)
    Label lSkipEvery;
    const char *everyEnv = std::getenv("CASE5_XOR_SENDEVERY");
    const int every = everyEnv ? std::atoi(everyEnv) : 1;
    if (every > 1 && state.flagCase5Xor.isValid()) {
        auto t = state.ra.alloc_sub<uint32_t>();
        and_(1, t, state.tileXorMs, uint16_t(every - 1));
        cmp(1 | ne | state.flagCase5Xor, t, uint16_t(every - 1));
        state.ra.safeRelease(t);
        jmpi(1 | state.flagCase5Xor, lSkipEvery);
    }
    const bool hdrCopy = state.tileXorStoreBase.isValid();
    if (hdrCopy)
        mov(2, state.tileXorStoreHdr[0].ud(0)(1), state.tileXorStoreBase.ud(0)(1));
    const char *lscEnv = std::getenv("CASE5_XOR_LSC");
    if (lscEnv && lscEnv[0] == '0' && hw < HW::Xe2)   // legacy oword block store (debug); Xe2 is LSC only
        store(16, block_oword(nFolds / 4), A64, state.tileXorStoreHdr, state.tileXorStoreData);
    else {
        // CASE5_XOR_STORE_CACHE=uc|wt|wb selects the LSC cache policy (default: hardware default).
        const char *cc = std::getenv("CASE5_XOR_STORE_CACHE");
        auto spec = block(D32, nFolds);
        // Default: hardware default (write-back in L2). L1S_L3UC is 0.4% better at K=4096 but 3%
        // worse at K=8192 where the write stream competes with A/B panels for DRAM.
        if (cc && !std::strcmp(cc, "s")) spec = spec | CacheSettingsLSC::L1S_L3UC;
        else if (cc && !std::strcmp(cc, "uc")) spec = spec | CacheSettingsLSC::L1UC_L3UC;
        else if (cc && !std::strcmp(cc, "wt")) spec = spec | CacheSettingsLSC::L1WT_L3UC;
        else if (cc && !std::strcmp(cc, "wb")) spec = spec | CacheSettingsLSC::L1WT_L3WB;
        store(1, spec, A64, state.tileXorStoreHdr, state.tileXorStoreData);
    }
    // debug: CASE5_XOR_SENDSAME=1 keeps writing the same 16 bytes (isolates the memory footprint)
    const char *same = std::getenv("CASE5_XOR_SENDSAME");
    if (same && same[0] == 49) return;
    if (hdrCopy)
        eadd(1, state.tileXorStoreBase.uq(0), state.tileXorStoreBase.uq(0), state.tileXorCountBytes,
             strategy, state);
    else
        eadd(1, state.tileXorStoreHdr[0].uq(0), state.tileXorStoreHdr[0].uq(0), state.tileXorCountBytes,
             strategy, state);
    if (every > 1) mark(lSkipEvery);
}

// tile_xor dword index of (spatial id sid, milestone/slot index slot) for problem.case5XorLayout:
// 0 = [slot][tile] (sid + slot * tile_count), 2 = [tile][slot] (sid * slots + slot). dst may alias sid.
template <HW hw>
void Generator<hw>::case5TileXorIndex(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                      const Subregister &dst, const Subregister &sid, const Subregister &slot,
                                      int slots)
{
    if (problem.case5XorLayout == 2) {
        mulConstant(1, dst, sid, std::max(1, slots));
        add(1, dst, dst, slot);
    } else
        emad(1, dst, sid, slot, state.inputs.tileCount, strategy, state);
}

// Phase test without a long-lived flag: flagCase5Xor is set transiently from the GRF phase value.
template <HW hw>
void Generator<hw>::case5JumpIfPhase(GEMMState &state, bool phaseB, Label &target)
{
    // Static phases: jump unconditionally when the block is for the other phase, else never.
    if (state.tileXorPhaseMode == 0 || state.tileXorPhaseMode == 1) {
        const bool allB = (state.tileXorPhaseMode == 1);
        if (phaseB == allB) jmpi(1, target);
        return;
    }
    if (phaseB)
        cmp(1 | ne | state.flagCase5Xor, state.tileXorPhase, uint16_t(0));
    else
        cmp(1 | eq | state.flagCase5Xor, state.tileXorPhase, uint16_t(0));
    jmpi(1 | state.flagCase5Xor, target);
}

// Which threads fold where. Phase A (flag clear): rows [0,osys) after pass 0 of the milestone step,
// the rest after pass 0 of the next step (or the final flush). Phase B (flag set): everything at the
// end of the milestone step. Half of the co-resident threads in each group keeps the systolic pipe
// busy with their dpas chains while the other half issues its fold. CASE5_XOR_PHASE selects the
// parity source: lidm (default), lidn, lidm2, tile, all-a, all-b.
template <HW hw>
void Generator<hw>::case5InitTileXorPhase(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                          GEMMState &state, const Subregister &idM, const Subregister &idN)
{
    if (!state.tileXorPhase.isValid()) return;
    const char *mode = std::getenv("CASE5_XOR_PHASE");
    // Default: every thread on phase A (verified, 192 TOPS at 32768x16384x4096). Phase B (fold at the
    // step end) verifies at small sizes but hangs the GPU at full size (TDR) for lidn/lidm2/tile/all-b;
    // it stays selectable for debugging only.
    std::string m = mode ? mode : "all-a";
    state.tileXorPhaseMode = (m == "all-a") ? 0 : (m == "all-b") ? 1 : 2;
    if (m != "all-a")
        std::fprintf(stderr, "Case5: CASE5_XOR_PHASE=%s is experimental (phase-B threads hung the GPU at full size)\n", m.c_str());
    auto tmp = state.ra.alloc_sub<uint32_t>();
    if (m == "all-b") {
        mov(1, tmp, uint16_t(1));
    } else if (m == "all-a") {
        mov(1, tmp, uint16_t(0));
    } else if (m == "lidn" && state.lidN.isValid()) {
        mov(1, tmp, state.lidN);
    } else if (m == "lidm2" && state.lidM.isValid()) {
        shr(1, tmp, state.lidM, uint16_t(1));
    } else if (m == "tile") {
        add(1, tmp, idM, idN);
    } else if (state.lidM.isValid()) {
        mov(1, tmp, state.lidM);
    } else {
        mov(1, tmp, uint16_t(0));
    }
    and_(1, state.tileXorPhase, tmp, uint16_t(1));
    state.ra.safeRelease(tmp);
}

// Whole sub-tile fold into foldReg (dword-wise, not yet reduced). On 32-byte GRFs it uses SIMD16
// over the two row halves of a column when their GRFs are adjacent (colMajor C, subM == 2*simd),
// else the per-row-chunk fold (which on 64-byte GRFs already covers 16 rows per operand).
template <HW hw>
void Generator<hw>::case5FoldSubtileFull(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                         GEMMState &state, const GRF &foldReg, int sr, int sc)
{
    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subN = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];
    const int simd = elementsPerGRF(hw, DataType::ud);
    const char *pairEnv = std::getenv("CASE5_XOR_PAIR");
    const bool pairs = GRF::bytes(hw) == 32 && state.C_layout.colMajor() && subM == 2 * simd
            && state.tileXorPairTmp.isValid() && !(pairEnv && pairEnv[0] == 48);
    if (!pairs) {
        case5FoldSubtileRows(problem, strategy, state, foldReg, sr, sc, 0, subM, true);
        return;
    }
    auto F = state.tileXorPairTmp;
    const int row0 = sr * subM, col0 = sc * subN;
    bool first = true;
    for (int col = col0; col < col0 + subN; col++) {
        Subregister c0 = state.C_layout.find(row0, col, state.C_regs[0]);
        Subregister c1 = state.C_layout.find(row0 + simd, col, state.C_regs[0]);
        const bool adjacent = (c1.getBase() == c0.getBase() + 1) && c0.getOffset() == 0 && c1.getOffset() == 0;
        if (adjacent) {
            if (first) mov(16, F[0].ud(), c0.ud(0)(1));
            else xor_(16, F[0].ud(), F[0].ud(), c0.ud(0)(1));
        } else {
            if (first) {
                mov(simd, F[0].ud(), c0.ud(0)(1));
                mov(simd, F[1].ud(), c1.ud(0)(1));
            } else {
                xor_(simd, F[0].ud(), F[0].ud(), c0.ud(0)(1));
                xor_(simd, F[1].ud(), F[1].ud(), c1.ud(0)(1));
            }
        }
        first = false;
    }
    xor_(simd, foldReg.ud(), F[0].ud(), F[1].ud());
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorChainHook(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                              GEMMState &state, int h, int opCount, int x, int nx,
                                              int osys, bool globalCM, bool preChain, bool xafStep)
{
    if (!problem.case5TileXor || !problem.case5TileXorInterleaved || problem.case5TileXorNop) return;
    if (!globalCM || state.C_regs.empty() || state.C_regs[0].empty()) return;
    const int unrollK = strategy.unroll[LoopK];
    const bool lastStep = ((h + opCount) % unrollK) == 0;
    const bool firstStep = (h % unrollK) == 0;
    const bool firstPass = (x == 0);
    const bool lastPass = (x + osys >= nx);
    const bool singlePass = (nx <= osys);
    // With xaf the step is one atomic chain: no pass-0 boundary, so the upper rows are folded at the
    // end of the milestone step and the lower rows right before the next step's chain (preChain).
    // (xafStep is false for the boundary steps of the hybrid mode, which behave as with af.)
    const bool xaf = xafStep;
    // Round-robin emission of the sub-tile chains (CASE5_XOR_RR, default on).
    const bool rr = case5EnvOn("CASE5_XOR_RR", true);
    auto rrBegin = [&]() { if (rr) { g_c5FoldRecord = true; g_c5FoldOps.clear(); g_c5FoldChainNext = 0; } };
    auto rrEnd = [&]() { if (rr) case5FoldEmitPending(); };
    // CASE5_XOR_TOPAT=prechain (af only): the upper rows [0, osys), final since pass 0 of the last
    // step, are folded at the start of the next milestone's first step, before its first chain
    // (right after the barrier, while the step's loads are in flight), instead of after pass 0 of
    // the last step. The final flush then folds all rows.
    const char *topAtEnv = std::getenv("CASE5_XOR_TOPAT");
    const bool topPrechain = topAtEnv && !std::strcmp(topAtEnv, "prechain") && !strategy.extendedAtomicFMA
            && !singlePass && state.tileXorPhaseMode == 0;
    // Debug: CASE5_XOR_FOLD=0 drops the fold work, CASE5_XOR_GATHER=0 the reduction / line / store
    // (both break the result; they attribute the cost).
    const bool dbgFold = case5EnvOn("CASE5_XOR_FOLD", true), dbgGather = case5EnvOn("CASE5_XOR_GATHER", true);
    if (preChain) {
        if (!firstStep || singlePass) return;
        if (!xaf && !topPrechain) return;
        const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
        const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
        const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
        Label lSkip;
        cmp(1 | eq | state.flagCase5Xor, state.tileXorMs, uint16_t(0));
        jmpi(1 | state.flagCase5Xor, lSkip);
        if (!xaf) {
            if (dbgFold) {
                rrBegin();
                for (int sr = 0; sr < subGridM; sr++) {
                    const int re = std::min(subM, osys - sr * subM);
                    if (re <= 0) continue;
                    for (int sc = 0; sc < subGridN; sc++)
                        case5FoldSubtileRows(problem, strategy, state, state.tileXorFolds[sr * subGridN + sc], sr, sc,
                                             0, re, true);
                }
                rrEnd();
            }
            mark(lSkip);
            return;
        }
        rrBegin();
        for (int sr = 0; sr < subGridM; sr++) {
            const int rb = std::max(0, osys - sr * subM), re = subM;
            if (re <= rb) continue;
            const bool init = (sr * subM >= osys);
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, state.tileXorFolds[sr * subGridN + sc], sr, sc, rb, re,
                                     init);
        }
        rrEnd();
        case5TileXorGather(problem, strategy, state, true, true);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
        return;
    }
    // CASE5_XOR_GATHER_AT=step1: the reduction / line update / store move from the pass-0 boundary of
    // the first step (where the bottom-rows fold sits) to the pass-0 boundary of the second step, so
    // each cluster runs under its own freshly issued dpas pass. Needs per-pass chains in the middle
    // steps (af, not xaf), at least 3 steps per milestone and phase-A threads.
    const char *gaEnv = std::getenv("CASE5_XOR_GATHER_AT");
    const bool secondStep = ((h % unrollK) == opCount);
    const bool gatherStep1 = gaEnv && !std::strcmp(gaEnv, "step1") && !strategy.extendedAtomicFMA
            && !singlePass && state.tileXorPhaseMode == 0 && unrollK >= 3 * opCount;
    if (!lastStep && !firstStep && !(secondStep && gatherStep1)) return;
    if (!firstPass && !lastPass) return;

    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    auto foldAt = [&](int sr, int sc) { return state.tileXorFolds[sr * subGridN + sc]; };

    if (secondStep && firstPass && gatherStep1) {
        Label lSkip;
        cmp(1 | eq | state.flagCase5Xor, state.tileXorMs, uint16_t(0));
        jmpi(1 | state.flagCase5Xor, lSkip);
        case5TileXorGather(problem, strategy, state, true, true);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
    }

    auto bottomRows = [&]() {
        rrBegin();
        for (int sr = 0; sr < subGridM; sr++) {
            const int rb = std::max(0, osys - sr * subM), re = subM;
            if (re <= rb) continue;
            const bool init = (sr * subM >= osys);
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, foldAt(sr, sc), sr, sc, rb, re, init);
        }
        rrEnd();
    };
    auto topRows = [&]() {
        rrBegin();
        for (int sr = 0; sr < subGridM; sr++) {
            const int re = std::min(subM, osys - sr * subM);
            if (re <= 0) continue;
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, foldAt(sr, sc), sr, sc, 0, re, true);
        }
        rrEnd();
    };

    // CASE5_XOR_GATHER_LATE=1: the reduction / line update / store move from the pass-0 boundary of
    // the first step (where the bottom-rows fold already sits) to its last-pass boundary, so the two
    // clusters overlap different dpas chains. Phase-A threads only (phase B gathers at the step end).
    const bool gatherLate = (case5EnvOn("CASE5_XOR_GATHER_LATE", false) || gatherStep1) && !singlePass
            && state.tileXorPhaseMode == 0;
    if (firstStep && firstPass && !singlePass) {
        // Phase A threads finish the previous milestone (rows [osys, unrollM) still intact);
        // nothing is pending in the first iteration.
        Label lSkip, lSend;
        cmp(1 | eq | state.flagCase5Xor, state.tileXorMs, uint16_t(0));
        jmpi(1 | state.flagCase5Xor, lSkip);
        const char *bsend = std::getenv("CASE5_XOR_BSEND");
        const bool bSendHere = bsend && !std::strcmp(bsend, "here");
        case5JumpIfPhase(state, true, bSendHere ? lSkip : lSend);   // phase B: gathered already
        if (dbgFold) bottomRows();
        if (!gatherLate && dbgGather) case5TileXorGather(problem, strategy, state, true, true);
        mark(lSend);
        if (!gatherLate && dbgGather) case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
    }
    if (firstStep && lastPass && gatherLate && !gatherStep1 && dbgGather) {
        Label lSkip;
        cmp(1 | eq | state.flagCase5Xor, state.tileXorMs, uint16_t(0));
        jmpi(1 | state.flagCase5Xor, lSkip);
        case5TileXorGather(problem, strategy, state, true, true);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
    }
    const char *topAt = topAtEnv;
    const bool topAtPass0 = !(topAt && !std::strcmp(topAt, "end")) && !xaf && !topPrechain;
    if (lastStep && firstPass && !singlePass && topAtPass0 && dbgFold) {
        // (debug) phase A folds rows [0, osys) right after pass 0: waits for that chain to retire.
        Label lSkip;
        case5JumpIfPhase(state, true, lSkip);
        topRows();
        mark(lSkip);
    }
    if (lastStep && lastPass && !singlePass && !topAtPass0 && !topPrechain && dbgFold) {
        // Phase A folds rows [0, osys) at the end of the step: pass 0 retired while pass 1 issued,
        // so the reads do not stall, and the xors overlap the in-flight pass-1 chain.
        Label lSkip;
        case5JumpIfPhase(state, true, lSkip);
        topRows();
        mark(lSkip);
    }
    const char *bposEnv = std::getenv("CASE5_XOR_BPOS");
    const bool bAtChain = bposEnv && !std::strcmp(bposEnv, "chain");
    if (lastStep && lastPass && (bAtChain || singlePass)) {
        // Phase B threads (all threads when there is a single row pass) fold and store here.
        Label lSkip;
        if (!singlePass) case5JumpIfPhase(state, false, lSkip);
        const char *bf = std::getenv("CASE5_XOR_BFOLD");
        const char *bs = std::getenv("CASE5_XOR_BSTORE");
        if (!(bf && bf[0] == 48))
            for (int sr = 0; sr < subGridM; sr++)
                for (int sc = 0; sc < subGridN; sc++)
                    case5FoldSubtileFull(problem, strategy, state, foldAt(sr, sc), sr, sc);
        if (const char *bp = std::getenv("CASE5_XOR_BPAD")) {
            // debug: N dependent ALU instructions on the fold scratch, no memory, no C reads
            const int n = std::atoi(bp);
            for (int i = 0; i < n; i++)
                xor_(8, state.tileXorStoreData.ud(), state.tileXorStoreData.ud(), state.tileXorFolds[i % subGridN].ud());
        }
        if (!(bs && bs[0] == 48))
            case5TileXorFastStore(problem, strategy, state, true, false);
        if (!singlePass) mark(lSkip);
    }
}

// Case 5.7: after the k loop, fold the SLM-staged line blocks of each sub-tile into its 16-word message
// (fold16: msg = rotl(msg, 13) ^ block, wrap cases: xor), in registers the finished C tile no longer
// needs. The stores, the judges and the hash then read tileXorFoldAcc as they always did.
template <HW hw>
void Generator<hw>::case5TileXorLoadAccFromSlm(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                               GEMMState &state)
{
    if (!problem.case5TileXorAccSlm || state.tileXorFoldAcc.isValid() || !state.tileXorAccSlmOff.isValid())
        return;
    const int nFolds = std::max(1, problem.case5XorSubGridM) * std::max(1, problem.case5XorSubGridN);
    const int nBlocks = std::max(1, problem.case5XorMaxMilestones / 16);
    const int grfPerLine = std::max(1, 16 * 4 / GRF::bytes(hw));
    gemmCase5ReleaseAccForBlake3(strategy, state);
    state.tileXorFoldAcc = state.ra.alloc_range(grfPerLine * nFolds, getHint(HintType::LongTerm, strategy));
    auto tmp = state.ra.alloc_range(grfPerLine);
    auto hdr = state.ra.alloc_range(1);
    for (int fi = 0; fi < nFolds; fi++) {
        GRFRange acc(state.tileXorFoldAcc.getBase() + fi * grfPerLine, grfPerLine);
        for (int b = 0; b < nBlocks; b++) {
            add(1, hdr[0].ud(0), state.tileXorAccSlmOff, uint32_t((b * nFolds + fi) * 64));
            if (b == 0) {
                load(1, acc, block(D32, 16), SLM, hdr);
            } else {
                load(1, tmp, block(D32, 16), SLM, hdr);
                if (problem.case5TileXorFold16)
                    rol(16, acc[0].ud(), acc[0].ud(), uint16_t(13));
                xor_(16, acc[0].ud(), acc[0].ud(), tmp[0].ud());
            }
        }
    }
    state.ra.safeRelease(hdr);
    state.ra.safeRelease(tmp);
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorFinalFlush(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                               GEMMState &state)
{
    if (!problem.case5TileXor || !problem.case5TileXorInterleaved || problem.case5TileXorNop) return;
    if (case5NoStore()) return;
    if (state.C_regs.empty() || state.C_regs[0].empty()) return;
    const int osys = systolicParams(problem).osys;
    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const int nx = strategy.unroll[LoopM];
    if (nx > osys) {                            // single pass: everything was folded in the loop
        // Phase A threads still owe the lower rows of the last milestone; phase B threads only the send.
        Label lSkip, lSend;
        const char *bsend = std::getenv("CASE5_XOR_BSEND");
        const bool bSendHere = bsend && !std::strcmp(bsend, "here");
        case5JumpIfPhase(state, true, bSendHere ? lSkip : lSend);
        const bool rr = case5EnvOn("CASE5_XOR_RR", true);
        // CASE5_XOR_TOPAT=prechain left the upper rows of the last milestone unfolded as well.
        const char *topAtEnv = std::getenv("CASE5_XOR_TOPAT");
        const bool topPrechain = topAtEnv && !std::strcmp(topAtEnv, "prechain") && !strategy.extendedAtomicFMA
                && state.tileXorPhaseMode == 0;
        if (rr) { g_c5FoldRecord = true; g_c5FoldOps.clear(); g_c5FoldChainNext = 0; }
        for (int sr = 0; sr < subGridM; sr++) {
            const int rb = topPrechain ? 0 : std::max(0, osys - sr * subM), re = subM;
            if (re <= rb) continue;
            const bool init = topPrechain || (sr * subM >= osys);
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, state.tileXorFolds[sr * subGridN + sc], sr, sc, rb,
                                     re, init);
        }
        if (rr) case5FoldEmitPending();
        case5TileXorGather(problem, strategy, state, true, true);
        mark(lSend);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
    }
    // Folded message / interleaved 5.4 and 5.5: every block has been folded; write each sub-tile's
    // 16-word message (or wrapped slots) as one line, or hand the messages to the cooperative jackpot.
    // Case 5.7: fold the SLM-staged blocks into the message now (C is finished, its registers are free).
    if (problem.case5TileXorAccSlm && (problem.case5JackpotCoop || !problem.case5TileXorBlake3))
        case5TileXorLoadAccFromSlm(problem, strategy, state);
    if (state.tileXorFoldAcc.isValid() && problem.case5JackpotCoop) {
        gemmCase5TileXorJackpotCoop(problem, strategy, state);
    } else if (state.tileXorFoldAcc.isValid() && !problem.case5TileXorBlake3) {
        const int grfPerLine = std::max(1, 16 * 4 / GRF::bytes(hw));
        if (!state.tileXorLineHdr.empty()) {
            for (int fi = 0; fi < subGridM * subGridN; fi++) {
                GRFRange hdr = state.tileXorLineHdr[state.tileXorLineHdrPacked ? 0 : fi];
                if (state.tileXorLineHdrPacked) {
                    mov(1, state.tileXorStoreHdr[0].uq(0), state.tileXorLineHdr[0][0].uq(fi));
                    hdr = state.tileXorStoreHdr;
                }
                store(1, block(D32, 16), A64, hdr,
                      GRFRange(state.tileXorFoldAcc.getBase() + fi * grfPerLine, grfPerLine));
            }
        } else {
            // No headers kept in the loop (fold16, wrap): sub-tile (sr, sc) of this thread is spatial id
            // (row * subGridM + sr) * tile_cols + col * subGridN + sc, 16 words each in tile_xor.
            const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
            auto hdr = state.ra.alloc_range(2);
            auto off = state.ra.alloc_sub<uint32_t>();
            auto rowOff = state.ra.alloc_sub<uint32_t>();
            auto colOff = state.ra.alloc_sub<uint32_t>();
            mulConstant(1, rowOff, state.tileXorRow, subGridM);
            mulConstant(1, colOff, state.tileXorCol, subGridN);
            for (int sr = 0; sr < subGridM; sr++) {
                for (int sc = 0; sc < subGridN; sc++) {
                    const int fi = sr * subGridN + sc;
                    add(1, off, rowOff, uint16_t(sr));
                    emad(1, off, colOff, off, state.inputs.tileCols, strategy, state);
                    if (sc) add(1, off, off, uint16_t(sc));
                    mulConstant(1, off, off, 16 * 4);
                    eadd(1, hdr[0].uq(0), state.inputs.tileXor, off, strategy, state);
                    store(1, block(D32, 16), A64, hdr,
                          GRFRange(state.tileXorFoldAcc.getBase() + fi * grfPerLine, grfPerLine));
                }
            }
            state.ra.safeRelease(colOff);
            state.ra.safeRelease(rowOff);
            state.ra.safeRelease(off);
            state.ra.safeRelease(hdr);
        }
    }
}

GEMMSTONE_NAMESPACE_END

