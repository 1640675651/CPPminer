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

#include <cstdlib>
#include <cstring>
#include <cstdio>

GEMMSTONE_NAMESPACE_START

using namespace ngen;

size_t case5TileXorSlmByteOffset(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy,
                                 bool computeMax);

constexpr int kFoldTreeFanin = 8;

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
            // num_ms (raw milestones) or 16 (folded jackpot message); one header per sub-tile line,
            // advanced by B*4 bytes at each flush (the folded message is stored once at the end).
            const int words = problem.case5TileXorFold16 ? 16 : numMs;
            mulConstant(1, lc, lc, words);
            const int nLines = int(state.tileXorLineHdr.size());
            for (int fi = 0; fi < nLines; fi++) {
                eadd(1, state.tileXorLineHdr[fi][0].uq(0), state.inputs.tileXor, lc, strategy, state);
                if (fi + 1 < nLines) add(1, lc, lc, uint32_t(words * 4));
            }
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

    if (problem.case5TileXorSlm && state.tileXorLocalId.isValid()) {
        const int wgM = strategy.wg[LoopM];
        const int wgN = strategy.wg[LoopN];
        auto localM = state.ra.alloc_sub<uint32_t>();
        auto localN = state.ra.alloc_sub<uint32_t>();
        mod(localM, idM, wgM, strategy, state);
        mod(localN, idN, wgN, strategy, state);
        emad(1, state.tileXorLocalId, localM, localN, wgM, strategy, state);
        state.ra.safeRelease(localM);
        state.ra.safeRelease(localN);
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
    if (problem.case5TileXorSlm)
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
    // Interleaved XMX fold: one 2-GRF (16 lane) accumulator per sub-tile so the SIMD16 bfn chain
    // accumulates directly (no per-chunk collapse) and the four scalars come out of one packed
    // reduction. The 1-GRF fold handles alias the first GRF of each range. CASE5_XOR_WIDE=0 disables.
    state.tileXorFoldWide.clear();
    {
        const char *wideEnv = std::getenv("CASE5_XOR_WIDE");
        if (problem.case5TileXorInterleaved && split && subGridM == 1 && (subGridN == 4 || subGridN == 8)
                && !(wideEnv && wideEnv[0] == '0')) {
            state.tileXorFoldWide.resize(nFolds);
            bool ok = true;
            for (int i = 0; i < nFolds; i++) {
                state.tileXorFoldWide[i] = state.ra.try_alloc_range(2, getHint(HintType::LongTerm, strategy));
                ok = ok && state.tileXorFoldWide[i].isValid();
            }
            if (!ok) {
                for (auto &r : state.tileXorFoldWide) state.ra.safeRelease(r);
                state.tileXorFoldWide.clear();
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
    // Fast milestone store (one oword message per milestone) for 1xN sub-tile grids with A64 tile_xor.
    if (split && subGridM == 1 && (subGridN == 4 || subGridN == 8) && !problem.case5TileXorWrap
            && !problem.case5TileXorWrapGrf && !problem.case5TileXorSlm
            && state.inputs.surfaceTileXor == InterfaceHandler::noSurface && state.inputs.tileXor.isValid()) {
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
        state.tileXorLineBuf = state.ra.try_alloc_range(grfPerLine * subGridN + 1, getHint(HintType::LongTerm, strategy));
        bool ok = state.tileXorLineBuf.isValid();
        // The folded message is stored once, at the end, with an address computed there (in the
        // dead line buffer); only the raw-milestone store needs per-line headers for the whole loop.
        if (!problem.case5TileXorFold16) {
            state.tileXorLineHdr.resize(subGridN);
            for (int i = 0; i < subGridN; i++) {
                state.tileXorLineHdr[i] = state.ra.try_alloc_range(1, getHint(HintType::LongTerm, strategy));
                ok = ok && state.tileXorLineHdr[i].isValid();
            }
        }
        // The flush test needs a flag (set transiently right before its jmpi); period-1 paths
        // without the interleaved fold do not allocate one otherwise.
        if (ok && !state.flagCase5Xor.isValid())
            state.flagCase5Xor = state.raVFlag.alloc();
        // Folded message (case5TileXorFold16): one 16-word accumulator per sub-tile, laid out like
        // the lines.
        if (ok && problem.case5TileXorFold16) {
            if (B != 16)
                throw std::runtime_error("Case5 fold16 needs 16-milestone line blocks (K/128 % 16 == 0)");
            state.tileXorFoldAcc = state.ra.try_alloc_range(grfPerLine * subGridN, getHint(HintType::LongTerm, strategy));
            if (!state.tileXorFoldAcc.isValid())
                throw std::runtime_error("Case5 fold16: insufficient registers for the fold accumulators");
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
        throw std::runtime_error("Case5 interleaved XOR needs the fast store path (1x4/1x8 sub-tiles, A64)");
    if (problem.case5TileXorWrapGrf) {
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
    // Folded message accumulators start at zero (rotl(0) ^ block0 = block0).
    if (state.tileXorFoldAcc.isValid()) {
        const int dw = elementsPerGRF(hw, DataType::ud);
        for (int g = 0; g < state.tileXorFoldAcc.getLen(); g += 2)
            mov(2 * dw, state.tileXorFoldAcc[g].ud(), uint16_t(0));
    }
    // Count mode (CASE5_XOR_GATE=count): panels left until the first milestone store.
    const bool countGate = (problem.case5XorPeriod > 1) && !problem.case5TileXorIncremental && case5GateCount();
    mov(1, state.tileXorPanel, uint16_t(countGate ? problem.case5XorPeriod : 0));
    if (state.tileXorSlmOff.isValid())
        case5EmitTileXorSlmByteOffConstMs(problem, strategy, state, state.tileXorSlmOff, 0, 0);
    case5ZeroFoldGrfs(strategy, state, nFolds);
    if (problem.case5TileXorWrapGrf) {
        const int simd = elementsPerGRF(hw, DataType::ud);
        for (int i = 0; i < kWrapGrfPerFold * nFolds; i++)
            mov(simd, state.tileXorWrapGrf[i].ud(), uint16_t(0));
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
                for (int r = 4; r < nGrf; r += 2)
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

        for (int i = 0; i < nFolds; i++)
            mov(simd, folds[i].ud(), uint16_t(0));

        auto foldAt = [&](int sr, int sc) { return folds[sr * subGridN + sc]; };

        if (state.C_layout.colMajor()) {
            for (int col = 0; col < unrollN; col++) {
                const int sc = col / subN;
                for (int row = 0; row < unrollM; row += simd) {
                    const int sr = row / subM;
                    Subregister C0 = state.C_layout.find(row, col, state.C_regs[0]);
                    xorSimdIntoFold(foldAt(sr, sc), C0);
                }
            }
        } else {
            for (int row = 0; row < unrollM; row++) {
                const int sr = row / subM;
                for (int col = 0; col < unrollN; col += simd) {
                    const int sc = col / subN;
                    Subregister C0 = state.C_layout.find(row, col, state.C_regs[0]);
                    xorSimdIntoFold(foldAt(sr, sc), C0);
                }
            }
        }

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
            auto hdr = state.ra.alloc_range(1);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
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
            auto hdr = state.ra.alloc_range(1);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
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
            auto hdr = state.ra.alloc_range(1);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
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
            auto hdr = state.ra.alloc_range(1);
            auto byteOff = state.ra.alloc_sub<uint32_t>();
            shl(1, byteOff, idx, uint16_t(2));
            eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
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

    // Fast path (colMajor C): per 8-row chunk, consecutive columns of a dpas block sit in adjacent
    // GRFs, so a SIMD16 operand covers two columns; bfn 0x96 (s0^s1^s2) folds two such operands per
    // instruction into a 2-GRF accumulator F, which is then folded into foldReg. Roughly 4x fewer
    // instructions than the SIMD8 chain. CASE5_XOR_SIMD16=0 / CASE5_XOR_BFN=0 disable the parts.
    const char *s16Env = std::getenv("CASE5_XOR_SIMD16");
    const char *bfnEnv = std::getenv("CASE5_XOR_BFN");
    int wideIdx = -1;
    for (size_t i = 0; i < state.tileXorFoldWide.size(); i++)
        if (state.tileXorFoldWide[i].isValid() && state.tileXorFoldWide[i][0].getBase() == foldReg.getBase())
            wideIdx = int(i);
    const bool useWide = (wideIdx >= 0) && state.C_layout.colMajor() && !(s16Env && s16Env[0] == '0');
    const bool useS16 = state.C_layout.colMajor() && state.tileXorPairTmp.isValid()
            && !(s16Env && s16Env[0] == '0');
    const bool useBfn = !(bfnEnv && bfnEnv[0] == '0');
    if (useWide) {
        // Wide path: the 16-lane accumulator A (2 GRFs) takes the SIMD16 units directly; the
        // reduction to a scalar happens once per milestone in case5TileXorGather.
        auto A = state.tileXorFoldWide[wideIdx];
        for (int row = row0; row < row1; row += simd) {
            std::vector<Subregister> units;
            std::vector<bool> width16;
            for (int col = col0; col < col1; ) {
                Subregister c0 = state.C_layout.find(row, col, state.C_regs[0]);
                bool pair = false;
                if (col + 1 < col1) {
                    Subregister c1 = state.C_layout.find(row, col + 1, state.C_regs[0]);
                    pair = (c1.getBase() == c0.getBase() + 1) && c0.getOffset() == 0 && c1.getOffset() == 0;
                }
                units.push_back(c0);
                width16.push_back(pair);
                col += pair ? 2 : 1;
            }
            size_t i = 0;
            while (i < units.size()) {
                if (width16[i]) {
                    if (first) {
                        if (i + 1 < units.size() && width16[i + 1]) {
                            xor_(16, A[0].ud(), units[i].ud(0)(1), units[i + 1].ud(0)(1));
                            i += 2;
                        } else {
                            mov(16, A[0].ud(), units[i].ud(0)(1));
                            i += 1;
                        }
                        first = false;
                    } else if (useBfn && i + 1 < units.size() && width16[i + 1]) {
                        bfn(16, 0x96, A[0].ud(), A[0].ud(), units[i].ud(0)(1), units[i + 1].ud(0)(1));
                        i += 2;
                    } else {
                        xor_(16, A[0].ud(), A[0].ud(), units[i].ud(0)(1));
                        i += 1;
                    }
                } else {
                    if (first) {
                        mov(16, A[0].ud(), uint16_t(0));
                        first = false;
                    }
                    xor_(simd, A[0].ud(), A[0].ud(), units[i].ud(0)(1));
                    i += 1;
                }
            }
        }
        if (first) mov(16, A[0].ud(), uint16_t(0));
        return;
    }
    if (useS16) {
        auto F = state.tileXorPairTmp;
        for (int row = row0; row < row1; row += simd) {
            std::vector<Subregister> units;   // operand starts; width16 marks a 2-GRF operand
            std::vector<bool> width16;
            for (int col = col0; col < col1; ) {
                Subregister c0 = state.C_layout.find(row, col, state.C_regs[0]);
                bool pair = false;
                if (col + 1 < col1) {
                    Subregister c1 = state.C_layout.find(row, col + 1, state.C_regs[0]);
                    pair = (c1.getBase() == c0.getBase() + 1) && c0.getOffset() == 0 && c1.getOffset() == 0;
                }
                units.push_back(c0);
                width16.push_back(pair);
                col += pair ? 2 : 1;
            }
            // Accumulate all SIMD16 units into F, SIMD8 units into F[0].
            bool fFirst = true;
            size_t i = 0;
            while (i < units.size()) {
                if (width16[i]) {
                    if (fFirst) {
                        if (i + 1 < units.size() && width16[i + 1]) {
                            xor_(16, F[0].ud(), units[i].ud(0)(1), units[i + 1].ud(0)(1));
                            i += 2;
                        } else {
                            mov(16, F[0].ud(), units[i].ud(0)(1));
                            i += 1;
                        }
                        fFirst = false;
                    } else if (useBfn && i + 1 < units.size() && width16[i + 1]) {
                        bfn(16, 0x96, F[0].ud(), F[0].ud(), units[i].ud(0)(1), units[i + 1].ud(0)(1));
                        i += 2;
                    } else {
                        xor_(16, F[0].ud(), F[0].ud(), units[i].ud(0)(1));
                        i += 1;
                    }
                } else {
                    if (fFirst) {
                        mov(16, F[0].ud(), uint16_t(0));
                        fFirst = false;
                    }
                    xor_(simd, F[0].ud(), F[0].ud(), units[i].ud(0)(1));
                    i += 1;
                }
            }
            if (fFirst) continue;
            if (first) {
                xor_(simd, foldReg.ud(), F[0].ud(), F[1].ud());
                first = false;
            } else if (useBfn) {
                bfn(simd, 0x96, foldReg.ud(), foldReg.ud(), F[0].ud(), F[1].ud());
            } else {
                xor_(simd, foldReg.ud(), foldReg.ud(), F[0].ud());
                xor_(simd, foldReg.ud(), foldReg.ud(), F[1].ud());
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
bool Generator<hw>::case5TileXorFastStoreOk(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                            GEMMState &state)
{
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    if (subGridM != 1 || (subGridN != 4 && subGridN != 8)) return false;
    if (problem.case5TileXorWrap || problem.case5TileXorWrapGrf || problem.case5TileXorSlm) return false;
    if (state.inputs.surfaceTileXor != InterfaceHandler::noSurface) return false;
    if (!state.inputs.tileXor.isValid()) return false;
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
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const int simd = elementsPerGRF(hw, DataType::ud);
    auto &data = state.tileXorStoreData;
    if (reduce && !state.tileXorFoldWide.empty() && int(state.tileXorFoldWide.size()) == subGridN
            && (subGridN == 4 || subGridN == 8) && simd == 8) {
        // Packed reduction of n 16-lane accumulators W[i] to n scalars in data.ud(0..n-1), reusing
        // the accumulator GRFs as scratch (every GRF written was already consumed):
        //   16 -> 8 lanes: slot i = W[i][0] ^ W[i][1]            (n SIMD8, lands in W[0..n/2))
        //    8 -> 4:       per range r: (8;4,1) ^ (8;4,1)+4      (n/2 SIMD8, lands in W[n/2..3n/4))
        //    4 -> 2:       per range q: (4;2,1) ^ (4;2,1)+2      (n/4 SIMD8, lands in W[3n/4..))
        //    2 -> 1:       (2;1,0) ^ (2;1,0)+1 -> data           (1 SIMDn)
        auto &W = state.tileXorFoldWide;
        const int n = subGridN;
        for (int i = 0; i < n; i++)
            xor_(8, W[i / 2][i % 2].ud(), W[i][0].ud(), W[i][1].ud());
        for (int r = 0; r < n / 2; r++)
            xor_(8, W[n / 2 + r / 2][r % 2].ud(), W[r][0].ud(0)(8, 4, 1), W[r][0].ud(4)(8, 4, 1));
        for (int q = 0; q < n / 4; q++)
            xor_(8, W[n / 2 + n / 4 + q / 2][q % 2].ud(), W[n / 2 + q][0].ud(0)(4, 2, 1),
                 W[n / 2 + q][0].ud(2)(4, 2, 1));
        xor_(n, data.ud(0)(1), W[n / 2 + n / 4][0].ud(0)(2, 1, 0), W[n / 2 + n / 4][0].ud(1)(2, 1, 0));
    } else {
        for (int fi = 0; fi < subGridN; fi++) {
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
        for (int fi = 0; fi < subGridN; fi++) {
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
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
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
        if (state.tileXorFoldAcc.isValid()) {
            // The line holds milestones 16b..16b+15 in slot order, so the protocol fold
            // msg[w] = rotl(msg[w], 13) ^ x[16b + w] is one rotate and one xor per 16 lanes.
            for (int fi = 0; fi < subGridN; fi++) {
                GRF acc = state.tileXorFoldAcc[fi * grfPerLine];
                GRF line = state.tileXorLineBuf[fi * grfPerLine];
                rol(16, acc.ud(), acc.ud(), uint16_t(13));
                xor_(16, acc.ud(), acc.ud(), line.ud());
            }
            mark(lSkip);
            return;
        }
        const char *cc = std::getenv("CASE5_XOR_STORE_CACHE");
        for (int fi = 0; fi < subGridN; fi++) {
            GRFRange line(state.tileXorLineBuf.getBase() + fi * grfPerLine, grfPerLine);
            auto spec = block(D32, B);
            if (cc && !std::strcmp(cc, "s")) spec = spec | CacheSettingsLSC::L1S_L3UC;
            else if (cc && !std::strcmp(cc, "uc")) spec = spec | CacheSettingsLSC::L1UC_L3UC;
            else if (cc && !std::strcmp(cc, "wt")) spec = spec | CacheSettingsLSC::L1WT_L3UC;
            else if (cc && !std::strcmp(cc, "wb")) spec = spec | CacheSettingsLSC::L1WT_L3WB;
            store(1, spec, A64, state.tileXorLineHdr[fi], line);
            eadd(1, state.tileXorLineHdr[fi][0].uq(0), state.tileXorLineHdr[fi][0].uq(0),
                 state.tileXorCountBytes, strategy, state);
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
    if (lscEnv && lscEnv[0] == '0')
        store(16, block_oword(subGridN / 4), A64, state.tileXorStoreHdr, state.tileXorStoreData);
    else {
        // CASE5_XOR_STORE_CACHE=uc|wt|wb selects the LSC cache policy (default: hardware default).
        const char *cc = std::getenv("CASE5_XOR_STORE_CACHE");
        auto spec = block(D32, subGridN);
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

// Whole sub-tile fold into foldReg (dword-wise, not yet reduced). Uses SIMD16 over the two row
// halves of a column when their GRFs are adjacent (colMajor C, subM == 2*simd), else SIMD8.
template <HW hw>
void Generator<hw>::case5FoldSubtileFull(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                         GEMMState &state, const GRF &foldReg, int sr, int sc)
{
    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subN = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];
    const int simd = elementsPerGRF(hw, DataType::ud);
    const char *pairEnv = std::getenv("CASE5_XOR_PAIR");
    const bool pairs = state.C_layout.colMajor() && subM == 2 * simd && state.tileXorPairTmp.isValid()
            && !(pairEnv && pairEnv[0] == 48);
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
    if (preChain) {
        if (!xaf || !firstStep || singlePass) return;
        const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
        const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
        const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
        Label lSkip;
        cmp(1 | eq | state.flagCase5Xor, state.tileXorMs, uint16_t(0));
        jmpi(1 | state.flagCase5Xor, lSkip);
        for (int sr = 0; sr < subGridM; sr++) {
            const int rb = std::max(0, osys - sr * subM), re = subM;
            if (re <= rb) continue;
            const bool init = (sr * subM >= osys);
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, state.tileXorFolds[sr * subGridN + sc], sr, sc, rb, re,
                                     init);
        }
        case5TileXorGather(problem, strategy, state, true, true);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
        return;
    }
    if (!lastStep && !firstStep) return;
    if (!firstPass && !lastPass) return;

    const int subM = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    auto foldAt = [&](int sr, int sc) { return state.tileXorFolds[sr * subGridN + sc]; };

    auto bottomRows = [&]() {
        for (int sr = 0; sr < subGridM; sr++) {
            const int rb = std::max(0, osys - sr * subM), re = subM;
            if (re <= rb) continue;
            const bool init = (sr * subM >= osys);
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, foldAt(sr, sc), sr, sc, rb, re, init);
        }
    };
    auto topRows = [&]() {
        for (int sr = 0; sr < subGridM; sr++) {
            const int re = std::min(subM, osys - sr * subM);
            if (re <= 0) continue;
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, foldAt(sr, sc), sr, sc, 0, re, true);
        }
    };

    if (firstStep && firstPass && !singlePass) {
        // Phase A threads finish the previous milestone (rows [osys, unrollM) still intact);
        // nothing is pending in the first iteration.
        Label lSkip, lSend;
        cmp(1 | eq | state.flagCase5Xor, state.tileXorMs, uint16_t(0));
        jmpi(1 | state.flagCase5Xor, lSkip);
        const char *bsend = std::getenv("CASE5_XOR_BSEND");
        const bool bSendHere = bsend && !std::strcmp(bsend, "here");
        case5JumpIfPhase(state, true, bSendHere ? lSkip : lSend);   // phase B: gathered already
        bottomRows();
        case5TileXorGather(problem, strategy, state, true, true);
        mark(lSend);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
    }
    const char *topAt = std::getenv("CASE5_XOR_TOPAT");
    const bool topAtPass0 = !(topAt && !std::strcmp(topAt, "end")) && !xaf;
    if (lastStep && firstPass && !singlePass && topAtPass0) {
        // (debug) phase A folds rows [0, osys) right after pass 0: waits for that chain to retire.
        Label lSkip;
        case5JumpIfPhase(state, true, lSkip);
        topRows();
        mark(lSkip);
    }
    if (lastStep && lastPass && !singlePass && !topAtPass0) {
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
        for (int sr = 0; sr < subGridM; sr++) {
            const int rb = std::max(0, osys - sr * subM), re = subM;
            if (re <= rb) continue;
            const bool init = (sr * subM >= osys);
            for (int sc = 0; sc < subGridN; sc++)
                case5FoldSubtileRows(problem, strategy, state, state.tileXorFolds[sr * subGridN + sc], sr, sc, rb,
                                     re, init);
        }
        case5TileXorGather(problem, strategy, state, true, true);
        mark(lSend);
        case5TileXorSend(problem, strategy, state, true);
        mark(lSkip);
    }
    // Folded message: every block has been folded; write each sub-tile's 16 words as one line at
    // tile_xor + (sid0 + fi) * 64 bytes. The line buffer is dead now, so its pad GRF holds the header.
    if (state.tileXorFoldAcc.isValid() && problem.case5JackpotCoop) {
        gemmCase5TileXorJackpotCoop(problem, strategy, state);
    } else if (state.tileXorFoldAcc.isValid()) {
        const int grfPerLine = std::max(1, 16 * 4 / GRF::bytes(hw));
        GRF hdr = state.tileXorLineBuf[grfPerLine * subGridN];
        auto lr = hdr.ud(4), lc = hdr.ud(5);
        mulConstant(1, lr, state.tileXorRow, subGridM);
        mulConstant(1, lc, state.tileXorCol, subGridN);
        emad(1, lc, lc, lr, state.inputs.tileCols, strategy, state);
        shl(1, lc, lc, uint16_t(6));
        eadd(1, hdr.uq(0), state.inputs.tileXor, lc, strategy, state);
        for (int fi = 0; fi < subGridN; fi++) {
            if (fi > 0)
                eadd(1, hdr.uq(0), hdr.uq(0), uint32_t(64), strategy, state);
            store(1, block(D32, 16), A64, GRFRange(hdr.getBase(), 1),
                  GRFRange(state.tileXorFoldAcc.getBase() + fi * grfPerLine, grfPerLine));
        }
    }
}

GEMMSTONE_NAMESPACE_END

