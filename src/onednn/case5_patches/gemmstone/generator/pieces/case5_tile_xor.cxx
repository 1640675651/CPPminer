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
    const int slotsPerThread = case5XorOutputMilestoneCount(problem) * nFolds;
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
    const int slotsPerThread = case5XorOutputMilestoneCount(problem) * nFolds;
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
    state.tileXorFold = state.ra.alloc(getHint(HintType::LongTerm, strategy));
    const int nFolds = split ? (subGridM * subGridN) : 1;
    if (split) {
        state.tileXorFolds.resize(nFolds);
        state.tileXorFolds[0] = state.tileXorFold;
        for (int i = 1; i < nFolds; i++)
            state.tileXorFolds[i] = state.ra.alloc(getHint(HintType::LongTerm, strategy));
    } else
        state.tileXorFolds.clear();
    if (problem.case5XorPeriod > 1)
        state.flagCase5Xor = state.raVFlag.alloc();
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
    mov(1, state.tileXorPanel, uint16_t(0));
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
    const bool wrap = problem.case5TileXorWrap;
    const bool wrapGrf = problem.case5TileXorWrapGrf;
    const int wrapGrfStoreMode = problem.case5TileXorWrapGrfStoreMode;
    const bool wrapGrfStoreInLoop = wrapGrf && (wrapGrfStoreMode == 1);
    const bool wrapGrfStageUnified = problem.case5TileXorWrapGrfStageUnified;
    const int outputMs = case5XorOutputMilestoneCount(problem);
    const uint16_t outputMsMask = static_cast<uint16_t>(outputMs - 1);

    add(1, state.tileXorPanel, state.tileXorPanel, uint16_t(1));

    Label lSkipXor;
    Subregister rem;
    if (gateStore) {
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
        mov(simd, fold.ud(), uint16_t(0));
        for (int r = 0; r < state.C_regs[0].getLen(); r++)
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
        if (wrapGrf) {
            auto slotMs = state.ra.alloc_sub<uint32_t>();
            and_(1, slotMs, state.tileXorMs, outputMsMask);
            storeWithWrap(Subregister(), slotMs, fold);
            state.ra.safeRelease(slotMs);
        } else {
            auto idx = state.ra.alloc_sub<uint32_t>();
            if (!wrap && !problem.case5TileXorSlm) {
                emad(1, idx, state.spatialId, state.tileXorMs, state.inputs.tileCount, strategy, state);
                storeFoldAtIdx(idx, state.tileXorMs, fold);
            } else {
                auto slotMs = state.ra.alloc_sub<uint32_t>();
                if (wrap)
                    and_(1, slotMs, state.tileXorMs, outputMsMask);
                else
                    mov(1, slotMs, state.tileXorMs);
                emad(1, idx, state.spatialId, slotMs, state.inputs.tileCount, strategy, state);
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

        if (wrapGrf) {
            auto slotMs = state.ra.alloc_sub<uint32_t>();
            and_(1, slotMs, state.tileXorMs, outputMsMask);
            for (int fi = 0; fi < nFolds; fi++)
                storeWithWrap(Subregister(), slotMs, state.tileXorFolds[fi], fi);
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
                    emad(1, idxs[fi], idxs[fi], slotMs, state.inputs.tileCount, strategy, state);
                else
                    emad(1, idxs[fi], idxs[fi], state.tileXorMs, state.inputs.tileCount, strategy, state);

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

    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);
    const int nFolds = split ? (subGridM * subGridN) : 1;
    const int maxMs = case5XorOutputMilestoneCount(problem);
    const bool useSurface = (state.inputs.surfaceTileXor != InterfaceHandler::noSurface);

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
            mov(1, msImm, uint32_t(ms));
            emad(1, idx, state.spatialId, msImm, state.inputs.tileCount, strategy, state);
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
                auto idx = state.ra.alloc_sub<uint32_t>();
                emad(1, idx, idxs[fi], msImm, state.inputs.tileCount, strategy, state);
                storeGlobalAtIdx(idx, state.tileXorFolds[fi]);
                state.ra.safeRelease(idx);
            }
        }
        state.ra.safeRelease(msImm);

        for (int fi = 0; fi < nFolds; fi++)
            state.ra.safeRelease(idxs[fi]);
    }

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

    auto loadWrapGrfAtSlot = [&](int slot, int fi, const GRF &dst) {
        const GRF &wrapBase = state.tileXorWrapGrf[fi * kWrapGrfPerFold];
        mov(1, a0[0], wrapBase.getBase() * GRF::bytes(hw) + slot * int(sizeof(uint32_t)));
        mov(1, dst.ud(0), indirect[a0].ud(0));
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
            emad(1, idx, state.spatialId, msImm, state.inputs.tileCount, strategy, state);
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
                emad(1, idx, idxs[fi], msImm, state.inputs.tileCount, strategy, state);
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

GEMMSTONE_NAMESPACE_END
