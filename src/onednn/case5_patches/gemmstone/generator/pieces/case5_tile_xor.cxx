/*******************************************************************************
 * Case5 milestoned tile XOR: once per case5XorPeriod unrollK panels
 * (Case 3.4 KR=128 → period=4 when unrollK=32), XOR-reduce live C_regs to one
 * uint32 and store to tile_xor[ms * tile_count + spatial_id].
 * Case5.1 (case5TileXorNop): same gate/store; fold replaced by const 0x51CACE51.
 *
 * The K-loop invokes this every unrollK panel; fold+store run only when
 * panel % xorPeriod == 0. Uses raVFlag (not state.ra.alloc_flag) so flagAP is
 * not stolen. Sequencer period must stay unrollK (ls.getUnroll() == unrollK).
 *******************************************************************************/

#include "hw_utils.hpp"
#include "gemmstone/generator.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;

template <HW hw>
void Generator<hw>::gemmCase5TileXorSetup(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                          GEMMState &state)
{
    if (!problem.case5TileXor)
        return;

    const auto unrollM = strategy.unroll[LoopM];
    const auto unrollN = strategy.unroll[LoopN];

    state.tileXorMs = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.tileXorPanel = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.spatialId = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.tileXorFold = state.ra.alloc(getHint(HintType::LongTerm, strategy));
    if (problem.case5XorPeriod > 1)
        state.flagCase5Xor = state.raVFlag.alloc();

    mov(1, state.tileXorMs, uint16_t(0));
    mov(1, state.tileXorPanel, uint16_t(0));

    // spatial_id = (i0 / unrollM) * tile_cols + (j0 / unrollN)
    auto tileRow = state.ra.alloc_sub<uint32_t>();
    auto tileCol = state.ra.alloc_sub<uint32_t>();
    if ((unrollM & (unrollM - 1)) == 0)
        shr(1, tileRow, state.i0, uint16_t(ilog2(unrollM)));
    else
        divDown(tileRow, state.i0, uint32_t(unrollM), strategy, state);
    if ((unrollN & (unrollN - 1)) == 0)
        shr(1, tileCol, state.j0, uint16_t(ilog2(unrollN)));
    else
        divDown(tileCol, state.j0, uint32_t(unrollN), strategy, state);

    // spatialId = tileCol + tileRow * tileCols
    emad(1, state.spatialId, tileCol, tileRow, state.inputs.tileCols, strategy, state);

    state.ra.safeRelease(tileRow);
    state.ra.safeRelease(tileCol);
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorStore(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                          GEMMState &state)
{
    if (!problem.case5TileXor)
        return;
    if (state.C_regs.empty() || state.C_regs[0].empty())
        return;

    const int xorPeriod = (problem.case5XorPeriod > 0) ? problem.case5XorPeriod : 1;

    add(1, state.tileXorPanel, state.tileXorPanel, uint16_t(1));

    Label lSkipXor;
    Subregister rem;
    if (xorPeriod > 1) {
        // Skip fold/store unless panel is a multiple of xorPeriod (KR boundary).
        // case5XorPeriod is always a power of two (milestone_k / unrollK).
        rem = state.ra.alloc_sub<uint32_t>();
        and_(1, rem, state.tileXorPanel, uint16_t(xorPeriod - 1));
        cmp(1 | nz | state.flagCase5Xor, rem, 0);
        jmpi(1 | state.flagCase5Xor, lSkipXor);
    }

    const int simd = elementsPerGRF(hw, DataType::ud);
    GRF fold = state.tileXorFold.isValid() ? state.tileXorFold : state.ra.alloc();
    const bool releaseFold = !state.tileXorFold.isValid();

    if (problem.case5TileXorNop) {
        // Case 5.1 ablation: keep panel gate + global store; skip C GRF fold.
        mov(1, fold.ud(0), uint32_t(0x51CACE51u));
    } else {
        // fold = XOR of all C GRFs (does not modify C_regs).
        mov(simd, fold.ud(), uint16_t(0));
        for (int r = 0; r < state.C_regs[0].getLen(); r++)
            xor_(simd, fold.ud(), fold.ud(), state.C_regs[0][r].ud());

        // Tree-reduce within the fold GRF → scalar ud(0).
        for (int n = simd >> 1; n >= 1; n >>= 1)
            xor_(n, fold.ud(0)(1), fold.ud(0)(1), fold.ud(n)(1));
    }
    // dword_index = ms * tile_count + spatial_id
    auto idx = state.ra.alloc_sub<uint32_t>();
    emad(1, idx, state.spatialId, state.tileXorMs, state.inputs.tileCount, strategy, state);

    const bool useSurface = (state.inputs.surfaceTileXor != InterfaceHandler::noSurface);
    if (useSurface) {
        auto hdr = state.ra.alloc().ud();
        shl(1, hdr, idx, uint16_t(2)); // byte offset
        store(1, surface_dword(ChannelMask::r), Surface(state.inputs.surfaceTileXor), hdr, fold);
        state.ra.safeRelease(hdr);
    } else if (state.inputs.tileXor.isValid()) {
        auto hdr = state.ra.alloc_range(1);
        auto byteOff = state.ra.alloc_sub<uint32_t>();
        shl(1, byteOff, idx, uint16_t(2));
        eadd(1, hdr[0].uq(0), state.inputs.tileXor, byteOff, strategy, state);
        store(1, scattered_dword(), A64, hdr, fold);
        state.ra.safeRelease(byteOff);
        state.ra.safeRelease(hdr);
    }

    add(1, state.tileXorMs, state.tileXorMs, uint16_t(1));

    state.ra.safeRelease(idx);
    if (releaseFold)
        state.ra.safeRelease(fold);

    if (xorPeriod > 1) {
        mark(lSkipXor);
        state.ra.safeRelease(rem);
    }
}

GEMMSTONE_NAMESPACE_END
