/*******************************************************************************

 * Case 5.6: fused in-register BLAKE3 + jackpot after wrap-GRF fold (no tile_xor flush).

 * Key/bound are scalar kernel args (same contract as case5_blake3.cxx).

 ******************************************************************************/



#include "alloc_utils.hpp"

#include "hw_utils.hpp"

#include "kernel_queries.hpp"

#include "gemmstone/generator.hpp"



GEMMSTONE_NAMESPACE_START



using namespace ngen;



namespace {



constexpr int kJackpotWords = 16;

constexpr int kBlake3DigestWords = 8;



constexpr uint32_t kBlake3IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,

                                   0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

constexpr uint8_t kBlake3Perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};



inline int grfDwords(HW hw) {

    return GRF::bytes(hw) / int(sizeof(uint32_t));

}



template <HW hw>

Subregister grfUd(GRF &grf, int dword) {

    return grf.ud(dword % grfDwords(hw));

}



template <HW hw, typename GeneratorT>

void emitBlake3G(GeneratorT &g, GRF vGrf[2], int a, int b, int c, int d, GRF mGrf[2], int mx, int my,

                 GRF &tmp) {

    const Subregister va = grfUd<hw>(vGrf[a / grfDwords(hw)], a);

    const Subregister vb = grfUd<hw>(vGrf[b / grfDwords(hw)], b);

    const Subregister vc = grfUd<hw>(vGrf[c / grfDwords(hw)], c);

    const Subregister vd = grfUd<hw>(vGrf[d / grfDwords(hw)], d);

    const Subregister mxSub = grfUd<hw>(mGrf[mx / grfDwords(hw)], mx);

    const Subregister mySub = grfUd<hw>(mGrf[my / grfDwords(hw)], my);

    g.add(1, va, va, vb);

    g.add(1, va, va, mxSub);

    g.xor_(1, tmp.ud(0), vd, va);

    g.ror(1, vd, tmp.ud(0), uint16_t(16));

    g.add(1, vc, vc, vd);

    g.xor_(1, tmp.ud(0), vb, vc);

    g.ror(1, vb, tmp.ud(0), uint16_t(12));

    g.add(1, va, va, vb);

    g.add(1, va, va, mySub);

    g.xor_(1, tmp.ud(0), vd, va);

    g.ror(1, vd, tmp.ud(0), uint16_t(8));

    g.add(1, vc, vc, vd);

    g.xor_(1, tmp.ud(0), vb, vc);

    g.ror(1, vb, tmp.ud(0), uint16_t(7));

}



template <HW hw, typename GeneratorT>

void emitBlake3Compress64(GeneratorT &g, GEMMState &state, const GRF &keyGrf, const GRF mGrfIn[2],

                          GRF &outGrf) {

    const int dpg = grfDwords(hw);

    GRF vGrf[2] = {state.ra.alloc(), state.ra.alloc()};

    GRF mGrf[2] = {state.ra.alloc(), state.ra.alloc()};

    GRF tGrf[2] = {state.ra.alloc(), state.ra.alloc()};

    GRF tmp = state.ra.alloc();



    for (int i = 0; i < 8; ++i) {

        g.mov(1, grfUd<hw>(vGrf[0], i), keyGrf.ud(i));

        g.mov(1, grfUd<hw>(mGrf[0], i), mGrfIn[0].ud(i));

        g.mov(1, grfUd<hw>(mGrf[1], i), mGrfIn[1].ud(i));

    }

    for (int i = 0; i < 4; ++i) {

        g.mov(1, grfUd<hw>(vGrf[1], i), uint32_t(kBlake3IV[i]));

    }

    g.mov(1, grfUd<hw>(vGrf[1], 4), uint32_t(0));

    g.mov(1, grfUd<hw>(vGrf[1], 5), uint32_t(0));

    g.mov(1, grfUd<hw>(vGrf[1], 6), uint32_t(64));

    g.mov(1, grfUd<hw>(vGrf[1], 7), uint32_t(0x1Bu));



    auto mAt = [&](int i) -> Subregister { return grfUd<hw>(mGrf[i / dpg], i); };



    for (int round = 0; round < 7; ++round) {

        emitBlake3G<hw>(g, vGrf, 0, 4, 8, 12, mGrf, 0, 1, tmp);

        emitBlake3G<hw>(g, vGrf, 1, 5, 9, 13, mGrf, 2, 3, tmp);

        emitBlake3G<hw>(g, vGrf, 2, 6, 10, 14, mGrf, 4, 5, tmp);

        emitBlake3G<hw>(g, vGrf, 3, 7, 11, 15, mGrf, 6, 7, tmp);

        emitBlake3G<hw>(g, vGrf, 0, 5, 10, 15, mGrf, 8, 9, tmp);

        emitBlake3G<hw>(g, vGrf, 1, 6, 11, 12, mGrf, 10, 11, tmp);

        emitBlake3G<hw>(g, vGrf, 2, 7, 8, 13, mGrf, 12, 13, tmp);

        emitBlake3G<hw>(g, vGrf, 3, 4, 9, 14, mGrf, 14, 15, tmp);

        if (round < 6) {

            for (int i = 0; i < 8; ++i) {

                g.mov(1, tGrf[0].ud(i), mAt(kBlake3Perm[i]));

                g.mov(1, tGrf[1].ud(i), mAt(kBlake3Perm[i + 8]));

            }

            for (int i = 0; i < 8; ++i) {

                g.mov(1, mGrf[0].ud(i), tGrf[0].ud(i));

                g.mov(1, mGrf[1].ud(i), tGrf[1].ud(i));

            }

        }

    }

    for (int i = 0; i < 8; ++i) {

        g.xor_(1, outGrf.ud(i), vGrf[0].ud(i), vGrf[1].ud(i));

    }



    state.ra.safeRelease(tmp);

    state.ra.safeRelease(tGrf[1]);

    state.ra.safeRelease(tGrf[0]);

    state.ra.safeRelease(mGrf[1]);

    state.ra.safeRelease(mGrf[0]);

    state.ra.safeRelease(vGrf[1]);

    state.ra.safeRelease(vGrf[0]);

}



bool jackpotScalarsReady(const GEMMState &state) {
    if (!state.inputs.jackpotKeyWords[0].isValid() || !state.inputs.jackpotDigestBuf.isValid()) {
        return false;
    }
    return state.inputs.tileCount.isValid();
}



} // namespace



template <HW hw>

void Generator<hw>::gemmCase5ReleaseAccForBlake3(const GEMMStrategy &strategy, GEMMState &state) {

    (void)strategy;

    safeReleaseRanges(state.C_regs, state);

    safeReleaseRanges(state.Cr_regs, state);

    safeReleaseRanges(state.C_addrs[0], state);

    safeReleaseRanges(state.C_addrs[1], state);

    safeReleaseRanges(state.Cp_regs, state);

    safeReleaseRanges(state.Cp_addrs, state);

    safeReleaseRanges(state.A_regs, state);

    safeReleaseRanges(state.B_regs, state);

    safeReleaseRanges(state.Ar_regs, state);

    safeReleaseRanges(state.Br_regs, state);

    safeReleaseRanges(state.Ap_regs, state);

    safeReleaseRanges(state.Bp_regs, state);

    safeReleaseRanges(state.tempMul_regs, state);

    safeReleaseRanges(state.tileXorSlmAddrs, state);

    for (auto &f : state.tileXorFolds) {

        if (f.isValid()) {

            state.ra.safeRelease(f);

            f.invalidate();

        }

    }

    if (state.tileXorFold.isValid()) {

        state.ra.safeRelease(state.tileXorFold);

        state.tileXorFold.invalidate();

    }

}



template <HW hw>

void Generator<hw>::case5StoreGlobalDword(const GEMMStrategy &strategy, GEMMState &state,

                                          const Subregister &base, const Subregister &val) {

    auto hdr = state.ra.alloc_range(1);

    eadd(1, hdr[0].uq(0), base, uint32_t(0), strategy, state);

    store(1, scattered_dword(), A64, hdr[0], val);

    state.ra.safeRelease(hdr);

}



template <HW hw>

void Generator<hw>::case5StoreGlobalDwordAtOffset(const GEMMStrategy &strategy, GEMMState &state,

                                                  const Subregister &base, int dword_offset,

                                                  const Subregister &val) {

    auto hdr = state.ra.alloc_range(1);

    auto byteOff = state.ra.alloc_sub<uint32_t>();

    mov(1, byteOff, uint32_t(dword_offset * int(sizeof(uint32_t))));

    eadd(1, hdr[0].uq(0), base, byteOff, strategy, state);

    store(1, scattered_dword(), A64, hdr[0], val);

    state.ra.safeRelease(byteOff);

    state.ra.safeRelease(hdr);

}



template <HW hw>

void Generator<hw>::gemmCase5FusedJackpotFromGrf(const GEMMProblem &problem,

                                                 const GEMMStrategy &strategy, GEMMState &state) {

    if (!problem.case5TileXor || !problem.case5FuseJackpot || !problem.case5TileXorWrapGrf
        || problem.case5TileXorWrapGrfStoreMode != 2)
        return;



    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;

    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;

    const bool split = (subGridM > 1) || (subGridN > 1);



    for (auto &g : state.tileXorWrapGrf) {

        if (g.isValid()) {

            state.ra.claim(g);

        }

    }



    auto stageWrapMsg = [&](int fi, GRF stagedMsg[2]) {

        const GRF &wrapBase = state.tileXorWrapGrf[fi * 2];

        for (int lane = 0; lane < kJackpotWords; ++lane) {

            mov(1, a0[0], wrapBase.getBase() * GRF::bytes(hw) + lane * int(sizeof(uint32_t)));

            if (lane < grfDwords(hw)) {

                mov(1, stagedMsg[0].ud(lane), indirect[a0].ud(0));

            } else {

                mov(1, stagedMsg[1].ud(lane - grfDwords(hw)), indirect[a0].ud(0));

            }

        }

    };



    GRF stagedMsg[2] = {state.ra.alloc(getHint(HintType::LongTerm, strategy)),
                        state.ra.alloc(getHint(HintType::LongTerm, strategy))};

    gemmCase5ReleaseAccForBlake3(strategy, state);

    GRF keyGrf = state.ra.alloc();
    for (int i = 0; i < kBlake3DigestWords; ++i) {
        mov(1, keyGrf.ud(i), state.inputs.jackpotKeyWords[i]);
    }

    auto storePanelWord = [&](const Subregister &base, const Subregister &idx, const Subregister &val) {
        auto hdr = state.ra.alloc_range(1);
        auto byteOff = state.ra.alloc_sub<uint32_t>();
        shl(1, byteOff, idx, uint16_t(2));
        eadd(1, hdr[0].uq(0), base, byteOff, strategy, state);
        store(1, scattered_dword(), A64, hdr, val);
        state.ra.safeRelease(byteOff);
        state.ra.safeRelease(hdr);
    };

    auto storeStagedMsgAtSpatial = [&](const Subregister &spatial, const GRF stagedMsg[2]) {
        if (!state.inputs.jackpotMsgBuf.isValid()) {
            return;
        }
        auto wordImm = state.ra.alloc_sub<uint32_t>();
        auto idx = state.ra.alloc_sub<uint32_t>();
        auto wordVal = state.ra.alloc_sub<uint32_t>();
        for (int w = 0; w < kJackpotWords; ++w) {
            mov(1, wordImm, uint32_t(w));
            emad(1, idx, spatial, wordImm, state.inputs.tileCount, strategy, state);
            if (w < grfDwords(hw)) {
                mov(1, wordVal, stagedMsg[0].ud(w));
            } else {
                mov(1, wordVal, stagedMsg[1].ud(w - grfDwords(hw)));
            }
            storePanelWord(state.inputs.jackpotMsgBuf, idx, wordVal);
        }
        state.ra.safeRelease(wordVal);
        state.ra.safeRelease(idx);
        state.ra.safeRelease(wordImm);
    };

    auto computeDigestAndStore = [&](const Subregister &spatial) {
        Label labelSkip;

        if (state.inputs.tileCount.isValid()) {
            cmp(1 | ge | f0[0], spatial, state.inputs.tileCount);
            jmpi(1 | f0[0], labelSkip);
        }

        GRF digest = state.ra.alloc();
        emitBlake3Compress64<hw>(*this, state, keyGrf, stagedMsg, digest);

        auto wordImm = state.ra.alloc_sub<uint32_t>();
        auto idx = state.ra.alloc_sub<uint32_t>();
        auto digestW = state.ra.alloc_sub<uint32_t>();
        for (int w = 0; w < kBlake3DigestWords; ++w) {
            mov(1, wordImm, uint32_t(w));
            emad(1, idx, spatial, wordImm, state.inputs.tileCount, strategy, state);
            mov(1, digestW, digest.ud(w));
            storePanelWord(state.inputs.jackpotDigestBuf, idx, digestW);
        }

        storeStagedMsgAtSpatial(spatial, stagedMsg);

        state.ra.safeRelease(digestW);
        state.ra.safeRelease(idx);
        state.ra.safeRelease(wordImm);
        state.ra.safeRelease(digest);

        mark(labelSkip);
    };

    auto hashFold = [&](int fi, const Subregister &spatial) {
        stageWrapMsg(fi, stagedMsg);
        computeDigestAndStore(spatial);
    };



    if (!split) {

        hashFold(0, state.spatialId);

    } else {

        for (int sr = 0; sr < subGridM; sr++) {

            for (int sc = 0; sc < subGridN; sc++) {

                const int fi = sr * subGridN + sc;

                auto logicalRow = state.ra.alloc_sub<uint32_t>();

                auto logicalCol = state.ra.alloc_sub<uint32_t>();

                auto spatial = state.ra.alloc_sub<uint32_t>();



                mulConstant(1, logicalRow, state.tileXorRow, subGridM);

                add(1, logicalRow, logicalRow, uint16_t(sr));

                mulConstant(1, logicalCol, state.tileXorCol, subGridN);

                add(1, logicalCol, logicalCol, uint16_t(sc));

                emad(1, spatial, logicalCol, logicalRow, state.inputs.tileCols, strategy, state);



                hashFold(fi, spatial);



                state.ra.safeRelease(spatial);

                state.ra.safeRelease(logicalCol);

                state.ra.safeRelease(logicalRow);

            }

        }

    }



    state.ra.safeRelease(keyGrf);
    state.ra.safeRelease(stagedMsg[1]);
    state.ra.safeRelease(stagedMsg[0]);
}



GEMMSTONE_NAMESPACE_END


