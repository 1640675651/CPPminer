/*******************************************************************************

 * Case 5.6: keyed BLAKE3 b3_compress64 on 16 wrap-GRF slot words per logical tile.

 * Message: contiguous 16 dwords in tileXorWrapGrf[fi*2 .. fi*2+1] (same as 5.5 staging).

 ******************************************************************************/



#include "alloc_utils.hpp"

#include "hw_utils.hpp"

#include "kernel_queries.hpp"

#include "gemmstone/generator.hpp"



GEMMSTONE_NAMESPACE_START



using namespace ngen;



namespace {



constexpr int kBlake3MsgWords = 16;

constexpr int kBlake3DigestWords = 8;

constexpr uint32_t kJackpotFoundBoundBytes = kBlake3DigestWords * uint32_t(sizeof(uint32_t));

constexpr uint32_t kJackpotFoundFlagOff = kJackpotFoundBoundBytes;
constexpr uint32_t kJackpotFoundTRowsOff = kJackpotFoundFlagOff + uint32_t(sizeof(uint32_t));
constexpr uint32_t kJackpotFoundTColsOff = kJackpotFoundTRowsOff + uint32_t(sizeof(uint32_t));



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

void Generator<hw>::gemmCase5TileXorBlake3FromGrf(const GEMMProblem &problem,

                                                  const GEMMStrategy &strategy, GEMMState &state) {

    if (!problem.case5TileXor || !problem.case5TileXorBlake3 || !problem.case5TileXorWrapGrf) {

        return;

    }



    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;

    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;

    const bool split = (subGridM > 1) || (subGridN > 1);

    const int hashMr = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];

    const int hashNr = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];

    const bool gpuJudge = problem.case5FuseJackpot && state.inputs.blake3Beats.isValid()

            && state.inputs.foundFlag.isValid() && state.inputs.trBase.isValid()

            && state.inputs.tcBase.isValid();

    if (!gpuJudge) {

        return;

    }



    for (auto &g : state.tileXorWrapGrf) {

        if (g.isValid()) {

            state.ra.claim(g);

        }

    }



    auto stageWrapMsg = [&](int fi, GRF stagedMsg[2]) {

        const GRF &wrapBase = state.tileXorWrapGrf[fi * 2];

        for (int lane = 0; lane < kBlake3MsgWords; ++lane) {

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

    stageWrapMsg(0, stagedMsg);



    gemmCase5ReleaseAccForBlake3(strategy, state);



    GRF keyGrf = state.ra.alloc();

    for (int i = 0; i < kBlake3DigestWords; ++i) {

        if (state.inputs.blake3KeyWords[i].isValid()) {

            mov(1, keyGrf.ud(i), state.inputs.blake3KeyWords[i]);

        } else {

            mov(1, keyGrf.ud(i), uint32_t(0xA5A5A5A5u ^ static_cast<uint32_t>(i * 0x9E3779B9u)));

        }

    }



    // Bound in found_flag[0:7]; found int at byte offset kJackpotFoundFlagOff (32).



    auto storeBeatAtSpatial = [&](const Subregister &spat, const Subregister &beat) {

        auto hdr = state.ra.alloc_range(1);

        auto byteOff = state.ra.alloc_sub<uint32_t>();

        GRF word = state.ra.alloc();

        shl(1, byteOff, spat, uint16_t(2));

        eadd(1, hdr[0].uq(0), state.inputs.blake3Beats, byteOff, strategy, state);

        mov(1, word.ud(0), beat);

        store(1, scattered_dword(), A64, hdr[0], word);

        state.ra.safeRelease(word);

        state.ra.safeRelease(byteOff);

        state.ra.safeRelease(hdr);

    };



    auto claimFound = [&](const Subregister &beat, const Subregister &logicalRow,

                          const Subregister &logicalCol) {

        auto hdr = state.ra.alloc_range(1);

        GRF foundWord = state.ra.alloc();

        eadd(1, hdr[0].uq(0), state.inputs.foundFlag, kJackpotFoundFlagOff, strategy, state);

        mov(1, foundWord.ud(0), beat);

        atomic(AtomicOp::add, 1, foundWord, scattered_dword(), A64, hdr[0], foundWord);

        state.ra.safeRelease(foundWord);

        state.ra.safeRelease(hdr);



        GRF lrG = state.ra.alloc();

        GRF lcG = state.ra.alloc();

        GRF tRowsGrf = state.ra.alloc();

        GRF tColsGrf = state.ra.alloc();

        mov(1, lrG.ud(0), logicalRow);

        mov(1, lcG.ud(0), logicalCol);

        if (state.inputs.trBase.isValid()) {

            add(1, lrG.ud(0), lrG.ud(0), state.inputs.trBase);

        }

        if (state.inputs.tcBase.isValid()) {

            add(1, lcG.ud(0), lcG.ud(0), state.inputs.tcBase);

        }

        mulConstant(1, tRowsGrf.ud(0), lrG.ud(0), hashMr);

        mulConstant(1, tColsGrf.ud(0), lcG.ud(0), hashNr);



        cmp(1 | ne | f0[0], beat, uint32_t(0));



        {

            auto rowHdr = state.ra.alloc_range(1);

            eadd(1, rowHdr[0].uq(0), state.inputs.foundFlag, kJackpotFoundTRowsOff, strategy, state);

            atomic(AtomicOp::mov, 1 | f0[0], scattered_dword(), A64, rowHdr[0], tRowsGrf);

            state.ra.safeRelease(rowHdr);

        }

        {

            auto colHdr = state.ra.alloc_range(1);

            eadd(1, colHdr[0].uq(0), state.inputs.foundFlag, kJackpotFoundTColsOff, strategy, state);

            atomic(AtomicOp::mov, 1 | f0[0], scattered_dword(), A64, colHdr[0], tColsGrf);

            state.ra.safeRelease(colHdr);

        }



        state.ra.safeRelease(tColsGrf);

        state.ra.safeRelease(tRowsGrf);

        state.ra.safeRelease(lcG);

        state.ra.safeRelease(lrG);

    };



    auto hashFold = [&](int fi, const Subregister &spatial, const Subregister &logicalRow,

                        const Subregister &logicalCol) {

        // Pin spat/row/col BEFORE any ra.alloc in this lambda (found-load/blake3 clobber alloc_sub).

        GRF spatKeepGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

        GRF lrKeepGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

        GRF lcKeepGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

        mov(1, spatKeepGrf.ud(0), spatial);

        mov(1, lrKeepGrf.ud(0), logicalRow);

        mov(1, lcKeepGrf.ud(0), logicalCol);



        Label labelSkip;

        if (state.inputs.tileCount.isValid()) {

            cmp(1 | ge | f0[0], spatKeepGrf.ud(0), state.inputs.tileCount);

            jmpi(1 | f0[0], labelSkip);

        }

        if (gpuJudge) {

            auto hdr = state.ra.alloc_range(1);

            GRF foundWord = state.ra.alloc();

            eadd(1, hdr[0].uq(0), state.inputs.foundFlag, kJackpotFoundFlagOff, strategy, state);

            load(1, foundWord, scattered_dword(), A64, hdr[0]);

            cmp(1 | ne | f0[0], foundWord.ud(0), 0);

            state.ra.safeRelease(foundWord);

            state.ra.safeRelease(hdr);

            jmpi(1 | f0[0], labelSkip);

        }

        if (fi != 0) {

            stageWrapMsg(fi, stagedMsg);

        }



        GRF digest = state.ra.alloc();

        emitBlake3Compress64<hw>(*this, state, keyGrf, stagedMsg, digest);



        // Pin digest in a LongTerm GRF: later temps must not reuse the BLAKE3 result.

        GRF digKeep = state.ra.alloc(getHint(HintType::LongTerm, strategy));

        for (int w = 0; w < kBlake3DigestWords; ++w) {

            mov(1, digKeep.ud(w), digest.ud(w));

        }

        state.ra.safeRelease(digest);



        // Gen12LP in-kernel compare bug: see src/onednn/case5_gen12_jackpot_cmp.md

        if (gpuJudge) {

            GRF beatGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

            GRF dWordGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

            GRF gtMaskGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

            GRF ltMaskGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

            GRF resolvedGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

            GRF tmpGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

            auto beatVal = beatGrf.ud(0);

            auto dWord = dWordGrf.ud(0);

            auto gtMask = gtMaskGrf.ud(0);

            auto ltMask = ltMaskGrf.ud(0);

            auto resolved = resolvedGrf.ud(0);

            auto tmp = tmpGrf.ud(0);

            mov(1, beatVal, uint32_t(0));

            mov(1, resolved, uint32_t(0));

            for (int w = kBlake3DigestWords - 1; w >= 0; --w) {

                GRF bWord = state.ra.alloc();

                GRF wordDecGrf = state.ra.alloc();

                GRF activeGrf = state.ra.alloc();

                GRF notResolvedGrf = state.ra.alloc();

                GRF notActiveGrf = state.ra.alloc();

                GRF beatCandGrf = state.ra.alloc();

                if (state.inputs.blake3BoundWords[w].isValid()) {

                    mov(1, bWord.ud(0), state.inputs.blake3BoundWords[w]);

                } else {

                    mov(1, bWord.ud(0), uint32_t(0));

                }

                mov(1, dWord, digKeep.ud(w));

                cmp(1 | gt, gtMask, dWord, bWord.ud(0));

                cmp(1 | lt, ltMask, dWord, bWord.ud(0));

                or_(1, wordDecGrf.ud(0), gtMask, ltMask);

                not_(1, notResolvedGrf.ud(0), resolved);

                and_(1, activeGrf.ud(0), wordDecGrf.ud(0), notResolvedGrf.ud(0));

                shr(1, beatCandGrf.ud(0), ltMask, uint16_t(31));

                not_(1, notActiveGrf.ud(0), activeGrf.ud(0));

                and_(1, beatVal, beatVal, notActiveGrf.ud(0));

                and_(1, tmp, beatCandGrf.ud(0), activeGrf.ud(0));

                or_(1, beatVal, beatVal, tmp);

                or_(1, resolved, resolved, activeGrf.ud(0));

                state.ra.safeRelease(beatCandGrf);

                state.ra.safeRelease(notActiveGrf);

                state.ra.safeRelease(notResolvedGrf);

                state.ra.safeRelease(activeGrf);

                state.ra.safeRelease(wordDecGrf);

                state.ra.safeRelease(bWord);

            }

            // All words equal => beat (OpenCL digest_beats_target returns true).

            not_(1, tmp, resolved);

            and_(1, tmp, tmp, uint32_t(1));

            or_(1, beatVal, beatVal, tmp);

            storeBeatAtSpatial(spatKeepGrf.ud(0), beatVal);

            claimFound(beatVal, lrKeepGrf.ud(0), lcKeepGrf.ud(0));

            state.ra.safeRelease(tmpGrf);

            state.ra.safeRelease(resolvedGrf);

            state.ra.safeRelease(ltMaskGrf);

            state.ra.safeRelease(gtMaskGrf);

            state.ra.safeRelease(dWordGrf);

            state.ra.safeRelease(beatGrf);

        }

        state.ra.safeRelease(digKeep);

        mark(labelSkip);

        state.ra.safeRelease(lcKeepGrf);

        state.ra.safeRelease(lrKeepGrf);

        state.ra.safeRelease(spatKeepGrf);

    };



    if (!split) {

        // Use gemm tile indices directly (same as OpenCL); float divMod(spatial) is off-by-one.

        hashFold(0, state.spatialId, state.tileXorRow, state.tileXorCol);

    } else {

        for (int sr = 0; sr < subGridM; sr++) {

            for (int sc = 0; sc < subGridN; sc++) {

                const int fi = sr * subGridN + sc;

                // Full GRFs: alloc_sub parents are clobbered before hashFold can snapshot.

                GRF logicalRowGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

                GRF logicalColGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

                GRF spatialGrf = state.ra.alloc(getHint(HintType::LongTerm, strategy));

                auto logicalRow = logicalRowGrf.ud(0);

                auto logicalCol = logicalColGrf.ud(0);

                auto spatial = spatialGrf.ud(0);



                mulConstant(1, logicalRow, state.tileXorRow, subGridM);

                add(1, logicalRow, logicalRow, uint16_t(sr));

                mulConstant(1, logicalCol, state.tileXorCol, subGridN);

                add(1, logicalCol, logicalCol, uint16_t(sc));

                emad(1, spatial, logicalCol, logicalRow, state.inputs.tileCols, strategy, state);



                hashFold(fi, spatial, logicalRow, logicalCol);



                state.ra.safeRelease(spatialGrf);

                state.ra.safeRelease(logicalColGrf);

                state.ra.safeRelease(logicalRowGrf);

            }

        }

    }



    state.ra.safeRelease(keyGrf);

    state.ra.safeRelease(stagedMsg[1]);

    state.ra.safeRelease(stagedMsg[0]);

}



GEMMSTONE_NAMESPACE_END

