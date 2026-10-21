/*******************************************************************************

 * Case 5.6: keyed BLAKE3 b3_compress64 on 16 wrap-GRF slot words per logical tile.

 * Message: contiguous 16 dwords in tileXorWrapGrf[fi*2 .. fi*2+1] (same as 5.5 staging).

 ******************************************************************************/



#include <cstdio>
#include <cstdlib>
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
constexpr uint32_t kJackpotFoundCoordsOff =
        (kJackpotFoundFlagOff + uint32_t(sizeof(uint32_t)) + 7u) & ~7u;



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



namespace {
template <typename GeneratorT, typename OpV, typename OpM, typename OpT>
void emitBlake3RoundsWide(GeneratorT &g, OpV v, OpM m, OpT tmp, int esize, bool useAdd3);
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

    const bool gpuJudge = problem.case5FuseJackpot && state.inputs.foundFlag.isValid()

            && state.inputs.trBase.isValid()

            && state.inputs.tcBase.isValid();

    if (!gpuJudge) {

        return;

    }



    for (auto &g : state.tileXorWrapGrf) {
        if (g.isValid()) {
            state.ra.claim(g);
        }
    }

    // SIMD-N judge (default for N = subGridM x subGridN logical tiles per thread, N = 2..8 a power of
    // two; lane fi = sr * subGridN + sc): the thread's N logical tiles are hashed together, one tile per lane, in a single N-wide keyed BLAKE3 compression
    // (emitBlake3RoundsWide), instead of N one-lane compressions. Messages are transposed from the
    // wrap-GRF slots (tile fi, word w -> lane fi of message register w). The digest compare is
    // branchless, in GRF masks (Gen12LP: a jmpi right after a cmp can read a stale flag, see
    // src/onednn/case5_gen12_jackpot_cmp.md), and only winning lanes touch the found buffer, through
    // predicated atomics. CASE5_JACKPOT_SIMDN=0 restores the per-tile judge below.
    {
        const char *simdEnv = std::getenv("CASE5_JACKPOT_SIMDN");
        const int N = subGridM * subGridN;
        const int dpg = grfDwords(hw);
        if (!(simdEnv && simdEnv[0] == '0') && N >= 2 && N <= 8 && N <= dpg
                && (N & (N - 1)) == 0 && state.inputs.tileCols.isValid()) {
            gemmCase5ReleaseAccForBlake3(strategy, state);

            GRF lrG = state.ra.alloc(), lcG = state.ra.alloc(), sidG = state.ra.alloc();
            for (int fi = 0; fi < N; fi++) {
                const int sr = fi / subGridN, sc = fi % subGridN;   // same tile order as the per-tile judge
                mulConstant(1, lrG.ud(fi), state.tileXorRow, subGridM);
                if (sr) add(1, lrG.ud(fi), lrG.ud(fi), uint16_t(sr));
                mulConstant(1, lcG.ud(fi), state.tileXorCol, subGridN);
                if (sc) add(1, lcG.ud(fi), lcG.ud(fi), uint16_t(sc));
                emad(1, sidG.ud(fi), lcG.ud(fi), lrG.ud(fi), state.inputs.tileCols, strategy, state);
            }

            GRFRange m = state.ra.alloc_range(kBlake3MsgWords), v = state.ra.alloc_range(16);
            GRFRange t = state.ra.alloc_range(4);
            // Wrap slot w of tile fi: tileXorWrapGrf[fi * 2 + w / dpg] dword w % dpg (2 GRFs per tile).
            for (int fi = 0; fi < N; fi++)
                for (int w = 0; w < kBlake3MsgWords; w++)
                    mov(1, m[w].ud(fi), state.tileXorWrapGrf[fi * 2 + w / dpg].ud(w % dpg));

            auto V = [&](int i) { return v[i].ud(); };
            auto Mw = [&](int w) { return m[w].ud(); };
            auto T = [&](int k) { return t[k].ud(); };
            for (int i = 0; i < kBlake3DigestWords; ++i) {
                if (state.inputs.blake3KeyWords[i].isValid()) {
                    mov(N, V(i), state.inputs.blake3KeyWords[i]);
                } else {
                    mov(N, V(i), uint32_t(0xA5A5A5A5u ^ static_cast<uint32_t>(i * 0x9E3779B9u)));
                }
            }
            for (int i = 0; i < 4; i++) mov(N, V(8 + i), uint32_t(kBlake3IV[i]));
            mov(N, V(12), uint32_t(0));
            mov(N, V(13), uint32_t(0));
            mov(N, V(14), uint32_t(64));      // block length
            mov(N, V(15), uint32_t(0x1Bu));   // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
            emitBlake3RoundsWide(*this, V, Mw, T, N, /*useAdd3=*/false);
            for (int i = 0; i < kBlake3DigestWords; i++) xor_(N, V(i), V(i), V(8 + i));   // digest in V(0..7)
            state.ra.safeRelease(m);

            // beat = digest <= bound, lexicographic from word 7 down (digest_beats_target), as
            // all-ones / zero lane masks.
            GRF beat = state.ra.alloc(), resolved = state.ra.alloc(), ltM = state.ra.alloc();
            GRF gtM = state.ra.alloc(), decM = state.ra.alloc(), bnd = state.ra.alloc();
            mov(8, beat.ud(), uint32_t(0));
            mov(8, resolved.ud(), uint32_t(0));
            for (int w = kBlake3DigestWords - 1; w >= 0; --w) {
                if (state.inputs.blake3BoundWords[w].isValid()) {
                    mov(1, bnd.ud(0), state.inputs.blake3BoundWords[w]);
                } else {
                    mov(1, bnd.ud(0), uint32_t(0));
                }
                cmp(N | lt, ltM.ud(), V(w), bnd.ud(0));
                cmp(N | gt, gtM.ud(), V(w), bnd.ud(0));
                or_(N, decM.ud(), ltM.ud(), gtM.ud());          // decided at this word
                not_(N, T(0), resolved.ud());
                and_(N, T(0), T(0), decM.ud());                 // ... and not before
                and_(N, T(1), T(0), ltM.ud());
                or_(N, beat.ud(), beat.ud(), T(1));
                or_(N, resolved.ud(), resolved.ud(), decM.ud());
            }
            not_(N, T(0), resolved.ud());                       // all words equal => beat
            or_(N, beat.ud(), beat.ud(), T(0));
            if (state.inputs.tileCount.isValid()) {             // tiles past the panel never win
                cmp(N | lt, ltM.ud(), sidG.ud(), state.inputs.tileCount);
                and_(N, beat.ud(), beat.ud(), ltM.ud());
            }

            // Winners: found = 1 and their packed (t_rows, t_cols), one predicated atomic each.
            auto hdrF = state.ra.alloc_range(1), hdrC = state.ra.alloc_range(1);
            GRF one = state.ra.alloc(), coords = state.ra.alloc();
            eadd(1, hdrF[0].uq(0), state.inputs.foundFlag, kJackpotFoundFlagOff, strategy, state);
            eadd(1, hdrC[0].uq(0), state.inputs.foundFlag, kJackpotFoundCoordsOff, strategy, state);
            mov(1, one.ud(0), uint32_t(1));
            for (int fi = 0; fi < N; fi++) {
                add(1, T(2), lrG.ud(fi), state.inputs.trBase);
                mulConstant(1, coords.ud(0), t[2].ud(0), hashMr);
                add(1, T(3), lcG.ud(fi), state.inputs.tcBase);
                mulConstant(1, coords.ud(1), t[3].ud(0), hashNr);
                cmp(1 | ne | f0[0], null.ud(), beat.ud(fi), uint32_t(0));
                atomic(AtomicOp::mov, 1 | f0[0], scattered_dword(), A64, hdrF[0], one);
                atomic(AtomicOp::mov, 1 | f0[0], scattered_qword(), A64, hdrC[0], coords);
            }
            state.ra.safeRelease(coords);
            state.ra.safeRelease(one);
            state.ra.safeRelease(hdrC);
            state.ra.safeRelease(hdrF);
            state.ra.safeRelease(bnd);
            state.ra.safeRelease(decM);
            state.ra.safeRelease(gtM);
            state.ra.safeRelease(ltM);
            state.ra.safeRelease(resolved);
            state.ra.safeRelease(beat);
            state.ra.safeRelease(t);
            state.ra.safeRelease(v);
            state.ra.safeRelease(sidG);
            state.ra.safeRelease(lcG);
            state.ra.safeRelease(lrG);
            return;
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



    // Bound in found_flag[0:7]; found int at +32; packed (t_rows,t_cols) qword at +40.



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

        GRF coordsGrf = state.ra.alloc();

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



        mov(1, coordsGrf.ud(0), tRowsGrf.ud(0));

        mov(1, coordsGrf.ud(1), tColsGrf.ud(0));

        {

            auto coordHdr = state.ra.alloc_range(1);

            eadd(1, coordHdr[0].uq(0), state.inputs.foundFlag, kJackpotFoundCoordsOff, strategy,

                 state);

            atomic(AtomicOp::mov, 1 | f0[0], scattered_qword(), A64, coordHdr[0], coordsGrf);

            state.ra.safeRelease(coordHdr);

        }



        state.ra.safeRelease(coordsGrf);

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



/*******************************************************************************
 * Cooperative fused jackpot on the fold16 message (XMX fast-store path, case5JackpotCoop).
 *  1. every thread writes its sub-tiles' 16-word messages, spatial ids and packed (t_rows, t_cols)
 *     into an SLM region past gemmstone's own;
 *  2. SLM fence + barrier;
 *  3. the first ceil(tiles/16) threads each take 16 tiles as SIMD16 lanes: four 4-word vector loads
 *     return the messages word-major for free, one keyed BLAKE3 compression (the four G chains of a
 *     half-round interleaved) hashes 16 tiles per instruction, a lexicographic SIMD16 compare against
 *     the bound gives a lane predicate, and winners store found = 1 and their packed coordinates
 *     (the per-tile judge's found-buffer contract). With blake3_out bound the digests are stored
 *     too (harness verification). All other threads exit at once.
 * Versus the per-tile Case 5.6 judge (one SIMD1 compression per tile per thread) this is 16x fewer
 * instructions per tile and the dependent chain sits on a quarter of the threads.
 ******************************************************************************/
namespace {

// The 7 rounds at execution size esize. v(i)/m(i) give state and message word i as 16-lane
// regions, tmp(k) the temporary of G chain k. The message schedule is applied statically (round
// r reads word perm_r[i]); the four independent G functions of a half-round are emitted in lockstep.
template <typename GeneratorT, typename OpV, typename OpM, typename OpT>
void emitBlake3RoundsWide(GeneratorT &g, OpV v, OpM m, OpT tmp, int esize, bool useAdd3) {
    int perm[16];
    for (int i = 0; i < 16; i++) perm[i] = i;
    auto G4 = [&](const int (&a)[4], const int (&b)[4], const int (&c)[4], const int (&d)[4],
                  const int (&mx)[4], const int (&my)[4]) {
        // a = a + b + m: one 3-input add on XeHP+ (12 instead of 14 instructions per G)
        if (useAdd3) {
            for (int k = 0; k < 4; k++) g.add3(esize, v(a[k]), v(a[k]), v(b[k]), m(mx[k]));
        } else {
            for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), v(b[k]));
            for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), m(mx[k]));
        }
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(d[k]), v(a[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(d[k]), tmp(k), uint16_t(16));
        for (int k = 0; k < 4; k++) g.add(esize, v(c[k]), v(c[k]), v(d[k]));
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(b[k]), v(c[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(b[k]), tmp(k), uint16_t(12));
        if (useAdd3) {
            for (int k = 0; k < 4; k++) g.add3(esize, v(a[k]), v(a[k]), v(b[k]), m(my[k]));
        } else {
            for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), v(b[k]));
            for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), m(my[k]));
        }
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(d[k]), v(a[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(d[k]), tmp(k), uint16_t(8));
        for (int k = 0; k < 4; k++) g.add(esize, v(c[k]), v(c[k]), v(d[k]));
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(b[k]), v(c[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(b[k]), tmp(k), uint16_t(7));
    };
    const int ca[4] = {0, 1, 2, 3}, cb[4] = {4, 5, 6, 7}, cc[4] = {8, 9, 10, 11}, cd[4] = {12, 13, 14, 15};
    const int da[4] = {0, 1, 2, 3}, db[4] = {5, 6, 7, 4}, dc[4] = {10, 11, 8, 9}, dd[4] = {15, 12, 13, 14};
    for (int round = 0; round < 7; ++round) {
        const int mx0[4] = {perm[0], perm[2], perm[4], perm[6]},   my0[4] = {perm[1], perm[3], perm[5], perm[7]};
        const int mx1[4] = {perm[8], perm[10], perm[12], perm[14]}, my1[4] = {perm[9], perm[11], perm[13], perm[15]};
        G4(ca, cb, cc, cd, mx0, my0);    // columns
        G4(da, db, dc, dd, mx1, my1);    // diagonals
        int next[16];
        for (int i = 0; i < 16; i++) next[i] = perm[kBlake3Perm[i]];
        for (int i = 0; i < 16; i++) perm[i] = next[i];
    }
}

} // namespace

// Cooperative jackpot: the found buffer (bound to the tile_xor argument) also carries the
// per-panel arguments, so the k loop keeps one pointer register. Byte offsets (host side:
// case33_gemm_onednn.cpp kJackpotCoopArgsOff): 0 bound[8], 32 found, 40 coords, 64 key[8],
// 96 tile_cols, 100 tr_base, 104 tc_base, 108 tile_count. One 128-byte block load covers them.
static constexpr uint32_t kCoopBufLoadBytes = 128;
static constexpr int kCoopBufKeyDw = 16, kCoopBufTileColsDw = 24, kCoopBufTrDw = 25, kCoopBufTcDw = 26,
                     kCoopBufTileCountDw = 27;
// CASE5_COOP_AUDIT: digests at tile_xor + 4096 + sid*32, messages at + 4096 + tile_count*32 + sid*64.
static constexpr uint32_t kCoopAuditOff = 4096;

template <HW hw>
void Generator<hw>::gemmCase5TileXorCoopStageArgs(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                                  GEMMState &state)
{
    // The 16x64 preferred strategy is at the register limit: release the jackpot's argument
    // registers (key, bound, found pointer, tile bases, tile_cols, ...) before the k loop. The
    // epilogue reloads their values from the found buffer through the tile_xor pointer.
    if (!problem.case5JackpotCoop || !state.tileXorLidFlat.isValid()) return;
    auto &in = state.inputs;
    if (std::getenv("CASE5_DEBUG_RA"))
        std::fprintf(stderr, "[coop] wg %dx%d activeThreads=%d slmA=%zu slmB=%zu slmBuffers=%d unrollKSLM=%d\n",
                     strategy.wg[LoopM], strategy.wg[LoopN], strategy.activeThreads,
                     size_t(strategy.slmABufSize(problem)), size_t(strategy.slmBBufSize(problem)),
                     strategy.slmBuffers, strategy.unrollKSLM);
    if (!in.tileXor.isValid()) stub("cooperative jackpot needs the tile_xor argument (found buffer)");
    state.tileXorCoopJudge = in.foundFlag.isValid() && in.trBase.isValid() && in.tcBase.isValid();
    state.tileXorCoopDigestOut = in.blake3Out.isValid();      // harness digest-out mode keeps its pointer
    for (int i = 0; i < kBlake3DigestWords; i++) {
        state.ra.safeRelease(in.blake3KeyWords[i]);
        state.ra.safeRelease(in.blake3BoundWords[i]);
    }
    state.ra.safeRelease(in.foundFlag);
    state.ra.safeRelease(in.trBase);
    state.ra.safeRelease(in.tcBase);
    state.ra.safeRelease(in.tileCols);
    state.ra.safeRelease(in.tileCount);
    state.ra.safeRelease(in.xorPeriod);
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorJackpotCoop(const GEMMProblem &problem, const GEMMStrategy &strategy,
                                                 GEMMState &state)
{
    if (!problem.case5JackpotCoop || !state.tileXorFoldAcc.isValid() || !state.tileXorLidFlat.isValid())
        return;
    if (hw < HW::XeHPG) stub("cooperative jackpot needs LSC messages");
    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const int tilesPerThread = subGridM * subGridN;
    const int nTiles = case5JackpotCoopTiles(problem, strategy);
    // Lanes per hashing thread (tiles per SIMD instruction). 8: twice as many threads share the
    // work group's hashing tail and SIMD8 int ops issue in one cycle (CASE5_COOP_LANES=16 for A/B).
    const char *lanesEnv = std::getenv("CASE5_COOP_LANES");
    const int L = (lanesEnv && std::atoi(lanesEnv) == 16) ? 16 : 8;
    const int nHash = (nTiles + L - 1) / L;
    const int hashMr = (problem.case5XorSubM > 0) ? problem.case5XorSubM : strategy.unroll[LoopM];
    const int hashNr = (problem.case5XorSubN > 0) ? problem.case5XorSubN : strategy.unroll[LoopN];
    const int dpg = grfDwords(hw);
    const uint32_t blake3BlockLen = 64, blake3Flags = 0x1Bu;   // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
    if (subGridM != 1 || tilesPerThread > dpg || (nTiles % 2) != 0 || nTiles <= 0)
        stub("cooperative jackpot: unsupported sub-tile grid");
    const uint32_t slmBase = 0;                   // the k loop's SLM buffers are dead after the barrier below
    const uint32_t msgBase = slmBase;
    const uint32_t sidBase = msgBase + uint32_t(nTiles) * 64;
    const uint32_t coordBase = sidBase + uint32_t(nTiles) * 4;      // nTiles even: 8-byte aligned
    // CASE5_COOP_SKIP (timing only; no shares): 1 = no epilogue at all, 2 = no hashing, 4 = no stores.
    const char *skipEnv = std::getenv("CASE5_COOP_SKIP");
    const int coopSkip = skipEnv ? std::atoi(skipEnv) : 0;
    if (coopSkip & 1) return;
    const bool judge = state.tileXorCoopJudge && !(coopSkip & 4);   // captured before the arguments were released
    const bool audit = problem.case5JackpotCoopAudit;
    const bool digestOut = (state.tileXorCoopDigestOut || audit) && !(coopSkip & 4);

    // The accumulators and operand registers are dead after the k loop: release them first so the
    // scratch below (and the hashing threads' 90 GRFs) can be allocated in a full register file.
        if (std::getenv("CASE5_DEBUG_RA")) std::fprintf(stderr, "[ra] %s: %d GRFs\n", "coop entry", state.ra.countAllocedRegisters());
    gemmCase5ReleaseAccForBlake3(strategy, state);
        if (std::getenv("CASE5_DEBUG_RA")) std::fprintf(stderr, "[ra] %s: %d GRFs\n", "coop after release", state.ra.countAllocedRegisters());

    // Default: every thread hashes its own N = tilesPerThread tiles, one SIMD lane per sub-tile,
    // straight from the fold registers: a register transpose instead of SLM tables, no barriers.
    // The busiest thread runs one N-lane hash, like a hashing thread of the SLM variant
    // (problem.case5JackpotCoopSlm, CASE5_COOP_OWN=0), which measured the same speed at N = 4.
    if (!problem.case5JackpotCoopSlm) {
        const int N = tilesPerThread;
        if (subGridM != 1 || N < 1 || N > 8 || (N & (N - 1)) || dpg != 8)
            stub("own-tile jackpot: needs a 1 x N sub-tile grid, N a power of two <= 8, and 32-byte GRFs");
        // With the hash skipped this kernel never completed on the A770 (cause not found); refuse
        // to build it rather than hang the GPU. Timing knob only.
        if (coopSkip & 2) stub("own-tile jackpot: CASE5_COOP_SKIP=2 hangs, not supported");
        const int wpg = 8 / N;                                  // message words per GRF in the packed transpose
        GRF hdr = state.ra.alloc(), sidG = state.ra.alloc();
        GRFRange coordG = state.ra.alloc_range(2);              // SIMD8 qword payload (lanes 0..N-1 used)
        GRFRange args = state.ra.alloc_range(int(kCoopBufLoadBytes) / GRF::bytes(hw));
        GRFRange hdrQ = state.ra.alloc_range(1);
        auto ARG = [&](int i) { return args[i / dpg].ud(i % dpg); };
        const auto trBase = ARG(kCoopBufTrDw), tcBase = ARG(kCoopBufTcDw), tileCols = ARG(kCoopBufTileColsDw);
        const auto foundFlag = state.inputs.tileXor, blake3Out = state.inputs.blake3Out;
        emov(1, hdrQ[0].uq(0), state.inputs.tileXor, strategy, state);
        load(1, args, block(D32, int(kCoopBufLoadBytes) / 4), A64, hdrQ);
        {
            GRF lr = state.ra.alloc(), lc = state.ra.alloc();
            for (int fi = 0; fi < N; fi++) {
                mulConstant(1, lr.ud(0), state.tileXorRow, subGridM);
                mulConstant(1, lc.ud(0), state.tileXorCol, subGridN);
                add(1, lc.ud(0), lc.ud(0), uint16_t(fi));
                emad(1, sidG.ud(fi), lc.ud(0), lr.ud(0), tileCols, strategy, state);  // spatial id
                if (judge) {
                    add(1, lr.ud(0), lr.ud(0), trBase);
                    add(1, lc.ud(0), lc.ud(0), tcBase);
                }
                mulConstant(1, coordG[(2 * fi) / dpg].ud((2 * fi) % dpg), lr.ud(0), hashMr);          // t_rows
                mulConstant(1, coordG[(2 * fi + 1) / dpg].ud((2 * fi + 1) % dpg), lc.ud(0), hashNr);  // t_cols
            }
            state.ra.safeRelease(lc);
            state.ra.safeRelease(lr);
        }
        if (audit) {
            // owner copy of each message: tile_xor + 4096 + tile_count*32 + sid*64
            GRFRange ahdr = state.ra.alloc_range(1);
            GRF aoff = state.ra.alloc();
            shl(1, aoff.ud(0), ARG(kCoopBufTileCountDw), uint16_t(5));
            add(1, aoff.ud(0), aoff.ud(0), uint32_t(kCoopAuditOff));
            for (int fi = 0; fi < N; fi++) {
                shl(1, aoff.ud(1), sidG.ud(fi), uint16_t(6));
                add(1, aoff.ud(1), aoff.ud(1), aoff.ud(0));
                eadd(1, ahdr[0].uq(0), state.inputs.tileXor, aoff.ud(1), strategy, state);
                store(1, block(D32, 16), A64, ahdr, GRFRange(state.tileXorFoldAcc.getBase() + fi * 2, 2));
            }
            state.ra.safeRelease(aoff);
            state.ra.safeRelease(ahdr);
        }
        {
            GRFRange v = state.ra.alloc_range(16), P = state.ra.alloc_range(16 / wpg), tmp = state.ra.alloc_range(4);
            GRFRange hdr64 = state.ra.alloc_range(2), ones = state.ra.alloc_range(1);
            GRFRange A;
            if (wpg > 1) A = state.ra.alloc_range(16);
            GRF scal = state.ra.alloc(), dtmp = state.ra.alloc(), hdrS = state.ra.alloc();
            auto V = [&](int i) { return v[i].ud(); };
            // Word w of the N tiles: N lanes at the start of a GRF. The transpose packs wpg words per
            // GRF (P); words not at a GRF start are copied to A[w], because add3 with an operand that
            // starts mid-register computed wrong sums (two-source ops on it were fine).
            auto Mw = [&](int w) { return (w % wpg) ? A[w].ud(0)(1) : P[w / wpg].ud(0)(1); };
            auto T = [&](int k) { return tmp[k].ud(); };
            // transpose: tile fi's word w (fold GRF 2 fi + w/8, dword w%8) -> P[w/wpg] dword (w%wpg)*N + fi;
            // one mov per (tile, wpg consecutive words): 2 N^2 movs
            const int dstStride = (wpg > 1) ? N : 1;
            for (int fi = 0; fi < N; fi++)
                for (int w = 0; w < kBlake3MsgWords; w += wpg)
                    mov(wpg, P[w / wpg].ud(fi)(dstStride),
                        GRF(state.tileXorFoldAcc.getBase() + 2 * fi + w / 8).ud(w % 8)(1));
            for (int w = 0; w < kBlake3MsgWords; w++)
                if (w % wpg) mov(N, A[w].ud(), P[w / wpg].ud((w % wpg) * N)(1));
            mov(1, scal.uw(0), uint16_t((1u << N) - 1));                      // valid lanes
            if (audit) {
                // hasher copy: tile_xor + 4096 + tile_count*96 + sid*64 + 4w (stores stay SIMD8, masked)
                mulConstant(1, hdr.ud(1), ARG(kCoopBufTileCountDw), 96);
                add(1, hdr.ud(1), hdr.ud(1), uint32_t(kCoopAuditOff));
                shl(8, hdrS.ud(), sidG.ud(), uint16_t(6));
                add(8, hdrS.ud(), hdrS.ud(), hdr.ud(1));
                eadd(8, hdr64[0].uq(), state.inputs.tileXor, hdrS.ud(), strategy, state);
                mov(1, f1[0], scal.uw(0));
                for (int w = 0; w < kBlake3MsgWords; w++) {
                    if (w) eadd(8, hdr64[0].uq(), hdr64[0].uq(), uint32_t(4), strategy, state);
                    mov(N, dtmp.ud(), Mw(w));
                    store(8 | f1[0], scattered_dword(), A64, hdr64[0], dtmp);
                }
            }
            for (int i = 0; i < kBlake3DigestWords; i++) mov(N, V(i), ARG(kCoopBufKeyDw + i));
            for (int i = 0; i < 4; i++) mov(N, V(8 + i), uint32_t(kBlake3IV[i]));
            mov(N, V(12), uint32_t(0));
            mov(N, V(13), uint32_t(0));
            mov(N, V(14), blake3BlockLen);
            mov(N, V(15), blake3Flags);
            {
                const char *a3 = std::getenv("CASE5_COOP_ADD3");
                const bool useAdd3 = hw >= HW::XeHP && !(a3 && a3[0] == '0');
                emitBlake3RoundsWide(*this, V, Mw, T, N, useAdd3);
            }
            for (int i = 0; i < kBlake3DigestWords; i++) xor_(N, V(i), V(i), V(8 + i));   // digest in V(0..7)
            if (digestOut) {
                const auto outBase = audit ? state.inputs.tileXor : blake3Out;
                shl(8, hdrS.ud(), sidG.ud(), uint16_t(5));
                if (audit) add(8, hdrS.ud(), hdrS.ud(), uint32_t(kCoopAuditOff));
                eadd(8, hdr64[0].uq(), outBase, hdrS.ud(), strategy, state);
                mov(1, f1[0], scal.uw(0));
                for (int w = 0; w < kBlake3DigestWords; w++) {
                    if (w) eadd(8, hdr64[0].uq(), hdr64[0].uq(), uint32_t(4), strategy, state);
                    store(8 | f1[0], scattered_dword(), A64, hdr64[0], v[w]);
                }
            }
            if (judge) {
                // beats = digest < bound lexicographically from word 7 down, equal counts as a hit
                mov(1, scal.uw(1), scal.uw(0));
                for (int w = 0; w < kBlake3DigestWords; w++) {
                    cmp(N | lt | f1[0], null.ud(), V(w), ARG(w));
                    cmp(N | eq | f1[1], null.ud(), V(w), ARG(w));
                    mov(1, scal.uw(2), f1[0]);
                    mov(1, scal.uw(3), f1[1]);
                    and_(1, scal.uw(3), scal.uw(3), scal.uw(1));
                    or_(1, scal.uw(1), scal.uw(2), scal.uw(3));
                }
                and_(1, scal.uw(1), scal.uw(1), scal.uw(0));
                mov(1, f0[1], scal.uw(1));
                mov(8, ones[0].ud(), uint32_t(1));
                eadd(8, hdr64[0].uq(), foundFlag, uint32_t(kJackpotFoundFlagOff), strategy, state);
                store(8 | f0[1], scattered_dword(), A64, hdr64[0], ones[0]);      // found = 1 (any winner)
                eadd(8, hdr64[0].uq(), foundFlag, uint32_t(kJackpotFoundCoordsOff), strategy, state);
                store(8 | f0[1], scattered_qword(), A64, hdr64[0], coordG[0]);    // one winner's (t_rows, t_cols)
            }
            state.ra.safeRelease(hdrS);
            state.ra.safeRelease(dtmp);
            state.ra.safeRelease(scal);
            if (wpg > 1) state.ra.safeRelease(A);
            state.ra.safeRelease(ones);
            state.ra.safeRelease(hdr64);
            state.ra.safeRelease(tmp);
            state.ra.safeRelease(P);
            state.ra.safeRelease(v);
        }
        state.ra.safeRelease(state.tileXorLineBuf);
        state.ra.safeRelease(state.tileXorFoldAcc);
        state.ra.safeRelease(hdrQ);
        state.ra.safeRelease(args);
        state.ra.safeRelease(coordG);
        state.ra.safeRelease(sidG);
        state.ra.safeRelease(hdr);
        return;
    }

    // ---- 1. this thread's tiles -> SLM ----
    GRF hdr = state.ra.alloc(), tIdx = state.ra.alloc(), sidG = state.ra.alloc();
    GRFRange coordG = state.ra.alloc_range(std::max(1, (2 * tilesPerThread + dpg - 1) / dpg));
    // Reload the jackpot arguments parked in SLM by gemmCase5TileXorCoopStageArgs (this thread's
    // own stores, long complete): key, bound, found pointer, tile bases, tile_cols.
    GRFRange args = state.ra.alloc_range(int(kCoopBufLoadBytes) / GRF::bytes(hw));
    GRFRange hdrQ = state.ra.alloc_range(1);
    auto ARG = [&](int i) { return args[i / dpg].ud(i % dpg); };
    const auto trBase = ARG(kCoopBufTrDw), tcBase = ARG(kCoopBufTcDw), tileCols = ARG(kCoopBufTileColsDw);
    const auto foundFlag = state.inputs.tileXor, blake3Out = state.inputs.blake3Out;
    emov(1, hdrQ[0].uq(0), state.inputs.tileXor, strategy, state);
    load(1, args, block(D32, int(kCoopBufLoadBytes) / 4), A64, hdrQ);
    // Every thread's last operand loads completed before its final dpas, so after this barrier
    // the k loop's SLM buffers can hold the tile tables.
    useTempAndR0(state, [&](GRF temp, GRF r0_info) { slmBarrier(temp, r0_info, strategy); });
    mulConstant(1, tIdx.ud(0), state.tileXorLidFlat, tilesPerThread);        // this thread's first tile
    for (int fi = 0; fi < tilesPerThread; fi++) {
        mulConstant(1, hdr.ud(0), tIdx.ud(0), 64);
        add(1, hdr.ud(0), hdr.ud(0), uint32_t(msgBase + fi * 64));
        store(1, block(D32, 16), SLM, hdr, GRFRange(state.tileXorFoldAcc.getBase() + fi * 2, 2));
    }
    {
        GRF lr = state.ra.alloc(), lc = state.ra.alloc();
        for (int fi = 0; fi < tilesPerThread; fi++) {
            const int sr = fi / subGridN, sc = fi % subGridN;
            mulConstant(1, lr.ud(0), state.tileXorRow, subGridM);
            add(1, lr.ud(0), lr.ud(0), uint16_t(sr));
            mulConstant(1, lc.ud(0), state.tileXorCol, subGridN);
            add(1, lc.ud(0), lc.ud(0), uint16_t(sc));
            emad(1, sidG.ud(fi), lc.ud(0), lr.ud(0), tileCols, strategy, state);  // spatial id
            if (judge) {
                add(1, lr.ud(0), lr.ud(0), trBase);
                add(1, lc.ud(0), lc.ud(0), tcBase);
            }
            mulConstant(1, coordG[(2 * fi) / dpg].ud((2 * fi) % dpg), lr.ud(0), hashMr);          // t_rows
            mulConstant(1, coordG[(2 * fi + 1) / dpg].ud((2 * fi + 1) % dpg), lc.ud(0), hashNr);  // t_cols
        }
        state.ra.safeRelease(lc);
        state.ra.safeRelease(lr);
    }
    if (audit) {
        // The owner thread also writes each tile's message (from its own registers) so the host
        // can recompute the digest: tile_xor + 4096 + tile_count*32 + sid*64.
        GRFRange ahdr = state.ra.alloc_range(1);
        GRF aoff = state.ra.alloc();
        shl(1, aoff.ud(0), ARG(kCoopBufTileCountDw), uint16_t(5));
        add(1, aoff.ud(0), aoff.ud(0), uint32_t(kCoopAuditOff));
        for (int fi = 0; fi < tilesPerThread; fi++) {
            shl(1, aoff.ud(1), sidG.ud(fi), uint16_t(6));
            add(1, aoff.ud(1), aoff.ud(1), aoff.ud(0));
            eadd(1, ahdr[0].uq(0), state.inputs.tileXor, aoff.ud(1), strategy, state);
            store(1, block(D32, 16), A64, ahdr, GRFRange(state.tileXorFoldAcc.getBase() + fi * 2, 2));
        }
        state.ra.safeRelease(aoff);
        state.ra.safeRelease(ahdr);
    }
    mulConstant(1, hdr.ud(0), tIdx.ud(0), 4);
    add(1, hdr.ud(0), hdr.ud(0), uint32_t(sidBase));
    store(1, block(D32, tilesPerThread), SLM, hdr, sidG);
    mulConstant(1, hdr.ud(0), tIdx.ud(0), 8);
    add(1, hdr.ud(0), hdr.ud(0), uint32_t(coordBase));
    store(1, block(D32, 2 * tilesPerThread), SLM, hdr, coordG);
    state.ra.safeRelease(coordG);
    state.ra.safeRelease(sidG);
    state.ra.safeRelease(tIdx);

    // ---- 2. every message visible to the work group ----
    useTempAndR0(state, [&](GRF temp, GRF r0_info) {
        slmfence(temp, r0_info);
        fencewait();
        slmBarrier(temp, r0_info, strategy);
    });

    // ---- 3. hashing threads: lane l of thread t handles tile L t + l ----
    Label lEnd;
    cmp(1 | ge | f0[0], null.ud(), state.tileXorLidFlat, uint32_t((coopSkip & 2) ? 0 : nHash));
    jmpi(1 | f0[0], lEnd);
    const int gpl = std::max(1, L / dpg);                 // GRFs per L-lane dword vector
    const int halves = (L + 7) / 8;                       // SIMD8 pieces of the 64-bit address math

    state.ra.safeRelease(state.tileXorLineBuf);
    // Audit mode: the A64 message stores above still read the fold accumulators (only the SLM
    // stores are fenced), so keep those registers out of the hashing allocations.
    if (!audit) state.ra.safeRelease(state.tileXorFoldAcc);
        if (std::getenv("CASE5_DEBUG_RA")) std::fprintf(stderr, "[ra] %s: %d GRFs\n", "coop before hash allocs", state.ra.countAllocedRegisters());

    GRFRange v = state.ra.try_alloc_range(16 * gpl), m = state.ra.try_alloc_range(16 * gpl), tmp = state.ra.try_alloc_range(4 * gpl);
    GRFRange laneIdx = state.ra.try_alloc_range(gpl), hdr16 = state.ra.try_alloc_range(gpl), sid16 = state.ra.try_alloc_range(gpl);
    GRFRange coord16 = state.ra.try_alloc_range(2 * gpl), hdr64 = state.ra.try_alloc_range(2 * halves), ones = state.ra.try_alloc_range(gpl);
    GRF scal = state.ra.try_alloc();
    if (!v.isValid() || !m.isValid() || !tmp.isValid() || !laneIdx.isValid() || !hdr16.isValid() || !sid16.isValid()
            || !coord16.isValid() || !hdr64.isValid() || !ones.isValid() || !scal.isValid())
        stub("cooperative jackpot: insufficient registers");
    auto V = [&](int i) { return v[gpl * i].ud(); };
    auto Mw = [&](int w) { return m[gpl * w].ud(); };
    auto T = [&](int k) { return tmp[gpl * k].ud(); };

    mov(8, scal.uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    mov(8, scal.uw(8)(1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
    mov(L, laneIdx[0].ud(), scal.uw(0)(1));                               // lane index (dwords)
    mulConstant(1, hdr.ud(0), state.tileXorLidFlat, L);
    add(L, laneIdx[0].ud(), laneIdx[0].ud(), hdr.ud(0));                 // tile index per lane
    cmp(L | lt | f1[0], null.ud(), laneIdx[0].ud(), uint32_t(nTiles));   // lanes past the last tile hash garbage
    mov(1, scal.uw(0), f1[0]);                                            // valid-lane mask
    if (L < 16) and_(1, scal.uw(0), scal.uw(0), uint16_t((1u << L) - 1));
    // messages: four 4-word vector loads land word-major (word w of all L tiles in m[gpl*w ..])
    shl(L, hdr16[0].ud(), laneIdx[0].ud(), uint16_t(6));
    add(L, hdr16[0].ud(), hdr16[0].ud(), uint32_t(msgBase));
    for (int k = 0; k < 4; k++) {
        load(L, m[4 * gpl * k], scattered(D32, 4), SLM, hdr16[0]);
        if (k < 3) add(L, hdr16[0].ud(), hdr16[0].ud(), uint32_t(16));
    }
    shl(L, hdr16[0].ud(), laneIdx[0].ud(), uint16_t(2));
    add(L, hdr16[0].ud(), hdr16[0].ud(), uint32_t(sidBase));
    load(L, sid16[0], D32, SLM, hdr16[0]);                                // spatial ids
    shl(L, hdr16[0].ud(), laneIdx[0].ud(), uint16_t(3));
    add(L, hdr16[0].ud(), hdr16[0].ud(), uint32_t(coordBase));
    load(L, coord16[0], D64, SLM, hdr16[0]);                              // packed (t_rows, t_cols)
    if (audit) {
        // Third audit copy: the message as loaded by the hashing lane, at
        // tile_xor + 4096 + tile_count*96 + sid*64 + 4w.
        mulConstant(1, hdr.ud(1), ARG(kCoopBufTileCountDw), 96);
        add(1, hdr.ud(1), hdr.ud(1), uint32_t(kCoopAuditOff));
        shl(L, hdr16[0].ud(), sid16[0].ud(), uint16_t(6));
        add(L, hdr16[0].ud(), hdr16[0].ud(), hdr.ud(1));
        for (int h = 0; h < halves; h++)
            eadd(8, hdr64[2 * h].uq(), state.inputs.tileXor, hdr16[h].ud(), strategy, state);
        mov(1, f1[0], scal.uw(0));
        for (int w = 0; w < kBlake3MsgWords; w++) {
            if (w)
                for (int h = 0; h < halves; h++)
                    eadd(8, hdr64[2 * h].uq(), hdr64[2 * h].uq(), uint32_t(4), strategy, state);
            store(L | f1[0], scattered_dword(), A64, hdr64[0], m[gpl * w]);
        }
    }

    // keyed BLAKE3, 16 tiles per instruction: v[0..7] key, v[8..11] IV, counter 0, block length, flags
    for (int i = 0; i < kBlake3DigestWords; i++) mov(L, V(i), ARG(kCoopBufKeyDw + i));
    for (int i = 0; i < 4; i++) mov(L, V(8 + i), uint32_t(kBlake3IV[i]));
    mov(L, V(12), uint32_t(0));
    mov(L, V(13), uint32_t(0));
    mov(L, V(14), blake3BlockLen);
    mov(L, V(15), blake3Flags);
    {
        const char *a3 = std::getenv("CASE5_COOP_ADD3");          // 0: two adds (A/B timing)
        const bool useAdd3 = hw >= HW::XeHP && !(a3 && a3[0] == '0');
        emitBlake3RoundsWide(*this, V, Mw, T, L, useAdd3);
    }
    for (int i = 0; i < kBlake3DigestWords; i++) xor_(L, V(i), V(i), V(8 + i));    // digest in V(0..7)

    if (digestOut) {
        // blake3_out + sid*32 + 4w (audit: tile_xor + 4096 + sid*32), SIMD8 halves for the 64-bit math
        const auto outBase = audit ? state.inputs.tileXor : blake3Out;
        shl(L, hdr16[0].ud(), sid16[0].ud(), uint16_t(5));
        if (audit) add(L, hdr16[0].ud(), hdr16[0].ud(), uint32_t(kCoopAuditOff));
        for (int h = 0; h < halves; h++)
            eadd(8, hdr64[2 * h].uq(), outBase, hdr16[h].ud(), strategy, state);
        mov(1, f1[0], scal.uw(0));
        for (int w = 0; w < kBlake3DigestWords; w++) {
            if (w)
                for (int h = 0; h < halves; h++)
                    eadd(8, hdr64[2 * h].uq(), hdr64[2 * h].uq(), uint32_t(4), strategy, state);
            store(L | f1[0], scattered_dword(), A64, hdr64[0], v[gpl * w]);
        }
    }
    if (judge) {
        // beats = digest < bound lexicographically from word 7 down, equal counts as a hit:
        // res_w = lt_w | (eq_w & res_{w-1}), res_{-1} = valid lanes
        mov(1, scal.uw(1), scal.uw(0));
        for (int w = 0; w < kBlake3DigestWords; w++) {
            cmp(L | lt | f1[0], null.ud(), V(w), ARG(w));               // bound words at +0
            cmp(L | eq | f1[1], null.ud(), V(w), ARG(w));
            mov(1, scal.uw(2), f1[0]);
            mov(1, scal.uw(3), f1[1]);
            and_(1, scal.uw(3), scal.uw(3), scal.uw(1));
            or_(1, scal.uw(1), scal.uw(2), scal.uw(3));
        }
        and_(1, scal.uw(1), scal.uw(1), scal.uw(0));
        mov(1, f0[1], scal.uw(1));
        mov(L, ones[0].ud(), uint32_t(1));
        for (int h = 0; h < halves; h++)
            eadd(8, hdr64[2 * h].uq(), foundFlag, uint32_t(kJackpotFoundFlagOff), strategy, state);
        store(L | f0[1], scattered_dword(), A64, hdr64[0], ones[0]);          // found = 1 (any winner)
        for (int h = 0; h < halves; h++)
            eadd(8, hdr64[2 * h].uq(), foundFlag, uint32_t(kJackpotFoundCoordsOff), strategy, state);
        store(L | f0[1], scattered_qword(), A64, hdr64[0], coord16[0]);      // one winner's (t_rows, t_cols)
    }
    mark(lEnd);

    state.ra.safeRelease(scal);
    state.ra.safeRelease(ones);
    state.ra.safeRelease(hdr64);
    state.ra.safeRelease(coord16);
    state.ra.safeRelease(sid16);
    state.ra.safeRelease(hdr16);
    state.ra.safeRelease(laneIdx);
    state.ra.safeRelease(tmp);
    state.ra.safeRelease(m);
    state.ra.safeRelease(v);
    state.ra.safeRelease(args);
    state.ra.safeRelease(hdrQ);
    state.ra.safeRelease(hdr);
}

GEMMSTONE_NAMESPACE_END


