/*
 * Pearl GEMM + milestone XOR kernel for the Hexagon v66 cDSP (gemm_xor Case 1.1):
 * register tiles of 4 rows x 128 columns of C; after every 128 values of k
 * (one Pearl rank-128 milestone) the int32 elements of each 4 x 64 half
 * (columns 0..63 and 64..127, the Pearl hash tile) are XORed in registers.
 *
 * Data (prepared by pearlx_imp.c):
 *   A, per row tile and K block: chunks of 128 bytes; chunk c = 4 rows x 32
 *     values of k ([row][k], signed). One extra readable 128 bytes must follow
 *     the last chunk used (the loop loads one chunk ahead).
 *   B, per column tile and K block, in VTCM: chunks x 8 groups x 4 vectors;
 *     vector (group g, q) holds B[4g..4g+3][32q + lane], signed.
 *   Accumulators in memory: row r of column block q at base + 512r + 128q.
 *
 * vrmpyz(Vu.b, Rt.b) is the signed x signed form of the v66 Z-buffer multiply
 * (vrmpyz(Vu.b, Rt.ub) treats Z as unsigned). Z is a 256-byte window indexed by
 * the low 8 bits of addresses: a Z load from address a fills the half selected
 * by bit 7 of a. Vdddd.w += vrmpyz(Vu.b, Rt.b++) multiplies the 32 x 4 bytes of
 * Vu with window bytes (Rt - 100 + 32N + t) mod 256, t = 0..3, into vector N of
 * the quad; Rt.b++ adds 4. So for a chunk loaded from address a, Rt = a + 100,
 * a + 104, .., a + 128 steps vector N through row N's 32 values of k.
 *
 * Packet rules (assembler and hardware, checked on this DSP): a packet with
 * vrmpyz may also hold a .tmp vector load and a Z load, and no other HVX
 * instruction at all, so the XOR needs packets of its own. A Z load may share
 * a packet with a VTCM load; with a load from ordinary memory it crashes the
 * DSP process. A packet without vrmpyz takes up to 4 vxor; one vdeal plus one
 * vxor; or a vector store, a vector load and one vdeal or two vxor.
 *
 * Accumulators: quad q (columns 32q..32q+31) is v(4q+3):(4q), vector 4q+r = row r.
 * Each loop packet is one VTCM weight load (.tmp) + one vrmpyz: 32 vrmpyz in
 * 32 packets per chunk, 128 per milestone. The XOR adds a fold of 5 packets
 * per milestone, the last step of each milestone deferred into the next
 * milestone's fold; the reduction to words rides in the tile transition's
 * packets. Measured: 4.4-4.6% over the same GEMM without the XOR.
 */

/* Registers. AZ, BZ and RT are used by the inner-loop macros. */
#define AZ      r0                  /* A: next chunk */
#define BZ      r1                  /* B: next weight vector */
#define NMILE   r2
#define INM     r3                  /* starting accumulators + 1024 */
#define INSTEP  r4
#define OUTM    r5                  /* final accumulators + 1024 */
#define RT      r6
#define OSTEP   r15
#define XO      r8                  /* XOR record line of the current tile */
#define XOSTEP  r9
#define ROT     r10
#define AZT     r11                 /* A of the current tile */
#define AZSTEP  r12
#define BZ0     r13
#define TMP     r14
#define DEAL    r7                  /* vdeal takes its size register from r0..r7 */
#define L2CTL   r17:16              /* l2fetch box: one tile's A (+ its partial sums) */
#define NLEFT   r18                 /* tiles after the first */

/* One group: 4 vrmpyz (column blocks q = 0..3) at the same Rt, then Rt += 4. */
#define GROUP                                                           \
    { v24.tmp = vmem(BZ++#1) ; v3:0.w   += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v7:4.w   += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v11:8.w  += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v15:12.w += vrmpyz(v24.b, RT.b++) }

/* One chunk (32 values of k). The first packet loads the NEXT chunk into Z (AZ
 * points at it): being 128 bytes further on, it lands in the other half of the
 * window, so the current chunk stays readable. Rt runs chunk address + 100 ..
 * + 128; the final packet moves it on by 100: 128 = one chunk in total with
 * the 7 increments of 4. (A macro puts a packet on one line: ';' separates.) */
#define CHUNK                                                           \
    { v24.tmp = vmem(BZ++#1) ; v3:0.w   += vrmpyz(v24.b, RT.b) ; z = vmem(AZ+#0) } \
    { v24.tmp = vmem(BZ++#1) ; v7:4.w   += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v11:8.w  += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v15:12.w += vrmpyz(v24.b, RT.b++) }    \
    GROUP GROUP GROUP GROUP GROUP GROUP                                 \
    { v24.tmp = vmem(BZ++#1) ; v3:0.w   += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v7:4.w   += vrmpyz(v24.b, RT.b)   }    \
    { v24.tmp = vmem(BZ++#1) ; v11:8.w  += vrmpyz(v24.b, RT.b) ; AZ = add(AZ, #128) } \
    { v24.tmp = vmem(BZ++#1) ; v15:12.w += vrmpyz(v24.b, RT.b) ; RT = add(RT, #100) }

/* Store vX of the finished tile and load its starting value for the next,
 * with one more HVX instruction (a vdeal, or one or two vxor) of the XOR
 * reduction. ioff includes its '#': '#param' would stringize in a macro. */
#define SWAP(v, ioff) { vmem(OUTM+ioff) = v ; v = vmem(INM+ioff) }
#define SWAPX(v, ioff, x) { vmem(OUTM+ioff) = v ; v = vmem(INM+ioff) ; x }
#define SWAPXX(v, ioff, x, y) { vmem(OUTM+ioff) = v ; v = vmem(INM+ioff) ; x ; y }


    .text

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define LBL(x) CAT(LP, x)               /* labels of the pearlx_xor.inc copy */

/* Fold for milestone i: XOR each half's 8 accumulators down to one vector,
 * XA (quads 0-1: v0..v7, columns 0..63) and XB (quads 2-3: v8..v15, columns
 * 64..127), and finish the previous milestone's pair PA, PB (PP = PB:PA).
 * vdeal splits the 64 words of PB:PA into even and odd words, so XORing the
 * two results gives lanes 0..15 = XORs of PA's word pairs and lanes 16..31 =
 * those of PB: one vector FPREV that keeps the halves apart. The pairs
 * alternate between v31:30 (even i) and v29:28 (odd i), so a milestone's pair
 * survives into the next fold; v25..v27 are temporaries. 15 vxor and a vdeal:
 * 5 packets, since a packet with a vdeal takes only one vxor. Quads 0..3 got
 * their last vrmpyz 4..1 packets before the first of these packets; quads 2
 * and 3 are first read in packets 2 and 3. */
#define FOLD5(XA, XB, PA, PB, PP, FPREV)                                                     \
    { v25 = vxor(v0, v1)   ; v26 = vxor(v2, v3)  ; v27 = vxor(v4, v5)   ; XA = vxor(v6, v7) }    \
    { v25 = vxor(v25, v26) ; v27 = vxor(v27, XA) ; v26 = vxor(v8, v9)   ; XB = vxor(v10, v11) }  \
    { XA = vxor(v25, v27)  ; v26 = vxor(v26, XB) ; v25 = vxor(v12, v13) ; v27 = vxor(v14, v15) } \
    { v25 = vxor(v25, v27) ; PP = vdeal(PB, PA, DEAL) }                                         \
    { XB = vxor(v26, v25)  ; FPREV = vxor(PA, PB) }

/* Tile transition: milestone 7's deal and XOR, then three rounds of dealing
 * two vectors' words apart and XORing the halves (8 vectors -> one; half h of
 * milestone m in lanes 4m + 2h and 4m + 2h + 1), one round on the vector with
 * itself (word 2m + h in lanes 0..15), and a vror into place. */
#define TRANSITION5                                                     \
    {   INM = add(INM, INSTEP) ; AZT = add(AZT, AZSTEP) ;               \
        v29:28 = vdeal(v29, v28, DEAL) }                                \
    SWAPX(v0, #-8, v17:16 = vdeal(v17, v16, DEAL))                      \
    SWAPXX(v1, #-4, v23 = vxor(v28, v29), v16 = vxor(v16, v17))         \
    SWAPX(v2, #0, v19:18 = vdeal(v19, v18, DEAL))                       \
    SWAPX(v3, #4, v21:20 = vdeal(v21, v20, DEAL))                       \
    SWAPX(v4, #-7, v23:22 = vdeal(v23, v22, DEAL))                      \
    SWAPXX(v5, #-3, v18 = vxor(v18, v19), v20 = vxor(v20, v21))         \
    SWAPX(v6, #1, v17:16 = vdeal(v18, v16, DEAL))                       \
    SWAPXX(v7, #5, v22 = vxor(v22, v23), v16 = vxor(v16, v17))          \
    SWAPX(v8, #-6, v19:18 = vdeal(v22, v20, DEAL))                      \
    SWAPX(v9, #-2, v18 = vxor(v18, v19))                                \
    SWAPX(v10, #2, v17:16 = vdeal(v18, v16, DEAL))                      \
    SWAPX(v11, #6, v16 = vxor(v16, v17))                                \
    SWAPX(v12, #-5, v17:16 = vdeal(v16, v16, DEAL))                     \
    SWAPX(v13, #-1, v16 = vxor(v16, v17))                               \
    SWAPX(v14, #3, v16 = vror(v16, ROT))                                \
    SWAP(v15, #7)

#define NAME pearlx_rows
#define LP px_
#define FOLD FOLD5
#define TRANSITION TRANSITION5
#include "pearlx_xor.inc"
#undef NAME
#undef LP
#undef FOLD
#undef TRANSITION
