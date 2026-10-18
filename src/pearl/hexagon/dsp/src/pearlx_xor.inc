/*
 * void NAME(const struct pearlx_rows_args *a)   (see pearlx_imp.c)
 *
 * ntiles consecutive row tiles for one column tile and K block, with the
 * milestone XOR. pearlx.S includes this with NAME, LP (the label prefix), FOLD
 * and TRANSITION defined.
 *
 * The K block's milestones are unrolled (copies c0..c7, each 4 chunks + a
 * fold), so each milestone's XOR lands in its own register v16 + i with no
 * copying; a block with fewer milestones starts at copy 8 - nmile. The fold of
 * milestone i also finishes milestone i - 1; copy 0 finishes a nonexistent one
 * into v24 (a .tmp register, otherwise unused), and a block starting later
 * leaves garbage only in lanes the store masks off.
 *
 * Between tiles, TRANSITION's 16 swap packets (store tile t's accumulators,
 * load tile t + 1's) also run the reduction of v16..v23 to words and a vror
 * into place. A masked store writes bytes [qlo, qhi) of the tile's record
 * line. The last tile runs the same transition: the extra loads and prefetches
 * touch the region's padding tiles.
 */
    .global NAME
    .type   NAME, @function
    .balign 32
NAME:
    {   allocframe(#16) }
    {   memd(r29+#0) = r17:16 }
    {   memd(r29+#8) = r19:18 }
    {   AZT = memw(r0+#0)
        AZSTEP = memw(r0+#4)
    }
    {   BZ0 = memw(r0+#8)
        NMILE = memw(r0+#12)
    }
    {   INM = memw(r0+#16)
        INSTEP = memw(r0+#20)
    }
    {   OUTM = memw(r0+#24)
        OSTEP = memw(r0+#28)
    }
    {   XO = memw(r0+#32)
        XOSTEP = memw(r0+#36)
    }
    {   NLEFT = memw(r0+#40)        /* here: all the tiles */
        ROT = memw(r0+#44)
    }
    {   r16 = memw(r0+#48)          /* l2fetch box: width << 16 | height */
        r17 = memw(r0+#52)          /* and row stride */
    }
    {   TMP = memw(r0+#56)
        DEAL = memw(r0+#60)
    }
    {   q1 = vsetq2(DEAL) }         /* bytes [0, qhi) */
    {   q2 = vsetq(TMP) }           /* bytes [0, qlo) */
    {   q0 = and(q1, !q2)
        DEAL = #-4                  /* vdeal of 4-byte elements */
        INM = add(INM, #1024)
        OUTM = add(OUTM, #1024)
    }
    /* Entry: copy 8 - nmile of the unrolled milestones (all copies are the
     * same size). Both addresses are PC-relative: a label difference would
     * need an absolute relocation, which a position-independent library can't
     * take. (add(pc) only issues in slot 3: one per packet.) RT is free until
     * the first z load. */
    {   TMP = add(pc, ##LBL(c0)@PCREL) }
    {   r19 = add(pc, ##LBL(c1)@PCREL) }
    {   r19 = sub(r19, TMP)
        RT = sub(#8, NMILE)
    }
    {   r19 = mpyi(r19, RT) }
    {   r19 = add(r19, TMP) }
    /* First tile's accumulators: quad 0 first, since the loop needs it first. */
    {   v0 = vmem(INM+#-8) }
    {   v1 = vmem(INM+#-4) }
    {   v2 = vmem(INM+#0) }
    {   v3 = vmem(INM+#4) }
    {   v4 = vmem(INM+#-7) }
    {   v5 = vmem(INM+#-3) }
    {   v6 = vmem(INM+#1) }
    {   v7 = vmem(INM+#5) }
    {   v8 = vmem(INM+#-6) }
    {   v9 = vmem(INM+#-2) }
    {   v10 = vmem(INM+#2) }
    {   v11 = vmem(INM+#6) }
    {   v12 = vmem(INM+#-5) }
    {   v13 = vmem(INM+#-1) }
    {   v14 = vmem(INM+#3) }
    {   v15 = vmem(INM+#7) }
    {   z = vmem(AZT+#0)
        AZ = add(AZT, #128)
        RT = add(AZT, #100)
    }
    {   BZ = BZ0
        TMP = add(AZT, AZSTEP)
    }
    {   l2fetch(TMP, L2CTL) }       /* only ALU instructions may share its packet */
    {   loop1(LBL(tile), NLEFT) }

    .balign 32
LBL(tile):
    {   jumpr r19 }
LBL(c0):
    CHUNK CHUNK CHUNK CHUNK FOLD(v30, v31, v28, v29, v29:28, v24)
LBL(c1):
    CHUNK CHUNK CHUNK CHUNK FOLD(v28, v29, v30, v31, v31:30, v16)
    CHUNK CHUNK CHUNK CHUNK FOLD(v30, v31, v28, v29, v29:28, v17)
    CHUNK CHUNK CHUNK CHUNK FOLD(v28, v29, v30, v31, v31:30, v18)
    CHUNK CHUNK CHUNK CHUNK FOLD(v30, v31, v28, v29, v29:28, v19)
    CHUNK CHUNK CHUNK CHUNK FOLD(v28, v29, v30, v31, v31:30, v20)
    CHUNK CHUNK CHUNK CHUNK FOLD(v30, v31, v28, v29, v29:28, v21)
    CHUNK CHUNK CHUNK CHUNK FOLD(v28, v29, v30, v31, v31:30, v22)

    /* Tile done: milestone 7 (in v28:v29) and the reduction of v16..v23,
     * inside the swap packets. */
    TRANSITION
    {   z = vmem(AZT+#0)
        AZ = add(AZT, #128)
        RT = add(AZT, #100)
    }
    {   BZ = BZ0
        TMP = add(AZT, AZSTEP)
        OUTM = add(OUTM, OSTEP)
        if (q0) vmem(XO+#0) = v16
    }
    {   l2fetch(TMP, L2CTL)
        XO = add(XO, XOSTEP)
    }:endloop1

    {   r17:16 = memd(r29+#0) }
    {   r19:18 = memd(r29+#8) }
    {   dealloc_return }
    .size NAME, .-NAME
