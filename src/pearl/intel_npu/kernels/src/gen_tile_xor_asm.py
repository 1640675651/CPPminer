"""Generate the hand-scheduled SHAVE assembly behind tile_xor_acc: tx_strips_g4 (G = 4 partial sums per pass,
described here), tx_strips_g8 (eight partials in one pass with a loop body shared by both halves of a tile, see
generate_g8) and tx_xor_strips (XOR-only, see generate_xor).

    void tx_strips_g4(const unsigned *records);

processes a list of strips. A strip is `ntiles` consecutive 8x16 int32 tiles down one 16-column strip. A tile is done
in four 2-row quarters (8 int32x4 vectors); in each quarter, for every partial g, the VAU does 8 adds
(acc += partial_g) and 8 xors into X_g, one VAU op per bundle. Memory traffic is interleaved so that loads are issued
8 cycles before use. The strip loop is part of the routine: C code running between strips evicts the tile loop
(about 2 KiB) from the instruction cache, which cost about 1000 stall cycles per strip.

Rules of the NPU3720 ACT-SHAVE this schedule obeys (found by measurement; the assembler checks only some of them):
  * LSU load latency 7, VAU add 2, VAU xor 1. An XOR issued right after an ADD would write the vector register
    file in the same cycle (port conflict), hence one bubble per slot: a slot is 8 adds, 1 bubble, 8 xors.
  * Loads and stores are 128-bit post-incrementing (LDI.128 / STI.128). Half the LSU operations of 64-bit
    accesses, and much fewer stalls. A post-incremented pointer may be used again only 2 cycles later, and must
    not be written by another unit within 2 cycles (the write-back races); LSU0/LSU1 accesses to the two row
    pointers are issued together in the same bundle (issuing them on alternate cycles hangs the device).
  * Every 128-bit access needs idle LSUs in the cycles before and after it (otherwise the device hangs).
  * SAU arithmetic cannot share a bundle with the VAU, so pointer updates use the IAU (one per bundle).
  * PEU.PC1C predicates every op of its bundle, so branches have a bundle of their own (6 delay slots).
  * The instruction cache holds the G = 4 tile loop but not the twice as long G = 8 one (or a loop unrolled to two
    tiles per iteration): those run at about the same active cycles with 2-3x the stall cycles. Eight partials
    therefore use a two-quarter body shared by both halves of a tile (generate_g8); other larger counts run as
    several G = 4 passes (tile_xor_acc.c).

Software pipeline (b = 17 g is the start of slot g of a 17 G-cycle quarter):
  b + j        VAU  S[o_j] += T[o_j]             S: accumulator set of this quarter (A or B, alternating)
  b + 8        bubble
  b + 9 + j    VAU  X_g ^= S[o_j]               (first quarter of a tile: X_g = S[o_0] ^ S[o_1], ...)
  b + 9 + 2k   LSU  T <- partial (g + 1) % G     (row 0 on LSU0, row 1 on LSU1; slot G-1 loads the next quarter)
  slot 0       LSU  O (previous quarter's set) -> acc_out, cycles 0, 2, 4, 6
  slot G // 2  LSU  O <- acc_in of the next quarter, cycles b, b + 2, b + 4, b + 6
  o = 0 4 1 5 2 6 3 7 (row 0 and row 1 vectors interleaved, matching the load order)
In the first quarter of the next tile, X_g is folded to one word by SAU.XORX and stored straight to xor_out through
a pointer per partial. The first tile of a strip runs through a peeled copy of that quarter (and of the head of the
second one) without those stores, then jumps into the loop.

Registers: v0-v7 A, v8-v15 B, v16-v23 T, v24.. X (callee-saved, spilled); integer registers from i0 upwards,
skipping i19 (stack pointer) and i29 (record pointer); i20 and above are callee-saved and spilled.
A strip record (32-bit words): ai0 ai1 air2 | p00 p01 ... p(G-1)0 p(G-1)1 | pr2 | ao0 ao1 an0 an1 aor2 |
                               xp0 ... xp(G-1) xr | ntiles | more
  ai*:  acc_in rows 0/1 of the first quarter; air2 = 2 * row stride - 64 (-64 for the 64-byte zero buffer)
  p*:   rows 0/1 of the first quarter of partial g; pr2 = 2 * row stride - 64
  ao*:  a 128-byte dummy for the first (garbage) store; an*: acc_out rows 0/1 of the first quarter;
        aor2 = 2 * row stride (ao is reloaded from an every quarter)
  xp*:  xor_out word of the first tile for partial g; xr = xor_out row stride in bytes
  more: nonzero if another record follows
"""

import re
import sys


A = [f"v{i}" for i in range(0, 8)]
B = [f"v{i}" for i in range(8, 16)]
T = [f"v{i}" for i in range(16, 24)]
ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
SLOT = 17
# i19 is the stack pointer; i29 holds the record pointer; i30 (return address) is spilled and reused.
INT_REGS = [f"i{k}" for k in range(19)] + [f"i{k}" for k in range(20, 29)] + ["i30"]
CALLEE_SAVED_INT = {f"i{k}" for k in range(20, 31)}
RECORDS = "i29"
# The loop is entered at this cycle of quarter 1, after a peeled copy of the first tile's quarter 0 and quarter 1
# head that leaves out the stores of the (nonexistent) previous tile's XOR words.
ENTRY_CYCLE = SLOT + 8


class Layout:
    def __init__(self, g):
        assert 4 <= g <= 8, "G < 4 leaves no slot free of 128-bit accumulator traffic for the result stores"
        self.G = g
        self.X = [f"v{24 + k}" for k in range(g)]
        names = ["ai0", "ai1", "air2"] + [f"p{k}{r}" for k in range(g) for r in (0, 1)] + \
                ["pr2", "ao0", "ao1", "an0", "an1", "aor2"] + [f"xp{k}" for k in range(g)] + ["xr", "cnt", "more"]
        folded = [f"x{k}" for k in range(4)]   # folded XOR words (rotating)
        assert len(names) + len(folded) <= len(INT_REGS), f"G = {g} needs more integer registers"
        self.args = list(names)            # record order == register order
        self.reg = dict(zip(names + folded, INT_REGS))
        self.R = [self.reg[x] for x in folded]
        self.P = [(self.reg[f"p{k}0"], self.reg[f"p{k}1"]) for k in range(g)]
        self.quarter_len = SLOT * g


def free_iau(ops, c):
    """First cycle >= c without an IAU op."""
    while any(cc == c and op.startswith("IAU") for cc, op in ops):
        c += 1
    return c


def quarter(L, qq, first=False):
    """Bundles (lists of ops) of quarter qq (0..3) of the tile loop body (first: without previous-tile stores)."""
    r, G, X = L.reg, L.G, L.X
    S, O = (A, B) if qq % 2 == 0 else (B, A)
    acc_load_slot = G // 2
    ops = []
    for g in range(G):
        b = SLOT * g
        nxt = L.P[(g + 1) % G]
        for j, i in enumerate(ORDER):
            ops.append((b + j, f"VAU.ADD.i32 {S[i]} {S[i]} {T[i]}"))
            c = b + 9 + j
            if qq == 0:
                if j == 0:
                    ops.append((c, f"VAU.XOR {X[g]} {S[ORDER[0]]} {S[ORDER[1]]}"))
                elif j < 7:
                    ops.append((c, f"VAU.XOR {X[g]} {X[g]} {S[ORDER[j + 1]]}"))
            else:
                ops.append((c, f"VAU.XOR {X[g]} {X[g]} {S[i]}"))
            ops.append((b + 9 + 2 * (j // 2), f"LSU{j % 2}.LDI.128 {T[i]} {nxt[i // 4]}"))
        for k in range(4):
            if g == 0:
                ops += [(b + 2 * k, f"LSU0.STI.128 {O[k]} {r['ao0']}"),
                        (b + 2 * k, f"LSU1.STI.128 {O[k + 4]} {r['ao1']}")]
            if g == acc_load_slot:
                ops += [(b + 2 * k, f"LSU0.LDI.128 {O[k]} {r['ai0']}"),
                        (b + 2 * k, f"LSU1.LDI.128 {O[k + 4]} {r['ai1']}")]
    # Pointer updates (IAU), each >= 3 cycles after the last post-increment of the register.
    for g in range(G):      # partial (g + 1) % G was loaded in slot g; slot g + 1 starts at SLOT (g + 1)
        p = L.P[(g + 1) % G]
        c = (SLOT * (g + 1) + 1) % L.quarter_len
        ops += [(c, f"IAU.ADD {p[0]} {p[0]} {r['pr2']}"), (c + 1, f"IAU.ADD {p[1]} {p[1]} {r['pr2']}")]
    ops += [(9, f"IAU.ADD {r['ao0']} {r['an0']} 0"), (10, f"IAU.ADD {r['ao1']} {r['an1']} 0"),
            (11, f"IAU.ADD {r['an0']} {r['an0']} {r['aor2']}"), (12, f"IAU.ADD {r['an1']} {r['an1']} {r['aor2']}")]
    c = SLOT * acc_load_slot + 9
    ops += [(c, f"IAU.ADD {r['ai0']} {r['ai0']} {r['air2']}"), (c + 1, f"IAU.ADD {r['ai1']} {r['ai1']} {r['air2']}")]
    last = SLOT * (G - 1)
    if qq == 3:
        ops += [(last + 6, f"IAU.INCS {r['cnt']} -1"), (last + 7, f"CMU.CMZ.i32 {r['cnt']}")]
    # Tile XOR results. In the first quarter of the next tile, X_g is folded to one word by SAU.XORX in slot g's
    # bubble (the SAU cannot share a bundle with the VAU; X_g is re-initialized the cycle after) and stored to
    # xor_out, whose pointer then moves on by a row. The stores go to slots without 128-bit accumulator traffic
    # (a 128-bit access needs idle LSU neighbours); the last one moves to slot 1 of the second quarter.
    if qq == 0:
        used = {}
        for g in range(G):
            ops.append((SLOT * g + 8, f"SAU.XORX.x32 {L.R[g % len(L.R)]} {X[g]}"))
            if g == G - 1 or first:
                continue
            slot = next(s for s in range(g + 1, G) if s not in (0, acc_load_slot))
            k = used.get(slot, 0)
            used[slot] = k + 1
            assert k < 4, (G, slot)
            store = SLOT * slot + 2 * k
            # the register is reused by the fold of partial g + len(R)
            assert g + len(L.R) >= G or store < SLOT * (g + len(L.R)) + 8, (G, g)
            ops.append((store, f"LSU0.STO.32 {L.R[g % len(L.R)]} {r[f'xp{g}']} 0"))
            ops.append((free_iau(ops, store + 3), f"IAU.ADD {r[f'xp{g}']} {r[f'xp{g}']} {r['xr']}"))
    if qq == 1 and not first:
        ops.append((SLOT, f"LSU0.STO.32 {L.R[(G - 1) % len(L.R)]} {r[f'xp{G - 1}']} 0"))
        c = free_iau(ops, SLOT + 3)
        assert c < ENTRY_CYCLE, c
        ops.append((c, f"IAU.ADD {r[f'xp{G - 1}']} {r[f'xp{G - 1}']} {r['xr']}"))
    bundles = [[] for _ in range(L.quarter_len)]
    for c, op in ops:
        unit = op[:4] if op.startswith("LSU") else op.split(".")[0]
        assert all((o[:4] if o.startswith("LSU") else o.split(".")[0]) != unit for o in bundles[c]), (G, qq, c, op)
        bundles[c].append(op)
    return bundles


def check_wide_access_isolation(quarters):
    """Every 128-bit LSU access needs idle LSUs in the cycles before and after it (otherwise the device hangs)."""
    flat = [b for q in quarters for b in q]
    n = len(flat)
    for c, bundle in enumerate(flat):
        if any(re.match(r"LSU\d\.\w+\.128", op) for op in bundle):
            for dc in (-1, 1):
                lsu = [op for op in flat[(c + dc) % n] if op.startswith("LSU")]
                if lsu:
                    raise AssertionError(f"128-bit access at {c} next to {lsu} at {c + dc}")


def check_post_increment_hazards(quarters):
    """A post-incrementing access on pointer p at cycle c: another post-incrementing access may use p again from
    c+2 on; anything else (an IAU update) must wait until c+3."""
    flat = [b for q in quarters for b in q]
    n = len(flat)
    for c, bundle in enumerate(flat):
        for op in bundle:
            m = re.match(r"LSU\d\.(?:LDI|STI)\.\S+ \S+ (i\d+)", op)
            if not m:
                continue
            for dc in (1, 2):
                for other in flat[(c + dc) % n]:
                    if dc == 2 and re.match(r"LSU\d\.(?:LDI|STI)\.", other):
                        continue
                    if re.search(rf"\b{m.group(1)}\b", other):
                        raise AssertionError(f"post-increment hazard: '{op}' at {c}, '{other}' at {c + dc}")


def emit(bundle):
    if not bundle:
        return ["        NOP"]
    if bundle[0].startswith("NOP"):
        return [f"        {bundle[0]}"]
    return [f"        {bundle[0]}"] + [f"            || {op}" for op in bundle[1:]]


def generate(g):
    L = Layout(g)
    r, X = L.reg, L.X
    name = f"tx_strips_g{g}"
    saved_int = sorted({x for x in L.reg.values() if x in CALLEE_SAVED_INT} | {RECORDS, "i30"},
                       key=lambda x: int(x[1:]))
    frame = 16 * len(X) + 8 * len(saved_int)
    frame = (frame + 15) // 16 * 16
    head = [[f"IAU.SUB i19 i19 {frame}"]]
    for k, v in enumerate(X):
        head.append([f"LSU0.STO.64 {v}.0 i19 {16 * k}", f"LSU1.STO.64 {v}.1 i19 {16 * k + 8}"])
    for k, reg in enumerate(saved_int):
        head.append([f"LSU0.STO.32 {reg} i19 {16 * len(X) + 4 * k}"])
    head.append([f"CMU.CP.32 {RECORDS} i18"])
    # Per strip: load the record, fill the pipeline (accumulator of quarter 0 into A, partial 0 into T).
    strip = []
    args = [r[n] for n in L.args]
    for k in range(0, len(args), 2):
        strip.append([f"LSU{m}.LDO.32 {args[k + m]} {RECORDS} {4 * (k + m)}" for m in range(2) if k + m < len(args)])
    strip.append(["NOP 6"])
    for k in range(4):
        strip.append([f"LSU0.LDI.128 {A[k]} {r['ai0']}", f"LSU1.LDI.128 {A[k + 4]} {r['ai1']}"])
        strip.append(["NOP"])
    for k in range(4):
        strip.append([f"LSU0.LDI.128 {T[k]} {L.P[0][0]}", f"LSU1.LDI.128 {T[k + 4]} {L.P[0][1]}"])
        strip.append(["NOP"])
    strip += [["NOP"], [f"IAU.ADD {r['ai0']} {r['ai0']} {r['air2']}"],
              [f"IAU.ADD {r['ai1']} {r['ai1']} {r['air2']}"], ["NOP 6"]]
    lines = [f".code      .text.{name}", f".L.text.{name}:", "    .align     16", f"\t.type\t{name},@function",
             f"    {name}:"]
    for b in head:
        lines += emit(b)
    lines.append(f".L{name}_strip:")
    for b in strip:
        lines += emit(b)
    # The loop's first P0 update (quarter cycle 1) expects the prologue's post-increments, as in steady state.
    peeled = quarter(L, 0, first=True) + quarter(L, 1, first=True)[:ENTRY_CYCLE]
    for c, b in enumerate(peeled):
        if c == len(peeled) - 6:
            lines.append(f"        BRU.BRA .L{name}_entry")
        lines += emit(b)
    lines.append(f".L{name}_loop:")
    quarters = [quarter(L, qq) for qq in range(4)]
    check_post_increment_hazards(quarters)
    check_wide_access_isolation(quarters)
    for qq, bundles in enumerate(quarters):
        for c, b in enumerate(bundles):
            if qq == 1 and c == ENTRY_CYCLE:
                lines.append(f".L{name}_entry:")
            if qq == 3 and c == L.quarter_len - 6:
                lines += ["        PEU.PC1C NEQ", f"            || BRU.BRA .L{name}_loop"]
            lines += emit(b)
    # End of a strip: last quarter's accumulator (set B) and the last tile's XOR words; then the next record.
    tail = [["NOP 2"]]
    for k in range(4):
        tail.append([f"LSU0.STI.128 {B[k]} {r['ao0']}", f"LSU1.STI.128 {B[k + 4]} {r['ao1']}"])
        tail.append(["NOP"])
    for g, v in enumerate(X):      # no VAU ops here, so SAU.XORX is free to issue
        tail += [[f"SAU.XORX.x32 {L.R[0]} {v}"], ["NOP 3"], [f"LSU0.STO.32 {L.R[0]} {r[f'xp{g}']} 0"]]
    tail += [[f"CMU.CMZ.i32 {r['more']}"], ["NOP 2"], ["PEU.PC1C NEQ", f"BRU.BRA .L{name}_strip"],
             [f"IAU.ADD {RECORDS} {RECORDS} {4 * len(args)}"], ["NOP 5"]]
    for k, v in enumerate(X):
        tail.append([f"LSU0.LDO.64 {v}.0 i19 {16 * k}", f"LSU1.LDO.64 {v}.1 i19 {16 * k + 8}"])
    for k, reg in enumerate(saved_int):
        tail.append([f"LSU0.LDO.32 {reg} i19 {16 * len(X) + 4 * k}"])
    tail.append(["NOP 6"])
    for b in tail:
        lines += emit(b)
    lines += ["        BRU.JMP i30", f"        IAU.ADD i19 i19 {frame}", "        NOP 5",
              f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}", ""]
    return "\n".join(lines) + "\n"


G8 = 8
G8_SLOT_IAU = (5, 6)          # cycles of a slot at which the next partial's row pointers are recomputed
G8_OFF_UPDATE = 110           # first of four IAU cycles advancing the row offsets by two rows (after slot 6's use)
G8_FOLD_SLOTS = (1, 2, 3, 7)  # X_0..X_3 are folded in the bubbles of these odd-quarter slots (X_3 late: R3 is a temp)
G8_TILE_END = 12              # cycles of the tile-end block (plus the branch bundle)
# The peeled even quarter branches to the odd quarter's first bundle.
G8_ENTRY_CYCLE = 0


class LayoutG8:
    """Registers of tx_strips_g8. The record order equals the register order (loaded in pairs by the prologue).

    b0..b7: partial g at the strip's first quarter (row 0); off0, off1: byte offsets of rows 0 and 1 of the quarter
    being loaded (0 and the row stride at first; both advance by two rows per quarter). ai*: acc_in rows 0/1,
    air2 = 2 * acc_in row stride - 64 (-64 for the 64-byte zero buffer with stride 0); an*: acc_out rows 0/1,
    aor2m = 2 * acc_out row stride - 64. xq0..xq3: xor_out words of partials 0..3 of the first tile; xplane4 =
    4 * partial stride of xor_out; xback = xor_out row stride - 4 * partial stride. cnt = half tiles (4 rows each:
    2 * ntiles for 8-row tiles, 4 * ntiles for 16-row tiles).
    The record ends with `more` (nonzero if another record follows), read by the tail only.
    """

    def __init__(self):
        self.G = G8
        self.X = [f"v{24 + k}" for k in range(G8)]
        names = [f"b{k}" for k in range(G8)] + ["off0", "off1", "ai0", "ai1", "air2", "an0", "an1", "aor2m"] + \
                [f"xq{k}" for k in range(4)] + ["xplane4", "xback", "cnt"]
        temps = ["w0", "w1"] + [f"R{k}" for k in range(4)]
        assert len(names) + len(temps) == len(INT_REGS), (len(names), len(temps))
        self.args = list(names)
        self.reg = dict(zip(names + temps, INT_REGS))
        self.R = [self.reg[f"R{k}"] for k in range(4)]
        self.quarter_len = SLOT * G8
        self.record_words = len(names) + 1   # + more


def place(ops, length, where):
    bundles = [[] for _ in range(length)]
    for c, op in ops:
        unit = op[:4] if op.startswith("LSU") else op.split(".")[0]
        assert 0 <= c < length, (where, c, op)
        assert all((o[:4] if o.startswith("LSU") else o.split(".")[0]) != unit for o in bundles[c]), (where, c, op)
        bundles[c].append(op)
    return bundles


def quarter_g8(L, parity, first=False, tile_rows=8):
    """Bundles of one quarter of the shared G = 8 body. parity 0: accumulate into A, store/load B; 1: the reverse.
    first: the peeled first quarter of a strip (no store of the nonexistent previous quarter). tile_rows: 8 or 16,
    i.e. 2 or 4 half tiles (two quarters, 4 rows) per tile."""
    r, X = L.reg, L.X
    S, O = (A, B) if parity == 0 else (B, A)
    w0, w1 = r["w0"], r["w1"]
    ops = []
    for g in range(G8):
        b = SLOT * g
        nxt = r[f"b{(g + 1) % G8}"]
        for j, i in enumerate(ORDER):
            ops.append((b + j, f"VAU.ADD.i32 {S[i]} {S[i]} {T[i]}"))
            ops.append((b + 9 + j, f"VAU.XOR {X[g]} {X[g]} {S[i]}"))
            ops.append((b + 9 + 2 * (j // 2), f"LSU{j % 2}.LDI.128 {T[i]} {(w0, w1)[i // 4]}"))
        # Row pointers of the next partial: the quarter's offsets, or the next quarter's for partial 0 (slot 7),
        # whose offsets advanced at G8_OFF_UPDATE.
        ops += [(b + G8_SLOT_IAU[0], f"IAU.ADD {w0} {nxt} {r['off0']}"),
                (b + G8_SLOT_IAU[1], f"IAU.ADD {w1} {nxt} {r['off1']}")]
        for k in range(4):
            if g == 0 and not first:
                ops += [(b + 2 * k, f"LSU0.STI.128 {O[k]} {r['an0']}"),
                        (b + 2 * k, f"LSU1.STI.128 {O[k + 4]} {r['an1']}")]
            if g == G8 // 2:
                ops += [(b + 2 * k, f"LSU0.LDI.128 {O[k]} {r['ai0']}"),
                        (b + 2 * k, f"LSU1.LDI.128 {O[k + 4]} {r['ai1']}")]
    if not first:
        ops += [(9, f"IAU.ADD {r['an0']} {r['an0']} {r['aor2m']}"), (10, f"IAU.ADD {r['an1']} {r['an1']} {r['aor2m']}")]
    c = SLOT * (G8 // 2) + 9
    ops += [(c, f"IAU.ADD {r['ai0']} {r['ai0']} {r['air2']}"), (c + 1, f"IAU.ADD {r['ai1']} {r['ai1']} {r['air2']}")]
    # off0 += 2 s, off1 += 2 s with s = off1 - off0 (R3 is free here: X_3 is folded later, in slot 7).
    c = G8_OFF_UPDATE
    tmp = L.R[3]
    ops += [(c, f"IAU.SUB {tmp} {r['off1']} {r['off0']}"), (c + 1, f"IAU.ADD {r['off0']} {r['off1']} {tmp}"),
            (c + 2, f"IAU.ADD {r['off1']} {r['off1']} {tmp}"), (c + 3, f"IAU.ADD {r['off1']} {r['off1']} {tmp}")]
    if parity == 1:
        # Fold X_0..X_3 once their last XOR of the quarter is done (slot g ends at 17 g + 16; the bubble of slot
        # s is at 17 s + 8). The tile-end block folds the rest.
        for k, s in enumerate(G8_FOLD_SLOTS):
            assert SLOT * s + 8 > SLOT * k + 16
            ops.append((SLOT * s + 8, f"SAU.XORX.x32 {L.R[k]} {X[k]}"))
        # Half-tile counter: not a multiple of the half tiles per tile -> back to the body, else tile end (falls
        # through the delay slots). The folds above run every odd quarter; the tile end stores the last ones.
        shift = {8: 31, 16: 30}[tile_rows]
        ops += [(115, f"IAU.INCS {r['cnt']} -1"), (116, f"IAU.SHL {tmp} {r['cnt']} {shift}"), (118, f"CMU.CMZ.i32 {tmp}")]
        assert 118 < SLOT * G8_FOLD_SLOTS[3] + 8
    return place(ops, L.quarter_len, ("quarter", parity, first))


def tile_end_g8(L):
    """Bundles after an even half-tile count: fold X_4..X_7, store the eight words of the finished tile, zero X,
    advance the word pointers, and branch back unless cnt is 0. The SAU folds (cycles 0-3) and the VAU resets
    (4-11) cannot share bundles; the branch is at cycle 6 and the rest fills its delay slots."""
    r, X, R = L.reg, L.X, L.R
    xq = [r[f"xq{k}"] for k in range(4)]
    ops = [(0, f"CMU.CMZ.i32 {r['cnt']}")]
    for k in range(4):
        ops += [(k, f"LSU0.STO.32 {R[k]} {xq[k]} 0"),               # X_k's word (folded in the odd quarter)
                (k, f"SAU.XORX.x32 {R[k]} {X[4 + k]}"),              # the store reads R_k first; SAU writes later
                (k + 1, f"IAU.ADD {xq[k]} {xq[k]} {r['xplane4']}"),
                (k + 4, f"LSU0.STO.32 {R[k]} {xq[k]} 0"),           # X_{4+k}'s word, 4 cycles after its fold
                (k + 5, f"IAU.ADD {xq[k]} {xq[k]} {r['xback']}"),
                (k + 4, f"VAU.XOR {X[k]} {X[k]} {X[k]}"),
                (k + 8, f"VAU.XOR {X[4 + k]} {X[4 + k]} {X[4 + k]}")]
    return place(ops, G8_TILE_END, "tile end")


def generate_g8(tile_rows=8):
    """Single-pass G = 8 strips: tx_strips_g8(const unsigned *records) for 8-row tiles, tx_strips_g8_r16 for
    16-row tiles (cnt = 4 * ntiles there); see LayoutG8 for the record.

    The loop body is one even and one odd quarter (272 cycles, the size of the G = 4 tile loop, so it stays in the
    instruction cache); a tile is two passes. The per-tile work (folding X_g to the tile words, zeroing X_g) sits
    in a block entered only after even half-tile counts. X_0..X_3 are folded in the odd quarter's bubbles, X_4..X_7
    in that block. Partial row pointers are recomputed every slot from eight bases and two row offsets, so eight
    partials fit the integer register file. The peeled copy of the first even quarter leaves out the store of the
    (nonexistent) previous quarter's accumulator.
    """
    L = LayoutG8()
    r, X = L.reg, L.X
    name = "tx_strips_g8" if tile_rows == 8 else f"tx_strips_g8_r{tile_rows}"
    saved_int = sorted({x for x in L.reg.values() if x in CALLEE_SAVED_INT} | {RECORDS, "i30"},
                       key=lambda x: int(x[1:]))
    frame = 16 * len(X) + 8 * len(saved_int)
    frame = (frame + 15) // 16 * 16
    head = [[f"IAU.SUB i19 i19 {frame}"]]
    for k, v in enumerate(X):
        head.append([f"LSU0.STO.64 {v}.0 i19 {16 * k}", f"LSU1.STO.64 {v}.1 i19 {16 * k + 8}"])
    for k, reg in enumerate(saved_int):
        head.append([f"LSU0.STO.32 {reg} i19 {16 * len(X) + 4 * k}"])
    head.append([f"CMU.CP.32 {RECORDS} i18"])
    strip = []
    args = [r[n] for n in L.args]
    for k in range(0, len(args), 2):
        strip.append([f"LSU{m}.LDO.32 {args[k + m]} {RECORDS} {4 * (k + m)}" for m in range(2) if k + m < len(args)])
    strip.append(["NOP 6"])
    # Row pointers of partial 0, quarter 0; X_g = 0; then the accumulator of quarter 0 into A, partial 0 into T.
    strip += [[f"IAU.ADD {r['w0']} {r['b0']} {r['off0']}", f"VAU.XOR {X[0]} {X[0]} {X[0]}"],
              [f"IAU.ADD {r['w1']} {r['b0']} {r['off1']}", f"VAU.XOR {X[1]} {X[1]} {X[1]}"]]
    for k in range(4):
        strip.append([f"LSU0.LDI.128 {A[k]} {r['ai0']}", f"LSU1.LDI.128 {A[k + 4]} {r['ai1']}"])
        strip.append([f"VAU.XOR {X[2 + k]} {X[2 + k]} {X[2 + k]}"])
    for k in range(4):
        strip.append([f"LSU0.LDI.128 {T[k]} {r['w0']}", f"LSU1.LDI.128 {T[k + 4]} {r['w1']}"])
        strip.append([f"VAU.XOR {X[6 + k]} {X[6 + k]} {X[6 + k]}"] if k < 2 else ["NOP"])
    strip += [["NOP"], [f"IAU.ADD {r['ai0']} {r['ai0']} {r['air2']}"],
              [f"IAU.ADD {r['ai1']} {r['ai1']} {r['air2']}"], ["NOP 6"]]
    lines = [f".code      .text.{name}", f".L.text.{name}:", "    .align     16", f"\t.type\t{name},@function",
             f"    {name}:"]
    for b in head:
        lines += emit(b)
    lines.append(f".L{name}_strip:")
    for b in strip:
        lines += emit(b)
    peeled = quarter_g8(L, 0, first=True, tile_rows=tile_rows)
    for c, b in enumerate(peeled):
        if c == len(peeled) - 6:
            lines.append(f"        BRU.BRA .L{name}_entry")
        lines += emit(b)
    even, odd, end = quarter_g8(L, 0, tile_rows=tile_rows), quarter_g8(L, 1, tile_rows=tile_rows), tile_end_g8(L)
    check_post_increment_hazards([even, odd, end])
    check_wide_access_isolation([even, odd, end])
    check_post_increment_hazards([peeled, odd])
    lines.append(f".L{name}_loop:")
    for c, b in enumerate(even):
        lines += emit(b)
    lines.append(f".L{name}_entry:")
    for c, b in enumerate(odd):
        if c == L.quarter_len - 6:
            lines += ["        PEU.PC1C NEQ", f"            || BRU.BRA .L{name}_loop"]
        lines += emit(b)
    for c, b in enumerate(end):
        if c == G8_TILE_END - 6:
            lines += ["        PEU.PC1C NEQ", f"            || BRU.BRA .L{name}_loop"]
        lines += emit(b)
    # End of a strip: the last quarter's accumulator (set B); the tile words were stored by the tile-end block.
    tail = [["NOP 2"]]
    for k in range(4):
        tail.append([f"LSU0.STI.128 {B[k]} {r['an0']}", f"LSU1.STI.128 {B[k + 4]} {r['an1']}"])
        tail.append(["NOP"])
    more = L.R[0]
    tail += [[f"LSU0.LDO.32 {more} {RECORDS} {4 * (L.record_words - 1)}"], ["NOP 6"], [f"CMU.CMZ.i32 {more}"], ["NOP 2"],
             ["PEU.PC1C NEQ", f"BRU.BRA .L{name}_strip"],
             [f"IAU.ADD {RECORDS} {RECORDS} {4 * L.record_words}"], ["NOP 5"]]
    for k, v in enumerate(X):
        tail.append([f"LSU0.LDO.64 {v}.0 i19 {16 * k}", f"LSU1.LDO.64 {v}.1 i19 {16 * k + 8}"])
    for k, reg in enumerate(saved_int):
        tail.append([f"LSU0.LDO.32 {reg} i19 {16 * len(X) + 4 * k}"])
    tail.append(["NOP 6"])
    for b in tail:
        lines += emit(b)
    lines += ["        BRU.JMP i30", f"        IAU.ADD i19 i19 {frame}", "        NOP 5",
              f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}", ""]
    return "\n".join(lines) + "\n"


XOR_TILE = 35  # cycles per 8x16 tile in the XOR-only loop: 32 vector XORs + a 3-cycle hole for the word store


def generate_xor():
    """XOR-only strips (kernel acc_mode 3: inputs are cumulative sums already).

        void tx_xor_strips(const unsigned *records);
    Record (32-bit words): p0 p1 q0 q1 pr4 | xd xr xw | ntiles more
      p0/p1: rows 0/1 of the strip's first quarter, q0/q1: rows 2/3 (the pointer pairs alternate per quarter so
      that each can be advanced 3 cycles after its last post-increment); pr4 = 4 * row stride - 64
      xw: xor_out word of the first tile; xr: xor_out row stride; xd: a dummy word for the first (garbage) store

    Per tile (35 cycles): quarter q (cycles 8q..8q+7) XORs the 8 vectors of set q % 2 into X while loading the
    next quarter into set (q + 1) % 2 (pairs at 8q + 2j, idle neighbours); cycle 32 folds X with SAU.XORX into
    R0, cycle 33 stores the previous tile's word (copied to R1 at cycle 2) through xs. Three pointers rotate at
    the start of every tile: xs <- xp <- xn, xn += xr, starting from xd, xw and xw + xr, so the first store hits
    the dummy and the tail stores the last word through xp. No callee-saved registers are used.
    """
    name = "tx_xor_strips"
    S = [[f"v{k}" for k in range(8)], [f"v{8 + k}" for k in range(8)]]
    X = "v16"
    p0, p1, q0, q1, pr4, xp, xr, xn, cnt, more, xs, R0, R1, rec = [f"i{k}" for k in range(14)]
    P = [(p0, p1), (q0, q1)]
    ORDER8 = [0, 4, 1, 5, 2, 6, 3, 7]
    ops = []
    for q in range(4):
        cur, nxt = S[q % 2], S[(q + 1) % 2]
        for k, i in enumerate(ORDER8):
            c = 8 * q + k
            if q == 0 and k == 0:
                ops.append((c, f"VAU.XOR {X} {cur[0]} {cur[4]}"))
            elif q == 0 and k == 1:
                continue
            else:
                ops.append((c, f"VAU.XOR {X} {X} {cur[i]}"))
        pa, pb = P[(q + 1) % 2]   # the next quarter's rows
        for j in range(4):
            ops += [(8 * q + 2 * j, f"LSU0.LDI.128 {nxt[j]} {pa}"), (8 * q + 2 * j, f"LSU1.LDI.128 {nxt[4 + j]} {pb}")]
        ops += [(8 * q + 9, f"IAU.ADD {pa} {pa} {pr4}"), (8 * q + 10, f"IAU.ADD {pb} {pb} {pr4}")]
    ops += [(0, f"IAU.ADD {xs} {xp} 0"), (1, f"IAU.ADD {xp} {xn} 0"), (2, f"IAU.ADD {xn} {xn} {xr}"),
            (2, f"CMU.CP.32 {R1} {R0}"),
            (20, f"IAU.INCS {cnt} -1"), (22, f"CMU.CMZ.i32 {cnt}"),
            (32, f"SAU.XORX.x32 {R0} {X}"), (33, f"LSU0.STO.32 {R1} {xs} 0")]
    bundles = [[] for _ in range(XOR_TILE)]
    for c, op in ops:
        unit = op[:4] if op.startswith("LSU") else op.split(".")[0]
        assert all((o[:4] if o.startswith("LSU") else o.split(".")[0]) != unit for o in bundles[c]), (c, op)
        bundles[c].append(op)
    check_post_increment_hazards([bundles])
    check_wide_access_isolation([bundles])
    lines = [f".code      .text.{name}", f".L.text.{name}:", "    .align     16", f"	.type	{name},@function",
             f"    {name}:", f"        CMU.CP.32 {rec} i18"]
    lines.append(f".L{name}_strip:")
    args = [p0, p1, q0, q1, pr4, xp, xr, xn, cnt, more]   # xp is loaded with xd, xn with xw
    strip = []
    for k in range(0, len(args), 2):
        strip.append([f"LSU{m}.LDO.32 {args[k + m]} {rec} {4 * (k + m)}" for m in range(2)])
    strip.append(["NOP 6"])
    for j in range(4):   # quarter 0 of the first tile into set 0
        strip.append([f"LSU0.LDI.128 {S[0][j]} {p0}", f"LSU1.LDI.128 {S[0][4 + j]} {p1}"])
        strip.append(["NOP"])
    strip += [["NOP"], [f"IAU.ADD {p0} {p0} {pr4}"], [f"IAU.ADD {p1} {p1} {pr4}"], ["NOP 6"]]
    for b in strip:
        lines += emit(b)
    lines.append(f".L{name}_loop:")
    for c, b in enumerate(bundles):
        if c == XOR_TILE - 6:   # the branch bundle plus 6 delay-slot bundles end the tile
            lines += ["        PEU.PC1C NEQ", f"            || BRU.BRA .L{name}_loop"]
        lines += emit(b)
    tail = [["NOP"], [f"IAU.ADD {xs} {xp} 0"], ["NOP 2"], [f"LSU0.STO.32 {R0} {xs} 0"],
            [f"CMU.CMZ.i32 {more}"], ["NOP 2"], ["PEU.PC1C NEQ", f"BRU.BRA .L{name}_strip"],
            [f"IAU.ADD {rec} {rec} {4 * len(args)}"], ["NOP 5"]]
    for b in tail:
        lines += emit(b)
    lines += ["        BRU.JMP i30", "        NOP 6", f".Lfunc_end_{name}:", f"	.size	{name}, .Lfunc_end_{name}-{name}", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    groups = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4]
    text = "".join(generate(g) for g in groups) + generate_g8() + generate_g8(16) + generate_xor()
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(text)
    else:
        sys.stdout.write(text)
