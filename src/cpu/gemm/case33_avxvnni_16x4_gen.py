#!/usr/bin/env python3
"""Generate the 16x4 GEMM + milestone-XOR kernels for MASM and GAS.

Port of gemm_xor_benchmark Case 3.5 (cpu/case35_avx_vnni_masm.asm,
cpu/case35_vnni_masm.asm, cpu/case35_avx2.cpp and cpu/case35_ssse3.cpp). Both outputs
carry the same instruction streams; only the dialect and the calling convention
differ, so MSVC and GCC/Clang builds run identical code:

    case33_avxvnni_16x4_masm.asm   Windows x64, ml64
    case33_avxvnni_16x4_gas.S      System V x86-64 (ELF), GNU as / LLVM

Regenerate both after editing this script:

    python src/cpu/gemm/case33_avxvnni_16x4_gen.py

Four kernels with the same arguments (correction only for the u8s8 ones):

    /* AVX-VNNI: VPDPBUSD, unsigned A+128 times signed B, corrected per panel. */
    void case33_avxvnni_16x4_xor(const int8_t *a, const int8_t *b, int blocks,
                                 uint32_t *out, const int32_t *correction);
    /* AVX-VNNI-INT8: VPDPBSSD, signed A times signed B; no offset, no correction. */
    void case33_avxvnni_int8_16x4_xor(const int8_t *a, const int8_t *b, int blocks,
                                      uint32_t *out);
    /* AVX2 / SSSE3 (legacy SSE encoding): (V)PMADDUBSW on A+128 times B, then
     * (V)PMADDWD by ones, corrected per panel. PMADDUBSW saturates its int16 pair
     * sums, so these are exact only while |B| <= 64 (Pearl's zero-B noise is). */
    void case33_avx2_16x4_xor(...);    /* as case33_avxvnni_16x4_xor */
    void case33_ssse3_16x4_xor(...);   /* as case33_avxvnni_16x4_xor */

a           Rows 0..7 of the 16-row tile: one packed 8-row A tile (A+128 as
            unsigned bytes for the u8s8 kernels, plain signed A for AVX-VNNI-INT8),
            [K/128 panel][K/4][8 rows][4 K bytes], 1024 bytes per panel. Rows 8..15
            are the next 8-row tile, blocks * 1024 bytes on.
b           One 4-column B quarter, signed, [K/128 panel][K/4][4 cols][4 K bytes],
            512 contiguous bytes per panel.
blocks      K / 128 panels (milestones), >= 1.
out         Per milestone: out[2*ms] = XOR of the 8x4 C block of rows 0..7,
            out[2*ms+1] = XOR of rows 8..15. C is cumulative over K and wraps
            modulo 2^32.
correction  u8s8 kernels only. Per panel and column, 128 * sum(B) over that panel:
            [K/128][4].

AVX2: C in ymm0..7 as below, A in ymm8/9, a K loop of four K/4 steps per iteration,
reduction after each panel (no interleaving). SSSE3: as Case 3.5, each panel runs as
two 8-row halves, one half's eight accumulators in xmm0..7 and the other parked on the
stack. Neither spills inside the K loop, unlike the MSVC builds of the intrinsics
versions (11% and 4.5% faster sustained on a Core Ultra 9 275HX).

AVX-VNNI schedule, as in Case 3.5:
  * C lives in ymm0..7 (column c: rows 0..7 in ymm<c>, rows 8..15 in ymm<c+4>),
    A rows 0..7 / 8..15 in ymm8 / ymm9. Each B broadcast feeds two dot products,
    and is refilled with the broadcast six columns ahead (six-register ring
    ymm10..15), also across K=4 steps.
  * AVX-VNNI: before each panel the correction is subtracted from C; the
    unsigned*signed dots then restore the exact signed cumulative C.
  * Each panel is fully unrolled. The final column updates fold C into the
    milestone's vector XOR (ymm12 = [rows 0..7 | rows 8..15], 4 lanes each).
    Except after the last panel, its horizontal reduction and stores are spread
    over the next panel's first eight column updates, using ymm13; those use a
    four-register ring (ymm14, 15, 10, 11) before the six-register ring resumes.
  * No stack spills. No load reaches past the last panel of A or B.
"""

from pathlib import Path

K_STEPS = 32            # K/4 steps per 128-deep panel
COLS = 4
COLUMNS = K_STEPS * COLS  # broadcast columns per panel
A_PANEL = 1024
B_PANEL = 512
RING = 6


class Kernel:
    """signed: no correction argument. vex: AVX encoding (VEX register saves, vzeroupper);
    the SSSE3 kernel must stay legacy SSE for CPUs without AVX. park: stack bytes for
    parked accumulators (the body addresses them as [rsp+PARK+...])."""

    def __init__(self, name, signed, body, vex=True, park=0):
        self.name = name
        self.signed = signed
        self.body = body
        self.vex = vex
        self.park = park

# Neutral instruction text: Intel operand order, decimal immediates. "VNNI"
# becomes the dot product of the kernel and dialect (VPDPBUSD explicitly VEX
# encoded, otherwise the assemblers may pick the AVX-512 EVEX form; VPDPBSSD
# only exists as VEX). Labels are "@name".


def vnni(acc, a, b):
    return f"VNNI ymm{acc}, ymm{a}, ymm{b}"


def bcast(reg, col):
    return f"vpbroadcastd ymm{reg}, dword ptr [r11+{col * 4}]"


def load_a(step):
    return [f"vmovdqu ymm8, ymmword ptr [r10+{step * 32}]",
            f"vmovdqu ymm9, ymmword ptr [rdx+{step * 32}]"]


def correct_panel(kernel):
    if kernel.signed:
        return []
    out = []
    for c in range(COLS):
        out += [f"vpbroadcastd ymm14, dword ptr [rcx+{c * 4}]",
                f"vpsubd ymm{c}, ymm{c}, ymm14",
                f"vpsubd ymm{c + 4}, ymm{c + 4}, ymm14"]
    out.append("add rcx, 16")
    return out


def ring_reg(g):
    return 10 + g % RING


def ring_column(g):
    """Column g with the six-register ring: two dots, then refill six ahead."""
    out = []
    if g % COLS == 0:
        out += load_a(g // COLS)
    c, r = g % COLS, ring_reg(g)
    out += [vnni(c, 8, r), vnni(c + 4, 9, r)]
    if g + RING < COLUMNS:
        out.append(bcast(r, g + RING))
    return out


def body(kernel):
    out = []
    out += ["vpxor ymm%d, ymm%d, ymm%d" % (i, i, i) for i in range(8)]

    # First panel: no previous milestone to finish.
    out += correct_panel(kernel)
    out += [bcast(ring_reg(g), g) for g in range(RING)]
    for g in range(2 * COLS):
        out += ring_column(g)
    out.append("jmp @body")

    # Later panels: finish the previous milestone (pending in ymm12) between the
    # first eight column updates, with a four-register ring 14, 15, 10, 11.
    out += ["ALIGN64", "@next:"]
    out += correct_panel(kernel)
    four = [14, 15, 10, 11]
    out += [bcast(four[c], c) for c in range(COLS)]
    finish = [
        ["vpshufd ymm13, ymm12, 78"],          # swap 64-bit halves per lane
        ["vpxor ymm12, ymm12, ymm13"],
        ["vpshufd ymm13, ymm12, 177"],         # swap neighbouring dwords
        ["vpxor ymm12, ymm12, ymm13"],         # every dword = XOR of its 128-bit half
        ["vmovd dword ptr [r9], xmm12"],       # rows 0..7
        ["vextracti128 xmm13, ymm12, 1"],
        ["vmovd dword ptr [r9+4], xmm13"],     # rows 8..15
        ["add r9, 8"],
    ]
    for g in range(2 * COLS):
        if g % COLS == 0:
            out += load_a(g // COLS)
        c, r = g % COLS, four[g % COLS]
        out += [vnni(c, 8, r), vnni(c + 4, 9, r)]
        out += finish[g]
        if g < COLS:
            out.append(bcast(r, g + COLS))
    out += [bcast(ring_reg(g), g) for g in range(2 * COLS, 2 * COLS + RING)]

    out.append("@body:")
    last = COLUMNS - COLS
    for g in range(2 * COLS, last):
        out += ring_column(g)

    # Last K step: fold each column into the milestone XOR as soon as it is final.
    # ymm12/13 are free (their ring slots had nothing left to load); ymm14/15
    # are free once columns 0 and 1 are done.
    g = last
    r = [ring_reg(g + c) for c in range(COLS)]
    out += load_a(g // COLS)
    out += [vnni(0, 8, r[0]), vnni(4, 9, r[0]),
            vnni(1, 8, r[1]), vnni(5, 9, r[1]),
            "vpxor ymm12, ymm0, ymm1",
            "vpxor ymm13, ymm4, ymm5",
            vnni(2, 8, r[2]),
            "vpxor ymm12, ymm12, ymm2",
            vnni(6, 9, r[2]),
            "vpxor ymm13, ymm13, ymm6",
            vnni(3, 8, r[3]),
            "vpxor ymm12, ymm12, ymm3",
            vnni(7, 9, r[3]),
            "vpxor ymm13, ymm13, ymm7",
            # [rows 0..7 | rows 8..15], each 128-bit half reduced to four lanes
            "vperm2i128 ymm14, ymm12, ymm13, 32",
            "vperm2i128 ymm15, ymm12, ymm13, 49",
            "vpxor ymm12, ymm14, ymm15"]
    out += [f"add r10, {A_PANEL}", f"add rdx, {A_PANEL}", f"add r11, {B_PANEL}",
            "dec r8d", "jnz @next"]

    # The last milestone has no next panel to hide its reduction in.
    out += [line for step in finish[:-1] for line in step]
    return out


# AVX2 / SSSE3 u8s8 (gemm_xor_benchmark Case 3.5 case35_avx2.cpp / case35_ssse3.cpp):
# PMADDUBSW (A+128 unsigned times B signed, pairs saturate to int16: exact while |B| <= 64),
# then PMADDWD by ones into int32. A K loop of UNROLL K/4 steps per iteration; the
# out-of-order core overlaps the independent column chains, so the temporaries just rotate.
UNROLL = 4
STEP_ITERS = K_STEPS // UNROLL


def correct_panel_avx2():
    out = []
    for c in range(COLS):
        out += [f"vpbroadcastd ymm14, dword ptr [rcx+{c * 4}]",
                f"vpsubd ymm{c}, ymm{c}, ymm14",
                f"vpsubd ymm{c + 4}, ymm{c + 4}, ymm14"]
    out.append("add rcx, 16")
    return out


def avx2_body(_kernel):
    """C in ymm0..7 as for AVX-VNNI, A in ymm8/9, ones (int16) in ymm15; B broadcasts
    ymm10/13 and products ymm11/14 and ymm12 alternate between columns."""
    out = ["vpxor ymm%d, ymm%d, ymm%d" % (i, i, i) for i in range(8)]
    out += ["vpcmpeqw ymm15, ymm15, ymm15", "vpsrlw ymm15, ymm15, 15", "ALIGN64", "@panel:"]
    out += correct_panel_avx2()
    out += [f"mov eax, {STEP_ITERS}", "ALIGN64", "@step:"]
    for s in range(UNROLL):
        out += [f"vmovdqu ymm8, ymmword ptr [r10+{s * 32}]",
                f"vmovdqu ymm9, ymmword ptr [rdx+{s * 32}]"]
        for c in range(COLS):
            b, t0, t1 = (10, 11, 12) if c % 2 == 0 else (13, 14, 12)
            out += [f"vpbroadcastd ymm{b}, dword ptr [r11+{(s * COLS + c) * 4}]",
                    f"vpmaddubsw ymm{t0}, ymm8, ymm{b}",
                    f"vpmaddubsw ymm{t1}, ymm9, ymm{b}",
                    f"vpmaddwd ymm{t0}, ymm{t0}, ymm15",
                    f"vpmaddwd ymm{t1}, ymm{t1}, ymm15",
                    f"vpaddd ymm{c}, ymm{c}, ymm{t0}",
                    f"vpaddd ymm{c + 4}, ymm{c + 4}, ymm{t1}"]
    out += [f"add r10, {UNROLL * 32}", f"add rdx, {UNROLL * 32}", f"add r11, {UNROLL * 16}",
            "dec eax", "jnz @step"]
    out += ["vpxor ymm10, ymm0, ymm1", "vpxor ymm11, ymm2, ymm3", "vpxor ymm10, ymm10, ymm11",
            "vpxor ymm12, ymm4, ymm5", "vpxor ymm13, ymm6, ymm7", "vpxor ymm12, ymm12, ymm13",
            # [rows 0..7 | rows 8..15], each 128-bit half reduced to one dword
            "vperm2i128 ymm13, ymm10, ymm12, 32",
            "vperm2i128 ymm14, ymm10, ymm12, 49",
            "vpxor ymm10, ymm13, ymm14",
            "vpshufd ymm11, ymm10, 78", "vpxor ymm10, ymm10, ymm11",
            "vpshufd ymm11, ymm10, 177", "vpxor ymm10, ymm10, ymm11",
            "vmovd dword ptr [r9], xmm10",
            "vextracti128 xmm11, ymm10, 1",
            "vmovd dword ptr [r9+4], xmm11",
            "add r9, 8", "dec r8d", "jnz @panel"]
    return out


PARK = "PARK"   # stack offset of the parked accumulators, filled in per dialect


def ssse3_steps(a_reg):
    """One 8-row half over a panel: column c in xmm(2c) rows 0..3 and xmm(2c+1) rows 4..7, A
    in xmm8/9, the K group's 16 B bytes in xmm10 (one load; pshufd broadcasts each column into
    xmm11, rather than Case 3.5's B stored twice for movddup, which would double the job's
    packed B), products xmm12/13, ones (int16) in xmm15."""
    out = [f"mov eax, {STEP_ITERS}", "ALIGN64", f"@step_{a_reg}:"]
    for s in range(UNROLL):
        out += [f"movdqu xmm8, xmmword ptr [{a_reg}+{s * 32}]",
                f"movdqu xmm9, xmmword ptr [{a_reg}+{s * 32 + 16}]",
                f"movdqu xmm10, xmmword ptr [r11+{s * 16}]"]
        for c in range(COLS):
            out += [f"pshufd xmm11, xmm10, {c * 0x55}",
                    "movdqa xmm12, xmm8",
                    "pmaddubsw xmm12, xmm11",
                    "movdqa xmm13, xmm9",
                    "pmaddubsw xmm13, xmm11",
                    "pmaddwd xmm12, xmm15",
                    "pmaddwd xmm13, xmm15",
                    f"paddd xmm{2 * c}, xmm12",
                    f"paddd xmm{2 * c + 1}, xmm13"]
    out += [f"add {a_reg}, {UNROLL * 32}", f"add r11, {UNROLL * 16}", "dec eax",
            f"jnz @step_{a_reg}"]
    return out


def ssse3_parked_body(_kernel):
    """As case35_ssse3.cpp: per panel, rows 0..7 then rows 8..15, each half's eight
    accumulators in xmm0..7 and the other half's parked on the stack (256 bytes at PARK)."""
    out = ["pxor xmm0, xmm0"] + [f"movdqu xmmword ptr [rsp+{PARK}+{i * 16}], xmm0"
                                 for i in range(16)]
    out += ["pcmpeqw xmm15, xmm15", "psrlw xmm15, 15", "ALIGN64", "@panel:"]
    for half, a_reg in ((0, "r10"), (1, "rdx")):
        out += [f"movdqu xmm{i}, xmmword ptr [rsp+{PARK}+{(half * 8 + i) * 16}]"
                for i in range(8)]
        for c in range(COLS):
            out += [f"movd xmm14, dword ptr [rcx+{c * 4}]",
                    "pshufd xmm14, xmm14, 0",
                    f"psubd xmm{2 * c}, xmm14",
                    f"psubd xmm{2 * c + 1}, xmm14"]
        out += ssse3_steps(a_reg)
        out += [f"movdqu xmmword ptr [rsp+{PARK}+{(half * 8 + i) * 16}], xmm{i}"
                for i in range(8)]
        out += ["movdqa xmm12, xmm0"] + [f"pxor xmm12, xmm{i}" for i in range(1, 8)]
        out += ["pshufd xmm13, xmm12, 78", "pxor xmm12, xmm13",
                "pshufd xmm13, xmm12, 177", "pxor xmm12, xmm13",
                f"movd dword ptr [r9+{half * 4}], xmm12"]
        if half == 0:
            out.append(f"sub r11, {B_PANEL}")   # rows 8..15 read the same B panel
    out += ["add rcx, 16", "add r9, 8", "dec r8d", "jnz @panel"]
    return out


KERNELS = [Kernel("case33_avxvnni_16x4_xor", signed=False, body=body),
           Kernel("case33_avxvnni_int8_16x4_xor", signed=True, body=body),
           Kernel("case33_avx2_16x4_xor", signed=False, body=avx2_body),
           Kernel("case33_ssse3_16x4_xor", signed=False, body=ssse3_parked_body, vex=False,
                  park=256)]


HEADER = ("AVX-VNNI / AVX-VNNI-INT8 / AVX2 / SSSE3 16x4 GEMM + milestone XOR (port of\n"
          "gemm_xor_benchmark Case 3.5). Generated by case33_avxvnni_16x4_gen.py, together\n"
          "with its MASM/GAS twin; do not edit.")


def masm_proc(kernel):
    proc = kernel.name
    dot = "vpdpbssd " if kernel.signed else "vex vpdpbusd "
    sub = 168 + kernel.park  # xmm6..15 (Win64 callee-saved) + alignment, then parked C
    if kernel.signed:
        args = "; Windows x64: RCX=a, RDX=b, R8D=blocks, R9=out."
    else:
        args = "; Windows x64: RCX=a, RDX=b, R8D=blocks, R9=out, [rsp+40]=correction."
    mov = "vmovdqu" if kernel.vex else "movdqu"
    lines = [args,
             f"PUBLIC {proc}",
             f"{proc} PROC FRAME",
             f"    sub rsp, {sub}",
             f"    .allocstack {sub}"]
    for i in range(6, 16):
        off = (i - 6) * 16
        lines += [f"    {mov} xmmword ptr [rsp+{off}], xmm{i}",
                  f"    .savexmm128 xmm{i}, {off}"]
    lines += ["    .endprolog",
              "    mov r10, rcx",
              "    mov r11, rdx"]
    if not kernel.signed:
        lines.append(f"    mov rcx, qword ptr [rsp+{sub + 8 + 32}]")
    lines += ["    mov edx, r8d",
              "    shl rdx, 10",
              "    add rdx, r10"]
    for ins in kernel.body(kernel):
        if ins == "ALIGN64":
            lines.append("    ALIGN 64")
        elif ins.startswith("@") and ins.endswith(":"):
            lines.append(f"{proc}_{ins[1:-1]}:")
        else:
            ins = ins.replace("VNNI ", dot).replace(PARK, "160")
            ins = ins.replace("@", f"{proc}_")
            lines.append(f"    {ins}")
    for i in range(6, 16):
        lines.append(f"    {mov} xmm{i}, xmmword ptr [rsp+{(i - 6) * 16}]")
    if kernel.vex:
        lines.append("    vzeroupper")
    lines += [f"    add rsp, {sub}",
              "    ret",
              f"{proc} ENDP",
              ""]
    return lines


def masm():
    lines = [f"; {line}" for line in HEADER.splitlines()] + [
             "; Kernel registers: R10=a rows 0..7, RDX=a rows 8..15, R11=b, RCX=correction.",
             "",
             "CASE33_16X4_TEXT SEGMENT ALIGN(64) 'CODE'",
             ""]
    for kernel in KERNELS:
        lines += masm_proc(kernel)
    lines += ["CASE33_16X4_TEXT ENDS",
              "END",
              ""]
    return "\n".join(lines)


def gas_proc(kernel):
    proc = kernel.name
    dot = "vpdpbssd " if kernel.signed else "{vex} vpdpbusd "
    if kernel.signed:
        args = "/* System V x86-64: RDI=a, RSI=b, EDX=blocks, RCX=out. */"
    else:
        args = "/* System V x86-64: RDI=a, RSI=b, EDX=blocks, RCX=out, R8=correction. */"
    lines = [args,
             "    .p2align 6",
             f"    .globl {proc}",
             f"    .type {proc}, @function",
             f"{proc}:",
             "    mov r10, rdi",
             "    mov r11, rsi",
             "    mov r9, rcx"]
    if not kernel.signed:
        lines.append("    mov rcx, r8")
    lines.append("    mov r8d, edx")
    if kernel.park:
        lines.append(f"    sub rsp, {kernel.park}")
    lines += ["    mov edx, edx",
              "    shl rdx, 10",
              "    add rdx, r10"]
    for ins in kernel.body(kernel):
        if ins == "ALIGN64":
            lines.append("    .p2align 6")
        elif ins.startswith("@") and ins.endswith(":"):
            lines.append(f".L{proc}_{ins[1:-1]}:")
        else:
            ins = ins.replace("VNNI ", dot).replace(PARK, "0")
            ins = ins.replace("@", f".L{proc}_")
            lines.append(f"    {ins}")
    if kernel.vex:
        lines.append("    vzeroupper")
    if kernel.park:
        lines.append(f"    add rsp, {kernel.park}")
    lines += ["    ret",
              f"    .size {proc}, .-{proc}",
              ""]
    return lines


def gas():
    lines = ["/* " + "\n * ".join(HEADER.splitlines()),
             " * Kernel registers: R10=a rows 0..7, RDX=a rows 8..15, R11=b, RCX=correction. */",
             "",
             "    .intel_syntax noprefix",
             "    .text",
             ""]
    for kernel in KERNELS:
        lines += gas_proc(kernel)
    lines += ['    .section .note.GNU-stack,"",@progbits',
              ""]
    return "\n".join(lines)


def main():
    here = Path(__file__).resolve().parent
    (here / "case33_avxvnni_16x4_masm.asm").write_text(masm(), newline="\n")
    (here / "case33_avxvnni_16x4_gas.S").write_text(gas(), newline="\n")


if __name__ == "__main__":
    main()
