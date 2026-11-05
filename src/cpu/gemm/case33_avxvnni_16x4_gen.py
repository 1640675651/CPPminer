#!/usr/bin/env python3
"""Generate the AVX-VNNI 16x4 u8s8 GEMM + milestone-XOR kernel for MASM and GAS.

Port of gemm_xor_benchmark Case 3.5 (cpu/case35_avx_vnni_masm.asm). Both outputs
carry the same instruction stream; only the dialect and the calling convention
differ, so MSVC and GCC/Clang builds run identical code:

    case33_avxvnni_16x4_masm.asm   Windows x64, ml64
    case33_avxvnni_16x4_gas.S      System V x86-64 (ELF), GNU as / LLVM

Regenerate both after editing this script:

    python src/cpu/gemm/case33_avxvnni_16x4_gen.py

C signature:

    void case33_avxvnni_16x4_xor(const int8_t *a, const int8_t *b, int blocks,
                                 uint32_t *out, const int32_t *correction);

a           Rows 0..7 of the 16-row tile: one packed 8-row A tile, A+128 as
            unsigned bytes, [K/128 panel][K/4][8 rows][4 K bytes], 1024 bytes per
            panel. Rows 8..15 are the next 8-row tile, blocks * 1024 bytes on.
b           One 4-column B quarter, signed, [K/128 panel][K/4][4 cols][4 K bytes],
            512 contiguous bytes per panel.
blocks      K / 128 panels (milestones), >= 1.
out         Per milestone: out[2*ms] = XOR of the 8x4 C block of rows 0..7,
            out[2*ms+1] = XOR of rows 8..15. C is cumulative over K and wraps
            modulo 2^32.
correction  Per panel and column, 128 * sum(B) over that panel: [K/128][4].

Schedule, as in Case 3.5:
  * C lives in ymm0..7 (column c: rows 0..7 in ymm<c>, rows 8..15 in ymm<c+4>),
    A rows 0..7 / 8..15 in ymm8 / ymm9. Each B broadcast feeds two dot products,
    and is refilled with the broadcast six columns ahead (six-register ring
    ymm10..15), also across K=4 steps.
  * Before each panel the correction is subtracted from C; the unsigned*signed
    dots then restore the exact signed cumulative C.
  * Each panel is fully unrolled. The final column updates fold C into the
    milestone's vector XOR (ymm12 = [rows 0..7 | rows 8..15], 4 lanes each).
    Except after the last panel, its horizontal reduction and stores are spread
    over the next panel's first eight column updates, using ymm13; those use a
    four-register ring (ymm14, 15, 10, 11) before the six-register ring resumes.
  * No stack spills. No load reaches past the last panel of A or B.
"""

from pathlib import Path

PROC = "case33_avxvnni_16x4_xor"
K_STEPS = 32            # K/4 steps per 128-deep panel
COLS = 4
COLUMNS = K_STEPS * COLS  # broadcast columns per panel
A_PANEL = 1024
B_PANEL = 512
RING = 6

# Neutral instruction text: Intel operand order, decimal immediates. "VNNI"
# becomes the explicitly VEX-encoded VPDPBUSD of each dialect (otherwise the
# assemblers may pick the AVX-512 EVEX form). Labels are "@name".


def vnni(acc, a, b):
    return f"VNNI ymm{acc}, ymm{a}, ymm{b}"


def bcast(reg, col):
    return f"vpbroadcastd ymm{reg}, dword ptr [r11+{col * 4}]"


def load_a(step):
    return [f"vmovdqu ymm8, ymmword ptr [r10+{step * 32}]",
            f"vmovdqu ymm9, ymmword ptr [rdx+{step * 32}]"]


def correct_panel():
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


def body():
    out = []
    out += ["vpxor ymm%d, ymm%d, ymm%d" % (i, i, i) for i in range(8)]

    # First panel: no previous milestone to finish.
    out += correct_panel()
    out += [bcast(ring_reg(g), g) for g in range(RING)]
    for g in range(2 * COLS):
        out += ring_column(g)
    out.append("jmp @body")

    # Later panels: finish the previous milestone (pending in ymm12) between the
    # first eight column updates, with a four-register ring 14, 15, 10, 11.
    out += ["ALIGN64", "@next:"]
    out += correct_panel()
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


HEADER = ("AVX-VNNI 16x4 u8s8 GEMM + milestone XOR (port of gemm_xor_benchmark Case 3.5).\n"
          "Generated by case33_avxvnni_16x4_gen.py, together with its MASM/GAS twin; do not edit.")


def masm():
    sub = 168  # xmm6..15 (Win64 callee-saved) + alignment
    lines = [f"; {line}" for line in HEADER.splitlines()] + [
             "; Windows x64: RCX=a, RDX=b, R8D=blocks, R9=out, [rsp+40]=correction.",
             "; Kernel registers: R10=a rows 0..7, RDX=a rows 8..15, R11=b, RCX=correction.",
             "",
             "CASE33_16X4_TEXT SEGMENT ALIGN(64) 'CODE'",
             "",
             f"PUBLIC {PROC}",
             f"{PROC} PROC FRAME",
             f"    sub rsp, {sub}",
             f"    .allocstack {sub}"]
    for i in range(6, 16):
        off = (i - 6) * 16
        lines += [f"    vmovdqu xmmword ptr [rsp+{off}], xmm{i}",
                  f"    .savexmm128 xmm{i}, {off}"]
    lines += ["    .endprolog",
              "    mov r10, rcx",
              "    mov r11, rdx",
              f"    mov rcx, qword ptr [rsp+{sub + 8 + 32}]",
              "    mov edx, r8d",
              "    shl rdx, 10",
              "    add rdx, r10"]
    for ins in body():
        if ins == "ALIGN64":
            lines.append("    ALIGN 64")
        elif ins.startswith("@") and ins.endswith(":"):
            lines.append(f"{PROC}_{ins[1:-1]}:")
        else:
            ins = ins.replace("VNNI ", "vex vpdpbusd ")
            ins = ins.replace("@", f"{PROC}_")
            lines.append(f"    {ins}")
    for i in range(6, 16):
        lines.append(f"    vmovdqu xmm{i}, xmmword ptr [rsp+{(i - 6) * 16}]")
    lines += ["    vzeroupper",
              f"    add rsp, {sub}",
              "    ret",
              f"{PROC} ENDP",
              "",
              "CASE33_16X4_TEXT ENDS",
              "END",
              ""]
    return "\n".join(lines)


def gas():
    lines = ["/* " + "\n * ".join(HEADER.splitlines())] + [
             " * System V x86-64: RDI=a, RSI=b, EDX=blocks, RCX=out, R8=correction.",
             " * Kernel registers: R10=a rows 0..7, RDX=a rows 8..15, R11=b, RCX=correction. */",
             "",
             "    .intel_syntax noprefix",
             "    .text",
             "    .p2align 6",
             f"    .globl {PROC}",
             f"    .type {PROC}, @function",
             f"{PROC}:",
             "    mov r10, rdi",
             "    mov r11, rsi",
             "    mov r9, rcx",
             "    mov rcx, r8",
             "    mov r8d, edx",
             "    mov edx, edx",
             "    shl rdx, 10",
             "    add rdx, r10"]
    for ins in body():
        if ins == "ALIGN64":
            lines.append("    .p2align 6")
        elif ins.startswith("@") and ins.endswith(":"):
            lines.append(f".L{PROC}_{ins[1:-1]}:")
        else:
            ins = ins.replace("VNNI ", "{vex} vpdpbusd ")
            ins = ins.replace("@", f".L{PROC}_")
            lines.append(f"    {ins}")
    lines += ["    vzeroupper",
              "    ret",
              f"    .size {PROC}, .-{PROC}",
              "",
              '    .section .note.GNU-stack,"",@progbits',
              ""]
    return "\n".join(lines)


def main():
    here = Path(__file__).resolve().parent
    (here / "case33_avxvnni_16x4_masm.asm").write_text(masm(), newline="\n")
    (here / "case33_avxvnni_16x4_gas.S").write_text(gas(), newline="\n")


if __name__ == "__main__":
    main()
