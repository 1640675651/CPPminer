#!/usr/bin/env python3
"""Summarise `-Xptxas -v` output of the fused CUTLASS kernels.

Build with ptxas verbose output captured, e.g.
  CUDAFLAGS="-Xptxas -v" ./build.sh --backend cpu,cuda --cuda-arch '75;86;89' > build.log 2>&1
then
  scripts/ptxas_table.py build.log

Prints one row per (kernel variant, arch): registers, stack frame, spill
stores/loads. Shared memory of these kernels is dynamic (sizeof SharedStorage,
printed by the miner in the kernel name line), so ptxas reports 0 smem.
"""
import re
import sys

ENTRY = re.compile(r"Compiling entry function '(_Z\S*FusedKernelEntry\S*)' for '(sm_\d+)'")
FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REGS = re.compile(r"Used (\d+) registers")


def short(name):
    arch = re.search(r"arch4(Sm\d+)", name)
    # TB, warp and instruction shapes are the first three <int,int,int>
    # template argument packs (later ones are back-references).
    shapes = re.findall(r"ILi(\d+)ELi(\d+)ELi(\d+)E", name)
    stages = re.search(r"ILi\d+ELi\d+ELi\d+EEELi(\d+)ELi(\d+)E", name)
    kind = "Case9" if "GemmTypesCase9" in name else "Case10"
    s = [("x".join(t)) for t in shapes[:3]]
    tag = f"{kind} {arch.group(1) if arch else '?'} TB {s[0] if s else '?'}"
    if len(s) > 2:
        tag += f" warp {s[1]} instr {s[2]}"
    if stages:
        tag += f" stages {stages.group(1)}"
    return tag


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = {}
    cur = None
    frame = None
    with open(sys.argv[1], errors="replace") as f:
        for line in f:
            m = ENTRY.search(line)
            if m:
                cur = (short(m.group(1)), m.group(2))
                frame = None
                continue
            if cur is None:
                continue
            m = FRAME.search(line)
            if m:
                frame = m.groups()
                continue
            m = REGS.search(line)
            if m and frame is not None:
                rows[cur] = (int(m.group(1)),) + tuple(int(x) for x in frame)
                cur = None
    print(f"{'kernel':<62} {'arch':<6} {'regs':>4} {'stack':>5} {'spill st/ld':>11}")
    for (k, arch), (regs, stack, st, ld) in sorted(rows.items()):
        print(f"{k:<62} {arch:<6} {regs:>4} {stack:>5} {st:>5}/{ld:<5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
