#!/usr/bin/env python3
"""Per-milestone instruction mix of the fused CUTLASS kernels (SASS).

Usage:
  cuobjdump -sass -arch sm_86 build/cmake/cppminer > k86.sass
  scripts/sass_milestone_stats.py k86.sass [name-filter ...]

For every FusedKernelEntry function in the dump (optionally only those whose
mangled name contains all filters) it finds the main loop -- the backward
branch that encloses the tensor-core instructions (IMMA/HMMA) -- and prints
the opcode histogram of one loop iteration. One iteration of the Case-10
multistage loop is exactly one milestone (kMilestoneIters K-tiles + the XOR /
reduce-scatter / jackpot-fold callback). For the Sm75 2-stage loop
(MmaMilestone) one iteration is one K-tile and the callback sits behind a
branch; the script then reports the loop and the callback block separately
(the code between the loop's last IMMA and the backward branch).

Columns: total instructions, IMMA, and the opcodes the milestone code uses
(LOP3 = 3-input logic, SHFL, SEL, SHF = funnel shift/rotate, LDL/STL = local
memory, i.e. dynamically indexed register arrays).
"""
import re
import sys
from collections import Counter

FUNC_RE = re.compile(r"Function : (\S+)")
INSN_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
KEY = ["IMMA", "HMMA", "LDSM", "LDGSTS", "LOP3", "xor3", "xor2", "SHFL", "SEL", "SHF", "LDL", "STL",
       "IADD3", "IMAD", "ISETP", "MOV", "BAR", "BRA", "BRX"]


def parse(path):
    funcs = {}
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            m = FUNC_RE.search(line)
            if m:
                cur = m.group(1)
                funcs[cur] = []
                continue
            if cur is None:
                continue
            m = INSN_RE.search(line)
            if m:
                addr = int(m.group(1), 16)
                op = m.group(3)
                funcs[cur].append((addr, op, m.group(4)))
    return funcs


def base(op):
    return op.split(".")[0]


def loops_of(insns):
    """Backward-branch loops containing tensor-core instructions, outermost
    first; nested/overlapping loops are reported once (the largest)."""
    found = []
    idx = {a: i for i, (a, _, _) in enumerate(insns)}
    for i, (addr, op, args) in enumerate(insns):
        if base(op) != "BRA":
            continue
        m = re.search(r"0x([0-9a-f]+)", args)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt >= addr or tgt not in idx:
            continue
        j = idx[tgt]
        body = insns[j:i + 1]
        n_mma = sum(1 for _, o, _ in body if base(o) in ("IMMA", "HMMA"))
        if n_mma:
            found.append((j, i, n_mma))
    found.sort(key=lambda x: x[0] - x[1])  # largest first
    out = []
    for j, i, n in found:
        if any(j >= a and i <= b for a, b, _ in out):
            continue
        out.append((j, i, n))
    return sorted(out)


def hist(body):
    c = Counter(base(o) for _, o, _ in body)
    # XOR flavours: 3-input (LUT 0x96) vs 2-input (LUT 0x3c) LOP3.
    for _, o, args in body:
        if base(o) == "LOP3" and not args.lstrip().startswith(("P", "UP")):
            if "0x96" in args:
                c["xor3"] += 1
            elif "0x3c" in args:
                c["xor2"] += 1
    return c


def row(label, c, total):
    cols = " ".join(f"{k}={c.get(k, 0)}" for k in KEY if c.get(k, 0))
    print(f"  {label:<10} total={total:<5} {cols}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    funcs = parse(sys.argv[1])
    filters = sys.argv[2:]
    for name, insns in funcs.items():
        if "FusedKernelEntry" not in name or not all(f in name for f in filters):
            continue
        lps = loops_of(insns)
        short = re.findall(r"ILi(\d+)ELi(\d+)ELi(\d+)E", name)
        arch = re.search(r"arch4(Sm\d+)", name)
        tag = (arch.group(1) if arch else "?") + " " + " / ".join("x".join(s) for s in short[:3])
        print(f"{tag}  [{len(insns)} insns]")
        if not lps:
            print("  (no tensor-core loop found)")
            continue
        for j, i, _ in lps:
            body = insns[j:i + 1]
            # The tile-xor dump loop (--align-test-prod) stores to global.
            kind = "dump-loop" if any(base(o) == "STG" for _, o, _ in body) else "mine-loop"
            row(kind, hist(body), len(body))
        j, i, _ = lps[-1]
        rest = insns[i + 1:]
        row("tail", hist(rest), len(rest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
