"""Rough static cycle count per basic block of moviCompile SHAVE assembly: one cycle per bundle, NOP n = n.

    python asm_cycles.py tile_jackpot.3720xx.s [N]    # the N blocks with the most VAU ops (default 8)

The .s files come out of movi.py next to the ELFs (../build_kernels.ps1 -KeepAsm).
"""
import re
import sys
from collections import Counter

blocks = []
cur = {"label": "<entry>", "cycles": 0, "vau": 0, "lsu": 0}
for line in open(sys.argv[1]):
    s = line.split("//")[0].strip()
    if not s or s.startswith("."):
        if re.match(r"^\.LBB\S*:", s) or (s.endswith(":") and not s.startswith(".")):
            blocks.append(cur)
            cur = {"label": s, "cycles": 0, "vau": 0, "lsu": 0}
        continue
    if s.endswith(":"):
        blocks.append(cur)
        cur = {"label": s, "cycles": 0, "vau": 0, "lsu": 0}
        continue
    body = s[2:].strip() if s.startswith("||") else s
    if not s.startswith("||"):
        m = re.match(r"NOP\s*(\d*)", body)
        cur["cycles"] += int(m.group(1) or 1) if m else 1
    cur["vau"] += body.startswith("VAU.")
    cur["lsu"] += body.startswith("LSU")
blocks.append(cur)
for b in sorted(blocks, key=lambda b: -b["vau"])[: int(sys.argv[2]) if len(sys.argv) > 2 else 8]:
    print(f'{b["label"]:12s} cycles={b["cycles"]:5d} vau={b["vau"]:4d} lsu={b["lsu"]:4d}')
