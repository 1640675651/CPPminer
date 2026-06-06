#!/usr/bin/env python3
"""
Host bridge for pearl_mine.cu plain_proof mode.

Matrix generation and noise precompute run natively in C (pearl_noise.c).
Python is used only for Merkle proof assembly and verification (pearl_mining).

  build  — assemble PlainProof base64 from a GPU hit (t_rows, t_cols)
  verify — offline verify a plain_proof against pool target (BE hex)
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miner1 import MiningConfig, generate_ab  # noqa: E402
from plain_proof_mine import (  # noqa: E402
    BZMINER_COLS_PATTERN as SCATTERED_COLS_PATTERN,
    BZMINER_ROWS_PATTERN as SCATTERED_ROWS_PATTERN,
    CONTIGUOUS_COLS_PATTERN,
    CONTIGUOUS_ROWS_PATTERN,
    build_plain_proof,
    bzminer_mining_config as scattered_mining_config,
    contiguous_mining_config,
    job_key,
)


def _cfg(m: int, n: int, k: int, r: int) -> MiningConfig:
    return MiningConfig(m=m, n=n, k=k, r=r)


def _seed(header: bytes, nonce: int) -> bytes:
    if nonce == 0:
        return header
    return __import__("blake3").blake3(header + struct.pack("<Q", nonce)).digest()


def _load_ab(header: bytes, cfg: MiningConfig, nonce: int, a_file: str | None, b_file: str | None):
    if a_file and b_file:
        A = np.frombuffer(Path(a_file).read_bytes(), dtype=np.int8).reshape(cfg.m, cfg.k)
        bt = np.frombuffer(Path(b_file).read_bytes(), dtype=np.int8).reshape(cfg.n, cfg.k)
        return A, bt
    seed = _seed(header, nonce)
    A, B = generate_ab(seed, cfg)
    return A, np.ascontiguousarray(B.T)


def verify_share(header_bytes: bytes, plain_proof, pool_target_be: int) -> tuple[bool, str]:
    import pearl_mining

    header = pearl_mining.IncompleteBlockHeader.from_bytes(list(header_bytes))
    target_bytes = pool_target_be.to_bytes(32, "big")
    return pearl_mining.verify_plain_proof_with_pool_target(header, plain_proof, target_bytes)


def cmd_build(args: argparse.Namespace) -> int:
    header = Path(args.header).read_bytes()
    cfg = _cfg(args.m, args.n, args.k, args.r)
    if args.contiguous_tiles:
        pearl_cfg = contiguous_mining_config(args.k, args.r)
        rows_pat = CONTIGUOUS_ROWS_PATTERN
        cols_pat = CONTIGUOUS_COLS_PATTERN
    else:
        pearl_cfg = scattered_mining_config(args.k, args.r)
        rows_pat = SCATTERED_ROWS_PATTERN
        cols_pat = SCATTERED_COLS_PATTERN

    A, bt = _load_ab(header, cfg, args.nonce, args.a_file, args.b_file)

    t_rows = args.t_rows
    t_cols = args.t_cols
    a_rows = [t_rows + i for i in rows_pat]
    bt_rows = [t_cols + i for i in cols_pat]

    key = job_key(header, pearl_cfg)
    pp = build_plain_proof(A, bt, key, a_rows, bt_rows, cfg.m, cfg.n, cfg.k, cfg.r)
    b64 = pp.to_base64()
    if args.out_b64:
        Path(args.out_b64).write_text(b64, encoding="utf-8")
    sys.stdout.write(b64)
    sys.stdout.flush()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    import pearl_mining

    header = Path(args.header).read_bytes()
    if args.proof_file:
        b64 = Path(args.proof_file).read_text(encoding="utf-8").strip()
    elif args.b64:
        b64 = args.b64.strip()
    else:
        b64 = sys.stdin.read().strip()
    pp = pearl_mining.PlainProof.from_base64(b64)
    target = int(args.target_hex, 16)
    ok, msg = verify_share(header, pp, target)
    if ok:
        print(f"verify OK: {msg}", flush=True)
        return 0
    print(f"verify FAIL: {msg}", file=sys.stderr, flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="plain_proof host bridge for pearl_mine.cu")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="build plain_proof base64")
    p_build.add_argument("--header", required=True)
    p_build.add_argument("--m", type=int, required=True)
    p_build.add_argument("--n", type=int, required=True)
    p_build.add_argument("--k", type=int, default=4096)
    p_build.add_argument("--r", type=int, default=256)
    p_build.add_argument("--nonce", type=int, default=0)
    p_build.add_argument("--t-rows", type=int, required=True)
    p_build.add_argument("--t-cols", type=int, required=True)
    p_build.add_argument("--a-file", help="A int8 raw (else regenerate)")
    p_build.add_argument("--b-file", help="B^T int8 raw (else regenerate)")
    p_build.add_argument("--out-b64", help="also write base64 to this file")
    p_build.add_argument(
        "--contiguous-tiles",
        action="store_true",
        help="contiguous 8x16 tile rows/cols (debug; default is production scattered layout)",
    )
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="verify plain_proof base64 against pool target")
    p_verify.add_argument("--header", required=True)
    p_verify.add_argument("--target-hex", required=True, help="pool target as 64-char BE hex")
    g = p_verify.add_mutually_exclusive_group(required=True)
    g.add_argument("--proof-file")
    g.add_argument("--b64")
    p_verify.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
