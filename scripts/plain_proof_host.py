#!/usr/bin/env python3
"""
Self-contained plain_proof CLI for CPminer.

Build and verify use cp_proof_ffi (Rust cdylib from rust/cp-proof-ffi). No pearl-miner
or pearl_mining Python packages required.

  build  — assemble PlainProof base64 from matrices + tile anchor (t_rows, t_cols)
  verify — offline verify a plain_proof against pool target (BE hex)

Requires: Python 3.10+, numpy, blake3 (pip install blake3 numpy)
Optional: cp_proof_ffi shared library (built by build.ps1 / cargo in rust/cp-proof-ffi)
"""
from __future__ import annotations

import argparse
import ctypes
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Precomputed 52-byte mining configs (match cp_noise.c / GPU job_key).
SCATTERED_CONFIG = bytes([
    0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x07, 0x01, 0x01, 0x03, 0x00, 0x00, 0x00, 0x01,
    0x0f, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
])
CONTIGUOUS_CONFIG = bytes([
    0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
])
CUTLASS_CONFIG = bytes([
    0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x03, 0x01, 0x03, 0x00, 0x00, 0x1f, 0x07,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
])

TILE_LAYOUT_SCATTERED = 0
TILE_LAYOUT_CONTIGUOUS = 1
TILE_LAYOUT_CUTLASS = 2

HEADER_SIZE = 76
PLAIN_PROOF_B64_MAX = 4 * 1024 * 1024
ERR_BUF = 4096

SCRIPTS_DIR = Path(__file__).resolve().parent
CPMINER_ROOT = SCRIPTS_DIR.parent


def _blake3_stream(seed: bytes, tag: bytes, nbytes: int) -> bytes:
    import blake3

    return blake3.blake3(tag + seed).digest(length=nbytes)


def generate_ab(seed: bytes, m: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic A (m×k) and B (k×n) int8 matrices from seed (matches cp_noise.c)."""
    raw_a = _blake3_stream(seed, b"matrix_A", m * k)
    a = (np.frombuffer(raw_a, dtype=np.uint8) % 128 - 64).astype(np.int8).reshape(m, k)
    raw_b = _blake3_stream(seed, b"matrix_B", k * n)
    b = (np.frombuffer(raw_b, dtype=np.uint8) % 128 - 64).astype(np.int8).reshape(k, n)
    return a, b


def effective_seed(header: bytes, nonce: int) -> bytes:
    if nonce == 0:
        return header
    import blake3

    return blake3.blake3(header + struct.pack("<Q", nonce)).digest()


def _ffi_lib_candidates() -> list[Path]:
    env = os.environ.get("CP_PROOF_FFI")
    if env:
        yield Path(env)
    rel = CPMINER_ROOT / "rust" / "cp-proof-ffi" / "target" / "release"
    for name in ("cp_proof_ffi.dll", "libcp_proof_ffi.so", "libcp_proof_ffi.dylib"):
        yield rel / name


def _load_ffi() -> ctypes.CDLL:
    last_err: Exception | None = None
    for path in _ffi_lib_candidates():
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            break
        except OSError as e:
            last_err = e
    else:
        hint = (
            "Build cp_proof_ffi first: cd CPminer/rust/cp-proof-ffi && cargo build --release\n"
            "Or set CP_PROOF_FFI to the shared library path."
        )
        msg = f"cp_proof_ffi not found ({last_err})" if last_err else "cp_proof_ffi not found"
        raise SystemExit(f"{msg}\n{hint}")

    lib.cp_proof_build.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.cp_proof_build.restype = ctypes.c_int

    lib.cp_proof_verify.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.cp_proof_verify.restype = ctypes.c_int
    return lib


_FFI: ctypes.CDLL | None = None


def ffi() -> ctypes.CDLL:
    global _FFI
    if _FFI is None:
        _FFI = _load_ffi()
    return _FFI


def _err_msg(err_buf: ctypes.Array) -> str:
    raw = bytes(err_buf)
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    return raw.decode("utf-8", errors="replace") or "unknown error"


def proof_build(
    header: bytes,
    mining_config: bytes,
    a: np.ndarray,
    bt: np.ndarray,
    m: int,
    n: int,
    k: int,
    rank: int,
    t_rows: int,
    t_cols: int,
    tile_layout: int,
) -> str:
    if len(header) != HEADER_SIZE:
        raise ValueError(f"header must be {HEADER_SIZE} bytes, got {len(header)}")
    if len(mining_config) != 52:
        raise ValueError(f"mining_config must be 52 bytes, got {len(mining_config)}")
    a = np.ascontiguousarray(a, dtype=np.int8)
    bt = np.ascontiguousarray(bt, dtype=np.int8)
    if a.shape != (m, k):
        raise ValueError(f"A shape {a.shape} != ({m}, {k})")
    if bt.shape != (n, k):
        raise ValueError(f"B^T shape {bt.shape} != ({n}, {k})")

    out = ctypes.create_string_buffer(PLAIN_PROOF_B64_MAX)
    err = ctypes.create_string_buffer(ERR_BUF)
    header_buf = (ctypes.c_uint8 * len(header)).from_buffer_copy(header)
    cfg_arr = (ctypes.c_uint8 * 52).from_buffer_copy(mining_config)

    rc = ffi().cp_proof_build(
        ctypes.cast(header_buf, ctypes.c_void_p),
        len(header),
        cfg_arr,
        52,
        a.ctypes.data_as(ctypes.c_void_p),
        bt.ctypes.data_as(ctypes.c_void_p),
        m,
        n,
        k,
        rank,
        t_rows,
        t_cols,
        tile_layout,
        out,
        PLAIN_PROOF_B64_MAX,
        err,
        ERR_BUF,
    )
    if rc != 0:
        raise RuntimeError(_err_msg(err))
    return out.value.decode("ascii")


def proof_verify(header: bytes, proof_b64: str, pool_target_be: int) -> None:
    if len(header) != HEADER_SIZE:
        raise ValueError(f"header must be {HEADER_SIZE} bytes, got {len(header)}")
    target = pool_target_be.to_bytes(32, "big")
    err = ctypes.create_string_buffer(ERR_BUF)
    header_buf = (ctypes.c_uint8 * len(header)).from_buffer_copy(header)
    b64_buf = proof_b64.strip().encode("ascii")
    b64_c = ctypes.create_string_buffer(b64_buf)
    target_arr = (ctypes.c_uint8 * 32).from_buffer_copy(target)

    rc = ffi().cp_proof_verify(
        ctypes.cast(header_buf, ctypes.c_void_p),
        len(header),
        ctypes.cast(b64_c, ctypes.c_void_p),
        len(b64_buf),
        ctypes.cast(target_arr, ctypes.c_void_p),
        err,
        ERR_BUF,
    )
    if rc != 0:
        raise RuntimeError(_err_msg(err))


def _tile_params(args: argparse.Namespace) -> tuple[bytes, int]:
    if args.cutlass_tiles:
        return CUTLASS_CONFIG, TILE_LAYOUT_CUTLASS
    if args.contiguous_tiles:
        return CONTIGUOUS_CONFIG, TILE_LAYOUT_CONTIGUOUS
    return SCATTERED_CONFIG, TILE_LAYOUT_SCATTERED


def _load_ab(
    header: bytes,
    m: int,
    n: int,
    k: int,
    nonce: int,
    a_file: str | None,
    b_file: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if a_file and b_file:
        a = np.frombuffer(Path(a_file).read_bytes(), dtype=np.int8).reshape(m, k)
        bt = np.frombuffer(Path(b_file).read_bytes(), dtype=np.int8).reshape(n, k)
        return a, bt
    seed = effective_seed(header, nonce)
    a, b = generate_ab(seed, m, n, k)
    return a, np.ascontiguousarray(b.T)


def cmd_build(args: argparse.Namespace) -> int:
    header = Path(args.header).read_bytes()
    mining_cfg, tile_layout = _tile_params(args)
    a, bt = _load_ab(header, args.m, args.n, args.k, args.nonce, args.a_file, args.b_file)
    b64 = proof_build(
        header,
        mining_cfg,
        a,
        bt,
        args.m,
        args.n,
        args.k,
        args.r,
        args.t_rows,
        args.t_cols,
        tile_layout,
    )
    if args.out_b64:
        Path(args.out_b64).write_text(b64, encoding="utf-8")
    sys.stdout.write(b64)
    sys.stdout.flush()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    header = Path(args.header).read_bytes()
    if args.proof_file:
        b64 = Path(args.proof_file).read_text(encoding="utf-8").strip()
    elif args.b64:
        b64 = args.b64.strip()
    else:
        b64 = sys.stdin.read().strip()
    target = int(args.target_hex, 16)
    try:
        proof_verify(header, b64, target)
    except RuntimeError as e:
        print(f"verify FAIL: {e}", file=sys.stderr, flush=True)
        return 1
    print("verify OK: Mining solution verified against pool target", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CPminer plain_proof build/verify (cp_proof_ffi)")
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
    p_build.add_argument("--a-file", help="A int8 raw (else regenerate from header)")
    p_build.add_argument("--b-file", help="B^T int8 raw (else regenerate)")
    p_build.add_argument("--out-b64", help="also write base64 to this file")
    p_build.add_argument(
        "--contiguous-tiles",
        action="store_true",
        help="contiguous 8x16 tile rows/cols (debug; default is production scattered layout)",
    )
    p_build.add_argument(
        "--cutlass-tiles",
        action="store_true",
        help="CUTLASS Case 7.1 epilogue scatter (16 A rows + 8 B^T rows)",
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
