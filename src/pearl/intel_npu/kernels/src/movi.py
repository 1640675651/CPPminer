"""Build ACT-SHAVE kernels for the NPU compiler with the MoviTools DLLs (MVC_DEPEND), with full control of flags.

npunlock's shavecc fixes -O2 and offers no way to read the assembly, so this drives the same three DLL exports
directly (moviCompile64 `main`, moviAsm64 `process`, moviLLD64 `process`), each stage in a fresh subprocess
because the DLLs keep global state.

The NPU compiler links its kernels with the same script used here (.text at 0x1d000000, .arg.data at
0x1e000000) and takes the entry point from e_entry, so the result can replace or extend the embedded kernels:

    python movi.py tile_jackpot.c tile_jackpot --out DIR    # -> DIR/tile_jackpot.3720xx{,_lsu0_wo}.{elf,s}
    python movi.py tile_jackpot.c tile_jackpot --out DIR --asm   # also print the generated assembly

The patched compiler loads every <name>.3720xx*.elf from NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR (the miner points it
at kernels/intel_npu next to the executable). ../build_kernels.ps1 builds both of the miner's kernels.

MOVITOOLS_DIR: the MVC_DEPEND directory (see ../../README.md, "SHAVE kernels"). NPU_COMPILER_SRC: the npu_compiler
checkout, for its kernel linker script (default: the prepare script's third_party/npu_compiler-src).
"""

import argparse
import ctypes
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # src/pearl/intel_npu/kernels/src -> repository root
# MoviTools (moviCompile64/moviAsm64/moviLLD64 DLLs and lib/): the MVC_DEPEND directory. NPUNLOCK_MOVITOOLS_DIR is
# accepted too (npunlock's name for it).
MOVI_DIR = Path(os.environ.get("MOVITOOLS_DIR") or os.environ.get("NPUNLOCK_MOVITOOLS_DIR") or "")
NPU_COMPILER_DIR = Path(os.environ.get("NPU_COMPILER_SRC") or REPO / "third_party" / "npu_compiler-src")
LINKER_SCRIPT_PATH = NPU_COMPILER_DIR / "sw_runtime_kernels" / "kernels" / "prebuild" / "shave_kernel.ld"

# Flags of the compiler's own kernels (sw_runtime_kernels/kernels/CMakeLists.txt), in -cc1 form.
DEFAULT_CFLAGS = ["-O3", "-ffunction-sections", "-fdata-sections",
                  "-mllvm", "-shave-align-stack=32"]
# MTL DDR split-access workaround used for the *_lsu0_wo variant.
LSU0_WO_CFLAGS = ["-mllvm", "-shave-lsu-load-policy=use-only-lsu1", "-mllvm", "-shave-lsu-store-policy=prefer-lsu0"]


class _Buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _BufferList(ctypes.Structure):
    _fields_ = [("items", ctypes.POINTER(_Buffer)), ("count", ctypes.c_size_t)]


def _argv(args):
    arr = (ctypes.c_char_p * len(args))(*[a.encode() for a in args])
    return len(args), arr


def _stage_in_process(stage, args, inputs):
    """Runs one DLL stage in this process. Returns (return code, output bytes, diagnostic bytes)."""
    os.add_dll_directory(str(MOVI_DIR / "bin"))
    dll = ctypes.CDLL(str(MOVI_DIR / "bin" / {"compile": "moviCompile64.dll", "asm": "moviAsm64.dll",
                                              "link": "moviLLD64.dll"}[stage]))
    argc, argv = _argv(args)
    out, out_size = ctypes.c_void_p(), ctypes.c_size_t()
    diag, diag_size = ctypes.c_void_p(), ctypes.c_size_t()
    if stage in ("compile", "asm"):
        src = ctypes.create_string_buffer(inputs[0], len(inputs[0]) + 1)  # the compiler reads a terminator
        fn = dll.main if stage == "compile" else dll.process
        size_type = ctypes.c_size_t if stage == "compile" else ctypes.c_uint32
        fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p, size_type,
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        rc = fn(argc, argv, ctypes.cast(src, ctypes.c_void_p), len(inputs[0]), ctypes.byref(out),
                ctypes.byref(out_size), ctypes.byref(diag), ctypes.byref(diag_size))
        data = ctypes.string_at(out, out_size.value) if out.value else b""
        text = ctypes.string_at(diag, diag_size.value) if diag.value else b""
    else:
        keep = [ctypes.create_string_buffer(b, len(b)) for b in inputs]
        items = (_Buffer * len(inputs))(*[_Buffer(ctypes.cast(k, ctypes.c_void_p), len(b))
                                          for k, b in zip(keep, inputs)])
        lists = (_BufferList * 8)()
        lists[0] = _BufferList(items, len(inputs))
        outputs = (_Buffer * 8)()
        dll.process.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p, ctypes.c_void_p]
        rc = dll.process(argc, argv, ctypes.cast(lists, ctypes.c_void_p), ctypes.cast(outputs, ctypes.c_void_p))
        data = ctypes.string_at(outputs[0].data, outputs[0].size) if outputs[0].data else b""
        text = b""
    return rc, data, text


def _stage(stage, args, inputs):
    proc = subprocess.run([sys.executable, "-I", __file__, "--_stage"], input=pickle.dumps((stage, args, inputs)),
                          capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"{stage} worker crashed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")
    rc, data, text = pickle.loads(proc.stdout)
    if rc != 0 or not data:
        raise RuntimeError(f"{stage} failed ({rc}):\n{text.decode(errors='replace')}")
    return data, text.decode(errors="replace")


def build(source: bytes, defines=(), cflags=None, lsu0_wo=False, entry="controlled_act", extra_asm=b""):
    """C source (+ hand-written assembly appended to the compiler's) -> (assembly text, linked ELF bytes)."""
    flags = list(cflags or DEFAULT_CFLAGS) + (LSU0_WO_CFLAGS if lsu0_wo else [])
    args = (["moviCompile.dll", "-cc1", "-triple", "shave", "-target-cpu", "3720xx", "-S", "-x", "c", "entry.c"]
            + flags + [f"-D{d}" for d in defines] + ["-o", "-"])
    asm, warnings = _stage("compile", args, [source])
    if extra_asm:
        marker = b".nowarnend"
        cut = asm.rfind(marker)
        asm = asm[:cut] + extra_asm + bytes([10]) + asm[cut:]
    if os.environ.get("MOVI_KEEP_ASM"):
        Path(os.environ["MOVI_KEEP_ASM"]).write_bytes(asm)
    if warnings.strip():
        print(warnings, file=sys.stderr)
    obj, _ = _stage("asm", ["moviAsm.dll", "--cv", "3720xx", "--noSPrefixing"], [asm])
    link_args = ["moviLLD.dll", "-flavor", "gnu", "-EL", "-e", entry, "-z", "max-page-size=0x10", "--gc-sections",
                 str(MOVI_DIR / "lib" / "mlibm.a")]
    elf, _ = _stage("link", link_args, [obj, LINKER_SCRIPT_PATH.read_bytes()])
    return asm.decode(errors="replace"), elf


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_stage":
        stage, args, inputs = pickle.loads(sys.stdin.buffer.read())
        result = _stage_in_process(stage, args, inputs)
        sys.stdout.buffer.write(pickle.dumps(result))
        sys.stdout.flush()
        os._exit(0)  # skip DLL teardown

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("name", help="kernel name the compiler looks up, e.g. tile_xor")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("-D", dest="defines", action="append", default=[])
    parser.add_argument("--asm", action="store_true", help="print the generated assembly")
    parser.add_argument("--extra-asm", action="append", default=[],
                        help="hand-written assembly file(s) appended to the compiled C (e.g. from gen_tile_xor_asm.py)")
    args = parser.parse_args()
    if not (MOVI_DIR / "bin" / "moviCompile64.dll").is_file():
        sys.exit("set MOVITOOLS_DIR to the MoviTools MVC_DEPEND directory (bin/moviCompile64.dll, lib/mlibm.a)")
    if not LINKER_SCRIPT_PATH.is_file():
        sys.exit(f"kernel linker script not found: {LINKER_SCRIPT_PATH} (set NPU_COMPILER_SRC to the npu_compiler "
                 "checkout)")

    source = Path(args.source).read_bytes()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix, lsu0_wo in (("3720xx", False), ("3720xx_lsu0_wo", True)):
        extra = b"".join(Path(f).read_bytes() for f in args.extra_asm)
        asm, elf = build(source, args.defines, lsu0_wo=lsu0_wo, extra_asm=extra)
        (out / f"{args.name}.{suffix}.elf").write_bytes(elf)
        (out / f"{args.name}.{suffix}.s").write_text(asm)
        print(f"wrote {out / f'{args.name}.{suffix}.elf'} ({len(elf)} bytes)")
        if args.asm and not lsu0_wo:
            print(asm)


if __name__ == "__main__":
    main()
