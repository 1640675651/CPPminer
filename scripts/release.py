#!/usr/bin/env python3
"""Package a Windows release: cppminer.exe + every non-system DLL it needs + kernels, zipped.

Usage (after build.ps1):
  python scripts/release.py                       # version from CHANGELOG.md, output in dist/
  python scripts/release.py --version v0.5 --suffix windows_x64_cuda
  python scripts/release.py --no-vcrt             # rely on the user's VC++ redistributable

DLLs are found by reading the PE import (and delay-import) tables of the exe, then of every
DLL that gets bundled, so new dependencies are picked up automatically:
  - next to the exe (cp_*_wgpu_ffi.dll, ...)      -> bundled
  - CUDA runtime (cudart64_*.dll)                  -> bundled from CUDA_PATH\\bin
  - VC++ runtime (msvcp140, vcruntime140, vcomp140) -> bundled from the VS redist folder
  - OpenCL.dll                                     -> skipped (installed by the GPU driver)
  - Windows system DLLs / api-ms-* sets            -> skipped
Anything else is an error, so a release never ships with a missing dependency.

A build with the Intel NPU backend (the exe imports openvino.dll) also gets a second zip,
<name>_intel_npu_runtime.zip: the OpenVINO NPU plugin, the patched NPU compiler (~85 MB) and the
NPU kernels, with the same top folder so it extracts over the main one. OpenVINO loads the plugin
and the compiler at run time (they are not imported, hence NPU_RUNTIME_DLLS). openvino.dll itself
stays in the main zip: the exe imports it, and cannot delay-load it (OpenVINO's headers import a
data symbol), so without it the exe would not start at all.
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")

DRIVER_DLLS = {"opencl.dll", "nvcuda.dll", "vulkan-1.dll"}
VCRT_PATTERNS = ("msvcp140", "vcruntime140", "vcomp140", "concrt140")
# Intel NPU runtime, next to the exe after build.ps1: OpenVINO loads the plugin, which loads the
# patched compiler through the loader.
NPU_RUNTIME_DLLS = ("openvino_intel_npu_plugin.dll", "openvino_intel_npu_compiler_loader.dll",
                    "openvino_intel_npu_compiler.dll")
NPU_KERNEL_DIR = os.path.join("kernels", "intel_npu")


def pe_imports(path):
    """Names of DLLs imported by a PE file: normal and delay-load import tables."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        raise ValueError(f"{path}: not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path}: bad PE signature")
    nsect, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    pe32plus = magic == 0x20B
    image_base = (struct.unpack_from("<Q", data, opt + 24)[0] if pe32plus
                  else struct.unpack_from("<I", data, opt + 28)[0])
    ddir = opt + (112 if pe32plus else 96)
    sections = []
    for i in range(nsect):
        s = opt + opt_size + 40 * i
        vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, max(vsize, rawsize), rawptr))

    def off(rva):
        for va, size, rawptr in sections:
            if va <= rva < va + size:
                return rva - va + rawptr
        raise ValueError(f"{path}: RVA 0x{rva:x} outside sections")

    def cstr(rva):
        o = off(rva)
        return data[o:data.index(b"\0", o)].decode("ascii")

    names = []
    imp_rva, imp_size = struct.unpack_from("<II", data, ddir + 8 * 1)
    if imp_rva and imp_size:
        o = off(imp_rva)
        while True:
            desc = struct.unpack_from("<5I", data, o)
            if not any(desc):
                break
            names.append(cstr(desc[3]))
            o += 20
    dly_rva, dly_size = struct.unpack_from("<II", data, ddir + 8 * 13)
    if dly_rva and dly_size:
        o = off(dly_rva)
        while True:
            attrs, name_rva = struct.unpack_from("<II", data, o)
            if not name_rva:
                break
            names.append(cstr(name_rva if attrs & 1 else name_rva - image_base))
            o += 32
    return names


def find_vs_redist():
    """Folders holding the app-local VC++ CRT and OpenMP DLLs (x64), newest toolset."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           r"Microsoft Visual Studio\Installer\vswhere.exe")
    roots = []
    if os.path.exists(vswhere):
        out = subprocess.run([vswhere, "-all", "-products", "*", "-property", "installationPath"],
                             capture_output=True, text=True).stdout
        roots = [l.strip() for l in out.splitlines() if l.strip()]
    dirs = []
    for root in roots:
        versions = [d for d in glob.glob(os.path.join(root, r"VC\Redist\MSVC\*"))
                    if re.fullmatch(r"[\d.]+", os.path.basename(d))]
        versions.sort(key=lambda d: [int(x) for x in os.path.basename(d).split(".")], reverse=True)
        for v in versions:
            found = glob.glob(os.path.join(v, r"x64\Microsoft.VC*.CRT")) + \
                    glob.glob(os.path.join(v, r"x64\Microsoft.VC*.OpenMP"))
            if found:
                dirs.extend(found)
                return dirs
    return dirs


def find_cuda_bin():
    dirs = []
    if os.environ.get("CUDA_PATH"):
        dirs.append(os.path.join(os.environ["CUDA_PATH"], "bin"))
    dirs += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"),
                   reverse=True)
    return dirs


def locate(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def is_system_dll(lname):
    if lname.startswith(("api-ms-win-", "ext-ms-")):
        return True
    return os.path.isfile(os.path.join(SYSTEM32, lname))


def resolve_dlls(roots, exe_dir, bundle_vcrt, bundle_cudart, elsewhere=None):
    """DLLs needed by the PE files in roots (themselves included unless one is the exe).
    elsewhere: lower name -> reason, for DLLs shipped in another zip (not followed)."""
    vcrt_dirs = find_vs_redist() if bundle_vcrt else []
    cuda_dirs = find_cuda_bin() if bundle_cudart else []
    bundled = {}   # lower name -> source path
    skipped = dict(elsewhere or {})   # lower name -> reason
    errors = []
    queue = []
    for root in roots:
        if not root.lower().endswith(".exe"):
            bundled[os.path.basename(root).lower()] = root
        queue.append(root)
    seen_files = set()
    while queue:
        cur = queue.pop()
        if cur in seen_files:
            continue
        seen_files.add(cur)
        for name in pe_imports(cur):
            lname = name.lower()
            if lname in bundled or lname in skipped:
                continue
            local = locate(name, [exe_dir])
            if local:
                bundled[lname] = local
                queue.append(local)
            elif lname.startswith("cudart64_"):
                if not bundle_cudart:
                    skipped[lname] = "CUDA runtime (--no-cudart)"
                    continue
                src = locate(name, cuda_dirs)
                if src:
                    bundled[lname] = src
                else:
                    errors.append(f"{name}: not found under CUDA_PATH\\bin (needed by {os.path.basename(cur)})")
            elif lname.startswith(VCRT_PATTERNS):
                if not bundle_vcrt:
                    skipped[lname] = "VC++ redistributable (--no-vcrt)"
                    continue
                src = locate(name, vcrt_dirs) or locate(name, [SYSTEM32])
                if src:
                    if src.startswith(SYSTEM32):
                        print(f"warning: {name} taken from System32 (VS redist folder not found)")
                    bundled[lname] = src
                else:
                    errors.append(f"{name}: VC++ runtime DLL not found")
            elif lname in DRIVER_DLLS:
                skipped[lname] = "installed by the GPU driver"
            elif is_system_dll(lname):
                skipped[lname] = "Windows"
            else:
                errors.append(f"{name}: unknown dependency of {os.path.basename(cur)}; "
                              f"put it next to the exe or teach release.py about it")
    return bundled, skipped, errors


def default_version():
    try:
        with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
            for line in f:
                m = re.match(r"##\s+(v[\w.\-]+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return "dev"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exe", default=os.path.join(ROOT, "cppminer.exe"))
    ap.add_argument("--version", default=default_version())
    ap.add_argument("--suffix", default="windows_x64", help="name suffix, e.g. windows_x64_cuda")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--no-vcrt", action="store_true", help="do not bundle the VC++ runtime DLLs")
    ap.add_argument("--no-cudart", action="store_true", help="do not bundle cudart64_*.dll")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"{exe} not found; run build.ps1 first", file=sys.stderr)
        return 1

    exe_dir = os.path.dirname(exe)
    vcrt, cudart = not args.no_vcrt, not args.no_cudart
    has_npu = "openvino.dll" in (n.lower() for n in pe_imports(exe))

    bundled, skipped, errors = resolve_dlls([exe], exe_dir, vcrt, cudart)
    npu_files = []
    if has_npu:
        roots = []
        for n in NPU_RUNTIME_DLLS:
            p = os.path.join(exe_dir, n)
            if os.path.isfile(p):
                roots.append(p)
            else:
                errors.append(f"{n}: Intel NPU runtime file missing next to the exe (run build.ps1 "
                              f"-Backend ...,IntelNpu)")
        # Everything the NPU runtime needs that the main zip does not already carry.
        npu_dlls, _, npu_errors = resolve_dlls(roots, exe_dir, vcrt, cudart,
                                               elsewhere={n: "main zip" for n in bundled})
        errors += npu_errors
        npu_files = [(src, os.path.basename(src)) for src in sorted(npu_dlls.values())]
        elfs = sorted(glob.glob(os.path.join(exe_dir, NPU_KERNEL_DIR, "*.elf")))
        if not elfs:
            errors.append(f"{NPU_KERNEL_DIR}\\*.elf: Intel NPU kernels missing next to the exe")
        npu_files += [(p, os.path.join(NPU_KERNEL_DIR, os.path.basename(p))) for p in elfs]
    if errors:
        for e in errors:
            print("error: " + e, file=sys.stderr)
        return 1

    name = f"cppminer_{args.version}_{args.suffix}"
    files = [(exe, os.path.basename(exe))]
    files += [(src, os.path.basename(src)) for src in sorted(bundled.values())]
    for cl in sorted(glob.glob(os.path.join(exe_dir, "kernels", "*.cl"))):
        files.append((cl, os.path.join("kernels", os.path.basename(cl))))

    zips = [package(args.out, name, name, files)]
    if npu_files:
        # Same top folder as the main zip, so it extracts over it.
        zips.append(package(args.out, name + "_intel_npu_runtime", name, npu_files))

    if skipped:
        print("not bundled:")
        for lname, why in sorted(skipped.items()):
            if why != "Windows":
                print(f"  {lname:<34} {why}")
        n_sys = sum(1 for why in skipped.values() if why == "Windows")
        print(f"  ({n_sys} Windows system DLLs)")
    for zip_path in zips:
        print(f"zip: {zip_path} ({os.path.getsize(zip_path) / 1048576:.1f} MiB)")
        print(f"sha256: {sha256(zip_path)}")
    return 0


def package(out, zip_name, top, files):
    """Stages files (source, relative path) in out/zip_name and zips them under top/."""
    stage = os.path.join(out, zip_name)
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    for src, rel in files:
        dst = os.path.join(stage, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    zip_path = os.path.join(out, zip_name + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for _, rel in files:
            z.write(os.path.join(stage, rel), os.path.join(top, rel))

    print(f"{zip_name}:")
    for src, rel in files:
        print(f"  {rel:<50} {os.path.getsize(src) / 1048576:8.2f} MiB  <- {src}")
    return zip_path


if __name__ == "__main__":
    sys.exit(main())
