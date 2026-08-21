# Termux build notes

## Missing `ar` (jemalloc / Rust)

`tikv-jemalloc-sys` fails to build without `ar`. `pkg install binutils` does not provide it.

```bash
cd "$PREFIX/bin"
ln -s llvm-ar ar
```

## OpenCL + libc++ conflict

Vendor `libOpenCL.so` depends on Android/system (and often VNDK) libs.  
`cppminer` (Termux clang) depends on Termux `$PREFIX/lib/libc++_shared.so`.

If system/vendor paths come first on `LD_LIBRARY_PATH`, the process can load system libc++ and die with:

```text
cannot locate symbol "_ZTVNSt6__ndk117bad_function_callE"
```

**Fix:** statically link libc++, then put the full vendor/system search path first so the OpenCL ICD can resolve its deps.

```bash
pkg install ndk-multilib
rm -rf build
./build.sh --backend cpu,opencl
```

CMake links `$PREFIX/<triple>/lib/libc++_static.a` + `libc++abi.a` with `-nostdlib++` (auto on Termux, or `-DCP_STATIC_LIBCXX=ON`).

On Termux, CMake prefers `/vendor/lib64/libOpenCL.so` over ocl-icd when present.

Verify the binary no longer `NEEDED`s shared libc++:

```bash
readelf -d ./cppminer | grep NEEDED
```

## `LD_LIBRARY_PATH` (device-specific)

`/vendor/lib64` alone is often **not** enough. MediaTek (e.g. mt6835) needs system + system_ext + chip EGL + VNDK dirs so `libOpenCL.so` can `dlopen` its dependencies.

Working `clinfo` path (mt6835):

```bash
export LD_LIBRARY_PATH=/system/lib64:/system_ext/lib64:/vendor/lib64:/vendor/lib64/egl/mt6835:/vendor/lib64/mt6835:/apex/com.android.vndk.v33/lib64

# optional: append Termux libs last if something still needs them
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PREFIX/lib"

clinfo
./cppminer --backend opencl --list-devices
```

Notes:

- Use absolute paths (`/system_ext/...`, `/apex/...`).
- Replace `mt6835` / `vndk.v33` with your SoC / VNDK version (`ls /vendor/lib64/egl /apex`).
- Only safe for `cppminer` after a `CP_STATIC_LIBCXX` build.

## 0 OpenCL platforms / devices

Diagnose:

```bash
ls -l /vendor/lib64/libOpenCL.so /vendor/lib/libOpenCL.so 2>/dev/null
readelf -d ./cppminer | grep -i opencl
pkg install clinfo
OCL_ICD_DEBUG=1 clinfo   # with the LD_LIBRARY_PATH above
```

### A) Linked to vendor `libOpenCL` (preferred on Termux rebuild)

Use the same `LD_LIBRARY_PATH` that makes `clinfo` work.

### B) Linked to Termux ocl-icd (`$PREFIX/lib/libOpenCL.so`)

ocl-icd only loads ICDs listed under `$PREFIX/etc/OpenCL/vendors/*.icd`. Create one:

```bash
mkdir -p "$PREFIX/etc/OpenCL/vendors"
echo /vendor/lib64/libOpenCL.so > "$PREFIX/etc/OpenCL/vendors/mali.icd"
# (or adreno.icd — name is arbitrary; path must be the real .so)
```

If `OCL_ICD_DEBUG` shows `dlopen` namespace errors, rebuild so CMake links the vendor `libOpenCL` directly, or try clvk.
