#!/usr/bin/env sh
# Copy the oneAPI SYCL runtime that kernels/libcp_esimd.so needs next to it, so
# a release runs the ESIMD scan without oneAPI installed (libcp_esimd's RPATH
# starts with $ORIGIN). Only the Intel GPU compute runtime (OpenCL ICD) is then
# required on the target machine.
#   scripts/package_esimd_runtime.sh <kernels-dir> [oneapi-root]
set -eu

KDIR=${1:?usage: package_esimd_runtime.sh <kernels-dir> [oneapi-root]}
ONEAPI=${2:-${ONEAPI_ROOT:-/opt/intel/oneapi}}
LIB="$KDIR/libcp_esimd.so"
[ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 1; }

# libsycl loads the OpenCL Unified Runtime adapter with dlopen, so ldd of the
# library alone does not list it.
ADAPTER=$(ls "$ONEAPI"/compiler/latest/lib/libur_adapter_opencl.so.0 2>/dev/null || true)
[ -n "$ADAPTER" ] || { echo "UR OpenCL adapter not found under $ONEAPI" >&2; exit 1; }

for obj in "$LIB" "$ADAPTER"; do
    # Resolve with the oneAPI tree on the path: libumf/libhwloc/libsvml etc.
    LD_LIBRARY_PATH="$ONEAPI/compiler/latest/lib:$ONEAPI/umf/latest/lib:$ONEAPI/tcm/latest/lib" \
        ldd "$obj" | awk '/=> \// {print $3}' | while read -r dep; do
        name=$(basename "$dep")
        case "$dep" in
            */libOpenCL.so*) ;; # the miner ships/uses the system ICD loader
            "$ONEAPI"/*|/opt/intel/oneapi/*)
                [ -f "$KDIR/$name" ] || cp -L "$dep" "$KDIR/$name"
                echo "$name"
                ;;
            *)
                # Already bundled by an earlier run: $ORIGIN resolves it here.
                [ "$(cd "$(dirname "$dep")" && pwd)" = "$(cd "$KDIR" && pwd)" ] && echo "$name"
                ;;
        esac
    done
done | sort -u > "$KDIR/.esimd_runtime_files"
cp -L "$ADAPTER" "$KDIR/libur_adapter_opencl.so.0"
echo libur_adapter_opencl.so.0 >> "$KDIR/.esimd_runtime_files"

# Intel's license terms for the redistributed runtime files.
mkdir -p "$KDIR/oneapi-licensing"
for f in "$ONEAPI"/licensing/latest/licensing/*/license.htm; do
    [ -f "$f" ] && cp -L "$f" "$KDIR/oneapi-licensing/license.htm"
done
for f in "$ONEAPI"/compiler/latest/share/doc/compiler/licensing/c/LICENSE \
         "$ONEAPI"/compiler/latest/share/doc/compiler/licensing/c/third-party-programs.txt; do
    [ -f "$f" ] && cp -L "$f" "$KDIR/oneapi-licensing/compiler-$(basename "$f")"
done
echo "bundled $(wc -l < "$KDIR/.esimd_runtime_files") runtime libraries into $KDIR:"
cat "$KDIR/.esimd_runtime_files"
du -sh "$KDIR"
