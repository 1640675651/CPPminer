#!/usr/bin/env sh
# Linux/macOS counterpart of prepare_onednn_deps.bat: fetch oneDNN, vendor
# nGEN + gemmstone into src/onednn/third_party and apply case5_patches.
#   ./prepare_onednn_deps.sh          vendor once, then only re-apply patches
#   ./prepare_onednn_deps.sh refresh  re-vendor from the oneDNN checkout
# ONEDNN_SRC=<checkout> uses an existing oneDNN tree instead of cloning.
set -eu

cd "$(dirname "$0")"

ONEDNN_TAG="${ONEDNN_TAG:-$(cat case5_patches/ONEDNN_TAG 2>/dev/null || echo v3.13.2)}"
FETCH_DIR="../../third_party/onednn-src"
NGEN_DEST="third_party/ngen"
GEMMSTONE_DEST="third_party/gemmstone"
PATCHES="case5_patches"
TAG_STAMP="../../third_party/.case5_onednn_tag"
REFRESH=0
[ "${1:-}" = "refresh" ] && REFRESH=1

valid_root() {
    [ -f "$1/third_party/ngen/ngen.hpp" ] &&
    [ -f "$1/src/gpu/intel/gemm/jit/include/gemmstone/generator.hpp" ]
}

apply_patches() {
    [ -f "$PATCHES/gemmstone/gemmstone_config.hpp" ] || {
        echo "Missing Case5 patch overlay: $PATCHES/gemmstone/gemmstone_config.hpp" >&2
        exit 1
    }
    echo "Applying Case5 patches from $PATCHES ..."
    cp -R "$PATCHES/ngen/." "$NGEN_DEST/"
    cp -R "$PATCHES/gemmstone/." "$GEMMSTONE_DEST/"
}

if [ "$REFRESH" = 0 ] && [ -f "$NGEN_DEST/ngen.hpp" ] &&
   [ -f "$GEMMSTONE_DEST/include/gemmstone/generator.hpp" ] &&
   [ -f "$TAG_STAMP" ] && [ "$(cat "$TAG_STAMP")" = "$ONEDNN_TAG" ]; then
    echo "Case5 deps already vendored ($ONEDNN_TAG); re-applying patches..."
    apply_patches
    exit 0
fi

if [ -n "${ONEDNN_SRC:-}" ]; then
    ROOT="$ONEDNN_SRC"
else
    ROOT="$FETCH_DIR"
    if ! valid_root "$ROOT"; then
        rm -rf "$ROOT"
        mkdir -p "$(dirname "$ROOT")"
        echo "Cloning oneDNN $ONEDNN_TAG into $ROOT ..."
        git clone --depth 1 --branch "$ONEDNN_TAG" \
            https://github.com/uxlfoundation/oneDNN.git "$ROOT"
    fi
fi
valid_root "$ROOT" || {
    echo "oneDNN checkout at $ROOT is missing gemmstone JIT or third_party/ngen" >&2
    exit 1
}

echo "Vendoring Case5 deps from oneDNN: $ROOT"
mkdir -p third_party "$NGEN_DEST"
cp -R "$ROOT/third_party/ngen/." "$NGEN_DEST/"
rm -rf "$GEMMSTONE_DEST"
mkdir -p "$GEMMSTONE_DEST"
cp -R "$ROOT/src/gpu/intel/gemm/jit/include" "$GEMMSTONE_DEST/include"
cp -R "$ROOT/src/gpu/intel/gemm/jit/generator" "$GEMMSTONE_DEST/generator"
apply_patches
echo "$ONEDNN_TAG" > "$TAG_STAMP"
echo "Case5 deps ready: $NGEN_DEST and $GEMMSTONE_DEST (oneDNN $ONEDNN_TAG)"
