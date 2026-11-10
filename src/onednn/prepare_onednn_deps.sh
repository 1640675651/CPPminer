#!/usr/bin/env sh
# Linux/macOS counterpart of prepare_onednn_deps.bat: vendor nGEN and gemmstone (with the Case 5
# modifications) from the case5 fork of oneDNN into src/onednn/third_party:
#   third_party/ngen       <- third_party/ngen
#   third_party/gemmstone  <- src/gpu/intel/gemm/jit/{include,generator,selector,gemmstone_config.hpp}
# Source: ONEDNN_SRC (a local checkout) or a shallow clone of ONEDNN_REPO at ONEDNN_REF in the repo
# root's third_party/onednn-src. The copy runs on every call (rsync when present: unchanged files keep
# their timestamps, so they do not rebuild).
#   ./prepare_onednn_deps.sh          use third_party/onednn-src as it is (clone if missing / other ref)
#   ./prepare_onednn_deps.sh refresh  re-fetch ONEDNN_REF
set -eu

cd "$(dirname "$0")"

ONEDNN_REPO="${ONEDNN_REPO:-https://github.com/1640675651/oneDNN.git}"
ONEDNN_REF="${ONEDNN_REF:-v3.13.2-prl-patch}"
FETCH_DIR="../../third_party/onednn-src"
FETCH_STAMP="../../third_party/.case5_onednn_ref"
NGEN_DEST="third_party/ngen"
GEMMSTONE_DEST="third_party/gemmstone"
REFRESH=0
[ "${1:-}" = "refresh" ] && REFRESH=1

valid_root() {
    [ -f "$1/third_party/ngen/ngen.hpp" ] &&
    [ -f "$1/src/gpu/intel/gemm/jit/include/gemmstone/generator.hpp" ] &&
    [ -f "$1/src/gpu/intel/gemm/jit/gemmstone_config.hpp" ] &&
    [ -f "$1/src/gpu/intel/gemm/jit/generator/pieces/case5_tile_xor.cxx" ]
}

mirror() {   # mirror <src dir> <dst dir>
    mkdir -p "$2"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete "$1/" "$2/"
    else
        rm -rf "$2"
        mkdir -p "$2"
        cp -R "$1/." "$2/"
    fi
}

if [ -n "${ONEDNN_SRC:-}" ]; then
    ROOT="$ONEDNN_SRC"
else
    ROOT="$FETCH_DIR"
    WANT="$ONEDNN_REPO $ONEDNN_REF"
    HAVE="$(tr -d '\r' < "$FETCH_STAMP" 2>/dev/null || true)"   # the .bat writes CRLF
    if [ "$REFRESH" = 1 ] || [ "$HAVE" != "$WANT" ] || ! valid_root "$ROOT"; then
        rm -rf "$ROOT" "$FETCH_STAMP"
        mkdir -p "$(dirname "$ROOT")"
        echo "Cloning $ONEDNN_REPO ($ONEDNN_REF) into $ROOT ..."
        git clone --depth 1 --branch "$ONEDNN_REF" "$ONEDNN_REPO" "$ROOT"
        valid_root "$ROOT" && echo "$WANT" > "$FETCH_STAMP"
    fi
fi
valid_root "$ROOT" || {
    echo "oneDNN tree at $ROOT is not the case5 fork (missing nGEN, gemmstone or the Case 5 sources)" >&2
    echo "  ONEDNN_REPO=$ONEDNN_REPO ONEDNN_REF=$ONEDNN_REF" >&2
    exit 1
}

JIT="$ROOT/src/gpu/intel/gemm/jit"
mkdir -p third_party
mirror "$ROOT/third_party/ngen" "$NGEN_DEST"
mirror "$JIT/include" "$GEMMSTONE_DEST/include"
mirror "$JIT/generator" "$GEMMSTONE_DEST/generator"
mirror "$JIT/selector" "$GEMMSTONE_DEST/selector"
cp -p "$JIT/gemmstone_config.hpp" "$GEMMSTONE_DEST/gemmstone_config.hpp"
echo "Case5 deps ready: $NGEN_DEST and $GEMMSTONE_DEST from $ROOT @ $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
