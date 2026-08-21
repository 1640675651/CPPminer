#!/usr/bin/env bash
# Build cppminer on Linux/macOS/WSL.
# Default: CPU backend only (no CUDA Toolkit required).
#
# Usage:
#   ./build.sh
#   ./build.sh --backend cpu
#   ./build.sh --backend cuda --cuda-arch 61
#   ./build.sh --backend cpu,opencl
#   ./build.sh --backend cpu,cuda,opencl --cuda-arch 75
#   ./build.sh --backend cuda --enable-cublas
#
# Dependencies downloaded:
#   BLAKE3 1.8.5          (always, for CPU hashing)
#   OpenCL-Headers          (only if OpenCL backend is enabled)
#   CUTLASS 2.11.0          (only if CUDA backend is enabled)
#   cp-proof-ffi (Rust)     (built by CMake via cargo, for proof build/verify)
#
# Install cmake + a C++ compiler first:
#   Debian/Ubuntu:  sudo apt install cmake g++ gcc
#   RHEL/Fedora:    sudo dnf install cmake gcc-c++ gcc
#   macOS:          brew install cmake libomp
# Proof FFI also needs Rust (rustup / cargo) and vendored crates under third_party/.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKENDS=("cpu")
CUDA_ARCH=""
ENABLE_CUBLAS=0
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
B3_DIR="${BUILD_DIR}/b3"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            IFS=',' read -ra BACKENDS <<< "$2"
            BACKENDS=("${BACKENDS[@]// /}")          # strip spaces
            shift 2
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --enable-cublas)
            ENABLE_CUBLAS=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend CPU[,CUDA[,OpenCl]]  Backends to build (default: cpu)"
            echo "  --cuda-arch ARCH               CUDA compute arch e.g. 75, 86 (default: auto)"
            echo "  --enable-cublas                Link cuBLAS (requires CUDA)"
            echo "  --help, -h                     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2; exit 1
            ;;
    esac
done

# ── Validate backends ─────────────────────────────────────────────────────────
ENABLE_CPU=0
ENABLE_CUDA=0
ENABLE_OPENCL=0
for b in "${BACKENDS[@]}"; do
    case "$b" in
        cpu)    ENABLE_CPU=1 ;;
        cuda)   ENABLE_CUDA=1 ;;
        opencl) ENABLE_OPENCL=1 ;;
        *)
            echo "Unknown backend: $b (valid: cpu, cuda, opencl)" >&2
            exit 1
            ;;
    esac
done

if (( ENABLE_CUDA && ENABLE_CUBLAS == 0 )); then
    :  # fine, cuBLAS optional
fi
if (( ENABLE_CUBLAS && ENABLE_CUDA == 0 )); then
    echo "--enable-cublas requires --backend cuda" >&2; exit 1
fi
if (( ENABLE_CPU == 0 && ENABLE_CUDA == 0 && ENABLE_OPENCL == 0 )); then
    echo "Enable at least one backend: --backend cpu,cuda,OpenCl" >&2; exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
log()    { printf "=== %s ===\n" "$*"; }
fail()   { echo "ERROR: $*" >&2; exit 1; }

# ── Ensure-Blake3 ─────────────────────────────────────────────────────────────
ensure_blake3() {
    local src_dir="${PROJECT_ROOT}/third_party/blake3"
    if [[ -f "${src_dir}/blake3.c" && -f "${src_dir}/blake3_neon.c" &&
          $(sed -n 's/^#define BLAKE3_VERSION_STRING "\(.*\)"/\1/p' "${src_dir}/blake3.h") == "1.8.5" ]]; then
        log "BLAKE3 already present at ${src_dir}"
        return
    fi

    log "Fetching/upgrading BLAKE3 1.8.5"
    mkdir -p "${src_dir}"

    local base="https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/1.8.5/c"
    local files=(
        blake3.c blake3.h blake3_dispatch.c blake3_portable.c blake3_impl.h
        blake3_sse2.c blake3_sse41.c blake3_avx2.c blake3_avx512.c blake3_neon.c
    )

    local url
    for f in "${files[@]}"; do
        url="${base}/${f}"
        if command -v curl &>/dev/null; then
            curl -fSL --retry 3 --retry-delay 2 -o "${src_dir}/${f}" "$url" \
                || fail "Failed to download ${url}"
        elif command -v wget &>/dev/null; then
            wget -q --retry-connrefused --tries=3 -O "${src_dir}/${f}" "$url" \
                || fail "Failed to download ${url}"
        else
            fail "Neither curl nor wget found — cannot download BLAKE3"
        fi
    done

    # Copy to build staging
    mkdir -p "${B3_DIR}"
    cp "${src_dir}"/* "${B3_DIR}/"
}

# ── Ensure-OpenCL-Headers ─────────────────────────────────────────────────────
ensure_opencl_headers() {
    local hdr_dir="${PROJECT_ROOT}/third_party/opencl-headers"
    if [[ -f "${hdr_dir}/CL/cl.h" ]]; then
        log "OpenCL-Headers already present at ${hdr_dir}"
        return
    fi

    log "Fetching Khronos OpenCL-Headers"
    if ! command -v git &>/dev/null; then
        fail "git required to fetch OpenCL headers"
    fi

    rm -rf "${hdr_dir}"
    git clone --depth 1 --branch v2024.10.24 \
        https://github.com/KhronosGroup/OpenCL-Headers.git "${hdr_dir}" \
        || fail "OpenCL-Headers clone failed"

    [[ -f "${hdr_dir}/CL/cl.h" ]] || fail "OpenCL headers missing after clone"
}

# ── Ensure-CUTLASS ────────────────────────────────────────────────────────────
ensure_cutlass() {
    local cutlass_dir="${PROJECT_ROOT}/third_party/cutlass"
    if [[ -f "${cutlass_dir}/include/cutlass/cutlass.h" ]]; then
        log "CUTLASS already present at ${cutlass_dir}"
        return
    fi

    log "Fetching CUTLASS v2.11.0"
    local zip_path="${BUILD_DIR}/cutlass-v2.11.0.zip"

    mkdir -p "${BUILD_DIR}"

    if command -v curl &>/dev/null; then
        curl -fSL --retry 3 --retry-delay 2 \
            -o "${zip_path}" \
            "https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.11.0.zip" \
            || fail "Failed to download CUTLASS"
    elif command -v wget &>/dev/null; then
        wget -q --retry-connrefused --tries=3 \
            -O "${zip_path}" \
            "https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.11.0.zip" \
            || fail "Failed to download CUTLASS"
    else
        fail "Neither curl nor wget found — cannot download CUTLASS"
    fi

    mkdir -p "${PROJECT_ROOT}/third_party"
    unzip -qo "${zip_path}" -d "${PROJECT_ROOT}/third_party"

    local extracted="${PROJECT_ROOT}/third_party/cutlass-2.11.0"
    [[ -d "${cutlass_dir}" ]] && rm -rf "${cutlass_dir}"
    mv "${extracted}" "${cutlass_dir}"
    rm -f "${zip_path}"

    [[ -f "${cutlass_dir}/include/cutlass/cutlass.h" ]] \
        || fail "CUTLASS fetch failed: missing include/cutlass/cutlass.h"
}

# ── Auto-detect CUDA root ────────────────────────────────────────────────────
find_cuda_root() {
    local candidates=(
        "/usr/local/cuda"
        "/usr/local/cuda-"*
        "${CUDA_HOME}"
    )
    local c
    for c in "${candidates[@]}"; do
        [[ -z "$c" || ! -d "$c" ]] && continue
        if [[ -x "${c}/bin/nvcc" ]]; then
            echo "$c"
            return 0
        fi
    done
    # Fallback: find latest CUDA directory
    if ls /usr/local/cuda-* &>/dev/null; then
        local latest
        latest=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -r | head -n1)
        if [[ -x "${latest}/bin/nvcc" ]]; then
            echo "$latest"; return 0
        fi
    fi
    return 1
}

# ── Auto-detect CUDA arch ────────────────────────────────────────────────────
detect_cuda_arch() {
    if [[ -n "$CUDA_ARCH" ]]; then
        echo "$CUDA_ARCH"; return
    fi
    if command -v nvidia-smi &>/dev/null; then
        local cap
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
        if [[ "$cap" =~ ([0-9]+)\.([0-9]+) ]]; then
            echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"; return
        fi
    fi
    echo "75"  # conservative default
}

# ── Find cmake ────────────────────────────────────────────────────────────────
find_cmake() {
    if command -v cmake &>/dev/null; then
        which cmake
        return 0
    fi
    return 1
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "Backends: CPU=${ENABLE_CPU} CUDA=${ENABLE_CUDA} OpenCL=${ENABLE_OPENCL} CUBLAS=${ENABLE_CUBLAS}"

ensure_blake3

if (( ENABLE_CUDA )); then
    CUDA_ROOT=$(find_cuda_root) || fail "CUDA Toolkit not found (needed for --backend cuda). Set --cuda-arch or install CUDA."
    log "CUDA: ${CUDA_ROOT}"
    CUDA_ARCH=$(detect_cuda_arch)
    log "CUDA arch: ${CUDA_ARCH}"
    ensure_cutlass
fi

if (( ENABLE_OPENCL )); then
    ensure_opencl_headers
fi

# ── CMake build (if cmake available) ─────────────────────────────────────────
CMAKE_EXE=""
if CMAKE_EXE=$(find_cmake); then
    log "cmake: ${CMAKE_EXE}"
else
    log "cmake not found; skipping CMake build (compile manually)"
fi

if [[ -n "$CMAKE_EXE" ]]; then
    CMAKE_BUILD_DIR="${BUILD_DIR}/cmake"
    mkdir -p "${CMAKE_BUILD_DIR}"

    cmake_args=(
        -S "${PROJECT_ROOT}"
        -B "${CMAKE_BUILD_DIR}"
        -DCMAKE_BUILD_TYPE=Release
        -DCP_ENABLE_CPU=$(( ENABLE_CPU ? 1 : 0 ))
        -DCP_ENABLE_CUDA=$(( ENABLE_CUDA ? 1 : 0 ))
        -DCP_ENABLE_OPENCL=$(( ENABLE_OPENCL ? 1 : 0 ))
        -DCP_ENABLE_CUBLAS=$(( ENABLE_CUBLAS ? 1 : 0 ))
    )

    if (( ENABLE_CUDA )); then
        cmake_args+=(-DCP_CUDA_ARCH="${CUDA_ARCH}")
    fi

    log "CMake configure"
    cmake "${cmake_args[@]}" \
        || fail "CMake configure failed"

    log "CMake build"
    cmake --build "${CMAKE_BUILD_DIR}" --config Release \
        || fail "CMake build failed"

    # Locate the built binary
    exe=$(find "${CMAKE_BUILD_DIR}" -maxdepth 2 -name cppminer -o -name cppminer.exe 2>/dev/null | head -n1)
    if [[ -z "$exe" ]]; then
        fail "cppminer not found after build"
    fi
    cp -f "$exe" "${PROJECT_ROOT}/cppminer"
    log "Done: ${PROJECT_ROOT}/cppminer"
else
    log "Skipping CMake build (cmake not available)"
fi

# ── Help summary ──────────────────────────────────────────────────────────────
log "Dependencies installed:"
log "  BLAKE3:      ${B3_DIR}"
(( ENABLE_OPENCL )) && log "  OpenCL-Headers: ${PROJECT_ROOT}/third_party/opencl-headers"
(( ENABLE_CUDA )) && log "  CUTLASS:     ${PROJECT_ROOT}/third_party/cutlass"
log "  cp-proof-ffi: built by CMake (cargo) when available"

if [[ -f "${PROJECT_ROOT}/cppminer" ]]; then
    "${PROJECT_ROOT}/cppminer" --help 2>&1 | head -n 15 || true
fi
