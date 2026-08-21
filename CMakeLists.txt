cmake_minimum_required(VERSION 3.18)

# Backends (enable independently).
option(CP_ENABLE_CPU    "Build CPU Case 3.3 worker" ON)
option(CP_ENABLE_CUDA   "Build CUDA/CUTLASS worker" OFF)
option(CP_ENABLE_OPENCL "Build OpenCL Case 3.3 worker" OFF)
option(CP_ENABLE_CUBLAS "Link cuBLAS for --cublas-period debug path" OFF)
option(CP_CUDA_ARCH     "CUDA arch e.g. 75; empty = native" "")
# Termux + vendor OpenCL: driver needs /vendor libc++, miner needs Termux libc++.
# Static libc++ lets LD_LIBRARY_PATH prefer /vendor without breaking the binary.
option(CP_STATIC_LIBCXX "Statically link libc++ (Termux/vendor OpenCL)" OFF)
if(NOT CP_STATIC_LIBCXX AND DEFINED ENV{PREFIX} AND
   ("$ENV{PREFIX}" MATCHES "/com\\.termux/" OR DEFINED ENV{TERMUX_VERSION}))
    set(CP_STATIC_LIBCXX ON)
    message(STATUS "Termux detected: enabling CP_STATIC_LIBCXX for vendor OpenCL coexistence")
endif()

if(CP_ENABLE_CUBLAS AND NOT CP_ENABLE_CUDA)
    message(FATAL_ERROR "CP_ENABLE_CUBLAS requires CP_ENABLE_CUDA")
endif()

if(NOT CP_ENABLE_CPU AND NOT CP_ENABLE_CUDA AND NOT CP_ENABLE_OPENCL)
    message(FATAL_ERROR "Enable at least one of CP_ENABLE_CPU, CP_ENABLE_CUDA, CP_ENABLE_OPENCL")
endif()

set(_cp_langs CXX C)
if(CP_ENABLE_CUDA)
    list(APPEND _cp_langs CUDA)
endif()
project(cppminer LANGUAGES ${_cp_langs})

set(CMAKE_CXX_STANDARD 17)
if(CP_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

set(CP_COMMON_SOURCES
    src/common/main.cpp
    src/common/cp_util.cpp
    src/common/cp_pool.cpp
    src/common/cp_job_ctrl.cpp
    src/common/cp_mine.cpp
    src/common/cp_fee.cpp
    src/common/cp_share_queue.cpp
    src/common/cp_state.cpp
    src/common/cp_worker.cpp
    src/common/cp_noise.c
)

set(CP_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${CMAKE_SOURCE_DIR}/src/opencl
)

# BLAKE3
set(B3_DIR ${CMAKE_SOURCE_DIR}/third_party/blake3)
if(NOT EXISTS ${B3_DIR}/blake3.c)
    message(FATAL_ERROR "Run build.ps1 once to fetch BLAKE3, or vendor third_party/blake3")
endif()
list(APPEND CP_INCLUDE_DIRS ${B3_DIR})

set(CP_BLAKE3_SOURCES
    ${B3_DIR}/blake3.c
    ${B3_DIR}/blake3_dispatch.c
    ${B3_DIR}/blake3_portable.c
)

# The Rust proof FFI and the miner both use BLAKE3. On AArch64, build each
# NEON implementation and namespace the miner's C symbols to avoid collisions.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    list(APPEND CP_BLAKE3_SOURCES ${B3_DIR}/blake3_neon.c)
    set(CP_BLAKE3_NAMESPACE_DEFINITIONS
        blake3_version=cp_blake3_version
        blake3_hasher_init=cp_blake3_hasher_init
        blake3_hasher_init_keyed=cp_blake3_hasher_init_keyed
        blake3_hasher_init_derive_key=cp_blake3_hasher_init_derive_key
        blake3_hasher_init_derive_key_raw=cp_blake3_hasher_init_derive_key_raw
        blake3_hasher_update=cp_blake3_hasher_update
        blake3_hasher_finalize=cp_blake3_hasher_finalize
        blake3_hasher_finalize_seek=cp_blake3_hasher_finalize_seek
        blake3_hasher_reset=cp_blake3_hasher_reset
        blake3_compress_in_place=cp_blake3_compress_in_place
        blake3_compress_subtree_wide=cp_blake3_compress_subtree_wide
        blake3_compress_xof=cp_blake3_compress_xof
        blake3_xof_many=cp_blake3_xof_many
        blake3_hash_many=cp_blake3_hash_many
        blake3_simd_degree=cp_blake3_simd_degree
        blake3_compress_in_place_portable=cp_blake3_compress_in_place_portable
        blake3_compress_xof_portable=cp_blake3_compress_xof_portable
        blake3_hash_many_portable=cp_blake3_hash_many_portable
        blake3_hash_many_neon=cp_blake3_hash_many_neon)
    set_source_files_properties(
        ${CP_BLAKE3_SOURCES}
        src/common/cp_noise.c
        PROPERTIES COMPILE_DEFINITIONS
            "${CP_BLAKE3_NAMESPACE_DEFINITIONS}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    list(APPEND CP_BLAKE3_SOURCES
        ${B3_DIR}/blake3_sse2.c
        ${B3_DIR}/blake3_sse41.c
        ${B3_DIR}/blake3_avx2.c
        ${B3_DIR}/blake3_avx512.c)
    if(MSVC)
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set_source_files_properties(${B3_DIR}/blake3_sse2.c
                PROPERTIES COMPILE_OPTIONS "/arch:SSE2")
        endif()
        # MSVC has no SSE4.1-specific switch; this is BLAKE3's upstream setting.
        set_source_files_properties(${B3_DIR}/blake3_sse41.c
            PROPERTIES COMPILE_OPTIONS "/arch:AVX")
        set_source_files_properties(${B3_DIR}/blake3_avx2.c
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(${B3_DIR}/blake3_avx512.c
            PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
        set_source_files_properties(${B3_DIR}/blake3_sse2.c
            PROPERTIES COMPILE_OPTIONS "-msse2")
        set_source_files_properties(${B3_DIR}/blake3_sse41.c
            PROPERTIES COMPILE_OPTIONS "-msse4.1")
        set_source_files_properties(${B3_DIR}/blake3_avx2.c
            PROPERTIES COMPILE_OPTIONS "-mavx2")
        set_source_files_properties(${B3_DIR}/blake3_avx512.c
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vl")
    endif()
endif()

set(CP_SOURCES ${CP_COMMON_SOURCES} ${CP_BLAKE3_SOURCES})

if(CP_ENABLE_CPU)
    list(APPEND CP_SOURCES
        src/cpu/cp_cpu_worker.cpp
        src/cpu/cp_cpu_affinity.cpp
        src/cpu/gemm/case33_gemm_xor.cpp
        src/cpu/gemm/case33_cpu_features.cpp
        src/cpu/gemm/case33_gemm_xor_neon.cpp
        src/cpu/gemm/case33_gemm_xor_dotprod.cpp
    )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        list(APPEND CP_SOURCES
            src/cpu/gemm/case33_gemm_xor_avxvnni.cpp
            src/cpu/gemm/case33_gemm_xor_avx2.cpp
            src/cpu/gemm/case33_gemm_xor_ssse3.cpp)
        add_compile_definitions(CASE33_BUILD_X86=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        add_compile_definitions(CASE33_BUILD_AARCH64=1)
        if(MSVC)
            set_source_files_properties(src/cpu/gemm/case33_gemm_xor_dotprod.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:armv8.2")
        else()
            set_source_files_properties(src/cpu/gemm/case33_gemm_xor_dotprod.cpp
                PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
        endif()
    endif()
    add_compile_definitions(CP_ENABLE_CPU=1)
else()
    add_compile_definitions(CP_ENABLE_CPU=0)
endif()

if(CP_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND CP_SOURCES
        src/cuda/cp_gpu.cu
        src/cuda/cp_cutlass_gemm.cu
        src/cuda/cp_cuda_worker.cpp
    )
    list(APPEND CP_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src/cuda/cutlass
        ${CMAKE_SOURCE_DIR}/third_party/cutlass_override
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/examples/35_gemm_softmax
    )
    add_compile_definitions(CP_ENABLE_CUDA=1)
    if(CP_ENABLE_CUBLAS)
        add_compile_definitions(CP_ENABLE_CUBLAS=1)
    else()
        add_compile_definitions(CP_ENABLE_CUBLAS=0)
    endif()
    if(CP_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${CP_CUDA_ARCH})
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
else()
    add_compile_definitions(CP_ENABLE_CUDA=0 CP_ENABLE_CUBLAS=0)
endif()

if(CP_ENABLE_OPENCL)
    # Prefer vendored Khronos headers + Windows x64 import lib (no vendor SDK).
    set(_OCL_HDR "${CMAKE_SOURCE_DIR}/third_party/opencl-headers")
    set(_OCL_LIB_WIN "${CMAKE_SOURCE_DIR}/third_party/opencl/lib/x64/OpenCL.lib")
    if(WIN32 AND EXISTS "${_OCL_LIB_WIN}" AND EXISTS "${_OCL_HDR}/CL/cl.h")
        add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
        set_target_properties(OpenCL::OpenCL PROPERTIES
            IMPORTED_LOCATION "${_OCL_LIB_WIN}"
            INTERFACE_INCLUDE_DIRECTORIES "${_OCL_HDR}")
        set(OpenCL_FOUND TRUE)
        message(STATUS "Using vendored OpenCL: ${_OCL_LIB_WIN}")
    endif()
    if(NOT OpenCL_FOUND)
        find_package(OpenCL QUIET)
    endif()
    if(NOT OpenCL_FOUND AND EXISTS "${_OCL_HDR}/CL/cl.h")
        find_library(OpenCL_LIBRARY NAMES OpenCL OpenCL.so)
        if(OpenCL_LIBRARY)
            add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
            set_target_properties(OpenCL::OpenCL PROPERTIES
                IMPORTED_LOCATION "${OpenCL_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${_OCL_HDR}")
            set(OpenCL_FOUND TRUE)
            message(STATUS "Using system OpenCL + vendored headers: ${OpenCL_LIBRARY}")
        endif()
    endif()
    if(NOT OpenCL_FOUND)
        find_package(OpenCL REQUIRED)
    endif()
    list(APPEND CP_SOURCES
        src/opencl/cp_opencl_worker.cpp
        src/opencl/case33_gemm_ocl.cpp
        src/opencl/case33_ocl_prep.cpp
        src/opencl/cp_ocl_align_test.cpp
        src/opencl/cp_ocl_prep_profile.cpp
        src/opencl/case32_prepack.cpp
        src/opencl/case32_layout.cpp
        src/opencl/opencl_context.cpp
    )
    add_compile_definitions(CP_ENABLE_OPENCL=1)
else()
    add_compile_definitions(CP_ENABLE_OPENCL=0)
endif()

add_executable(cppminer ${CP_SOURCES})
target_include_directories(cppminer PRIVATE ${CP_INCLUDE_DIRS})

if(CP_ENABLE_CUDA)
    target_link_libraries(cppminer PRIVATE CUDA::cudart)
    if(CP_ENABLE_CUBLAS)
        target_link_libraries(cppminer PRIVATE CUDA::cublas)
    endif()
endif()

if(CP_ENABLE_OPENCL)
    target_link_libraries(cppminer PRIVATE OpenCL::OpenCL)
    add_custom_command(TARGET cppminer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cppminer>/kernels"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/opencl/kernels/case33_gemm_xor.cl"
            "$<TARGET_FILE_DIR:cppminer>/kernels/case33_gemm_xor.cl"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/opencl/kernels/cp_ocl_blake3.cl"
            "$<TARGET_FILE_DIR:cppminer>/kernels/cp_ocl_blake3.cl"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/opencl/kernels/cp_ocl_merkle.cl"
            "$<TARGET_FILE_DIR:cppminer>/kernels/cp_ocl_merkle.cl"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/opencl/kernels/cp_ocl_prep.cl"
            "$<TARGET_FILE_DIR:cppminer>/kernels/cp_ocl_prep.cl"
        COMMENT "Copy OpenCL kernels next to cppminer")
endif()

# Optional in-process proof FFI via cargo (CMake builds it; falls back to stub).
set(CP_PROOF_FFI_DIR ${CMAKE_SOURCE_DIR}/rust/cp-proof-ffi)
set(CP_PROOF_FFI_MANIFEST ${CP_PROOF_FFI_DIR}/Cargo.toml)
find_program(CARGO_EXECUTABLE cargo
    HINTS
        "$ENV{HOME}/.cargo/bin"
        "$ENV{USERPROFILE}/.cargo/bin"
        "$ENV{CARGO_HOME}/bin"
        "$ENV{CONDA_PREFIX}/bin"
        "$ENV{CONDA_PREFIX}/Library/bin"
        "$ENV{CONDA_PREFIX}/Scripts"
)

set(_cp_proof_deps_ok TRUE)
foreach(_dep
        ${CMAKE_SOURCE_DIR}/third_party/pearl-blake3/Cargo.toml
        ${CMAKE_SOURCE_DIR}/third_party/zk-pow/Cargo.toml
        ${CMAKE_SOURCE_DIR}/third_party/plonky2/plonky2/Cargo.toml)
    if(NOT EXISTS ${_dep})
        set(_cp_proof_deps_ok FALSE)
    endif()
endforeach()

# Match CMAKE_BUILD_TYPE when set; default Release (multi-config / empty type).
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_cp_cargo_args build)
    set(_cp_cargo_profile debug)
else()
    set(_cp_cargo_args build --release)
    set(_cp_cargo_profile release)
endif()
if(WIN32)
    set(_cp_proof_lib ${CP_PROOF_FFI_DIR}/target/${_cp_cargo_profile}/cp_proof_ffi.lib)
else()
    set(_cp_proof_lib ${CP_PROOF_FFI_DIR}/target/${_cp_cargo_profile}/libcp_proof_ffi.a)
endif()

set(_cp_link_proof_ffi FALSE)
if(CARGO_EXECUTABLE AND _cp_proof_deps_ok AND EXISTS ${CP_PROOF_FFI_MANIFEST})
    file(GLOB _cp_proof_rs CONFIGURE_DEPENDS
        ${CP_PROOF_FFI_DIR}/src/*.rs)
    add_custom_command(
        OUTPUT ${_cp_proof_lib}
        COMMAND ${CMAKE_COMMAND} -E env
            CARGO_TARGET_DIR=${CP_PROOF_FFI_DIR}/target
            ${CARGO_EXECUTABLE} ${_cp_cargo_args}
            --manifest-path ${CP_PROOF_FFI_MANIFEST}
        WORKING_DIRECTORY ${CP_PROOF_FFI_DIR}
        DEPENDS
            ${CP_PROOF_FFI_MANIFEST}
            ${CP_PROOF_FFI_DIR}/Cargo.lock
            ${_cp_proof_rs}
            ${CMAKE_SOURCE_DIR}/third_party/pearl-blake3/Cargo.toml
            ${CMAKE_SOURCE_DIR}/third_party/zk-pow/Cargo.toml
            ${CMAKE_SOURCE_DIR}/third_party/plonky2/plonky2/Cargo.toml
        COMMENT "Building cp-proof-ffi (cargo ${_cp_cargo_profile})"
        VERBATIM
        USES_TERMINAL
    )
    add_custom_target(cp_proof_ffi_build DEPENDS ${_cp_proof_lib})
    add_dependencies(cppminer cp_proof_ffi_build)
    set(_cp_link_proof_ffi TRUE)
    message(STATUS "cp_proof_ffi: cargo will build ${_cp_proof_lib}")
elseif(EXISTS ${_cp_proof_lib})
    # build.ps1 / build.sh may prebuild the crate when cmake cannot find cargo.
    set(_cp_link_proof_ffi TRUE)
    message(STATUS "cp_proof_ffi: linking prebuilt ${_cp_proof_lib}")
else()
    if(NOT CARGO_EXECUTABLE)
        message(WARNING "cargo not found; using proof stub (no share submit proofs)")
    elseif(NOT _cp_proof_deps_ok)
        message(WARNING "pearl-blake3/zk-pow/plonky2 missing; using proof stub")
    else()
        message(WARNING "cp-proof-ffi manifest missing; using proof stub")
    endif()
    target_sources(cppminer PRIVATE src/common/cp_proof_stub.cpp)
endif()

if(_cp_link_proof_ffi)
    # C++ directly references cp_proof_build/cp_proof_verify. Normal archive linking pulls
    # their FFI object without force-loading Rust's bundled BLAKE3 objects, which would clash
    # with the C BLAKE3 implementation used by the miner on ELF linkers.
    target_link_libraries(cppminer PRIVATE ${_cp_proof_lib})
    if(WIN32)
        target_link_libraries(cppminer PRIVATE userenv ws2_32 bcrypt ntdll advapi32)
    elseif(APPLE)
        target_link_libraries(cppminer PRIVATE
            pthread dl m
            "-framework Security"
            "-framework CoreFoundation"
            "-framework SystemConfiguration")
    else()
        target_link_libraries(cppminer PRIVATE pthread dl m)
    endif()
endif()

if(MSVC)
    target_compile_options(cppminer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/openmp>
        $<$<COMPILE_LANGUAGE:C>:/openmp>
    )
    # AVX2 / AVX-VNNI only in dedicated x86 ukernel translation units.
    if(CP_ENABLE_CPU AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        set_source_files_properties(src/cpu/gemm/case33_gemm_xor_ssse3.cpp
            PROPERTIES COMPILE_OPTIONS "-mssse3")
        set_source_files_properties(src/cpu/gemm/case33_gemm_xor_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(src/cpu/gemm/case33_gemm_xor_avxvnni.cpp
            PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2"
                COMPILE_DEFINITIONS "__AVXVNNI__")

    endif()
else()
    # Apple Clang has no built-in OpenMP; use Homebrew libomp when present.
    if(APPLE AND NOT OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
        foreach(_omp_prefix /usr/local/opt/libomp /opt/homebrew/opt/libomp)
            if(EXISTS "${_omp_prefix}/lib/libomp.dylib")
                set(OpenMP_ROOT "${_omp_prefix}")
                break()
            endif()
        endforeach()
        unset(_omp_prefix)
    endif()
    if(APPLE AND OpenMP_ROOT AND NOT OpenMP_CXX_FOUND)
        # AppleClang needs -Xpreprocessor; plain -fopenmp is rejected.
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include" CACHE STRING "" FORCE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include" CACHE STRING "" FORCE)
        set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cppminer PRIVATE OpenMP::OpenMP_CXX)
    elseif(NOT APPLE)
        # GNU/Clang on Linux/etc.: -fopenmp is usually enough.
        target_compile_options(cppminer PRIVATE -fopenmp)
        target_link_options(cppminer PRIVATE -fopenmp)
    else()
        message(FATAL_ERROR
            "OpenMP not found (required for CPU build on macOS).\n"
            "  brew install libomp\n"
            "  Then reconfigure, or pass -DOpenMP_ROOT=$(brew --prefix libomp)")
    endif()
    if(CP_ENABLE_CPU AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        set_source_files_properties(src/cpu/gemm/case33_gemm_xor_ssse3.cpp
            PROPERTIES COMPILE_OPTIONS "-mssse3")
        set_source_files_properties(src/cpu/gemm/case33_gemm_xor_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2")
        set_source_files_properties(src/cpu/gemm/case33_gemm_xor_avxvnni.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
    endif()
endif()

if(WIN32)
    target_link_libraries(cppminer PRIVATE ws2_32)
endif()

if(CP_STATIC_LIBCXX AND NOT MSVC)
    # Embed libc++ so /vendor/lib64 on LD_LIBRARY_PATH (needed by Adreno/Mali
    # OpenCL ICDs) cannot replace Termux's libc++_shared at process start.
    #
    # Termux: -static-libstdc++ still emits -lc++_shared and fails (no static
    # lib in $PREFIX/lib). Use ndk-multilib's libc++_static.a + libc++abi.a.
    set(_cp_libcxx_hints)
    if(DEFINED ENV{PREFIX})
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            list(APPEND _cp_libcxx_hints "$ENV{PREFIX}/aarch64-linux-android/lib")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv7l|arm)$")
            list(APPEND _cp_libcxx_hints "$ENV{PREFIX}/arm-linux-androideabi/lib")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
            list(APPEND _cp_libcxx_hints "$ENV{PREFIX}/x86_64-linux-android/lib")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)$")
            list(APPEND _cp_libcxx_hints "$ENV{PREFIX}/i686-linux-android/lib")
        endif()
        list(APPEND _cp_libcxx_hints "$ENV{PREFIX}/lib")
    endif()
    find_library(CP_LIBCXX_STATIC NAMES c++_static libc++_static
        HINTS ${_cp_libcxx_hints} NO_DEFAULT_PATH)
    find_library(CP_LIBCXXABI_STATIC NAMES c++abi libc++abi
        HINTS ${_cp_libcxx_hints} NO_DEFAULT_PATH)
    if(NOT CP_LIBCXX_STATIC)
        find_library(CP_LIBCXX_STATIC NAMES c++_static libc++_static
            HINTS ${_cp_libcxx_hints})
    endif()
    if(NOT CP_LIBCXXABI_STATIC)
        find_library(CP_LIBCXXABI_STATIC NAMES c++abi libc++abi
            HINTS ${_cp_libcxx_hints})
    endif()
    if(CP_LIBCXX_STATIC AND CP_LIBCXXABI_STATIC)
        target_link_options(cppminer PRIVATE -nostdlib++)
        target_link_libraries(cppminer PRIVATE
            ${CP_LIBCXX_STATIC} ${CP_LIBCXXABI_STATIC})
        find_library(CP_LIBUNWIND_STATIC NAMES unwind
            HINTS ${_cp_libcxx_hints})
        if(CP_LIBUNWIND_STATIC)
            target_link_libraries(cppminer PRIVATE ${CP_LIBUNWIND_STATIC})
        endif()
        message(STATUS "Linking libc++ statically: ${CP_LIBCXX_STATIC}")
    elseif(DEFINED ENV{TERMUX_VERSION} OR
           (DEFINED ENV{PREFIX} AND "$ENV{PREFIX}" MATCHES "/com\\.termux/"))
        message(FATAL_ERROR
            "CP_STATIC_LIBCXX requires static libc++ on Termux.\n"
            "  pkg install ndk-multilib\n"
            "  Then reconfigure, or pass -DCP_STATIC_LIBCXX=OFF")
    else()
        # Desktop clang/gcc often supports this; Termux does not without ndk-multilib.
        target_link_options(cppminer PRIVATE -static-libstdc++)
        message(STATUS "Linking libc++ via -static-libstdc++ (no libc++_static.a found)")
    endif()
    unset(_cp_libcxx_hints)
endif()

message(STATUS "CPPminer backends: CPU=${CP_ENABLE_CPU} CUDA=${CP_ENABLE_CUDA} OpenCL=${CP_ENABLE_OPENCL} CUBLAS=${CP_ENABLE_CUBLAS}")
