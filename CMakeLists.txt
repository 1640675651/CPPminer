cmake_minimum_required(VERSION 3.18)

# Backends (enable independently). OpenCL stub reserved for a later worker.
option(CP_ENABLE_CPU    "Build CPU Case 3.3 worker" ON)
option(CP_ENABLE_CUDA   "Build CUDA/CUTLASS worker" OFF)
option(CP_ENABLE_OPENCL "Build OpenCL worker (not yet)" OFF)
option(CP_CUDA_ARCH     "CUDA arch e.g. 75; empty = native" "")

if(NOT CP_ENABLE_CPU AND NOT CP_ENABLE_CUDA AND NOT CP_ENABLE_OPENCL)
    message(FATAL_ERROR "Enable at least one of CP_ENABLE_CPU, CP_ENABLE_CUDA, CP_ENABLE_OPENCL")
endif()
if(CP_ENABLE_OPENCL)
    message(FATAL_ERROR "OpenCL worker is not implemented yet (CP_ENABLE_OPENCL=ON)")
endif()

set(_cp_langs CXX C)
if(CP_ENABLE_CUDA)
    list(APPEND _cp_langs CUDA)
endif()
project(cpminer LANGUAGES ${_cp_langs})

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
    src/common/cp_state.cpp
    src/common/cp_worker.cpp
    src/common/cp_noise.c
)

set(CP_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/cuda
)

# BLAKE3
set(B3_DIR ${CMAKE_SOURCE_DIR}/third_party/blake3)
if(NOT EXISTS ${B3_DIR}/blake3.c)
    message(FATAL_ERROR "Run build.ps1 once to fetch BLAKE3, or vendor third_party/blake3")
endif()
list(APPEND CP_INCLUDE_DIRS ${B3_DIR})

add_compile_definitions(
    BLAKE3_NO_AVX512 BLAKE3_NO_AVX2 BLAKE3_NO_SSE41 BLAKE3_NO_SSE2
)

set(CP_SOURCES ${CP_COMMON_SOURCES}
    ${B3_DIR}/blake3.c
    ${B3_DIR}/blake3_dispatch.c
    ${B3_DIR}/blake3_portable.c
)

if(CP_ENABLE_CPU)
    list(APPEND CP_SOURCES
        src/cpu/cp_cpu_worker.cpp
        src/cpu/gemm/case33_gemm_xor.cpp
    )
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
    if(CP_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${CP_CUDA_ARCH})
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
else()
    add_compile_definitions(CP_ENABLE_CUDA=0)
endif()

add_compile_definitions(CP_ENABLE_OPENCL=0)

add_executable(cpminer ${CP_SOURCES})
target_include_directories(cpminer PRIVATE ${CP_INCLUDE_DIRS})

if(CP_ENABLE_CUDA)
    target_link_libraries(cpminer PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Optional in-process proof FFI (built by build.ps1 / cargo).
set(CP_PROOF_FFI_DIR ${CMAKE_SOURCE_DIR}/rust/cp-proof-ffi/target/release)
if(WIN32)
    set(_cp_proof_lib ${CP_PROOF_FFI_DIR}/cp_proof_ffi.lib)
else()
    set(_cp_proof_lib ${CP_PROOF_FFI_DIR}/libcp_proof_ffi.a)
endif()
if(EXISTS ${_cp_proof_lib})
    target_link_libraries(cpminer PRIVATE ${_cp_proof_lib})
    if(WIN32)
        target_link_libraries(cpminer PRIVATE userenv ws2_32 bcrypt ntdll advapi32)
    else()
        target_link_libraries(cpminer PRIVATE pthread dl m)
    endif()
    message(STATUS "Linking cp_proof_ffi from ${_cp_proof_lib}")
else()
    message(WARNING "cp_proof_ffi not found; using stub (proof build/verify unavailable)")
    target_sources(cpminer PRIVATE src/common/cp_proof_stub.cpp)
endif()

if(MSVC)
    target_compile_options(cpminer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/openmp>
        $<$<COMPILE_LANGUAGE:C>:/openmp>
    )
    if(CP_ENABLE_CPU)
        target_compile_options(cpminer PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        )
    endif()
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cpminer PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_compile_options(cpminer PRIVATE -fopenmp)
        target_link_options(cpminer PRIVATE -fopenmp)
    endif()
    if(CP_ENABLE_CPU)
        target_compile_options(cpminer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
    endif()
endif()

if(WIN32)
    target_link_libraries(cpminer PRIVATE ws2_32)
endif()

message(STATUS "CPminer backends: CPU=${CP_ENABLE_CPU} CUDA=${CP_ENABLE_CUDA} OpenCL=${CP_ENABLE_OPENCL}")
