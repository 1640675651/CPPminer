cmake_minimum_required(VERSION 3.18)

# Backends (enable independently).
option(CP_ENABLE_CPU    "Build CPU Case 3.3 worker" ON)
option(CP_ENABLE_CUDA   "Build CUDA/CUTLASS worker" OFF)
option(CP_ENABLE_OPENCL "Build OpenCL Case 3.3 worker" OFF)
option(CP_CUDA_ARCH     "CUDA arch e.g. 75; empty = native" "")

if(NOT CP_ENABLE_CPU AND NOT CP_ENABLE_CUDA AND NOT CP_ENABLE_OPENCL)
    message(FATAL_ERROR "Enable at least one of CP_ENABLE_CPU, CP_ENABLE_CUDA, CP_ENABLE_OPENCL")
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
    ${CMAKE_SOURCE_DIR}/src/opencl
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

if(CP_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(NOT OpenCL_FOUND)
        set(_ONEAPI_CANDIDATES)
        if(DEFINED ENV{ONEAPI_ROOT})
            list(APPEND _ONEAPI_CANDIDATES "$ENV{ONEAPI_ROOT}/compiler/latest"
                 "$ENV{ONEAPI_ROOT}")
        endif()
        foreach(_ver IN ITEMS 2026.1 2025.3 2023.2.0)
            list(APPEND _ONEAPI_CANDIDATES
                 "C:/Program Files (x86)/Intel/oneAPI/compiler/${_ver}")
        endforeach()
        foreach(_root IN LISTS _ONEAPI_CANDIDATES)
            if(NOT OpenCL_FOUND AND EXISTS "${_root}")
                find_path(OpenCL_INCLUDE_DIR CL/cl.h
                    HINTS "${_root}/windows/include/sycl"
                          "${_root}/include/sycl"
                          "${_root}/windows/lib/oclfpga/host/include"
                    NO_DEFAULT_PATH)
                find_library(OpenCL_LIBRARY NAMES OpenCL
                    HINTS "${_root}/windows/lib" "${_root}/lib"
                    NO_DEFAULT_PATH)
                if(OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY)
                    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                    set_target_properties(OpenCL::OpenCL PROPERTIES
                        IMPORTED_LOCATION "${OpenCL_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIR}")
                    set(OpenCL_FOUND TRUE)
                    message(STATUS "Using OpenCL from oneAPI (${_root})")
                endif()
            endif()
        endforeach()
    endif()
    if(NOT OpenCL_FOUND)
        set(_OCL_HDR "${CMAKE_SOURCE_DIR}/third_party/opencl-headers")
        if(EXISTS "${_OCL_HDR}/CL/cl.h")
            find_library(OpenCL_LIBRARY NAMES OpenCL
                HINTS
                    "C:/Program Files (x86)/Intel/oneAPI/compiler/2023.2.0/windows/lib"
                    "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3/windows/lib"
                    "C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1/windows/lib")
            if(OpenCL_LIBRARY)
                add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                set_target_properties(OpenCL::OpenCL PROPERTIES
                    IMPORTED_LOCATION "${OpenCL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${_OCL_HDR}")
                set(OpenCL_FOUND TRUE)
                message(STATUS "Using vendored OpenCL headers + ${OpenCL_LIBRARY}")
            endif()
        endif()
    endif()
    if(NOT OpenCL_FOUND)
        find_package(OpenCL REQUIRED)
    endif()
    list(APPEND CP_SOURCES
        src/opencl/cp_opencl_worker.cpp
        src/opencl/case33_gemm_ocl.cpp
        src/opencl/case32_prepack.cpp
        src/opencl/opencl_context.cpp
    )
    add_compile_definitions(CP_ENABLE_OPENCL=1)
else()
    add_compile_definitions(CP_ENABLE_OPENCL=0)
endif()

add_executable(cpminer ${CP_SOURCES})
target_include_directories(cpminer PRIVATE ${CP_INCLUDE_DIRS})

if(CP_ENABLE_CUDA)
    target_link_libraries(cpminer PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(CP_ENABLE_OPENCL)
    target_link_libraries(cpminer PRIVATE OpenCL::OpenCL)
    add_custom_command(TARGET cpminer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cpminer>/kernels"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/opencl/kernels/case33_gemm_xor.cl"
            "$<TARGET_FILE_DIR:cpminer>/kernels/case33_gemm_xor.cl"
        COMMENT "Copy OpenCL kernels next to cpminer")
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
