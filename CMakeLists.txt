cmake_minimum_required(VERSION 3.18)
project(cpminer LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(CP_CUDA_ARCH "CUDA arch e.g. 75; empty = native" "")

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# BLAKE3
set(B3_DIR ${CMAKE_SOURCE_DIR}/third_party/blake3)
if(NOT EXISTS ${B3_DIR}/blake3.c)
    message(FATAL_ERROR "Run build.ps1 once to fetch BLAKE3, or vendor third_party/blake3")
endif()
include_directories(${B3_DIR})

add_compile_definitions(
    BLAKE3_NO_AVX512 BLAKE3_NO_AVX2 BLAKE3_NO_SSE41 BLAKE3_NO_SSE2
)

find_package(CUDAToolkit REQUIRED)

set(CP_SOURCES
    src/main.cpp
    src/cp_util.cpp
    src/cp_pool.cpp
    src/cp_job_ctrl.cpp
    src/cp_mine.cpp
    src/cp_state.cpp
    src/cp_noise.c
    ${B3_DIR}/blake3.c
    ${B3_DIR}/blake3_dispatch.c
    ${B3_DIR}/blake3_portable.c
)

set(CP_CUDA src/cp_gpu.cu)

if(CP_CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${CP_CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(cpminer ${CP_SOURCES} ${CP_CUDA})
target_link_libraries(cpminer CUDA::cudart)

if(MSVC)
    target_compile_options(cpminer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
    target_compile_options(cpminer PRIVATE $<$<COMPILE_LANGUAGE:C>:/openmp>)
else()
    target_compile_options(cpminer PRIVATE -fopenmp)
endif()

if(WIN32)
    target_link_libraries(cpminer ws2_32)
endif()
