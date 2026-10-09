# Hexagon SDK project for the CPPminer Hexagon backend (built with build.ps1, not by the
# miner's CMake): the pearlx skel library loaded into the cDSP, plus an Android
# self-test that checks it against a CPU reference.
cmake_minimum_required(VERSION 3.14.3)

project(pearlx C CXX ASM)
enable_language(ASM)

cmake_policy(SET CMP0115 OLD)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
)

set(NO_QURT_INC 1)

if(${OS_TYPE} MATCHES "HLOS")

    # Android: FastRPC stub library + self-test executable
    add_library(pearlx SHARED ${CMAKE_CURRENT_BINARY_DIR}/pearlx_stub)
    add_executable(pearlx_test ${CMAKE_CURRENT_SOURCE_DIR}/src/pearlx_test)

    link_options(pearlx_test)
    include_directories(${common_incs})
    build_idl(inc/pearlx.idl pearlx)

    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(pearlx ${dsprpc})
    link_custom_library(pearlx_test ${dsprpc})
    add_dependencies(pearlx_test pearlx)
    target_link_libraries(pearlx_test pearlx)

    copy_binaries(pearlx_test pearlx)

else()

    # Hexagon: skel library loaded into the DSP process
    add_library(pearlx_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/pearlx_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pearlx_imp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pearlx.S
    )

    include_directories(${common_incs})
    build_idl(inc/pearlx.idl pearlx_skel)
    link_custom_library(pearlx_skel worker_pool)

    copy_binaries(pearlx_skel)

endif()
