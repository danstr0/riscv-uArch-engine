cmake_minimum_required(VERSION 3.20)

project(rvsim
    VERSION     0.5.0
    LANGUAGES   CXX
    DESCRIPTION "RV32I CPU architecture engine"
)

# ── Global settings ─────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strict warnings.
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wconversion -Wsign-conversion
    -Wshadow -Wnon-virtual-dtor
    -Wold-style-cast -Wcast-align
    -Woverloaded-virtual
    -Wnull-dereference
    -Wformat=2
)

# ── Detect std::format support ──────────────────────────────────────────
# GCC < 13 and Clang < 17 lack <format> - pull in {fmt} and provide a shim 
# header so source code can use #include <format> and std::format() 
# unconditionally.

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
    #include <format>
    int main() { auto s = std::format(\"{}\", 42); return 0; }
" HAS_STD_FORMAT)

if(NOT HAS_STD_FORMAT)
    message(STATUS "std::format not available — fetching {fmt} library")
    include(FetchContent)
    FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        11.1.4
    )
    FetchContent_MakeAvailable(fmt)

    # Shim header: #include <format> → fmt/format.h
    file(WRITE "${CMAKE_BINARY_DIR}/include/format"
"#pragma once
// Auto-generated shim: maps std::format to fmt::format.
#include <fmt/format.h>
namespace std {
    using fmt::format;
}
")
    set(FORMAT_SHIM_DIR "${CMAKE_BINARY_DIR}/include")
    set(FORMAT_LIB fmt::fmt)
else()
    message(STATUS "std::format available — using standard library")
    set(FORMAT_SHIM_DIR "")
    set(FORMAT_LIB "")
endif()

# ── Core library ────────────────────────────────────────────────────────

add_library(rvsim_core STATIC
    src/core/decoder.cpp
    src/core/memory.cpp
    src/core/sparse_memory.cpp
    src/core/executor.cpp
    src/core/cpu.cpp
    src/core/cache.cpp
    src/core/pipeline.cpp
    src/core/plic.cpp
    src/core/timer.cpp
    src/core/nic.cpp
    src/core/coherence.cpp
    src/core/multicore.cpp
    src/core/assembler.cpp
    src/core/config.cpp
)

target_include_directories(rvsim_core PUBLIC src)
if(FORMAT_SHIM_DIR)
    target_include_directories(rvsim_core SYSTEM BEFORE PUBLIC "${FORMAT_SHIM_DIR}")
endif()
if(FORMAT_LIB)
    target_link_libraries(rvsim_core PUBLIC ${FORMAT_LIB})
endif()
 
# ── Main executable ──────────────────────────────────────────────────────────
 
add_executable(rvsim src/main.cpp)
target_link_libraries(rvsim PRIVATE rvsim_core)
 
# ── Tests ────────────────────────────────────────────────────────────────────
 
enable_testing()
 
add_executable(rvsim_tests
    test/test_main.cpp
    test/test_decoder.cpp
    test/test_memory.cpp
    test/test_executor.cpp
    test/test_programs.cpp
    test/test_cpu.cpp
    test/test_cache.cpp
    test/test_pipeline.cpp
    test/test_rv32m.cpp
    test/test_rv32a.cpp
    test/test_rvv.cpp
    test/test_csr.cpp
    test/test_plic.cpp
    test/test_timer.cpp
    test/test_nic.cpp
    test/test_coherence.cpp
    test/test_multicore.cpp
    test/test_assembler.cpp
    test/test_config.cpp
    test/test_sparse_memory.cpp
)
 
target_link_libraries(rvsim_tests PRIVATE rvsim_core)
 
add_test(NAME rvsim_tests COMMAND rvsim_tests)
 
add_custom_target(check
    COMMAND rvsim_tests
    DEPENDS rvsim_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running RISC-V simulator tests"
)

# ── Benchmarks ──────────────────────────────────────────────────────────
# TODO: re-enable later, in benchmarking+documentation update.

#add_executable(bench_cache benchmark/bench_cache.cpp)
#target_link_libraries(bench_cache PRIVATE rvsim_core)

#add_executable(bench_pipeline benchmark/bench_pipeline.cpp)
#target_link_libraries(bench_pipeline PRIVATE rvsim_core)
