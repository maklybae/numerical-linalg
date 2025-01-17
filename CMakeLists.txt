cmake_minimum_required(VERSION 3.29.2)
project(numerical-linalg_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

add_compile_options(
    -g

    -fsanitize=address

    -Werror

    -Wall
    -Wextra
    -Wpedantic

    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wctor-dtor-privacy
    -Wenum-compare
    -Wfloat-equal
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Woverloaded-virtual
    -Wredundant-decls
    -Wsign-conversion
    -Wsign-promo
)

add_link_options(-fsanitize=address)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)
