cmake_minimum_required(VERSION 3.29.2)
project(numerical-linalg_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(BUILD_GMOCK OFF)
set(INSTALL_GMOCK OFF)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-Wcast-align -Wcast-qual -Wconversion -Wctor-dtor-privacy)
    add_compile_options(-Wenum-compare -Wfloat-equal -Wnon-virtual-dtor)
    add_compile_options(-Wold-style-cast -Woverloaded-virtual -Wredundant-decls)
    add_compile_options(-Wsign-conversion -Wsign-promo)

    # Проверка ОС перед включением санитайзеров
    if (UNIX)
        add_compile_options(-fsanitize=address)
        add_link_options(-fsanitize=address)
    endif()
endif()

enable_testing()

file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)
