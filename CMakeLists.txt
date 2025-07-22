cmake_minimum_required(VERSION 3.18)


project(_spsolve LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(DEV_MODULE Development.Module)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "CXX Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)


find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP REQUIRED)


nanobind_add_module(
  _spsolve
  STABLE_ABI
  NB_STATIC
  src/spsolve.h
  src/spsolve.cpp
  src/spsolve_triangular_C.cpp
  src/spsolve_triangular_F.cpp
)

target_include_directories(_spsolve PRIVATE src)


if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(_spsolve PRIVATE /arch:AVX2 /O2)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_spsolve PRIVATE -mavx2 -O3 -fopenmp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_options(_spsolve PRIVATE /arch:AVX2 /O3 /Qiopenmp)
    target_compile_options(_spsolve PRIVATE /Qopenmp-simd- /Wno-debug-option-simd)
    target_compile_options(_spsolve PRIVATE /Wno-deprecated-literal-operator)
  else()
    target_compile_options(_spsolve PRIVATE -arch AVX2 -O3 -fiopenmp)
    target_compile_options(_spsolve PRIVATE -qno-openmp-simd -Wno-debug-option-simd)
    target_compile_options(_spsolve PRIVATE -Wno-deprecated-literal-operator)
  endif()
endif()

target_link_libraries(_spsolve PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS _spsolve LIBRARY DESTINATION spsolve)