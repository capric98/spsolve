cmake_minimum_required(VERSION 3.21)

project(_spsolve LANGUAGES CXX)

option(SP_USE_OPENMP "Enable OpenMP parallelism" ON)
option(SP_ARCH_NATIVE "Enable host-specific compiler optimizations" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEV_MODULE Development.Module)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release MinSizeRel RelWithDebInfo)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

if(SP_USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

nanobind_add_module(
  _spsolve
  NB_STATIC
  src/spsolve.cpp
  src/sparse_lu.cpp
  src/sparse_lu.hpp
  src/spsolve_triangular.hpp
  src/spsolve_triangular_C.hpp
  src/spsolve_triangular_F.hpp
)

target_include_directories(_spsolve PRIVATE src)
target_link_libraries(_spsolve PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(_spsolve PRIVATE /O2 /EHsc)
  if(SP_ARCH_NATIVE)
    target_compile_options(_spsolve PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(_spsolve PRIVATE -O3)
  if(SP_ARCH_NATIVE)
    target_compile_options(_spsolve PRIVATE -march=native)
  endif()
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(_spsolve PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "Building with OpenMP support")
elseif(SP_USE_OPENMP)
  message(WARNING "OpenMP was not found; triangular kernels will be serial (LU still uses std::thread)")
else()
  message(STATUS "OpenMP disabled; triangular kernels will be serial (LU still uses std::thread)")
endif()

install(TARGETS _spsolve LIBRARY DESTINATION spsolve)
