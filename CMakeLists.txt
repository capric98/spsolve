cmake_minimum_required(VERSION 3.21)


project(_spsolve LANGUAGES CXX)
option(SP_USE_MKL "Enable to build with MKL support" ON)

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


set(SPSOLVE_SOURCES
  src/spsolve.cpp
  src/spsolve_triangular_C.hpp
  src/spsolve_triangular_F.hpp
  src/spsolve_pardiso.hpp
)

nanobind_add_module(
  _spsolve
  STABLE_ABI
  NB_STATIC
  ${SPSOLVE_SOURCES}
)

target_include_directories(_spsolve PRIVATE src)


include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "${CMAKE_CXX_COMPILER_ID} supports /arch:AVX2")
    target_compile_options(_spsolve PRIVATE /arch:AVX2)
  endif()
  target_compile_options(_spsolve PRIVATE /O2)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "${CMAKE_CXX_COMPILER_ID} supports -mavx2")
    target_compile_options(_spsolve PRIVATE -mavx2)
  endif()
  target_compile_options(_spsolve PRIVATE -O3 -fopenmp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
      message(STATUS "${CMAKE_CXX_COMPILER_ID} supports /arch:AVX2")
      target_compile_options(_spsolve PRIVATE /arch:AVX2)
    endif()
    target_compile_options(_spsolve PRIVATE /O3 /Qiopenmp)
    target_compile_options(_spsolve PRIVATE /Qopenmp-simd- /Wno-debug-option-simd)
    target_compile_options(_spsolve PRIVATE /Wno-deprecated-literal-operator)
  else()
    check_cxx_compiler_flag("-arch AVX2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
      message(STATUS "${CMAKE_CXX_COMPILER_ID} supports -arch AVX2")
      target_compile_options(_spsolve PRIVATE -arch AVX2)
    endif()
    target_compile_options(_spsolve PRIVATE -O3 -fiopenmp)
    target_compile_options(_spsolve PRIVATE -qno-openmp-simd -Wno-debug-option-simd)
    target_compile_options(_spsolve PRIVATE -Wno-deprecated-literal-operator)
  endif()
endif()

target_link_libraries(_spsolve PRIVATE OpenMP::OpenMP_CXX)


# Use MKL by default
if(SP_USE_MKL)
  # execute_process(
  #   COMMAND "${Python_EXECUTABLE}" cmake_debug.py
  #   OUTPUT_VARIABLE PYTHON_DEBUG_MSG
  #   OUTPUT_STRIP_TRAILING_WHITESPACE
  # )
  # message(STATUS "Python DBG: ${PYTHON_DEBUG_MSG}")

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os, sys; print(os.path.join(sys.prefix, 'Library'), end='')"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; from os.path import dirname, join; print(join(dirname(dirname(dirname(dirname(nanobind.source_dir())))), 'Library'))"
    OUTPUT_VARIABLE PYTHON_BUILD_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(MKL_SEARCH_PATHS
    "${PYTHON_LIB_DIR}"
    "${PYTHON_BUILD_LIB_DIR}"
  )

  message(STATUS "MKL Search Paths: ${MKL_SEARCH_PATHS}")

  find_path(MKL_INCLUDE_DIR NAMES mkl.h
    HINTS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "MKL Include Directory"
  )

  message(STATUS "MKL Include Directory: ${MKL_INCLUDE_DIR}")

  find_library(MKL_LIBRARY NAMES mkl_rt mkl_rt.2
    HINTS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES lib libs bin
    DOC "MKL Runtime Library"
  )

  if(MKL_INCLUDE_DIR AND MKL_LIBRARY)
    message(STATUS "Found MKL headers: ${MKL_INCLUDE_DIR}")
    message(STATUS "Found MKL library: ${MKL_LIBRARY}")
    add_compile_definitions(SP_USE_MKL)
    
    target_include_directories(_spsolve PRIVATE ${MKL_INCLUDE_DIR})
    target_link_libraries(_spsolve PRIVATE ${MKL_LIBRARY})
  else()
    message(FATAL_ERROR "MKL not found in ${MKL_SEARCH_PATHS}. Make sure mkl and mkl-include are installed.")
  endif()

endif()


install(TARGETS _spsolve LIBRARY DESTINATION spsolve)