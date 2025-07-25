cmake_minimum_required(VERSION 3.21)


project(_spsolve LANGUAGES CXX)
option(SP_USE_MKL "Enable to build with MKL support" OFF)

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


if(SP_USE_MKL)
  set(MKL_INTERFACE_FULL "intel_lp64")
  find_package(MKL CONFIG REQUIRED)
  if(MKL_FOUND)
    message(STATUS "Found MKL with: ${MKL_IMPORTED_TARGETS}")
    add_compile_definitions(SP_USE_MKL)
  else()
    message(WARNFATAL_ERROR "MKL not found!")
  endif()

  # target_link_libraries(_spsolve PRIVATE MKL::MKL)
  target_compile_options(_spsolve PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(_spsolve PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(_spsolve PRIVATE $<LINK_ONLY:MKL::MKL>)
  install(FILES "$<TARGET_RUNTIME_DLLS:_spsolve>" DESTINATION spsolve)

  if(WIN32)
    set(MKL_HINTS
      "${MKLROOT}"
      "${ONEAPI_ROOT}/mkl/latest"
    )
    find_file(IOMP5_LIB NAMES libiomp5md.dll HINTS ${MKL_HINTS} DOC "Intel OpenMP Compatibility Library")
    message(STATUS "Found OpenMP runtime: ${IOMP5_LIB}")
    if(IOMP5_LIB)
      install(FILES ${IOMP5_LIB} DESTINATION spsolve)
    endif()
  endif()

endif()


install(TARGETS _spsolve LIBRARY DESTINATION spsolve)