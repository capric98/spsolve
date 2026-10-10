cmake_minimum_required(VERSION 3.21)

project(_spsolve LANGUAGES C CXX)

option(SP_USE_OPENMP "Enable OpenMP parallelism" ON)
option(SP_ARCH_NATIVE "Enable host-specific compiler optimizations" OFF)
option(SP_ENABLE_AVX2 "Build runtime-dispatched AVX2+FMA dense kernels (x86-64 only)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEV_MODULE Development.Module)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release MinSizeRel RelWithDebInfo)
endif()

find_package(Python 3.14 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(ORDERING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/suitesparse_ordering")
add_library(spsolve_ordering STATIC
  "${ORDERING_DIR}/suitesparse_config_shim.c"
  "${ORDERING_DIR}/amd_1.c"
  "${ORDERING_DIR}/amd_2.c"
  "${ORDERING_DIR}/amd_aat.c"
  "${ORDERING_DIR}/amd_control.c"
  "${ORDERING_DIR}/amd_defaults.c"
  "${ORDERING_DIR}/amd_dump.c"
  "${ORDERING_DIR}/amd_info.c"
  "${ORDERING_DIR}/amd_order.c"
  "${ORDERING_DIR}/amd_post_tree.c"
  "${ORDERING_DIR}/amd_postorder.c"
  "${ORDERING_DIR}/amd_preprocess.c"
  "${ORDERING_DIR}/amd_valid.c"
  "${ORDERING_DIR}/amd_l1.c"
  "${ORDERING_DIR}/amd_l2.c"
  "${ORDERING_DIR}/amd_l_aat.c"
  "${ORDERING_DIR}/amd_l_control.c"
  "${ORDERING_DIR}/amd_l_defaults.c"
  "${ORDERING_DIR}/amd_l_dump.c"
  "${ORDERING_DIR}/amd_l_info.c"
  "${ORDERING_DIR}/amd_l_order.c"
  "${ORDERING_DIR}/amd_l_post_tree.c"
  "${ORDERING_DIR}/amd_l_postorder.c"
  "${ORDERING_DIR}/amd_l_preprocess.c"
  "${ORDERING_DIR}/amd_l_valid.c"
  "${ORDERING_DIR}/colamd.c"
  "${ORDERING_DIR}/colamd_l.c"
)
target_include_directories(spsolve_ordering PUBLIC "${ORDERING_DIR}")
target_compile_definitions(spsolve_ordering PRIVATE NPRINT NDEBUG)
set_target_properties(spsolve_ordering PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET hidden
)

if(SP_USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

nanobind_add_module(
  _spsolve
  NB_STATIC
  src/spsolve.cpp
  src/sparse_lu.cpp
  src/dense_kernels.cpp
  src/dense_kernels.hpp
  src/sparse_lu.hpp
  src/sparse_ldlt.hpp
  src/sparse_lu_panels.hpp
  src/spsolve_triangular.hpp
  src/spsolve_triangular_C.hpp
  src/spsolve_triangular_F.hpp
)

target_include_directories(_spsolve PRIVATE src)
target_include_directories(_spsolve PRIVATE "${ORDERING_DIR}")
target_link_libraries(_spsolve PRIVATE Threads::Threads spsolve_ordering)

# AVX2+FMA dense kernels live in their own translation unit, the only one
# compiled for AVX2; everything else targets the baseline ISA. The kernels
# are selected at run time after a CPUID/XCR0 check (see src/dense_kernels.cpp
# and the rules at the top of src/dense_kernels_avx2.cpp). Never enable
# link-time code generation (/GL, -flto) for this target.
set(SP_X86_64 OFF)
if(CMAKE_OSX_ARCHITECTURES)
  if("x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES)
    set(SP_X86_64 ON)
  endif()
elseif(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
  # The compiler's target: Visual Studio generators report the host
  # processor in CMAKE_SYSTEM_PROCESSOR even for -A Win32 or -A ARM64.
  if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    set(SP_X86_64 ON)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(SP_X86_64 ON)
endif()
# 1 or 0, not ON/OFF: the value is tested by the preprocessor.
if(SP_ENABLE_AVX2 AND SP_X86_64)
  set(SP_HAVE_AVX2_KERNELS 1)
else()
  set(SP_HAVE_AVX2_KERNELS 0)
endif()
target_compile_definitions(_spsolve PRIVATE SPSOLVE_HAVE_AVX2_KERNELS=${SP_HAVE_AVX2_KERNELS})

if(SP_HAVE_AVX2_KERNELS)
  target_sources(_spsolve PRIVATE src/dense_kernels_avx2.cpp)
  # Per-source options follow the target's options on the command line.
  if(MSVC)
    # SP_ARCH_NATIVE already compiles every source with /arch:AVX2; repeating
    # it would only trigger warning D9025.
    if(NOT SP_ARCH_NATIVE)
      set_source_files_properties(src/dense_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    endif()
  else()
    # The target-wide -ffp-contract=off below also covers this file: its
    # running order multiplies and subtracts separately, which GCC would
    # otherwise fuse under -mfma. Explicit FMA intrinsics are unaffected.
    list(LENGTH CMAKE_OSX_ARCHITECTURES SP_OSX_ARCH_COUNT)
    if(APPLE AND SP_OSX_ARCH_COUNT GREATER 1)
      # Universal build: only the x86_64 slice may use AVX2; the source is
      # empty in the other slices.
      set_source_files_properties(src/dense_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS
        "-Xarch_x86_64;-mavx2;-Xarch_x86_64;-mfma")
    else()
      set_source_files_properties(src/dense_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS
        "-mavx2;-mfma")
    endif()
  endif()
  message(STATUS "Building runtime-dispatched AVX2+FMA dense kernels")
else()
  if(NOT SP_ENABLE_AVX2)
    message(STATUS "AVX2+FMA dense kernels disabled (SP_ENABLE_AVX2=OFF); using sse2/scalar kernels")
  else()
    message(STATUS "AVX2+FMA dense kernels not built: target is not x86-64; using baseline kernels")
  endif()
endif()

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
  # The whole module is built without floating-point contraction, so every
  # scalar and vector path (the dense kernels and the scalar walks: sparse-row
  # updates and dense-row updates from sparse pivots outside panels in
  # sparse_lu.cpp, sparse-source pivot-tail materialization in
  # sparse_lu_panels.hpp) rounds a*b and the following subtraction
  # separately. Exact singular detection relies on those paths rounding
  # identically; GCC's C++ default -ffp-contract=fast would fuse them under
  # -march=native or -mfma. MSVC does not contract under its default
  # /fp:precise.
  target_compile_options(_spsolve PRIVATE -ffp-contract=off)
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
