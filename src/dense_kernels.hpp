#pragma once

// Dense tile kernels shared by the row-hybrid LU panel flush and the
// supernodal LDL.T rank-k update, plus their runtime selection.
//
// This header deliberately holds declarations and PODs only: no inline
// function bodies and no standard headers that instantiate floating-point
// code (<cmath>, <algorithm>, <vector>, ...). It is included by translation
// units compiled for wider instruction sets (for example an AVX2 kernel TU).
// Any inline function or template instantiated in such a TU is emitted as a
// COMDAT with VEX encoding, and the linker keeps one arbitrary copy of each
// COMDAT for the whole module. A baseline caller could then execute an AVX2
// copy of, say, std::min<double> on a CPU without AVX2. Keeping shared code
// out of this header keeps every wide instruction inside its own TU, behind
// the runtime CPU check.

#include <cstddef>

// CMake defines this; the fallback keeps non-CMake builds on baseline kernels.
#ifndef SPSOLVE_HAVE_AVX2_KERNELS
#define SPSOLVE_HAVE_AVX2_KERNELS 0
#endif

namespace dense_kernels {

// All matrices are row-major with explicit leading dimensions. The GEMM
// kernels compute C[i][j] -= sum_k A[i][k] * B[k][j] in one of two fixed
// accumulation orders. The baseline kernels round each term's multiply
// separately; a fused multiply-add kernel keeps the same order but rounds
// each term once:
//
//   running:            c = (...((c - a0*b0) - a1*b1) ...) - a(k-1)*b(k-1)
//   sum_then_subtract:  s = (...((0 + a0*b0) + a1*b1) ...) + a(k-1)*b(k-1);
//                       c = c - s
//
// The row-hybrid LU flush uses running: it is the recurrence of its eager
// rank-1 updates and of pivot-tail materialization, so a dense row receives
// the same arithmetic whichever way its tail is updated. The supernodal LDL.T
// update uses sum_then_subtract: its targets are often much larger than the
// individual terms (a dominant diagonal), and summing the terms first rounds
// at their scale, with a single rounding at the target's scale.
//
// In either order an element's result depends only on its own operands,
// never on its position in a tile, the tile shape, the call path, or the
// calling thread. Each kernel returns false if and only if some element it
// wrote is not finite; it may stop early in that case, and the caller must
// discard the factorization.
enum class Accumulate : unsigned char { running, sum_then_subtract };

// C[i][j] -= sum_k A[i][k] * B[k][j] for i < m, j < n, k < kc.
using GemmSub = bool (*)(double* c, std::size_t ldc,
    const double* a, std::size_t lda, const double* b, std::size_t ldb,
    std::size_t m, std::size_t n, std::size_t kc, Accumulate order) noexcept;

// As GemmSub, with output row i starting at c_rows[i] (column 0 of the tile).
using GemmSubRows = bool (*)(double* const* c_rows,
    const double* a, std::size_t lda, const double* b, std::size_t ldb,
    std::size_t m, std::size_t n, std::size_t kc, Accumulate order) noexcept;

// As GemmSub, writing only elements with i + diagonal_shift >= j. Elements
// above that diagonal are neither read nor written, and microtiles entirely
// above it are skipped.
using GemmSubLower = bool (*)(double* c, std::size_t ldc,
    const double* a, std::size_t lda, const double* b, std::size_t ldb,
    std::size_t m, std::size_t n, std::size_t kc,
    std::ptrdiff_t diagonal_shift, Accumulate order) noexcept;

// y[i] -= alpha * x[i] for i < n.
using AxpySub = bool (*)(double* y, const double* x, double alpha,
    std::size_t n) noexcept;

struct Kernels {
    const char* name;  // "avx2-fma", "sse2", or "scalar"
    GemmSub gemm_sub;
    GemmSubRows gemm_sub_rows;
    GemmSubLower gemm_sub_lower;
    AxpySub axpy_sub;
};

#if SPSOLVE_HAVE_AVX2_KERNELS
// Defined by the separately compiled AVX2+FMA translation unit.
extern const Kernels avx2_kernels;
#endif

// The currently selected kernels. The first call initializes the selection:
// the best kernel this build and CPU support, unless SPSOLVE_FORCE_KERNEL
// (auto|avx2|avx2-fma|sse2|scalar, case-insensitive) names a usable one;
// unusable values are ignored. A factorization must call this once and use
// that snapshot throughout, so a concurrent select() cannot mix kernels.
[[nodiscard]] const Kernels& active();

// Select kernels by name (case-insensitive): "auto" (best available),
// "avx2" or "avx2-fma", "sse2", "scalar"; nullptr restores the startup
// selection. Returns the new selection, or nullptr, leaving the selection
// unchanged, when the name is unknown or unsupported by this build or CPU.
[[nodiscard]] const Kernels* select(const char* name);

// Usable kernels, best first, terminated by nullptr.
[[nodiscard]] const Kernels* const* available();

[[nodiscard]] const char* active_name();

// True when the CPU and operating system support AVX2 and FMA (CPUID plus
// XCR0 YMM state). Always false on non-x86 targets.
[[nodiscard]] bool cpu_supports_avx2_fma() noexcept;

} // namespace dense_kernels
