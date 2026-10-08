// AVX2+FMA dense tile kernels: dense_kernels::avx2_kernels ("avx2-fma").
//
// Build and dispatch rules. CMake compiles this file, and only this file,
// with /arch:AVX2 (MSVC) or -mavx2 -mfma (GCC/Clang), so any code emitted
// here may contain VEX-encoded instructions that fault on a CPU without AVX2.
// None of it may run before dense_kernels.cpp has confirmed AVX2, FMA and the
// operating system's YMM state (cpu_supports_avx2_fma) and selected the table
// below. To keep it that way:
//
// - The whole file is inside the SPSOLVE_HAVE_AVX2_KERNELS && x86-64 guard;
//   dense_kernels.cpp references avx2_kernels under the same condition.
// - It includes only <immintrin.h>, <cstddef> and "dense_kernels.hpp"
//   (declarations only); a typedef-only header such as <cstdint> is the only
//   safe addition. An inline function or template from any other standard
//   header, once instantiated here, is emitted as a COMDAT with VEX encoding,
//   and the linker keeps one copy of each COMDAT for the whole module: a
//   baseline caller of, say, std::min<std::size_t> could then execute this
//   file's copy on a CPU without AVX2. Small helpers are therefore written
//   by hand below.
// - Every helper has internal linkage (anonymous namespace). Only
//   avx2_kernels and its four entry points are external, and baseline code
//   reaches them only through that table.
// - avx2_kernels is constinit, so loading the module runs no code from here.
// - Never enable link-time code generation (/GL, -flto) for this target: it
//   could inline across the instruction-set boundary.
// - Never pass -mno-vzeroupper: both compilers emit vzeroupper before
//   returning from code that used YMM registers, which avoids AVX-SSE
//   transition penalties in the baseline callers.
// - GCC and Clang compile this file, like the rest of the module, with
//   -ffp-contract=off (set target-wide in CMakeLists.txt). GCC implements
//   _mm256_mul_pd and _mm256_sub_pd as vector arithmetic and, by default for
//   C++, would fuse them into the FMA instructions that -mfma enables. The
//   FMA intrinsics used on purpose below are unaffected. MSVC does not
//   contract under its default /fp:precise.
//
// Arithmetic. Each output element performs its kc terms in ascending k:
//
//   running:            c = (...((c - a0*b0) - a1*b1) ...) - a(k-1)*b(k-1),
//                       each product rounded, then each subtraction
//                       (vmulpd + vsubpd: two roundings per term);
//   sum_then_subtract:  s = fma(a(k-1), b(k-1), ... fma(a1, b1, fma(a0, b0, +0)));
//                       c = c - s (vfmadd: one rounding per term).
//
// running deliberately does not fuse, and is bitwise identical to the sse2
// and scalar kernels; it gains from four-wide vectors only. The row-hybrid LU
// applies a panel's terms to a dense row's tail either in this flush or, for
// a row that becomes a pivot first, in the scalar pivot-tail materialization
// of src/sparse_lu_panels.hpp, which rounds twice per term. Both must agree
// bit for bit so that, for example, a duplicated row cancels to an exact
// zero and is reported as singular; with a fused flush the duplicate kept a
// residue near 1e-18 and the singular matrix factored. Fusing running needs
// every such path to call these kernels first. sum_then_subtract (the
// supernodal LDL.T update) has no alternative path, so it fuses; its results
// differ from the sse2 and scalar kernels in the last bits.
//
// The 4x8 microkernel holds a 4-row by 8-column block in eight YMM
// accumulators (running: loaded from C; sum_then_subtract: zero, with one
// c - s per register before the store). Per k it loads two YMM of B,
// broadcasts four elements of A and issues eight vmulpd + vsubpd pairs
// (running) or eight vfmadd (sum_then_subtract). Edges use 1-4 rows by 8 (two
// YMM), 4 (one YMM), 2 (one XMM) or 1 (the low XMM lane, _sd forms) columns
// with the same per-element operations, so an element's bits never depend on
// its tile, the tile shape, the call path or the calling thread, and nothing
// outside the requested rows and columns is read from C or written.
// gemm_sub_lower computes each microtile that the diagonal crosses in an
// aligned 4x8 scratch block (elements on or below the diagonal are copied in,
// the rest start at zero) and stores back only the elements on or below the
// diagonal; microtiles entirely above it are skipped.
//
// axpy_sub computes every element as one running term, y - round(alpha*x),
// four at a time and then one at a time with the same two roundings, so no
// element's result depends on n or on its position, and it matches gemm_sub
// in the running order (and the sse2 and scalar axpy_sub) bit for bit.
//
// Finiteness. Every stored register is folded into a witness as x - x (+0
// for finite x, NaN otherwise, kept by OR) and the witness is tested once per
// block with _CMP_NEQ_UQ against zero; a diagonal-crossing block tests exactly
// the elements it stores. As in the sse2 kernel, a GEMM entry point returns
// false at the first block with a non-finite stored element, leaving later
// blocks uncomputed, and never because of an element it does not write.

#if SPSOLVE_HAVE_AVX2_KERNELS && (defined(__x86_64__) || defined(_M_X64))

#include "dense_kernels.hpp"

#include <immintrin.h>

#include <cstddef>

#if defined(_MSC_VER) && !defined(__clang__)
#define SPSOLVE_AVX2_INLINE __forceinline
#else
#define SPSOLVE_AVX2_INLINE inline __attribute__((always_inline))
#endif

namespace dense_kernels {
namespace {

// Row r of a lower update writes columns [0, lower_limit(r)).
[[nodiscard]] SPSOLVE_AVX2_INLINE std::size_t lower_limit(
    std::size_t row, std::ptrdiff_t diagonal_shift, std::size_t columns
) noexcept {
    const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(row) + diagonal_shift + 1;
    if (limit <= 0) return 0;
    const auto bound = static_cast<std::size_t>(limit);
    return bound < columns ? bound : columns;
}

// W consecutive columns of one row in one register: W == 4 in a YMM, W == 2
// in an XMM, W == 1 in the low lane of an XMM whose upper lane stays zero
// (every _sd operation below takes its upper lane from an operand loaded by
// _mm_load_sd or created by _mm_setzero_pd).
template <int W>
struct Part;

template <>
struct Part<4> {
    using V = __m256d;
    static SPSOLVE_AVX2_INLINE V zero() noexcept { return _mm256_setzero_pd(); }
    static SPSOLVE_AVX2_INLINE V load(const double* p) noexcept { return _mm256_loadu_pd(p); }
    static SPSOLVE_AVX2_INLINE void store(double* p, V v) noexcept { _mm256_storeu_pd(p, v); }
    static SPSOLVE_AVX2_INLINE V splat(const double* p) noexcept { return _mm256_broadcast_sd(p); }
    static SPSOLVE_AVX2_INLINE V mul(V x, V y) noexcept { return _mm256_mul_pd(x, y); }
    // acc + x*y rounded once.
    static SPSOLVE_AVX2_INLINE V fmadd(V x, V y, V acc) noexcept { return _mm256_fmadd_pd(x, y, acc); }
    static SPSOLVE_AVX2_INLINE V sub(V x, V y) noexcept { return _mm256_sub_pd(x, y); }
    // x - x is +0 for finite x and NaN otherwise; OR retains any NaN bits.
    static SPSOLVE_AVX2_INLINE V note(V witness, V v) noexcept {
        return _mm256_or_pd(witness, _mm256_sub_pd(v, v));
    }
    [[nodiscard]] static SPSOLVE_AVX2_INLINE bool finite(V witness) noexcept {
        return _mm256_movemask_pd(_mm256_cmp_pd(witness, _mm256_setzero_pd(), _CMP_NEQ_UQ)) == 0;
    }
};

template <>
struct Part<2> {
    using V = __m128d;
    static SPSOLVE_AVX2_INLINE V zero() noexcept { return _mm_setzero_pd(); }
    static SPSOLVE_AVX2_INLINE V load(const double* p) noexcept { return _mm_loadu_pd(p); }
    static SPSOLVE_AVX2_INLINE void store(double* p, V v) noexcept { _mm_storeu_pd(p, v); }
    static SPSOLVE_AVX2_INLINE V splat(const double* p) noexcept { return _mm_loaddup_pd(p); }
    static SPSOLVE_AVX2_INLINE V mul(V x, V y) noexcept { return _mm_mul_pd(x, y); }
    static SPSOLVE_AVX2_INLINE V fmadd(V x, V y, V acc) noexcept { return _mm_fmadd_pd(x, y, acc); }
    static SPSOLVE_AVX2_INLINE V sub(V x, V y) noexcept { return _mm_sub_pd(x, y); }
    static SPSOLVE_AVX2_INLINE V note(V witness, V v) noexcept {
        return _mm_or_pd(witness, _mm_sub_pd(v, v));
    }
    [[nodiscard]] static SPSOLVE_AVX2_INLINE bool finite(V witness) noexcept {
        return _mm_movemask_pd(_mm_cmp_pd(witness, _mm_setzero_pd(), _CMP_NEQ_UQ)) == 0;
    }
};

template <>
struct Part<1> {
    using V = __m128d;
    static SPSOLVE_AVX2_INLINE V zero() noexcept { return _mm_setzero_pd(); }
    static SPSOLVE_AVX2_INLINE V load(const double* p) noexcept { return _mm_load_sd(p); }
    static SPSOLVE_AVX2_INLINE void store(double* p, V v) noexcept { _mm_store_sd(p, v); }
    static SPSOLVE_AVX2_INLINE V splat(const double* p) noexcept { return _mm_load_sd(p); }
    // The _sd forms take the upper lane from their first operand.
    static SPSOLVE_AVX2_INLINE V mul(V x, V y) noexcept { return _mm_mul_sd(x, y); }
    static SPSOLVE_AVX2_INLINE V fmadd(V x, V y, V acc) noexcept { return _mm_fmadd_sd(x, y, acc); }
    static SPSOLVE_AVX2_INLINE V sub(V x, V y) noexcept { return _mm_sub_sd(x, y); }
    static SPSOLVE_AVX2_INLINE V note(V witness, V v) noexcept {
        return _mm_or_pd(witness, _mm_sub_sd(v, v));
    }
    [[nodiscard]] static SPSOLVE_AVX2_INLINE bool finite(V witness) noexcept {
        // Only the low lane carries data.
        return (_mm_movemask_pd(_mm_cmp_pd(witness, _mm_setzero_pd(), _CMP_NEQ_UQ)) & 1) == 0;
    }
};

// One term of either order: running subtracts the rounded product x*y from
// the output (two roundings, as in the sse2 and scalar kernels), while
// sum_then_subtract adds x*y to a sum that started at zero with one fused
// multiply-add (one rounding).
template <typename P, bool Sum>
[[nodiscard]] SPSOLVE_AVX2_INLINE typename P::V apply_term(
    typename P::V acc, typename P::V x, typename P::V y
) noexcept {
    if constexpr (Sum) return P::fmadd(x, y, acc);
    else return P::sub(acc, P::mul(x, y));
}

// Update the MR x NR block whose output row r starts at rows[r] + col; NR == 8
// uses two registers per row. The rows use named registers rather than arrays
// so that every compiler keeps the accumulators in registers; rows beyond MR
// stay zero and are unused. The block always computes and stores all of its
// elements and reports whether they are all finite.
template <int MR, int NR, bool Sum>
[[nodiscard]] bool avx2_block(
    double* const* rows, std::size_t col, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    static_assert(MR >= 1 && MR <= 4 && (NR == 1 || NR == 2 || NR == 4 || NR == 8));
    using P = Part<NR == 8 ? 4 : NR>;
    using V = typename P::V;
    constexpr bool wide = NR == 8;
    const V zero = P::zero();
    double* const out0 = rows[0] + col;
    [[maybe_unused]] double* out1 = nullptr;
    [[maybe_unused]] double* out2 = nullptr;
    [[maybe_unused]] double* out3 = nullptr;
    const double* const a0 = a;
    [[maybe_unused]] const double* a1 = nullptr;
    [[maybe_unused]] const double* a2 = nullptr;
    [[maybe_unused]] const double* a3 = nullptr;
    if constexpr (MR > 1) { out1 = rows[1] + col; a1 = a + lda; }
    if constexpr (MR > 2) { out2 = rows[2] + col; a2 = a + 2 * lda; }
    if constexpr (MR > 3) { out3 = rows[3] + col; a3 = a + 3 * lda; }

    [[maybe_unused]] V c0 = zero, c0h = zero, c1 = zero, c1h = zero;
    [[maybe_unused]] V c2 = zero, c2h = zero, c3 = zero, c3h = zero;
    if constexpr (!Sum) {
        c0 = P::load(out0);
        if constexpr (wide) c0h = P::load(out0 + 4);
        if constexpr (MR > 1) {
            c1 = P::load(out1);
            if constexpr (wide) c1h = P::load(out1 + 4);
        }
        if constexpr (MR > 2) {
            c2 = P::load(out2);
            if constexpr (wide) c2h = P::load(out2 + 4);
        }
        if constexpr (MR > 3) {
            c3 = P::load(out3);
            if constexpr (wide) c3h = P::load(out3 + 4);
        }
    }
    const double* source = b + col;
    for (std::size_t k = 0; k < kc; ++k, source += ldb) {
        const V b0 = P::load(source);
        [[maybe_unused]] V b4 = zero;
        if constexpr (wide) b4 = P::load(source + 4);
        const V x0 = P::splat(a0 + k);
        c0 = apply_term<P, Sum>(c0, x0, b0);
        if constexpr (wide) c0h = apply_term<P, Sum>(c0h, x0, b4);
        if constexpr (MR > 1) {
            const V x1 = P::splat(a1 + k);
            c1 = apply_term<P, Sum>(c1, x1, b0);
            if constexpr (wide) c1h = apply_term<P, Sum>(c1h, x1, b4);
        }
        if constexpr (MR > 2) {
            const V x2 = P::splat(a2 + k);
            c2 = apply_term<P, Sum>(c2, x2, b0);
            if constexpr (wide) c2h = apply_term<P, Sum>(c2h, x2, b4);
        }
        if constexpr (MR > 3) {
            const V x3 = P::splat(a3 + k);
            c3 = apply_term<P, Sum>(c3, x3, b0);
            if constexpr (wide) c3h = apply_term<P, Sum>(c3h, x3, b4);
        }
    }
    if constexpr (Sum) {
        // The single final subtraction c - s.
        c0 = P::sub(P::load(out0), c0);
        if constexpr (wide) c0h = P::sub(P::load(out0 + 4), c0h);
        if constexpr (MR > 1) {
            c1 = P::sub(P::load(out1), c1);
            if constexpr (wide) c1h = P::sub(P::load(out1 + 4), c1h);
        }
        if constexpr (MR > 2) {
            c2 = P::sub(P::load(out2), c2);
            if constexpr (wide) c2h = P::sub(P::load(out2 + 4), c2h);
        }
        if constexpr (MR > 3) {
            c3 = P::sub(P::load(out3), c3);
            if constexpr (wide) c3h = P::sub(P::load(out3 + 4), c3h);
        }
    }
    V witness = zero;
    P::store(out0, c0);
    witness = P::note(witness, c0);
    if constexpr (wide) {
        P::store(out0 + 4, c0h);
        witness = P::note(witness, c0h);
    }
    if constexpr (MR > 1) {
        P::store(out1, c1);
        witness = P::note(witness, c1);
        if constexpr (wide) {
            P::store(out1 + 4, c1h);
            witness = P::note(witness, c1h);
        }
    }
    if constexpr (MR > 2) {
        P::store(out2, c2);
        witness = P::note(witness, c2);
        if constexpr (wide) {
            P::store(out2 + 4, c2h);
            witness = P::note(witness, c2h);
        }
    }
    if constexpr (MR > 3) {
        P::store(out3, c3);
        witness = P::note(witness, c3);
        if constexpr (wide) {
            P::store(out3 + 4, c3h);
            witness = P::note(witness, c3h);
        }
    }
    return P::finite(witness);
}

// Columns [0, end) of MR rows: eight at a time, then a four-, a two- and a
// one-column edge.
template <int MR, bool Sum>
[[nodiscard]] bool avx2_span(
    double* const* rows, std::size_t end,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    std::size_t j = 0;
    for (; j + 8 <= end; j += 8) {
        if (!avx2_block<MR, 8, Sum>(rows, j, a, lda, b, ldb, kc)) return false;
    }
    if (j + 4 <= end) {
        if (!avx2_block<MR, 4, Sum>(rows, j, a, lda, b, ldb, kc)) return false;
        j += 4;
    }
    if (j + 2 <= end) {
        if (!avx2_block<MR, 2, Sum>(rows, j, a, lda, b, ldb, kc)) return false;
        j += 2;
    }
    if (j < end) return avx2_block<MR, 1, Sum>(rows, j, a, lda, b, ldb, kc);
    return true;
}

template <bool Sum>
[[nodiscard]] bool avx2_rows_in(
    double* const* rows, std::size_t count, std::size_t end,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    switch (count) {
    case 4: return avx2_span<4, Sum>(rows, end, a, lda, b, ldb, kc);
    case 3: return avx2_span<3, Sum>(rows, end, a, lda, b, ldb, kc);
    case 2: return avx2_span<2, Sum>(rows, end, a, lda, b, ldb, kc);
    case 1: return avx2_span<1, Sum>(rows, end, a, lda, b, ldb, kc);
    default: return true;
    }
}

// Up to four rows over columns [0, end) in the requested order.
[[nodiscard]] bool avx2_rows(
    double* const* rows, std::size_t count, std::size_t end,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc,
    Accumulate order
) noexcept {
    return order == Accumulate::running
        ? avx2_rows_in<false>(rows, count, end, a, lda, b, ldb, kc)
        : avx2_rows_in<true>(rows, count, end, a, lda, b, ldb, kc);
}

// The MR x NR block at column col that the diagonal crosses: row r writes
// only columns below limits[r]. The block is computed in a scratch copy that
// holds those outputs and zeros elsewhere; only the written outputs are
// stored back and tested for finiteness.
template <int MR, int NR, bool Sum>
[[nodiscard]] bool avx2_crossing(
    double* const* rows, std::size_t col, const std::size_t* limits,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    alignas(32) double scratch[4][8];
    double* const tile[4] = {scratch[0], scratch[1], scratch[2], scratch[3]};
    for (int r = 0; r < MR; ++r) {
        for (int j = 0; j < NR; ++j) {
            const std::size_t column = col + static_cast<std::size_t>(j);
            scratch[r][j] = column < limits[r] ? rows[r][column] : 0.0;
        }
    }
    // Scratch column j is output column col + j. The block's own flag also
    // covers elements that are not written back, so it is not used.
    (void)avx2_block<MR, NR, Sum>(tile, 0, a, lda, b + col, ldb, kc);
    __m128d witness = _mm_setzero_pd();
    for (int r = 0; r < MR; ++r) {
        for (int j = 0; j < NR; ++j) {
            const std::size_t column = col + static_cast<std::size_t>(j);
            if (column < limits[r]) {
                const double value = scratch[r][j];
                rows[r][column] = value;
                witness = Part<1>::note(witness, _mm_set_sd(value));
            }
        }
    }
    return Part<1>::finite(witness);
}

// limits ascend with the row, so limits[0] bounds the columns that every row
// writes.
template <int MR, int NR, bool Sum>
[[nodiscard]] SPSOLVE_AVX2_INLINE bool avx2_lower_block(
    double* const* rows, std::size_t col, const std::size_t* limits,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    if (col + NR <= limits[0]) return avx2_block<MR, NR, Sum>(rows, col, a, lda, b, ldb, kc);
    return avx2_crossing<MR, NR, Sum>(rows, col, limits, a, lda, b, ldb, kc);
}

// Columns [0, limits[MR - 1]) of MR rows with the avx2_span decomposition;
// columns at or beyond limits[MR - 1] are above the diagonal in every row.
template <int MR, bool Sum>
[[nodiscard]] bool avx2_lower_span(
    double* const* rows, const std::size_t* limits,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    const std::size_t end = limits[MR - 1];
    std::size_t j = 0;
    for (; j + 8 <= end; j += 8) {
        if (!avx2_lower_block<MR, 8, Sum>(rows, j, limits, a, lda, b, ldb, kc)) return false;
    }
    if (j + 4 <= end) {
        if (!avx2_lower_block<MR, 4, Sum>(rows, j, limits, a, lda, b, ldb, kc)) return false;
        j += 4;
    }
    if (j + 2 <= end) {
        if (!avx2_lower_block<MR, 2, Sum>(rows, j, limits, a, lda, b, ldb, kc)) return false;
        j += 2;
    }
    if (j < end) return avx2_lower_block<MR, 1, Sum>(rows, j, limits, a, lda, b, ldb, kc);
    return true;
}

template <bool Sum>
[[nodiscard]] bool avx2_lower_rows_in(
    double* const* rows, std::size_t count, const std::size_t* limits,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    switch (count) {
    case 4: return avx2_lower_span<4, Sum>(rows, limits, a, lda, b, ldb, kc);
    case 3: return avx2_lower_span<3, Sum>(rows, limits, a, lda, b, ldb, kc);
    case 2: return avx2_lower_span<2, Sum>(rows, limits, a, lda, b, ldb, kc);
    case 1: return avx2_lower_span<1, Sum>(rows, limits, a, lda, b, ldb, kc);
    default: return true;
    }
}

} // namespace

bool avx2_gemm_sub(
    double* c, std::size_t ldc, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; i += 4) {
        const std::size_t count = m - i < 4 ? m - i : 4;
        double* rows[4] = {};
        for (std::size_t r = 0; r < count; ++r) rows[r] = c + (i + r) * ldc;
        if (!avx2_rows(rows, count, n, a + i * lda, lda, b, ldb, kc, order)) return false;
    }
    return true;
}

bool avx2_gemm_sub_rows(
    double* const* c_rows, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; i += 4) {
        const std::size_t count = m - i < 4 ? m - i : 4;
        if (!avx2_rows(c_rows + i, count, n, a + i * lda, lda, b, ldb, kc, order)) return false;
    }
    return true;
}

bool avx2_gemm_sub_lower(
    double* c, std::size_t ldc, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    std::ptrdiff_t diagonal_shift, Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; i += 4) {
        const std::size_t count = m - i < 4 ? m - i : 4;
        std::size_t limits[4] = {};
        for (std::size_t r = 0; r < count; ++r) limits[r] = lower_limit(i + r, diagonal_shift, n);
        if (limits[count - 1] == 0) continue;  // The whole group is above the diagonal.
        double* rows[4] = {};
        for (std::size_t r = 0; r < count; ++r) rows[r] = c + (i + r) * ldc;
        const double* group = a + i * lda;
        const bool finite = order == Accumulate::running
            ? avx2_lower_rows_in<false>(rows, count, limits, group, lda, b, ldb, kc)
            : avx2_lower_rows_in<true>(rows, count, limits, group, lda, b, ldb, kc);
        if (!finite) return false;
    }
    return true;
}

bool avx2_axpy_sub(double* y, const double* x, double alpha, std::size_t n) noexcept {
    // Each element is one running term, y - round(alpha * x): four at a time,
    // then one at a time with the same two roundings (upper lanes stay zero).
    const __m256d scale = _mm256_set1_pd(alpha);
    __m256d witness = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d value = _mm256_sub_pd(_mm256_loadu_pd(y + i),
            _mm256_mul_pd(scale, _mm256_loadu_pd(x + i)));
        _mm256_storeu_pd(y + i, value);
        witness = Part<4>::note(witness, value);
    }
    const __m128d scale_low = _mm_set_sd(alpha);
    __m128d tail = _mm_setzero_pd();
    for (; i < n; ++i) {
        const __m128d value = _mm_sub_sd(_mm_load_sd(y + i), _mm_mul_sd(scale_low, _mm_load_sd(x + i)));
        _mm_store_sd(y + i, value);
        tail = Part<1>::note(tail, value);
    }
    return Part<4>::finite(witness) && Part<1>::finite(tail);
}

constinit const Kernels avx2_kernels{
    "avx2-fma", &avx2_gemm_sub, &avx2_gemm_sub_rows, &avx2_gemm_sub_lower, &avx2_axpy_sub};

} // namespace dense_kernels

#endif // SPSOLVE_HAVE_AVX2_KERNELS && x86-64
