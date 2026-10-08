#include "dense_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define SPSOLVE_DENSE_SSE2 1
#include <emmintrin.h>
#else
#define SPSOLVE_DENSE_SSE2 0
#endif

// The AVX2 translation unit defines avx2_kernels only on x86-64, which can be
// narrower than the build-wide switch (an Apple universal build compiles both
// translation units for every architecture slice).
#if SPSOLVE_HAVE_AVX2_KERNELS && (defined(__x86_64__) || defined(_M_X64))
#define SPSOLVE_DENSE_AVX2 1
#else
#define SPSOLVE_DENSE_AVX2 0
#endif

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#define SPSOLVE_CPUID_MSVC 1
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define SPSOLVE_CPUID_GNU 1
#include <cpuid.h>
#endif

// This translation unit must not contract c - a*b or s + a*b into a fused
// multiply-add: the baseline kernels promise two roundings per term on every
// compiler and target. CMake passes -ffp-contract=off for non-MSVC compilers; MSVC does
// not contract under its default /fp:precise. The AVX2 kernel, which fuses
// only the sum_then_subtract order, is the separately compiled
// dense_kernels_avx2.cpp.

namespace dense_kernels {
namespace {

// Row r of a lower update writes columns [0, lower_limit(r)).
[[nodiscard]] std::size_t lower_limit(
    std::size_t row, std::ptrdiff_t diagonal_shift, std::size_t columns
) noexcept {
    const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(row) + diagonal_shift + 1;
    if (limit <= 0) return 0;
    return std::min(columns, static_cast<std::size_t>(limit));
}

// Portable reference kernels: one output row at a time, k ascending, one
// multiply plus one subtraction (running) or addition (sum_then_subtract)
// per term.
[[nodiscard]] bool scalar_row(
    double* c, const double* a, const double* b, std::size_t ldb,
    std::size_t columns, std::size_t kc, Accumulate order
) noexcept {
    if (order == Accumulate::running) {
        for (std::size_t k = 0; k < kc; ++k) {
            const double multiplier = a[k];
            const double* source = b + k * ldb;
            for (std::size_t j = 0; j < columns; ++j) c[j] -= multiplier * source[j];
        }
    } else {
        constexpr std::size_t block = 64;
        double sums[block];
        for (std::size_t first = 0; first < columns; first += block) {
            const std::size_t count = std::min(block, columns - first);
            for (std::size_t j = 0; j < count; ++j) sums[j] = 0.0;
            for (std::size_t k = 0; k < kc; ++k) {
                const double multiplier = a[k];
                const double* source = b + k * ldb + first;
                for (std::size_t j = 0; j < count; ++j) sums[j] += multiplier * source[j];
            }
            for (std::size_t j = 0; j < count; ++j) c[first + j] -= sums[j];
        }
    }
    for (std::size_t j = 0; j < columns; ++j) {
        if (!std::isfinite(c[j])) return false;
    }
    return true;
}

bool scalar_gemm_sub(
    double* c, std::size_t ldc, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; ++i) {
        if (!scalar_row(c + i * ldc, a + i * lda, b, ldb, n, kc, order)) return false;
    }
    return true;
}

bool scalar_gemm_sub_rows(
    double* const* c_rows, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; ++i) {
        if (!scalar_row(c_rows[i], a + i * lda, b, ldb, n, kc, order)) return false;
    }
    return true;
}

bool scalar_gemm_sub_lower(
    double* c, std::size_t ldc, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    std::ptrdiff_t diagonal_shift, Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t limit = lower_limit(i, diagonal_shift, n);
        if (limit > 0 && !scalar_row(c + i * ldc, a + i * lda, b, ldb, limit, kc, order)) {
            return false;
        }
    }
    return true;
}

bool scalar_axpy_sub(double* y, const double* x, double alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) y[i] -= alpha * x[i];
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(y[i])) return false;
    }
    return true;
}

const Kernels scalar_table{
    "scalar", &scalar_gemm_sub, &scalar_gemm_sub_rows, &scalar_gemm_sub_lower,
    &scalar_axpy_sub};

#if SPSOLVE_DENSE_SSE2

// NR = 4 holds a row's four outputs in two registers, NR = 2 in one, and
// NR = 1 in the low lane, using the scalar _sd forms. Every lane performs the
// same per-term multiply and subtract (or add) sequence, so edges match full
// microtiles in either accumulation order.
template <int NR>
[[nodiscard]] inline __m128d load_part(const double* p) noexcept {
    if constexpr (NR == 1) return _mm_load_sd(p);
    else return _mm_loadu_pd(p);
}

template <int NR>
inline void store_part(double* p, __m128d value) noexcept {
    if constexpr (NR == 1) _mm_store_sd(p, value);
    else _mm_storeu_pd(p, value);
}

template <int NR>
[[nodiscard]] inline __m128d subtract_product(__m128d c, __m128d a, __m128d b) noexcept {
    if constexpr (NR == 1) return _mm_sub_sd(c, _mm_mul_sd(a, b));
    else return _mm_sub_pd(c, _mm_mul_pd(a, b));
}

// One term of either order: running subtracts a*b from the output, while
// sum_then_subtract adds it to a sum that started at zero.
template <int NR, bool Sum>
[[nodiscard]] inline __m128d apply_term(__m128d value, __m128d a, __m128d b) noexcept {
    if constexpr (!Sum) return subtract_product<NR>(value, a, b);
    else if constexpr (NR == 1) return _mm_add_sd(value, _mm_mul_sd(a, b));
    else return _mm_add_pd(value, _mm_mul_pd(a, b));
}

// The single final subtraction of sum_then_subtract: c - s.
template <int NR>
[[nodiscard]] inline __m128d subtract_sum(const double* c, __m128d sum) noexcept {
    if constexpr (NR == 1) return _mm_sub_sd(_mm_load_sd(c), sum);
    else return _mm_sub_pd(_mm_loadu_pd(c), sum);
}

// x - x is +0 for finite x and NaN otherwise; OR retains any NaN bits.
[[nodiscard]] inline __m128d note_finite(__m128d witness, __m128d value) noexcept {
    return _mm_or_pd(witness, _mm_sub_pd(value, value));
}

// Update the MR x NR block whose output row r starts at rows[r] + col. The
// rows use named registers rather than arrays so that every compiler keeps
// the accumulators in registers; rows beyond MR stay zero and are unused.
// Running mode accumulates into the loaded outputs. Sum mode accumulates
// from zero and loads each output only for its final subtraction, which also
// keeps the eight accumulators within the sixteen XMM registers.
template <int MR, int NR, bool Sum>
[[nodiscard]] bool sse2_block(
    double* const* rows, std::size_t col, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    static_assert(MR >= 1 && MR <= 4 && (NR == 1 || NR == 2 || NR == 4));
    const __m128d zero = _mm_setzero_pd();
    [[maybe_unused]] __m128d c0 = zero, c0h = zero, c1 = zero, c1h = zero;
    [[maybe_unused]] __m128d c2 = zero, c2h = zero, c3 = zero, c3h = zero;
    if constexpr (!Sum) {
        c0 = load_part<NR>(rows[0] + col);
        if constexpr (NR == 4) c0h = _mm_loadu_pd(rows[0] + col + 2);
        if constexpr (MR > 1) {
            c1 = load_part<NR>(rows[1] + col);
            if constexpr (NR == 4) c1h = _mm_loadu_pd(rows[1] + col + 2);
        }
        if constexpr (MR > 2) {
            c2 = load_part<NR>(rows[2] + col);
            if constexpr (NR == 4) c2h = _mm_loadu_pd(rows[2] + col + 2);
        }
        if constexpr (MR > 3) {
            c3 = load_part<NR>(rows[3] + col);
            if constexpr (NR == 4) c3h = _mm_loadu_pd(rows[3] + col + 2);
        }
    }
    const double* source = b + col;
    for (std::size_t k = 0; k < kc; ++k, source += ldb) {
        const __m128d b0 = load_part<NR>(source);
        [[maybe_unused]] __m128d b2 = zero;
        if constexpr (NR == 4) b2 = _mm_loadu_pd(source + 2);
        const __m128d a0 = _mm_set1_pd(a[k]);
        c0 = apply_term<NR, Sum>(c0, a0, b0);
        if constexpr (NR == 4) c0h = apply_term<NR, Sum>(c0h, a0, b2);
        if constexpr (MR > 1) {
            const __m128d a1 = _mm_set1_pd(a[lda + k]);
            c1 = apply_term<NR, Sum>(c1, a1, b0);
            if constexpr (NR == 4) c1h = apply_term<NR, Sum>(c1h, a1, b2);
        }
        if constexpr (MR > 2) {
            const __m128d a2 = _mm_set1_pd(a[2 * lda + k]);
            c2 = apply_term<NR, Sum>(c2, a2, b0);
            if constexpr (NR == 4) c2h = apply_term<NR, Sum>(c2h, a2, b2);
        }
        if constexpr (MR > 3) {
            const __m128d a3 = _mm_set1_pd(a[3 * lda + k]);
            c3 = apply_term<NR, Sum>(c3, a3, b0);
            if constexpr (NR == 4) c3h = apply_term<NR, Sum>(c3h, a3, b2);
        }
    }
    if constexpr (Sum) {
        c0 = subtract_sum<NR>(rows[0] + col, c0);
        if constexpr (NR == 4) c0h = subtract_sum<NR>(rows[0] + col + 2, c0h);
        if constexpr (MR > 1) {
            c1 = subtract_sum<NR>(rows[1] + col, c1);
            if constexpr (NR == 4) c1h = subtract_sum<NR>(rows[1] + col + 2, c1h);
        }
        if constexpr (MR > 2) {
            c2 = subtract_sum<NR>(rows[2] + col, c2);
            if constexpr (NR == 4) c2h = subtract_sum<NR>(rows[2] + col + 2, c2h);
        }
        if constexpr (MR > 3) {
            c3 = subtract_sum<NR>(rows[3] + col, c3);
            if constexpr (NR == 4) c3h = subtract_sum<NR>(rows[3] + col + 2, c3h);
        }
    }
    __m128d witness = zero;
    store_part<NR>(rows[0] + col, c0);
    witness = note_finite(witness, c0);
    if constexpr (NR == 4) {
        _mm_storeu_pd(rows[0] + col + 2, c0h);
        witness = note_finite(witness, c0h);
    }
    if constexpr (MR > 1) {
        store_part<NR>(rows[1] + col, c1);
        witness = note_finite(witness, c1);
        if constexpr (NR == 4) {
            _mm_storeu_pd(rows[1] + col + 2, c1h);
            witness = note_finite(witness, c1h);
        }
    }
    if constexpr (MR > 2) {
        store_part<NR>(rows[2] + col, c2);
        witness = note_finite(witness, c2);
        if constexpr (NR == 4) {
            _mm_storeu_pd(rows[2] + col + 2, c2h);
            witness = note_finite(witness, c2h);
        }
    }
    if constexpr (MR > 3) {
        store_part<NR>(rows[3] + col, c3);
        witness = note_finite(witness, c3);
        if constexpr (NR == 4) {
            _mm_storeu_pd(rows[3] + col + 2, c3h);
            witness = note_finite(witness, c3h);
        }
    }
    int mask = _mm_movemask_pd(_mm_cmpneq_pd(witness, zero));
    if constexpr (NR == 1) mask &= 1;
    return mask == 0;
}

// Columns [begin, end) of MR rows: four at a time, then a two- and a
// one-column edge.
template <int MR, bool Sum>
[[nodiscard]] bool sse2_span(
    double* const* rows, std::size_t begin, std::size_t end,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    std::size_t j = begin;
    for (; j + 4 <= end; j += 4) {
        if (!sse2_block<MR, 4, Sum>(rows, j, a, lda, b, ldb, kc)) return false;
    }
    if (j + 2 <= end) {
        if (!sse2_block<MR, 2, Sum>(rows, j, a, lda, b, ldb, kc)) return false;
        j += 2;
    }
    if (j < end) return sse2_block<MR, 1, Sum>(rows, j, a, lda, b, ldb, kc);
    return true;
}

template <bool Sum>
[[nodiscard]] bool sse2_rows_in(
    double* const* rows, std::size_t count, std::size_t begin, std::size_t end,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc
) noexcept {
    switch (count) {
    case 4: return sse2_span<4, Sum>(rows, begin, end, a, lda, b, ldb, kc);
    case 3: return sse2_span<3, Sum>(rows, begin, end, a, lda, b, ldb, kc);
    case 2: return sse2_span<2, Sum>(rows, begin, end, a, lda, b, ldb, kc);
    case 1: return sse2_span<1, Sum>(rows, begin, end, a, lda, b, ldb, kc);
    default: return true;
    }
}

// Up to four rows over columns [begin, end) in the requested order.
[[nodiscard]] bool sse2_rows(
    double* const* rows, std::size_t count, std::size_t begin, std::size_t end,
    const double* a, std::size_t lda, const double* b, std::size_t ldb, std::size_t kc,
    Accumulate order
) noexcept {
    return order == Accumulate::running
        ? sse2_rows_in<false>(rows, count, begin, end, a, lda, b, ldb, kc)
        : sse2_rows_in<true>(rows, count, begin, end, a, lda, b, ldb, kc);
}

bool sse2_gemm_sub(
    double* c, std::size_t ldc, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; i += 4) {
        const std::size_t count = std::min<std::size_t>(4, m - i);
        double* rows[4] = {};
        for (std::size_t r = 0; r < count; ++r) rows[r] = c + (i + r) * ldc;
        if (!sse2_rows(rows, count, 0, n, a + i * lda, lda, b, ldb, kc, order)) return false;
    }
    return true;
}

bool sse2_gemm_sub_rows(
    double* const* c_rows, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; i += 4) {
        const std::size_t count = std::min<std::size_t>(4, m - i);
        if (!sse2_rows(c_rows + i, count, 0, n, a + i * lda, lda, b, ldb, kc, order)) {
            return false;
        }
    }
    return true;
}

bool sse2_gemm_sub_lower(
    double* c, std::size_t ldc, const double* a, std::size_t lda,
    const double* b, std::size_t ldb, std::size_t m, std::size_t n, std::size_t kc,
    std::ptrdiff_t diagonal_shift, Accumulate order
) noexcept {
    for (std::size_t i = 0; i < m; i += 4) {
        const std::size_t count = std::min<std::size_t>(4, m - i);
        // Limits grow with the row, so the first row bounds the columns that
        // every row in the group writes and the last row bounds any write.
        const std::size_t shared = lower_limit(i, diagonal_shift, n);
        if (lower_limit(i + count - 1, diagonal_shift, n) == 0) continue;
        double* rows[4] = {};
        for (std::size_t r = 0; r < count; ++r) rows[r] = c + (i + r) * ldc;
        if (!sse2_rows(rows, count, 0, shared, a + i * lda, lda, b, ldb, kc, order)) {
            return false;
        }
        for (std::size_t r = 1; r < count; ++r) {
            const std::size_t limit = lower_limit(i + r, diagonal_shift, n);
            if (limit > shared && !sse2_rows(rows + r, 1, shared, limit,
                    a + (i + r) * lda, lda, b, ldb, kc, order)) {
                return false;
            }
        }
    }
    return true;
}

bool sse2_axpy_sub(double* y, const double* x, double alpha, std::size_t n) noexcept {
    const __m128d zero = _mm_setzero_pd();
    const __m128d scale = _mm_set1_pd(alpha);
    __m128d witness = zero;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128d low = subtract_product<2>(_mm_loadu_pd(y + i), scale, _mm_loadu_pd(x + i));
        const __m128d high = subtract_product<2>(_mm_loadu_pd(y + i + 2), scale,
            _mm_loadu_pd(x + i + 2));
        _mm_storeu_pd(y + i, low);
        _mm_storeu_pd(y + i + 2, high);
        witness = note_finite(note_finite(witness, low), high);
    }
    for (; i < n; ++i) {
        // The upper lane stays 0 - 0 = 0 and cannot set the mask.
        const __m128d value = subtract_product<1>(_mm_load_sd(y + i), scale, _mm_load_sd(x + i));
        _mm_store_sd(y + i, value);
        witness = note_finite(witness, value);
    }
    return _mm_movemask_pd(_mm_cmpneq_pd(witness, zero)) == 0;
}

const Kernels sse2_table{
    "sse2", &sse2_gemm_sub, &sse2_gemm_sub_rows, &sse2_gemm_sub_lower, &sse2_axpy_sub};

#endif // SPSOLVE_DENSE_SSE2

// Usable kernels, best first, then nullptr. Filled once by initialize().
const Kernels* usable[4] = {};
const Kernels* startup = nullptr;
std::once_flag initialized;
std::atomic<const Kernels*> selected{nullptr};

[[nodiscard]] std::string environment(const char* name) {
#if defined(_MSC_VER)
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return {};
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
#endif
}

[[nodiscard]] const Kernels* usable_named(const char* name) noexcept {
    for (const Kernels* const* entry = usable; *entry != nullptr; ++entry) {
        if (std::strcmp((*entry)->name, name) == 0) return *entry;
    }
    return nullptr;
}

[[nodiscard]] const Kernels* lookup(const char* name) {
    std::string key(name);
    for (char& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (key == "auto") return usable[0];
    if (key == "avx2") key = "avx2-fma";
    return usable_named(key.c_str());
}

void initialize() {
    std::call_once(initialized, [] {
        std::size_t count = 0;
#if SPSOLVE_DENSE_AVX2
        // The only way baseline code reaches the AVX2 translation unit.
        if (cpu_supports_avx2_fma()) usable[count++] = &avx2_kernels;
#endif
#if SPSOLVE_DENSE_SSE2
        usable[count++] = &sse2_table;
#endif
        usable[count++] = &scalar_table;
        usable[count] = nullptr;
        const Kernels* choice = usable[0];
        const std::string forced = environment("SPSOLVE_FORCE_KERNEL");
        if (!forced.empty()) {
            if (const Kernels* named = lookup(forced.c_str())) choice = named;
        }
        startup = choice;
        selected.store(choice, std::memory_order_release);
    });
}

} // namespace

const Kernels& active() {
    initialize();
    return *selected.load(std::memory_order_acquire);
}

const Kernels* select(const char* name) {
    initialize();
    const Kernels* choice = name == nullptr ? startup : lookup(name);
    if (choice != nullptr) selected.store(choice, std::memory_order_release);
    return choice;
}

const Kernels* const* available() {
    initialize();
    return usable;
}

const char* active_name() {
    return active().name;
}

bool cpu_supports_avx2_fma() noexcept {
#if defined(SPSOLVE_CPUID_MSVC)
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 7) return false;
    __cpuidex(registers, 1, 0);
    const unsigned int ecx = static_cast<unsigned int>(registers[2]);
    const bool fma = (ecx & (1u << 12)) != 0;
    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx = (ecx & (1u << 28)) != 0;
    if (!fma || !osxsave || !avx) return false;
    // The operating system must save XMM (bit 1) and YMM (bit 2) state.
    if ((_xgetbv(0) & 0x6u) != 0x6u) return false;
    __cpuidex(registers, 7, 0);
    return (static_cast<unsigned int>(registers[1]) & (1u << 5)) != 0;
#elif defined(SPSOLVE_CPUID_GNU)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(1, 0, &eax, &ebx, &ecx, &edx)) return false;
    const bool fma = (ecx & (1u << 12)) != 0;
    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx = (ecx & (1u << 28)) != 0;
    if (!fma || !osxsave || !avx) return false;
    // Inline xgetbv: the _xgetbv intrinsic would require compiling with -mxsave.
    unsigned int xcr0_low = 0, xcr0_high = 0;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0u));
    (void)xcr0_high;
    if ((xcr0_low & 0x6u) != 0x6u) return false;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1u << 5)) != 0;
#else
    return false;
#endif
}

} // namespace dense_kernels
