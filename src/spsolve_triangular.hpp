#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spsolve_detail {

inline int triangular_thread_count(int requested, std::int64_t tasks) noexcept {
    if (tasks <= 1) return 1;
#ifdef _OPENMP
    const int maximum = std::max(1, omp_get_max_threads());
#else
    const int maximum = 1;
#endif
    const int desired = requested <= 0 ? maximum : std::max(1, requested);
    return std::min<int>(desired, static_cast<int>(std::min<std::int64_t>(
        tasks, static_cast<std::int64_t>(maximum))));
}

template <typename Int>
void validate_triangular_csr(
    const double* data,
    const Int* indices,
    const Int* indptr,
    std::int64_t data_size,
    std::int64_t index_size,
    std::int64_t n,
    bool lower,
    bool unit_diagonal,
    std::vector<double>& diagonal
) {
    if (n < 0 || index_size != data_size) {
        throw std::invalid_argument("invalid CSR array sizes");
    }
    if (static_cast<std::int64_t>(indptr[0]) != 0 ||
        static_cast<std::int64_t>(indptr[n]) != data_size) {
        throw std::invalid_argument("invalid CSR indptr endpoints");
    }

    diagonal.assign(static_cast<std::size_t>(n), unit_diagonal ? 1.0 : 0.0);
    for (std::int64_t row = 0; row < n; ++row) {
        const auto begin = static_cast<std::int64_t>(indptr[row]);
        const auto end = static_cast<std::int64_t>(indptr[row + 1]);
        if (begin < 0 || end < begin || end > data_size) {
            throw std::invalid_argument("CSR indptr must be monotonic and in bounds");
        }
        for (std::int64_t position = begin; position < end; ++position) {
            const auto column = static_cast<std::int64_t>(indices[position]);
            if (column < 0 || column >= n) {
                throw std::invalid_argument("CSR column index is out of bounds");
            }
            if (data[position] != 0.0 &&
                ((lower && column > row) || (!lower && column < row))) {
                throw std::invalid_argument(
                    lower ? "A is not lower triangular" : "A is not upper triangular");
            }
            if ((!unit_diagonal || column != row) &&
                !std::isfinite(data[position])) {
                throw std::invalid_argument("A contains NaN or infinity");
            }
            if (!unit_diagonal && column == row) {
                diagonal[static_cast<std::size_t>(row)] += data[position];
            }
        }
        const double value = diagonal[static_cast<std::size_t>(row)];
        if (!unit_diagonal && (value == 0.0 || !std::isfinite(value))) {
            throw std::runtime_error("A is singular: zero or non-finite diagonal entry");
        }
    }
}

template <typename Int>
void solve_triangular_csr(
    const double* data,
    const Int* indices,
    const Int* indptr,
    std::int64_t data_size,
    std::int64_t index_size,
    double* rhs,
    std::int64_t n,
    std::int64_t nrhs,
    std::int64_t row_stride,
    std::int64_t column_stride,
    bool lower,
    bool unit_diagonal,
    int requested_threads
) {
    std::vector<double> diagonal;
    validate_triangular_csr(
        data, indices, indptr, data_size, index_size, n,
        lower, unit_diagonal, diagonal);
    if (n == 0 || nrhs == 0) return;

    // Adjacent right-hand sides form vectorizable blocks for C layout. Fortran
    // layout uses one independent, sequential RHS per task.
    const std::int64_t block_width = column_stride == 1 ? 4 : 1;
    const std::int64_t tasks = (nrhs + block_width - 1) / block_width;
    const int threads = triangular_thread_count(requested_threads, tasks);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (std::int64_t task = 0; task < tasks; ++task) {
        const std::int64_t first = task * block_width;
        const std::int64_t last = std::min(nrhs, first + block_width);

        if (lower) {
            for (std::int64_t row = 0; row < n; ++row) {
                double* const destination = rhs + row * row_stride;
                const auto begin = static_cast<std::int64_t>(indptr[row]);
                const auto end = static_cast<std::int64_t>(indptr[row + 1]);
                for (std::int64_t position = begin; position < end; ++position) {
                    const auto column = static_cast<std::int64_t>(indices[position]);
                    if (column == row) continue;
                    const double value = data[position];
                    const double* const source = rhs + column * row_stride;
                    for (std::int64_t right = first; right < last; ++right) {
                        destination[right * column_stride] -=
                            value * source[right * column_stride];
                    }
                }
                if (!unit_diagonal) {
                    const double value = diagonal[static_cast<std::size_t>(row)];
                    for (std::int64_t right = first; right < last; ++right) {
                        destination[right * column_stride] /= value;
                    }
                }
            }
        } else {
            for (std::int64_t row = n; row-- > 0;) {
                double* const destination = rhs + row * row_stride;
                const auto begin = static_cast<std::int64_t>(indptr[row]);
                const auto end = static_cast<std::int64_t>(indptr[row + 1]);
                for (std::int64_t position = begin; position < end; ++position) {
                    const auto column = static_cast<std::int64_t>(indices[position]);
                    if (column == row) continue;
                    const double value = data[position];
                    const double* const source = rhs + column * row_stride;
                    for (std::int64_t right = first; right < last; ++right) {
                        destination[right * column_stride] -=
                            value * source[right * column_stride];
                    }
                }
                if (!unit_diagonal) {
                    const double value = diagonal[static_cast<std::size_t>(row)];
                    for (std::int64_t right = first; right < last; ++right) {
                        destination[right * column_stride] /= value;
                    }
                }
            }
        }
    }
}

} // namespace spsolve_detail
