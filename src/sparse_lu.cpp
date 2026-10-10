#include "sparse_lu.hpp"
#include "dense_kernels.hpp"
#include "sparse_ldlt.hpp"
#include "sparse_lu_panels.hpp"

#include "amd.h"
#include "colamd.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using index_t = NativeSparseLU::index_t;
using Entry = NativeSparseLU::Entry;
using Clock = std::chrono::steady_clock;

// A dense row stores exactly the columns [dense_offset, dense_end()); every
// column outside that range is an exact zero. Storage ends at n or at a
// multiple of 64, and grows on demand before any write beyond its end.
struct WorkRow {
    std::vector<Entry> active;
    std::vector<Entry> scratch;
    std::vector<Entry> lower;
    std::vector<double> dense;
    index_t dense_offset = 0;
    bool dense_mode = false;

    [[nodiscard]] index_t dense_end() const noexcept {
        return dense_offset + static_cast<index_t>(dense.size());
    }

    // Zero-fills new columns. Rounding to 64 bounds padding at 63 columns,
    // while std::vector's geometric capacity keeps reallocations logarithmic.
    void grow_dense(index_t needed_end, index_t n) {
        if (needed_end <= dense_end()) return;
        const index_t rounded = (needed_end + 63) / 64 * 64;
        dense.resize(static_cast<std::size_t>(std::min(n, rounded) - dense_offset), 0.0);
    }
};

struct Addition {
    index_t col;
    index_t row;
};

[[nodiscard]] int available_threads() noexcept {
    const unsigned int hardware = std::thread::hardware_concurrency();
    return hardware == 0 ? 1 : static_cast<int>(hardware);
}

[[nodiscard]] int resolve_threads(int requested) noexcept {
    const int maximum = available_threads();
    if (requested <= 0) {
        return maximum;
    }
    return std::max(1, std::min(requested, maximum));
}

[[nodiscard]] std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

// Sparse pivot steps have a barrier between them, but their row updates are
// independent. Keeping this small worker team alive across all pivots avoids
// repeatedly creating an OpenMP team, which is particularly expensive on
// Windows. The calling thread participates and counts toward thread_count.
class FactorExecutor {
public:
    explicit FactorExecutor(int thread_count)
        : thread_count_(std::max(1, thread_count)) {}

    FactorExecutor(const FactorExecutor&) = delete;
    FactorExecutor& operator=(const FactorExecutor&) = delete;

    ~FactorExecutor() {
        {
            std::lock_guard lock(job_mutex_);
            stopping_ = true;
        }
        job_ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    // Runs function(index, worker) for every index below count; worker is in
    // [0, max_workers) (all threads when max_workers is zero), 0 being the
    // calling thread. After an item throws, no further item starts and the
    // first exception is rethrown here once every worker has stopped.
    template <typename Function>
    void parallel_for(std::size_t count, Function&& function, int max_workers = 0) {
        if (count == 0) return;
        if (thread_count_ == 1) {
            for (std::size_t index = 0; index < count; ++index) function(index, 0);
            return;
        }
        start_workers();
        {
            std::lock_guard lock(job_mutex_);
            count_ = count;
            worker_limit_ = max_workers > 0 ? max_workers : thread_count_;
            next_.store(0, std::memory_order_relaxed);
            failed_.store(false, std::memory_order_relaxed);
            worker_error_ = nullptr;
            completed_workers_ = 0;
            job_ = std::forward<Function>(function);
            ++generation_;
        }
        job_ready_.notify_all();
        execute(0);

        {
            std::unique_lock lock(job_mutex_);
            job_done_.wait(lock, [&] {
                return completed_workers_ == workers_.size();
            });
            job_ = {};
        }
        if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
    }

    // True once an item of the current parallel_for has thrown. A long item
    // polls it to stop early; the error is already captured and rethrown.
    [[nodiscard]] bool failed() const noexcept {
        return failed_.load(std::memory_order_relaxed);
    }

private:
    void start_workers() {
        if (!workers_.empty()) return;
        workers_.reserve(static_cast<std::size_t>(thread_count_ - 1));
        for (int worker_id = 1; worker_id < thread_count_; ++worker_id) {
            workers_.emplace_back([this, worker_id] { worker_loop(worker_id); });
        }
    }

    void worker_loop(int worker_id) {
        std::size_t observed_generation = 0;
        std::unique_lock lock(job_mutex_);
        while (true) {
            job_ready_.wait(lock, [&] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) return;
            observed_generation = generation_;
            lock.unlock();
            execute(worker_id);
            lock.lock();
            ++completed_workers_;
            if (completed_workers_ == workers_.size()) job_done_.notify_one();
        }
    }

    void execute(int worker_id) noexcept {
        if (worker_id >= worker_limit_) return;
        while (!failed_.load(std::memory_order_relaxed)) {
            const std::size_t index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) break;
            try {
                job_(index, worker_id);
            } catch (...) {
                failed_.store(true, std::memory_order_relaxed);
                std::lock_guard lock(error_mutex_);
                if (worker_error_ == nullptr) worker_error_ = std::current_exception();
            }
        }
    }

    int thread_count_;
    std::vector<std::thread> workers_;
    std::mutex job_mutex_;
    std::mutex error_mutex_;
    std::condition_variable job_ready_;
    std::condition_variable job_done_;
    std::function<void(std::size_t, int)> job_;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> failed_{false};
    std::size_t count_ = 0;
    int worker_limit_ = 1;
    std::size_t generation_ = 0;
    std::size_t completed_workers_ = 0;
    bool stopping_ = false;
    std::exception_ptr worker_error_;
};

// Validates the CSC arrays and, in the same pass, accumulates every row's
// largest magnitude for equilibration into row_max (resized to n), in the
// order and with the expression of a separate pass. The caller checks the
// maxima (zero rows, scaling overflow) after ordering, as before.
template <typename Int>
void validate_csc(
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t data_size,
    index_t indices_size,
    index_t indptr_size,
    index_t n,
    bool check_finite,
    std::vector<double>& row_max
) {
    if (n < 0) {
        throw nb::value_error("matrix dimension must be non-negative");
    }
    if (indptr_size != n + 1) {
        throw nb::value_error("indptr length must equal n + 1");
    }
    if (data_size != indices_size) {
        throw nb::value_error("data and indices must have equal length");
    }
    if (indptr[0] != 0 || static_cast<index_t>(indptr[n]) != data_size) {
        throw nb::value_error("invalid CSC indptr endpoints");
    }
    row_max.assign(static_cast<std::size_t>(n), 0.0);
    for (index_t col = 0; col < n; ++col) {
        const index_t begin = static_cast<index_t>(indptr[col]);
        const index_t end = static_cast<index_t>(indptr[col + 1]);
        if (begin < 0 || end < begin || end > data_size) {
            throw nb::value_error("CSC indptr must be monotonic and within data bounds");
        }
        Int previous = static_cast<Int>(-1);
        for (index_t p = begin; p < end; ++p) {
            const Int row = indices[p];
            if (row < 0 || static_cast<index_t>(row) >= n) {
                throw nb::value_error("CSC row index is out of bounds");
            }
            if (row <= previous) {
                throw nb::value_error("native factorization requires canonical sorted CSC indices");
            }
            previous = row;
            if (check_finite && !std::isfinite(data[p])) {
                throw nb::value_error("matrix contains NaN or infinity");
            }
            row_max[static_cast<std::size_t>(row)] = std::max(
                row_max[static_cast<std::size_t>(row)], std::abs(data[p]));
        }
    }
}

// The largest |a_ij| * row_scale[i] of one column. std::max(acc, x) keeps acc
// when x is NaN, and the accumulators start at +0 and never hold a NaN, so
// every accumulator is the maximum of its non-NaN terms. max is exact and
// commutative on those, so four interleaved accumulators return bitwise the
// value of one running maximum while shortening its dependency chain.
template <typename Int>
[[nodiscard]] double scaled_column_max(
    const double* data,
    const Int* indices,
    index_t begin,
    index_t end,
    const double* row_scale
) {
    double first = 0.0, second = 0.0, third = 0.0, fourth = 0.0;
    index_t p = begin;
    for (; p + 4 <= end; p += 4) {
        first = std::max(first, std::abs(data[p]) *
            row_scale[static_cast<std::size_t>(indices[p])]);
        second = std::max(second, std::abs(data[p + 1]) *
            row_scale[static_cast<std::size_t>(indices[p + 1])]);
        third = std::max(third, std::abs(data[p + 2]) *
            row_scale[static_cast<std::size_t>(indices[p + 2])]);
        fourth = std::max(fourth, std::abs(data[p + 3]) *
            row_scale[static_cast<std::size_t>(indices[p + 3])]);
    }
    for (; p < end; ++p) {
        first = std::max(first, std::abs(data[p]) *
            row_scale[static_cast<std::size_t>(indices[p])]);
    }
    return std::max(std::max(first, second), std::max(third, fourth));
}

// Per row of the scaled, column-permuted matrix: the number of nonzero scaled
// values and the first and last factor columns holding one (first and last
// are meaningful only for a nonzero count). The row engine sizes its setup
// storage from these before filling it.
struct SetupCounts {
    std::vector<index_t> count;
    std::vector<index_t> first;
    std::vector<index_t> last;
};

// One pass over the columns in factor order. Each value is computed with the
// expression of the setup fill, data * row_scale * col_scale, so the counts
// see exactly its nonzeros (a product can underflow to zero).
template <typename Int>
void count_scaled_rows(
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t n,
    const std::vector<index_t>& perm_c,
    const std::vector<double>& row_scale,
    const std::vector<double>& col_scale,
    SetupCounts& counts
) {
    counts.count.assign(static_cast<std::size_t>(n), 0);
    counts.first.assign(static_cast<std::size_t>(n), 0);
    counts.last.assign(static_cast<std::size_t>(n), 0);
    for (index_t new_col = 0; new_col < n; ++new_col) {
        const index_t original_col = perm_c[static_cast<std::size_t>(new_col)];
        const index_t begin = static_cast<index_t>(indptr[original_col]);
        const index_t end = static_cast<index_t>(indptr[original_col + 1]);
        const double scale = col_scale[static_cast<std::size_t>(original_col)];
        for (index_t p = begin; p < end; ++p) {
            const auto row = static_cast<std::size_t>(indices[p]);
            const double value = data[p] * row_scale[row] * scale;
            if (value != 0.0) {
                if (counts.count[row]++ == 0) counts.first[row] = new_col;
                counts.last[row] = new_col;
            }
        }
    }
}

template <typename Int>
[[nodiscard]] std::vector<index_t> checked_permutation(
    const std::vector<Int>& raw,
    index_t n,
    const char* name
) {
    std::vector<index_t> permutation(static_cast<std::size_t>(n));
    std::vector<unsigned char> seen(static_cast<std::size_t>(n), 0);
    for (index_t position = 0; position < n; ++position) {
        const index_t col = static_cast<index_t>(raw[static_cast<std::size_t>(position)]);
        if (col < 0 || col >= n || seen[static_cast<std::size_t>(col)] != 0) {
            throw std::runtime_error(std::string(name) + " returned an invalid permutation");
        }
        seen[static_cast<std::size_t>(col)] = 1;
        permutation[static_cast<std::size_t>(position)] = col;
    }
    return permutation;
}

template <typename Int>
[[nodiscard]] std::vector<index_t> suitesparse_amd_order(
    const Int* indices,
    const Int* indptr,
    index_t n,
    double& estimated_l_nnz,
    double& estimated_max_front
) {
    double info[AMD_INFO]{};
    if constexpr (std::is_same_v<Int, std::int32_t>) {
        if (n > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error("matrix is too large for 32-bit AMD ordering");
        }
        std::vector<std::int32_t> permutation(static_cast<std::size_t>(n));
        const int status = amd_order(
            static_cast<std::int32_t>(n), indptr, indices,
            permutation.data(), nullptr, info);
        if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
            throw std::runtime_error("AMD ordering failed with status " +
                std::to_string(status));
        }
        estimated_l_nnz = info[AMD_LNZ] + static_cast<double>(n);
        estimated_max_front = info[AMD_DMAX];
        return checked_permutation(permutation, n, "AMD");
    } else {
        std::vector<std::int64_t> permutation(static_cast<std::size_t>(n));
        const int status = amd_l_order(
            n, indptr, indices, permutation.data(), nullptr, info);
        if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
            throw std::runtime_error("AMD ordering failed with status " +
                std::to_string(status));
        }
        estimated_l_nnz = info[AMD_LNZ] + static_cast<double>(n);
        estimated_max_front = info[AMD_DMAX];
        return checked_permutation(permutation, n, "AMD");
    }
}

template <typename Int>
[[nodiscard]] std::vector<index_t> suitesparse_colamd_order(
    const Int* indices,
    const Int* indptr,
    index_t n,
    index_t nnz
) {
    const std::size_t recommended = [&] {
        if constexpr (std::is_same_v<Int, std::int32_t>) {
            if (n > std::numeric_limits<std::int32_t>::max() ||
                nnz > std::numeric_limits<std::int32_t>::max()) {
                return std::size_t{0};
            }
            return colamd_recommended(
                static_cast<std::int32_t>(nnz), static_cast<std::int32_t>(n),
                static_cast<std::int32_t>(n));
        } else {
            return colamd_l_recommended(nnz, n, n);
        }
    }();
    if (recommended == 0 || recommended >
        static_cast<std::size_t>(std::numeric_limits<Int>::max())) {
        throw std::runtime_error("COLAMD workspace size overflowed");
    }

    std::vector<Int> workspace(recommended, Int{0});
    std::copy(indices, indices + nnz, workspace.begin());
    std::vector<Int> pointers(static_cast<std::size_t>(n + 1));
    std::copy(indptr, indptr + n + 1, pointers.begin());
    double knobs[COLAMD_KNOBS]{};
    Int stats[COLAMD_STATS]{};
    int ok = 0;
    if constexpr (std::is_same_v<Int, std::int32_t>) {
        colamd_set_defaults(knobs);
        ok = colamd(
            static_cast<std::int32_t>(n), static_cast<std::int32_t>(n),
            static_cast<std::int32_t>(recommended), workspace.data(), pointers.data(),
            knobs, stats);
    } else {
        colamd_l_set_defaults(knobs);
        ok = colamd_l(n, n, static_cast<std::int64_t>(recommended),
            workspace.data(), pointers.data(), knobs, stats);
    }
    if (ok == 0) {
        throw std::runtime_error("COLAMD ordering failed with status " +
            std::to_string(static_cast<index_t>(stats[COLAMD_STATUS])));
    }
    return checked_permutation(pointers, n, "COLAMD");
}

[[nodiscard]] const Entry* find_entry(const std::vector<Entry>& row, index_t col) {
    const auto it = std::lower_bound(row.begin(), row.end(), col,
        [](const Entry& entry, index_t needle) { return entry.col < needle; });
    return (it != row.end() && it->col == col) ? &*it : nullptr;
}

[[nodiscard]] std::tuple<std::vector<double>, std::vector<index_t>, std::vector<index_t>>
csc_to_csr(
    index_t n,
    const std::vector<index_t>& col_ptr,
    const std::vector<index_t>& row_indices,
    const std::vector<double>& values
) {
    std::vector<index_t> row_ptr(static_cast<std::size_t>(n + 1), 0);
    for (const index_t row : row_indices) {
        ++row_ptr[static_cast<std::size_t>(row + 1)];
    }
    std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
    std::vector<index_t> next = row_ptr;
    std::vector<index_t> columns(values.size());
    std::vector<double> output(values.size());
    for (index_t col = 0; col < n; ++col) {
        for (index_t p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const index_t row = row_indices[static_cast<std::size_t>(p)];
            const index_t destination = next[static_cast<std::size_t>(row)]++;
            columns[static_cast<std::size_t>(destination)] = col;
            output[static_cast<std::size_t>(destination)] = values[static_cast<std::size_t>(p)];
        }
    }
    return {std::move(output), std::move(columns), std::move(row_ptr)};
}

[[nodiscard]] double work_row_value(const WorkRow& row, index_t col) {
    if (!row.dense_mode) {
        const Entry* entry = find_entry(row.active, col);
        return entry == nullptr ? 0.0 : entry->value;
    }
    if (col < row.dense_offset ||
        col - row.dense_offset >= static_cast<index_t>(row.dense.size())) {
        return 0.0;
    }
    return row.dense[static_cast<std::size_t>(col - row.dense_offset)];
}

// Density is relative to the row's own remaining span, so a locally dense row
// of a globally sparse matrix (for example, a block-diagonal block) converts.
[[nodiscard]] bool should_use_dense(const WorkRow& row, index_t first_col) {
    if (row.dense_mode || row.active.size() < 64) return false;
    const index_t span = row.active.back().col -
        std::max(first_col, row.active.front().col) + 1;
    return static_cast<std::uint64_t>(row.active.size()) * 4 >=
        static_cast<std::uint64_t>(span);
}

// Storage starts at the first remaining nonzero, so later blocks carry no
// leading zeros, and ends after the last one rounded up to 64 columns.
void convert_to_dense(WorkRow& row, index_t n, index_t first_col) {
    // Every caller passed should_use_dense, which requires at least 64 entries.
    assert(!row.active.empty());
    row.dense_offset = std::max(first_col, row.active.front().col);
    // Callers' active entries all lie at or after first_col, so end > offset.
    const index_t end = std::min(n, (row.active.back().col + 1 + 63) / 64 * 64);
    row.dense.assign(static_cast<std::size_t>(end - row.dense_offset), 0.0);
    for (const Entry& entry : row.active) {
        if (entry.col >= row.dense_offset) {
            row.dense[static_cast<std::size_t>(entry.col - row.dense_offset)] = entry.value;
        }
    }
    std::vector<Entry>().swap(row.active);
    std::vector<Entry>().swap(row.scratch);
    row.dense_mode = true;
}

void update_sparse_row(
    WorkRow& row,
    const std::vector<Entry>& pivot,
    index_t pivot_col,
    double multiplier,
    std::vector<Addition>& additions,
    index_t row_id
) {
    row.lower.push_back({pivot_col, multiplier});
    row.scratch.clear();
    row.scratch.reserve(std::max(row.active.size(), pivot.size()));

    auto target = std::upper_bound(row.active.begin(), row.active.end(), pivot_col,
        [](index_t needle, const Entry& entry) { return needle < entry.col; });
    auto source = std::upper_bound(pivot.begin(), pivot.end(), pivot_col,
        [](index_t needle, const Entry& entry) { return needle < entry.col; });

    while (target != row.active.end() || source != pivot.end()) {
        if (source == pivot.end() ||
            (target != row.active.end() && target->col < source->col)) {
            row.scratch.push_back(*target);
            ++target;
            continue;
        }
        if (target == row.active.end() || source->col < target->col) {
            const double value = -multiplier * source->value;
            if (value != 0.0) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite value generated during LU factorization");
                }
                row.scratch.push_back({source->col, value});
                additions.push_back({source->col, row_id});
            }
            ++source;
            continue;
        }

        const double value = target->value - multiplier * source->value;
        if (value != 0.0) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite value generated during LU factorization");
            }
            row.scratch.push_back({target->col, value});
        }
        ++target;
        ++source;
    }

    row.active.swap(row.scratch);
    row.scratch.clear();
}

// Dense-pivot updates subtract multiplier * source over the pivot row's whole
// stored range with kernels.axpy_sub (multiply, then subtract, per element)
// and reject a non-finite result through the kernel flag. There is no
// zero-source skip: for a finite target it can change at most the sign of a
// zero, which is unobservable because zeros are never emitted as factor
// entries or candidates, and the branch-free loop vectorizes.
void update_dense_row(
    WorkRow& row,
    const WorkRow& pivot_row,
    const std::vector<Entry>& pivot,
    index_t pivot_col,
    index_t n,
    double multiplier,
    const dense_kernels::Kernels& kernels
) {
    // The target is a candidate at pivot_col, which therefore lies inside
    // [dense_offset, dense_end()). Every later write is above pivot_col, so
    // only the end of the storage can be short; it grows before writing.
    row.lower.push_back({pivot_col, multiplier});
    row.dense[static_cast<std::size_t>(pivot_col - row.dense_offset)] = 0.0;

    if (pivot_row.dense_mode) {
        const index_t first = std::max(pivot_col + 1, pivot_row.dense_offset);
        const index_t end = pivot_row.dense_end();
        if (first >= end) return;
        row.grow_dense(end, n);
        if (!kernels.axpy_sub(
                row.dense.data() + (first - row.dense_offset),
                pivot_row.dense.data() + (first - pivot_row.dense_offset),
                multiplier, static_cast<std::size_t>(end - first))) {
            sparse_lu_panels::throw_non_finite();
        }
        return;
    }

    if (!pivot.empty()) row.grow_dense(pivot.back().col + 1, n);
    for (const Entry& entry : pivot) {
        if (entry.col <= pivot_col) continue;
        double& target = row.dense[
            static_cast<std::size_t>(entry.col - row.dense_offset)];
        target -= multiplier * entry.value;
        if (!std::isfinite(target)) {
            throw std::runtime_error(
                "non-finite value generated during LU factorization");
        }
    }
}

// True iff every stored entry (row, col) has a stored transpose (col, row)
// with an exactly equal value. LDL.T uses only one triangle; without a caller
// symmetry assumption, accepting unequal entries would factor another matrix.
//
// O(nnz) transpose walk over sorted CSC: columns are visited in increasing
// order, so the entries (row, col) that name column `row` as their transpose
// arrive in increasing col, which is the sorted order of column `row`. Each
// must therefore be the next unconsumed entry of column `row` (cursor[row])
// with row index col and an equal value. If A is exactly symmetric every
// step succeeds. If every step succeeds, each entry has an equal transpose
// (the consumed one). Consumed positions are distinct, so this is the same
// test as a lookup per entry. Degree check: when column col starts, its
// entries above the diagonal must all have been consumed by row col's
// entries left of the diagonal; a remaining one has no transpose.
template <typename Int>
[[nodiscard]] bool is_exactly_symmetric(
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t n
) {
    std::vector<index_t> cursor(static_cast<std::size_t>(n));
    for (index_t col = 0; col < n; ++col) {
        cursor[static_cast<std::size_t>(col)] = static_cast<index_t>(indptr[col]);
    }
    for (index_t col = 0; col < n; ++col) {
        const index_t end = static_cast<index_t>(indptr[col + 1]);
        const index_t next = cursor[static_cast<std::size_t>(col)];
        if (next < end && static_cast<index_t>(indices[next]) < col) return false;
        for (index_t p = static_cast<index_t>(indptr[col]); p < end; ++p) {
            const auto row = static_cast<std::size_t>(indices[p]);
            const index_t transpose = cursor[row];
            if (transpose >= static_cast<index_t>(indptr[row + 1]) ||
                static_cast<index_t>(indices[transpose]) != col ||
                data[p] != data[transpose]) {
                return false;
            }
            cursor[row] = transpose + 1;
        }
    }
    return true;
}

template <typename Int>
[[nodiscard]] bool try_factorize_symmetric_positive_definite(
    const std::shared_ptr<NativeSparseLU>& result,
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t n,
    const dense_kernels::Kernels& kernels
) {
    const auto symbolic_start = Clock::now();
    if (!is_exactly_symmetric(data, indices, indptr, n)) return false;

    std::vector<index_t> inverse_permutation(static_cast<std::size_t>(n));
    for (index_t k = 0; k < n; ++k) {
        inverse_permutation[static_cast<std::size_t>(
            result->perm_c_[static_cast<std::size_t>(k)])] = k;
    }

    std::vector<index_t> parent(static_cast<std::size_t>(n), -1);
    std::vector<index_t> column_counts(static_cast<std::size_t>(n), 0);
    std::vector<index_t> flags(static_cast<std::size_t>(n), -1);
    for (index_t k = 0; k < n; ++k) {
        flags[static_cast<std::size_t>(k)] = k;
        const index_t original_col = result->perm_c_[static_cast<std::size_t>(k)];
        for (index_t p = static_cast<index_t>(indptr[original_col]);
             p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
            index_t node = inverse_permutation[static_cast<std::size_t>(indices[p])];
            if (node >= k) continue;
            while (flags[static_cast<std::size_t>(node)] != k) {
                if (parent[static_cast<std::size_t>(node)] < 0) {
                    parent[static_cast<std::size_t>(node)] = k;
                }
                ++column_counts[static_cast<std::size_t>(node)];
                flags[static_cast<std::size_t>(node)] = k;
                node = parent[static_cast<std::size_t>(node)];
            }
        }
    }

    std::vector<index_t> strict_col_ptr(static_cast<std::size_t>(n + 1), 0);
    for (index_t k = 0; k < n; ++k) {
        const index_t current = strict_col_ptr[static_cast<std::size_t>(k)];
        const index_t count = column_counts[static_cast<std::size_t>(k)];
        if (count > std::numeric_limits<index_t>::max() - current) {
            throw std::overflow_error("symmetric factor size overflowed");
        }
        strict_col_ptr[static_cast<std::size_t>(k + 1)] = current + count;
    }
    const index_t strict_nnz = strict_col_ptr.back();
    std::vector<double> diagonal(static_cast<std::size_t>(n), 0.0);
    // Narrow fronts retain the single symbolic count pass and simplicial
    // numeric path. The parent/count run test gives the exact fundamental
    // nodes (sparse_ldlt::for_each_node), so their width and internal update
    // work are known before any row index is formed. Wide work must also
    // amortize driver overhead across the whole matrix, including
    // disconnected narrow nodes.
    constexpr index_t minimum_blocked_width = 320;
    constexpr double minimum_work_per_column = 3072.0;
    const auto node_work = [](index_t width, index_t height) {
        // Sum the lower-triangle update terms over this node's pivots:
        // (R-W)*W*(W-1)/2 rectangular-tail terms plus W*(W-1)*(W+1)/6
        // diagonal-block terms. Convert before multiplying to avoid overflow.
        const double w = static_cast<double>(width);
        const double r = static_cast<double>(height);
        return w * (w - 1.0) * (3.0 * r - 2.0 * w + 1.0) / 6.0;
    };
    const double required_work = minimum_work_per_column * static_cast<double>(n);
    double useful_work = 0.0;
    index_t node_count = 0;
    index_t maximum_width = 1;
    index_t maximum_front = 0;
    sparse_ldlt::for_each_node(parent, column_counts, [&](index_t first, index_t last) {
        const index_t height = 1 + column_counts[static_cast<std::size_t>(first)];
        ++node_count;
        maximum_width = std::max(maximum_width, last - first);
        maximum_front = std::max(maximum_front, height);
        if (last - first >= minimum_blocked_width) useful_work += node_work(last - first, height);
    });
    const bool blocked = useful_work > 0.0 && useful_work >= required_work;
    // The simplicial kernel fills the CSC factor column by column; the
    // blocked engine writes it at assembly, from the node blocks and tails.
    std::vector<index_t> strict_rows(blocked ? 0 : static_cast<std::size_t>(strict_nnz));
    std::vector<double> strict_values(blocked ? 0 : static_cast<std::size_t>(strict_nnz));
    sparse_ldlt::Pattern nodes;
    if (blocked) {
        nodes.nodes.reserve(static_cast<std::size_t>(node_count + 1));
        nodes.nodes.push_back(0);
        sparse_ldlt::for_each_node(parent, column_counts, [&](index_t, index_t last) {
            nodes.nodes.push_back(last);
        });
        sparse_ldlt::build_pattern(nodes, indices, indptr, result->perm_c_,
            inverse_permutation, parent, strict_col_ptr, flags);
    }
    result->symbolic_seconds_ = std::chrono::duration<double>(
        Clock::now() - symbolic_start).count();
    const auto numeric_start = Clock::now();
    std::vector<double> accumulator(blocked ? 0 : static_cast<std::size_t>(n), 0.0);
    std::vector<index_t> pattern(blocked ? 0 : static_cast<std::size_t>(n));
    std::vector<index_t> path(blocked ? 0 : static_cast<std::size_t>(n));
    std::fill(flags.begin(), flags.end(), -1);
    std::fill(column_counts.begin(), column_counts.end(), 0);

    std::vector<double> symmetric_scale(static_cast<std::size_t>(n));
    for (index_t row = 0; row < n; ++row) {
        symmetric_scale[static_cast<std::size_t>(row)] =
            std::sqrt(result->row_scale_[static_cast<std::size_t>(row)]);
    }

    const std::uint64_t base_workspace = sizeof(index_t) * (
        inverse_permutation.capacity() + parent.capacity() + column_counts.capacity() +
        flags.capacity() + strict_col_ptr.capacity() + strict_rows.capacity() +
        pattern.capacity() + path.capacity()) + sizeof(double) * (
        strict_values.capacity() + diagonal.capacity() + accumulator.capacity() +
        symmetric_scale.capacity());
    sparse_ldlt::NumericInfo blocked_info;
    result->peak_workspace_bytes_ = base_workspace;
    std::uint64_t numeric_updates = 0;
    // The blocked engine factors into contiguous node blocks; they become the
    // CSC factor only at assembly, after every pivot has passed. Its worker
    // team (started on first use) serves both phases.
    std::unique_ptr<double[]> blocks;
    std::optional<FactorExecutor> executor;
    sparse_ldlt::CscFactor csc{strict_rows, strict_values, static_cast<std::size_t>(strict_nnz)};
    if (blocked) {
        executor.emplace(result->factor_threads_);
        const bool succeeded = sparse_ldlt::factorize(
            data, indices, indptr, result->perm_c_, inverse_permutation,
            symmetric_scale, nodes, blocks, csc, diagonal, kernels, *executor,
            result->factor_threads_, blocked_info, base_workspace);
        result->peak_workspace_bytes_ = blocked_info.workspace_bytes;
        if (!succeeded) return false;
        numeric_updates = blocked_info.update_flops / 2;
    } else {
        // The simplicial kernel reads and writes through local base
        // pointers: MSVC does not use type-based alias analysis, so with
        // vector subscripts it reloaded the data pointers of the escaped
        // factor vectors after every store of the innermost loop.
        double* const accumulated = accumulator.data();
        index_t* const reach = pattern.data();
        index_t* const stack = path.data();
        index_t* const marks = flags.data();
        index_t* const counts = column_counts.data();
        index_t* const rows = strict_rows.data();
        double* const values = strict_values.data();
        double* const pivots = diagonal.data();
        const index_t* const order = result->perm_c_.data();
        const index_t* const inverse = inverse_permutation.data();
        const index_t* const parents = parent.data();
        const index_t* const starts = strict_col_ptr.data();
        const double* const scales = symmetric_scale.data();
        for (index_t k = 0; k < n; ++k) {
            index_t top = n;
            marks[k] = k;
            accumulated[k] = 0.0;
            const index_t original_col = order[k];
            for (index_t p = static_cast<index_t>(indptr[original_col]);
                 p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
                const index_t original_row = static_cast<index_t>(indices[p]);
                index_t node = inverse[original_row];
                if (node > k) continue;
                accumulated[node] += data[p] * scales[original_row] * scales[original_col];

                index_t length = 0;
                while (marks[node] != k) {
                    stack[length++] = node;
                    marks[node] = k;
                    node = parents[node];
                }
                while (length > 0) {
                    reach[--top] = stack[--length];
                }
            }

            double pivot = accumulated[k];
            accumulated[k] = 0.0;
            for (index_t position = top; position < n; ++position) {
                const index_t source = reach[position];
                const double source_value = accumulated[source];
                accumulated[source] = 0.0;
                const index_t first = starts[source];
                const index_t last = first + counts[source];
                numeric_updates += static_cast<std::uint64_t>(last - first);
                for (index_t q = first; q < last; ++q) {
                    accumulated[rows[q]] -= values[q] * source_value;
                }
                const double denominator = pivots[source];
                if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
                const double multiplier = source_value / denominator;
                if (!std::isfinite(multiplier)) return false;
                pivot -= multiplier * source_value;
                const index_t destination = last;
                rows[destination] = k;
                values[destination] = multiplier;
                ++counts[source];
            }
            if (!(pivot > 0.0) || !std::isfinite(pivot)) return false;
            pivots[k] = pivot;
        }
    }
    result->numeric_seconds_ = std::chrono::duration<double>(
        Clock::now() - numeric_start).count();

    const auto assembly_start = Clock::now();
    if (blocked) {
        // The CSC factor, unless phase A has allocated it, is allocated only
        // now; it coexists with the blocks until they are released.
        sparse_ldlt::allocate_factor(csc, *executor, result->factor_threads_);
        result->peak_workspace_bytes_ = std::max(result->peak_workspace_bytes_,
            base_workspace + nodes.bytes() + sizeof(double) * nodes.block_ptr.back() +
            sizeof(index_t) * strict_rows.capacity() + sizeof(double) * strict_values.capacity());
        sparse_ldlt::assemble(nodes, strict_col_ptr, blocks.get(), strict_rows, strict_values,
            *executor, result->factor_threads_);
        blocks.reset();
        executor.reset();
    }
    result->column_storage_ = true;
    result->symmetric_ldlt_ = true;
    result->numeric_engine_ = blocked ? "symmetric-supernodal-ldlt" : "symmetric-ldlt-csc";
    result->l_col_ptr_ = std::move(strict_col_ptr);
    result->l_row_indices_ = std::move(strict_rows);
    result->l_values_ = std::move(strict_values);
    result->ldlt_diagonal_ = std::move(diagonal);

    result->perm_r_ = result->perm_c_;
    result->pivot_swaps_ = 0;
    result->row_scale_ = symmetric_scale;
    result->col_scale_ = symmetric_scale;
    if (strict_nnz > std::numeric_limits<index_t>::max() - n ||
        strict_nnz + n > std::numeric_limits<index_t>::max() / 2) {
        throw std::overflow_error("symmetric factor size overflowed");
    }
    result->factor_nnz_ = 2 * (strict_nnz + n);
    result->fill_ratio_ = result->input_nnz_ == 0
        ? 0.0
        : static_cast<double>(result->factor_nnz_) /
            static_cast<double>(result->input_nnz_);
    result->row_updates_ = static_cast<index_t>(std::min<std::uint64_t>(
        numeric_updates, static_cast<std::uint64_t>(std::numeric_limits<index_t>::max())));
    result->supernodes_ = node_count;
    result->max_supernode_width_ = maximum_width;
    result->largest_front_ = maximum_front;
    result->parallel_panels_ = blocked_info.parallel_panels;
    // Concurrent subtrees count as one more parallel step.
    result->parallel_steps_ = blocked_info.parallel_panels + (blocked_info.subtrees > 1 ? 1 : 0);
    result->spd_subtrees_ = blocked_info.subtrees;
    result->spd_subtree_seconds_ = blocked_info.subtree_seconds;
    result->dense_update_flops_ = blocked_info.update_flops;
    result->spd_panel_entries_ = blocked_info.maximum_panel_entries;
    result->spd_source_seconds_ = blocked_info.source_seconds;
    result->spd_panel_seconds_ = blocked_info.panel_seconds;
    result->spd_pack_seconds_ = blocked_info.pack_seconds;
    result->assembly_seconds_ = std::chrono::duration<double>(
        Clock::now() - assembly_start).count();
    result->factor_seconds_ = result->symbolic_seconds_ +
        result->numeric_seconds_ + result->assembly_seconds_;
    return true;
}

// Every result field this function writes must be listed in ProbeRestart,
// which restores them when the COLAMD probe restarts in the row engine.
template <typename Int>
bool factorize_left_looking(
    const std::shared_ptr<NativeSparseLU>& result,
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t n,
    double diag_pivot_thresh,
    bool bounded_probe = false
) {
    const auto numeric_start = Clock::now();
    result->numeric_engine_ = "left-looking-csc";
    result->column_storage_ = true;
    result->l_col_ptr_.assign(static_cast<std::size_t>(n + 1), 0);
    result->u_col_ptr_.assign(static_cast<std::size_t>(n + 1), 0);

    // Bound discarded COLAMD work by the input size, and leave genuinely wide
    // or expensive columns to the row engine. These limits use this ordering
    // and its actual pivots, rather than an unrelated symmetric graph.
    constexpr index_t probe_front_limit = 192;
    constexpr std::uint64_t probe_column_update_limit = 8'192;
    const auto scaled_budget = [&](std::uint64_t multiplier) {
        const auto nnz = static_cast<std::uint64_t>(result->input_nnz_);
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        return nnz > maximum / multiplier ? maximum : nnz * multiplier;
    };
    const std::uint64_t probe_update_limit = scaled_budget(64);
    const std::uint64_t probe_factor_limit = scaled_budget(20);

    // A column's reach contains its own rows, so a probe with an input
    // column wider than the reach limit is certain to restart (or to raise
    // an error first) before it finishes.
    bool restart_certain = false;
    for (index_t col = 0; bounded_probe && col < n; ++col) {
        if (static_cast<index_t>(indptr[col + 1]) - static_cast<index_t>(indptr[col]) >
            probe_front_limit) {
            restart_certain = true;
            break;
        }
    }

    // Staged reserve. Each factor array first gets 1.15 * E + n entries, with
    // E the larger of 2n and AMD's estimate of nnz(L) (zero unless
    // MMD_AT_PLUS_A). After the first max(256, n / 16) columns, L's and U's
    // sizes are extrapolated to n columns; an array whose extrapolated size
    // exceeds its capacity is reserved for 1.15 times that size plus n, but
    // never more than the ceiling 1.15 * max(E, 2 * nnz(A)) + n. Without the
    // step COLAMD factors (no estimate) grew each array several times, while
    // reserving the ceiling up front left factors without fill most of it
    // unused. The step depends only on n, the factor so far and the current
    // capacity; the capacity (the standard library's growth policy) can
    // decide whether it happens, which affects memory only, never the
    // factors. A probe that must restart skips it, since its arrays are
    // discarded. No array of L or U exceeds a dense triangle, so no reserve
    // does either.
    const double dense_triangle =
        0.5 * static_cast<double>(n) * (static_cast<double>(n) + 1.0);
    const auto bounded_reserve = [&](double entries) {
        return static_cast<std::size_t>(std::min({
            entries,
            dense_triangle,
            static_cast<double>(std::numeric_limits<std::size_t>::max() / 4)}));
    };
    const double estimate = std::max<double>(
        static_cast<double>(2 * n), result->ordering_estimated_l_nnz_);
    const std::size_t reserve =
        bounded_reserve(estimate * 1.15 + static_cast<double>(n));
    const std::size_t reserve_ceiling = bounded_reserve(
        std::max(estimate, 2.0 * static_cast<double>(result->input_nnz_)) * 1.15 +
        static_cast<double>(n));
    const index_t reserve_checkpoint =
        restart_certain ? n : std::max<index_t>(256, n / 16);
    result->l_row_indices_.reserve(reserve);
    result->l_values_.reserve(reserve);
    result->u_row_indices_.reserve(reserve);
    result->u_values_.reserve(reserve);
    const auto extend_reserve = [&](std::vector<index_t>& rows,
                                    std::vector<double>& entries) {
        const double projected = static_cast<double>(entries.size()) *
            (static_cast<double>(n) / static_cast<double>(reserve_checkpoint));
        if (projected <= static_cast<double>(entries.capacity())) return;
        const std::size_t wanted = std::min(
            reserve_ceiling, bounded_reserve(projected * 1.15 + static_cast<double>(n)));
        if (wanted > entries.capacity()) {
            rows.reserve(wanted);
            entries.reserve(wanted);
        }
    };

    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    std::vector<index_t> inverse_row_permutation(static_cast<std::size_t>(n), -1);
    std::vector<index_t> reach(static_cast<std::size_t>(n));
    std::vector<index_t> node_stack(static_cast<std::size_t>(n));
    std::vector<index_t> edge_stack(static_cast<std::size_t>(n));
    std::vector<index_t> mark(static_cast<std::size_t>(n), -1);
    std::vector<index_t> row_at_position(static_cast<std::size_t>(n));
    std::vector<index_t> position_of_row(static_cast<std::size_t>(n));
    std::iota(row_at_position.begin(), row_at_position.end(), index_t{0});
    std::iota(position_of_row.begin(), position_of_row.end(), index_t{0});

    result->peak_workspace_bytes_ = static_cast<std::uint64_t>(n) *
        (sizeof(double) + 7 * sizeof(index_t));

    // Depth-first search from an unmarked root through the columns of L
    // factored so far. Each row is written to reach[--top] when it finishes,
    // so reach[top, n) is a topological order. An unpivoted row has no
    // edges and finishes as soon as it is found, so it is written then
    // without being pushed. The current node's factor column and edge range
    // stay in locals; the stacks hold each suspended node's factor column
    // and next edge. This visits and writes rows exactly as a search that
    // pushes every row and reloads a resumed node's state from its row.
    auto depth_first_reach = [&](index_t root, index_t top, index_t generation) {
        mark[static_cast<std::size_t>(root)] = generation;
        index_t col = inverse_row_permutation[static_cast<std::size_t>(root)];
        if (col < 0) {
            reach[static_cast<std::size_t>(--top)] = root;
            return top;
        }
        const index_t* const l_rows = result->l_row_indices_.data();
        const index_t* const l_ptr = result->l_col_ptr_.data();
        index_t head = 0;
        index_t edge = l_ptr[col] + 1;
        index_t edge_end = l_ptr[col + 1];
        for (;;) {
            if (edge < edge_end) {
                const index_t next = l_rows[edge++];
                if (mark[static_cast<std::size_t>(next)] == generation) continue;
                mark[static_cast<std::size_t>(next)] = generation;
                const index_t next_col =
                    inverse_row_permutation[static_cast<std::size_t>(next)];
                if (next_col < 0) {
                    reach[static_cast<std::size_t>(--top)] = next;
                    continue;
                }
                node_stack[static_cast<std::size_t>(head)] = col;
                edge_stack[static_cast<std::size_t>(head)] = edge;
                ++head;
                col = next_col;
                edge = l_ptr[col] + 1;
                edge_end = l_ptr[col + 1];
                continue;
            }
            reach[static_cast<std::size_t>(--top)] =
                row_at_position[static_cast<std::size_t>(col)];
            if (head == 0) return top;
            --head;
            col = node_stack[static_cast<std::size_t>(head)];
            edge = edge_stack[static_cast<std::size_t>(head)];
            edge_end = l_ptr[col + 1];
        }
    };

    for (index_t factor_col = 0; factor_col < n; ++factor_col) {
        result->l_col_ptr_[static_cast<std::size_t>(factor_col)] =
            static_cast<index_t>(result->l_values_.size());
        result->u_col_ptr_[static_cast<std::size_t>(factor_col)] =
            static_cast<index_t>(result->u_values_.size());

        const index_t original_col =
            result->perm_c_[static_cast<std::size_t>(factor_col)];
        const index_t column_begin = static_cast<index_t>(indptr[original_col]);
        const index_t column_end = static_cast<index_t>(indptr[original_col + 1]);
        // values is zero on entry: only rows of a column's reach are written,
        // and the end of every column resets its whole reach. So one pass
        // scatters the scaled column before the search and finds the search
        // start: the column's pivoted row of smallest pivot position. An
        // unpivoted row's position, -1, is the largest value as unsigned, so
        // the minimum skips unpivoted rows without a branch.
        const double column_scale =
            result->col_scale_[static_cast<std::size_t>(original_col)];
        std::uint64_t first_position = std::numeric_limits<std::uint64_t>::max();
        for (index_t p = column_begin; p < column_end; ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
            first_position = std::min(first_position, static_cast<std::uint64_t>(
                inverse_row_permutation[static_cast<std::size_t>(row)]));
            values[static_cast<std::size_t>(row)] += data[p] *
                result->row_scale_[static_cast<std::size_t>(row)] * column_scale;
        }
        // Search first from that row, then from the column's rows in stored
        // order. In a band the first search typically reaches every other
        // pivoted row of the column, so the traversal repeats from column to
        // column instead of following the row labels. One loop covers both
        // (index column_begin - 1 stands for the first root), which keeps a
        // single call site of the search.
        const bool has_first =
            first_position != std::numeric_limits<std::uint64_t>::max();
        const index_t first_root = has_first
            ? row_at_position[static_cast<std::size_t>(first_position)]
            : index_t{0};
        index_t top = n;
        for (index_t p = column_begin - (has_first ? 1 : 0); p < column_end; ++p) {
            const index_t row = p < column_begin
                ? first_root
                : static_cast<index_t>(indices[p]);
            if (mark[static_cast<std::size_t>(row)] != factor_col) {
                top = depth_first_reach(row, top, factor_col);
            }
        }
        if (bounded_probe) {
            result->numeric_probe_max_front_ = std::max(
                result->numeric_probe_max_front_, n - top);
            if (n - top > probe_front_limit) return false;
        }

        std::uint64_t column_updates = 0;
        for (index_t p = top; p < n; ++p) {
            const index_t row = reach[static_cast<std::size_t>(p)];
            const index_t source_col =
                inverse_row_permutation[static_cast<std::size_t>(row)];
            if (source_col < 0) continue;
            const double coefficient = values[static_cast<std::size_t>(row)];
            const index_t first =
                result->l_col_ptr_[static_cast<std::size_t>(source_col)] + 1;
            const index_t last =
                result->l_col_ptr_[static_cast<std::size_t>(source_col + 1)];
            if (bounded_probe) {
                const auto updates = static_cast<std::uint64_t>(last - first);
                if (updates > probe_column_update_limit - column_updates ||
                    updates > probe_update_limit - result->numeric_probe_updates_) {
                    return false;
                }
                column_updates += updates;
                result->numeric_probe_updates_ += updates;
            }
            result->row_updates_ += last - first;
            for (index_t q = first; q < last; ++q) {
                const index_t target =
                    result->l_row_indices_[static_cast<std::size_t>(q)];
                double& target_value = values[static_cast<std::size_t>(target)];
                target_value -= result->l_values_[static_cast<std::size_t>(q)] *
                    coefficient;
                if (!std::isfinite(target_value)) {
                    throw std::runtime_error(
                        "non-finite value generated during LU factorization");
                }
            }
        }

        // Every candidate is tested for finiteness: a NaN magnitude compares
        // false with the running maximum and would otherwise be skipped, so a
        // column whose only candidate is non-finite would look singular.
        index_t pivot_row = -1;
        double maximum_value = 0.0;
        for (index_t p = top; p < n; ++p) {
            const index_t row = reach[static_cast<std::size_t>(p)];
            if (inverse_row_permutation[static_cast<std::size_t>(row)] >= 0) continue;
            const double magnitude = std::abs(values[static_cast<std::size_t>(row)]);
            if (!std::isfinite(magnitude)) {
                throw std::runtime_error(
                    "non-finite value generated during LU factorization");
            }
            if (magnitude > maximum_value ||
                (magnitude == maximum_value && magnitude != 0.0 &&
                 (pivot_row < 0 ||
                  position_of_row[static_cast<std::size_t>(row)] <
                  position_of_row[static_cast<std::size_t>(pivot_row)]))) {
                maximum_value = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_row < 0 || maximum_value == 0.0) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(factor_col));
        }

        const index_t preferred_row = original_col;
        if (inverse_row_permutation[static_cast<std::size_t>(preferred_row)] < 0) {
            const double preferred = values[static_cast<std::size_t>(preferred_row)];
            if (preferred != 0.0 &&
                std::abs(preferred) >= diag_pivot_thresh * maximum_value) {
                pivot_row = preferred_row;
            }
        }
        // The scan above checked every candidate, so these are defensive.
        const double pivot_value = values[static_cast<std::size_t>(pivot_row)];
        if (!std::isfinite(pivot_value)) {
            throw std::runtime_error("non-finite value generated during LU factorization");
        }
        if (pivot_value == 0.0) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(factor_col));
        }

        for (index_t p = top; p < n; ++p) {
            const index_t row = reach[static_cast<std::size_t>(p)];
            const index_t pivot_position =
                inverse_row_permutation[static_cast<std::size_t>(row)];
            const double value = values[static_cast<std::size_t>(row)];
            if (pivot_position >= 0 && value != 0.0) {
                result->u_row_indices_.push_back(pivot_position);
                result->u_values_.push_back(value);
            }
        }
        result->u_row_indices_.push_back(factor_col);
        result->u_values_.push_back(pivot_value);

        const index_t old_position =
            position_of_row[static_cast<std::size_t>(pivot_row)];
        const index_t displaced_row =
            row_at_position[static_cast<std::size_t>(factor_col)];
        std::swap(row_at_position[static_cast<std::size_t>(factor_col)],
                  row_at_position[static_cast<std::size_t>(old_position)]);
        position_of_row[static_cast<std::size_t>(pivot_row)] = factor_col;
        position_of_row[static_cast<std::size_t>(displaced_row)] = old_position;
        if (old_position != factor_col) ++result->pivot_swaps_;
        inverse_row_permutation[static_cast<std::size_t>(pivot_row)] = factor_col;

        result->l_row_indices_.push_back(pivot_row);
        result->l_values_.push_back(1.0);
        for (index_t p = top; p < n; ++p) {
            const index_t row = reach[static_cast<std::size_t>(p)];
            if (row == pivot_row ||
                inverse_row_permutation[static_cast<std::size_t>(row)] >= 0) {
                values[static_cast<std::size_t>(row)] = 0.0;
                continue;
            }
            const double value = values[static_cast<std::size_t>(row)] / pivot_value;
            if (value != 0.0) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "non-finite multiplier generated during LU factorization");
                }
                result->l_row_indices_.push_back(row);
                result->l_values_.push_back(value);
            }
            values[static_cast<std::size_t>(row)] = 0.0;
        }
        values[static_cast<std::size_t>(pivot_row)] = 0.0;
        result->largest_front_ = std::max(
            result->largest_front_, n - top);
        if (bounded_probe) {
            ++result->numeric_probe_columns_;
            const auto entries = static_cast<std::uint64_t>(result->l_values_.size()) +
                static_cast<std::uint64_t>(result->u_values_.size());
            if (entries > probe_factor_limit) return false;
        }
        if (factor_col + 1 == reserve_checkpoint && reserve_checkpoint < n) {
            extend_reserve(result->l_row_indices_, result->l_values_);
            extend_reserve(result->u_row_indices_, result->u_values_);
        }
    }

    result->l_col_ptr_[static_cast<std::size_t>(n)] =
        static_cast<index_t>(result->l_values_.size());
    result->u_col_ptr_[static_cast<std::size_t>(n)] =
        static_cast<index_t>(result->u_values_.size());
    for (index_t& row : result->l_row_indices_) {
        row = inverse_row_permutation[static_cast<std::size_t>(row)];
        if (row < 0) throw std::runtime_error("LU row permutation is incomplete");
    }
    result->perm_r_ = std::move(row_at_position);
    result->factor_nnz_ = static_cast<index_t>(
        result->l_values_.size() + result->u_values_.size());
    result->fill_ratio_ = result->input_nnz_ == 0
        ? 0.0
        : static_cast<double>(result->factor_nnz_) /
            static_cast<double>(result->input_nnz_);
    result->supernodes_ = n;
    result->numeric_seconds_ = std::chrono::duration<double>(
        Clock::now() - numeric_start).count();
    result->factor_seconds_ = result->numeric_seconds_;
    return true;
}

// Restart state of the COLAMD probe. factorize_left_looking reads perm_c_,
// the scales, input_nnz_ and ordering_estimated_l_nnz_, and writes only the
// fields below besides numeric_probe_*, which the caller sets after the probe
// either way. The permutation and scales are kept as the pristine snapshot
// that the probe's workspace accounting counts. restore() move-assigns every
// saved field, which also releases the discarded CSC factor buffers. Keep
// this list in sync with every result field factorize_left_looking writes.
struct ProbeRestart {
    std::vector<index_t> perm_c;
    std::vector<double> row_scale;
    std::vector<double> col_scale;
    std::string numeric_engine;
    bool column_storage = false;
    std::vector<index_t> perm_r;
    std::vector<index_t> l_col_ptr;
    std::vector<index_t> l_row_indices;
    std::vector<double> l_values;
    std::vector<index_t> u_col_ptr;
    std::vector<index_t> u_row_indices;
    std::vector<double> u_values;
    index_t factor_nnz = 0;
    index_t row_updates = 0;
    index_t pivot_swaps = 0;
    index_t largest_front = 0;
    index_t supernodes = 0;
    double fill_ratio = 0.0;
    double numeric_seconds = 0.0;
    double factor_seconds = 0.0;
    std::uint64_t peak_workspace_bytes = 0;

    void save(const NativeSparseLU& state) {
        perm_c = state.perm_c_;
        row_scale = state.row_scale_;
        col_scale = state.col_scale_;
        numeric_engine = state.numeric_engine_;
        column_storage = state.column_storage_;
        perm_r = state.perm_r_;
        l_col_ptr = state.l_col_ptr_;
        l_row_indices = state.l_row_indices_;
        l_values = state.l_values_;
        u_col_ptr = state.u_col_ptr_;
        u_row_indices = state.u_row_indices_;
        u_values = state.u_values_;
        factor_nnz = state.factor_nnz_;
        row_updates = state.row_updates_;
        pivot_swaps = state.pivot_swaps_;
        largest_front = state.largest_front_;
        supernodes = state.supernodes_;
        fill_ratio = state.fill_ratio_;
        numeric_seconds = state.numeric_seconds_;
        factor_seconds = state.factor_seconds_;
        peak_workspace_bytes = state.peak_workspace_bytes_;
    }

    void restore(NativeSparseLU& state) {
        state.perm_c_ = std::move(perm_c);
        state.row_scale_ = std::move(row_scale);
        state.col_scale_ = std::move(col_scale);
        state.numeric_engine_ = std::move(numeric_engine);
        state.column_storage_ = column_storage;
        state.perm_r_ = std::move(perm_r);
        state.l_col_ptr_ = std::move(l_col_ptr);
        state.l_row_indices_ = std::move(l_row_indices);
        state.l_values_ = std::move(l_values);
        state.u_col_ptr_ = std::move(u_col_ptr);
        state.u_row_indices_ = std::move(u_row_indices);
        state.u_values_ = std::move(u_values);
        state.factor_nnz_ = factor_nnz;
        state.row_updates_ = row_updates;
        state.pivot_swaps_ = pivot_swaps;
        state.largest_front_ = largest_front;
        state.supernodes_ = supernodes;
        state.fill_ratio_ = fill_ratio;
        state.numeric_seconds_ = numeric_seconds;
        state.factor_seconds_ = factor_seconds;
        state.peak_workspace_bytes_ = peak_workspace_bytes;
    }
};

template <typename Int>
std::shared_ptr<NativeSparseLU> factorize_typed(
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t data_size,
    index_t indices_size,
    index_t indptr_size,
    index_t n,
    const std::string& permc_spec,
    double diag_pivot_thresh,
    int num_threads,
    bool check_finite
) {
    std::vector<double> row_max;
    validate_csc(data, indices, indptr, data_size, indices_size, indptr_size,
                 n, check_finite, row_max);
    if (!(diag_pivot_thresh >= 0.0 && diag_pivot_thresh <= 1.0)) {
        throw nb::value_error("diag_pivot_thresh must be between 0 and 1");
    }

    const std::string ordering = uppercase(permc_spec);
    if (ordering != "NATURAL" && ordering != "COLAMD" &&
        ordering != "MMD_AT_PLUS_A") {
        throw nb::value_error(
            "native permc_spec must be 'COLAMD', 'MMD_AT_PLUS_A', or 'NATURAL'");
    }

    auto result = std::make_shared<NativeSparseLU>();
    result->n_ = n;
    result->input_nnz_ = data_size;
    result->factor_threads_ = resolve_threads(num_threads);
    // One snapshot per factorization: a concurrent selection change from
    // another thread cannot mix kernels within this factor.
    const dense_kernels::Kernels& kernels = dense_kernels::active();
    result->dense_kernel_ = kernels.name;
    result->ordering_ = ordering;
    result->row_scale_.assign(static_cast<std::size_t>(n), 1.0);
    result->col_scale_.assign(static_cast<std::size_t>(n), 1.0);

    if (n == 0) {
        result->fill_ratio_ = 0.0;
        return result;
    }

    const auto ordering_start = Clock::now();
    if (ordering == "NATURAL") {
        result->perm_c_.resize(static_cast<std::size_t>(n));
        std::iota(result->perm_c_.begin(), result->perm_c_.end(), index_t{0});
        result->ordering_backend_ = "builtin";
    } else if (ordering == "COLAMD") {
        result->perm_c_ = suitesparse_colamd_order(indices, indptr, n, data_size);
        result->ordering_backend_ = "SuiteSparse COLAMD 3.3.5";
    } else {
        result->perm_c_ = suitesparse_amd_order(
            indices, indptr, n, result->ordering_estimated_l_nnz_,
            result->ordering_estimated_max_front_);
        result->ordering_backend_ = "SuiteSparse AMD 3.3.4";
    }
    result->ordering_seconds_ = std::chrono::duration<double>(
        Clock::now() - ordering_start).count();

    // Equilibrate by infinity norms. Zero rows and columns are structurally
    // singular. validate_csc accumulated the row maxima; the checks run here,
    // after ordering, rows and then columns, each in natural order.
    for (index_t row = 0; row < n; ++row) {
        const double maximum = row_max[static_cast<std::size_t>(row)];
        if (maximum == 0.0) {
            throw std::runtime_error("factor is exactly singular: matrix contains a zero row");
        }
        const double scale = 1.0 / maximum;
        if (!std::isfinite(scale)) {
            throw std::runtime_error("matrix row scaling overflowed");
        }
        result->row_scale_[static_cast<std::size_t>(row)] = scale;
    }
    std::vector<double>().swap(row_max);

    for (index_t col = 0; col < n; ++col) {
        const double maximum = scaled_column_max(
            data, indices, static_cast<index_t>(indptr[col]),
            static_cast<index_t>(indptr[col + 1]), result->row_scale_.data());
        if (maximum == 0.0) {
            throw std::runtime_error("factor is exactly singular: matrix contains a zero column");
        }
        const double scale = 1.0 / maximum;
        if (!std::isfinite(scale)) {
            throw std::runtime_error("matrix column scaling overflowed");
        }
        result->col_scale_[static_cast<std::size_t>(col)] = scale;
    }

    const auto spd_start = Clock::now();
    if (try_factorize_symmetric_positive_definite(
            result, data, indices, indptr, n, kernels)) {
        return result;
    }
    const auto spd_workspace = result->peak_workspace_bytes_;
    if (result->symbolic_seconds_ > 0.0) {
        result->failed_spd_seconds_ = std::chrono::duration<double>(
            Clock::now() - spd_start).count();
        result->symbolic_seconds_ = 0.0;
    }
    const auto finish_lu = [&] {
        result->numeric_seconds_ += result->failed_spd_seconds_;
        result->factor_seconds_ += result->failed_spd_seconds_;
        result->peak_workspace_bytes_ = std::max(
            result->peak_workspace_bytes_, spd_workspace);
    };

    if (ordering == "COLAMD") {
        // Restart from pristine state if actual CSC work is unsuitable. The
        // snapshot holds the requested permutation and equilibration, which
        // the probe only reads (the workspace counted below), and every field
        // factorize_left_looking writes; no partial factors exist yet.
        const auto probe_start = Clock::now();
        ProbeRestart restart_state;
        restart_state.save(*result);
        const bool completed = factorize_left_looking(
            result, data, indices, indptr, n, diag_pivot_thresh, true);
        const double probe_seconds = std::chrono::duration<double>(
            Clock::now() - probe_start).count();
        const auto probe_columns = result->numeric_probe_columns_;
        const auto probe_front = result->numeric_probe_max_front_;
        const auto probe_updates = result->numeric_probe_updates_;
        // The pristine snapshot is temporary workspace. On fallback, count
        // the discarded CSC factor buffers as workspace as well.
        std::uint64_t probe_workspace = result->peak_workspace_bytes_ +
            static_cast<std::uint64_t>(n) * (sizeof(index_t) + 2 * sizeof(double));
        if (!completed) {
            probe_workspace += sizeof(index_t) * (
                result->l_col_ptr_.capacity() + result->u_col_ptr_.capacity() +
                result->l_row_indices_.capacity() + result->u_row_indices_.capacity());
            probe_workspace += sizeof(double) * (
                result->l_values_.capacity() + result->u_values_.capacity());
            restart_state.restore(*result);
        }
        result->numeric_probe_columns_ = probe_columns;
        result->numeric_probe_max_front_ = probe_front;
        result->numeric_probe_updates_ = probe_updates;
        result->numeric_probe_seconds_ = probe_seconds;
        result->numeric_probe_fallback_ = !completed;
        result->peak_workspace_bytes_ = std::max(
            result->peak_workspace_bytes_, probe_workspace);
        if (completed) {
            result->numeric_seconds_ = probe_seconds;
            result->factor_seconds_ = probe_seconds;
            finish_lu();
            return result;
        }
    }

    const double predicted_factor_nnz =
        2.0 * result->ordering_estimated_l_nnz_;
    const double predicted_fill = data_size == 0
        ? 0.0
        : predicted_factor_nnz / static_cast<double>(data_size);
    const bool use_left_looking = ordering != "NATURAL" &&
        result->ordering_estimated_max_front_ > 0.0 &&
        result->ordering_estimated_max_front_ <= 192.0 &&
        predicted_fill <= 20.0;
    if (use_left_looking) {
        factorize_left_looking(
            result, data, indices, indptr, n, diag_pivot_thresh);
        finish_lu();
        return result;
    }

    // Setup: rows are sized from their counts before any value is stored.
    // A row that passes the dense test of should_use_dense(row, 0) on its
    // initial entries (at least 64, filling at least 25% of the span from
    // its first to its last column) gets the dense tail convert_to_dense
    // would build, [first, min(n, first multiple of 64 above last)), and
    // receives its values there directly. Sparse rows receive sorted Entry
    // lists with exact capacity. Only sparse rows are listed in
    // active_columns: a setup-dense row is found through dense_rows, or
    // through the panel block inside a panel (see the candidate scan).
    SetupCounts setup_counts;
    count_scaled_rows(data, indices, indptr, n, result->perm_c_, result->row_scale_,
                      result->col_scale_, setup_counts);
    std::vector<WorkRow> rows(static_cast<std::size_t>(n));
    std::vector<std::vector<index_t>> active_columns(static_cast<std::size_t>(n));
    // Initially dense rows wait in pending_dense, ordered by (dense_offset,
    // row_id), until the factorization reaches their first stored column. A
    // dense row has exact zeros before dense_offset, so it cannot be a
    // candidate earlier; the scan below then visits only reachable rows.
    std::vector<index_t> dense_rows;
    std::vector<index_t> pending_dense;
    std::size_t pending_dense_cursor = 0;
    index_t converted_dense_rows = 0;
    index_t active_dense_rows = 0;
    {
        // Dense rows' storage, indexed by row (null for a sparse row); a value
        // in factor column c lands at dense[c - first].
        std::vector<double*> dense_storage(static_cast<std::size_t>(n), nullptr);
        for (index_t row_id = 0; row_id < n; ++row_id) {
            const auto slot = static_cast<std::size_t>(row_id);
            auto& row = rows[slot];
            const index_t count = setup_counts.count[slot];
            if (count == 0) continue;
            const index_t first = setup_counts.first[slot];
            const index_t last = setup_counts.last[slot];
            if (count >= 64 && static_cast<std::uint64_t>(count) * 4 >=
                    static_cast<std::uint64_t>(last - first + 1)) {
                const index_t end = std::min(n, (last + 1 + 63) / 64 * 64);
                row.dense_offset = first;
                row.dense.assign(static_cast<std::size_t>(end - first), 0.0);
                row.dense_mode = true;
                dense_storage[slot] = row.dense.data();
                pending_dense.push_back(row_id);
                ++converted_dense_rows;
                ++active_dense_rows;
            } else {
                row.active.reserve(static_cast<std::size_t>(count));
            }
        }
        std::vector<index_t> column_rows;
        for (index_t new_col = 0; new_col < n; ++new_col) {
            const index_t original_col = result->perm_c_[static_cast<std::size_t>(new_col)];
            const double scale = result->col_scale_[static_cast<std::size_t>(original_col)];
            column_rows.clear();
            for (index_t p = static_cast<index_t>(indptr[original_col]);
                 p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
                const auto row = static_cast<std::size_t>(indices[p]);
                const double value = data[p] * result->row_scale_[row] * scale;
                if (value == 0.0) continue;
                if (double* dense = dense_storage[row]) {
                    dense[static_cast<std::size_t>(new_col - setup_counts.first[row])] = value;
                } else {
                    rows[row].active.push_back({new_col, value});
                    column_rows.push_back(static_cast<index_t>(row));
                }
            }
            if (!column_rows.empty()) {
                active_columns[static_cast<std::size_t>(new_col)].assign(
                    column_rows.begin(), column_rows.end());
            }
        }
    }
    setup_counts = SetupCounts{};
    std::stable_sort(pending_dense.begin(), pending_dense.end(),
        [&](index_t left, index_t right) {
            return rows[static_cast<std::size_t>(left)].dense_offset <
                rows[static_cast<std::size_t>(right)].dense_offset;
        });

    std::vector<index_t> row_at_position(static_cast<std::size_t>(n));
    std::vector<index_t> position_of_row(static_cast<std::size_t>(n));
    std::iota(row_at_position.begin(), row_at_position.end(), index_t{0});
    std::iota(position_of_row.begin(), position_of_row.end(), index_t{0});
    std::vector<index_t> seen(static_cast<std::size_t>(n), -1);
    result->u_rows_.resize(static_cast<std::size_t>(n));

    const auto factor_start = Clock::now();
    constexpr std::uint64_t parallel_work_threshold = 8'192;
    FactorExecutor executor(result->factor_threads_);
    sparse_lu_panels::DensePanel panel;
    bool dense_panels_enabled = false;
    bool panel_active = false;
    // Per-pivot scratch, cleared at each step instead of reallocated. The
    // addition and conversion slots are indexed by worker.
    std::vector<index_t> candidates;
    std::vector<double> candidate_values;
    std::vector<index_t> targets;
    std::vector<std::vector<Addition>> additions(
        static_cast<std::size_t>(result->factor_threads_));
    std::vector<std::vector<index_t>> dense_conversions(
        static_cast<std::size_t>(result->factor_threads_));

    for (index_t pivot_col = 0; pivot_col < n; ++pivot_col) {
        bool panel_started = false;
        if (!panel_active) {
            // Initial activation needs enough dense work to amortize packing.
            // Once enabled, finish short trailing panels with the same recurrence.
            if (!dense_panels_enabled && active_dense_rows >= 32 && n - pivot_col >= 128) {
                dense_panels_enabled = true;
            }
            if (dense_panels_enabled && active_dense_rows > 0) {
                panel.begin(pivot_col, pivot_col + std::min(n - pivot_col, sparse_lu_panels::width),
                            n, result->factor_threads_);
                panel_active = true;
                panel_started = true;
            }
        }
        // Admit pending rows that start in this column, or anywhere in the
        // current panel. The cursor depends only on offsets and the current
        // column or panel end, never on the thread count.
        const index_t admit_end = panel_active ? panel.end() : pivot_col + 1;
        while (pending_dense_cursor < pending_dense.size() &&
               rows[static_cast<std::size_t>(pending_dense[pending_dense_cursor])].dense_offset <
                   admit_end) {
            dense_rows.push_back(pending_dense[pending_dense_cursor++]);
        }
        // A new panel copies its columns of every dense row that can be a
        // candidate inside it into its block (see DensePanel).
        if (panel_started) panel.gather(dense_rows, rows, position_of_row);

        candidates.clear();
        candidate_values.clear();
        const auto add_candidate = [&](index_t row_id, double value) {
            if (value == 0.0) return;
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite value generated during LU factorization");
            }
            candidates.push_back(row_id);
            candidate_values.push_back(value);
        };
        // The column lists hold sparse rows. A setup-dense row is never listed:
        // it is in dense_rows from its first stored column on, and is found by
        // the dense loop below or, inside a panel, as a member in the block.
        // A row converted during the factorization stays listed by its sparse
        // entries; it is examined once, through the bounded read, and the
        // dense loop skips rows already seen. Inside a panel, members are
        // examined in the block instead, and no other dense row can have a
        // nonzero in a panel column.
        for (const index_t row_id : active_columns[static_cast<std::size_t>(pivot_col)]) {
            if (panel_active && panel.is_member(row_id)) continue;
            if (seen[static_cast<std::size_t>(row_id)] == pivot_col) continue;
            seen[static_cast<std::size_t>(row_id)] = pivot_col;
            if (position_of_row[static_cast<std::size_t>(row_id)] < pivot_col) continue;
            const auto& row = rows[static_cast<std::size_t>(row_id)];
            const double value = work_row_value(row, pivot_col);
            if (panel_active && row.dense_mode && value != 0.0) {
                throw std::logic_error("dense LU panel: a dense candidate is not a panel member");
            }
            add_candidate(row_id, value);
        }
        sparse_lu_panels::DensePanel::Candidate block_candidate;
        if (panel_active) {
            block_candidate = panel.scan_column(pivot_col);
        } else {
            for (const index_t row_id : dense_rows) {
                if (seen[static_cast<std::size_t>(row_id)] == pivot_col) continue;
                seen[static_cast<std::size_t>(row_id)] = pivot_col;
                if (position_of_row[static_cast<std::size_t>(row_id)] < pivot_col) continue;
                const auto& row = rows[static_cast<std::size_t>(row_id)];
                // Columns outside a dense row's storage are exact zeros, and a
                // released pivot row has empty storage.
                if (!row.dense_mode || pivot_col < row.dense_offset ||
                    pivot_col >= row.dense_end()) {
                    continue;
                }
                add_candidate(row_id,
                    row.dense[static_cast<std::size_t>(pivot_col - row.dense_offset)]);
            }
        }
        if (candidates.empty() && block_candidate.count == 0) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(pivot_col));
        }

        // Largest magnitude, ties to the smaller current position. The block
        // maximum merges by the same rule, so the choice does not depend on
        // which list holds a candidate.
        index_t maximum_row = -1;
        index_t maximum_position = 0;
        double maximum_value = 0.0;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const double magnitude = std::abs(candidate_values[index]);
            const index_t position = position_of_row[static_cast<std::size_t>(candidates[index])];
            if (magnitude > maximum_value ||
                (magnitude == maximum_value && position < maximum_position)) {
                maximum_value = magnitude;
                maximum_position = position;
                maximum_row = candidates[index];
            }
        }
        if (block_candidate.count > 0 &&
            (block_candidate.magnitude > maximum_value ||
             (block_candidate.magnitude == maximum_value &&
              block_candidate.position < maximum_position))) {
            maximum_value = block_candidate.magnitude;
            maximum_position = block_candidate.position;
            maximum_row = block_candidate.row;
        }
        if (!std::isfinite(maximum_value)) {
            throw std::runtime_error("non-finite value generated during LU factorization");
        }
        if (maximum_value == 0.0) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(pivot_col));
        }

        index_t pivot_row = maximum_row;
        const index_t preferred_row = result->perm_c_[static_cast<std::size_t>(pivot_col)];
        if (preferred_row >= 0 && preferred_row < n &&
            position_of_row[static_cast<std::size_t>(preferred_row)] >= pivot_col) {
            const double preferred = panel_active && panel.is_member(preferred_row)
                ? panel.value_of(preferred_row, pivot_col)
                : work_row_value(rows[static_cast<std::size_t>(preferred_row)], pivot_col);
            if (preferred != 0.0 &&
                std::abs(preferred) >= diag_pivot_thresh * maximum_value) {
                pivot_row = preferred_row;
            }
        }

        const index_t old_position = position_of_row[static_cast<std::size_t>(pivot_row)];
        const index_t displaced_row = row_at_position[static_cast<std::size_t>(pivot_col)];
        std::swap(row_at_position[static_cast<std::size_t>(pivot_col)],
                  row_at_position[static_cast<std::size_t>(old_position)]);
        position_of_row[static_cast<std::size_t>(pivot_row)] = pivot_col;
        position_of_row[static_cast<std::size_t>(displaced_row)] = old_position;
        if (old_position != pivot_col) ++result->pivot_swaps_;
        if (panel_active) {
            panel.set_position(pivot_row, pivot_col);
            panel.set_position(displaced_row, old_position);
        }

        auto& pivot_work_row = rows[static_cast<std::size_t>(pivot_row)];
        const bool member_pivot = panel_active && panel.is_member(pivot_row);
        const double pivot_value = member_pivot
            ? panel.value_of(pivot_row, pivot_col)
            : work_row_value(pivot_work_row, pivot_col);
        if (pivot_value == 0.0 || !std::isfinite(pivot_value)) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(pivot_col));
        }
        auto& pivot_output = result->u_rows_[static_cast<std::size_t>(pivot_col)];
        // Inside a panel a dense pivot row is retained, unchanged, until the
        // panel's flush; its U row is built on demand or at that flush.
        const bool retained_pivot = pivot_work_row.dense_mode && panel_active;
        if (retained_pivot) {
            if (!member_pivot) {
                throw std::logic_error("dense LU panel: a dense pivot is not a panel member");
            }
            panel.pivot_dense(pivot_row, pivot_col, rows, result->u_rows_, kernels);
        } else if (pivot_work_row.dense_mode) {
            // Columns at or beyond dense_end() are exact zeros.
            sparse_lu_panels::append_dense_u_row(pivot_work_row, pivot_col, pivot_output);
        } else {
            pivot_output = std::move(pivot_work_row.active);
        }
        const auto& pivot = result->u_rows_[static_cast<std::size_t>(pivot_col)];
        if (panel_active && !retained_pivot) panel.pivot_sparse(pivot_col, pivot);

        // Inside a panel the candidate list holds sparse rows only; the dense
        // targets are the block's candidates other than the pivot.
        targets.clear();
        for (const index_t row_id : candidates) {
            if (row_id != pivot_row &&
                position_of_row[static_cast<std::size_t>(row_id)] > pivot_col) {
                targets.push_back(row_id);
            }
        }
        const index_t block_targets = block_candidate.count - (member_pivot ? 1 : 0);

        const std::uint64_t pivot_work = pivot_work_row.dense_mode
            ? static_cast<std::uint64_t>(pivot_work_row.dense_end() - pivot_col)
            : static_cast<std::uint64_t>(pivot.size());
        const std::uint64_t work = static_cast<std::uint64_t>(targets.size()) *
            pivot_work;
        if (!retained_pivot) {
            result->largest_front_ = std::max<index_t>(
                result->largest_front_, static_cast<index_t>(pivot.size()));
        }
        result->row_updates_ += static_cast<index_t>(targets.size()) + block_targets;
        if (panel_active) {
            // Dense targets: one serial block elimination, no team barrier.
            panel.eliminate(pivot_col, pivot_value, block_targets, kernels);
            // Sparse targets merge the complete U row; build it serially
            // before any worker reads it.
            if (retained_pivot && !targets.empty()) {
                panel.ensure_u_entries(pivot_col, rows, result->u_rows_, result->largest_front_);
            }
        }
        const bool run_parallel = result->factor_threads_ > 1 &&
            targets.size() > 1 && work >= parallel_work_threshold;
        const int addition_slots = run_parallel ? result->factor_threads_ : 1;
        for (int slot = 0; slot < addition_slots; ++slot) {
            additions[static_cast<std::size_t>(slot)].clear();
            dense_conversions[static_cast<std::size_t>(slot)].clear();
        }

        auto update_target = [&](std::size_t target_index, int worker_id) {
            const index_t row_id = targets[target_index];
            auto& row = rows[static_cast<std::size_t>(row_id)];
            const double coefficient = work_row_value(row, pivot_col);
            if (coefficient == 0.0) return;
            const double multiplier = coefficient / pivot_value;
            if (!std::isfinite(multiplier)) {
                throw std::runtime_error(
                    "non-finite multiplier generated during LU factorization");
            }
            if (row.dense_mode) {
                update_dense_row(
                    row, pivot_work_row, pivot, pivot_col, n, multiplier, kernels);
            } else {
                auto& local_additions = additions[static_cast<std::size_t>(worker_id)];
                const std::size_t additions_before_update = local_additions.size();
                update_sparse_row(
                    row, pivot, pivot_col, multiplier,
                    local_additions, row_id);
                if (should_use_dense(row, pivot_col + 1)) {
                    if (panel_active) {
                        panel.note_conversion(row_id, worker_id);
                    } else {
                        convert_to_dense(row, n, pivot_col + 1);
                        local_additions.resize(additions_before_update);
                        dense_conversions[static_cast<std::size_t>(worker_id)].push_back(row_id);
                    }
                }
            }
        };
        if (run_parallel) {
            ++result->parallel_steps_;
            executor.parallel_for(targets.size(), update_target);
        } else {
            for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
                update_target(target_index, 0);
            }
        }

        for (int slot = 0; slot < addition_slots; ++slot) {
            for (const Addition& addition : additions[static_cast<std::size_t>(slot)]) {
                active_columns[static_cast<std::size_t>(addition.col)].push_back(addition.row);
            }
        }
        // Worker slots depend on scheduling; sort so dense_rows does not.
        const auto conversions_begin = static_cast<std::ptrdiff_t>(dense_rows.size());
        for (int slot = 0; slot < addition_slots; ++slot) {
            const auto& local = dense_conversions[static_cast<std::size_t>(slot)];
            dense_rows.insert(dense_rows.end(), local.begin(), local.end());
            converted_dense_rows += static_cast<index_t>(local.size());
            active_dense_rows += static_cast<index_t>(local.size());
        }
        std::sort(dense_rows.begin() + conversions_begin, dense_rows.end());
        if (pivot_work_row.dense_mode) {
            // A retained pivot row is released by the panel's flush.
            if (!retained_pivot) {
                std::vector<double>().swap(pivot_work_row.dense);
                pivot_work_row.dense_mode = false;
            }
            --active_dense_rows;
        }
        std::vector<index_t>().swap(active_columns[static_cast<std::size_t>(pivot_col)]);
        if (panel_active && pivot_col + 1 == panel.end()) {
            panel.flush(rows, result->u_rows_, result->largest_front_,
                        kernels, executor, result->factor_threads_);
            for (index_t row_id : panel.conversions()) {
                auto& row = rows[static_cast<std::size_t>(row_id)];
                if (position_of_row[static_cast<std::size_t>(row_id)] >= panel.end() &&
                    should_use_dense(row, panel.end())) {
                    convert_to_dense(row, n, panel.end());
                    dense_rows.push_back(row_id);
                    ++converted_dense_rows;
                    ++active_dense_rows;
                }
            }
            dense_rows.erase(std::remove_if(dense_rows.begin(), dense_rows.end(),
                [&](index_t row_id) { return !rows[static_cast<std::size_t>(row_id)].dense_mode; }),
                dense_rows.end());
            panel_active = false;
            if (active_dense_rows == 0) dense_panels_enabled = false;
        }
    }

    result->factor_seconds_ = std::chrono::duration<double>(
        Clock::now() - factor_start).count() + result->numeric_probe_seconds_;
    result->numeric_seconds_ = result->factor_seconds_;
    result->perm_r_ = row_at_position;
    result->l_rows_.resize(static_cast<std::size_t>(n));
    result->factor_nnz_ = 0;
    for (index_t position = 0; position < n; ++position) {
        const index_t row_id = row_at_position[static_cast<std::size_t>(position)];
        auto& output = result->l_rows_[static_cast<std::size_t>(position)];
        output = std::move(rows[static_cast<std::size_t>(row_id)].lower);
        output.push_back({position, 1.0});
        result->factor_nnz_ += static_cast<index_t>(output.size());
        result->factor_nnz_ += static_cast<index_t>(
            result->u_rows_[static_cast<std::size_t>(position)].size());
    }
    result->fill_ratio_ = data_size == 0
        ? 0.0
        : static_cast<double>(result->factor_nnz_) / static_cast<double>(data_size);
    result->dense_rows_ = converted_dense_rows;
    result->lu_dense_panels_ = panel.flushes();
    result->lu_parallel_panels_ = panel.parallel_flushes();
    result->lu_panel_columns_ = panel.maximum_columns();
    result->lu_dense_update_flops_ = panel.update_flops();
    result->lu_panel_workspace_bytes_ = panel.workspace_bytes();
    result->parallel_steps_ += panel.parallel_flushes();
    finish_lu();
    return result;
}

template <typename Int>
std::shared_ptr<NativeSparseLU> dispatch_factorize(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indices_flex,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indptr_flex,
    index_t n,
    const std::string& permc_spec,
    double diag_pivot_thresh,
    int num_threads,
    bool check_finite
) {
    const auto indices = static_cast<nb::ndarray<const Int, nb::ndim<1>, nb::c_contig>>(
        indices_flex);
    const auto indptr = static_cast<nb::ndarray<const Int, nb::ndim<1>, nb::c_contig>>(
        indptr_flex);
    return factorize_typed(
        data.data(), indices.data(), indptr.data(),
        static_cast<index_t>(data.size()), static_cast<index_t>(indices.size()),
        static_cast<index_t>(indptr.size()), n, permc_spec, diag_pivot_thresh,
        num_threads, check_finite);
}

} // namespace

std::shared_ptr<NativeSparseLU> sparse_lu_factorize(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indptr,
    index_t n,
    const std::string& permc_spec,
    double diag_pivot_thresh,
    int num_threads,
    bool check_finite
) {
    if (indices.dtype() != indptr.dtype()) {
        throw nb::type_error("indices and indptr must have the same dtype");
    }
    if (indices.dtype() == nb::dtype<std::int32_t>()) {
        return dispatch_factorize<std::int32_t>(
            data, indices, indptr, n, permc_spec, diag_pivot_thresh,
            num_threads, check_finite);
    }
    if (indices.dtype() == nb::dtype<std::int64_t>()) {
        return dispatch_factorize<std::int64_t>(
            data, indices, indptr, n, permc_spec, diag_pivot_thresh,
            num_threads, check_finite);
    }
    throw nb::type_error("indices and indptr must use int32 or int64");
}

void NativeSparseLU::solve_inplace(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig>& rhs,
    const std::string& trans,
    int num_threads
) const {
    if (static_cast<index_t>(rhs.shape(0)) != n_) {
        throw nb::value_error("right-hand side row count does not match factorization");
    }
    const std::string operation = uppercase(trans);
    if (operation != "N" && operation != "T" && operation != "H") {
        throw nb::value_error("trans must be 'N', 'T', or 'H'");
    }

    const index_t nrhs = static_cast<index_t>(rhs.shape(1));
    if (n_ == 0 || nrhs == 0) return;
    double* output = rhs.data();
    std::vector<double> work(static_cast<std::size_t>(n_ * nrhs));
    const int requested_threads = resolve_threads(num_threads);
    constexpr index_t block_width = 4;
    const index_t blocks = (nrhs + block_width - 1) / block_width;
    const int use_threads = std::min<int>(requested_threads, static_cast<int>(blocks));

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(use_threads) if(use_threads > 1)
#endif
    for (index_t block = 0; block < blocks; ++block) {
        const index_t first_rhs = block * block_width;
        const index_t last_rhs = std::min(nrhs, first_rhs + block_width);

        if (operation == "N") {
            for (index_t row = 0; row < n_; ++row) {
                const index_t original = perm_r_[static_cast<std::size_t>(row)];
                for (index_t col = first_rhs; col < last_rhs; ++col) {
                    work[static_cast<std::size_t>(row * nrhs + col)] =
                        row_scale_[static_cast<std::size_t>(original)] *
                        output[static_cast<std::size_t>(original * nrhs + col)];
                }
            }

            if (column_storage_) {
                if (symmetric_ldlt_) {
                    for (index_t factor_col = 0; factor_col < n_; ++factor_col) {
                        const index_t first =
                            l_col_ptr_[static_cast<std::size_t>(factor_col)];
                        const index_t last =
                            l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        for (index_t p = first; p < last; ++p) {
                            const index_t row =
                                l_row_indices_[static_cast<std::size_t>(p)];
                            const double value = l_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(row * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(
                                        factor_col * nrhs + col)];
                            }
                        }
                    }
                    for (index_t factor_col = 0; factor_col < n_; ++factor_col) {
                        const double diagonal =
                            ldlt_diagonal_[static_cast<std::size_t>(factor_col)];
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(factor_col * nrhs + col)] /=
                                diagonal;
                        }
                    }
                    for (index_t factor_col = n_; factor_col-- > 0;) {
                        const index_t first =
                            l_col_ptr_[static_cast<std::size_t>(factor_col)];
                        const index_t last =
                            l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        for (index_t p = first; p < last; ++p) {
                            const index_t row =
                                l_row_indices_[static_cast<std::size_t>(p)];
                            const double value = l_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(factor_col * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(row * nrhs + col)];
                            }
                        }
                    }
                } else {
                    for (index_t factor_col = 0; factor_col < n_; ++factor_col) {
                        const index_t first =
                            l_col_ptr_[static_cast<std::size_t>(factor_col)] + 1;
                        const index_t last =
                            l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        for (index_t p = first; p < last; ++p) {
                            const index_t row =
                                l_row_indices_[static_cast<std::size_t>(p)];
                            const double value = l_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(row * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(
                                        factor_col * nrhs + col)];
                            }
                        }
                    }
                    for (index_t factor_col = n_; factor_col-- > 0;) {
                        const index_t first =
                            u_col_ptr_[static_cast<std::size_t>(factor_col)];
                        const index_t last =
                            u_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        if (last <= first ||
                            u_row_indices_[static_cast<std::size_t>(last - 1)] !=
                                factor_col ||
                            u_values_[static_cast<std::size_t>(last - 1)] == 0.0) {
                            throw std::runtime_error(
                                "factor contains a zero U diagonal");
                        }
                        const double diagonal =
                            u_values_[static_cast<std::size_t>(last - 1)];
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(factor_col * nrhs + col)] /=
                                diagonal;
                        }
                        for (index_t p = first; p + 1 < last; ++p) {
                            const index_t row =
                                u_row_indices_[static_cast<std::size_t>(p)];
                            const double value = u_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(row * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(
                                        factor_col * nrhs + col)];
                            }
                        }
                    }
                }
            } else {
                for (index_t row = 0; row < n_; ++row) {
                    const auto& lower = l_rows_[static_cast<std::size_t>(row)];
                    for (const Entry& entry : lower) {
                        if (entry.col >= row) break;
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(row * nrhs + col)] -=
                                entry.value * work[static_cast<std::size_t>(entry.col * nrhs + col)];
                        }
                    }
                }

                for (index_t row = n_; row-- > 0;) {
                    const auto& upper = u_rows_[static_cast<std::size_t>(row)];
                    const Entry* diagonal = find_entry(upper, row);
                    if (diagonal == nullptr || diagonal->value == 0.0) {
                        throw std::runtime_error("factor contains a zero U diagonal");
                    }
                    for (const Entry& entry : upper) {
                        if (entry.col <= row) continue;
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(row * nrhs + col)] -=
                                entry.value * work[static_cast<std::size_t>(entry.col * nrhs + col)];
                        }
                    }
                    for (index_t col = first_rhs; col < last_rhs; ++col) {
                        work[static_cast<std::size_t>(row * nrhs + col)] /= diagonal->value;
                    }
                }
            }

            for (index_t col_index = 0; col_index < n_; ++col_index) {
                const index_t original = perm_c_[static_cast<std::size_t>(col_index)];
                for (index_t col = first_rhs; col < last_rhs; ++col) {
                    output[static_cast<std::size_t>(original * nrhs + col)] =
                        col_scale_[static_cast<std::size_t>(original)] *
                        work[static_cast<std::size_t>(col_index * nrhs + col)];
                }
            }
        } else {
            for (index_t col_index = 0; col_index < n_; ++col_index) {
                const index_t original = perm_c_[static_cast<std::size_t>(col_index)];
                for (index_t col = first_rhs; col < last_rhs; ++col) {
                    work[static_cast<std::size_t>(col_index * nrhs + col)] =
                        col_scale_[static_cast<std::size_t>(original)] *
                        output[static_cast<std::size_t>(original * nrhs + col)];
                }
            }

            if (column_storage_) {
                if (symmetric_ldlt_) {
                    for (index_t factor_col = 0; factor_col < n_; ++factor_col) {
                        const index_t first =
                            l_col_ptr_[static_cast<std::size_t>(factor_col)];
                        const index_t last =
                            l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        for (index_t p = first; p < last; ++p) {
                            const index_t row =
                                l_row_indices_[static_cast<std::size_t>(p)];
                            const double value = l_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(row * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(
                                        factor_col * nrhs + col)];
                            }
                        }
                    }
                    for (index_t factor_col = 0; factor_col < n_; ++factor_col) {
                        const double diagonal =
                            ldlt_diagonal_[static_cast<std::size_t>(factor_col)];
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(factor_col * nrhs + col)] /=
                                diagonal;
                        }
                    }
                    for (index_t factor_col = n_; factor_col-- > 0;) {
                        const index_t first =
                            l_col_ptr_[static_cast<std::size_t>(factor_col)];
                        const index_t last =
                            l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        for (index_t p = first; p < last; ++p) {
                            const index_t row =
                                l_row_indices_[static_cast<std::size_t>(p)];
                            const double value = l_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(factor_col * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(row * nrhs + col)];
                            }
                        }
                    }
                } else {
                    for (index_t factor_col = 0; factor_col < n_; ++factor_col) {
                        const index_t first =
                            u_col_ptr_[static_cast<std::size_t>(factor_col)];
                        const index_t last =
                            u_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        if (last <= first ||
                            u_values_[static_cast<std::size_t>(last - 1)] == 0.0) {
                            throw std::runtime_error(
                                "factor contains a zero U diagonal");
                        }
                        for (index_t p = first; p + 1 < last; ++p) {
                            const index_t row =
                                u_row_indices_[static_cast<std::size_t>(p)];
                            const double value =
                                u_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(factor_col * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(
                                        row * nrhs + col)];
                            }
                        }
                        const double diagonal =
                            u_values_[static_cast<std::size_t>(last - 1)];
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(factor_col * nrhs + col)] /=
                                diagonal;
                        }
                    }
                    for (index_t factor_col = n_; factor_col-- > 0;) {
                        const index_t first =
                            l_col_ptr_[static_cast<std::size_t>(factor_col)] + 1;
                        const index_t last =
                            l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
                        for (index_t p = first; p < last; ++p) {
                            const index_t row =
                                l_row_indices_[static_cast<std::size_t>(p)];
                            const double value = l_values_[static_cast<std::size_t>(p)];
                            for (index_t col = first_rhs; col < last_rhs; ++col) {
                                work[static_cast<std::size_t>(factor_col * nrhs + col)] -=
                                    value * work[static_cast<std::size_t>(
                                        row * nrhs + col)];
                            }
                        }
                    }
                }
            } else {
                // U^T is lower triangular. Scatter each solved component forward.
                for (index_t row = 0; row < n_; ++row) {
                    const auto& upper = u_rows_[static_cast<std::size_t>(row)];
                    const Entry* diagonal = find_entry(upper, row);
                    if (diagonal == nullptr || diagonal->value == 0.0) {
                        throw std::runtime_error("factor contains a zero U diagonal");
                    }
                    for (index_t col = first_rhs; col < last_rhs; ++col) {
                        work[static_cast<std::size_t>(row * nrhs + col)] /= diagonal->value;
                    }
                    for (const Entry& entry : upper) {
                        if (entry.col <= row) continue;
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(entry.col * nrhs + col)] -=
                                entry.value * work[static_cast<std::size_t>(row * nrhs + col)];
                        }
                    }
                }

                // L^T is unit upper triangular.
                for (index_t row = n_; row-- > 0;) {
                    const auto& lower = l_rows_[static_cast<std::size_t>(row)];
                    for (const Entry& entry : lower) {
                        if (entry.col >= row) break;
                        for (index_t col = first_rhs; col < last_rhs; ++col) {
                            work[static_cast<std::size_t>(entry.col * nrhs + col)] -=
                                entry.value * work[static_cast<std::size_t>(row * nrhs + col)];
                        }
                    }
                }
            }

            for (index_t row = 0; row < n_; ++row) {
                const index_t original = perm_r_[static_cast<std::size_t>(row)];
                for (index_t col = first_rhs; col < last_rhs; ++col) {
                    output[static_cast<std::size_t>(original * nrhs + col)] =
                        row_scale_[static_cast<std::size_t>(original)] *
                        work[static_cast<std::size_t>(row * nrhs + col)];
                }
            }
        }
    }
}

std::tuple<std::vector<double>, std::vector<index_t>, std::vector<index_t>>
NativeSparseLU::export_l_csr() const {
    if (column_storage_) {
        if (symmetric_ldlt_) {
            std::vector<double> values;
            std::vector<index_t> rows;
            std::vector<index_t> column_ptr(
                static_cast<std::size_t>(n_ + 1), 0);
            values.reserve(static_cast<std::size_t>(l_values_.size() + n_));
            rows.reserve(static_cast<std::size_t>(l_row_indices_.size() + n_));
            for (index_t col = 0; col < n_; ++col) {
                column_ptr[static_cast<std::size_t>(col)] =
                    static_cast<index_t>(values.size());
                rows.push_back(col);
                values.push_back(1.0);
                const index_t first = l_col_ptr_[static_cast<std::size_t>(col)];
                const index_t last = l_col_ptr_[static_cast<std::size_t>(col + 1)];
                for (index_t p = first; p < last; ++p) {
                    rows.push_back(l_row_indices_[static_cast<std::size_t>(p)]);
                    values.push_back(l_values_[static_cast<std::size_t>(p)]);
                }
            }
            column_ptr[static_cast<std::size_t>(n_)] =
                static_cast<index_t>(values.size());
            return csc_to_csr(n_, column_ptr, rows, values);
        }
        return csc_to_csr(n_, l_col_ptr_, l_row_indices_, l_values_);
    }
    std::vector<double> data;
    std::vector<index_t> indices;
    std::vector<index_t> indptr(static_cast<std::size_t>(n_ + 1), 0);
    data.reserve(static_cast<std::size_t>(factor_nnz_));
    indices.reserve(static_cast<std::size_t>(factor_nnz_));
    for (index_t row = 0; row < n_; ++row) {
        for (const Entry& entry : l_rows_[static_cast<std::size_t>(row)]) {
            indices.push_back(entry.col);
            data.push_back(entry.value);
        }
        indptr[static_cast<std::size_t>(row + 1)] = static_cast<index_t>(data.size());
    }
    return {std::move(data), std::move(indices), std::move(indptr)};
}

std::tuple<std::vector<double>, std::vector<index_t>, std::vector<index_t>>
NativeSparseLU::export_u_csr() const {
    if (column_storage_) {
        if (symmetric_ldlt_) {
            std::vector<double> data;
            std::vector<index_t> indices;
            std::vector<index_t> indptr(static_cast<std::size_t>(n_ + 1), 0);
            data.reserve(static_cast<std::size_t>(l_values_.size() + n_));
            indices.reserve(static_cast<std::size_t>(l_row_indices_.size() + n_));
            for (index_t row = 0; row < n_; ++row) {
                const double diagonal =
                    ldlt_diagonal_[static_cast<std::size_t>(row)];
                indices.push_back(row);
                data.push_back(diagonal);
                const index_t first = l_col_ptr_[static_cast<std::size_t>(row)];
                const index_t last = l_col_ptr_[static_cast<std::size_t>(row + 1)];
                for (index_t p = first; p < last; ++p) {
                    indices.push_back(l_row_indices_[static_cast<std::size_t>(p)]);
                    data.push_back(
                        diagonal * l_values_[static_cast<std::size_t>(p)]);
                }
                indptr[static_cast<std::size_t>(row + 1)] =
                    static_cast<index_t>(data.size());
            }
            return {std::move(data), std::move(indices), std::move(indptr)};
        }
        return csc_to_csr(n_, u_col_ptr_, u_row_indices_, u_values_);
    }
    std::vector<double> data;
    std::vector<index_t> indices;
    std::vector<index_t> indptr(static_cast<std::size_t>(n_ + 1), 0);
    data.reserve(static_cast<std::size_t>(factor_nnz_));
    indices.reserve(static_cast<std::size_t>(factor_nnz_));
    for (index_t row = 0; row < n_; ++row) {
        for (const Entry& entry : u_rows_[static_cast<std::size_t>(row)]) {
            indices.push_back(entry.col);
            data.push_back(entry.value);
        }
        indptr[static_cast<std::size_t>(row + 1)] = static_cast<index_t>(data.size());
    }
    return {std::move(data), std::move(indices), std::move(indptr)};
}

nb::dict NativeSparseLU::stats() const {
    nb::dict output;
    output["ordering"] = ordering_;
    output["ordering_backend"] = ordering_backend_;
    output["numeric_engine"] = numeric_engine_;
    output["dense_kernel"] = dense_kernel_;
    output["input_nnz"] = input_nnz_;
    output["factor_nnz"] = factor_nnz_;
    output["fill_ratio"] = fill_ratio_;
    output["ordering_seconds"] = ordering_seconds_;
    output["factor_seconds"] = factor_seconds_;
    output["symbolic_seconds"] = symbolic_seconds_;
    output["numeric_seconds"] = numeric_seconds_;
    output["assembly_seconds"] = assembly_seconds_;
    output["num_threads"] = factor_threads_;
    output["threading_backend"] =
        ((!column_storage_ && factor_threads_ > 1) || parallel_steps_ > 0)
            ? "std::thread" : "serial";
    output["parallel_steps"] = parallel_steps_;
    output["pivot_swaps"] = pivot_swaps_;
    output["row_updates"] = row_updates_;
    output["largest_front"] = largest_front_;
    output["dense_rows"] = dense_rows_;
    output["lu_dense_panels"] = lu_dense_panels_;
    output["lu_parallel_panels"] = lu_parallel_panels_;
    output["lu_panel_columns"] = lu_panel_columns_;
    output["lu_dense_update_flops"] = lu_dense_update_flops_;
    output["lu_panel_workspace_bytes"] = lu_panel_workspace_bytes_;
    output["supernodes"] = supernodes_;
    output["max_supernode_width"] = max_supernode_width_;
    output["parallel_panels"] = parallel_panels_;
    output["dense_update_flops"] = dense_update_flops_;
    output["peak_workspace_bytes"] = peak_workspace_bytes_;
    output["spd_panel_entries"] = spd_panel_entries_;
    output["failed_spd_seconds"] = failed_spd_seconds_;
    output["spd_source_seconds"] = spd_source_seconds_;
    output["spd_panel_seconds"] = spd_panel_seconds_;
    output["spd_pack_seconds"] = spd_pack_seconds_;
    output["spd_subtrees"] = spd_subtrees_;
    output["spd_subtree_seconds"] = spd_subtree_seconds_;
    output["ordering_estimated_l_nnz"] = ordering_estimated_l_nnz_;
    output["ordering_estimated_max_front"] = ordering_estimated_max_front_;
    output["numeric_probe_columns"] = numeric_probe_columns_;
    output["numeric_probe_max_front"] = numeric_probe_max_front_;
    output["numeric_probe_updates"] = numeric_probe_updates_;
    output["numeric_probe_seconds"] = numeric_probe_seconds_;
    output["numeric_probe_fallback"] = numeric_probe_fallback_;
    return output;
}

void bind_sparse_lu(nb::module_& module) {
    nb::class_<NativeSparseLU>(module, "_NativeSparseLU")
        .def("_solve_inplace", &NativeSparseLU::solve_inplace,
             nb::arg("rhs").noconvert(), nb::arg("trans") = "N",
             nb::arg("num_threads") = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("_export_l_csr", &NativeSparseLU::export_l_csr)
        .def("_export_u_csr", &NativeSparseLU::export_u_csr)
        .def("_perm_r", &NativeSparseLU::perm_r)
        .def("_perm_c", &NativeSparseLU::perm_c)
        .def("_row_scale", &NativeSparseLU::row_scale)
        .def("_col_scale", &NativeSparseLU::col_scale)
        .def("_stats", &NativeSparseLU::stats)
        .def_prop_ro("size", &NativeSparseLU::size)
        .def_prop_ro("nnz", &NativeSparseLU::factor_nnz);

    module.def("_splu_factorize", &sparse_lu_factorize,
        nb::arg("data"), nb::arg("indices"), nb::arg("indptr"), nb::arg("n"),
        nb::arg("permc_spec") = "COLAMD", nb::arg("diag_pivot_thresh") = 1.0,
        nb::arg("num_threads") = 0, nb::arg("check_finite") = true,
        nb::call_guard<nb::gil_scoped_release>());
}
