#include "sparse_lu.hpp"
#include "sparse_ldlt.hpp"
#include "sparse_lu_panels.hpp"

#include "amd.h"
#include "colamd.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
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

struct WorkRow {
    std::vector<Entry> active;
    std::vector<Entry> scratch;
    std::vector<Entry> lower;
    std::vector<double> dense;
    index_t dense_offset = 0;
    bool dense_mode = false;
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

    template <typename Function>
    void parallel_for(std::size_t count, Function&& function) {
        if (count == 0) return;
        if (thread_count_ == 1) {
            for (std::size_t index = 0; index < count; ++index) function(index, 0);
            return;
        }
        start_workers();
        {
            std::lock_guard lock(job_mutex_);
            count_ = count;
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
    std::size_t generation_ = 0;
    std::size_t completed_workers_ = 0;
    bool stopping_ = false;
    std::exception_ptr worker_error_;
};

template <typename Int>
void validate_csc(
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t data_size,
    index_t indices_size,
    index_t indptr_size,
    index_t n,
    bool check_finite
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

[[nodiscard]] bool should_use_dense(
    const WorkRow& row,
    index_t n,
    index_t first_col
) {
    const index_t remaining = n - first_col;
    return !row.dense_mode && remaining >= 64 && row.active.size() >= 64 &&
        static_cast<std::uint64_t>(row.active.size()) * 4 >=
            static_cast<std::uint64_t>(remaining);
}

void convert_to_dense(WorkRow& row, index_t n, index_t first_col) {
    row.dense_offset = first_col;
    row.dense.assign(static_cast<std::size_t>(n - first_col), 0.0);
    for (const Entry& entry : row.active) {
        if (entry.col >= first_col) {
            row.dense[static_cast<std::size_t>(entry.col - first_col)] = entry.value;
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

void update_dense_row(
    WorkRow& row,
    const WorkRow& pivot_row,
    const std::vector<Entry>& pivot,
    index_t pivot_col,
    index_t n,
    double multiplier
) {
    row.lower.push_back({pivot_col, multiplier});
    row.dense[static_cast<std::size_t>(pivot_col - row.dense_offset)] = 0.0;

    if (pivot_row.dense_mode) {
        const index_t first = std::max(pivot_col + 1, pivot_row.dense_offset);
        for (index_t col = first; col < n; ++col) {
            const double source = pivot_row.dense[
                static_cast<std::size_t>(col - pivot_row.dense_offset)];
            if (source == 0.0) continue;
            double& target = row.dense[
                static_cast<std::size_t>(col - row.dense_offset)];
            target -= multiplier * source;
            if (!std::isfinite(target)) {
                throw std::runtime_error(
                    "non-finite value generated during LU factorization");
            }
        }
        return;
    }

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

void update_dense_panel_row(
    WorkRow& row, const std::vector<Entry>& pivot,
    index_t pivot_col, index_t panel_end, double multiplier
) {
    row.lower.push_back({pivot_col, multiplier});
    row.dense[static_cast<std::size_t>(pivot_col - row.dense_offset)] = 0.0;
    auto source = std::upper_bound(pivot.begin(), pivot.end(), pivot_col,
        [](index_t col, const Entry& entry) { return col < entry.col; });
    for (; source != pivot.end() && source->col < panel_end; ++source) {
        double& target = row.dense[static_cast<std::size_t>(source->col - row.dense_offset)];
        target -= multiplier * source->value;
        if (!std::isfinite(target)) {
            throw std::runtime_error("non-finite value generated during LU factorization");
        }
    }
}

template <typename Int>
[[nodiscard]] bool is_exactly_symmetric(
    const double* data,
    const Int* indices,
    const Int* indptr,
    index_t n
) {
    for (index_t col = 0; col < n; ++col) {
        for (index_t p = static_cast<index_t>(indptr[col]);
             p < static_cast<index_t>(indptr[col + 1]); ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
            const Int* first = indices + static_cast<index_t>(indptr[row]);
            const Int* last = indices + static_cast<index_t>(indptr[row + 1]);
            const Int needle = static_cast<Int>(col);
            const Int* transpose = std::lower_bound(first, last, needle);
            if (transpose == last || *transpose != needle) return false;
            const index_t transpose_position = static_cast<index_t>(transpose - indices);
            // LDL.T uses only one triangle. Without a caller symmetry assumption,
            // accepting unequal entries would factor a different matrix.
            if (data[p] != data[transpose_position]) return false;
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
    index_t n
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
    std::vector<index_t> strict_rows(static_cast<std::size_t>(strict_nnz));
    std::vector<double> strict_values(static_cast<std::size_t>(strict_nnz));
    std::vector<double> diagonal(static_cast<std::size_t>(n), 0.0);
    sparse_ldlt::Pattern nodes;
    bool blocked = false;
    // Narrow fronts retain the single symbolic count pass and simplicial
    // numeric path. Parent/count adjacency bounds fundamental-node width and
    // internal update work without constructing its pattern; exact tail
    // equality can only split these runs. Wide work must also amortize driver
    // overhead across the whole matrix, including disconnected narrow nodes.
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
    double possible_work = 0.0;
    for (index_t first = 0; first < n;) {
        index_t last = first + 1;
        while (last < n && parent[static_cast<std::size_t>(last - 1)] == last &&
               column_counts[static_cast<std::size_t>(last - 1)] ==
                   column_counts[static_cast<std::size_t>(last)] + 1) {
            ++last;
        }
        if (last - first >= minimum_blocked_width) {
            possible_work += node_work(last - first,
                1 + column_counts[static_cast<std::size_t>(first)]);
        }
        first = last;
    }
    if (possible_work > 0.0 && possible_work >= required_work) {
        std::fill(flags.begin(), flags.end(), -1);
        std::fill(column_counts.begin(), column_counts.end(), 0);
        for (index_t k = 0; k < n; ++k) {
            flags[static_cast<std::size_t>(k)] = k;
            const index_t original_col = result->perm_c_[static_cast<std::size_t>(k)];
            for (index_t p = static_cast<index_t>(indptr[original_col]);
                 p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
                index_t node = inverse_permutation[static_cast<std::size_t>(indices[p])];
                if (node >= k) continue;
                while (flags[static_cast<std::size_t>(node)] != k) {
                    const index_t destination = strict_col_ptr[static_cast<std::size_t>(node)] +
                        column_counts[static_cast<std::size_t>(node)]++;
                    if (destination >= strict_col_ptr[static_cast<std::size_t>(node + 1)]) {
                        throw std::logic_error("inconsistent symmetric symbolic counts");
                    }
                    strict_rows[static_cast<std::size_t>(destination)] = k;
                    flags[static_cast<std::size_t>(node)] = k;
                    node = parent[static_cast<std::size_t>(node)];
                }
            }
        }
        nodes = sparse_ldlt::find_nodes(parent, strict_col_ptr, strict_rows);
        double useful_work = 0.0;
        for (std::size_t node = 0; node + 1 < nodes.nodes.size(); ++node) {
            const index_t first = nodes.nodes[node];
            const index_t width = nodes.nodes[node + 1] - first;
            if (width >= minimum_blocked_width) {
                useful_work += node_work(width, 1 + strict_col_ptr[static_cast<std::size_t>(first + 1)] -
                    strict_col_ptr[static_cast<std::size_t>(first)]);
            }
        }
        blocked = useful_work >= required_work;
        if (blocked) sparse_ldlt::build_row_incidence(nodes, strict_col_ptr, strict_rows);
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
    if (blocked) {
        FactorExecutor executor(result->factor_threads_);
        const bool succeeded = sparse_ldlt::factorize(
            data, indices, indptr, result->perm_c_, inverse_permutation,
            symmetric_scale, strict_col_ptr, strict_rows, nodes, strict_values,
            diagonal, executor, result->factor_threads_, blocked_info, base_workspace);
        result->peak_workspace_bytes_ = blocked_info.workspace_bytes;
        if (!succeeded) return false;
        numeric_updates = blocked_info.update_flops / 2;
    } else for (index_t k = 0; k < n; ++k) {
        index_t top = n;
        flags[static_cast<std::size_t>(k)] = k;
        accumulator[static_cast<std::size_t>(k)] = 0.0;
        const index_t original_col = result->perm_c_[static_cast<std::size_t>(k)];
        for (index_t p = static_cast<index_t>(indptr[original_col]);
             p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
            const index_t original_row = static_cast<index_t>(indices[p]);
            index_t node = inverse_permutation[static_cast<std::size_t>(original_row)];
            if (node > k) continue;
            accumulator[static_cast<std::size_t>(node)] += data[p] *
                symmetric_scale[static_cast<std::size_t>(original_row)] *
                symmetric_scale[static_cast<std::size_t>(original_col)];

            index_t length = 0;
            while (flags[static_cast<std::size_t>(node)] != k) {
                path[static_cast<std::size_t>(length++)] = node;
                flags[static_cast<std::size_t>(node)] = k;
                node = parent[static_cast<std::size_t>(node)];
            }
            while (length > 0) {
                pattern[static_cast<std::size_t>(--top)] =
                    path[static_cast<std::size_t>(--length)];
            }
        }

        double pivot = accumulator[static_cast<std::size_t>(k)];
        accumulator[static_cast<std::size_t>(k)] = 0.0;
        for (index_t position = top; position < n; ++position) {
            const index_t source = pattern[static_cast<std::size_t>(position)];
            const double source_value = accumulator[static_cast<std::size_t>(source)];
            accumulator[static_cast<std::size_t>(source)] = 0.0;
            const index_t first = strict_col_ptr[static_cast<std::size_t>(source)];
            const index_t last = first + column_counts[static_cast<std::size_t>(source)];
            numeric_updates += static_cast<std::uint64_t>(last - first);
            for (index_t q = first; q < last; ++q) {
                accumulator[static_cast<std::size_t>(
                    strict_rows[static_cast<std::size_t>(q)])] -=
                    strict_values[static_cast<std::size_t>(q)] * source_value;
            }
            const double denominator = diagonal[static_cast<std::size_t>(source)];
            if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
            const double multiplier = source_value / denominator;
            if (!std::isfinite(multiplier)) return false;
            pivot -= multiplier * source_value;
            const index_t destination = last;
            strict_rows[static_cast<std::size_t>(destination)] = k;
            strict_values[static_cast<std::size_t>(destination)] = multiplier;
            ++column_counts[static_cast<std::size_t>(source)];
        }
        if (!(pivot > 0.0) || !std::isfinite(pivot)) return false;
        diagonal[static_cast<std::size_t>(k)] = pivot;
    }
    result->numeric_seconds_ = std::chrono::duration<double>(
        Clock::now() - numeric_start).count();

    if (nodes.nodes.empty()) {
        // Simplicial factors only need diagnostics, not an O(n) node list.
        nodes = sparse_ldlt::find_nodes(parent, strict_col_ptr, strict_rows, false);
    }

    const auto assembly_start = Clock::now();
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
    result->supernodes_ = nodes.node_count;
    result->max_supernode_width_ = nodes.maximum_width;
    result->largest_front_ = nodes.maximum_front;
    result->parallel_panels_ = blocked_info.parallel_panels;
    result->parallel_steps_ = blocked_info.parallel_panels;
    result->dense_update_flops_ = blocked_info.update_flops;
    result->spd_panel_entries_ = blocked_info.maximum_panel_entries;
    result->spd_source_seconds_ = blocked_info.source_seconds;
    result->spd_panel_seconds_ = blocked_info.panel_seconds;
    result->spd_pack_seconds_ = blocked_info.pack_seconds;
    if (!blocked) result->peak_workspace_bytes_ += nodes.nodes.capacity() * sizeof(index_t);
    result->assembly_seconds_ = std::chrono::duration<double>(
        Clock::now() - assembly_start).count();
    result->factor_seconds_ = result->symbolic_seconds_ +
        result->numeric_seconds_ + result->assembly_seconds_;
    return true;
}

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

    const double estimate = std::max<double>(
        static_cast<double>(2 * n), result->ordering_estimated_l_nnz_);
    const std::size_t reserve = static_cast<std::size_t>(std::min<double>(
        estimate * 1.15 + static_cast<double>(n),
        static_cast<double>(std::numeric_limits<std::size_t>::max() / 4)));
    result->l_row_indices_.reserve(reserve);
    result->l_values_.reserve(reserve);
    result->u_row_indices_.reserve(reserve);
    result->u_values_.reserve(reserve);

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

    auto depth_first_reach = [&](index_t root, index_t top, index_t generation) {
        index_t head = 0;
        node_stack[0] = root;
        while (head >= 0) {
            const index_t row = node_stack[static_cast<std::size_t>(head)];
            const index_t factor_col =
                inverse_row_permutation[static_cast<std::size_t>(row)];
            if (mark[static_cast<std::size_t>(row)] != generation) {
                mark[static_cast<std::size_t>(row)] = generation;
                edge_stack[static_cast<std::size_t>(head)] = factor_col < 0
                    ? 0
                    : result->l_col_ptr_[static_cast<std::size_t>(factor_col)] + 1;
            }

            const index_t edge_end = factor_col < 0
                ? 0
                : result->l_col_ptr_[static_cast<std::size_t>(factor_col + 1)];
            bool descended = false;
            auto& edge = edge_stack[static_cast<std::size_t>(head)];
            while (edge < edge_end) {
                const index_t next = result->l_row_indices_[static_cast<std::size_t>(edge++)];
                if (mark[static_cast<std::size_t>(next)] == generation) continue;
                node_stack[static_cast<std::size_t>(++head)] = next;
                descended = true;
                break;
            }
            if (!descended) {
                reach[static_cast<std::size_t>(--top)] = row;
                --head;
            }
        }
        return top;
    };

    for (index_t factor_col = 0; factor_col < n; ++factor_col) {
        result->l_col_ptr_[static_cast<std::size_t>(factor_col)] =
            static_cast<index_t>(result->l_values_.size());
        result->u_col_ptr_[static_cast<std::size_t>(factor_col)] =
            static_cast<index_t>(result->u_values_.size());

        const index_t original_col =
            result->perm_c_[static_cast<std::size_t>(factor_col)];
        index_t top = n;
        for (index_t p = static_cast<index_t>(indptr[original_col]);
             p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
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
            values[static_cast<std::size_t>(reach[static_cast<std::size_t>(p)])] = 0.0;
        }
        for (index_t p = static_cast<index_t>(indptr[original_col]);
             p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
            values[static_cast<std::size_t>(row)] += data[p] *
                result->row_scale_[static_cast<std::size_t>(row)] *
                result->col_scale_[static_cast<std::size_t>(original_col)];
        }

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

        index_t pivot_row = -1;
        double maximum_value = 0.0;
        for (index_t p = top; p < n; ++p) {
            const index_t row = reach[static_cast<std::size_t>(p)];
            if (inverse_row_permutation[static_cast<std::size_t>(row)] >= 0) continue;
            const double magnitude = std::abs(values[static_cast<std::size_t>(row)]);
            if (magnitude > maximum_value ||
                (magnitude == maximum_value && magnitude != 0.0 &&
                 (pivot_row < 0 ||
                  position_of_row[static_cast<std::size_t>(row)] <
                  position_of_row[static_cast<std::size_t>(pivot_row)]))) {
                maximum_value = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_row < 0 || maximum_value == 0.0 || !std::isfinite(maximum_value)) {
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
        const double pivot_value = values[static_cast<std::size_t>(pivot_row)];
        if (pivot_value == 0.0 || !std::isfinite(pivot_value)) {
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
    validate_csc(data, indices, indptr, data_size, indices_size, indptr_size,
                 n, check_finite);
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

    // Equilibrate by infinity norms. Zero rows and columns are structurally singular.
    std::vector<double> row_max(static_cast<std::size_t>(n), 0.0);
    for (index_t col = 0; col < n; ++col) {
        for (index_t p = static_cast<index_t>(indptr[col]);
             p < static_cast<index_t>(indptr[col + 1]); ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
            row_max[static_cast<std::size_t>(row)] = std::max(
                row_max[static_cast<std::size_t>(row)], std::abs(data[p]));
        }
    }
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

    std::vector<double> col_max(static_cast<std::size_t>(n), 0.0);
    for (index_t col = 0; col < n; ++col) {
        for (index_t p = static_cast<index_t>(indptr[col]);
             p < static_cast<index_t>(indptr[col + 1]); ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
            col_max[static_cast<std::size_t>(col)] = std::max(
                col_max[static_cast<std::size_t>(col)],
                std::abs(data[p]) * result->row_scale_[static_cast<std::size_t>(row)]);
        }
        if (col_max[static_cast<std::size_t>(col)] == 0.0) {
            throw std::runtime_error("factor is exactly singular: matrix contains a zero column");
        }
        const double scale = 1.0 / col_max[static_cast<std::size_t>(col)];
        if (!std::isfinite(scale)) {
            throw std::runtime_error("matrix column scaling overflowed");
        }
        result->col_scale_[static_cast<std::size_t>(col)] = scale;
    }

    const auto spd_start = Clock::now();
    if (try_factorize_symmetric_positive_definite(
            result, data, indices, indptr, n)) {
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
        // Restart from pristine state if actual CSC work is unsuitable. Copy
        // only the construction state; no partial factors exist at this point.
        const auto probe_start = Clock::now();
        NativeSparseLU restart_state = *result;
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
            *result = std::move(restart_state);
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

    std::vector<WorkRow> rows(static_cast<std::size_t>(n));
    std::vector<std::vector<index_t>> active_columns(static_cast<std::size_t>(n));
    for (index_t new_col = 0; new_col < n; ++new_col) {
        const index_t original_col = result->perm_c_[static_cast<std::size_t>(new_col)];
        for (index_t p = static_cast<index_t>(indptr[original_col]);
             p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
            const index_t row = static_cast<index_t>(indices[p]);
            const double value = data[p] *
                result->row_scale_[static_cast<std::size_t>(row)] *
                result->col_scale_[static_cast<std::size_t>(original_col)];
            if (value != 0.0) {
                rows[static_cast<std::size_t>(row)].active.push_back({new_col, value});
                active_columns[static_cast<std::size_t>(new_col)].push_back(row);
            }
        }
    }

    std::vector<index_t> dense_rows;
    index_t converted_dense_rows = 0;
    index_t active_dense_rows = 0;
    for (index_t row_id = 0; row_id < n; ++row_id) {
        auto& row = rows[static_cast<std::size_t>(row_id)];
        if (should_use_dense(row, n, 0)) {
            convert_to_dense(row, n, 0);
            dense_rows.push_back(row_id);
            ++converted_dense_rows;
            ++active_dense_rows;
        }
    }

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

    for (index_t pivot_col = 0; pivot_col < n; ++pivot_col) {
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
            }
        }
        std::vector<index_t> candidates;
        candidates.reserve(
            active_columns[static_cast<std::size_t>(pivot_col)].size() +
            dense_rows.size());
        for (const index_t row_id : active_columns[static_cast<std::size_t>(pivot_col)]) {
            if (seen[static_cast<std::size_t>(row_id)] == pivot_col) continue;
            seen[static_cast<std::size_t>(row_id)] = pivot_col;
            if (position_of_row[static_cast<std::size_t>(row_id)] < pivot_col) continue;
            if (work_row_value(rows[static_cast<std::size_t>(row_id)], pivot_col) != 0.0) {
                candidates.push_back(row_id);
            }
        }
        for (const index_t row_id : dense_rows) {
            if (seen[static_cast<std::size_t>(row_id)] == pivot_col) continue;
            seen[static_cast<std::size_t>(row_id)] = pivot_col;
            if (position_of_row[static_cast<std::size_t>(row_id)] < pivot_col) continue;
            if (work_row_value(rows[static_cast<std::size_t>(row_id)], pivot_col) != 0.0) {
                candidates.push_back(row_id);
            }
        }
        if (candidates.empty()) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(pivot_col));
        }

        index_t maximum_row = candidates.front();
        double maximum_value = 0.0;
        for (const index_t row_id : candidates) {
            const double magnitude = std::abs(
                work_row_value(rows[static_cast<std::size_t>(row_id)], pivot_col));
            if (magnitude > maximum_value ||
                (magnitude == maximum_value &&
                 position_of_row[static_cast<std::size_t>(row_id)] <
                 position_of_row[static_cast<std::size_t>(maximum_row)])) {
                maximum_value = magnitude;
                maximum_row = row_id;
            }
        }
        if (maximum_value == 0.0 || !std::isfinite(maximum_value)) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(pivot_col));
        }

        index_t pivot_row = maximum_row;
        const index_t preferred_row = result->perm_c_[static_cast<std::size_t>(pivot_col)];
        if (preferred_row >= 0 && preferred_row < n &&
            position_of_row[static_cast<std::size_t>(preferred_row)] >= pivot_col) {
            const double preferred = work_row_value(
                rows[static_cast<std::size_t>(preferred_row)], pivot_col);
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

        auto& pivot_work_row = rows[static_cast<std::size_t>(pivot_row)];
        const double pivot_value = work_row_value(pivot_work_row, pivot_col);
        if (pivot_value == 0.0 || !std::isfinite(pivot_value)) {
            throw std::runtime_error(
                "factor is exactly singular at column " + std::to_string(pivot_col));
        }
        auto& pivot_output = result->u_rows_[static_cast<std::size_t>(pivot_col)];
        if (pivot_work_row.dense_mode) {
            if (panel_active) {
                panel.materialize_pivot(pivot_row, pivot_col, rows, result->u_rows_);
            }
            const index_t first = std::max(pivot_col, pivot_work_row.dense_offset);
            std::size_t nonzeros = 0;
            for (index_t col = first; col < n; ++col) {
                nonzeros += pivot_work_row.dense[
                    static_cast<std::size_t>(col - pivot_work_row.dense_offset)] != 0.0;
            }
            pivot_output.reserve(nonzeros);
            for (index_t col = first; col < n; ++col) {
                const double value = pivot_work_row.dense[
                    static_cast<std::size_t>(col - pivot_work_row.dense_offset)];
                if (value != 0.0) pivot_output.push_back({col, value});
            }
        } else {
            pivot_output = std::move(pivot_work_row.active);
        }
        const auto& pivot = result->u_rows_[static_cast<std::size_t>(pivot_col)];

        std::vector<index_t> targets;
        targets.reserve(candidates.size());
        for (const index_t row_id : candidates) {
            if (row_id != pivot_row &&
                position_of_row[static_cast<std::size_t>(row_id)] > pivot_col) {
                targets.push_back(row_id);
            }
        }

        const std::uint64_t pivot_work = pivot_work_row.dense_mode
            ? static_cast<std::uint64_t>(pivot_work_row.dense.size())
            : static_cast<std::uint64_t>(pivot.size());
        const std::uint64_t work = static_cast<std::uint64_t>(targets.size()) *
            pivot_work;
        result->largest_front_ = std::max<index_t>(
            result->largest_front_, static_cast<index_t>(pivot.size()));
        result->row_updates_ += static_cast<index_t>(targets.size());
        if (panel_active) {
            // Only sparse targets perform expensive whole-tail work per pivot.
            // Cheap dense panel coefficients run serially without a team barrier.
            std::size_t sparse_count = 0;
            for (index_t row_id : targets) {
                auto& row = rows[static_cast<std::size_t>(row_id)];
                if (!row.dense_mode) {
                    targets[sparse_count++] = row_id;
                    continue;
                }
                const double multiplier = work_row_value(row, pivot_col) / pivot_value;
                if (!std::isfinite(multiplier)) {
                    throw std::runtime_error("non-finite multiplier generated during LU factorization");
                }
                update_dense_panel_row(row, pivot, pivot_col, panel.end(), multiplier);
                panel.note_dense_row(row_id);
            }
            targets.resize(sparse_count);
        }
        const bool run_parallel = result->factor_threads_ > 1 &&
            targets.size() > 1 && (panel_active
                ? static_cast<std::uint64_t>(targets.size()) * pivot_work : work)
                    >= parallel_work_threshold;
        const int addition_slots = run_parallel ? result->factor_threads_ : 1;
        std::vector<std::vector<Addition>> additions(
            static_cast<std::size_t>(addition_slots));
        std::vector<std::vector<index_t>> dense_conversions(
            static_cast<std::size_t>(addition_slots));

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
                    row, pivot_work_row, pivot, pivot_col, n, multiplier);
            } else {
                auto& local_additions = additions[static_cast<std::size_t>(worker_id)];
                const std::size_t additions_before_update = local_additions.size();
                update_sparse_row(
                    row, pivot, pivot_col, multiplier,
                    local_additions, row_id);
                if (should_use_dense(row, n, pivot_col + 1)) {
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

        for (const auto& local : additions) {
            for (const Addition& addition : local) {
                active_columns[static_cast<std::size_t>(addition.col)].push_back(addition.row);
            }
        }
        for (const auto& local : dense_conversions) {
            dense_rows.insert(dense_rows.end(), local.begin(), local.end());
            converted_dense_rows += static_cast<index_t>(local.size());
            active_dense_rows += static_cast<index_t>(local.size());
        }
        if (pivot_work_row.dense_mode) {
            std::vector<double>().swap(pivot_work_row.dense);
            pivot_work_row.dense_mode = false;
            --active_dense_rows;
        }
        std::vector<index_t>().swap(active_columns[static_cast<std::size_t>(pivot_col)]);
        if (panel_active && pivot_col + 1 == panel.end()) {
            panel.flush(rows, position_of_row, result->u_rows_, executor, result->factor_threads_);
            for (index_t row_id : panel.conversions()) {
                auto& row = rows[static_cast<std::size_t>(row_id)];
                if (position_of_row[static_cast<std::size_t>(row_id)] >= panel.end() &&
                    should_use_dense(row, n, panel.end())) {
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
        ((!column_storage_ && factor_threads_ > 1) || parallel_panels_ > 0)
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
