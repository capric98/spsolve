#include "sparse_lu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_set>
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
std::vector<std::vector<index_t>> build_pattern_rows(
    const Int* indices,
    const Int* indptr,
    index_t n
) {
    std::vector<std::vector<index_t>> rows(static_cast<std::size_t>(n));
    for (index_t col = 0; col < n; ++col) {
        for (index_t p = static_cast<index_t>(indptr[col]);
             p < static_cast<index_t>(indptr[col + 1]); ++p) {
            rows[static_cast<std::size_t>(indices[p])].push_back(col);
        }
    }
    return rows;
}

[[nodiscard]] std::vector<index_t> eliminate_minimum_degree(
    std::vector<std::unordered_set<index_t>> adjacency,
    std::vector<index_t> dense_penalty,
    std::vector<index_t> column_count,
    index_t n,
    std::size_t directed_edges,
    std::size_t dense_limit,
    std::size_t edge_budget
) {
    struct HeapItem {
        index_t score;
        index_t count;
        index_t col;
        bool operator>(const HeapItem& other) const noexcept {
            if (score != other.score) return score > other.score;
            if (count != other.count) return count > other.count;
            return col > other.col;
        }
    };

    auto score_of = [&](index_t col) -> index_t {
        return static_cast<index_t>(adjacency[static_cast<std::size_t>(col)].size()) +
               dense_penalty[static_cast<std::size_t>(col)];
    };

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>> heap;
    std::vector<unsigned char> active(static_cast<std::size_t>(n), 1);
    for (index_t col = 0; col < n; ++col) {
        heap.push({score_of(col), column_count[static_cast<std::size_t>(col)], col});
    }

    std::vector<index_t> permutation;
    permutation.reserve(static_cast<std::size_t>(n));

    while (static_cast<index_t>(permutation.size()) < n) {
        HeapItem item{};
        bool found = false;
        while (!heap.empty()) {
            item = heap.top();
            heap.pop();
            if (!active[static_cast<std::size_t>(item.col)]) continue;
            if (item.score != score_of(item.col)) continue;
            found = true;
            break;
        }
        if (!found) {
            for (index_t col = 0; col < n; ++col) {
                if (active[static_cast<std::size_t>(col)]) {
                    item = {score_of(col), column_count[static_cast<std::size_t>(col)], col};
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            throw std::runtime_error("column ordering failed to select all columns");
        }

        const index_t pivot = item.col;
        active[static_cast<std::size_t>(pivot)] = 0;
        permutation.push_back(pivot);

        std::vector<index_t> neighbors;
        neighbors.reserve(adjacency[static_cast<std::size_t>(pivot)].size());
        for (const index_t col : adjacency[static_cast<std::size_t>(pivot)]) {
            if (active[static_cast<std::size_t>(col)]) neighbors.push_back(col);
        }
        std::sort(neighbors.begin(), neighbors.end());

        for (const index_t col : neighbors) {
            adjacency[static_cast<std::size_t>(col)].erase(pivot);
        }

        const bool materialize_fill = neighbors.size() <= dense_limit &&
            directed_edges + neighbors.size() * neighbors.size() <= edge_budget;
        if (materialize_fill) {
            for (std::size_t a = 0; a < neighbors.size(); ++a) {
                const index_t ca = neighbors[a];
                for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
                    const index_t cb = neighbors[b];
                    if (adjacency[static_cast<std::size_t>(ca)].insert(cb).second) {
                        adjacency[static_cast<std::size_t>(cb)].insert(ca);
                        directed_edges += 2;
                    }
                }
            }
        } else {
            const index_t penalty = static_cast<index_t>(neighbors.size() / 8);
            for (const index_t col : neighbors) {
                dense_penalty[static_cast<std::size_t>(col)] += penalty;
            }
        }

        adjacency[static_cast<std::size_t>(pivot)].clear();
        for (const index_t col : neighbors) {
            heap.push({score_of(col), column_count[static_cast<std::size_t>(col)], col});
        }
    }

    return permutation;
}

// A clean-room column minimum-degree implementation. It constructs the sparse
// column-intersection graph while pruning dense rows, then performs lazy exact
// degree updates subject to a memory guard. The guard deliberately degrades to
// an approximate degree rather than allowing ordering memory to become dense.
[[nodiscard]] std::vector<index_t> column_amd_order(
    const std::vector<std::vector<index_t>>& pattern_rows,
    index_t n,
    index_t nnz
) {
    if (n == 0) return {};
    const auto dense_limit = static_cast<std::size_t>(
        std::max<index_t>(16, std::min<index_t>(128,
            static_cast<index_t>(10.0 * std::sqrt(static_cast<double>(n)))))
    );
    const std::size_t edge_budget = static_cast<std::size_t>(
        std::max<index_t>(100'000, std::min<index_t>(20'000'000, 16 * nnz + 8 * n))
    );
    std::vector<std::unordered_set<index_t>> adjacency(static_cast<std::size_t>(n));
    std::vector<index_t> dense_penalty(static_cast<std::size_t>(n), 0);
    std::vector<index_t> column_count(static_cast<std::size_t>(n), 0);
    std::size_t directed_edges = 0;

    for (const auto& row : pattern_rows) {
        for (const index_t col : row) {
            ++column_count[static_cast<std::size_t>(col)];
        }
        if (row.size() > dense_limit) {
            for (const index_t col : row) {
                dense_penalty[static_cast<std::size_t>(col)] +=
                    static_cast<index_t>(std::min<std::size_t>(row.size(), dense_limit));
            }
            continue;
        }
        for (std::size_t a = 0; a < row.size(); ++a) {
            for (std::size_t b = a + 1; b < row.size(); ++b) {
                if (directed_edges + 2 > edge_budget) break;
                const index_t ca = row[a];
                const index_t cb = row[b];
                if (adjacency[static_cast<std::size_t>(ca)].insert(cb).second) {
                    adjacency[static_cast<std::size_t>(cb)].insert(ca);
                    directed_edges += 2;
                }
            }
        }
    }
    return eliminate_minimum_degree(
        std::move(adjacency), std::move(dense_penalty), std::move(column_count),
        n, directed_edges, dense_limit, edge_budget);
}

// Minimum degree on the structural graph of A + A^T is often a better choice
// for PDE-like or nearly symmetric matrices. It is also cheaper to construct
// than the column-intersection graph used above.
[[nodiscard]] std::vector<index_t> symmetric_amd_order(
    const std::vector<std::vector<index_t>>& pattern_rows,
    index_t n,
    index_t nnz
) {
    if (n == 0) return {};
    const auto dense_limit = static_cast<std::size_t>(
        std::max<index_t>(16, std::min<index_t>(128,
            static_cast<index_t>(10.0 * std::sqrt(static_cast<double>(n)))))
    );
    const std::size_t edge_budget = static_cast<std::size_t>(
        std::max<index_t>(100'000, std::min<index_t>(20'000'000, 16 * nnz + 8 * n))
    );
    std::vector<std::unordered_set<index_t>> adjacency(static_cast<std::size_t>(n));
    std::vector<index_t> dense_penalty(static_cast<std::size_t>(n), 0);
    std::vector<index_t> column_count(static_cast<std::size_t>(n), 0);
    std::size_t directed_edges = 0;

    for (index_t row = 0; row < n; ++row) {
        for (const index_t col : pattern_rows[static_cast<std::size_t>(row)]) {
            ++column_count[static_cast<std::size_t>(col)];
            if (row == col) continue;
            if (directed_edges + 2 > edge_budget) {
                ++dense_penalty[static_cast<std::size_t>(row)];
                ++dense_penalty[static_cast<std::size_t>(col)];
                continue;
            }
            if (adjacency[static_cast<std::size_t>(row)].insert(col).second) {
                adjacency[static_cast<std::size_t>(col)].insert(row);
                directed_edges += 2;
            }
        }
    }
    return eliminate_minimum_degree(
        std::move(adjacency), std::move(dense_penalty), std::move(column_count),
        n, directed_edges, dense_limit, edge_budget);
}

[[nodiscard]] const Entry* find_entry(const std::vector<Entry>& row, index_t col) {
    const auto it = std::lower_bound(row.begin(), row.end(), col,
        [](const Entry& entry, index_t needle) { return entry.col < needle; });
    return (it != row.end() && it->col == col) ? &*it : nullptr;
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

    const auto pattern_rows = build_pattern_rows(indices, indptr, n);
    const auto ordering_start = Clock::now();
    if (ordering == "NATURAL") {
        result->perm_c_.resize(static_cast<std::size_t>(n));
        std::iota(result->perm_c_.begin(), result->perm_c_.end(), index_t{0});
    } else if (ordering == "COLAMD") {
        result->perm_c_ = column_amd_order(pattern_rows, n, data_size);
    } else {
        result->perm_c_ = symmetric_amd_order(pattern_rows, n, data_size);
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
    for (index_t row_id = 0; row_id < n; ++row_id) {
        auto& row = rows[static_cast<std::size_t>(row_id)];
        if (should_use_dense(row, n, 0)) {
            convert_to_dense(row, n, 0);
            dense_rows.push_back(row_id);
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

    for (index_t pivot_col = 0; pivot_col < n; ++pivot_col) {
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
        const bool run_parallel = result->factor_threads_ > 1 &&
            targets.size() > 1 && work >= parallel_work_threshold;
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
                    convert_to_dense(row, n, pivot_col + 1);
                    local_additions.resize(additions_before_update);
                    dense_conversions[static_cast<std::size_t>(worker_id)].push_back(
                        row_id);
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
        }
        if (pivot_work_row.dense_mode) {
            std::vector<double>().swap(pivot_work_row.dense);
            pivot_work_row.dense_mode = false;
        }
        std::vector<index_t>().swap(active_columns[static_cast<std::size_t>(pivot_col)]);
    }

    result->factor_seconds_ = std::chrono::duration<double>(
        Clock::now() - factor_start).count();
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
    result->dense_rows_ = static_cast<index_t>(dense_rows.size());
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
    output["input_nnz"] = input_nnz_;
    output["factor_nnz"] = factor_nnz_;
    output["fill_ratio"] = fill_ratio_;
    output["ordering_seconds"] = ordering_seconds_;
    output["factor_seconds"] = factor_seconds_;
    output["num_threads"] = factor_threads_;
    output["threading_backend"] = factor_threads_ > 1 ? "std::thread" : "serial";
    output["parallel_steps"] = parallel_steps_;
    output["pivot_swaps"] = pivot_swaps_;
    output["row_updates"] = row_updates_;
    output["largest_front"] = largest_front_;
    output["dense_rows"] = dense_rows_;
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
