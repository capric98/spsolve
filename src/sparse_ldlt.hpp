#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dense_kernels.hpp"

// Internal, portable supernodal LDL.T kernels. The caller supplies an exact
// filled pattern and publishes the CSC factors only after all pivots succeed.
namespace sparse_ldlt {

using index_t = std::int64_t;
constexpr index_t panel_width = 32;

inline std::size_t checked_product(index_t rows, index_t columns) {
    const auto maximum = std::vector<double>().max_size();
    if (rows < 0 || columns < 0 || (columns != 0 &&
        static_cast<std::uint64_t>(rows) > maximum /
            static_cast<std::uint64_t>(columns))) {
        throw std::overflow_error("symmetric panel allocation overflowed");
    }
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
}

struct Pattern {
    std::vector<index_t> nodes;
    std::vector<index_t> row_ptr;
    std::vector<index_t> row_sources;
    std::vector<index_t> column_nodes;
    // Offset of each node's dense block in the numeric phase's single block
    // array: block_ptr[node + 1] - block_ptr[node] = height * width.
    std::vector<std::size_t> block_ptr;
    index_t node_count = 0;
    index_t maximum_width = 1;
    index_t maximum_front = 0;
};

inline Pattern find_nodes(
    const std::vector<index_t>& parent,
    const std::vector<index_t>& col_ptr,
    const std::vector<index_t>& rows,
    bool collect_nodes = true
) {
    Pattern pattern;
    const index_t n = static_cast<index_t>(parent.size());
    if (collect_nodes) pattern.nodes.push_back(0);
    for (index_t first = 0; first < n;) {
        index_t last = first + 1;
        while (last < n) {
            const index_t previous_begin = col_ptr[static_cast<std::size_t>(last - 1)];
            const index_t previous_end = col_ptr[static_cast<std::size_t>(last)];
            const index_t current_end = col_ptr[static_cast<std::size_t>(last + 1)];
            if (parent[static_cast<std::size_t>(last - 1)] != last ||
                previous_end - previous_begin != current_end - previous_end + 1 ||
                previous_begin == previous_end ||
                rows[static_cast<std::size_t>(previous_begin)] != last ||
                !std::equal(rows.begin() + previous_begin + 1,
                            rows.begin() + previous_end, rows.begin() + previous_end)) {
                break;
            }
            ++last;
        }
        pattern.maximum_width = std::max(pattern.maximum_width, last - first);
        pattern.maximum_front = std::max(pattern.maximum_front,
            1 + col_ptr[static_cast<std::size_t>(first + 1)] -
                col_ptr[static_cast<std::size_t>(first)]);
        ++pattern.node_count;
        if (collect_nodes) pattern.nodes.push_back(last);
        first = last;
    }
    return pattern;
}

inline void build_row_incidence(
    Pattern& pattern,
    const std::vector<index_t>& col_ptr,
    const std::vector<index_t>& rows
) {
    const index_t n = static_cast<index_t>(col_ptr.size()) - 1;
    pattern.column_nodes.resize(static_cast<std::size_t>(n));
    for (std::size_t node = 0; node + 1 < pattern.nodes.size(); ++node) {
        for (index_t col = pattern.nodes[node]; col < pattern.nodes[node + 1]; ++col) {
            pattern.column_nodes[static_cast<std::size_t>(col)] = static_cast<index_t>(node);
        }
    }
    pattern.row_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    for (index_t row : rows) ++pattern.row_ptr[static_cast<std::size_t>(row + 1)];
    for (index_t row = 0; row < n; ++row) {
        pattern.row_ptr[static_cast<std::size_t>(row + 1)] +=
            pattern.row_ptr[static_cast<std::size_t>(row)];
    }
    auto next = pattern.row_ptr;
    pattern.row_sources.resize(rows.size());
    for (index_t col = 0; col < n; ++col) {
        for (index_t p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const index_t destination = next[static_cast<std::size_t>(rows[p])]++;
            pattern.row_sources[static_cast<std::size_t>(destination)] = col;
        }
    }
}

// Prefix sums of the node rectangles: each node of width W and height R (its
// columns, then its tail rows) owns R * W consecutive block entries.
inline void build_block_offsets(Pattern& pattern, const std::vector<index_t>& col_ptr) {
    const auto maximum = std::vector<double>().max_size();
    pattern.block_ptr.assign(1, 0);
    pattern.block_ptr.reserve(pattern.nodes.size());
    for (std::size_t node = 0; node + 1 < pattern.nodes.size(); ++node) {
        const index_t first = pattern.nodes[node];
        const index_t width = pattern.nodes[node + 1] - first;
        const index_t height = 1 + col_ptr[static_cast<std::size_t>(first + 1)] -
            col_ptr[static_cast<std::size_t>(first)];
        const auto entries = checked_product(height, width);
        const auto total = pattern.block_ptr.back();
        if (entries > maximum - total) {
            throw std::overflow_error("symmetric panel allocation overflowed");
        }
        pattern.block_ptr.push_back(total + entries);
    }
}

struct NumericInfo {
    std::uint64_t workspace_bytes = 0;
    std::uint64_t maximum_panel_entries = 0;
    std::uint64_t update_flops = 0;
    index_t parallel_panels = 0;
    double source_seconds = 0.0;
    double panel_seconds = 0.0;
    double pack_seconds = 0.0;
};

struct NonPositiveDefinite {};

// A parallel phase (a node's contributor strips or a panel's row strips) runs
// on the worker team only when its serial time spans several dispatches:
// FactorExecutor::parallel_for costs about 20 us at 4-8 threads, and 2 MFLOP
// (2^21) is about 35 us of avx2-fma tile work. It also needs two items per
// thread, counting at most eight threads: a small team is then mostly busy,
// while a larger team does not turn serial an update that eight threads would
// share (its extra workers only cost the dispatch). This is an efficiency
// heuristic. The decision only chooses which thread computes which entries:
// every entry receives the same arithmetic on either path, so it cannot change
// any result.
[[nodiscard]] inline bool worth_parallel(std::uint64_t flops, std::size_t items, int threads) {
    constexpr int counted_threads = 8;
    return threads > 1 && flops >= (std::uint64_t{1} << 21) &&
        items >= 2 * static_cast<std::size_t>(std::min(threads, counted_threads));
}

// Prior-node fan-in. All contributions to a node are applied in one phase over
// fixed strips of fan_in_rows node-rectangle rows (aligned at multiples of
// fan_in_rows, every column), so the partition depends on the node's shape
// only. A strip applies every contribution that reaches its rows, in
// ascending contributor order, to its own entries and writes nothing else.
// Kernel calls cover at most fan_in_rows x fan_in_columns output entries, the
// size of a worker's gather block.
constexpr index_t fan_in_rows = 16;
constexpr index_t fan_in_columns = 128;

// A contribution with fewer output rows times columns than this is applied by
// sparse outer products rather than dense kernel calls. The choice is made
// from the whole contribution's size, never from its part in one strip.
constexpr std::uint64_t sparse_update_entries = 256;

// Marks a contribution that has no packed operand (a sparse update).
constexpr std::size_t sparse_update = std::numeric_limits<std::size_t>::max();

// One prior node Q's update of the current node, F[R,J] -= L[R,Q] D_Q L[J,Q].T.
// Its target positions, the ascending node-rectangle rows of R, are
// FanIn::positions[first, first + rows). Those that are node columns (J) come
// first, so the output columns are the prefix [first, first + columns) of the
// output rows, and output entry (i, j) lies on or below the diagonal exactly
// when i >= j. left is row 0 of L[R,Q] inside Q's finished block, read in
// place with row stride rank.
struct Contribution {
    const double* left;
    const double* pivots;
    index_t rank;
    index_t rows;
    index_t columns;
    // The first and last target rows, positions[first] and
    // positions[first + rows - 1].
    index_t top;
    index_t bottom;
    std::size_t first;
    // Offset of the packed right operand D_Q L[J,Q].T ([rank x columns],
    // right[k][j] = d[k] * l[j][k]) in the node's operand buffer, or
    // sparse_update.
    std::size_t right;
};

// Dense operand storage that starts on a 64-byte boundary. The allocator
// only guarantees 16 bytes, and a panel operand that started 16 bytes into a
// cache line made the dense SPD blocks' strip phase about 30% slower (starts
// 0 or 32 bytes into a line were unaffected), in whichever processes the heap
// happened to place it there, so the start is aligned by hand. Contents are
// not initialized: every entry is written before it is read.
struct AlignedDoubles {
    static constexpr std::size_t alignment = 64 / sizeof(double);
    std::unique_ptr<double[]> storage;
    double* values = nullptr;
    std::size_t capacity = 0;

    // Replace the storage with room for entries. Old contents are not kept,
    // so the old block is released first; if the new allocation throws, the
    // object is left empty (no storage, values null, capacity zero).
    void allocate(std::size_t entries) {
        if (entries > std::numeric_limits<std::size_t>::max() - alignment) {
            throw std::overflow_error("symmetric panel allocation overflowed");
        }
        storage.reset();
        values = nullptr;
        capacity = 0;
        std::unique_ptr<double[]> fresh(new double[entries + alignment - 1]);
        const auto misalignment = reinterpret_cast<std::uintptr_t>(fresh.get()) % 64;
        double* const aligned = fresh.get() +
            (misalignment == 0 ? 0 : (64 - misalignment) / sizeof(double));
        storage = std::move(fresh);
        values = aligned;
        capacity = entries + alignment - 1;
    }
};

// Packed operand storage, shared by the contributor and panel phases. It only
// grows, geometrically; old contents are never kept. capacity is updated only
// after the allocation succeeded (it is zero while the buffer is empty).
struct OperandBuffer {
    AlignedDoubles buffer;
    std::size_t capacity = 0;

    [[nodiscard]] double* reserve(std::size_t entries) {
        if (entries > capacity) {
            const std::size_t grown = std::max(entries, capacity + capacity / 2);
            capacity = 0;
            buffer.allocate(grown);
            capacity = grown;
        }
        return buffer.values;
    }
};

// The current node's contributions, reused across nodes, plus one
// fan_in_rows x fan_in_columns gather block per worker, allocated once per
// factorization.
struct FanIn {
    std::vector<Contribution> contributions;
    std::vector<index_t> positions;
    AlignedDoubles scratch;

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return sizeof(Contribution) * contributions.capacity() +
            sizeof(index_t) * positions.capacity() + sizeof(double) * scratch.capacity;
    }
};

// Apply every contribution of the node to its rectangle front (row stride
// width, height rows). operands holds room for every packed D_Q L[J,Q].T at
// its offset (parallel) or for the largest one (otherwise). In parallel, all
// operands are packed first and the worker team runs the row strips, claimed
// from the bottom up because lower rows reach more columns: a strip visits
// the contributions in ascending contributor order, finds by binary search
// the output rows [ia, ib) of each that it holds, and applies them.
// Otherwise the calling thread packs and applies one whole contribution after
// another, as one strip of all rows. Output columns at or after the strip's
// last output row (j >= ib) are above the diagonal, so a strip applies
// output columns [0, min(columns, ib)), split into parts of at most
// fan_in_rows rows by fan_in_columns columns that start at or below the
// diagonal (row index >= first column index). A part receives the same
// arithmetic the whole contribution applies to those entries:
// - a sparse update (fewer than sparse_update_entries output entries in the
//   whole contribution) runs per source column k ascending,
//   f[i][j] -= (l[j][k] * d[k]) * l[i][k];
// - a kernel update sums the rank terms l[i][k] * (d[k] * l[j][k]) of each
//   entry from zero in ascending k and subtracts the sum once
//   (Accumulate::sum_then_subtract). Terms are often far smaller than the
//   dominant diagonal they update, so summing first avoids rounding at the
//   entry's scale once per term. When the part's columns are contiguous
//   positions, its rows that are also columns are contiguous too and are
//   updated in place by gemm_sub_lower, and its rows below every column are
//   updated in place by gemm_sub, or by gemm_sub_rows through one pointer
//   per row when their positions are not contiguous. A part with
//   noncontiguous columns is gathered into the worker's scratch block,
//   updated there, and scattered back.
// Only entries on or below the diagonal are read or written. A kernel
// element's result depends only on its own operands, never on the strip, the
// part's shape, or the path, so every entry receives the same operations in
// the same contributor order on both paths; strips are disjoint, so neither
// the thread that runs a strip nor the order of strips can change a result.
// The kernel's finiteness flag is authoritative on every path: a kernel may
// stop at its first non-finite block, so a false flag rejects the attempt
// before anything is scattered, and no path scatters an entry the kernel was
// not asked to compute.
template <typename Executor>
void apply_contributions(
    double* front, index_t width, index_t height, FanIn& fan_in, double* operands,
    bool parallel, const dense_kernels::Kernels& kernels, Executor& executor,
    NumericInfo& info
) {
    constexpr auto order = dense_kernels::Accumulate::sum_then_subtract;
    constexpr auto stride = static_cast<std::size_t>(fan_in_columns);
    const auto ldc = static_cast<std::size_t>(width);
    // D_Q L[J,Q].T of a kernel contribution as [rank x columns]. An overflow
    // here reaches the kernel's flag through output entry (j, j), which every
    // update computes.
    const auto pack = [](const Contribution& q, double* packed) {
        const double* const left = q.left;
        const double* const pivots = q.pivots;
        const index_t rank = q.rank;
        const index_t columns = q.columns;
        for (index_t j = 0; j < columns; ++j) {
            const double* const l = left + j * rank;
            for (index_t k = 0; k < rank; ++k) packed[k * columns + j] = pivots[k] * l[k];
        }
    };
    // Output rows [ia, ib) by columns [0, jb) of a sparse update.
    const auto update_sparse = [&](const Contribution& q, const index_t* p,
                                   index_t ia, index_t ib, index_t jb) {
        // Every target is a front entry, tested later as a pivot or an L
        // entry (see the finiteness note in factorize).
        double* const target = front;
        const index_t row_stride = width;
        const double* const left = q.left;
        const double* const pivots = q.pivots;
        const index_t rank = q.rank;
        for (index_t k = 0; k < rank; ++k) {
            const double pivot = pivots[k];
            for (index_t j = 0; j < jb; ++j) {
                const double weighted = left[j * rank + k] * pivot;
                double* const column = target + p[j];
                for (index_t i = std::max(ia, j); i < ib; ++i) {
                    column[p[i] * row_stride] -= weighted * left[i * rank + k];
                }
            }
        }
    };
    // Output rows [ia, ib) by columns [ja, jb) of a kernel update whose
    // operand is packed, with ia >= ja, at most fan_in_rows rows and at most
    // fan_in_columns columns.
    const auto update_part = [&](const Contribution& q, const index_t* p, const double* packed,
                                 index_t ia, index_t ib, index_t ja, index_t jb, double* scratch) {
        const double* const left = q.left;
        const index_t rank = q.rank;
        const auto n = static_cast<std::size_t>(jb - ja);
        const auto kc = static_cast<std::size_t>(rank);
        const double* const b = packed + static_cast<std::size_t>(ja);
        const auto ldb = static_cast<std::size_t>(q.columns);
        const index_t leftmost = p[ja];
        bool finite = true;
        if (p[jb - 1] - leftmost == jb - 1 - ja) {
            // Rows ia..min(ib, jb) - 1 are columns of the part, so they are
            // contiguous positions; local entry (r, c) is output entry
            // (ia + r, ja + c), on or below the diagonal when c <= r + ia - ja.
            const index_t cross_end = std::min(ib, jb);
            if (ia < cross_end) {
                finite = kernels.gemm_sub_lower(front + p[ia] * width + leftmost, ldc,
                    left + ia * rank, kc, b, ldb, static_cast<std::size_t>(cross_end - ia), n, kc,
                    static_cast<std::ptrdiff_t>(ia - ja), order);
            }
            const index_t below = std::max(ia, jb);
            if (finite && below < ib) {
                const auto m = static_cast<std::size_t>(ib - below);
                if (p[ib - 1] - p[below] == ib - 1 - below) {
                    finite = kernels.gemm_sub(front + p[below] * width + leftmost, ldc,
                        left + below * rank, kc, b, ldb, m, n, kc, order);
                } else {
                    double* row_pointers[fan_in_rows];
                    for (index_t i = below; i < ib; ++i) {
                        row_pointers[i - below] = front + p[i] * width + leftmost;
                    }
                    finite = kernels.gemm_sub_rows(row_pointers, left + below * rank, kc,
                        b, ldb, m, n, kc, order);
                }
            }
            if (!finite) throw NonPositiveDefinite{};
            return;
        }
        // Local entry (r, c) of the gathered part is output entry
        // (ia + r, ja + c), on or below the diagonal when c <= r + shift.
        const auto m = static_cast<std::size_t>(ib - ia);
        const auto shift = static_cast<std::size_t>(ia - ja);
        const bool lower = ia < jb - 1;
        const index_t* const columns = p + ja;
        for (std::size_t r = 0; r < m; ++r) {
            const double* const source = front + p[ia + static_cast<index_t>(r)] * width;
            double* const values = scratch + r * stride;
            const std::size_t count = lower ? std::min(n, r + shift + 1) : n;
            for (std::size_t c = 0; c < count; ++c) values[c] = source[columns[c]];
        }
        finite = lower
            ? kernels.gemm_sub_lower(scratch, stride, left + ia * rank, kc, b, ldb, m, n, kc,
                  static_cast<std::ptrdiff_t>(shift), order)
            : kernels.gemm_sub(scratch, stride, left + ia * rank, kc, b, ldb, m, n, kc, order);
        if (!finite) throw NonPositiveDefinite{};
        for (std::size_t r = 0; r < m; ++r) {
            double* const target = front + p[ia + static_cast<index_t>(r)] * width;
            const double* const values = scratch + r * stride;
            const std::size_t count = lower ? std::min(n, r + shift + 1) : n;
            for (std::size_t c = 0; c < count; ++c) target[columns[c]] = values[c];
        }
    };
    // Apply every contribution's output rows inside node rows [row_first,
    // row_end). Without separate operands, each kernel contribution is
    // packed at the start of operands just before it is applied.
    const auto update_strip = [&](index_t row_first, index_t row_end, double* scratch, bool separate) {
        for (const Contribution& q : fan_in.contributions) {
            if (q.bottom < row_first || q.top >= row_end) continue;
            const index_t* const p = fan_in.positions.data() + q.first;
            const index_t ia = q.top >= row_first ? 0 :
                static_cast<index_t>(std::lower_bound(p, p + q.rows, row_first) - p);
            const index_t ib = q.bottom < row_end ? q.rows :
                static_cast<index_t>(std::lower_bound(p + ia, p + q.rows, row_end) - p);
            if (ia >= ib) continue;
            const index_t jb = std::min(q.columns, ib);
            if (q.right == sparse_update) {
                update_sparse(q, p, ia, ib, jb);
                continue;
            }
            // Separate operands sit at their offsets; otherwise the buffer
            // only holds the one packed here.
            const double* const packed = separate ? operands + q.right : operands;
            if (!separate) {
                const auto pack_start = std::chrono::steady_clock::now();
                pack(q, operands);
                info.pack_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pack_start).count();
            }
            for (index_t j = 0; j < jb; j += fan_in_columns) {
                const index_t j_end = std::min(jb, j + fan_in_columns);
                for (index_t i = std::max(ia, j); i < ib; i += fan_in_rows) {
                    update_part(q, p, packed, i, std::min(ib, i + fan_in_rows), j, j_end, scratch);
                }
            }
        }
    };
    if (!parallel) {
        update_strip(0, height, fan_in.scratch.values, false);
        return;
    }
    const auto pack_start = std::chrono::steady_clock::now();
    for (const Contribution& q : fan_in.contributions) {
        if (q.right != sparse_update) pack(q, operands + q.right);
    }
    info.pack_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - pack_start).count();
    const auto strips = static_cast<std::size_t>((height + fan_in_rows - 1) / fan_in_rows);
    executor.parallel_for(strips, [&](std::size_t item, int worker) {
        const index_t row_first = static_cast<index_t>(strips - 1 - item) * fan_in_rows;
        update_strip(row_first, std::min(height, row_first + fan_in_rows), fan_in.scratch.values +
            static_cast<std::size_t>(worker) * static_cast<std::size_t>(fan_in_rows) * stride, true);
    });
    ++info.parallel_panels;
}

// True when no value is Inf or NaN (fabs(NaN) <= DBL_MAX is false).
[[nodiscard]] inline bool all_finite(const double* values, index_t count) noexcept {
    bool finite = true;
    for (index_t i = 0; i < count; ++i) {
        finite &= std::fabs(values[i]) <= std::numeric_limits<double>::max();
    }
    return finite;
}

// Factor the panel's diagonal block F[begin:end, begin:end] in place (the
// lower triangle of the row-major node rectangle, row stride width) and store
// its pivots in pivots[0, end - begin). Pivot p must be positive and finite;
// it divides column p, which is then copied to a contiguous buffer so that its
// rank-1 update of each later row i is an AXPY over j. Every element performs
// the same scalar sequence as before, value -= weighted * L[j][p] with
// weighted = L[i][p] * pivot, so an entry that cancels exactly still stores an
// exact zero. The block's L entries are not tested here; factorize scans them.
inline void factor_diagonal_block(
    double* front, index_t width, index_t begin, index_t end, double* pivots
) {
    double column[panel_width];
    for (index_t p = begin; p < end; ++p) {
        const double pivot = front[p * width + p];
        if (!(pivot > 0.0) || !std::isfinite(pivot)) throw NonPositiveDefinite{};
        pivots[p - begin] = pivot;
        for (index_t i = p + 1; i < end; ++i) {
            double& value = front[i * width + p];
            value /= pivot;
            column[i - begin] = value;
        }
        for (index_t i = p + 1; i < end; ++i) {
            double* const row = front + i * width;
            const double weighted = row[p] * pivot;
            for (index_t j = p + 1; j <= i; ++j) row[j] -= weighted * column[j - begin];
        }
    }
}

// Below-panel solve, L[i,P] = F[i,P] inv(L_PP.T) inv(D_P), of count rows
// x + r * stride (block entries each, holding the updated F[i,P] on entry), in
// AXPY form. For k ascending, x[k] /= d[k] completes L[i][k], and then
// x[p] -= x[k] * S[k][p] for every p > k, where weights holds
// S[k][p] = d[k] * L[p][k] with row stride panel_width. Each entry therefore
// subtracts its terms in ascending k, each a rounded product and a rounded
// subtraction, and finally divides by its own pivot. Rows are independent;
// four are interleaved so that their division and update chains overlap, and
// the update over p is contiguous in every row. Neither changes any entry's
// operations.
inline void solve_panel_rows(
    double* x, index_t stride, index_t count,
    const double* pivots, const double* weights, index_t block
) noexcept {
    index_t row = 0;
    for (; row + 4 <= count; row += 4) {
        double* const x0 = x + row * stride;
        double* const x1 = x0 + stride;
        double* const x2 = x1 + stride;
        double* const x3 = x2 + stride;
        for (index_t k = 0; k < block; ++k) {
            const double pivot = pivots[k];
            const double v0 = x0[k] / pivot;
            const double v1 = x1[k] / pivot;
            const double v2 = x2[k] / pivot;
            const double v3 = x3[k] / pivot;
            x0[k] = v0;
            x1[k] = v1;
            x2[k] = v2;
            x3[k] = v3;
            const double* const weight = weights + k * panel_width;
            for (index_t p = k + 1; p < block; ++p) {
                const double factor = weight[p];
                x0[p] -= v0 * factor;
                x1[p] -= v1 * factor;
                x2[p] -= v2 * factor;
                x3[p] -= v3 * factor;
            }
        }
    }
    for (; row < count; ++row) {
        double* const x0 = x + row * stride;
        for (index_t k = 0; k < block; ++k) {
            const double v0 = x0[k] / pivots[k];
            x0[k] = v0;
            const double* const weight = weights + k * panel_width;
            for (index_t p = k + 1; p < block; ++p) x0[p] -= v0 * weight[p];
        }
    }
}

// Numeric phase of the supernodal LDL.T. Every node's dense rectangle lives
// in one array, blocks (allocated here, pattern.block_ptr[node] being the
// node's offset): row-major [height x width], positions 0..width-1 being the
// node's own columns and then its tail rows in ascending order. A finished
// node's block holds its pivots on the diagonal and its L entries below it,
// and later nodes read it in place as their contributor operand; it is never
// written again. Each node zero-fills its own block before scattering into
// it, so blocks of nodes not yet reached are never read. Returns false, with
// nothing assembled, when a pivot or intermediate fails; on success, assemble
// copies the blocks into the CSC values.
template <typename Int, typename Executor>
bool factorize(
    const double* data, const Int* indices, const Int* indptr,
    const std::vector<index_t>& permutation,
    const std::vector<index_t>& inverse_permutation,
    const std::vector<double>& scale,
    const std::vector<index_t>& col_ptr, const std::vector<index_t>& rows,
    const Pattern& pattern, std::unique_ptr<double[]>& blocks,
    std::vector<double>& diagonal, const dense_kernels::Kernels& kernels,
    Executor& executor, int threads, NumericInfo& info, std::uint64_t base_workspace
) {
    const index_t n = static_cast<index_t>(permutation.size());
    const std::size_t block_entries = pattern.block_ptr.back();
    blocks.reset(new double[block_entries]);
    std::vector<index_t> row_position(static_cast<std::size_t>(n), -1);
    std::vector<index_t> mark(static_cast<std::size_t>(n), -1);
    std::vector<index_t> contributors;
    OperandBuffer right;
    FanIn fan_in;
    // One gather block per worker; each starts on a 64-byte boundary too.
    fan_in.scratch.allocate(static_cast<std::size_t>(std::max(threads, 1)) *
        checked_product(fan_in_rows, fan_in_columns));
    auto workspace = [&] {
        info.workspace_bytes = std::max(info.workspace_bytes, base_workspace +
            sizeof(index_t) * (row_position.capacity() + mark.capacity() +
                contributors.capacity() + pattern.nodes.capacity() +
                pattern.row_ptr.capacity() + pattern.row_sources.capacity() +
                pattern.column_nodes.capacity()) +
            sizeof(std::size_t) * pattern.block_ptr.capacity() +
            sizeof(double) * (block_entries + right.buffer.capacity) + fan_in.bytes());
    };
    workspace();
    try {
        for (std::size_t node = 0; node + 1 < pattern.nodes.size(); ++node) {
            const index_t first = pattern.nodes[node];
            const index_t last = pattern.nodes[node + 1];
            const index_t width = last - first;
            const index_t row_begin = col_ptr[static_cast<std::size_t>(first)];
            const index_t row_end = col_ptr[static_cast<std::size_t>(first + 1)];
            const index_t height = 1 + row_end - row_begin;
            row_position[static_cast<std::size_t>(first)] = 0;
            for (index_t p = row_begin; p < row_end; ++p) {
                row_position[static_cast<std::size_t>(rows[p])] = 1 + p - row_begin;
            }
            const auto panel_entries = pattern.block_ptr[node + 1] - pattern.block_ptr[node];
            double* const front = blocks.get() + pattern.block_ptr[node];
            std::fill_n(front, panel_entries, 0.0);
            info.maximum_panel_entries = std::max(info.maximum_panel_entries,
                static_cast<std::uint64_t>(panel_entries));
            contributors.clear();
            for (index_t col = first; col < last; ++col) {
                const index_t original_col = permutation[static_cast<std::size_t>(col)];
                for (index_t p = static_cast<index_t>(indptr[original_col]);
                     p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
                    const index_t original_row = static_cast<index_t>(indices[p]);
                    const index_t row = inverse_permutation[static_cast<std::size_t>(original_row)];
                    if (row < col) continue;
                    const index_t position = row_position[static_cast<std::size_t>(row)];
                    if (position < 0) throw std::logic_error("incomplete symmetric panel pattern");
                    const double value = data[p] * scale[static_cast<std::size_t>(original_row)] *
                        scale[static_cast<std::size_t>(original_col)];
                    if (!std::isfinite(value)) throw NonPositiveDefinite{};
                    front[static_cast<std::size_t>(position * width + col - first)] = value;
                }
                for (index_t p = pattern.row_ptr[static_cast<std::size_t>(col)];
                     p < pattern.row_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
                    const index_t source = pattern.row_sources[static_cast<std::size_t>(p)];
                    const index_t source_node = pattern.column_nodes[static_cast<std::size_t>(source)];
                    if (source >= first || mark[static_cast<std::size_t>(source_node)] == first) continue;
                    mark[static_cast<std::size_t>(source_node)] = first;
                    contributors.push_back(source_node);
                }
            }
            std::sort(contributors.begin(), contributors.end());
            workspace();
            const auto source_start = std::chrono::steady_clock::now();
            fan_in.contributions.clear();
            fan_in.positions.clear();
            std::size_t right_entries = 0;
            std::size_t largest_operand = 0;
            std::uint64_t source_flops = 0;
            for (index_t source_node : contributors) {
                const index_t source_first = pattern.nodes[static_cast<std::size_t>(source_node)];
                const index_t source_last = pattern.nodes[static_cast<std::size_t>(source_node + 1)];
                const index_t source_width = source_last - source_first;
                const auto source_rows = rows.begin() + col_ptr[static_cast<std::size_t>(source_first)];
                const auto source_end = rows.begin() + col_ptr[static_cast<std::size_t>(source_first + 1)];
                const auto common_begin = std::lower_bound(source_rows, source_end, first);
                // The source rows from the first one inside this node onwards
                // all lie in this node (elimination tree containment), so they
                // are a contiguous run of the source block's rows, starting at
                // block row common_row (row 0 is the diagonal row
                // source_first). Output columns, the rows among them that are
                // this node's columns (rows below last), come first: they are
                // a prefix of the output rows.
                const index_t common_row = 1 + static_cast<index_t>(common_begin - source_rows);
                const index_t active_rows = static_cast<index_t>(source_end - common_begin);
                const index_t active_cols = static_cast<index_t>(
                    std::lower_bound(common_begin, source_end, last) - common_begin);
                if (active_cols == 0) throw std::logic_error("empty symmetric contributor");
                const std::size_t first_position = fan_in.positions.size();
                fan_in.positions.resize(first_position + static_cast<std::size_t>(active_rows));
                index_t* const targets = fan_in.positions.data() + first_position;
                index_t missing = 0;
                for (index_t i = 0; i < active_rows; ++i) {
                    const index_t position = row_position[static_cast<std::size_t>(common_begin[i])];
                    targets[i] = position;
                    missing |= position;
                }
                if (missing < 0) throw std::logic_error("noncontiguous symmetric contributor");
                // L[R,Q] in place: active_rows full rows of the finished
                // source block (row stride source_width), each holding final,
                // scanned L entries in all source_width columns.
                const double* const source = blocks.get() +
                    pattern.block_ptr[static_cast<std::size_t>(source_node)] +
                    static_cast<std::size_t>(common_row * source_width);
                // Tiny intersections use sparse outer products, retaining the
                // same source order without packing structural zeros. Larger
                // ones are applied as one rank source_width update with
                // right = D_Q L[J,Q].T packed as [source_width x active_cols];
                // it holds at most the source block's entries, so the sum
                // over distinct contributors cannot overflow.
                const bool sparse = checked_product(active_rows, active_cols) < sparse_update_entries;
                fan_in.contributions.push_back(Contribution{source, diagonal.data() + source_first,
                    source_width, active_rows, active_cols, targets[0], targets[active_rows - 1],
                    first_position, sparse ? sparse_update : right_entries});
                if (!sparse) {
                    const auto operand = checked_product(source_width, active_cols);
                    right_entries += operand;
                    largest_operand = std::max(largest_operand, operand);
                }
                // Each source column applies active_rows - j terms to output
                // column j.
                source_flops += 2 * static_cast<std::uint64_t>(source_width) *
                    static_cast<std::uint64_t>(active_cols * active_rows -
                        active_cols * (active_cols - 1) / 2);
            }
            info.update_flops += source_flops;
            if (!fan_in.contributions.empty()) {
                // The strips depend on the node's shape only; worth_parallel
                // only decides which thread applies which entries.
                const auto strips = static_cast<std::size_t>((height + fan_in_rows - 1) / fan_in_rows);
                const bool parallel = worth_parallel(source_flops, strips, threads);
                double* const operands = right.reserve(parallel ? right_entries : largest_operand);
                workspace();
                apply_contributions(front, width, height, fan_in, operands, parallel,
                    kernels, executor, info);
            }
            info.source_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - source_start).count();
            // Finiteness. Original entries are tested when scattered. After
            // that no arithmetic result is tested where it is produced;
            // instead every node-rectangle entry on or below the diagonal ends
            // as a pivot or as an L entry, and both are tested: each pivot
            // before it is used (positive and finite), each panel's L
            // entries by one scan after their last write, before a later
            // node reads the block. This is complete because:
            // - entries combine only through +, -, * and division by a tested
            //   pivot, and these keep an Inf or NaN operand non-finite (Inf*0
            //   and Inf-Inf are NaN), so a non-finite entry stays non-finite
            //   until it is tested;
            // - each auxiliary product reaches an entry: a sparse update's
            //   weighted = l * d updates its own diagonal position with the
            //   nonzero l; the diagonal block's weighted = L[i][p] * pivot
            //   updates the future pivot (i, i) with L[i][p]; S[k][p] =
            //   d[k] * L[p][k] is formed only when rows below the block exist,
            //   and enters each such row's L[i][p];
            // - a contributor's in-place left operand holds only tested L
            //   entries, and its packed right operand d[k] * l[j][k] enters
            //   output entry (j, j) with the nonzero l[j][k] (a zero l gives
            //   a zero product). The fan-in strip holding row j of that
            //   diagonal entry applies the contribution's rows there, and
            //   every column up to that row, so a kernel part computes the
            //   entry and the kernel's flag sees it; the flag is
            //   authoritative, and a false flag (a kernel may stop early)
            //   rejects the attempt before any later use.
            // A rejected attempt assembles nothing and falls back to LU.
            constexpr auto order = dense_kernels::Accumulate::sum_then_subtract;
            // Trailing work is split into fixed strips of node rows, so the
            // partition depends on positions only. Rows are independent: a
            // strip reads only its own rows and the shared, read-only pivots,
            // S and right operand, and writes only its own rows. Sixteen rows
            // (four kernel row groups) keep panels with 256-512 rows below
            // them parallel at eight threads under worth_parallel's two items
            // per thread.
            constexpr index_t strip_rows = 16;
            double* const rect = front;
            const auto panel_start = std::chrono::steady_clock::now();
            for (index_t begin = 0; begin < width; begin += panel_width) {
                const index_t end = std::min(width, begin + panel_width);
                const index_t block = end - begin;
                double* const pivots = diagonal.data() + first + begin;
                factor_diagonal_block(rect, width, begin, end, pivots);
                for (index_t i = begin + 1; i < end; ++i) {
                    if (!all_finite(rect + i * width + begin, i - begin)) throw NonPositiveDefinite{};
                }
                // Multiply-subtract terms: the diagonal block, then the solve
                // and trailing update below.
                std::uint64_t terms = static_cast<std::uint64_t>((block - 1) * block * (block + 1) / 6);
                if (end < height) {
                    // L_RP = F_RP * inv(L_PP.T) * inv(D_P), then
                    // F_RJ -= (L_RP D_P) L_JP.T for the remaining node columns
                    // J = [end, width), lower part only.
                    double weights[panel_width * panel_width];
                    for (index_t k = 0; k < block; ++k) {
                        for (index_t p = k + 1; p < block; ++p) {
                            weights[k * panel_width + p] = pivots[k] * rect[(begin + p) * width + begin + k];
                        }
                    }
                    // Rows [end, width) form the shared right operand, so
                    // they are solved and scanned first, in order.
                    const index_t columns = width - end;
                    solve_panel_rows(rect + end * width + begin, width, columns, pivots, weights, block);
                    for (index_t i = end; i < width; ++i) {
                        if (!all_finite(rect + i * width + begin, block)) throw NonPositiveDefinite{};
                    }
                    double* const shared = columns > 0 ? right.reserve(checked_product(block, columns)) : nullptr;
                    if (columns > 0) {
                        const auto pack_start = std::chrono::steady_clock::now();
                        workspace();
                        for (index_t i = end; i < width; ++i) {
                            for (index_t k = 0; k < block; ++k) {
                                shared[k * columns + i - end] = rect[i * width + begin + k];
                            }
                        }
                        info.pack_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pack_start).count();
                    }
                    // Each strip solves and scans its rows at or below width
                    // (rows above it were solved above), packs L_RP D_P, and
                    // applies the rank-block update to its rows' entries in
                    // columns [end, min(row, width - 1)].
                    const auto strips = static_cast<std::size_t>(
                        (height - end + strip_rows - 1) / strip_rows);
                    // Strips are claimed bottom-up: lower rows update more
                    // columns, so the largest items start first.
                    auto strip = [&](std::size_t index, int) {
                        const index_t row_first = end +
                            static_cast<index_t>(strips - 1 - index) * strip_rows;
                        const index_t row_last = std::min(height, row_first + strip_rows);
                        const index_t solve_first = std::max(row_first, width);
                        if (solve_first < row_last) {
                            solve_panel_rows(rect + solve_first * width + begin, width,
                                row_last - solve_first, pivots, weights, block);
                            for (index_t i = solve_first; i < row_last; ++i) {
                                if (!all_finite(rect + i * width + begin, block)) throw NonPositiveDefinite{};
                            }
                        }
                        if (columns == 0) return;
                        double packed[strip_rows * panel_width];
                        for (index_t i = row_first; i < row_last; ++i) {
                            const double* const x = rect + i * width + begin;
                            double* const a = packed + (i - row_first) * panel_width;
                            for (index_t k = 0; k < block; ++k) a[k] = x[k] * pivots[k];
                        }
                        double* const c = rect + row_first * width + end;
                        const auto m = static_cast<std::size_t>(row_last - row_first);
                        const auto n_columns = static_cast<std::size_t>(columns);
                        const auto ldc = static_cast<std::size_t>(width);
                        const auto lda = static_cast<std::size_t>(panel_width);
                        const auto rank = static_cast<std::size_t>(block);
                        // Rows at or below width update every column; rows
                        // above it update only entries with row >= column.
                        const bool finite = row_first >= width
                            ? kernels.gemm_sub(c, ldc, packed, lda, shared, n_columns,
                                  m, n_columns, rank, order)
                            : kernels.gemm_sub_lower(c, ldc, packed, lda, shared, n_columns,
                                  m, n_columns, rank, static_cast<std::ptrdiff_t>(row_first - end), order);
                        if (!finite) throw NonPositiveDefinite{};
                    };
                    const index_t tail_rows = height - width;
                    const auto pair_terms = static_cast<std::uint64_t>(block * (block - 1) / 2);
                    // Entries with row >= column in rows [end, height) and
                    // columns [end, width).
                    const auto entries = static_cast<std::uint64_t>(
                        columns * (columns + 1) / 2 + tail_rows * columns);
                    const std::uint64_t strip_terms =
                        static_cast<std::uint64_t>(tail_rows) * pair_terms +
                        entries * static_cast<std::uint64_t>(block);
                    terms += static_cast<std::uint64_t>(columns) * pair_terms + strip_terms;
                    if (worth_parallel(2 * strip_terms, strips, threads)) {
                        executor.parallel_for(strips, strip);
                        ++info.parallel_panels;
                    } else {
                        for (std::size_t index = 0; index < strips; ++index) strip(index, 0);
                    }
                }
                info.update_flops += 2 * terms;
            }
            info.panel_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - panel_start).count();
            row_position[static_cast<std::size_t>(first)] = -1;
            for (index_t p = row_begin; p < row_end; ++p) {
                row_position[static_cast<std::size_t>(rows[p])] = -1;
            }
        }
    } catch (const NonPositiveDefinite&) {
        return false;
    }
    return true;
}

// Copy the strict lower entries of every factored node block into the CSC
// values, once all pivots have passed. Column first + c of a node stores
// block rows c + 1, ..., height - 1 of block column c, in this order: the
// node's later columns, then its tail rows (find_nodes accepts a column only
// when its strict rows are those of the previous column without its first).
// Entries that cancelled to zero are copied as stored zeros. Work items are
// the fixed column ranges [8 * item, 8 * item + 8), split at node boundaries;
// each part is copied row by row, so each block row is read once. Items write
// disjoint columns and only copy, so neither the partition nor the thread
// that runs an item can change a value. The copy is memory bound; it runs on
// the worker team for at least 2^18 entries (2 MB of values, about 0.15 ms
// on one thread) and two items per thread, counting at most eight threads.
template <typename Executor>
void assemble(
    const Pattern& pattern, const std::vector<index_t>& col_ptr,
    const double* blocks, std::vector<double>& values,
    Executor& executor, int threads
) {
    constexpr index_t group = 8;
    const index_t n = static_cast<index_t>(col_ptr.size()) - 1;
    const auto items = static_cast<std::size_t>((n + group - 1) / group);
    auto copy = [&](std::size_t item, int) {
        const index_t stop = std::min(n, static_cast<index_t>(item + 1) * group);
        for (index_t start = static_cast<index_t>(item) * group; start < stop;) {
            const index_t node = pattern.column_nodes[static_cast<std::size_t>(start)];
            const index_t first = pattern.nodes[static_cast<std::size_t>(node)];
            const index_t width = pattern.nodes[static_cast<std::size_t>(node + 1)] - first;
            const index_t height = 1 + col_ptr[static_cast<std::size_t>(first + 1)] -
                col_ptr[static_cast<std::size_t>(first)];
            const index_t begin = start - first;
            const index_t end = std::min(width, stop - first);
            const double* const block = blocks + pattern.block_ptr[static_cast<std::size_t>(node)];
            // Block row r of column col goes to CSC position
            // col_ptr[first + col] + r - (col + 1), that is, base + r.
            index_t base[group];
            for (index_t col = begin; col < end; ++col) {
                const index_t column = first + col;
                if (col_ptr[static_cast<std::size_t>(column + 1)] -
                    col_ptr[static_cast<std::size_t>(column)] != height - 1 - col) {
                    throw std::logic_error("inconsistent symmetric node pattern");
                }
                base[col - begin] = col_ptr[static_cast<std::size_t>(column)] - (col + 1);
            }
            for (index_t row = begin + 1; row < height; ++row) {
                const double* const source = block + row * width;
                const index_t row_end = std::min(end, row);
                for (index_t col = begin; col < row_end; ++col) {
                    values[static_cast<std::size_t>(base[col - begin] + row)] = source[col];
                }
            }
            start = first + end;
        }
    };
    constexpr int counted_threads = 8;
    if (threads > 1 && values.size() >= (std::size_t{1} << 18) &&
        items >= 2 * static_cast<std::size_t>(std::min(threads, counted_threads))) {
        executor.parallel_for(items, copy);
    } else {
        for (std::size_t item = 0; item < items; ++item) copy(item, 0);
    }
}

} // namespace sparse_ldlt
