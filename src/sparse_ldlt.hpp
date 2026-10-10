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

// Fundamental supernodes from the elimination tree and the strict column
// counts of the first symbolic pass: column j + 1 joins column j's node
// exactly when parent(j) = j + 1 and count(j) = count(j + 1) + 1. This is the
// exact supernode test, not an upper bound: when parent(j) = j + 1, every
// strict row of column j other than j + 1 is a strict row of column j + 1
// (the rows of a column below its parent are rows of the parent), so the
// count equality makes the two row sets equal. Calls visit(first, last) for
// each node [first, last) in ascending order.
template <typename Visit>
void for_each_node(
    const std::vector<index_t>& parent, const std::vector<index_t>& counts, Visit&& visit
) {
    const index_t n = static_cast<index_t>(parent.size());
    for (index_t first = 0; first < n;) {
        index_t last = first + 1;
        while (last < n && parent[static_cast<std::size_t>(last - 1)] == last &&
               counts[static_cast<std::size_t>(last - 1)] ==
                   counts[static_cast<std::size_t>(last)] + 1) {
            ++last;
        }
        visit(first, last);
        first = last;
    }
}

// One earlier node's contribution to a node, found by the symbolic phase. The
// source node's tail rows from index begin to the end of its tail all lie in
// the target node's rows (elimination-tree containment); the first columns of
// them are target columns, the rest target tail rows.
struct Source {
    index_t node;
    index_t begin;
    index_t columns;
};

// Supernodal symbolic structure. Node s owns the columns [nodes[s],
// nodes[s + 1]) and the tail rows tail_rows[tail_ptr[s], tail_ptr[s + 1]),
// ascending: the strict rows of its last column. Its rectangle has its
// columns, then its tail rows, as rows. By for_each_node, the strict rows of
// a node column j other than the last are j + 1 and the strict rows of
// column j + 1, so node column first + c holds exactly the node rows
// c + 1, ..., height - 1, in this order; assemble writes the CSC row indices
// of L from this, and checks every column's count against the first pass.
struct Pattern {
    std::vector<index_t> nodes;
    std::vector<index_t> column_nodes;
    // Node of the last column's parent, or -1 for a root.
    std::vector<index_t> parent_node;
    std::vector<index_t> tail_ptr;
    std::vector<index_t> tail_rows;
    // Contributors of each node in ascending node order:
    // sources[source_ptr[s], source_ptr[s + 1]).
    std::vector<index_t> source_ptr;
    std::vector<Source> sources;
    // Each node's fan-in multiply-subtract operations, as counted in
    // dense_update_flops, and its whole work estimate (see work_estimate).
    std::vector<std::uint64_t> fan_in_flops;
    std::vector<std::uint64_t> work;
    // Offset of each node's dense block in the numeric phase's single block
    // array: block_ptr[node + 1] - block_ptr[node] = height * width.
    std::vector<std::size_t> block_ptr;

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return sizeof(index_t) * (nodes.capacity() + column_nodes.capacity() +
                parent_node.capacity() + tail_ptr.capacity() + tail_rows.capacity() +
                source_ptr.capacity()) +
            sizeof(Source) * sources.capacity() +
            sizeof(std::uint64_t) * (fan_in_flops.capacity() + work.capacity()) +
            sizeof(std::size_t) * block_ptr.capacity();
    }
};

// Multiply-subtract operations of a node's panel loop in factorize (diagonal
// blocks, below-panel solves and trailing updates), twice the terms counted
// there.
[[nodiscard]] inline std::uint64_t panel_flops(index_t width, index_t height) noexcept {
    std::uint64_t terms = 0;
    for (index_t begin = 0; begin < width; begin += panel_width) {
        const index_t end = std::min(width, begin + panel_width);
        const index_t block = end - begin;
        terms += static_cast<std::uint64_t>((block - 1) * block * (block + 1) / 6);
        if (end < height) {
            const index_t columns = width - end;
            const index_t tail_rows = height - width;
            const auto pair_terms = static_cast<std::uint64_t>(block * (block - 1) / 2);
            const auto entries = static_cast<std::uint64_t>(
                columns * (columns + 1) / 2 + tail_rows * columns);
            terms += static_cast<std::uint64_t>(columns) * pair_terms +
                static_cast<std::uint64_t>(tail_rows) * pair_terms +
                entries * static_cast<std::uint64_t>(block);
        }
    }
    return 2 * terms;
}

// A node's cost for subtree scheduling, in flop units: its fan-in and panel
// operations plus allowances, measured on small nodes, for the memory-bound
// work (zero-fill, scatter and copy-out per block entry, one target-position
// list per contributor, and fixed per-node bookkeeping). It only balances the
// subtrees between workers and never affects a result.
[[nodiscard]] inline std::uint64_t work_estimate(
    std::uint64_t fan_in_flops, index_t width, index_t height, std::size_t entries, index_t sources
) noexcept {
    return fan_in_flops + panel_flops(width, height) + 8 * static_cast<std::uint64_t>(entries) +
        256 * static_cast<std::uint64_t>(sources) + 2048;
}

// Supernodal symbolic factorization over the node partition in
// pattern.nodes, for the permuted matrix B = P^T A P given by the CSC input
// and the permutation. col_ptr holds the strict column counts' prefix sums
// from the first pass; mark is n-sized scratch (overwritten).
//
// Tails. For a node s with columns J = [f, l), the strict rows of every column
// j in J that are >= l are exactly tail(s) (see Pattern). The strict rows of
// a column j are its own entries of B below j and the strict rows of its
// children in the elimination tree, without j. A child of a column in J is
// either the previous column of J or the last column of an earlier node c
// whose parent column lies in J (a child node of s), whose strict rows are
// tail(c). Hence tail(s) is the union of B's entries in the columns J and of
// the child nodes' tails, restricted to rows >= l: each of these sets lies in
// the strict rows of some column of J, so within tail(s), and the sets for
// column f alone already give all of tail(s), since f has no child in J.
// Each column of B is visited once and each tail once more by its parent
// node, so this costs O(nnz(B) + sum of tails) plus sorting each tail,
// instead of one step per entry of L for the row-subtree walk; the first
// pass's counts predict every tail's size, which is checked.
//
// Contributors. Node q contributes to every node t holding one of its tail
// rows. Its tail rows inside t's columns are one contiguous run of the sorted
// tail, and every later tail row is a row of t, so walking each tail once in
// ascending q lists every node's contributors in ascending order, with the
// run's start and length, in O(sum of tails).
template <typename Int>
void build_pattern(
    Pattern& pattern, const Int* indices, const Int* indptr,
    const std::vector<index_t>& permutation,
    const std::vector<index_t>& inverse_permutation,
    const std::vector<index_t>& parent, const std::vector<index_t>& col_ptr,
    std::vector<index_t>& mark
) {
    const index_t n = static_cast<index_t>(parent.size());
    const auto node_count = pattern.nodes.size() - 1;
    const auto maximum = std::vector<double>().max_size();
    pattern.column_nodes.resize(static_cast<std::size_t>(n));
    for (std::size_t node = 0; node < node_count; ++node) {
        for (index_t col = pattern.nodes[node]; col < pattern.nodes[node + 1]; ++col) {
            pattern.column_nodes[static_cast<std::size_t>(col)] = static_cast<index_t>(node);
        }
    }
    pattern.parent_node.resize(node_count);
    pattern.tail_ptr.assign(1, 0);
    pattern.tail_ptr.reserve(node_count + 1);
    pattern.block_ptr.assign(1, 0);
    pattern.block_ptr.reserve(node_count + 1);
    for (std::size_t node = 0; node < node_count; ++node) {
        const index_t first = pattern.nodes[node];
        const index_t last = pattern.nodes[node + 1];
        const index_t width = last - first;
        const index_t tail = col_ptr[static_cast<std::size_t>(last)] -
            col_ptr[static_cast<std::size_t>(last - 1)];
        const index_t height = width + tail;
        if (1 + col_ptr[static_cast<std::size_t>(first + 1)] -
                col_ptr[static_cast<std::size_t>(first)] != height) {
            throw std::logic_error("inconsistent symmetric node pattern");
        }
        // Tails sum to at most the strict entries, whose total did not
        // overflow.
        pattern.tail_ptr.push_back(pattern.tail_ptr.back() + tail);
        const index_t parent_col = parent[static_cast<std::size_t>(last - 1)];
        pattern.parent_node[node] = parent_col < 0 ? -1 :
            pattern.column_nodes[static_cast<std::size_t>(parent_col)];
        const auto entries = checked_product(height, width);
        const auto total = pattern.block_ptr.back();
        if (entries > maximum - total) {
            throw std::overflow_error("symmetric panel allocation overflowed");
        }
        pattern.block_ptr.push_back(total + entries);
    }

    // Child nodes as singly linked lists (head per node, next per child).
    std::vector<index_t> child(node_count, -1);
    std::vector<index_t> sibling(node_count, -1);
    for (std::size_t node = node_count; node-- > 0;) {
        const index_t up = pattern.parent_node[node];
        if (up < 0) continue;
        sibling[node] = child[static_cast<std::size_t>(up)];
        child[static_cast<std::size_t>(up)] = static_cast<index_t>(node);
    }
    pattern.tail_rows.resize(static_cast<std::size_t>(pattern.tail_ptr.back()));
    std::fill(mark.begin(), mark.end(), -1);
    for (std::size_t node = 0; node < node_count; ++node) {
        const index_t last = pattern.nodes[node + 1];
        const auto stamp = static_cast<index_t>(node);
        index_t* const out = pattern.tail_rows.data() + pattern.tail_ptr[node];
        const index_t expected = pattern.tail_ptr[node + 1] - pattern.tail_ptr[node];
        index_t size = 0;
        const auto add = [&](index_t row) {
            if (row < last || mark[static_cast<std::size_t>(row)] == stamp) return;
            if (size == expected) throw std::logic_error("inconsistent symmetric node pattern");
            mark[static_cast<std::size_t>(row)] = stamp;
            out[size++] = row;
        };
        for (index_t col = pattern.nodes[node]; col < last; ++col) {
            const index_t original_col = permutation[static_cast<std::size_t>(col)];
            for (index_t p = static_cast<index_t>(indptr[original_col]);
                 p < static_cast<index_t>(indptr[original_col + 1]); ++p) {
                add(inverse_permutation[static_cast<std::size_t>(indices[p])]);
            }
        }
        for (index_t c = child[node]; c >= 0; c = sibling[static_cast<std::size_t>(c)]) {
            for (index_t p = pattern.tail_ptr[static_cast<std::size_t>(c)];
                 p < pattern.tail_ptr[static_cast<std::size_t>(c + 1)]; ++p) {
                add(pattern.tail_rows[static_cast<std::size_t>(p)]);
            }
        }
        if (size != expected) throw std::logic_error("inconsistent symmetric node pattern");
        std::sort(out, out + size);
    }

    // Contributor lists, counted and then filled in ascending source order.
    pattern.source_ptr.assign(node_count + 1, 0);
    const auto for_each_source = [&](auto&& visit) {
        for (std::size_t node = 0; node < node_count; ++node) {
            const index_t* const tail = pattern.tail_rows.data() + pattern.tail_ptr[node];
            const index_t length = pattern.tail_ptr[node + 1] - pattern.tail_ptr[node];
            for (index_t begin = 0; begin < length;) {
                const index_t target = pattern.column_nodes[static_cast<std::size_t>(tail[begin])];
                const index_t target_last = pattern.nodes[static_cast<std::size_t>(target + 1)];
                index_t end = begin + 1;
                while (end < length && tail[end] < target_last) ++end;
                visit(node, target, begin, end, length);
                begin = end;
            }
        }
    };
    for_each_source([&](std::size_t, index_t target, index_t, index_t, index_t) {
        ++pattern.source_ptr[static_cast<std::size_t>(target + 1)];
    });
    for (std::size_t node = 0; node < node_count; ++node) {
        pattern.source_ptr[node + 1] += pattern.source_ptr[node];
    }
    pattern.sources.resize(static_cast<std::size_t>(pattern.source_ptr.back()));
    pattern.fan_in_flops.assign(node_count, 0);
    std::vector<index_t> next(pattern.source_ptr.begin(), pattern.source_ptr.end() - 1);
    for_each_source([&](std::size_t node, index_t target, index_t begin, index_t end, index_t length) {
        const index_t width = pattern.nodes[node + 1] - pattern.nodes[node];
        const index_t active_rows = length - begin;
        const index_t active_cols = end - begin;
        pattern.sources[static_cast<std::size_t>(next[static_cast<std::size_t>(target)]++)] =
            Source{static_cast<index_t>(node), begin, active_cols};
        // Each source column applies active_rows - j terms to output column j.
        pattern.fan_in_flops[static_cast<std::size_t>(target)] +=
            2 * static_cast<std::uint64_t>(width) *
            static_cast<std::uint64_t>(active_cols * active_rows -
                active_cols * (active_cols - 1) / 2);
    });
    pattern.work.resize(node_count);
    for (std::size_t node = 0; node < node_count; ++node) {
        const index_t width = pattern.nodes[node + 1] - pattern.nodes[node];
        const index_t height = width + pattern.tail_ptr[node + 1] - pattern.tail_ptr[node];
        pattern.work[node] = work_estimate(pattern.fan_in_flops[node], width, height,
            pattern.block_ptr[node + 1] - pattern.block_ptr[node],
            pattern.source_ptr[node + 1] - pattern.source_ptr[node]);
    }
}

struct NumericInfo {
    std::uint64_t workspace_bytes = 0;
    std::uint64_t maximum_panel_entries = 0;
    std::uint64_t update_flops = 0;
    index_t parallel_panels = 0;
    // Subtrees factored concurrently before the top nodes, and that phase's
    // wall time (both zero when it did not run).
    index_t subtrees = 0;
    double subtree_seconds = 0.0;
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
// Otherwise the thread that factors the node packs and applies one whole
// contribution after another, as one strip of all rows, without calling into
// the executor. Output columns at or after the strip's
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

// Per-thread numeric state, allocated once per factorization: the row map of
// the node being factored (map[row] is the rectangle position of each of that
// node's tail rows and -1 for every other row), the packed operand buffer,
// and the fan-in lists and gather blocks. Position is std::int32_t when
// n < 2^31: positions are below a node's height, at most n.
template <typename Position>
struct NodeWorkspace {
    std::vector<Position> map;
    OperandBuffer right;
    FanIn fan_in;

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return sizeof(Position) * map.capacity() +
            sizeof(double) * right.buffer.capacity + fan_in.bytes();
    }
};

// Read-only inputs of the numeric phase, plus the block array and the pivots,
// which each node writes only in its own block and its own columns.
template <typename Int>
struct NumericInput {
    const double* data;
    const Int* indices;
    const Int* indptr;
    const std::vector<index_t>& permutation;
    const std::vector<index_t>& inverse_permutation;
    const std::vector<double>& scale;
    const Pattern& pattern;
    double* blocks;
    double* diagonal;
    const dense_kernels::Kernels& kernels;
};

// Factor one node into its block (row-major [height x width], positions
// 0..width-1 being the node's own columns and then its tail rows in ascending
// order): zero-fill it, scatter B's entries, apply every contribution of the
// earlier nodes listed by the symbolic phase (in ascending node order), and
// factor its columns in panels. The finished block holds the node's pivots on
// its diagonal and its L entries below it, and is never written again; later
// nodes read it in place as their contributor operand. A node reads only the
// blocks and pivots of its contributors, which are its descendants in the
// elimination tree, so it may run as soon as they are finished. threads is
// the team size its parallel phases may use; with 1 (a phase A worker), the
// thread that calls this does all of the node's work and never calls into the
// executor. track is called whenever the workspace may have grown.
template <typename Int, typename Position, typename Executor, typename Track>
void factor_node(
    std::size_t node, const NumericInput<Int>& input, NodeWorkspace<Position>& workspace,
    Executor& executor, int threads, NumericInfo& info, Track&& track
) {
    const Pattern& pattern = input.pattern;
    const dense_kernels::Kernels& kernels = input.kernels;
    const index_t first = pattern.nodes[node];
    const index_t last = pattern.nodes[node + 1];
    const index_t width = last - first;
    const index_t* const tail = pattern.tail_rows.data() + pattern.tail_ptr[node];
    const index_t tail_rows = pattern.tail_ptr[node + 1] - pattern.tail_ptr[node];
    const index_t height = width + tail_rows;
    Position* const map = workspace.map.data();
    for (index_t i = 0; i < tail_rows; ++i) {
        map[static_cast<std::size_t>(tail[i])] = static_cast<Position>(width + i);
    }
    const auto panel_entries = pattern.block_ptr[node + 1] - pattern.block_ptr[node];
    double* const front = input.blocks + pattern.block_ptr[node];
    std::fill_n(front, panel_entries, 0.0);
    info.maximum_panel_entries = std::max(info.maximum_panel_entries,
        static_cast<std::uint64_t>(panel_entries));
    for (index_t col = first; col < last; ++col) {
        const index_t original_col = input.permutation[static_cast<std::size_t>(col)];
        for (index_t p = static_cast<index_t>(input.indptr[original_col]);
             p < static_cast<index_t>(input.indptr[original_col + 1]); ++p) {
            const index_t original_row = static_cast<index_t>(input.indices[p]);
            const index_t row = input.inverse_permutation[static_cast<std::size_t>(original_row)];
            if (row < col) continue;
            const index_t position = row < last ? row - first :
                static_cast<index_t>(map[static_cast<std::size_t>(row)]);
            if (position < 0) throw std::logic_error("incomplete symmetric panel pattern");
            const double value = input.data[p] * input.scale[static_cast<std::size_t>(original_row)] *
                input.scale[static_cast<std::size_t>(original_col)];
            if (!std::isfinite(value)) throw NonPositiveDefinite{};
            front[static_cast<std::size_t>(position * width + col - first)] = value;
        }
    }
    const auto source_start = std::chrono::steady_clock::now();
    FanIn& fan_in = workspace.fan_in;
    fan_in.contributions.clear();
    fan_in.positions.clear();
    std::size_t right_entries = 0;
    std::size_t largest_operand = 0;
    for (index_t s = pattern.source_ptr[node]; s < pattern.source_ptr[node + 1]; ++s) {
        const Source& source = pattern.sources[static_cast<std::size_t>(s)];
        const auto q = static_cast<std::size_t>(source.node);
        const index_t source_first = pattern.nodes[q];
        const index_t source_width = pattern.nodes[q + 1] - source_first;
        // The source's tail rows from source.begin on: block rows
        // source_width + source.begin, ... of its block (row 0 being the
        // diagonal row source_first), all rows of this node. The first
        // source.columns of them are this node's columns, so the output
        // columns are a prefix of the output rows.
        const index_t* const source_rows = pattern.tail_rows.data() +
            pattern.tail_ptr[q] + source.begin;
        const index_t active_rows = pattern.tail_ptr[q + 1] - pattern.tail_ptr[q] - source.begin;
        const index_t active_cols = source.columns;
        const std::size_t first_position = fan_in.positions.size();
        fan_in.positions.resize(first_position + static_cast<std::size_t>(active_rows));
        index_t* const targets = fan_in.positions.data() + first_position;
        for (index_t i = 0; i < active_cols; ++i) targets[i] = source_rows[i] - first;
        index_t missing = 0;
        for (index_t i = active_cols; i < active_rows; ++i) {
            const auto position = static_cast<index_t>(map[static_cast<std::size_t>(source_rows[i])]);
            targets[i] = position;
            missing |= position;
        }
        if (missing < 0) throw std::logic_error("noncontiguous symmetric contributor");
        // L[R,Q] in place: active_rows full rows of the finished source
        // block (row stride source_width), each holding final, scanned L
        // entries in all source_width columns.
        const double* const source_block = input.blocks + pattern.block_ptr[q] +
            static_cast<std::size_t>((source_width + source.begin) * source_width);
        // Tiny intersections use sparse outer products, retaining the same
        // source order without packing structural zeros. Larger ones are
        // applied as one rank source_width update with right = D_Q L[J,Q].T
        // packed as [source_width x active_cols]; it holds at most the source
        // block's entries, so the sum over distinct contributors cannot
        // overflow.
        const bool sparse = checked_product(active_rows, active_cols) < sparse_update_entries;
        fan_in.contributions.push_back(Contribution{source_block, input.diagonal + source_first,
            source_width, active_rows, active_cols, targets[0], targets[active_rows - 1],
            first_position, sparse ? sparse_update : right_entries});
        if (!sparse) {
            const auto operand = checked_product(source_width, active_cols);
            right_entries += operand;
            largest_operand = std::max(largest_operand, operand);
        }
    }
    const std::uint64_t source_flops = pattern.fan_in_flops[node];
    info.update_flops += source_flops;
    if (!fan_in.contributions.empty()) {
        // The strips depend on the node's shape only; worth_parallel only
        // decides which thread applies which entries.
        const auto strips = static_cast<std::size_t>((height + fan_in_rows - 1) / fan_in_rows);
        const bool parallel = worth_parallel(source_flops, strips, threads);
        double* const operands = workspace.right.reserve(parallel ? right_entries : largest_operand);
        track();
        apply_contributions(front, width, height, fan_in, operands, parallel,
            kernels, executor, info);
    }
    info.source_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - source_start).count();
    // Finiteness. Original entries are tested when scattered. After that no
    // arithmetic result is tested where it is produced; instead every
    // node-rectangle entry on or below the diagonal ends as a pivot or as an
    // L entry, and both are tested: each pivot before it is used (positive
    // and finite), each panel's L entries by one scan after their last write,
    // before a later node reads the block. This is complete because:
    // - entries combine only through +, -, * and division by a tested pivot,
    //   and these keep an Inf or NaN operand non-finite (Inf*0 and Inf-Inf
    //   are NaN), so a non-finite entry stays non-finite until it is tested;
    // - each auxiliary product reaches an entry: a sparse update's weighted =
    //   l * d updates its own diagonal position with the nonzero l; the
    //   diagonal block's weighted = L[i][p] * pivot updates the future pivot
    //   (i, i) with L[i][p]; S[k][p] = d[k] * L[p][k] is formed only when rows
    //   below the block exist, and enters each such row's L[i][p];
    // - a contributor's in-place left operand holds only tested L entries,
    //   and its packed right operand d[k] * l[j][k] enters output entry
    //   (j, j) with the nonzero l[j][k] (a zero l gives a zero product). The
    //   fan-in strip holding row j of that diagonal entry applies the
    //   contribution's rows there, and every column up to that row, so a
    //   kernel part computes the entry and the kernel's flag sees it; the flag
    //   is authoritative, and a false flag (a kernel may stop early) rejects
    //   the attempt before any later use.
    // A rejected attempt assembles nothing and falls back to LU.
    constexpr auto order = dense_kernels::Accumulate::sum_then_subtract;
    // Trailing work is split into fixed strips of node rows, so the partition
    // depends on positions only. Rows are independent: a strip reads only its
    // own rows and the shared, read-only pivots, S and right operand, and
    // writes only its own rows. Sixteen rows (four kernel row groups) keep
    // panels with 256-512 rows below them parallel at eight threads under
    // worth_parallel's two items per thread.
    constexpr index_t strip_rows = 16;
    double* const rect = front;
    const auto panel_start = std::chrono::steady_clock::now();
    for (index_t begin = 0; begin < width; begin += panel_width) {
        const index_t end = std::min(width, begin + panel_width);
        const index_t block = end - begin;
        double* const pivots = input.diagonal + first + begin;
        factor_diagonal_block(rect, width, begin, end, pivots);
        for (index_t i = begin + 1; i < end; ++i) {
            if (!all_finite(rect + i * width + begin, i - begin)) throw NonPositiveDefinite{};
        }
        // Multiply-subtract terms: the diagonal block, then the solve and
        // trailing update below (panel_flops repeats this count).
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
            // Rows [end, width) form the shared right operand, so they are
            // solved and scanned first, in order.
            const index_t columns = width - end;
            solve_panel_rows(rect + end * width + begin, width, columns, pivots, weights, block);
            for (index_t i = end; i < width; ++i) {
                if (!all_finite(rect + i * width + begin, block)) throw NonPositiveDefinite{};
            }
            double* const shared = columns > 0
                ? workspace.right.reserve(checked_product(block, columns)) : nullptr;
            if (columns > 0) {
                const auto pack_start = std::chrono::steady_clock::now();
                track();
                for (index_t i = end; i < width; ++i) {
                    for (index_t k = 0; k < block; ++k) {
                        shared[k * columns + i - end] = rect[i * width + begin + k];
                    }
                }
                info.pack_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pack_start).count();
            }
            // Each strip solves and scans its rows at or below width (rows
            // above it were solved above), packs L_RP D_P, and applies the
            // rank-block update to its rows' entries in columns
            // [end, min(row, width - 1)].
            const auto strips = static_cast<std::size_t>(
                (height - end + strip_rows - 1) / strip_rows);
            // Strips are claimed bottom-up: lower rows update more columns,
            // so the largest items start first.
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
                // Rows at or below width update every column; rows above it
                // update only entries with row >= column.
                const bool finite = row_first >= width
                    ? kernels.gemm_sub(c, ldc, packed, lda, shared, n_columns,
                          m, n_columns, rank, order)
                    : kernels.gemm_sub_lower(c, ldc, packed, lda, shared, n_columns,
                          m, n_columns, rank, static_cast<std::ptrdiff_t>(row_first - end), order);
                if (!finite) throw NonPositiveDefinite{};
            };
            const auto pair_terms = static_cast<std::uint64_t>(block * (block - 1) / 2);
            // Entries with row >= column in rows [end, height) and columns
            // [end, width).
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
    for (index_t i = 0; i < tail_rows; ++i) map[static_cast<std::size_t>(tail[i])] = -1;
}

// Subtree scheduling. With W(s) the summed work estimate of node s and all its
// descendants (pattern.work, accumulated bottom-up) and W_total the sum over
// the roots, a node with W(s) <= W_total / (2 * threads) whose parent exceeds
// that bound, or that is a root, heads a subtree. W only grows towards the
// roots, so the subtrees are disjoint and closed under descendants, and every
// other node (a top node) has only top-node ancestors. (A forest of
// comparable trees makes each tree a subtree instead; see plan_subtrees.)
// When there are at least two subtrees and two threads, phase A factors the
// subtrees concurrently, largest W first, each on one worker that runs its
// nodes serially in ascending order; phase B then factors the top nodes in
// ascending order on the calling thread with the worker team for their
// parallel phases, as before. Otherwise every node is a top node.
//
// The partition depends on the thread count, which is safe for the same
// reason as the fan-in strips (dense_kernels.hpp): a node's block is a
// function of B's entries in its columns and of its contributors' finished
// blocks and pivots only. Its contributors are its descendants (a node
// contributes only to nodes holding its tail rows, its ancestors), which are
// in the same subtree, factored earlier by the same worker, or top nodes
// with smaller ids or subtree nodes, all finished before it in phase B (the
// phase A parallel_for returns only after every worker has stopped). Each
// node applies its contributions in the fixed ascending order of the
// symbolic phase, every output entry receiving the same operations whether
// the fan-in runs whole on one thread or in strips (the kernel contract),
// and factors its panels in position-fixed strips. Nodes write only their own
// blocks and pivots; maps, operand buffers, fan-in lists and gather blocks
// are per worker. So neither the partition nor the thread that factors a
// node can change any bit of the factors.
struct SubtreePlan {
    // Nodes of each subtree, ascending: nodes[ptr[item], ptr[item + 1]),
    // items ordered by decreasing W, which weight holds.
    std::vector<index_t> ptr;
    std::vector<index_t> nodes;
    std::vector<std::uint64_t> weight;
    // Subtree item of each node, or -1 for a top node; empty without
    // phase A.
    std::vector<index_t> item;
    // Phase A team size (zero without phase A) and the largest per-node
    // needs of subtree nodes on the serial path, so that phase A workspaces
    // are sized once, on the calling thread, and never grow on a worker.
    int workers = 0;
    std::size_t operand_entries = 0;
    std::size_t contributions = 0;
    std::size_t positions = 0;

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return sizeof(index_t) * (ptr.capacity() + nodes.capacity() + item.capacity()) +
            sizeof(std::uint64_t) * weight.capacity();
    }
};

// Phase A runs the subtrees and extra_items other work items. Each of its
// workers holds an n-entry row map of map_entry_bytes per entry (4 for
// n < 2^31), one gather block, and an operand buffer, contribution list and
// target positions sized for the largest subtree node, so the team is limited
// to keep all of that within half the bytes of the node block array, with at
// least two workers. The limit bounds the per-thread growth of the workspace;
// it does not change any result.
inline SubtreePlan plan_subtrees(
    const Pattern& pattern, int threads, std::size_t n, std::size_t extra_items,
    std::size_t map_entry_bytes
) {
    SubtreePlan plan;
    const std::size_t node_count = pattern.nodes.size() - 1;
    if (threads < 2 || node_count < 2) return plan;
    std::vector<std::uint64_t> total(node_count, 0);
    std::uint64_t whole = 0;
    for (std::size_t node = 0; node < node_count; ++node) {
        total[node] += pattern.work[node];
        const index_t up = pattern.parent_node[node];
        if (up >= 0) {
            total[static_cast<std::size_t>(up)] += total[node];
        } else {
            whole += total[node];
        }
    }
    std::uint64_t bound = whole / (2 * static_cast<std::uint64_t>(threads));
    // A forest of comparable trees (disconnected components) runs as one
    // subtree per tree when there are at least half as many trees as counted
    // threads (at most eight, as in worth_parallel) and no tree holds more
    // than half of the work: each worker then factors a whole tree at serial
    // speed, while the top nodes of phase B, often dense, scale poorly (four
    // dense 600-wide SPD blocks: 9.3 ms in phase B at eight threads, about
    // 3.5-5 ms as four subtrees).
    std::size_t trees = 0;
    std::uint64_t largest_tree = 0;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (pattern.parent_node[node] >= 0) continue;
        ++trees;
        largest_tree = std::max(largest_tree, total[node]);
    }
    constexpr int counted_threads = 8;
    if (trees >= 2 && 2 * trees >= static_cast<std::size_t>(std::min(threads, counted_threads)) &&
        largest_tree <= whole / 2) {
        bound = std::max(bound, largest_tree);
    }
    std::vector<index_t> roots;
    for (std::size_t node = 0; node < node_count; ++node) {
        const index_t up = pattern.parent_node[node];
        if (total[node] <= bound && (up < 0 || total[static_cast<std::size_t>(up)] > bound)) {
            roots.push_back(static_cast<index_t>(node));
        }
    }
    if (roots.size() < 2) return plan;
    std::sort(roots.begin(), roots.end(), [&](index_t a, index_t b) {
        const auto wa = total[static_cast<std::size_t>(a)];
        const auto wb = total[static_cast<std::size_t>(b)];
        return wa != wb ? wa > wb : a < b;
    });
    plan.item.assign(node_count, -1);
    plan.weight.resize(roots.size());
    for (std::size_t r = 0; r < roots.size(); ++r) {
        plan.item[static_cast<std::size_t>(roots[r])] = static_cast<index_t>(r);
        plan.weight[r] = total[static_cast<std::size_t>(roots[r])];
    }
    // Parents have larger ids, so a descending pass sees a node's parent
    // first; a node below a subtree root joins its parent's subtree.
    for (std::size_t node = node_count; node-- > 0;) {
        const index_t up = pattern.parent_node[node];
        if (plan.item[node] < 0 && up >= 0) plan.item[node] = plan.item[static_cast<std::size_t>(up)];
    }
    plan.ptr.assign(roots.size() + 1, 0);
    for (std::size_t node = 0; node < node_count; ++node) {
        if (plan.item[node] >= 0) ++plan.ptr[static_cast<std::size_t>(plan.item[node] + 1)];
    }
    for (std::size_t r = 0; r < roots.size(); ++r) plan.ptr[r + 1] += plan.ptr[r];
    plan.nodes.resize(static_cast<std::size_t>(plan.ptr.back()));
    std::vector<index_t> next(plan.ptr.begin(), plan.ptr.end() - 1);
    for (std::size_t node = 0; node < node_count; ++node) {
        const index_t item = plan.item[node];
        if (item < 0) continue;
        plan.nodes[static_cast<std::size_t>(next[static_cast<std::size_t>(item)]++)] =
            static_cast<index_t>(node);
        // Serial-path needs of this node: its contribution list, its target
        // positions, and the larger of its largest kernel operand and its
        // first panel's shared operand.
        std::size_t positions = 0;
        std::size_t operand = 0;
        for (index_t s = pattern.source_ptr[node]; s < pattern.source_ptr[node + 1]; ++s) {
            const Source& source = pattern.sources[static_cast<std::size_t>(s)];
            const auto q = static_cast<std::size_t>(source.node);
            const index_t rows = pattern.tail_ptr[q + 1] - pattern.tail_ptr[q] - source.begin;
            positions += static_cast<std::size_t>(rows);
            if (checked_product(rows, source.columns) >= sparse_update_entries) {
                operand = std::max(operand,
                    checked_product(pattern.nodes[q + 1] - pattern.nodes[q], source.columns));
            }
        }
        const index_t width = pattern.nodes[node + 1] - pattern.nodes[node];
        if (width > panel_width) {
            operand = std::max(operand, checked_product(panel_width, width - panel_width));
        }
        plan.contributions = std::max(plan.contributions, static_cast<std::size_t>(
            pattern.source_ptr[node + 1] - pattern.source_ptr[node]));
        plan.positions = std::max(plan.positions, positions);
        plan.operand_entries = std::max(plan.operand_entries, operand);
    }
    // The phase A workspaces together may hold at most half the bytes of
    // the node block array (at least two workers): a row map, a gather block,
    // and buffers sized for the largest subtree node each.
    const std::uint64_t worker_bytes = static_cast<std::uint64_t>(map_entry_bytes) * n +
        sizeof(double) * (plan.operand_entries + AlignedDoubles::alignment +
            static_cast<std::uint64_t>(fan_in_rows * fan_in_columns)) +
        sizeof(index_t) * plan.positions + sizeof(Contribution) * plan.contributions;
    const std::uint64_t budget = sizeof(double) * static_cast<std::uint64_t>(pattern.block_ptr.back()) / 2;
    const std::uint64_t memory_limit = std::max<std::uint64_t>(2, budget / std::max<std::uint64_t>(worker_bytes, 1));
    plan.workers = static_cast<int>(std::min<std::uint64_t>({static_cast<std::uint64_t>(threads),
        static_cast<std::uint64_t>(roots.size() + extra_items), memory_limit}));
    return plan;
}

// The strict lower CSC factor arrays that assembly fills: empty until
// allocated, with entries elements each.
struct CscFactor {
    std::vector<index_t>& rows;
    std::vector<double>& values;
    std::size_t entries;
};

// Allocate (and zero) the CSC arrays that are still empty: concurrently on
// the worker team when there are at least two threads, otherwise on the
// calling thread. Page faults dominate this cost (about 1.8 ms for each array of
// 3-D Poisson of side 20 with COLAMD, 1.86 million entries).
template <typename Executor>
void allocate_factor(CscFactor& csc, Executor& executor, int threads) {
    const auto allocate = [&](std::size_t item, int) {
        if (item == 0) {
            csc.rows.resize(csc.entries);
        } else {
            csc.values.resize(csc.entries);
        }
    };
    if (threads > 1 && csc.rows.size() != csc.entries && csc.values.size() != csc.entries) {
        executor.parallel_for(2, allocate);
        return;
    }
    if (csc.rows.size() != csc.entries) allocate(0, 0);
    if (csc.values.size() != csc.entries) allocate(1, 0);
}

template <typename Position, typename Int, typename Executor>
bool factorize_nodes(
    const NumericInput<Int>& input, CscFactor& csc, Executor& executor, int threads,
    NumericInfo& info, std::uint64_t base_workspace
) {
    const Pattern& pattern = input.pattern;
    const std::size_t node_count = pattern.nodes.size() - 1;
    const std::size_t n = input.permutation.size();
    const std::size_t block_entries = pattern.block_ptr.back();
    const std::size_t gather = checked_product(fan_in_rows, fan_in_columns);
    // Phase A also allocates the two CSC arrays as work items (below).
    constexpr std::size_t allocation_items = 2;
    const SubtreePlan plan = plan_subtrees(pattern, threads, n, allocation_items, sizeof(Position));
    // Workspace 0 belongs to the calling thread in both phases; phase A
    // workers 1, ..., workers - 1 get their own, released after phase A.
    std::vector<NodeWorkspace<Position>> workspaces(static_cast<std::size_t>(std::max(plan.workers, 1)));
    for (std::size_t w = 0; w < workspaces.size(); ++w) {
        NodeWorkspace<Position>& held = workspaces[w];
        held.map.assign(n, Position{-1});
        // Phase B's parallel fan-ins need one gather block per team thread;
        // a phase A worker needs one. Each starts on a 64-byte boundary.
        held.fan_in.scratch.allocate(w == 0
            ? static_cast<std::size_t>(std::max(threads, 1)) * gather : gather);
        if (plan.workers > 0) {
            static_cast<void>(held.right.reserve(plan.operand_entries));
            held.fan_in.contributions.reserve(plan.contributions);
            held.fan_in.positions.reserve(plan.positions);
        }
    }
    // Only the calling thread calls this, and never while phase A runs.
    auto workspace = [&] {
        std::uint64_t bytes = 0;
        for (const auto& held : workspaces) bytes += held.bytes();
        info.workspace_bytes = std::max(info.workspace_bytes, base_workspace + pattern.bytes() +
            plan.bytes() + sizeof(double) * block_entries + bytes +
            sizeof(index_t) * csc.rows.capacity() + sizeof(double) * csc.values.capacity());
    };
    workspace();
    try {
        if (plan.workers > 0) {
            // Phase A. Items are the subtrees and the allocation of each CSC
            // array, a memory-bound job that would otherwise run at assembly
            // on the calling thread; all are claimed in order of decreasing
            // estimated cost (an allocation counted at 8 flop units per
            // entry, like zero-filling a block), so idle workers absorb the
            // allocations. An allocation failure is left for assembly to
            // repeat on the calling thread, so it is reported only once every
            // pivot has passed; a rejected attempt falls back to LU instead
            // (earlier builds allocated both arrays in the symbolic phase and
            // raised the allocation failure before any pivot). A worker
            // abandons its subtree as soon as any item has failed; the
            // failure is rethrown here after all have stopped.
            const std::size_t subtrees = plan.ptr.size() - 1;
            std::vector<std::size_t> order(subtrees + allocation_items);
            for (std::size_t item = 0; item < order.size(); ++item) order[item] = item;
            const std::uint64_t allocation_cost = 8 * static_cast<std::uint64_t>(csc.entries);
            const auto cost = [&](std::size_t item) {
                return item < subtrees ? plan.weight[item] : allocation_cost;
            };
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return cost(a) > cost(b);
            });
            std::vector<NumericInfo> worker_info(workspaces.size());
            const auto subtree_start = std::chrono::steady_clock::now();
            const auto no_growth = [] {};
            executor.parallel_for(order.size(), [&](std::size_t index, int worker) {
                const std::size_t item = order[index];
                if (item >= subtrees) {
                    try {
                        if (item == subtrees) {
                            csc.rows.resize(csc.entries);
                        } else {
                            csc.values.resize(csc.entries);
                        }
                    } catch (const std::exception&) {
                    }
                    return;
                }
                const auto w = static_cast<std::size_t>(worker);
                for (index_t p = plan.ptr[item]; p < plan.ptr[item + 1]; ++p) {
                    if (executor.failed()) return;
                    // One thread per node: no call into the executor.
                    factor_node(static_cast<std::size_t>(plan.nodes[static_cast<std::size_t>(p)]),
                        input, workspaces[w], executor, 1, worker_info[w], no_growth);
                }
            }, plan.workers);
            info.subtree_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - subtree_start).count();
            info.subtrees = static_cast<index_t>(subtrees);
            workspace();
            // The phase timers add the slowest worker's time for each.
            double source = 0.0;
            double panel = 0.0;
            double pack = 0.0;
            for (const NumericInfo& part : worker_info) {
                info.update_flops += part.update_flops;
                info.maximum_panel_entries = std::max(info.maximum_panel_entries,
                    part.maximum_panel_entries);
                source = std::max(source, part.source_seconds);
                panel = std::max(panel, part.panel_seconds);
                pack = std::max(pack, part.pack_seconds);
            }
            info.source_seconds += source;
            info.panel_seconds += panel;
            info.pack_seconds += pack;
            workspaces.resize(1);
        }
        // Phase B.
        for (std::size_t node = 0; node < node_count; ++node) {
            if (!plan.item.empty() && plan.item[node] >= 0) continue;
            factor_node(node, input, workspaces[0], executor, threads, info, workspace);
        }
    } catch (const NonPositiveDefinite&) {
        // Every worker has stopped. Count the CSC arrays that a phase A
        // allocation item may have made before the failure; whether it ran
        // first depends on scheduling, so on this path the estimate can vary
        // between runs.
        workspace();
        return false;
    }
    return true;
}

// Numeric phase of the supernodal LDL.T. Every node's dense rectangle lives in
// one array, blocks (allocated here, pattern.block_ptr[node] being the node's
// offset). Each node zero-fills its own block before scattering into it, so
// blocks of nodes not yet reached are never read. Returns false, with nothing
// assembled, when a pivot or intermediate fails; on success, assemble copies
// the blocks into the CSC factor, whose arrays phase A may already have
// allocated (they are discarded with the attempt on failure).
template <typename Int, typename Executor>
bool factorize(
    const double* data, const Int* indices, const Int* indptr,
    const std::vector<index_t>& permutation,
    const std::vector<index_t>& inverse_permutation,
    const std::vector<double>& scale,
    const Pattern& pattern, std::unique_ptr<double[]>& blocks, CscFactor& csc,
    std::vector<double>& diagonal, const dense_kernels::Kernels& kernels,
    Executor& executor, int threads, NumericInfo& info, std::uint64_t base_workspace
) {
    blocks.reset(new double[pattern.block_ptr.back()]);
    const NumericInput<Int> input{data, indices, indptr, permutation, inverse_permutation,
        scale, pattern, blocks.get(), diagonal.data(), kernels};
    if (permutation.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return factorize_nodes<std::int32_t>(input, csc, executor, threads, info, base_workspace);
    }
    return factorize_nodes<index_t>(input, csc, executor, threads, info, base_workspace);
}

// Write the CSC row indices and values of the strict lower factor from the
// factored node blocks, once all pivots have passed. Column first + c of a
// node stores the node rows c + 1, ..., height - 1, in this order: the node's
// later columns first + c + 1, ..., last - 1, then its tail rows (see
// for_each_node), with the values of block column c in those block rows.
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
    const double* blocks, std::vector<index_t>& rows, std::vector<double>& values,
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
            const index_t* const tail = pattern.tail_rows.data() +
                pattern.tail_ptr[static_cast<std::size_t>(node)];
            const index_t tail_rows = pattern.tail_ptr[static_cast<std::size_t>(node + 1)] -
                pattern.tail_ptr[static_cast<std::size_t>(node)];
            const index_t height = width + tail_rows;
            const index_t begin = start - first;
            const index_t end = std::min(width, stop - first);
            const double* const block = blocks + pattern.block_ptr[static_cast<std::size_t>(node)];
            // Block row r of column col goes to CSC position
            // col_ptr[first + col] + r - (col + 1), that is, base + r.
            index_t base[group];
            for (index_t col = begin; col < end; ++col) {
                const index_t column = first + col;
                const index_t position = col_ptr[static_cast<std::size_t>(column)];
                if (col_ptr[static_cast<std::size_t>(column + 1)] - position != height - 1 - col) {
                    throw std::logic_error("inconsistent symmetric node pattern");
                }
                base[col - begin] = position - (col + 1);
                index_t* const out = rows.data() + position;
                index_t count = 0;
                for (index_t row = column + 1; row < first + width; ++row) out[count++] = row;
                std::copy(tail, tail + tail_rows, out + count);
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
