#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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

// A parallel phase (contributor tiles or a panel's row strips) runs on the
// worker team only when its serial time spans several dispatches:
// FactorExecutor::parallel_for costs about 20 us at 4-8 threads, and 2 MFLOP
// (2^21) is about 35 us of avx2-fma tile work. It also needs two items per
// thread, counting at most eight threads: a small team is then mostly busy,
// while a larger team does not turn serial an update that eight threads would
// share (its extra workers only cost the dispatch). This is an efficiency
// heuristic. The decision only chooses which thread computes an item: every
// item performs the same arithmetic on either path, so it cannot change any
// result.
[[nodiscard]] inline bool worth_parallel(std::uint64_t flops, std::size_t tiles, int threads) {
    constexpr int counted_threads = 8;
    return threads > 1 && flops >= (std::uint64_t{1} << 21) &&
        tiles >= 2 * static_cast<std::size_t>(std::min(threads, counted_threads));
}

// C -= left * right over the lower part of the selected front entries.
// left is [output rows x rank] with row stride left_stride (>= rank), right is
// [rank x output columns] with row stride equal to the output column count.
// A prior supernode's intersection is applied in one call with rank equal to
// its width: left is that node's block rows read in place (its L entries,
// left_stride being its width) and right its packed D * L.T (a node's own
// panels use the row strips in factorize instead). left is only read, so
// another node's finished block can serve as it. Output tiles are disjoint,
// and every entry visits the rank terms in order: the kernel sums the
// left * right terms from zero in ascending k and subtracts that sum from the
// entry once (Accumulate::sum_then_subtract). Terms are often far smaller
// than the dominant diagonal they update, so summing first avoids rounding at
// the entry's scale once per term. The kernel's finiteness flag is
// authoritative on every tile path, and no path scatters an entry the kernel
// was not asked to compute.
template <typename Executor>
void rank_update(
    double* front, index_t columns,
    const std::vector<index_t>& output_rows,
    const std::vector<index_t>& output_columns,
    const double* left, index_t left_stride,
    const double* right,
    index_t rank, const dense_kernels::Kernels& kernels,
    Executor& executor, int threads, NumericInfo& info
) {
    constexpr index_t tile_size = 16;
    const index_t rows = static_cast<index_t>(output_rows.size());
    const index_t cols = static_cast<index_t>(output_columns.size());
    const index_t row_tiles = (rows + tile_size - 1) / tile_size;
    const index_t col_tiles = (cols + tile_size - 1) / tile_size;
    const auto tiles = checked_product(row_tiles, col_tiles);
    std::uint64_t entries = 0;
    for (index_t row : output_rows) {
        entries += static_cast<std::uint64_t>(std::upper_bound(
            output_columns.begin(), output_columns.end(), row) - output_columns.begin());
    }
    const auto flops = entries * static_cast<std::uint64_t>(2 * rank);
    info.update_flops += flops;
    // Only entries with row >= col are updated. Output rows and columns are
    // ascending front positions; the path depends only on those positions.
    auto update_tile = [&](std::size_t tile, int) {
        const index_t row_first = static_cast<index_t>(tile /
            static_cast<std::size_t>(col_tiles)) * tile_size;
        const index_t col_first = static_cast<index_t>(tile %
            static_cast<std::size_t>(col_tiles)) * tile_size;
        const index_t row_last = std::min(rows, row_first + tile_size);
        const index_t col_last = std::min(cols, col_first + tile_size);
        const index_t top = output_rows[static_cast<std::size_t>(row_first)];
        const index_t bottom = output_rows[static_cast<std::size_t>(row_last - 1)];
        const index_t leftmost = output_columns[static_cast<std::size_t>(col_first)];
        const index_t rightmost = output_columns[static_cast<std::size_t>(col_last - 1)];
        if (bottom < leftmost) return;  // Entirely above the diagonal.
        const auto tile_rows = static_cast<std::size_t>(row_last - row_first);
        const auto tile_cols = static_cast<std::size_t>(col_last - col_first);
        const bool below = top >= rightmost;  // Every entry has row >= col.
        const double* a = left + static_cast<std::size_t>(row_first * left_stride);
        const double* b = right + static_cast<std::size_t>(col_first);
        const auto lda = static_cast<std::size_t>(left_stride);
        const auto ldb = static_cast<std::size_t>(cols);
        constexpr auto order = dense_kernels::Accumulate::sum_then_subtract;
        // The kernel flag is authoritative on every path. A kernel may stop
        // at its first non-finite block, leaving later entries uncomputed, so
        // a false flag rejects the SPD attempt before anything is scattered.
        if (bottom - top == row_last - 1 - row_first &&
            rightmost - leftmost == col_last - 1 - col_first) {
            // A contiguous block of the front: update it in place.
            double* c = front + static_cast<std::size_t>(top * columns + leftmost);
            const auto ldc = static_cast<std::size_t>(columns);
            const bool finite = below
                ? kernels.gemm_sub(c, ldc, a, lda, b, ldb, tile_rows, tile_cols,
                      static_cast<std::size_t>(rank), order)
                : kernels.gemm_sub_lower(c, ldc, a, lda, b, ldb, tile_rows, tile_cols,
                      static_cast<std::size_t>(rank), static_cast<std::ptrdiff_t>(top - leftmost),
                      order);
            if (!finite) throw NonPositiveDefinite{};
            return;
        }
        // Irregular intersection: gather into a scratch block, update it, and
        // scatter back exactly the entries the kernel computed. Output columns
        // are the node's own positions among the ascending output rows, which
        // come first, so output_columns is a prefix of output_rows and
        // output_columns[q] == output_rows[q]. Tiles start at multiples of 16,
        // so a tile that is neither skipped above nor entirely below the
        // diagonal has row_first == col_first: its local entry (i, j) is on
        // or below the diagonal exactly when j <= i, which is what
        // gemm_sub_lower with shift 0 computes.
        constexpr auto stride = static_cast<std::size_t>(tile_size);
        double values[stride * stride];
        const auto row_columns = [&](std::size_t i) {
            return below ? tile_cols : std::min(i + 1, tile_cols);
        };
        for (std::size_t i = 0; i < tile_rows; ++i) {
            const index_t row = output_rows[static_cast<std::size_t>(row_first) + i];
            for (std::size_t j = 0; j < row_columns(i); ++j) {
                values[i * stride + j] = front[static_cast<std::size_t>(row * columns +
                    output_columns[static_cast<std::size_t>(col_first) + j])];
            }
        }
        const bool finite = below
            ? kernels.gemm_sub(values, stride, a, lda, b, ldb, tile_rows, tile_cols,
                  static_cast<std::size_t>(rank), order)
            : kernels.gemm_sub_lower(values, stride, a, lda, b, ldb, tile_rows, tile_cols,
                  static_cast<std::size_t>(rank), 0, order);
        if (!finite) throw NonPositiveDefinite{};
        for (std::size_t i = 0; i < tile_rows; ++i) {
            const index_t row = output_rows[static_cast<std::size_t>(row_first) + i];
            for (std::size_t j = 0; j < row_columns(i); ++j) {
                front[static_cast<std::size_t>(row * columns +
                    output_columns[static_cast<std::size_t>(col_first) + j])] = values[i * stride + j];
            }
        }
    };
    if (worth_parallel(flops, tiles, threads)) {
        executor.parallel_for(tiles, update_tile);
        ++info.parallel_panels;
    } else {
        for (std::size_t tile = 0; tile < tiles; ++tile) update_tile(tile, 0);
    }
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
    std::vector<index_t> output_rows, output_columns;
    std::vector<double> right;
    auto workspace = [&] {
        info.workspace_bytes = std::max(info.workspace_bytes, base_workspace +
            sizeof(index_t) * (row_position.capacity() + mark.capacity() +
                contributors.capacity() + pattern.nodes.capacity() +
                pattern.row_ptr.capacity() + pattern.row_sources.capacity() +
                pattern.column_nodes.capacity() + output_rows.capacity() +
                output_columns.capacity()) +
            sizeof(std::size_t) * pattern.block_ptr.capacity() +
            sizeof(double) * (block_entries + right.capacity()));
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
                // this node's columns, come first: output_columns is a prefix
                // of output_rows.
                const index_t common_row = 1 + static_cast<index_t>(common_begin - source_rows);
                output_rows.clear();
                output_columns.clear();
                for (auto row = common_begin; row != source_end; ++row) {
                    const index_t position = row_position[static_cast<std::size_t>(*row)];
                    if (position < 0) throw std::logic_error("noncontiguous symmetric contributor");
                    output_rows.push_back(position);
                    if (position < width) output_columns.push_back(position);
                }
                const index_t active_rows = static_cast<index_t>(output_rows.size());
                const index_t active_cols = static_cast<index_t>(output_columns.size());
                workspace();
                if (active_cols == 0) throw std::logic_error("empty symmetric contributor");
                // L[R,Q] in place: active_rows full rows of the finished
                // source block (row stride source_width), each holding final,
                // scanned L entries in all source_width columns.
                const double* const source = blocks.get() +
                    pattern.block_ptr[static_cast<std::size_t>(source_node)] +
                    static_cast<std::size_t>(common_row * source_width);
                const double* const pivots = diagonal.data() + source_first;
                if (checked_product(active_rows, active_cols) < 256) {
                    // Tiny intersections use sparse outer products, retaining
                    // the same source order without packing structural zeros:
                    // per source column k, F[i][j] -= (l[j][k] * d[k]) *
                    // l[i][k]. Every target is a front entry, tested later as
                    // a pivot or an L entry (see the finiteness note below).
                    for (index_t k = 0; k < source_width; ++k) {
                        for (index_t j = 0; j < active_cols; ++j) {
                            const double weighted = source[j * source_width + k] * pivots[k];
                            for (index_t i = j; i < active_rows; ++i) {
                                double& value = front[static_cast<std::size_t>(
                                    output_rows[static_cast<std::size_t>(i)] * width +
                                    output_columns[static_cast<std::size_t>(j)])];
                                value -= weighted * source[i * source_width + k];
                            }
                        }
                    }
                    // Each source column applies active_rows - j terms to
                    // output column j.
                    info.update_flops += 2 * static_cast<std::uint64_t>(source_width) *
                        static_cast<std::uint64_t>(active_cols * active_rows -
                            active_cols * (active_cols - 1) / 2);
                    continue;
                }
                // Apply the whole contributor in one rank source_width update,
                // left = L[R,Q] read in place and right = D_Q L[J,Q].T packed
                // as [source_width x active_cols], right[k][j] = d[k] *
                // l[j][k]: each entry sums all of this node's terms
                // l[i][k] * (d[k] * l[j][k]) in ascending source column and
                // subtracts them once. right holds at most source_width times
                // the source node's height, the size of its block.
                right.resize(checked_product(source_width, active_cols));
                workspace();
                const auto pack_start = std::chrono::steady_clock::now();
                for (index_t j = 0; j < active_cols; ++j) {
                    const double* const l = source + j * source_width;
                    // An overflow here reaches the kernel's flag through
                    // output entry (j, j), which every update computes.
                    for (index_t k = 0; k < source_width; ++k) {
                        right[static_cast<std::size_t>(k * active_cols + j)] = pivots[k] * l[k];
                    }
                }
                info.pack_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pack_start).count();
                // left_stride == rank == source_width.
                rank_update(front, width, output_rows, output_columns, source, source_width,
                    right.data(), source_width, kernels, executor, threads, info);
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
            //   a zero product). That diagonal entry lies in a diagonal tile,
            //   which is never skipped, so the kernel's flag sees it; the
            //   flag is authoritative, and a false flag (a kernel may stop
            //   early) rejects the attempt before any later use.
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
                    if (columns > 0) {
                        const auto pack_start = std::chrono::steady_clock::now();
                        right.resize(checked_product(block, columns));
                        workspace();
                        for (index_t i = end; i < width; ++i) {
                            for (index_t k = 0; k < block; ++k) {
                                right[static_cast<std::size_t>(k * columns + i - end)] =
                                    rect[i * width + begin + k];
                            }
                        }
                        info.pack_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pack_start).count();
                    }
                    const double* const shared = right.data();
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
