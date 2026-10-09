#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
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
// Output tiles are disjoint, and every entry visits the rank terms in order:
// the kernel sums the left * right terms from zero in ascending k and
// subtracts that sum from the entry once (Accumulate::sum_then_subtract).
// Terms are often far smaller than the dominant diagonal they update, so
// summing first avoids rounding at the entry's scale once per term. A prior
// supernode's intersection is applied in one call with rank equal to its
// width (a node's own panels use the row strips in factorize instead). The
// kernel's finiteness flag is authoritative on every tile path, and no path
// scatters an entry the kernel was not asked to compute.
template <typename Executor>
void rank_update(
    std::vector<double>& front, index_t columns,
    const std::vector<index_t>& output_rows,
    const std::vector<index_t>& output_columns,
    const std::vector<double>& left, index_t left_stride,
    const std::vector<double>& right,
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
        const double* a = left.data() + static_cast<std::size_t>(row_first * left_stride);
        const double* b = right.data() + static_cast<std::size_t>(col_first);
        const auto lda = static_cast<std::size_t>(left_stride);
        const auto ldb = static_cast<std::size_t>(cols);
        constexpr auto order = dense_kernels::Accumulate::sum_then_subtract;
        // The kernel flag is authoritative on every path. A kernel may stop
        // at its first non-finite block, leaving later entries uncomputed, so
        // a false flag rejects the SPD attempt before anything is scattered.
        if (bottom - top == row_last - 1 - row_first &&
            rightmost - leftmost == col_last - 1 - col_first) {
            // A contiguous block of the front: update it in place.
            double* c = front.data() + static_cast<std::size_t>(top * columns + leftmost);
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

template <typename Int, typename Executor>
bool factorize(
    const double* data, const Int* indices, const Int* indptr,
    const std::vector<index_t>& permutation,
    const std::vector<index_t>& inverse_permutation,
    const std::vector<double>& scale,
    const std::vector<index_t>& col_ptr, const std::vector<index_t>& rows,
    const Pattern& pattern, std::vector<double>& values,
    std::vector<double>& diagonal, const dense_kernels::Kernels& kernels,
    Executor& executor, int threads, NumericInfo& info, std::uint64_t base_workspace
) {
    const index_t n = static_cast<index_t>(permutation.size());
    std::vector<index_t> row_position(static_cast<std::size_t>(n), -1);
    std::vector<index_t> mark(static_cast<std::size_t>(n), -1);
    std::vector<index_t> contributors;
    std::vector<index_t> output_rows, output_columns, source_offsets;
    std::vector<double> front, left, right;
    auto workspace = [&] {
        info.workspace_bytes = std::max(info.workspace_bytes, base_workspace +
            sizeof(index_t) * (row_position.capacity() + mark.capacity() +
                contributors.capacity() + pattern.nodes.capacity() +
                pattern.row_ptr.capacity() + pattern.row_sources.capacity() +
                pattern.column_nodes.capacity() + output_rows.capacity() +
                output_columns.capacity() + source_offsets.capacity()) +
            sizeof(double) * (front.capacity() + left.capacity() + right.capacity()));
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
            const auto panel_entries = checked_product(height, width);
            front.assign(panel_entries, 0.0);
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
                const auto common_begin = std::lower_bound(rows.begin() +
                    col_ptr[static_cast<std::size_t>(source_first)], rows.begin() +
                    col_ptr[static_cast<std::size_t>(source_first + 1)], first);
                output_rows.clear();
                output_columns.clear();
                source_offsets.clear();
                for (auto row = common_begin; row != rows.begin() +
                     col_ptr[static_cast<std::size_t>(source_first + 1)]; ++row) {
                    const index_t position = row_position[static_cast<std::size_t>(*row)];
                    if (position < 0) continue;
                    output_rows.push_back(position);
                    source_offsets.push_back(static_cast<index_t>(row - common_begin));
                    if (position < width) output_columns.push_back(position);
                }
                const index_t active_rows = static_cast<index_t>(output_rows.size());
                const index_t active_cols = static_cast<index_t>(output_columns.size());
                workspace();
                if (active_cols == 0) throw std::logic_error("empty symmetric contributor");
                if (checked_product(active_rows, active_cols) < 256) {
                    // Tiny intersections use sparse outer products, retaining
                    // the same source order without packing structural zeros.
                    // Every target is a front entry, tested later as a pivot
                    // or an L entry (see the finiteness note below).
                    for (index_t source = source_first; source < source_last; ++source) {
                        const auto source_begin = std::lower_bound(rows.begin() +
                            col_ptr[static_cast<std::size_t>(source)], rows.begin() +
                            col_ptr[static_cast<std::size_t>(source + 1)], first);
                        const index_t offset = static_cast<index_t>(source_begin - rows.begin());
                        for (index_t j = 0; j < active_cols; ++j) {
                            const double weighted = values[static_cast<std::size_t>(
                                offset + source_offsets[static_cast<std::size_t>(j)])] *
                                diagonal[static_cast<std::size_t>(source)];
                            for (index_t i = j; i < active_rows; ++i) {
                                double& value = front[static_cast<std::size_t>(
                                    output_rows[static_cast<std::size_t>(i)] * width +
                                    output_columns[static_cast<std::size_t>(j)])];
                                value -= weighted * values[static_cast<std::size_t>(
                                    offset + source_offsets[static_cast<std::size_t>(i)])];
                            }
                        }
                    }
                    // Each source column applies active_rows - j terms to
                    // output column j.
                    info.update_flops += 2 * static_cast<std::uint64_t>(source_last - source_first) *
                        static_cast<std::uint64_t>(active_cols * active_rows -
                            active_cols * (active_cols - 1) / 2);
                    continue;
                }
                // Pack the whole contributor once, left = L[R,Q] D_Q as
                // [active_rows x source_width] and right = L[J,Q].T as
                // [source_width x active_cols], and apply it in one rank
                // source_width update: each entry sums all of this node's
                // terms in ascending source column and subtracts them once.
                // Each buffer holds at most source_width times the source
                // node's height, the size of that node's dense rectangle.
                const index_t source_width = source_last - source_first;
                left.resize(checked_product(active_rows, source_width));
                right.resize(checked_product(source_width, active_cols));
                workspace();
                const auto pack_start = std::chrono::steady_clock::now();
                for (index_t k = 0; k < source_width; ++k) {
                    const index_t source = source_first + k;
                    const auto source_begin = std::lower_bound(rows.begin() +
                        col_ptr[static_cast<std::size_t>(source)], rows.begin() +
                        col_ptr[static_cast<std::size_t>(source + 1)], first);
                    const index_t offset = static_cast<index_t>(source_begin - rows.begin());
                    for (index_t i = 0; i < active_rows; ++i) {
                        const double value = values[static_cast<std::size_t>(
                            offset + source_offsets[static_cast<std::size_t>(i)])];
                        // An overflow here reaches the kernel's flag through
                        // output column 0, which every packed row updates.
                        const double weighted = value * diagonal[static_cast<std::size_t>(source)];
                        left[static_cast<std::size_t>(i * source_width + k)] = weighted;
                        if (i < active_cols) right[static_cast<std::size_t>(k * active_cols + i)] = value;
                    }
                }
                info.pack_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pack_start).count();
                // left_stride == rank == source_width.
                rank_update(front, width, output_rows, output_columns, left, source_width, right,
                    source_width, kernels, executor, threads, info);
            }
            info.source_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - source_start).count();
            // Finiteness. Original entries are tested when scattered. After
            // that no arithmetic result is tested where it is produced;
            // instead every node-rectangle entry on or below the diagonal ends
            // as a pivot or as an L entry, and both are tested: each pivot
            // before it is used (positive and finite), each panel's L
            // entries by one scan after their last write, before the node is
            // published. This is complete because:
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
            // - packed kernel operands (l * d) reach the kernel's flag, which
            //   is authoritative: every packed row updates output column 0,
            //   which each kernel call computes, and a false flag (a kernel
            //   may stop early) rejects the attempt before any later use.
            // A rejected attempt publishes nothing and falls back to LU.
            constexpr auto order = dense_kernels::Accumulate::sum_then_subtract;
            // Trailing work is split into fixed strips of node rows, so the
            // partition depends on positions only. Rows are independent: a
            // strip reads only its own rows and the shared, read-only pivots,
            // S and right operand, and writes only its own rows. Sixteen rows
            // (four kernel row groups) keep panels with 256-512 rows below
            // them parallel at eight threads under worth_parallel's two items
            // per thread.
            constexpr index_t strip_rows = 16;
            double* const rect = front.data();
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
            for (index_t col = first; col < last; ++col) {
                for (index_t p = col_ptr[static_cast<std::size_t>(col)];
                     p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
                    const index_t position = row_position[static_cast<std::size_t>(rows[p])];
                    values[static_cast<std::size_t>(p)] = front[static_cast<std::size_t>(
                        position * width + col - first)];
                }
            }
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

} // namespace sparse_ldlt
