#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dense_kernels.hpp"

namespace sparse_lu_panels {

using index_t = std::int64_t;
constexpr index_t width = 32;
// Zero gaps narrower than this between packed U column segments are packed as
// exact zero columns, so the flush does not fragment into tiny segments.
constexpr index_t bridge_gap = 16;

inline std::size_t checked_product(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::vector<double>().max_size() / columns) {
        throw std::overflow_error("LU panel workspace allocation overflowed");
    }
    return rows * columns;
}

[[noreturn]] inline void throw_non_finite() {
    throw std::runtime_error("non-finite value generated during LU factorization");
}

// Appends the nonzeros of columns [first, dense_end()) of a dense row to an
// empty U row and rejects a non-finite one. This is the finiteness check of
// every U row built from dense storage (see DensePanel). |x| <= DBL_MAX is
// false exactly for infinities and NaNs, and stays an inline comparison.
template <typename Row, typename Entry>
void append_dense_u_row(const Row& row, index_t first, std::vector<Entry>& output) {
    const double* values = row.dense.data() + static_cast<std::size_t>(first - row.dense_offset);
    const auto count = static_cast<std::size_t>(row.dense_end() - first);
    std::size_t nonzeros = 0;
    for (std::size_t i = 0; i < count; ++i) nonzeros += values[i] != 0.0;
    output.reserve(nonzeros);
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        const double value = values[i];
        if (value != 0.0) {
            output.push_back({first + static_cast<index_t>(i), value});
            finite &= std::abs(value) <= std::numeric_limits<double>::max();
        }
    }
    if (!finite) throw_non_finite();
}

// Only dense rows defer updates. Sparse rows retain their original eager
// updates and fill adjacency.
//
// Lifecycle of a panel [first_, last_): begin(); per pivot, for a dense pivot
// row materialize_pivot() and record_dense_pivot(), and ensure_u_entries()
// when a sparse target needs its U row; eager in-panel updates mark dense
// targets with note_dense_row(); at the panel end flush() applies the
// deferred L_panel * U_panel update to the surviving dense targets and then
// finalizes the panel.
//
// A dense pivot row keeps its dense storage until that flush and is
// immutable after its pivot step, so it is the panel's source for its U
// values: later eager panel updates, later pivots' materialization, and the
// packed U operand read it directly. Its U entries are built only by
// ensure_u_entries(), on demand or when the panel is finalized; finalize()
// then releases the storage. Sparse pivot rows publish their U row at once.
//
// Non-finite detection (completeness argument). Panel arithmetic does not
// test each element it writes: the scalar U-entry walks of sparse sources in
// materialize_pivot() and of sparse pivots in the eager panel updates are
// unchecked, and axpy_sub and the flush GEMM report through their kernel
// flags, which are always tested. Instead every candidate value and pivot,
// every L multiplier, and every U row when its entries are built
// (append_dense_u_row; a sparse row checks each value an update writes) is
// checked. That covers every value the elimination produces (inputs are
// validated separately when check_finite is set). Let a non-finite value be
// stored in column c of row r before r is pivoted at position pos(r). If
// c >= pos(r), the value is part of r's U row and is checked when that row
// is built, before any factor is published. If c < pos(r), the value is
// still there at step c: arithmetic never turns a non-finite operand into a
// finite result, columns at or before the current pivot are never written
// except to clear an already checked coefficient, and deferred updates to
// column c are applied before step c (a flush precedes the next panel, and
// a panel keeps its own columns current). A non-finite value is nonzero, so
// r is a candidate at step c and the candidate check rejects it. Every row
// is pivoted, so no stored value escapes both checks. Detection may come
// later than the arithmetic step that produced the value; an exactly
// singular column reached in between is then reported first. Either way no
// factors are published.
class DensePanel {
public:
    // Nothing is allocated before the first begin(), so a run that never
    // activates a panel reports zero panel workspace.
    void begin(index_t first, index_t last, index_t n, int threads) {
        first_ = first;
        last_ = last;
        n_ = n;
        if (pending_mark_.empty()) {
            pending_mark_.assign(static_cast<std::size_t>(n), -1);
            conversion_mark_.assign(static_cast<std::size_t>(n), -1);
            local_conversions_.resize(static_cast<std::size_t>(threads));
        }
        pending_.clear();
        for (auto& local : local_conversions_) local.clear();
        conversions_.clear();
        const auto rank = static_cast<std::size_t>(last - first);
        pivot_rows_.assign(rank, -1);
        tail_end_.assign(rank, last);
        u_built_.assign(rank, 0);
        record_workspace();
    }

    [[nodiscard]] index_t end() const noexcept { return last_; }

    // The eager dense-column loop calls this serially; row workers only record
    // conversions in their own vectors, with a unique target row per pivot.
    void note_dense_row(index_t row) {
        if (pending_mark_[static_cast<std::size_t>(row)] != first_) {
            pending_mark_[static_cast<std::size_t>(row)] = first_;
            pending_.push_back(row);
        }
    }

    void note_conversion(index_t row, int worker) {
        if (conversion_mark_[static_cast<std::size_t>(row)] != first_) {
            conversion_mark_[static_cast<std::size_t>(row)] = first_;
            local_conversions_[static_cast<std::size_t>(worker)].push_back(row);
        }
    }

    const std::vector<index_t>& conversions() {
        for (const auto& local : local_conversions_) {
            conversions_.insert(conversions_.end(), local.begin(), local.end());
        }
        std::sort(conversions_.begin(), conversions_.end());
        record_workspace();
        return conversions_;
    }

    // Completes the deferred tail [last_, ...) of a dense row selected as the
    // pivot of column pivot_col, subtracting l * u for each earlier panel
    // pivot in ascending order: c = c - round(l * u), the recurrence of the
    // eager updates and of the flush (Accumulate::running), so a row's tail
    // receives identical arithmetic whichever path applies a term.
    template <typename Row, typename Entry>
    void materialize_pivot(index_t row_id, index_t pivot_col, std::vector<Row>& rows,
                           const std::vector<std::vector<Entry>>& u_rows,
                           const dense_kernels::Kernels& kernels) {
        if (pending_mark_[static_cast<std::size_t>(row_id)] != first_) return;
        auto& row = rows[static_cast<std::size_t>(row_id)];
        auto lower = std::lower_bound(row.lower.begin(), row.lower.end(), first_,
            [](const auto& entry, index_t col) { return entry.col < col; });
        for (; lower != row.lower.end() && lower->col < pivot_col; ++lower) {
            const double multiplier = lower->value;
            if (multiplier == 0.0) continue;
            const auto k = static_cast<std::size_t>(lower->col - first_);
            const index_t source_row = pivot_rows_[k];
            if (source_row >= 0) {
                // A retained dense pivot: one contiguous AXPY over its tail.
                const index_t end = tail_end_[k];
                if (end <= last_) continue;
                // The row starts at or before pivot_col < last_; extend its end.
                row.grow_dense(end, n_);
                const auto& source = rows[static_cast<std::size_t>(source_row)];
                if (!kernels.axpy_sub(
                        row.dense.data() + static_cast<std::size_t>(last_ - row.dense_offset),
                        source.dense.data() + static_cast<std::size_t>(last_ - source.dense_offset),
                        multiplier, static_cast<std::size_t>(end - last_))) {
                    throw_non_finite();
                }
                continue;
            }
            const auto& source = u_rows[static_cast<std::size_t>(lower->col)];
            auto entry = std::lower_bound(source.begin(), source.end(), last_,
                [](const auto& value, index_t col) { return value.col < col; });
            if (entry == source.end()) continue;
            row.grow_dense(source.back().col + 1, n_);
            // Unchecked: this row's U row is checked when it is built.
            for (; entry != source.end(); ++entry) {
                row.dense[static_cast<std::size_t>(entry->col - row.dense_offset)] -=
                    multiplier * entry->value;
            }
        }
    }

    // Retains a dense pivot row, already materialized, as the panel source of
    // its U row. tail_end_ is one past its last nonzero at or after last_.
    template <typename Row>
    void record_dense_pivot(index_t row_id, index_t pivot_col, const std::vector<Row>& rows) {
        const auto k = static_cast<std::size_t>(pivot_col - first_);
        const auto& row = rows[static_cast<std::size_t>(row_id)];
        index_t end = row.dense_end();
        while (end > last_ &&
               row.dense[static_cast<std::size_t>(end - 1 - row.dense_offset)] == 0.0) {
            --end;
        }
        pivot_rows_[k] = row_id;
        tail_end_[k] = std::max(end, last_);
    }

    // Builds the U row of a retained dense pivot once. Callers run serially.
    template <typename Row, typename Entry>
    void ensure_u_entries(index_t pivot_col, const std::vector<Row>& rows,
                          std::vector<std::vector<Entry>>& u_rows, index_t& largest_front) {
        const auto k = static_cast<std::size_t>(pivot_col - first_);
        if (pivot_rows_[k] < 0 || u_built_[k] != 0) return;
        auto& output = u_rows[static_cast<std::size_t>(pivot_col)];
        append_dense_u_row(rows[static_cast<std::size_t>(pivot_rows_[k])], pivot_col, output);
        largest_front = std::max(largest_front, static_cast<index_t>(output.size()));
        u_built_[k] = 1;
    }

    // Applies the deferred panel update to the surviving dense targets, then
    // finalizes the panel even when there is nothing to update: the last
    // panel never has targets, and its retained rows still need U rows.
    template <typename Row, typename Entry, typename Executor>
    void flush(std::vector<Row>& rows, const std::vector<index_t>& positions,
               std::vector<std::vector<Entry>>& u_rows, index_t& largest_front,
               const dense_kernels::Kernels& kernels, Executor& executor, int threads) {
        update_targets(rows, positions, u_rows, kernels, executor, threads);
        finalize(rows, u_rows, largest_front);
    }

    [[nodiscard]] index_t flushes() const noexcept { return flushes_; }
    [[nodiscard]] index_t parallel_flushes() const noexcept { return parallel_flushes_; }
    [[nodiscard]] index_t maximum_columns() const noexcept { return maximum_columns_; }
    [[nodiscard]] std::uint64_t update_flops() const noexcept { return update_flops_; }
    [[nodiscard]] std::uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }

private:
    // Packed U columns [start, end) start at packed_offset. Tile chunks reuse
    // the same layout.
    struct Segment {
        index_t start;
        index_t end;
        index_t packed_offset;
    };

    static void push_segment(std::vector<Segment>& segments, index_t start, index_t end) {
        const index_t offset = segments.empty()
            ? 0 : segments.back().packed_offset + (segments.back().end - segments.back().start);
        segments.push_back({start, end, offset});
    }

    // The packed U support is the union of the panel pivots' tails at or after
    // last_, as sorted disjoint column segments. A dense pivot contributes
    // [last_, tail_end_), so the dense pivots together form one segment from
    // last_; sparse pivots contribute their tail columns. Gaps narrower than
    // bridge_gap are packed as zero columns. Only positions decide segments.
    template <typename Entry>
    void build_segments(const std::vector<std::vector<Entry>>& u_rows) {
        segments_.clear();
        sparse_tails_.clear();
        index_t dense_end = last_;
        for (std::size_t k = 0; k < pivot_rows_.size(); ++k) {
            if (pivot_rows_[k] >= 0) {
                dense_end = std::max(dense_end, tail_end_[k]);
                continue;
            }
            const auto& source = u_rows[static_cast<std::size_t>(first_) + k];
            auto entry = std::lower_bound(source.begin(), source.end(), last_,
                [](const auto& value, index_t col) { return value.col < col; });
            for (; entry != source.end(); ++entry) sparse_tails_.push_back(entry->col);
        }
        std::sort(sparse_tails_.begin(), sparse_tails_.end());
        index_t start = last_;
        index_t end = dense_end;
        for (const index_t col : sparse_tails_) {
            if (col < end) continue;
            if (end > start && col - end < bridge_gap) {
                end = col + 1;
                continue;
            }
            if (end > start) push_segment(segments_, start, end);
            start = col;
            end = col + 1;
        }
        if (end > start) push_segment(segments_, start, end);
    }

    template <typename Row, typename Entry, typename Executor>
    void update_targets(std::vector<Row>& rows, const std::vector<index_t>& positions,
                        const std::vector<std::vector<Entry>>& u_rows,
                        const dense_kernels::Kernels& kernels, Executor& executor, int threads) {
        targets_.clear();
        for (index_t row : pending_) {
            if (positions[static_cast<std::size_t>(row)] >= last_ &&
                rows[static_cast<std::size_t>(row)].dense_mode) {
                targets_.push_back(row);
            }
        }
        if (targets_.empty()) { record_workspace(); return; }
        std::sort(targets_.begin(), targets_.end());

        build_segments(u_rows);
        if (segments_.empty()) { record_workspace(); return; }
        // Every target was a candidate inside this panel, so its storage
        // starts before last_ <= segments_.front().start. Extend only the
        // ends, and do so serially: resizing invalidates storage used by the
        // tiles. Each segment is then contiguous in every target.
        const index_t union_end = segments_.back().end;
        for (index_t row : targets_) {
            rows[static_cast<std::size_t>(row)].grow_dense(union_end, n_);
        }

        const std::size_t rank = static_cast<std::size_t>(last_ - first_);
        const auto cols = static_cast<std::size_t>(
            segments_.back().packed_offset + (segments_.back().end - segments_.back().start));
        packed_u_.assign(checked_product(rank, cols), 0.0);
        packed_l_.assign(checked_product(targets_.size(), rank), 0.0);
        for (std::size_t k = 0; k < rank; ++k) {
            double* output = packed_u_.data() + k * cols;
            const index_t source_row = pivot_rows_[k];
            if (source_row >= 0) {
                // A retained dense pivot covers [last_, tail_end_); copy each
                // segment's overlap with it. The rest of its packed row is zero.
                const auto& source = rows[static_cast<std::size_t>(source_row)];
                const index_t end = tail_end_[k];
                for (const Segment& segment : segments_) {
                    if (segment.start >= end) break;
                    const index_t stop = std::min(segment.end, end);
                    std::copy(
                        source.dense.data() + static_cast<std::size_t>(segment.start - source.dense_offset),
                        source.dense.data() + static_cast<std::size_t>(stop - source.dense_offset),
                        output + segment.packed_offset);
                }
                continue;
            }
            const auto& source = u_rows[static_cast<std::size_t>(first_) + k];
            auto entry = std::lower_bound(source.begin(), source.end(), last_,
                [](const auto& value, index_t col) { return value.col < col; });
            std::size_t segment = 0;
            for (; entry != source.end(); ++entry) {
                // Every sparse tail column lies in some segment.
                while (segments_[segment].end <= entry->col) ++segment;
                output[static_cast<std::size_t>(segments_[segment].packed_offset +
                    (entry->col - segments_[segment].start))] = entry->value;
            }
        }
        for (std::size_t target = 0; target < targets_.size(); ++target) {
            const auto& lower = rows[static_cast<std::size_t>(targets_[target])].lower;
            auto entry = std::lower_bound(lower.begin(), lower.end(), first_,
                [](const auto& value, index_t col) { return value.col < col; });
            for (; entry != lower.end() && entry->col < last_; ++entry) {
                packed_l_[target * rank + static_cast<std::size_t>(entry->col - first_)] = entry->value;
            }
        }

        // Tiles are 16 target rows by 64-column chunks of one segment.
        constexpr std::size_t tile_rows = 16;
        constexpr index_t tile_columns = 64;
        chunks_.clear();
        for (const Segment& segment : segments_) {
            for (index_t start = segment.start; start < segment.end; start += tile_columns) {
                chunks_.push_back({start, std::min(segment.end, start + tile_columns),
                                   segment.packed_offset + (start - segment.start)});
            }
        }
        record_workspace();
        ++flushes_;
        maximum_columns_ = std::max(maximum_columns_, static_cast<index_t>(cols));
        const auto flops = static_cast<std::uint64_t>(targets_.size()) * cols * rank * 2;
        update_flops_ += flops;

        const auto row_tiles = (targets_.size() + tile_rows - 1) / tile_rows;
        const auto chunk_count = chunks_.size();
        const auto tiles = checked_product(row_tiles, chunk_count);
        // Each output starts from its current value and subtracts its rank
        // terms in pivot order inside the kernel (Accumulate::running, the
        // recurrence of the eager and materialized updates). Every chunk is
        // contiguous in every target, so tiles update the rows in place; the
        // tiling depends only on column positions, never on the thread count.
        auto update_tile = [&](std::size_t tile, int) {
            const auto first_row = (tile / chunk_count) * tile_rows;
            const Segment& chunk = chunks_[tile % chunk_count];
            const auto row_count = std::min(targets_.size(), first_row + tile_rows) - first_row;
            double* outputs[tile_rows];
            for (std::size_t i = 0; i < row_count; ++i) {
                auto& row = rows[static_cast<std::size_t>(targets_[first_row + i])];
                outputs[i] = row.dense.data() + static_cast<std::size_t>(chunk.start - row.dense_offset);
            }
            // The flag is authoritative: a kernel may stop at its first
            // non-finite block, and the factorization is then discarded.
            if (!kernels.gemm_sub_rows(outputs, packed_l_.data() + first_row * rank, rank,
                    packed_u_.data() + chunk.packed_offset, cols, row_count,
                    static_cast<std::size_t>(chunk.end - chunk.start), rank,
                    dense_kernels::Accumulate::running)) {
                throw_non_finite();
            }
        };
        if (threads > 1 && tiles >= 4 && flops >= 262'144) {
            executor.parallel_for(tiles, update_tile);
            ++parallel_flushes_;
        } else {
            for (std::size_t tile = 0; tile < tiles; ++tile) update_tile(tile, 0);
        }
    }

    // Builds and checks the U rows of the retained dense pivots that do not
    // have them yet, then releases their dense storage.
    template <typename Row, typename Entry>
    void finalize(std::vector<Row>& rows, std::vector<std::vector<Entry>>& u_rows,
                  index_t& largest_front) {
        for (std::size_t k = 0; k < pivot_rows_.size(); ++k) {
            const index_t row_id = pivot_rows_[k];
            if (row_id < 0) continue;
            ensure_u_entries(first_ + static_cast<index_t>(k), rows, u_rows, largest_front);
            auto& row = rows[static_cast<std::size_t>(row_id)];
            std::vector<double>().swap(row.dense);
            row.dense_mode = false;
            pivot_rows_[k] = -1;
        }
    }

    void record_workspace() {
        std::uint64_t bytes = sizeof(index_t) * (pending_mark_.capacity() +
            conversion_mark_.capacity() + pending_.capacity() + targets_.capacity() +
            conversions_.capacity() + pivot_rows_.capacity() + tail_end_.capacity() +
            sparse_tails_.capacity());
        bytes += u_built_.capacity();
        bytes += sizeof(Segment) * (segments_.capacity() + chunks_.capacity());
        for (const auto& local : local_conversions_) bytes += sizeof(index_t) * local.capacity();
        bytes += sizeof(std::vector<index_t>) * local_conversions_.capacity();
        bytes += sizeof(double) * (packed_u_.capacity() + packed_l_.capacity());
        workspace_bytes_ = std::max(workspace_bytes_, bytes);
    }

    index_t first_ = 0, last_ = 0, n_ = 0;
    index_t flushes_ = 0, parallel_flushes_ = 0, maximum_columns_ = 0;
    std::uint64_t update_flops_ = 0, workspace_bytes_ = 0;
    std::vector<index_t> pending_mark_, conversion_mark_;
    std::vector<index_t> pending_, targets_, conversions_;
    std::vector<std::vector<index_t>> local_conversions_;
    // Per panel pivot k = column - first_: the retained dense row (or -1 for
    // a sparse pivot), one past its last nonzero at or after last_, and
    // whether its U row exists.
    std::vector<index_t> pivot_rows_, tail_end_;
    std::vector<unsigned char> u_built_;
    std::vector<index_t> sparse_tails_;
    std::vector<Segment> segments_, chunks_;
    std::vector<double> packed_u_, packed_l_;
};

} // namespace sparse_lu_panels
