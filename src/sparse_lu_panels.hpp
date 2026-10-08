#pragma once

#include <algorithm>
#include <bit>
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

[[noreturn]] inline void throw_non_finite_multiplier() {
    throw std::runtime_error("non-finite multiplier generated during LU factorization");
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
// Lifecycle of a panel [first_, last_): begin(), then gather() once the
// pending dense rows starting inside the panel are admitted. gather() copies
// the panel columns of every member, a dense row that has not been pivoted and
// whose storage meets the panel, into block_, a column-major
// members x (last_ - first_) array. Inside a panel every dense candidate is a
// member: pending rows start at or after last_, conversions wait for the
// flush, and the other dense rows are exact zeros in the panel's columns,
// which only candidates' updates could change. Per pivot column:
// scan_column() finds the members' candidates and their maximum (and
// value_of() reads one member); the caller merges that maximum with the
// sparse candidates'; pivot_dense() or pivot_sparse() records the pivot row's
// in-panel U values; eliminate() divides the block column by the pivot and
// applies one axpy_sub per later panel column with a nonzero U value
// (dgetf2). At the panel end flush() applies the deferred L_panel * U_panel
// update to the surviving targets, appends their panel multipliers to their
// L rows, builds the remaining U rows, and releases the retained rows.
//
// While a panel is active, a member's panel columns live in the block; its
// dense storage there is stale and, like every storage column before a row's
// current pivot column, never read again. Only pivot_dense() writes a block
// row back (the pivot's U values).
//
// Arithmetic identity. A block entry receives exactly the operations the
// row's dense storage received before the block existed: its multiplier is
// value / pivot (division), and each later panel column takes
// c = c - round(l * u) with u the pivot row's current value, in ascending
// pivot order. Members that are not candidates have multiplier +-0, so they
// change at most the sign of a zero; pivoted members' block rows are zeroed,
// and columns where u is zero are skipped, which again only affects zero
// signs. Zeros are never candidates or factor entries.
//
// A dense pivot row keeps its dense storage until the flush and is immutable
// after its pivot step, so it is the panel's source for its U values:
// pivot_dense() writes its block row back to the storage, later pivots'
// materialization and the packed U operand read it directly. Its U entries
// are built once, serially by ensure_u_entries() when a sparse target needs
// them, otherwise in the flush pass; the flush then releases the storage.
// Sparse pivot rows publish their U row at once.
//
// Storage extents follow the row-by-row recurrence: a row that was a
// candidate at panel step k grows, as if it had been updated in place, to
// min(last_, the end of that pivot's row) (step_extent_), so dense_end() and
// the pivot work estimates match it exactly.
//
// Non-finite detection (completeness argument). Panel arithmetic does not
// test each element it writes: the scalar U-entry walks of sparse sources in
// materialize_pivot() are unchecked, while axpy_sub and the flush GEMM report
// through their kernel flags, which are always tested. Instead every candidate
// value and pivot (scan_column() for members), every L multiplier
// (eliminate() tests a whole block column), and every U row when its entries
// are built (append_dense_u_row; a sparse row checks each value an update
// writes) is checked. That covers every value the elimination produces
// (inputs are validated separately when check_finite is set). Let a
// non-finite value be stored in column c of row r before r is pivoted at
// position pos(r), either in r's dense storage or, for c inside the active
// panel, in r's block row. If c >= pos(r), the value is part of r's U row
// (pivot_dense() copies the block row into the storage) and is checked when
// that row is built, before any factor is published. If c < pos(r), the value
// is still there at step c: arithmetic never turns a non-finite operand into
// a finite result, columns at or before the current pivot are never written
// except to clear an already checked coefficient or replace it by its
// checked multiplier, deferred updates to column c are applied before step c (a
// flush precedes the next panel, and the block keeps the panel's columns
// current), and a block column is scanned from the block. A non-finite value
// is nonzero, so r is a candidate at step c and the candidate check rejects
// it. Every row is pivoted, so no stored value escapes both checks. Detection
// may come later than the arithmetic step that produced the value; an exactly
// singular column reached in between is then reported first. Either way no
// factors are published.
class DensePanel {
public:
    // The members' candidates in one panel column: the largest magnitude, ties
    // to the smaller position, and the number of candidates.
    struct Candidate {
        index_t row = -1;
        index_t position = 0;
        double magnitude = 0.0;
        index_t count = 0;
    };

    // Nothing is allocated before the first begin(), so a run that never
    // activates a panel reports zero panel workspace.
    void begin(index_t first, index_t last, index_t n, int threads) {
        first_ = first;
        last_ = last;
        n_ = n;
        if (member_mark_.empty()) {
            member_mark_.assign(static_cast<std::size_t>(n), -1);
            member_slot_.assign(static_cast<std::size_t>(n), 0);
            conversion_mark_.assign(static_cast<std::size_t>(n), -1);
            local_conversions_.resize(static_cast<std::size_t>(threads));
        }
        for (auto& local : local_conversions_) local.clear();
        conversions_.clear();
        members_.clear();
        member_position_.clear();
        const auto rank = static_cast<std::size_t>(last - first);
        pivot_rows_.assign(rank, -1);
        tail_end_.assign(rank, last);
        u_built_.assign(rank, 0);
        std::fill(std::begin(step_extent_), std::end(step_extent_), index_t{0});
        record_workspace();
    }

    // Loads the members from dense_rows, which must already hold every
    // pending row that starts before last_. Member order (and so the block
    // layout) follows dense_rows; no result depends on it.
    template <typename Row>
    void gather(const std::vector<index_t>& dense_rows, const std::vector<Row>& rows,
                const std::vector<index_t>& positions) {
        for (const index_t row_id : dense_rows) {
            const auto& row = rows[static_cast<std::size_t>(row_id)];
            const index_t position = positions[static_cast<std::size_t>(row_id)];
            if (!row.dense_mode || position < first_ || row.dense_offset >= last_ ||
                row.dense_end() <= first_ ||
                member_mark_[static_cast<std::size_t>(row_id)] == first_) {
                continue;
            }
            member_mark_[static_cast<std::size_t>(row_id)] = first_;
            member_slot_[static_cast<std::size_t>(row_id)] = static_cast<index_t>(members_.size());
            members_.push_back(row_id);
            member_position_.push_back(position);
        }
        const std::size_t m = members_.size();
        const auto rank = static_cast<std::size_t>(last_ - first_);
        block_.resize(checked_product(m, rank));
        touched_.assign(m, 0);
        for (std::size_t slot = 0; slot < m; ++slot) {
            const auto& row = rows[static_cast<std::size_t>(members_[slot])];
            // Columns outside the storage are exact zeros.
            const auto start = static_cast<std::size_t>(std::max(first_, row.dense_offset) - first_);
            const auto stop = static_cast<std::size_t>(std::min(last_, row.dense_end()) - first_);
            const double* values = row.dense.data() +
                static_cast<std::size_t>(first_ + static_cast<index_t>(start) - row.dense_offset);
            double* output = block_.data() + slot;
            for (std::size_t k = 0; k < start; ++k) output[k * m] = 0.0;
            for (std::size_t k = start; k < stop; ++k) output[k * m] = values[k - start];
            for (std::size_t k = stop; k < rank; ++k) output[k * m] = 0.0;
        }
        record_workspace();
    }

    [[nodiscard]] index_t end() const noexcept { return last_; }

    [[nodiscard]] bool is_member(index_t row) const noexcept {
        return member_mark_[static_cast<std::size_t>(row)] == first_;
    }

    // Keeps a member's position current after a row swap.
    void set_position(index_t row, index_t position) noexcept {
        if (is_member(row)) {
            member_position_[static_cast<std::size_t>(member_slot_[static_cast<std::size_t>(row)])] = position;
        }
    }

    // The current value of a member in panel column pivot_col.
    [[nodiscard]] double value_of(index_t row, index_t pivot_col) const noexcept {
        return block_[static_cast<std::size_t>(pivot_col - first_) * members_.size() +
                      static_cast<std::size_t>(member_slot_[static_cast<std::size_t>(row)])];
    }

    // One contiguous pass over block column pivot_col: the unpivoted members
    // with a nonzero there are candidates (and targets unless one of them is
    // the pivot). Each is checked for finiteness and marked in touched_.
    [[nodiscard]] Candidate scan_column(index_t pivot_col) {
        Candidate best;
        const std::size_t m = members_.size();
        const auto k = static_cast<std::size_t>(pivot_col - first_);
        const double* column = block_.data() + k * m;
        const std::uint32_t bit = std::uint32_t{1} << k;
        bool finite = true;
        for (std::size_t slot = 0; slot < m; ++slot) {
            const double value = column[slot];
            const index_t position = member_position_[slot];
            if (value == 0.0 || position < pivot_col) continue;
            const double magnitude = std::abs(value);
            finite &= magnitude <= std::numeric_limits<double>::max();
            touched_[slot] |= bit;
            ++best.count;
            if (magnitude > best.magnitude ||
                (magnitude == best.magnitude && position < best.position)) {
                best.magnitude = magnitude;
                best.position = position;
                best.row = members_[slot];
            }
        }
        if (!finite) throw_non_finite();
        return best;
    }

    // A member selected as the pivot of column pivot_col: appends its
    // multipliers of earlier panel pivots to its L row, grows its storage as
    // its in-place updates would have, writes its block row [k, rank) back to
    // the storage as its in-panel U values (u_panel_) and clears it, completes
    // the deferred tail, and retains the row.
    template <typename Row, typename Entry>
    void pivot_dense(index_t row_id, index_t pivot_col, std::vector<Row>& rows,
                     const std::vector<std::vector<Entry>>& u_rows,
                     const dense_kernels::Kernels& kernels) {
        const std::size_t m = members_.size();
        const auto k = static_cast<std::size_t>(pivot_col - first_);
        const auto rank = static_cast<std::size_t>(last_ - first_);
        const auto slot = static_cast<std::size_t>(member_slot_[static_cast<std::size_t>(row_id)]);
        auto& row = rows[static_cast<std::size_t>(row_id)];
        const std::uint32_t earlier = touched_[slot] & ((std::uint32_t{1} << k) - 1);
        append_lower(row, slot, earlier);
        row.grow_dense(step_extent(earlier), n_);
        index_t nonzero_end = pivot_col;
        for (std::size_t j = k; j < rank; ++j) {
            double& value = block_[j * m + slot];
            u_panel_[j] = value;
            if (value != 0.0) nonzero_end = first_ + static_cast<index_t>(j) + 1;
            value = 0.0;
        }
        // The extent above already covers every nonzero (an update that can
        // write a column is one that grows the row to it); stay safe anyway.
        row.grow_dense(nonzero_end, n_);
        // The row was a candidate at pivot_col, so its storage starts at or
        // before it. Storage columns before pivot_col are never read again.
        const index_t stop = std::min(last_, row.dense_end());
        double* storage = row.dense.data();
        for (index_t col = pivot_col; col < stop; ++col) {
            storage[static_cast<std::size_t>(col - row.dense_offset)] =
                u_panel_[static_cast<std::size_t>(col - first_)];
        }
        if (earlier != 0) materialize_pivot(row_id, pivot_col, rows, u_rows, kernels);
        record_dense_pivot(row_id, pivot_col, rows);
        const index_t reach = std::min(last_, row.dense_end());
        step_extent_[k] = pivot_col + 1 < reach ? reach : 0;
    }

    // A sparse pivot row: its U entries inside the panel become u_panel_.
    template <typename Entry>
    void pivot_sparse(index_t pivot_col, const std::vector<Entry>& u) {
        const auto k = static_cast<std::size_t>(pivot_col - first_);
        const auto rank = static_cast<std::size_t>(last_ - first_);
        std::fill(u_panel_ + k + 1, u_panel_ + rank, 0.0);
        auto entry = std::upper_bound(u.begin(), u.end(), pivot_col,
            [](index_t col, const auto& value) { return col < value.col; });
        for (; entry != u.end() && entry->col < last_; ++entry) {
            u_panel_[static_cast<std::size_t>(entry->col - first_)] = entry->value;
        }
        step_extent_[k] = u.empty() ? 0 : std::min(last_, u.back().col + 1);
    }

    // Eliminates panel column pivot_col from the block: multipliers
    // value / pivot_value (division), one finiteness test for the column,
    // then column j -= multipliers * u_panel_[j] for every later column with
    // a nonzero U value. Pivoted members and non-candidates have multiplier
    // +-0, so no target list is needed. block_targets counts the members that
    // are candidates but not the pivot; with none, every multiplier is zero
    // and the block is left as it is.
    void eliminate(index_t pivot_col, double pivot_value, index_t block_targets,
                   const dense_kernels::Kernels& kernels) {
        if (block_targets == 0) return;
        const std::size_t m = members_.size();
        const auto k = static_cast<std::size_t>(pivot_col - first_);
        const auto rank = static_cast<std::size_t>(last_ - first_);
        double* multipliers = block_.data() + k * m;
        bool finite = true;
        for (std::size_t slot = 0; slot < m; ++slot) {
            const double multiplier = multipliers[slot] / pivot_value;
            multipliers[slot] = multiplier;
            finite &= std::abs(multiplier) <= std::numeric_limits<double>::max();
        }
        if (!finite) throw_non_finite_multiplier();
        for (std::size_t j = k + 1; j < rank; ++j) {
            const double u = u_panel_[j];
            if (u == 0.0) continue;
            // The flag is authoritative: the kernel may stop at a non-finite block.
            if (!kernels.axpy_sub(block_.data() + j * m, multipliers, u, m)) throw_non_finite();
        }
    }

    // The worker that updates a sparse row records its conversion; the
    // conversions are applied after the flush.
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

    // Ends the panel. Three kinds of independent work items run in one pass:
    // the deferred L_panel * U_panel update of the surviving targets (output
    // tiles), the targets' panel multipliers appended to their L rows (groups
    // of 16 targets), and the U rows of retained dense pivots not built yet.
    // Each item writes only its own rows' storage, L rows or U row, so the
    // results do not depend on which thread runs it. The pass uses the worker
    // team under the tile update's own condition; afterwards the retained
    // rows are released. The panel is finalized even when there is nothing
    // to update: the last panel never has targets, and its retained rows
    // still need U rows.
    template <typename Row, typename Entry, typename Executor>
    void flush(std::vector<Row>& rows, std::vector<std::vector<Entry>>& u_rows,
               index_t& largest_front, const dense_kernels::Kernels& kernels,
               Executor& executor, int threads) {
        collect_targets(rows);
        const bool update = prepare_update(rows, u_rows);
        open_u_.clear();
        for (std::size_t k = 0; k < pivot_rows_.size(); ++k) {
            if (pivot_rows_[k] >= 0 && u_built_[k] == 0) open_u_.push_back(static_cast<index_t>(k));
        }
        record_workspace();

        constexpr std::size_t tile_rows = 16;
        const std::size_t rank = static_cast<std::size_t>(last_ - first_);
        const std::size_t chunk_count = update ? chunks_.size() : 0;
        const std::size_t row_tiles = (targets_.size() + tile_rows - 1) / tile_rows;
        const std::size_t tiles = update ? checked_product(row_tiles, chunk_count) : 0;
        const std::size_t items = tiles + row_tiles + open_u_.size();
        const std::size_t cols = update ? static_cast<std::size_t>(
            segments_.back().packed_offset + (segments_.back().end - segments_.back().start)) : 0;
        // Items run U rows first, then L groups, then tiles, so the larger
        // irregular items do not trail. Each output starts from its current
        // value and subtracts its rank terms in pivot order inside the kernel
        // (Accumulate::running, the recurrence of the block and materialized
        // updates). Every chunk is contiguous in every target, so tiles update
        // the rows in place; the tiling depends only on column positions,
        // never on the thread count.
        auto work = [&](std::size_t item, int) {
            if (item < open_u_.size()) {
                // No other item reads a retained row (packed_u_ is a copy),
                // so its storage is released here.
                const auto k = static_cast<std::size_t>(open_u_[item]);
                auto& row = rows[static_cast<std::size_t>(pivot_rows_[k])];
                append_dense_u_row(row, first_ + static_cast<index_t>(k),
                                   u_rows[static_cast<std::size_t>(first_) + k]);
                std::vector<double>().swap(row.dense);
                row.dense_mode = false;
                return;
            }
            item -= open_u_.size();
            if (item < row_tiles) {
                const auto stop = std::min(targets_.size(), (item + 1) * tile_rows);
                for (std::size_t target = item * tile_rows; target < stop; ++target) {
                    const index_t row_id = targets_[target];
                    const auto slot = static_cast<std::size_t>(member_slot_[static_cast<std::size_t>(row_id)]);
                    append_lower(rows[static_cast<std::size_t>(row_id)], slot, touched_[slot]);
                }
                return;
            }
            item -= row_tiles;
            {
                const auto first_row = (item / chunk_count) * tile_rows;
                const Segment& chunk = chunks_[item % chunk_count];
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
            }
        };
        if (update && threads > 1 && tiles >= 4 && last_flops_ >= 262'144) {
            executor.parallel_for(items, work);
            ++parallel_flushes_;
        } else {
            for (std::size_t item = 0; item < items; ++item) work(item, 0);
        }

        for (const index_t k : open_u_) {
            largest_front = std::max(largest_front,
                static_cast<index_t>(u_rows[static_cast<std::size_t>(first_ + k)].size()));
            u_built_[static_cast<std::size_t>(k)] = 1;
        }
        // Release the retained rows whose U rows were built earlier.
        for (std::size_t k = 0; k < pivot_rows_.size(); ++k) {
            const index_t row_id = pivot_rows_[k];
            if (row_id < 0) continue;
            auto& row = rows[static_cast<std::size_t>(row_id)];
            if (row.dense_mode) {
                std::vector<double>().swap(row.dense);
                row.dense_mode = false;
            }
            pivot_rows_[k] = -1;
        }
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

    // Appends a member's multipliers for the panel columns in bits, in
    // ascending column order, to its L row: exactly the entries in-place
    // updates would have appended, one per step at which it was a target.
    // Reads only the block, so workers may run it for distinct rows.
    template <typename Row>
    void append_lower(Row& row, std::size_t slot, std::uint32_t bits) const {
        const std::size_t m = members_.size();
        for (; bits != 0; bits &= bits - 1) {
            const auto k = static_cast<std::size_t>(std::countr_zero(bits));
            row.lower.push_back({first_ + static_cast<index_t>(k), block_[k * m + slot]});
        }
    }

    // The storage end the in-place updates of the panel steps in bits would
    // have requested (0 for none).
    [[nodiscard]] index_t step_extent(std::uint32_t bits) const noexcept {
        index_t extent = 0;
        for (; bits != 0; bits &= bits - 1) {
            extent = std::max(extent, step_extent_[std::countr_zero(bits)]);
        }
        return extent;
    }

    // Completes the deferred tail [last_, ...) of a member selected as the
    // pivot of column pivot_col, subtracting l * u for each earlier panel
    // pivot in ascending order: c = c - round(l * u), the recurrence of the
    // block updates and of the flush (Accumulate::running), so a row's tail
    // receives identical arithmetic whichever path applies a term.
    template <typename Row, typename Entry>
    void materialize_pivot(index_t row_id, index_t pivot_col, std::vector<Row>& rows,
                           const std::vector<std::vector<Entry>>& u_rows,
                           const dense_kernels::Kernels& kernels) {
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

    // The flush targets are the members still unpivoted that were candidates
    // in the panel, sorted by row id. Their storage takes, serially, the
    // extent of their in-panel updates; their L entries are appended by the
    // flush pass. Storage columns inside the panel keep stale values: like
    // every column before the current pivot, they are never read again.
    template <typename Row>
    void collect_targets(std::vector<Row>& rows) {
        targets_.clear();
        for (std::size_t slot = 0; slot < members_.size(); ++slot) {
            if (touched_[slot] != 0 && member_position_[slot] >= last_) {
                targets_.push_back(members_[slot]);
            }
        }
        std::sort(targets_.begin(), targets_.end());
        for (const index_t row_id : targets_) {
            const auto slot = static_cast<std::size_t>(member_slot_[static_cast<std::size_t>(row_id)]);
            rows[static_cast<std::size_t>(row_id)].grow_dense(step_extent(touched_[slot]), n_);
        }
    }

    // The packed U support covers the panel pivots' nonzeros at or after
    // last_, as sorted disjoint column segments. The retained dense pivots'
    // tails [last_, tail_end_) are OR-ed into a column mask, and the mask's
    // columns are merged with the sparse pivots' tail columns; gaps narrower
    // than bridge_gap are packed as zero columns, wider gaps split segments.
    // Only positions decide segments.
    template <typename Row, typename Entry>
    void build_segments(const std::vector<Row>& rows, const std::vector<std::vector<Entry>>& u_rows) {
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
        const auto span = static_cast<std::size_t>(dense_end - last_);
        column_mask_.assign(span, 0);
        for (std::size_t k = 0; k < pivot_rows_.size(); ++k) {
            if (pivot_rows_[k] < 0 || tail_end_[k] <= last_) continue;
            const auto& source = rows[static_cast<std::size_t>(pivot_rows_[k])];
            const double* values = source.dense.data() + static_cast<std::size_t>(last_ - source.dense_offset);
            const auto count = static_cast<std::size_t>(tail_end_[k] - last_);
            for (std::size_t i = 0; i < count; ++i) {
                column_mask_[i] |= static_cast<unsigned char>(values[i] != 0.0);
            }
        }
        // Columns arrive in ascending order; a column inside the open segment
        // is already covered.
        bool open = false;
        index_t start = 0;
        index_t end = 0;
        const auto take = [&](index_t col) {
            if (open) {
                if (col < end) return;
                if (col - end < bridge_gap) {
                    end = col + 1;
                    return;
                }
                push_segment(segments_, start, end);
            }
            open = true;
            start = col;
            end = col + 1;
        };
        std::size_t next = 0;
        for (std::size_t i = 0; i < span; ++i) {
            if (column_mask_[i] == 0) continue;
            const index_t col = last_ + static_cast<index_t>(i);
            for (; next < sparse_tails_.size() && sparse_tails_[next] < col; ++next) {
                take(sparse_tails_[next]);
            }
            take(col);
        }
        for (; next < sparse_tails_.size(); ++next) take(sparse_tails_[next]);
        if (open) push_segment(segments_, start, end);
    }

    // Prepares the deferred update of the targets: segments, target storage
    // extended to the union end, packed operands, and 64-column tile chunks.
    // Returns false when there is nothing to update.
    template <typename Row, typename Entry>
    bool prepare_update(std::vector<Row>& rows, const std::vector<std::vector<Entry>>& u_rows) {
        if (targets_.empty()) return false;

        build_segments(rows, u_rows);
        if (segments_.empty()) return false;
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
                // A retained dense pivot's nonzeros at or after last_ lie in
                // [last_, tail_end_) and inside the segments; copy each
                // segment's overlap with that range. The rest of its packed row
                // is zero.
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
        // The block's multiplier columns, transposed. A target's entry is its
        // multiplier where it was a candidate and exactly zero elsewhere.
        const std::size_t m = members_.size();
        for (std::size_t target = 0; target < targets_.size(); ++target) {
            const auto slot = static_cast<std::size_t>(
                member_slot_[static_cast<std::size_t>(targets_[target])]);
            double* output = packed_l_.data() + target * rank;
            for (std::uint32_t bits = touched_[slot]; bits != 0; bits &= bits - 1) {
                const auto k = static_cast<std::size_t>(std::countr_zero(bits));
                output[k] = block_[k * m + slot];
            }
        }

        // Tiles are 16 target rows by 64-column chunks of one segment.
        constexpr index_t tile_columns = 64;
        chunks_.clear();
        for (const Segment& segment : segments_) {
            for (index_t start = segment.start; start < segment.end; start += tile_columns) {
                chunks_.push_back({start, std::min(segment.end, start + tile_columns),
                                   segment.packed_offset + (start - segment.start)});
            }
        }
        ++flushes_;
        maximum_columns_ = std::max(maximum_columns_, static_cast<index_t>(cols));
        last_flops_ = static_cast<std::uint64_t>(targets_.size()) * cols * rank * 2;
        update_flops_ += last_flops_;
        return true;
    }

    void record_workspace() {
        std::uint64_t bytes = sizeof(index_t) * (member_mark_.capacity() +
            member_slot_.capacity() + conversion_mark_.capacity() + members_.capacity() +
            member_position_.capacity() + targets_.capacity() + conversions_.capacity() +
            pivot_rows_.capacity() + tail_end_.capacity() + open_u_.capacity() +
            sparse_tails_.capacity());
        bytes += sizeof(std::uint32_t) * touched_.capacity();
        bytes += u_built_.capacity() + column_mask_.capacity();
        bytes += sizeof(Segment) * (segments_.capacity() + chunks_.capacity());
        for (const auto& local : local_conversions_) bytes += sizeof(index_t) * local.capacity();
        bytes += sizeof(std::vector<index_t>) * local_conversions_.capacity();
        bytes += sizeof(double) * (block_.capacity() + packed_u_.capacity() + packed_l_.capacity());
        workspace_bytes_ = std::max(workspace_bytes_, bytes);
    }

    index_t first_ = 0, last_ = 0, n_ = 0;
    index_t flushes_ = 0, parallel_flushes_ = 0, maximum_columns_ = 0;
    std::uint64_t update_flops_ = 0, workspace_bytes_ = 0, last_flops_ = 0;
    // Membership: member_mark_[row] == first_ marks a member of this panel,
    // whose slot in the block is member_slot_[row] (both n-sized).
    std::vector<index_t> member_mark_, member_slot_, conversion_mark_;
    // Per slot: row id, current position, and the panel columns at which the
    // member was a candidate (bit k for column first_ + k).
    std::vector<index_t> members_, member_position_;
    std::vector<std::uint32_t> touched_;
    // Column k of the panel occupies block_[k * members, (k + 1) * members).
    std::vector<double> block_;
    std::vector<index_t> targets_, conversions_;
    std::vector<std::vector<index_t>> local_conversions_;
    // Per panel pivot k = column - first_: the retained dense row (or -1 for
    // a sparse pivot), one past its last nonzero at or after last_, and
    // whether its U row exists.
    std::vector<index_t> pivot_rows_, tail_end_;
    std::vector<unsigned char> u_built_;
    // Panel steps whose retained pivot still needs its U row at the flush.
    std::vector<index_t> open_u_;
    // The current pivot's U values in the panel, and per panel step the
    // storage end its in-place updates request (0 for none).
    double u_panel_[width] = {};
    index_t step_extent_[width] = {};
    std::vector<index_t> sparse_tails_;
    std::vector<unsigned char> column_mask_;
    std::vector<Segment> segments_, chunks_;
    std::vector<double> packed_u_, packed_l_;
};

} // namespace sparse_lu_panels
