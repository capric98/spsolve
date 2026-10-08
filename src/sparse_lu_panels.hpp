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

inline std::size_t checked_product(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::vector<double>().max_size() / columns) {
        throw std::overflow_error("LU panel workspace allocation overflowed");
    }
    return rows * columns;
}

// Only dense rows defer updates. Sparse rows retain their original eager
// updates and fill adjacency; retained U rows own all panel source values.
class DensePanel {
public:
    void begin(index_t first, index_t last, index_t n, int threads) {
        first_ = first;
        last_ = last;
        n_ = n;
        if (pending_mark_.empty()) {
            pending_mark_.assign(static_cast<std::size_t>(n), -1);
            conversion_mark_.assign(static_cast<std::size_t>(n), -1);
            column_position_.assign(static_cast<std::size_t>(n), -1);
            local_conversions_.resize(static_cast<std::size_t>(threads));
        }
        pending_.clear();
        for (auto& local : local_conversions_) local.clear();
        conversions_.clear();
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

    template <typename Row, typename Sources>
    void materialize_pivot(index_t row_id, index_t pivot_col,
                           std::vector<Row>& rows, const Sources& sources) {
        if (pending_mark_[static_cast<std::size_t>(row_id)] != first_) return;
        auto& row = rows[static_cast<std::size_t>(row_id)];
        auto lower = std::lower_bound(row.lower.begin(), row.lower.end(), first_,
            [](const auto& entry, index_t col) { return entry.col < col; });
        for (; lower != row.lower.end() && lower->col < pivot_col; ++lower) {
            if (lower->value == 0.0) continue;
            const auto& source = sources[static_cast<std::size_t>(lower->col)];
            auto entry = std::lower_bound(source.begin(), source.end(), last_,
                [](const auto& value, index_t col) { return value.col < col; });
            if (entry == source.end()) continue;
            // The row starts at or before pivot_col < last_; extend its end.
            row.grow_dense(source.back().col + 1, n_);
            for (; entry != source.end(); ++entry) {
                double& value = row.dense[static_cast<std::size_t>(entry->col - row.dense_offset)];
                value -= lower->value * entry->value;
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite value generated during LU factorization");
                }
            }
        }
    }

    template <typename Row, typename Sources, typename Executor>
    void flush(std::vector<Row>& rows, const std::vector<index_t>& positions,
               const Sources& sources, const dense_kernels::Kernels& kernels,
               Executor& executor, int threads) {
        targets_.clear();
        for (index_t row : pending_) {
            if (positions[static_cast<std::size_t>(row)] >= last_ &&
                rows[static_cast<std::size_t>(row)].dense_mode) {
                targets_.push_back(row);
            }
        }
        if (targets_.empty()) { record_workspace(); return; }
        std::sort(targets_.begin(), targets_.end());

        // Clear only the previous support, then discover this panel's union.
        // Neither discovery nor numeric updates scan unrelated zero columns.
        for (index_t col : columns_) column_position_[static_cast<std::size_t>(col)] = -1;
        columns_.clear();
        for (index_t source_col = first_; source_col < last_; ++source_col) {
            const auto& source = sources[static_cast<std::size_t>(source_col)];
            auto entry = std::lower_bound(source.begin(), source.end(), last_,
                [](const auto& value, index_t col) { return value.col < col; });
            for (; entry != source.end(); ++entry) {
                auto& position = column_position_[static_cast<std::size_t>(entry->col)];
                if (position < 0) { position = 0; columns_.push_back(entry->col); }
            }
        }
        if (columns_.empty()) { record_workspace(); return; }
        std::sort(columns_.begin(), columns_.end());
        for (std::size_t col = 0; col < columns_.size(); ++col) {
            column_position_[static_cast<std::size_t>(columns_[col])] = static_cast<index_t>(col);
        }
        // Every target was a candidate inside this panel, so its storage
        // starts before last_ <= columns_.front(). Extend only the ends, and
        // do so serially: resizing invalidates storage used by the tiles.
        for (index_t row : targets_) {
            rows[static_cast<std::size_t>(row)].grow_dense(columns_.back() + 1, n_);
        }

        const std::size_t rank = static_cast<std::size_t>(last_ - first_);
        const std::size_t cols = columns_.size();
        packed_u_.assign(checked_product(rank, cols), 0.0);
        packed_l_.assign(checked_product(targets_.size(), rank), 0.0);
        for (index_t source_col = first_; source_col < last_; ++source_col) {
            const auto& source = sources[static_cast<std::size_t>(source_col)];
            auto entry = std::lower_bound(source.begin(), source.end(), last_,
                [](const auto& value, index_t col) { return value.col < col; });
            for (; entry != source.end(); ++entry) {
                if (!std::isfinite(entry->value)) {
                    throw std::runtime_error("non-finite value generated during LU factorization");
                }
                packed_u_[static_cast<std::size_t>(source_col - first_) * cols +
                    static_cast<std::size_t>(column_position_[static_cast<std::size_t>(entry->col)])] = entry->value;
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
        record_workspace();
        ++flushes_;
        maximum_columns_ = std::max(maximum_columns_, static_cast<index_t>(cols));
        const auto flops = static_cast<std::uint64_t>(targets_.size()) * cols * rank * 2;
        update_flops_ += flops;

        constexpr std::size_t tile_rows = 16;
        constexpr std::size_t tile_columns = 64;
        const auto row_tiles = (targets_.size() + tile_rows - 1) / tile_rows;
        const auto col_tiles = (cols + tile_columns - 1) / tile_columns;
        const auto tiles = checked_product(row_tiles, col_tiles);
        // Each output starts from its current value and subtracts its rank
        // terms in pivot order inside the kernel (Accumulate::running, the
        // recurrence of the eager and materialized updates). The in-place and
        // gathered paths therefore produce identical bits, and the path
        // depends only on this tile's column positions, never on the thread
        // count.
        auto update_tile = [&](std::size_t tile, int) {
            const auto first_row = (tile / col_tiles) * tile_rows;
            const auto first_col = (tile % col_tiles) * tile_columns;
            const auto row_count = std::min(targets_.size(), first_row + tile_rows) - first_row;
            const auto col_count = std::min(cols, first_col + tile_columns) - first_col;
            const double* l = packed_l_.data() + first_row * rank;
            const double* u = packed_u_.data() + first_col;
            const index_t first_column = columns_[first_col];
            bool finite = true;
            if (columns_[first_col + col_count - 1] - first_column ==
                static_cast<index_t>(col_count - 1)) {
                // Contiguous support: every target's dense storage covers
                // these columns (grown above), so update the rows in place.
                double* outputs[tile_rows];
                for (std::size_t i = 0; i < row_count; ++i) {
                    auto& row = rows[static_cast<std::size_t>(targets_[first_row + i])];
                    outputs[i] = row.dense.data() + static_cast<std::size_t>(first_column - row.dense_offset);
                }
                finite = kernels.gemm_sub_rows(outputs, l, rank, u, cols, row_count, col_count, rank,
                    dense_kernels::Accumulate::running);
            } else {
                // Gather, update, and write every scratch entry back only when
                // the kernel flag is true. A kernel may stop at its first
                // non-finite block, so a false flag throws before any
                // possibly uncomputed value can reach a row.
                double values[tile_rows * tile_columns];
                for (std::size_t i = 0; i < row_count; ++i) {
                    const auto& row = rows[static_cast<std::size_t>(targets_[first_row + i])];
                    for (std::size_t j = 0; j < col_count; ++j) {
                        values[i * tile_columns + j] = row.dense[static_cast<std::size_t>(
                            columns_[first_col + j] - row.dense_offset)];
                    }
                }
                finite = kernels.gemm_sub(values, tile_columns, l, rank, u, cols,
                    row_count, col_count, rank, dense_kernels::Accumulate::running);
                if (finite) {
                    for (std::size_t i = 0; i < row_count; ++i) {
                        auto& row = rows[static_cast<std::size_t>(targets_[first_row + i])];
                        for (std::size_t j = 0; j < col_count; ++j) {
                            row.dense[static_cast<std::size_t>(columns_[first_col + j] - row.dense_offset)] =
                                values[i * tile_columns + j];
                        }
                    }
                }
            }
            if (!finite) {
                throw std::runtime_error("non-finite value generated during LU factorization");
            }
        };
        if (threads > 1 && tiles >= 4 && flops >= 262'144) {
            executor.parallel_for(tiles, update_tile);
            ++parallel_flushes_;
        } else {
            for (std::size_t tile = 0; tile < tiles; ++tile) update_tile(tile, 0);
        }
    }

    [[nodiscard]] index_t flushes() const noexcept { return flushes_; }
    [[nodiscard]] index_t parallel_flushes() const noexcept { return parallel_flushes_; }
    [[nodiscard]] index_t maximum_columns() const noexcept { return maximum_columns_; }
    [[nodiscard]] std::uint64_t update_flops() const noexcept { return update_flops_; }
    [[nodiscard]] std::uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }

private:
    void record_workspace() {
        std::uint64_t bytes = sizeof(index_t) * (pending_mark_.capacity() +
            conversion_mark_.capacity() + column_position_.capacity() + pending_.capacity() +
            targets_.capacity() + columns_.capacity() + conversions_.capacity());
        for (const auto& local : local_conversions_) bytes += sizeof(index_t) * local.capacity();
        bytes += sizeof(std::vector<index_t>) * local_conversions_.capacity();
        bytes += sizeof(double) * (packed_u_.capacity() + packed_l_.capacity());
        workspace_bytes_ = std::max(workspace_bytes_, bytes);
    }

    index_t first_ = 0, last_ = 0, n_ = 0;
    index_t flushes_ = 0, parallel_flushes_ = 0, maximum_columns_ = 0;
    std::uint64_t update_flops_ = 0, workspace_bytes_ = 0;
    std::vector<index_t> pending_mark_, conversion_mark_, column_position_;
    std::vector<index_t> pending_, targets_, columns_, conversions_;
    std::vector<std::vector<index_t>> local_conversions_;
    std::vector<double> packed_u_, packed_l_;
};

} // namespace sparse_lu_panels
