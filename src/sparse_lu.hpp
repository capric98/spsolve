#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

class NativeSparseLU {
public:
    using index_t = std::int64_t;

    struct Entry {
        index_t col;
        double value;
    };

    void solve_inplace(
        nb::ndarray<double, nb::ndim<2>, nb::c_contig>& rhs,
        const std::string& trans,
        int num_threads
    ) const;

    [[nodiscard]] index_t size() const noexcept { return n_; }
    [[nodiscard]] index_t factor_nnz() const noexcept { return factor_nnz_; }
    [[nodiscard]] const std::vector<index_t>& perm_r() const noexcept { return perm_r_; }
    [[nodiscard]] const std::vector<index_t>& perm_c() const noexcept { return perm_c_; }
    [[nodiscard]] const std::vector<double>& row_scale() const noexcept { return row_scale_; }
    [[nodiscard]] const std::vector<double>& col_scale() const noexcept { return col_scale_; }

    [[nodiscard]] std::tuple<std::vector<double>, std::vector<index_t>, std::vector<index_t>>
    export_l_csr() const;

    [[nodiscard]] std::tuple<std::vector<double>, std::vector<index_t>, std::vector<index_t>>
    export_u_csr() const;

    [[nodiscard]] nb::dict stats() const;

public:
    // Internal construction state. The type itself is private to the extension;
    // Python only sees the explicitly bound read-only surface below.
    friend std::shared_ptr<NativeSparseLU> sparse_lu_factorize(
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig>&,
        nb::ndarray<nb::ndim<1>, nb::c_contig>&,
        nb::ndarray<nb::ndim<1>, nb::c_contig>&,
        index_t,
        const std::string&,
        double,
        int,
        bool
    );

    index_t n_ = 0;
    index_t input_nnz_ = 0;
    index_t factor_nnz_ = 0;
    int factor_threads_ = 1;
    std::string ordering_ = "NATURAL";
    std::string ordering_backend_ = "builtin";
    std::string numeric_engine_ = "row-hybrid";
    std::string dense_kernel_;
    double ordering_seconds_ = 0.0;
    double factor_seconds_ = 0.0;
    double symbolic_seconds_ = 0.0;
    double numeric_seconds_ = 0.0;
    double assembly_seconds_ = 0.0;
    double fill_ratio_ = 0.0;
    index_t pivot_swaps_ = 0;
    index_t row_updates_ = 0;
    index_t parallel_steps_ = 0;
    index_t largest_front_ = 0;
    index_t dense_rows_ = 0;
    index_t lu_dense_panels_ = 0;
    index_t lu_parallel_panels_ = 0;
    index_t lu_panel_columns_ = 0;
    std::uint64_t lu_dense_update_flops_ = 0;
    std::uint64_t lu_panel_workspace_bytes_ = 0;
    index_t supernodes_ = 0;
    index_t max_supernode_width_ = 1;
    index_t parallel_panels_ = 0;
    std::uint64_t dense_update_flops_ = 0;
    std::uint64_t peak_workspace_bytes_ = 0;
    std::uint64_t spd_panel_entries_ = 0;
    double failed_spd_seconds_ = 0.0;
    double spd_source_seconds_ = 0.0;
    double spd_panel_seconds_ = 0.0;
    double spd_pack_seconds_ = 0.0;
    index_t spd_subtrees_ = 0;
    double spd_subtree_seconds_ = 0.0;
    double ordering_estimated_l_nnz_ = 0.0;
    double ordering_estimated_max_front_ = 0.0;
    index_t numeric_probe_columns_ = 0;
    index_t numeric_probe_max_front_ = 0;
    std::uint64_t numeric_probe_updates_ = 0;
    double numeric_probe_seconds_ = 0.0;
    bool numeric_probe_fallback_ = false;

    // perm_r_[new row] and perm_c_[new column] contain original indices.
    std::vector<index_t> perm_r_;
    std::vector<index_t> perm_c_;
    std::vector<double> row_scale_;
    std::vector<double> col_scale_;
    bool column_storage_ = false;
    bool symmetric_ldlt_ = false;
    std::vector<double> ldlt_diagonal_;
    std::vector<index_t> l_col_ptr_;
    std::vector<index_t> l_row_indices_;
    std::vector<double> l_values_;
    std::vector<index_t> u_col_ptr_;
    std::vector<index_t> u_row_indices_;
    std::vector<double> u_values_;
    std::vector<std::vector<Entry>> l_rows_;
    std::vector<std::vector<Entry>> u_rows_;
};

std::shared_ptr<NativeSparseLU> sparse_lu_factorize(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indptr,
    NativeSparseLU::index_t n,
    const std::string& permc_spec,
    double diag_pivot_thresh,
    int num_threads,
    bool check_finite
);

void bind_sparse_lu(nb::module_& module);
