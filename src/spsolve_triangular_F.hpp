#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "spsolve_triangular.hpp"

namespace nb = nanobind;

template <typename Int>
void spsolve_triangular_F(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<const Int, nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<const Int, nb::ndim<1>, nb::c_contig>& indptr,
    nb::ndarray<double, nb::ndim<2>, nb::f_contig>& rhs,
    bool lower,
    bool unit_diagonal,
    int num_threads
) {
    if (indptr.size() != rhs.shape(0) + 1) {
        throw nb::value_error("indptr length must match the RHS row count");
    }
    spsolve_detail::solve_triangular_csr(
        data.data(), indices.data(), indptr.data(),
        static_cast<std::int64_t>(data.size()),
        static_cast<std::int64_t>(indices.size()),
        rhs.data(), static_cast<std::int64_t>(indptr.size()) - 1,
        static_cast<std::int64_t>(rhs.shape(1)), 1,
        static_cast<std::int64_t>(rhs.shape(0)),
        lower, unit_diagonal, num_threads);
}
