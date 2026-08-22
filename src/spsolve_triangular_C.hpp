#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "spsolve_triangular.hpp"

namespace nb = nanobind;

template <typename Int>
void spsolve_triangular_c_typed(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indices_flex,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indptr_flex,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig>& rhs,
    bool lower,
    bool unit_diagonal,
    int num_threads
) {
    if (indptr_flex.size() != rhs.shape(0) + 1) {
        throw nb::value_error("indptr length must match the RHS row count");
    }
    const auto indices = static_cast<
        nb::ndarray<const Int, nb::ndim<1>, nb::c_contig>>(indices_flex);
    const auto indptr = static_cast<
        nb::ndarray<const Int, nb::ndim<1>, nb::c_contig>>(indptr_flex);
    spsolve_detail::solve_triangular_csr(
        data.data(), indices.data(), indptr.data(),
        static_cast<std::int64_t>(data.size()),
        static_cast<std::int64_t>(indices.size()),
        rhs.data(), static_cast<std::int64_t>(indptr.size()) - 1,
        static_cast<std::int64_t>(rhs.shape(1)),
        static_cast<std::int64_t>(rhs.shape(1)), 1,
        lower, unit_diagonal, num_threads);
}

inline void spsolve_triangular_C(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<nb::ndim<1>, nb::c_contig>& indptr,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig>& rhs,
    bool lower,
    bool unit_diagonal,
    int num_threads
) {
    if (indices.dtype() != indptr.dtype()) {
        throw nb::type_error("indices and indptr must have the same dtype");
    }
    if (indices.dtype() == nb::dtype<std::int32_t>()) {
        spsolve_triangular_c_typed<std::int32_t>(
            data, indices, indptr, rhs, lower, unit_diagonal, num_threads);
        return;
    }
    if (indices.dtype() == nb::dtype<std::int64_t>()) {
        spsolve_triangular_c_typed<std::int64_t>(
            data, indices, indptr, rhs, lower, unit_diagonal, num_threads);
        return;
    }
    throw nb::type_error("indices and indptr must use int32 or int64");
}
