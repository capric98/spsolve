#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "sparse_lu.hpp"
#include "spsolve_triangular_C.hpp"
#include "spsolve_triangular_F.hpp"

namespace nb = nanobind;

namespace {

bool is_built_with_mkl() noexcept {
    // Retained for compatibility with releases that optionally linked MKL.
    return false;
}

constexpr const char* triangular_doc = R"doc(
Solve a real CSR triangular system in place.

This is the internal two-dimensional float64 kernel. The Python wrapper handles
canonicalization, dtype/layout conversion, vector RHSs, and complex RHS views.
)doc";

} // namespace

NB_MODULE(_spsolve, module) {
    module.doc() = "Native dependency-free sparse LU and triangular kernels.";

    module.def(
        "spsolve_triangular",
        &spsolve_triangular_C,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0,
        nb::call_guard<nb::gil_scoped_release>(),
        triangular_doc
    );
    module.def(
        "spsolve_triangular",
        &spsolve_triangular_F<std::int32_t>,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0,
        nb::call_guard<nb::gil_scoped_release>(),
        triangular_doc
    );
    module.def(
        "spsolve_triangular",
        &spsolve_triangular_F<std::int64_t>,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0,
        nb::call_guard<nb::gil_scoped_release>(),
        triangular_doc
    );

    bind_sparse_lu(module);
    module.def("is_built_with_mkl", &is_built_with_mkl);
}
