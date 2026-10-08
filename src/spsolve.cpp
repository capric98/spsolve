#include <cstdint>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "dense_kernels.hpp"
#include "sparse_lu.hpp"
#include "spsolve_triangular_C.hpp"
#include "spsolve_triangular_F.hpp"

namespace nb = nanobind;

namespace {

bool is_built_with_mkl() noexcept {
    // Retained for compatibility with releases that optionally linked MKL.
    return false;
}

nb::list available_dense_kernels() {
    nb::list names;
    for (const auto* const* entry = dense_kernels::available(); *entry != nullptr; ++entry) {
        names.append((*entry)->name);
    }
    return names;
}

// None restores the startup selection (SPSOLVE_FORCE_KERNEL or the best
// available kernel). Factorizations already running keep their snapshot.
void set_dense_kernel(const std::optional<std::string>& name) {
    if (!name.has_value()) {
        (void)dense_kernels::select(nullptr);
        return;
    }
    if (name->find('\0') == std::string::npos && dense_kernels::select(name->c_str()) != nullptr) {
        return;
    }
    std::string choices;
    for (const auto* const* entry = dense_kernels::available(); *entry != nullptr; ++entry) {
        choices += choices.empty() ? "'" : ", '";
        choices += (*entry)->name;
        choices += "'";
    }
    throw nb::value_error(("unknown or unsupported dense kernel '" + *name +
        "'; available kernels on this build and CPU: " + choices + ", or 'auto'").c_str());
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
    module.def("_dense_kernel", [] { return std::string(dense_kernels::active_name()); });
    module.def("_available_dense_kernels", &available_dense_kernels);
    module.def("_set_dense_kernel", &set_dense_kernel, nb::arg("name").none());
}
