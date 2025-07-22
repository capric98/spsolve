#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <spsolve.h>

namespace nb = nanobind;


// --- Nanobind Module Definition ---
// This macro creates the entry point for the Python module.
NB_MODULE(_spsolve, m) {
    m.doc() = "Nanobind implementation of a sparse triangular solver with OpenMP and AVX2.";
    // This defines the Python-callable function, mapping it to our C++ function.
    m.def("spsolve_triangular", &spsolve_triangular_C, // backward compatible
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b"),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, c_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
    m.def("spsolve_triangular_C", &spsolve_triangular_C, // backward compatible
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b"),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, c_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
    m.def("spsolve_triangular_F", &spsolve_triangular_F,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b"),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, f_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
}