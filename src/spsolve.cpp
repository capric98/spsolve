#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <spsolve_triangular_C.hpp>
#include <spsolve_triangular_F.hpp>

#ifdef SP_USE_MKL
#include <mkl.h>
#include <spsolve_pardiso.hpp>
#include <complex>
#endif

namespace nb = nanobind;


// --- Nanobind Module Definition ---
// This macro creates the entry point for the Python module.
NB_MODULE(_spsolve, m) {
    m.doc() = "Nanobind implementation of a sparse triangular solver with OpenMP and AVX2.";
    // This defines the Python-callable function, mapping it to our C++ function.
    m.def("spsolve_triangular", &spsolve_triangular_C<int32_t>,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32|int64]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32|int64]]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, c_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
    m.def("spsolve_triangular", &spsolve_triangular_C<int64_t>,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32|int64]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32|int64]]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, c_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
    m.def("spsolve_triangular", &spsolve_triangular_F<int32_t>,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32|int64]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32|int64]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, f_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
    m.def("spsolve_triangular", &spsolve_triangular_F<int64_t>,
        nb::arg("data"),
        nb::arg("indices"),
        nb::arg("indptr"),
        nb::arg("b").noconvert(),
        nb::arg("lower"),
        nb::arg("unit_diagonal") = false,
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    data (np.ndarray[float64]): Row indices of the sparse matrix.\n"
        "    indices (np.ndarray[int32|int64]): Column indices of the sparse matrix.\n"
        "    indptr (np.ndarray[int32|int64]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, f_contig]): The right-hand side matrix.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );


#ifdef SP_USE_MKL

    m.def("spsolve_pardiso", &spsolve_pardiso<double, double>,
        nb::arg("data"), nb::arg("indices"), nb::arg("indptr"), nb::arg("b"), nb::arg("x"),
        nb::arg("iparm"), nb::arg("M"), nb::arg("N"), nb::arg("nrhs"), nb::arg("mtype"),
        nb::arg("num_threads") = 0
    );
    m.def("spsolve_pardiso", &spsolve_pardiso<double, std::complex<double>>,
        nb::arg("data"), nb::arg("indices"), nb::arg("indptr"), nb::arg("b"), nb::arg("x"),
        nb::arg("iparm"), nb::arg("M"), nb::arg("N"), nb::arg("nrhs"), nb::arg("mtype"),
        nb::arg("num_threads") = 0
    );
    m.def("spsolve_pardiso", &spsolve_pardiso<std::complex<double>, double>,
        nb::arg("data"), nb::arg("indices"), nb::arg("indptr"), nb::arg("b"), nb::arg("x"),
        nb::arg("iparm"), nb::arg("M"), nb::arg("N"), nb::arg("nrhs"), nb::arg("mtype"),
        nb::arg("num_threads") = 0
    );
    m.def("spsolve_pardiso", &spsolve_pardiso<std::complex<double>, std::complex<double>>,
        nb::arg("data"), nb::arg("indices"), nb::arg("indptr"), nb::arg("b"), nb::arg("x"),
        nb::arg("iparm"), nb::arg("M"), nb::arg("N"), nb::arg("nrhs"), nb::arg("mtype"),
        nb::arg("num_threads") = 0
    );

#endif

    m.def("is_built_with_mkl", &is_built_with_mkl);

}