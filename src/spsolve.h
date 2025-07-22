#ifndef _SPSOLVE_H_INCL_GUARD
#define _SPSOLVE_H_INCL_GUARD

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


void spsolve_triangular(
    nb::ndarray<const double,  nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<const int, nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<const int, nb::ndim<1>, nb::c_contig>& indptr,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig>& b,
    bool lower, bool unit_diagonal, int num_threads
);


#endif