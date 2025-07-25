
bool is_built_with_mkl() {
#ifdef SP_USE_MKL
    return true;
#else
    return false;
#endif
}


#ifdef SP_USE_MKL
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <mkl.h>

namespace nb = nanobind;


template <typename T_A, typename T_b>
int spsolve_pardiso (
    nb::ndarray<const T_A, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<const MKL_INT, nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<const MKL_INT, nb::ndim<1>, nb::c_contig>& indptr,
    nb::ndarray<T_b, nb::ndim<2>, nb::f_contig>& b,
    nb::ndarray<T_b, nb::ndim<2>, nb::f_contig>& x,
    nb::ndarray<MKL_INT, nb::ndim<1>, nb::c_contig>& iparm,
    const MKL_INT& M,
    const MKL_INT& N,
    const MKL_INT& nrhs,
    const MKL_INT& mtype,
    int num_threads
) {

    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/pardiso.html
    const MKL_INT maxfct = 1;
    const MKL_INT mnum   = 1;
    const MKL_INT n      = M;
    const MKL_INT *ia    = indptr.data();
    const MKL_INT *ja    = indices.data();
    MKL_INT phase  = 13; // Analysis, numerical factorization, solve, iterative refinement
    MKL_INT msglvl = 0;
    MKL_INT error  = 0;

    std::vector<MKL_INT> pt(128, 0);
    std::vector<MKL_INT> perm(n, 0);


    // if (num_threads > mkl_get_max_threads()) {
    //     mkl_set_dynamic(1);
    //     mkl_set_num_threads(num_threads);
    //     printf("num_threads = %d, using %d threads\n", mkl_get_max_threads());
    // }


    for (int k=0; k<2; ++k) {

        MKL_INT perror = 0;
        pardiso(
            pt.data(),
            &maxfct, &mnum, &mtype, &phase, &n,
            data.data(), ia, ja, perm.data(), &nrhs,
            iparm.data(), &msglvl,
            b.data(),
            x.data(),
            &perror
        );

        phase = -1; // cleanup
        error = (error==0) ? perror : error;
    }

    return error;
}
#endif