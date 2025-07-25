#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <omp.h>
#include <stdexcept>

namespace nb = nanobind;

/**
 * @brief Solves a sparse triangular linear system (Lx=b or Ux=b) in-place.
 *   This function is a C++/nanobind implementation of a sparse triangular solve,
 *   optimized with OpenMP for multi-threading and AVX2 for vectorization. It operates
 *   directly on NumPy arrays provided from Python.
 * @param data:    NumPy array of non-zero values in the sparse array  (CSR format).
 * @param indices: NumPy array of column indices for the sparse array  (CSR format).
 * @param indptr:  NumPy array of index pointer array for the sparse array (CSR format).
 * @param b: The right-hand side array, which will be modified in-place to store the solution.
 * @param nnz: The number of non-zero elements in the sparse array .
 * @param lower: If true, performs forward substitution. If false, performs backward substitution.
 * @param num_threads: The number of OpenMP threads to use. If <= 0, it defaults to the maximum
 *        number of available threads.
 */
template <typename INT>
void spsolve_triangular_F(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<const INT, nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<const INT, nb::ndim<1>, nb::c_contig>& indptr,
    nb::ndarray<double, nb::ndim<2>, nb::f_contig>& b,
    bool lower, bool unit_diagonal, int num_threads
) {
    // get raw data pointers once at the beginning
    const auto* data_ptr    = data.data();
    const auto* indices_ptr = indices.data();
    const auto* ind_ptr     = indptr.data();

    const auto  nnz   = data.size();
    const auto  M     = indptr.size() - 1; // size of A (M x N where M = N for square matrix)
    const auto  nrhs  = b.shape(1);
    double*     b_ptr = b.data();

    // volatile bool flag_ill_zero_diag = false;

    // // when nrhs is small, grab more columns and use single column solver, to increase CPU usage
    // while ((para_max < 6) && (vec_cols > 4)) {
    //     residue += 4;
    //     vec_cols -= 4;
    //     para_max += vec_cols / 4 + residue; // para_max += 3
    // }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    num_threads = (num_threads<nrhs) ? num_threads : nrhs;


#if defined(_OPENMP) && _OPENMP >= 201503
    #pragma omp parallel num_threads(num_threads) proc_bind(spread)
#else
    #pragma omp parallel num_threads(num_threads)
#endif
    {


        double *b_i_ptr;

        if (lower) {

            // solve Lx=b using forward substitution
            #pragma omp for schedule(guided) nowait
            for (INT col = 0; col < nrhs; ++col) {
                const auto& b_col_ptr = b_ptr + col * M;
                // _mm256_maskload_pd, _mm256_masksave_pd are slow...
                for (INT i = 0; i < M; ++i) {
                    const auto& data_lpos = ind_ptr[i];
                    const auto& data_rpos = ind_ptr[i+1] - 1;
                    if ((i != 0) && (data_lpos > data_rpos)) { continue; } // empty row
                    // if (i != indices_ptr[data_rpos]) { flag_ill_zero_diag = true; continue; }

                    const auto& b_i = b_col_ptr + i;
                    auto b_i_temp = *(b_i);

                    for (INT k = data_lpos; k < data_rpos; ++k) {
                        const auto& j = indices_ptr[k];
                        const auto& v = data_ptr[k];
                        b_i_temp -= v * b_col_ptr[j];
                    }

                    if (!unit_diagonal) {
                        *(b_i) = b_i_temp / data_ptr[data_rpos];
                    } else {
                        *(b_i) = b_i_temp;
                    }

                }
            }

        } else {

            // solve Ux=b using backward substitution
            #pragma omp for schedule(guided) nowait
            for (INT col = 0; col < nrhs; ++col) {
                // _mm256_maskload_pd, _mm256_masksave_pd are slow...
                const auto& M_1 = M-1;
                const auto& b_col_ptr = b_ptr + col * M;
                for (INT i = M_1; i >= 0; --i) {
                    const auto& data_lpos = ind_ptr[i];
                    const auto& data_rpos = ind_ptr[i+1] - 1;
                    if ((i != M_1) && (data_lpos > data_rpos)) { continue; } // empty row
                    // if (i != indices_ptr[data_lpos]) { flag_ill_zero_diag = true; continue; }

                    const auto& b_i = b_col_ptr + i;
                    auto b_i_temp = *(b_i);

                    for (INT k = data_rpos; k > data_lpos; --k) {
                        const auto& j = indices_ptr[k];
                        const auto& v = data_ptr[k];
                        b_i_temp -= v * b_col_ptr[j];
                    }

                    if (!unit_diagonal) {
                        *(b_i) = b_i_temp / data_ptr[data_lpos];
                    } else {
                        *(b_i) = b_i_temp;
                    }

                }
            }

        }


    } // end of #pragma omp

    // if (flag_ill_zero_diag) {
    //     throw std::invalid_argument("ill-conditioned matrix: non-empty row with diag element of 0");
    // }

}