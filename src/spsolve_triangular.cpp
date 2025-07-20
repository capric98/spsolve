#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>

namespace nb = nanobind;

/**
 * @brief Solves a sparse triangular linear system (Lx=b or Ux=b) in-place.
 *   This function is a C++/nanobind implementation of a sparse triangular solve,
 *   optimized with OpenMP for multi-threading and AVX2 for vectorization. It operates
 *   directly on NumPy arrays provided from Python.
 * @param rows NumPy array of row indices for the sparse matrix (COO format).
 * @param cols NumPy array of column indices for the sparse matrix (COO format).
 * @param vals NumPy array of non-zero values for the sparse matrix (COO format).
 * @param b The right-hand side matrix, which will be modified in-place to store the solution.
 * @param nnz The number of non-zero elements in the sparse matrix.
 * @param lower If true, performs forward substitution. If false, performs backward substitution.
 * @param num_threads The number of OpenMP threads to use. If <= 0, it defaults to the maximum
 *        number of available threads.
 */
// data, indices, indptr, ans, nnz, num_rows, lower, OMP_NUM_THREADS
void spsolve_triangular(
    nb::ndarray<const double,  nb::ndim<1>, nb::c_contig>& data,
    nb::ndarray<const int, nb::ndim<1>, nb::c_contig>& indices,
    nb::ndarray<const int, nb::ndim<1>, nb::c_contig>& indptr,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig>& b,
    bool lower, int num_threads
) {
    // get raw data pointers once at the beginning
    const auto* data_ptr    = data.data();
    const auto* indices_ptr = indices.data();
    const auto* ind_ptr     = indptr.data();

    const auto  nnz      = data.size();
    const auto  num_rows = indptr.size() - 1;
    const auto  num_cols = b.shape(1);
    double*     b_ptr    = b.data();

    auto vec_cols = num_cols / 4 * 4;
    auto residue  = num_cols % 4;
    auto para_max = num_cols / 4 + residue;

    volatile bool flag_ill_zero_diag = false;

    // // when num_cols is small, grab more columns and use single column solver, to increase CPU usage
    // while ((para_max < 6) && (vec_cols > 4)) {
    //     residue += 4;
    //     vec_cols -= 4;
    //     para_max += vec_cols / 4 + residue; // para_max += 3
    // }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    num_threads = (num_threads<para_max) ? num_threads : para_max;


#if defined(_OPENMP) && _OPENMP >= 201503
    #pragma omp parallel num_threads(num_threads) proc_bind(spread)
#else
    #pragma omp parallel num_threads(num_threads)
#endif
    {


        double *b_i_ptr;
        __m256d b_i_vec, b_j_vec, val_vec;

        if (lower) {

            // solve Lx=b using forward substitution
            #pragma omp for schedule(guided) nowait
            for (int col = 0; col < vec_cols; col += 4) {
                // use AVX2
                for (int i = 0; i < num_rows; ++i) {
                    const auto& data_lpos = ind_ptr[i];
                    const auto& data_rpos = ind_ptr[i+1] - 1;
                    if ((i != 0) && (data_lpos > data_rpos)) { continue; } // empty row
                    if (i != indices_ptr[data_rpos]) { flag_ill_zero_diag = true; continue; }

                    b_i_ptr = b_ptr + i * num_cols + col;
                    b_i_vec = _mm256_load_pd(b_i_ptr);

                    for (int k = data_lpos; k < data_rpos; ++k) {
                        const auto& j = indices_ptr[k];
                        b_j_vec = _mm256_load_pd(b_ptr + j * num_cols + col);
                        val_vec = _mm256_set1_pd(data_ptr[k]);
                        b_i_vec = _mm256_sub_pd(b_i_vec, _mm256_mul_pd(val_vec, b_j_vec));
                    }

                    b_i_vec = _mm256_div_pd(b_i_vec, _mm256_set1_pd(data_ptr[data_rpos]));
                    _mm256_store_pd(b_i_ptr, b_i_vec);
                }
            }

            #pragma omp for schedule(guided) nowait
            for (int col = vec_cols; col < num_cols; ++col) {
                // _mm256_maskload_pd, _mm256_masksave_pd are slow...
                for (int i = 0; i < num_rows; ++i) {
                    const auto& data_lpos = ind_ptr[i];
                    const auto& data_rpos = ind_ptr[i+1] - 1;
                    if ((i != 0) && (data_lpos > data_rpos)) { continue; } // empty row
                    if (i != indices_ptr[data_rpos]) { flag_ill_zero_diag = true; continue; }

                    const auto& b_i = b_ptr + i * num_cols + col;

                    for (int k = data_lpos; k < data_rpos; ++k) {
                        const auto& j = indices_ptr[k];
                        const auto& v = data_ptr[k];
                        *(b_i) -= v * b_ptr[j * num_cols + col];
                    }

                    *(b_i) /= data_ptr[data_rpos];

                }
            }

        } else {

            // solve Ux=b using backward substitution
            #pragma omp for schedule(guided) nowait
            for (int col = 0; col < vec_cols; col += 4) {
                // use AVX2
                const auto num_rows_1 = num_rows-1;
                for (int i = num_rows_1; i >= 0; --i) {
                    const auto& data_lpos = ind_ptr[i];
                    const auto& data_rpos = ind_ptr[i+1] - 1;
                    if ((i != num_rows_1) && (data_lpos > data_rpos)) { continue; } // empty row
                    if (i != indices_ptr[data_lpos]) { flag_ill_zero_diag = true; continue; }

                    b_i_ptr = b_ptr + i * num_cols + col;
                    b_i_vec = _mm256_load_pd(b_i_ptr);

                    for (int k = data_rpos; k > data_lpos; --k) {
                        const auto& j = indices_ptr[k];
                        b_j_vec = _mm256_load_pd(b_ptr + j * num_cols + col);
                        val_vec = _mm256_set1_pd(data_ptr[k]);
                        b_i_vec = _mm256_sub_pd(b_i_vec, _mm256_mul_pd(val_vec, b_j_vec));
                    }

                    b_i_vec = _mm256_div_pd(b_i_vec, _mm256_set1_pd(data_ptr[data_lpos]));
                    _mm256_store_pd(b_i_ptr, b_i_vec);
                }
            }

            #pragma omp for schedule(guided) nowait
            for (int col = vec_cols; col < num_cols; ++col) {
                // _mm256_maskload_pd, _mm256_masksave_pd are slow...
                const auto num_rows_1 = num_rows-1;
                for (int i = num_rows_1; i >= 0; --i) {
                    const auto& data_lpos = ind_ptr[i];
                    const auto& data_rpos = ind_ptr[i+1] - 1;
                    if ((i != num_rows_1) && (data_lpos > data_rpos)) { continue; } // empty row
                    if (i != indices_ptr[data_lpos]) { flag_ill_zero_diag = true; continue; }

                    const auto& b_i = b_ptr + i * num_cols + col;

                    for (int k = data_rpos; k > data_lpos; --k) {
                        const auto& j = indices_ptr[k];
                        const auto& v = data_ptr[k];
                        *(b_i) -= v * b_ptr[j * num_cols + col];
                    }

                    *(b_i) /= data_ptr[data_lpos];

                }
            }

        }


    } // end of #pragma omp

    if (flag_ill_zero_diag) {
        throw std::invalid_argument("ill-conditioned matrix: non-empty row with diag element of 0");
    }

}