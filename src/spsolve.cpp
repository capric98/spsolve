#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <immintrin.h>
#include <omp.h>
#include <vector>

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
void spsolve_triangular(
    nb::ndarray<const int, nb::ndim<1>, nb::c_contig>& rows,
    nb::ndarray<const int, nb::ndim<1>, nb::c_contig>& cols,
    nb::ndarray<const double,  nb::ndim<1>, nb::c_contig>& vals,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig>& b,
    int nnz,
    bool lower,
    int num_threads
) {
    // get raw data pointers once at the beginning
    const auto* rows_ptr = rows.data();
    const auto* cols_ptr = cols.data();
    const auto* vals_ptr = vals.data();
    const auto  num_cols = b.shape(1);
    double*     b_ptr    = b.data();

    auto vec_cols = num_cols / 4 * 4;
    auto residue  = num_cols % 4;
    auto para_max = num_cols / 4 + residue;

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

        bool write_back_flag;
        double* b_i_ptr;
        __m256d b_i_vec, b_j_vec, val_vec;

        if (lower) {

            // solve Lx=b using forward substitution
            #pragma omp for schedule(guided) nowait
            for (int col = 0; col < vec_cols; col += 4) {
                // use AVX2
                write_back_flag = true;
                for (int k = 0; k < nnz; ++k) {
                    const auto& i = rows_ptr[k];
                    const auto& j = cols_ptr[k];

                    if (write_back_flag) {
                        b_i_ptr = b_ptr + i * num_cols + col;
                        b_i_vec = _mm256_load_pd(b_i_ptr);
                    }
                    val_vec = _mm256_set1_pd(vals_ptr[k]);

                    if (i == j) {
                        // b[i, col] = b[i, col] / val;
                        b_i_vec = _mm256_div_pd(b_i_vec, val_vec);
                        _mm256_store_pd(b_i_ptr, b_i_vec);
                        write_back_flag = true;
                    } else {
                        // b[i, col] - val * b[j, col];
                        b_j_vec = _mm256_load_pd(b_ptr + j * num_cols + col);
                        b_i_vec = _mm256_sub_pd(b_i_vec, _mm256_mul_pd(val_vec, b_j_vec));
                        write_back_flag = false;
                    }
                }
            }

            #pragma omp for schedule(guided) nowait
            for (int col = vec_cols; col < num_cols; ++col) {
                // _mm256_maskload_pd, _mm256_masksave_pd are slow...
                for (int k = 0; k < nnz; ++k) {
                    const auto& i = rows_ptr[k] * num_cols + col;
                    const auto& j = cols_ptr[k] * num_cols + col;
                    const auto& v = vals_ptr[k];

                    if (i == j) {
                        b_ptr[i] /= v;
                    } else {
                        b_ptr[i] -= v * b_ptr[j];
                    }
                }
            }

        } else {

            // solve Ux=b using backward substitution
            #pragma omp for schedule(guided) nowait
            for (int col = 0; col < vec_cols; col += 4) {
                // use AVX2
                write_back_flag = true;
                for (int k = nnz - 1; k >= 0; --k) {
                    const auto& i = rows_ptr[k];
                    const auto& j = cols_ptr[k];

                    if (write_back_flag) {
                        b_i_ptr = b_ptr + i * num_cols + col;
                        b_i_vec = _mm256_load_pd(b_i_ptr);
                    }
                    val_vec = _mm256_set1_pd(vals_ptr[k]);

                    if (i == j) {
                        // b[i, col] = b[i, col] / val;
                        b_i_vec = _mm256_div_pd(b_i_vec, val_vec);
                        _mm256_store_pd(b_i_ptr, b_i_vec);
                        write_back_flag = true;
                    } else {
                        // b[i, col] - val * b[j, col];
                        b_j_vec = _mm256_load_pd(b_ptr + j * num_cols + col);
                        b_i_vec = _mm256_sub_pd(b_i_vec, _mm256_mul_pd(val_vec, b_j_vec));
                        write_back_flag = false;
                    }
                }
            }

            #pragma omp for schedule(guided) nowait
            for (int col = vec_cols; col < num_cols; ++col) {
                // _mm256_maskload_pd, _mm256_masksave_pd are slow...
                for (int k = nnz - 1; k >=0 ; --k) {
                    const auto& i = rows_ptr[k] * num_cols + col;
                    const auto& j = cols_ptr[k] * num_cols + col;
                    const auto& v = vals_ptr[k];

                    if (i == j) {
                        b_ptr[i] /= v;
                    } else {
                        b_ptr[i] -= v * b_ptr[j];
                    }
                }
            }

        }
    } // end of #pragma omp


}


// --- Nanobind Module Definition ---
// This macro creates the entry point for the Python module.
NB_MODULE(_spsolve, m) {
    m.doc() = "Nanobind implementation of a sparse triangular solver with OpenMP and AVX2.";
    // This defines the Python-callable function, mapping it to our C++ function.
    m.def("spsolve_triangular", &spsolve_triangular,
        nb::arg("rows"),
        nb::arg("cols"),
        nb::arg("vals"),
        nb::arg("b"),
        nb::arg("nnz"),
        nb::arg("lower"),
        nb::arg("num_threads") = 0, // Default value for the argument
        nb::call_guard<nb::gil_scoped_release>(),
        "Solves a sparse triangular linear system (Lx=b or Ux=b).\n\n"
        "The matrix 'b' is modified in-place.\n\n"
        "Args:\n"
        "    rows (np.ndarray[int32]): Row indices of the sparse matrix.\n"
        "    cols (np.ndarray[int32]): Column indices of the sparse matrix.\n"
        "    vals (np.ndarray[float64]): Non-zero values of the sparse matrix.\n"
        "    b (np.ndarray[float64, ndim=2, c_contig]): The right-hand side matrix.\n"
        "    nnz (int): The number of non-zero elements.\n"
        "    lower (bool): If True, perform forward substitution. Otherwise, backward.\n"
        "    num_threads (int): Number of OpenMP threads to use. Defaults to max available."
    );
}