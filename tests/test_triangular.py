import unittest
from warnings import catch_warnings, simplefilter

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import SparseEfficiencyWarning, coo_array, csc_array, csr_array

from spsolve import spsolve_triangular


class TriangularSolveTests(unittest.TestCase):
    def test_vector_and_matrix_layouts(self):
        lower_matrix = np.array(
            [[2.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [0.5, 2.0, 4.0]]
        )
        for lower, dense in ((True, lower_matrix), (False, lower_matrix.T)):
            for layout in ("vector", "C", "F"):
                with self.subTest(lower=lower, layout=layout):
                    expected = np.arange(1.0, 10.0).reshape(3, 3)
                    if layout == "vector":
                        expected = expected[:, 0]
                    else:
                        expected = np.array(expected, order=layout)
                    rhs = dense @ expected
                    if layout == "F":
                        rhs = np.array(rhs, order="F")
                    result = spsolve_triangular(
                        csr_array(dense), rhs, lower=lower, num_threads=4
                    )
                    self.assertEqual(result.shape, expected.shape)
                    assert_allclose(result, expected)

    def test_missing_and_stored_unit_diagonal_are_ignored(self):
        effective = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 3.0, 1.0]])
        expected = np.array([2.0, -1.0, 4.0])
        rhs = effective @ expected

        missing = csr_array(effective)
        missing.setdiag(0.0)
        missing.eliminate_zeros()
        assert_allclose(
            spsolve_triangular(missing, rhs, unit_diagonal=True), expected
        )

        stored = csr_array(effective)
        stored.setdiag([7.0, -2.0, 99.0])
        assert_allclose(
            spsolve_triangular(stored, rhs, unit_diagonal=True), expected
        )

    def test_complex_rhs_and_fortran_input(self):
        dense = np.array([[2.0, 0.0], [1.0, 3.0]])
        expected = np.array([[1.0 + 2.0j, -3.0j], [4.0 - 1.0j, 2.0 + 5.0j]])
        rhs = np.array(dense @ expected, order="F")
        result = spsolve_triangular(csr_array(dense), rhs)
        self.assertEqual(result.dtype, np.complex128)
        assert_allclose(result, expected)

        c_rhs = np.array(dense @ expected, order="C")
        overwritten = spsolve_triangular(
            csr_array(dense), c_rhs, overwrite_b=True
        )
        self.assertIs(overwritten, c_rhs)
        assert_allclose(overwritten, expected)

    def test_overwrite_real_c_and_f_arrays(self):
        matrix = csr_array([[2.0, 0.0], [1.0, 2.0]])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        for order in ("C", "F"):
            with self.subTest(order=order):
                rhs = np.array(matrix @ expected, dtype=np.float64, order=order)
                result = spsolve_triangular(matrix, rhs, overwrite_b=True)
                self.assertIs(result, rhs)
                assert_allclose(result, expected)

    def test_noncontiguous_rhs_is_copied(self):
        matrix = csr_array([[2.0, 0.0], [1.0, 3.0]])
        expected = np.arange(8.0).reshape(2, 4)
        wide_rhs = np.empty((2, 8))
        wide_rhs[:, ::2] = matrix @ expected
        rhs = wide_rhs[:, ::2]
        self.assertFalse(rhs.flags.c_contiguous)
        result = spsolve_triangular(matrix, rhs, overwrite_b=True)
        self.assertIsNot(result, rhs)
        assert_allclose(result, expected)

    def test_csc_conversion_warns(self):
        matrix = csc_array([[2.0, 0.0], [1.0, 3.0]])
        with catch_warnings(record=True) as warnings:
            simplefilter("always")
            result = spsolve_triangular(matrix, np.array([2.0, 4.0]))
        self.assertTrue(any(issubclass(w.category, SparseEfficiencyWarning) for w in warnings))
        assert_allclose(result, [1.0, 1.0])

    def test_coo_conversion_and_explicit_zero_outside_triangle(self):
        matrix = coo_array([[2.0, 0.0], [1.0, 3.0]])
        with self.assertWarns(SparseEfficiencyWarning):
            assert_allclose(spsolve_triangular(matrix, [2.0, 4.0]), [1.0, 1.0])

        csr = csr_array(
            (
                np.array([2.0, 0.0, 1.0, 3.0]),
                np.array([0, 1, 0, 1]),
                np.array([0, 2, 4]),
            ),
            shape=(2, 2),
        )
        assert_allclose(spsolve_triangular(csr, [2.0, 4.0]), [1.0, 1.0])

    def test_validation(self):
        with self.assertRaises(ValueError):
            spsolve_triangular(
                csr_array([[1.0, 1.0], [0.0, 1.0]]), np.ones(2), lower=True
            )
        with self.assertRaises(np.linalg.LinAlgError):
            spsolve_triangular(
                csr_array([[0.0, 0.0], [1.0, 1.0]]), np.ones(2), lower=True
            )
        with self.assertRaises(TypeError):
            spsolve_triangular(csr_array(np.eye(2, dtype=complex)), np.ones(2))
        with self.assertRaises(ValueError):
            spsolve_triangular(csr_array(np.eye(2)), np.ones(2), num_threads=0)

    def test_empty_system(self):
        result = spsolve_triangular(
            csr_array((0, 0), dtype=np.float64), np.empty(0)
        )
        self.assertEqual(result.shape, (0,))


if __name__ == "__main__":
    unittest.main()
