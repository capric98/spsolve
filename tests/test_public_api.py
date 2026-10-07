import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import spsolve as scipy_spsolve

from spsolve import spsolve, spsolve_pardiso
from spsolve.spparms import get_max_threads, set_num_threads, set_prefer_order


def mixed_dtype_system(matrix_dtype, rhs_dtype, multiple):
    values = np.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
    if np.issubdtype(matrix_dtype, np.complexfloating):
        values = values + 1j * np.eye(3)
    matrix = csc_array(values, dtype=matrix_dtype)
    rhs = np.array([1.0, 2.0, -1.0])
    if multiple:
        rhs = np.column_stack((rhs, rhs[::-1]))
    if np.issubdtype(rhs_dtype, np.complexfloating):
        rhs = rhs + 1j * rhs[::-1]
    return matrix, rhs.astype(rhs_dtype)


class PublicApiTests(unittest.TestCase):
    def test_general_spsolve_native_and_scipy(self):
        matrix = csc_array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
        expected = np.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 5.0]])
        rhs = matrix @ expected
        for backend in ("native", "scipy", "AUTO"):
            with self.subTest(backend=backend):
                assert_allclose(
                    spsolve(matrix, rhs, backend=backend, num_threads=4),
                    expected,
                )

    def test_triangular_fast_path_and_vector_shape(self):
        matrix = csr_array([[2.0, 0.0], [1.0, 3.0]])
        result = spsolve(matrix, np.array([2.0, 4.0]), backend="native")
        self.assertEqual(result.shape, (2,))
        assert_allclose(result, [1.0, 1.0])

    def test_scipy_fallback_supports_complex_matrix(self):
        matrix = csc_array([[2.0 + 1.0j, 1.0], [1.0, 3.0 - 2.0j]])
        expected = np.array([1.0 - 1.0j, 2.0 + 0.5j])
        assert_allclose(spsolve(matrix, matrix @ expected), expected)

    def test_one_shot_joint_dtype_promotion_matches_scipy(self):
        dtype_pairs = (
            (np.float64, np.complex128),
            (np.float32, np.float64),
            (np.complex64, np.complex128),
            (np.float32, np.complex64),
            (np.float64, np.complex64),
            (np.complex64, np.float64),
            (np.float32, np.int64),
            (np.int32, np.float32),
            (np.int16, np.bool_),
            (np.bool_, np.bool_),
            (np.bool_, np.complex64),
            (np.int64, np.complex64),
            (np.float32, np.float32),
            (np.complex64, np.complex64),
        )
        for matrix_dtype, rhs_dtype in dtype_pairs:
            for multiple in (False, True):
                matrix, rhs = mixed_dtype_system(matrix_dtype, rhs_dtype, multiple)
                original_rhs = rhs.copy()
                original_matrix = matrix.copy()
                reference = scipy_spsolve(matrix.copy(), rhs, use_umfpack=False)
                for backend in ("AUTO", "SCIPY"):
                    with self.subTest(
                        matrix_dtype=matrix_dtype, rhs_dtype=rhs_dtype,
                        multiple=multiple, backend=backend,
                    ):
                        result = spsolve(matrix, rhs, backend=backend)
                        self.assertEqual(result.dtype, reference.dtype)
                        self.assertEqual(result.shape, rhs.shape)
                        assert_allclose(result, reference, rtol=5e-6, atol=5e-6)
                        assert_array_equal(rhs, original_rhs)
                        self.assertEqual(matrix.dtype, original_matrix.dtype)
                        assert_array_equal(matrix.data, original_matrix.data)
                        assert_array_equal(matrix.indices, original_matrix.indices)
                        assert_array_equal(matrix.indptr, original_matrix.indptr)

    def test_mixed_dtype_overwrite_requires_matching_solution_dtype(self):
        for matrix_dtype, rhs_dtype in (
            (np.float32, np.float64),
            (np.float64, np.complex128),
            (np.float64, np.complex64),
            (np.complex128, np.float64),
            (np.bool_, np.bool_),
            (np.complex64, np.complex64),
        ):
            for backend in ("auto", "scipy"):
                for multiple in (False, True):
                    for writable in (False, True):
                        with self.subTest(
                            matrix_dtype=matrix_dtype, rhs_dtype=rhs_dtype,
                            backend=backend, multiple=multiple, writable=writable,
                        ):
                            matrix, rhs = mixed_dtype_system(matrix_dtype, rhs_dtype, multiple)
                            original = rhs.copy()
                            reference = scipy_spsolve(matrix.copy(), rhs, use_umfpack=False)
                            rhs.flags.writeable = writable
                            result = spsolve(matrix, rhs, backend=backend, overwrite_b=True)
                            assert_allclose(result, reference, rtol=5e-6, atol=5e-6)
                            if writable and rhs.dtype == reference.dtype:
                                self.assertIs(result, rhs)
                            else:
                                self.assertIsNot(result, rhs)
                                assert_array_equal(rhs, original)

    def test_auto_complex_rhs_uses_scipy_if_native_default_is_enabled(self):
        matrix, rhs = mixed_dtype_system(np.float64, np.complex128, True)
        reference = scipy_spsolve(matrix.copy(), rhs, use_umfpack=False)
        with (
            patch("spsolve.sparse_lu.NATIVE_DEFAULT", True),
            patch(
                "spsolve.sparse_lu._splu_factorize",
                side_effect=AssertionError("complex RHS must use a supported backend"),
            ),
        ):
            result = spsolve(matrix, rhs, backend="AUTO")
        assert_allclose(result, reference, rtol=2e-14, atol=2e-14)

    def test_explicit_native_scope_and_triangular_complex_rhs(self):
        matrix, rhs = mixed_dtype_system(np.float64, np.complex128, False)
        with self.assertRaisesRegex(TypeError, "complex RHS"):
            spsolve(matrix, rhs, backend="NATIVE")
        with self.assertRaisesRegex(TypeError, "float64 A"):
            spsolve(matrix.astype(np.float32), rhs.real, backend="native")
        triangular = csr_array([[2.0, 0.0], [1.0, 3.0]])
        expected = np.array([1.0 + 2.0j, -3.0 + 0.5j])
        rhs = triangular @ expected
        result = spsolve(triangular, rhs, backend="NATIVE", overwrite_b=True)
        self.assertIs(result, rhs)
        assert_allclose(result, expected)

    def test_promoted_scipy_factors_keep_finite_input_checks(self):
        for backend in ("auto", "scipy"):
            for invalid in (np.nan, np.inf):
                with self.subTest(backend=backend, invalid=invalid):
                    matrix, rhs = mixed_dtype_system(np.float64, np.complex128, False)
                    matrix.data[0] = invalid
                    with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                        spsolve(matrix, rhs, backend=backend, check_finite=True)

    def test_deprecated_pardiso_alias_routes_to_native(self):
        matrix = csc_array([[3.0, 1.0], [1.0, 2.0]])
        expected = np.array([2.0, -1.0])
        with self.assertWarns(DeprecationWarning):
            result = spsolve_pardiso(matrix, matrix @ expected)
        assert_allclose(result, expected)

    def test_settings_validation(self):
        original = get_max_threads()
        try:
            set_num_threads(2)
            self.assertEqual(get_max_threads(), 2)
            with self.assertRaises(ValueError):
                set_num_threads(0)
            with self.assertRaises(ValueError):
                set_prefer_order("bad")
        finally:
            set_num_threads(original)

    def test_shape_and_backend_validation(self):
        with self.assertRaises(ValueError):
            spsolve(csc_array(np.ones((2, 3))), np.ones(2))
        with self.assertRaises(ValueError):
            spsolve(csc_array(np.eye(2)), np.ones(3))
        with self.assertRaises(ValueError):
            spsolve(csc_array(np.eye(2)), np.ones(2), backend="unknown")


if __name__ == "__main__":
    unittest.main()
