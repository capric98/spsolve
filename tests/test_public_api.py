import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csc_array, csr_array

from spsolve import spsolve, spsolve_pardiso
from spsolve.spparms import get_max_threads, set_num_threads, set_prefer_order


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
