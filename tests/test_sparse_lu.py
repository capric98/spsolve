import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array, diags, random_array

from spsolve import SparseLU, factorized, splu


def general_matrix(n: int, density: float, seed: int) -> csc_array:
    rng = np.random.default_rng(seed)
    matrix = random_array(
        (n, n),
        density=density,
        format="csc",
        rng=rng,
        data_sampler=lambda size: rng.uniform(-1.0, 1.0, size),
    )
    diagonal = np.asarray(abs(matrix).sum(axis=1)).ravel() + 1.0
    return csc_array(matrix + diags(diagonal, format="csc"), dtype=np.float64)


class SparseLUTests(unittest.TestCase):
    def test_factor_identity_and_public_metadata(self):
        matrix = general_matrix(40, 0.12, 10)
        for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
            with self.subTest(ordering=ordering):
                factor = splu(
                    matrix,
                    backend="native",
                    permc_spec=ordering,
                    num_threads=4,
                )
                self.assertIsInstance(factor, SparseLU)
                self.assertEqual(factor.shape, matrix.shape)
                self.assertEqual(factor.dtype, np.dtype(np.float64))
                self.assertEqual(factor.nnz, factor.L.nnz + factor.U.nnz)
                assert_allclose(factor.L.diagonal(), 1.0)

                scaled = csc_array(
                    matrix.multiply(factor.row_scale[:, None]).multiply(
                        factor.col_scale[None, :]
                    )
                )
                permuted = scaled[factor.perm_r, :][:, factor.perm_c]
                assert_allclose(
                    (permuted - factor.L @ factor.U).toarray(),
                    0.0,
                    atol=2e-14,
                )
                self.assertEqual(factor.stats["shape"], matrix.shape)
                self.assertEqual(factor.stats["ordering"], ordering)
                self.assertGreaterEqual(factor.stats["row_updates"], 0)

    def test_normal_transpose_and_hermitian_solves(self):
        matrix = general_matrix(60, 0.08, 11)
        factor = splu(
            matrix,
            backend="native",
            permc_spec="MMD_AT_PLUS_A",
            num_threads=4,
        )
        rng = np.random.default_rng(12)

        vector = rng.normal(size=matrix.shape[0])
        assert_allclose(factor.solve(matrix @ vector), vector, rtol=1e-12, atol=1e-12)

        multiple = rng.normal(size=(matrix.shape[0], 7))
        for trans in ("T", "H"):
            with self.subTest(trans=trans):
                assert_allclose(
                    factor.solve(matrix.T @ multiple, trans=trans, num_threads=4),
                    multiple,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_overwrite_and_factorized_callable(self):
        matrix = general_matrix(25, 0.2, 13)
        expected = np.arange(25, dtype=np.float64)
        rhs = np.asarray(matrix @ expected)
        factor = splu(matrix, backend="native")
        returned = factor.solve(rhs, overwrite_b=True)
        self.assertIs(returned, rhs)
        assert_allclose(rhs, expected, atol=1e-12)

        solve = factorized(matrix, backend="native")
        assert_allclose(solve(matrix @ expected), expected, atol=1e-12)

    def test_pivoting_with_zero_diagonal_and_zero_threshold(self):
        matrix = csc_array([[0.0, 1.0], [1.0, 1.0]])
        expected = np.array([2.0, -3.0])
        factor = splu(
            matrix,
            backend="native",
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
        )
        assert_allclose(factor.solve(matrix @ expected), expected)
        self.assertGreater(factor.stats["pivot_swaps"], 0)

    def test_canonicalization_and_int64_indices(self):
        matrix = csc_array(
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([0, 0, 1], dtype=np.int64),
                np.array([0, 2, 3], dtype=np.int64),
            ),
            shape=(2, 2),
        )
        factor = splu(matrix, backend="native", permc_spec="NATURAL")
        assert_allclose(factor.solve(np.array([6.0, 9.0])), [2.0, 3.0])

    def test_empty_factor(self):
        factor = splu(csc_array((0, 0), dtype=np.float64), backend="native")
        self.assertEqual(factor.shape, (0, 0))
        self.assertEqual(factor.nnz, 0)
        self.assertEqual(factor.L.shape, (0, 0))
        self.assertEqual(factor.solve(np.empty(0)).shape, (0,))

    def test_threaded_factor_is_deterministic(self):
        matrix = general_matrix(320, 0.025, 14)
        serial = splu(
            matrix,
            backend="native",
            permc_spec="MMD_AT_PLUS_A",
            num_threads=1,
        )
        threaded = splu(
            matrix,
            backend="native",
            permc_spec="MMD_AT_PLUS_A",
            num_threads=4,
        )
        assert_array_equal(serial.perm_r, threaded.perm_r)
        assert_array_equal(serial.perm_c, threaded.perm_c)
        assert_allclose(serial.L.toarray(), threaded.L.toarray(), rtol=0, atol=0)
        assert_allclose(serial.U.toarray(), threaded.U.toarray(), rtol=0, atol=0)

    def test_singular_and_invalid_inputs(self):
        with self.assertRaises(np.linalg.LinAlgError):
            splu(csc_array([[1.0, 2.0], [2.0, 4.0]]), backend="native")
        with self.assertRaises(np.linalg.LinAlgError):
            splu(csc_array([[1.0, 0.0], [0.0, 0.0]]), backend="native")
        with self.assertRaises(ValueError):
            splu(csc_array(np.eye(2)), backend="native", diag_pivot_thresh=2.0)
        with self.assertRaises(ValueError):
            splu(csc_array(np.eye(2)), backend="native", num_threads=0)
        with self.assertRaises(TypeError):
            splu(csc_array(np.eye(2), dtype=np.float32), backend="native")
        with self.assertRaises(ValueError):
            splu(csc_array([[1.0, np.inf], [0.0, 1.0]]), backend="native")


if __name__ == "__main__":
    unittest.main()
