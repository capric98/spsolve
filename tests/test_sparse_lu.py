import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import block_diag, csc_array, diags, random_array, tril

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
    def assert_factor_identity(self, matrix, factor, *, atol=2e-13):
        self.assertEqual(factor.nnz, factor.L.nnz + factor.U.nnz)
        scaled = csc_array(
            matrix.multiply(factor.row_scale[:, None]).multiply(
                factor.col_scale[None, :]
            )
        )
        permuted = scaled[factor.perm_r, :][:, factor.perm_c]
        assert_allclose(
            (permuted - factor.L @ factor.U).toarray(),
            0.0,
            atol=atol,
        )

    def test_adaptive_numeric_engines_and_stats(self):
        n = 300
        symmetric = csc_array(
            diags(
                [-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)],
                [-1, 0, 1],
                format="csc",
            )
        )
        spd = splu(
            symmetric,
            backend="native",
            permc_spec="MMD_AT_PLUS_A",
            num_threads=8,
        )
        self.assertEqual(spd.stats["ordering_backend"], "SuiteSparse AMD 3.3.4")
        self.assertEqual(spd.stats["numeric_engine"], "symmetric-ldlt-csc")
        self.assertEqual(spd.stats["threading_backend"], "serial")
        self.assertEqual(spd.stats["pivot_swaps"], 0)
        self.assertGreater(spd.stats["supernodes"], 0)
        self.assertGreaterEqual(spd.stats["max_supernode_width"], 1)
        assert_array_equal(spd.perm_r, spd.perm_c)
        self.assert_factor_identity(symmetric, spd)
        expected = np.linspace(-1.0, 1.0, n)
        assert_allclose(spd.solve(symmetric @ expected), expected, atol=1e-12)
        assert_allclose(
            spd.solve(symmetric.T @ expected, trans="T"),
            expected,
            atol=1e-12,
        )

        nonsymmetric = csc_array(
            diags(
                [-np.ones(n - 1), 4.0 * np.ones(n), 0.25 * np.ones(n - 2)],
                [-1, 0, 2],
                format="csc",
            )
        )
        narrow = splu(
            nonsymmetric,
            backend="native",
            permc_spec="MMD_AT_PLUS_A",
            num_threads=8,
        )
        self.assertEqual(narrow.stats["numeric_engine"], "left-looking-csc")
        self.assert_factor_identity(nonsymmetric, narrow)
        assert_allclose(narrow.solve(nonsymmetric @ expected), expected, atol=1e-12)
        assert_allclose(
            narrow.solve(nonsymmetric.T @ expected, trans="T"),
            expected,
            atol=1e-12,
        )

        for name in (
            "symbolic_seconds",
            "numeric_seconds",
            "assembly_seconds",
            "peak_workspace_bytes",
            "ordering_estimated_l_nnz",
            "ordering_estimated_max_front",
        ):
            self.assertIn(name, narrow.stats)

        spd_colamd = splu(
            symmetric,
            backend="native",
            permc_spec="COLAMD",
            num_threads=8,
        )
        self.assertEqual(spd_colamd.stats["numeric_engine"], "symmetric-ldlt-csc")
        self.assertEqual(
            spd_colamd.stats["ordering_backend"],
            "SuiteSparse COLAMD 3.3.5",
        )
        self.assert_factor_identity(symmetric, spd_colamd)

        narrow_colamd = splu(
            nonsymmetric,
            backend="native",
            permc_spec="COLAMD",
            num_threads=8,
        )
        self.assertEqual(narrow_colamd.stats["numeric_engine"], "left-looking-csc")
        self.assertEqual(
            narrow_colamd.stats["ordering_backend"], "SuiteSparse COLAMD 3.3.5"
        )
        self.assert_factor_identity(nonsymmetric, narrow_colamd)

    def test_colamd_probe_keeps_row_permuted_bands_in_csc(self):
        n = 1000
        base = csc_array(
            diags(
                [-0.2 * np.ones(n - 7), -np.ones(n - 1), 4.0 * np.ones(n),
                 0.5 * np.ones(n - 2), 0.1 * np.ones(n - 9)],
                [-7, -1, 0, 2, 9], format="csc",
            )
        )
        reference = splu(base, backend="native", permc_spec="COLAMD")
        expected = np.random.default_rng(322).normal(size=(n, 3))
        for seed in (321, 654):
            matrix = csc_array(base[np.random.default_rng(seed).permutation(n), :])
            factors = []
            for threads in (1, 4):
                with self.subTest(seed=seed, threads=threads):
                    factor = splu(
                        matrix, backend="native", permc_spec="COLAMD", num_threads=threads
                    )
                    factors.append(factor)
                    self.assertEqual(factor.stats["numeric_engine"], "left-looking-csc")
                    self.assertEqual(
                        factor.stats["ordering_backend"], "SuiteSparse COLAMD 3.3.5"
                    )
                    self.assertEqual(factor.stats["ordering_estimated_max_front"], 0.0)
                    self.assertEqual(factor.stats["ordering_estimated_l_nnz"], 0.0)
                    self.assertFalse(factor.stats["numeric_probe_fallback"])
                    self.assertEqual(factor.stats["numeric_probe_columns"], n)
                    self.assertLess(factor.stats["numeric_probe_max_front"], 40)
                    self.assertLessEqual(factor.nnz, reference.nnz + 2)
                    self.assert_factor_identity(matrix, factor)
                    for trans, operator in (("N", matrix), ("T", matrix.T)):
                        assert_allclose(
                            factor.solve(operator @ expected, trans=trans,
                                         iterative_refinement=False),
                            expected, rtol=2e-13, atol=2e-13,
                        )
            assert_array_equal(factors[0].perm_r, factors[1].perm_r)
            assert_array_equal(factors[0].perm_c, factors[1].perm_c)
            assert_array_equal(factors[0].L.toarray(), factors[1].L.toarray())
            assert_array_equal(factors[0].U.toarray(), factors[1].U.toarray())

    def test_colamd_probe_restarts_wide_fronts_deterministically(self):
        for dense_size in (140, 260):
            with self.subTest(dense_size=dense_size):
                rng = np.random.default_rng(555)
                dense = rng.uniform(-0.5, 0.5, size=(dense_size, dense_size))
                dense += 300.0 * np.eye(dense_size)
                matrix = csc_array(block_diag((np.eye(40), dense), format="csc"))
                matrix = csc_array(matrix[rng.permutation(matrix.shape[0]), :])
                expected = rng.normal(size=(matrix.shape[0], 3))
                factors = []
                for threads in (1, 4):
                    factor = splu(
                        matrix, backend="native", permc_spec="COLAMD", num_threads=threads
                    )
                    factors.append(factor)
                    self.assertEqual(factor.stats["numeric_engine"], "row-hybrid")
                    self.assertTrue(factor.stats["numeric_probe_fallback"])
                    self.assertGreater(factor.stats["numeric_probe_columns"], 0)
                    self.assertLess(factor.stats["numeric_probe_columns"], matrix.shape[0])
                    self.assertEqual(factor.stats["numeric_probe_max_front"], dense_size)
                    self.assertLessEqual(factor.stats["numeric_probe_updates"], 64 * matrix.nnz)
                    self.assertGreater(factor.stats["numeric_probe_seconds"], 0.0)
                    self.assertGreaterEqual(
                        factor.stats["numeric_seconds"], factor.stats["numeric_probe_seconds"]
                    )
                    self.assertGreater(factor.stats["peak_workspace_bytes"], 0)
                    self.assert_factor_identity(matrix, factor)
                    for trans, operator in (("N", matrix), ("T", matrix.T)):
                        assert_allclose(
                            factor.solve(operator @ expected, trans=trans,
                                         iterative_refinement=False),
                            expected, rtol=2e-13, atol=2e-13,
                        )
                self.assertGreater(factors[1].stats["parallel_steps"], 0)
                assert_array_equal(factors[0].perm_r, factors[1].perm_r)
                assert_array_equal(factors[0].perm_c, factors[1].perm_c)
                assert_array_equal(factors[0].L.toarray(), factors[1].L.toarray())
                assert_array_equal(factors[0].U.toarray(), factors[1].U.toarray())

    def test_colamd_probe_propagates_factorization_errors(self):
        singular = np.ones((260, 260))
        singular[:, 1] = 2.0
        with self.assertRaises(np.linalg.LinAlgError):
            splu(csc_array(singular), backend="native", permc_spec="COLAMD")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            splu(csc_array([[1e-310, 1.0], [1.0, 1.0]]), backend="native",
                 permc_spec="COLAMD", diag_pivot_thresh=0.0)

    def test_adaptive_csc_engines_on_random_patterns(self):
        for seed in range(5):
            with self.subTest(kind="spd", seed=seed):
                rng = np.random.default_rng(seed)
                lower = tril(
                    random_array(
                        (70, 70),
                        density=0.04,
                        format="csc",
                        rng=rng,
                        data_sampler=lambda size: rng.uniform(-0.5, 0.5, size),
                    ),
                    k=-1,
                    format="csc",
                )
                off_diagonal = lower + lower.T
                diagonal = np.asarray(abs(off_diagonal).sum(axis=1)).ravel() + 1.0
                matrix = csc_array(
                    off_diagonal + diags(diagonal, format="csc"),
                    dtype=np.float64,
                )
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
                    num_threads=8,
                )
                self.assertEqual(serial.stats["numeric_engine"], "symmetric-ldlt-csc")
                assert_array_equal(serial.perm_r, threaded.perm_r)
                assert_array_equal(serial.perm_c, threaded.perm_c)
                self.assert_factor_identity(matrix, threaded)

            with self.subTest(kind="nonsymmetric", seed=seed):
                matrix = general_matrix(70, 0.05, seed + 200)
                row_order = np.random.default_rng(seed + 400).permutation(70)
                matrix = csc_array(matrix[row_order, :])
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
                    num_threads=8,
                )
                self.assertEqual(serial.stats["numeric_engine"], "left-looking-csc")
                self.assertGreater(serial.stats["pivot_swaps"], 0)
                assert_array_equal(serial.perm_r, threaded.perm_r)
                assert_array_equal(serial.perm_c, threaded.perm_c)
                self.assert_factor_identity(matrix, threaded)

    def test_unequal_transpose_entries_do_not_use_symmetric_factorization(self):
        for scale in (1e-16, 1e-8, 1.0, 1e8, 1e16):
            nonsymmetric = scale * np.array([[1.0, 0.5], [3.0, 2.0]])
            near_symmetric = scale * np.array(
                [[4.0, 1.0, 0.0], [1.0, 3.0, -0.5], [0.0, -0.5, 2.0]]
            )
            near_symmetric[1, 0] = np.nextafter(near_symmetric[0, 1], np.inf)
            for kind, values in (
                ("nonsymmetric", nonsymmetric),
                ("near_symmetric", near_symmetric),
            ):
                matrix = csc_array(values)
                expected = np.column_stack(
                    (np.ones(matrix.shape[0]), np.arange(matrix.shape[0]) + 1.0)
                )
                for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
                    with self.subTest(kind=kind, scale=scale, ordering=ordering):
                        factor = splu(matrix, backend="native", permc_spec=ordering)
                        self.assertNotEqual(
                            factor.stats["numeric_engine"], "symmetric-ldlt-csc"
                        )
                        self.assert_factor_identity(matrix, factor, atol=2e-14)
                        for trans, operator in (("N", matrix), ("T", matrix.T)):
                            assert_allclose(
                                factor.solve(
                                    operator @ expected,
                                    trans=trans,
                                    iterative_refinement=False,
                                ),
                                expected,
                                rtol=2e-14,
                                atol=2e-14,
                            )

    def test_exactly_symmetric_spd_retains_ldlt_at_all_scales(self):
        values = np.array(
            [[4.0, 1.0, 0.0], [1.0, 3.0, -0.5], [0.0, -0.5, 2.0]]
        )
        expected = np.array([2.0, -3.0, 1.0])
        for scale in (1e-16, 1e-8, 1.0, 1e8, 1e16):
            matrix = csc_array(scale * values)
            for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
                with self.subTest(scale=scale, ordering=ordering):
                    factor = splu(matrix, backend="native", permc_spec=ordering)
                    self.assertEqual(
                        factor.stats["numeric_engine"], "symmetric-ldlt-csc"
                    )
                    self.assert_factor_identity(matrix, factor, atol=2e-14)
                    for trans, operator in (("N", matrix), ("T", matrix.T)):
                        assert_allclose(
                            factor.solve(
                                operator @ expected,
                                trans=trans,
                                iterative_refinement=False,
                            ),
                            expected,
                            rtol=2e-14,
                            atol=2e-14,
                        )

    def test_symmetric_indefinite_falls_back_to_pivoted_lu(self):
        expected = np.array([2.0, -3.0])
        for scale in (1e-16, 1.0, 1e16):
            matrix = csc_array(scale * np.array([[0.0, 1.0], [1.0, 0.0]]))
            for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
                with self.subTest(scale=scale, ordering=ordering):
                    factor = splu(
                        matrix,
                        backend="native",
                        permc_spec=ordering,
                        num_threads=4,
                    )
                    self.assertNotEqual(
                        factor.stats["numeric_engine"], "symmetric-ldlt-csc"
                    )
                    self.assert_factor_identity(matrix, factor)
                    for trans, operator in (("N", matrix), ("T", matrix.T)):
                        assert_allclose(
                            factor.solve(
                                operator @ expected,
                                trans=trans,
                                iterative_refinement=False,
                            ),
                            expected,
                        )

    def test_vendored_orderings_accept_both_index_widths(self):
        matrix = general_matrix(45, 0.08, 99)
        expected = np.arange(matrix.shape[0], dtype=np.float64)
        for ordering in ("MMD_AT_PLUS_A", "COLAMD"):
            factors = []
            for dtype in (np.int32, np.int64):
                converted = csc_array(
                    (
                        matrix.data.copy(),
                        matrix.indices.astype(dtype),
                        matrix.indptr.astype(dtype),
                    ),
                    shape=matrix.shape,
                )
                factors.append(
                    splu(
                        converted,
                        backend="native",
                        permc_spec=ordering,
                    )
                )
            assert_array_equal(factors[0].perm_c, factors[1].perm_c)
            for factor in factors:
                assert_allclose(
                    factor.solve(matrix @ expected),
                    expected,
                    atol=1e-11,
                )

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

                self.assert_factor_identity(matrix, factor, atol=2e-14)
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
