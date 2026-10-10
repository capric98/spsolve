import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import block_diag, csc_array, diags

from spsolve import splu


def dense_pivoted(n, seed=812):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-0.04, 0.04, (n, n)) + 4.0 * np.eye(n)
    # Dominant pivots repeatedly come from outside the current panel.
    return csc_array(matrix[np.roll(np.arange(n), 47), :])


def sparse_to_dense():
    rng = np.random.default_rng(643)
    n, split = 289, 96
    matrix = np.zeros((n, n))
    matrix[np.diag_indices(n)] = 8.0
    for row in range(split):
        cols = split + rng.choice(n - split, 20, replace=False)
        matrix[row, cols] = rng.uniform(-0.15, 0.15, 20)
    for row in range(split, n):
        cols = rng.choice(split, 20, replace=False)
        matrix[row, cols] = rng.uniform(-0.15, 0.15, 20)
    # Every initial row is sparse. Elimination fills the lower Schur block.
    assert np.max(np.count_nonzero(matrix, axis=1)) < 64
    return csc_array(matrix)


class BlockedLUTests(unittest.TestCase):
    def check_factor(self, matrix, factor, tolerance=3e-12):
        self.assertEqual(factor.nnz, factor.L.nnz + factor.U.nnz)
        scaled = csc_array(matrix.multiply(factor.row_scale[:, None])
                           .multiply(factor.col_scale[None, :]))
        error = scaled[factor.perm_r, :][:, factor.perm_c] - factor.L @ factor.U
        self.assertLess(np.max(abs(error.data), initial=0.0), 3e-13)
        expected = np.random.default_rng(371).normal(size=(matrix.shape[0], 3))
        for trans, operator in (("N", matrix), ("T", matrix.T), ("H", matrix.T)):
            assert_allclose(factor.solve(operator @ expected, trans=trans,
                                        iterative_refinement=False),
                            expected, rtol=tolerance, atol=tolerance)
        rhs = np.array(matrix @ expected[:, 0], copy=True)
        solution = factor.solve(rhs, overwrite_b=True, iterative_refinement=False)
        self.assertTrue(np.shares_memory(rhs, solution))
        assert_allclose(solution, expected[:, 0], rtol=tolerance, atol=tolerance)

    def assert_same_factors(self, serial, parallel):
        for name in ("perm_r", "perm_c", "row_scale", "col_scale"):
            assert_array_equal(getattr(serial, name), getattr(parallel, name))
        for name in ("L", "U"):
            first, second = getattr(serial, name), getattr(parallel, name)
            for field in ("indptr", "indices", "data"):
                assert_array_equal(getattr(first, field), getattr(second, field))

    def test_pivoted_panels_all_orderings_thresholds_and_index_widths(self):
        original = dense_pivoted(225)
        saved = original.data.copy()
        for order in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
            for threshold in (0.0, 0.1, 1.0):
                with self.subTest(order=order, threshold=threshold):
                    matrix = original.copy()
                    if threshold == 0.0:
                        # Zero threshold intentionally accepts arbitrarily small
                        # preferred pivots. Use stable preferred pivots here;
                        # the other thresholds exercise cross-panel row swaps.
                        matrix = csc_array(matrix[np.roll(np.arange(matrix.shape[0]), -47), :])
                    if threshold == 0.1:
                        matrix.indices = matrix.indices.astype(np.int64)
                        matrix.indptr = matrix.indptr.astype(np.int64)
                    serial = splu(matrix, backend="native", permc_spec=order,
                                  diag_pivot_thresh=threshold, num_threads=1)
                    parallel = splu(matrix, backend="native", permc_spec=order,
                                    diag_pivot_thresh=threshold, num_threads=4)
                    self.assertEqual(serial.stats["numeric_engine"], "row-hybrid")
                    self.assertGreater(serial.stats["lu_dense_panels"], 0)
                    self.assertGreater(serial.stats["lu_dense_update_flops"], 0)
                    self.assertGreater(serial.stats["lu_panel_workspace_bytes"], 0)
                    if threshold == 1.0:
                        self.assertGreater(serial.stats["pivot_swaps"], 32)
                    self.check_factor(matrix, serial)
                    self.check_factor(matrix, parallel)
                    self.assert_same_factors(serial, parallel)
        assert_array_equal(original.data, saved)

    def test_sparse_fill_conversion_and_panel_boundary_pivots(self):
        matrix = sparse_to_dense()
        for threshold in (0.0, 1.0):
            serial = splu(matrix, backend="native", permc_spec="NATURAL",
                          diag_pivot_thresh=threshold, num_threads=1)
            parallel = splu(matrix, backend="native", permc_spec="NATURAL",
                            diag_pivot_thresh=threshold, num_threads=4)
            self.assertGreater(serial.stats["dense_rows"], 32)
            self.assertGreater(serial.stats["lu_dense_panels"], 0)
            self.check_factor(matrix, serial)
            self.check_factor(matrix, parallel)
            self.assert_same_factors(serial, parallel)

    def test_short_panels_disconnected_support_and_cancellation(self):
        for n in (159, 160, 161):
            with self.subTest(n=n):
                first = dense_pivoted(n, n)
                second = dense_pivoted(131, n + 1)
                # Stored zeros and exact cancellations must not leave stale tails.
                row_indices = first.indices[::19]
                column_indices = np.repeat(np.arange(n), np.diff(first.indptr))[::19]
                zero_positions = np.arange(0, first.nnz, 19)[
                    row_indices != (column_indices + 47) % n]
                first.data[zero_positions] = 0.0
                matrix = csc_array(block_diag((first, second), format="csc"))
                # A binary rank-one border cancels exactly after its first pivot.
                cancel = np.eye(129)
                cancel[0, :] = 0.5
                cancel[0, 0] = 1.0
                cancel[1:, 0] = 0.5
                cancel[1:, 1:] += 0.25
                cancel[-1, 1] += 0.125  # Unsymmetric: require general LU.
                matrix = csc_array(block_diag((matrix, csc_array(cancel)), format="csc"))
                serial = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=4)
                self.assertGreater(serial.stats["lu_dense_panels"], 0)
                # Tail support comprises local blocks, rather than every n column.
                self.assertLess(serial.stats["lu_panel_columns"], 200)
                self.check_factor(matrix, serial)
                self.assert_same_factors(serial, parallel)

    def test_narrow_sparse_and_small_dense_controls_skip_panels(self):
        tridiagonal = diags([-np.ones(299), 4.0 * np.ones(300),
                             0.5 * np.ones(299)], [-1, 0, 1]).toarray()
        # NATURAL factors the tridiagonal matrix in the left-looking CSC
        # engine, as COLAMD does. A full last column, wider than the probe's
        # 192-row reach limit, keeps every row sparse and the NATURAL factor
        # in the row engine, so a narrow sparse control still runs there.
        bordered = tridiagonal.copy()
        bordered[:, -1] = 0.25
        bordered[-1, -1] = 4.0
        natural_engines = ("left-looking-csc", "row-hybrid", "row-hybrid")
        controls = (csc_array(tridiagonal), csc_array(bordered), dense_pivoted(96))
        for matrix, natural_engine in zip(controls, natural_engines):
            for order in ("NATURAL", "COLAMD"):
                factor = splu(matrix, backend="native", permc_spec=order, num_threads=4)
                if order == "NATURAL":
                    self.assertEqual(factor.stats["numeric_engine"], natural_engine)
                self.assertEqual(factor.stats["lu_dense_panels"], 0)
                self.assertEqual(factor.stats["lu_panel_workspace_bytes"], 0)
                self.check_factor(matrix, factor)

    def test_dense_panels_preserve_small_and_large_matrix_scales(self):
        original = dense_pivoted(193)
        for scale in (1e-16, 1e16):
            with self.subTest(scale=scale):
                matrix = original * scale
                serial = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=4)
                self.assertGreater(serial.stats["lu_dense_panels"], 0)
                self.check_factor(matrix, serial)
                self.assert_same_factors(serial, parallel)

    def test_panel_activation_rechecks_a_later_small_dense_phase(self):
        first = dense_pivoted(193)
        # The first component's last panel ends at global column 224, with
        # no dense rows left. Tail fill starts only after column 241, so eight
        # newly dense rows must satisfy the activation guard again.
        rng = np.random.default_rng(149)
        split = 96
        tail = np.diag(4.0 * np.ones(300))
        for row in range(split):
            columns = split + rng.choice(300 - split, 20, replace=False)
            tail[row, columns] = rng.uniform(-0.02, 0.02, 20)
        for row in range(split, split + 8):
            columns = rng.choice(np.arange(48, split), 20, replace=False)
            tail[row, columns] = rng.uniform(-0.02, 0.02, 20)
        self.assertLess(np.max(np.count_nonzero(tail, axis=1)), 64)
        matrix = csc_array(block_diag((first, csc_array(tail)), format="csc"))
        alone = splu(first, backend="native", permc_spec="NATURAL", num_threads=1)
        for threads in (1, 4):
            with self.subTest(threads=threads):
                combined = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=threads)
                self.assertEqual(combined.stats["dense_rows"], first.shape[0] + 8)
                self.assertEqual(combined.stats["lu_dense_panels"], alone.stats["lu_dense_panels"])
                self.check_factor(matrix, combined)

    def test_dense_singular_and_nonfinite_updates_raise(self):
        matrix = dense_pivoted(193).toarray()
        matrix[150, :] = matrix[149, :]
        for threads in (1, 4):
            with self.assertRaisesRegex(np.linalg.LinAlgError, "singular"):
                splu(csc_array(matrix), backend="native", permc_spec="NATURAL",
                     num_threads=threads)
        # Finite Wilkinson data overflows through actual updates, not input checks.
        n = 1030
        growth = np.eye(n) - np.tril(np.ones((n, n)), -1)
        growth[:, -1] = 1.0
        for threads in (1, 4):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                splu(csc_array(growth), backend="native", permc_spec="NATURAL",
                     num_threads=threads)


if __name__ == "__main__":
    unittest.main()
