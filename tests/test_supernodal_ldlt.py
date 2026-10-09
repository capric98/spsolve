import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import block_diag, csc_array, diags, eye, kron

from spsolve import splu


def poisson_3d(side):
    ident = eye(side, format="csc")
    off = diags([-np.ones(side - 1), -np.ones(side - 1)], [-1, 1], format="csc")
    diagonal = off + 6.0 * ident
    return csc_array(
        kron(kron(ident, ident), diagonal)
        + kron(kron(ident, off), ident)
        + kron(kron(off, ident), ident)
    )


def overlapping_cliques():
    # Known leaf widths straddle the node and 32-column panel boundaries.
    # Their long, noncontiguous, differing tails overlap only partially.
    widths = (15, 16, 17, 31, 32, 33)
    tail = 350
    n = sum(widths) + tail
    start_tail = n - tail
    rng = np.random.default_rng(785)
    mask = np.eye(n, dtype=bool)
    # A 320-column root makes blocked execution worthwhile; missing edges
    # involving the other tail rows still require symbolic fill discovery.
    mask[start_tail + 30:, start_tail + 30:] = True
    first = 0
    for node, width in enumerate(widths):
        last = first + width
        mask[first:last, first:last] = True
        selected = start_tail + rng.choice(tail, size=320 - 3 * node, replace=False)
        mask[first:last, selected] = True
        mask[selected, first:last] = True
        first = last
    raw = rng.uniform(-0.02, 0.02, (n, n))
    raw = np.tril(raw, -1)
    raw += raw.T
    raw *= mask
    raw[np.diag_indices(n)] = np.sum(abs(raw), axis=1) + 1.0
    row, col = np.nonzero(mask)
    return csc_array((raw[row, col], (row, col)), shape=(n, n)), widths


def fan_in_rejection(contiguous, overflow=True):
    # NATURAL order: ten 12-wide leaves coupled to 200 random rows of a dense
    # 400-wide root, then a 2-wide leaf coupled to 40 root rows (scattered, or
    # one contiguous run), then the root. The root's fan-in exceeds 2 MFLOP
    # over 25 strips, so it runs on the worker team at four threads and on the
    # calling thread at one. With overflow, the 2-wide leaf's subnormal pivots
    # (1e-308) make its L entries about 1e308, still finite, and its rank-2
    # kernel update sums two terms of about 1e308 into every diagonal output
    # entry (j, j), which overflows in a gathered (scattered rows) or in-place
    # (contiguous rows) kernel part of the fan-in; the kernel flag must reject
    # the SPD attempt. The second leaf column alternates signs so the two
    # columns are independent and the matrix stays well conditioned for the LU
    # fallback.
    rng = np.random.default_rng(5)
    leaves, width, root = 10, 12, 400
    pair = leaves * width
    first_root = pair + 2
    n = first_root + root
    dense = np.zeros((n, n))
    for leaf in range(leaves):
        cols = np.arange(leaf * width, (leaf + 1) * width)
        dense[np.ix_(cols, cols)] = 5e-4
        rows = first_root + np.sort(rng.choice(root, size=200, replace=False))
        dense[np.ix_(rows, cols)] = 0.005
        dense[np.ix_(cols, rows)] = 0.005
    rows = first_root + (np.arange(100, 140) if contiguous
                         else np.sort(rng.choice(root, size=40, replace=False)))
    coupling = 1.0 if overflow else 0.01
    signs = np.where(np.arange(rows.size) % 2 == 0, coupling, -coupling)
    dense[rows, pair] = dense[pair, rows] = coupling
    dense[rows, pair + 1] = dense[pair + 1, rows] = signs
    dense[pair, pair + 1] = dense[pair + 1, pair] = 1e-320 if overflow else 0.01
    dense[first_root:, first_root:] = 5e-4
    np.fill_diagonal(dense, 1.0)
    if overflow:
        # Subnormal values (1e-308 pivots, 1e-320 coupling): under
        # flush-to-zero/denormals-are-zero the attempt would be rejected at the
        # pivot instead, so the test would pass without reaching the fan-in.
        dense[pair, pair] = dense[pair + 1, pair + 1] = 1e-308
    return csc_array(dense)


class SupernodalLDLTTests(unittest.TestCase):
    def check_factor(self, matrix, factor, *, tolerance=3e-12):
        self.assertEqual(factor.nnz, factor.L.nnz + factor.U.nnz)
        scaled = csc_array(
            matrix.multiply(factor.row_scale[:, None]).multiply(factor.col_scale[None, :])
        )
        error = scaled[factor.perm_r, :][:, factor.perm_c] - factor.L @ factor.U
        self.assertLess(np.max(abs(error.data), initial=0.0), 3e-13)
        expected = np.random.default_rng(427).normal(size=(matrix.shape[0], 3))
        for trans, operator in (("N", matrix), ("T", matrix.T), ("H", matrix.T)):
            assert_allclose(
                factor.solve(operator @ expected, trans=trans, iterative_refinement=False),
                expected, rtol=tolerance, atol=tolerance,
            )
        assert_allclose(
            factor.solve(matrix @ expected[:, 0], iterative_refinement=False),
            expected[:, 0], rtol=tolerance, atol=tolerance,
        )

    def assert_same_factors(self, serial, parallel):
        for name in ("perm_r", "perm_c", "row_scale", "col_scale"):
            assert_array_equal(getattr(serial, name), getattr(parallel, name))
        for name in ("L", "U"):
            first, second = getattr(serial, name), getattr(parallel, name)
            assert_array_equal(first.indptr, second.indptr)
            assert_array_equal(first.indices, second.indices)
            assert_array_equal(first.data, second.data)

    def test_connected_grid_selects_blocked_updates_deterministically(self):
        for order in ("COLAMD", "MMD_AT_PLUS_A"):
            matrix = poisson_3d(16 if order == "COLAMD" else 20)
            with self.subTest(order=order):
                serial = splu(matrix, backend="native", permc_spec=order, num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec=order, num_threads=4)
                for factor in (serial, parallel):
                    expected_engine = ("symmetric-supernodal-ldlt" if order == "MMD_AT_PLUS_A"
                                       else "symmetric-ldlt-csc")
                    self.assertEqual(factor.stats["numeric_engine"], expected_engine)
                    if order == "MMD_AT_PLUS_A":
                        self.assertGreater(factor.stats["dense_update_flops"], 0)
                        self.assertGreater(factor.stats["spd_panel_entries"], 0)
                    self.assertLess(factor.stats["spd_panel_entries"], matrix.shape[0] ** 2 // 10)
                    self.check_factor(matrix, factor)
                if parallel.stats["num_threads"] > 1 and order == "MMD_AT_PLUS_A":
                    self.assertGreater(parallel.stats["parallel_panels"], 0)
                self.assert_same_factors(serial, parallel)

    def test_partial_contributors_panel_boundaries_and_scaled_inputs(self):
        original, widths = overlapping_cliques()
        original_data = original.data.copy()
        for scale in (1e-16, 1.0, 1e16):
            matrix = original * scale
            # Powers of two preserve exact transpose equality while varying
            # symmetric row/column scales substantially.
            exponents = np.random.default_rng(94).integers(-8, 9, matrix.shape[0])
            matrix.data = np.ldexp(matrix.data,
                exponents[matrix.indices] + np.repeat(exponents, np.diff(matrix.indptr)))
            matrix.indices = matrix.indices.astype(np.int64)
            matrix.indptr = matrix.indptr.astype(np.int64)
            with self.subTest(scale=scale):
                serial = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=4)
                self.assertEqual(serial.stats["numeric_engine"], "symmetric-supernodal-ldlt")
                self.assertGreater(serial.L.nnz + serial.U.nnz,
                                   matrix.nnz + matrix.shape[0] + 100)
                first = 0
                lower = serial.L.tocsc()
                for width in widths:
                    last = first + width
                    tails = []
                    for col in range(first, last):
                        rows = lower.indices[lower.indptr[col]:lower.indptr[col + 1]]
                        tails.append(rows[rows >= last])
                    for rows in tails[1:]:
                        assert_array_equal(rows, tails[0])
                    first = last
                self.check_factor(matrix, serial, tolerance=3e-10)
                self.check_factor(matrix, parallel, tolerance=3e-10)
                self.assert_same_factors(serial, parallel)
        assert_array_equal(original.data, original_data)

    def test_selection_boundary_and_all_orderings(self):
        for size in (319, 320, 351, 352, 353):
            dense = 0.125 * np.ones((size, size))
            dense[np.diag_indices(size)] = 4.0
            # A nonzero original entry cancels exactly after the first pivot;
            # it must remain in the filled factor pattern as a stored zero.
            dense[1, 2] = dense[2, 1] = 0.00390625
            matrix = csc_array(dense)
            for order in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
                with self.subTest(size=size, order=order):
                    factor = splu(matrix, backend="native", permc_spec=order, num_threads=4)
                    expected_engine = "symmetric-ldlt-csc" if size == 319 else "symmetric-supernodal-ldlt"
                    self.assertEqual(factor.stats["numeric_engine"], expected_engine)
                    self.check_factor(matrix, factor)
                    if order == "NATURAL":
                        lower = factor.L.tocsc()
                        self.assertEqual(lower[2, 1], 0.0)
                        self.assertIn(2, lower.indices[lower.indptr[1]:lower.indptr[2]])

    def test_late_indefinite_failure_keeps_lu_state_and_accounting(self):
        first = np.eye(320) + 0.01 * np.ones((320, 320))
        last = np.eye(65) + 0.01 * np.ones((65, 65))
        last[-1, -1] = -1.0
        matrix = csc_array(block_diag((first, last), format="csc"))
        for order in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
            with self.subTest(order=order):
                serial = splu(matrix, backend="native", permc_spec=order, num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec=order, num_threads=4)
                for factor in (serial, parallel):
                    self.assertNotIn("ldlt", factor.stats["numeric_engine"])
                    self.assertEqual(factor.stats["parallel_panels"], 0)
                    self.assertEqual(factor.stats["spd_panel_entries"], 0)
                    self.assertGreater(factor.stats["failed_spd_seconds"], 0.0)
                    self.assertGreaterEqual(factor.stats["factor_seconds"], factor.stats["failed_spd_seconds"])
                    self.assertGreater(factor.stats["peak_workspace_bytes"], 320 * 320 * 8)
                    self.check_factor(matrix, factor)
                self.assert_same_factors(serial, parallel)

    def test_overflow_inside_contributor_fan_in_falls_back(self):
        for contiguous in (False, True):
            with self.subTest(contiguous=contiguous):
                # The same structure without the overflow takes the blocked
                # path, so the rejection below happens inside it.
                control = fan_in_rejection(contiguous, overflow=False)
                serial = splu(control, backend="native", permc_spec="NATURAL", num_threads=1)
                parallel = splu(control, backend="native", permc_spec="NATURAL", num_threads=4)
                for factor in (serial, parallel):
                    self.assertEqual(factor.stats["numeric_engine"], "symmetric-supernodal-ldlt")
                if parallel.stats["num_threads"] > 1:
                    self.assertGreater(parallel.stats["parallel_panels"], 0)
                self.assert_same_factors(serial, parallel)

                matrix = fan_in_rejection(contiguous)
                serial = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=4)
                for factor in (serial, parallel):
                    self.assertNotIn("ldlt", factor.stats["numeric_engine"])
                    self.assertEqual(factor.stats["parallel_panels"], 0)
                    self.assertEqual(factor.stats["spd_panel_entries"], 0)
                    self.assertGreater(factor.stats["failed_spd_seconds"], 0.0)
                    self.check_factor(matrix, factor)
                self.assert_same_factors(serial, parallel)

    def test_mixed_narrow_and_wide_components_use_work_guard(self):
        for chain, width, engine in (
            (100_000, 320, "symmetric-ldlt-csc"),
            (20_000, 1000, "symmetric-supernodal-ldlt"),
        ):
            narrow = diags(
                [-np.ones(chain - 1), 4.0 * np.ones(chain), -np.ones(chain - 1)],
                [-1, 0, 1], format="csc",
            )
            wide = np.eye(width) + 0.01 * np.ones((width, width))
            matrix = csc_array(block_diag((narrow, wide), format="csc"))
            with self.subTest(chain=chain, width=width):
                serial = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
                parallel = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=4)
                for factor in (serial, parallel):
                    self.assertEqual(factor.stats["numeric_engine"], engine)
                    self.check_factor(matrix, factor)
                if engine == "symmetric-ldlt-csc":
                    self.assertEqual(serial.nnz, 502_718)
                    self.assertEqual(serial.stats["spd_panel_entries"], 0)
                    self.assertEqual(parallel.stats["parallel_panels"], 0)
                else:
                    # A fixed 5% wide-column criterion would incorrectly reject
                    # this case despite its dominant dense-node work.
                    self.assertLess(width / matrix.shape[0], 0.05)
                    self.assertGreater(serial.stats["spd_panel_entries"], 0)
                self.assert_same_factors(serial, parallel)


if __name__ == "__main__":
    unittest.main()
