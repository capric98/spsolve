import hashlib
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


def dense_boundary(size):
    dense = 0.125 * np.ones((size, size))
    dense[np.diag_indices(size)] = 4.0
    # A nonzero original entry cancels exactly after the first pivot; it must
    # remain in the filled factor pattern as a stored zero.
    dense[1, 2] = dense[2, 1] = 0.00390625
    return csc_array(dense)


def spd_forest(sizes=(330, 400, 450), side=12):
    # Disconnected dense SPD blocks plus a 3-D grid: several independent
    # elimination subtrees of very different work.
    blocks = [np.eye(size) + 0.01 * np.ones((size, size)) for size in sizes]
    return csc_array(block_diag(blocks + [poisson_3d(side)], format="csc"))


def chain_and_block(chain, width):
    narrow = diags(
        [-np.ones(chain - 1), 4.0 * np.ones(chain), -np.ones(chain - 1)],
        [-1, 0, 1], format="csc",
    )
    wide = np.eye(width) + 0.01 * np.ones((width, width))
    return csc_array(block_diag((narrow, wide), format="csc"))


def leading_blocks(defect=None, count=8, small=40, wide=330):
    # Eight small SPD blocks, then a wide one. Each small block is a subtree
    # factored concurrently on a worker (phase A); the wide block is a top
    # node (phase B). defect: "one" makes the fourth small block indefinite,
    # "all" every small block, "late" the wide block's last pivot.
    blocks = []
    for index in range(count):
        block = np.eye(small) + 0.01 * np.ones((small, small))
        if defect == "all" or (defect == "one" and index == 3):
            block[-1, -1] = -1.0
        blocks.append(csc_array(block))
    block = np.eye(wide) + 0.01 * np.ones((wide, wide))
    if defect == "late":
        block[-1, -1] = -1.0
    blocks.append(csc_array(block))
    return csc_array(block_diag(blocks, format="csc"))


def block_forest(first_pivot=None, size=321, count=4):
    # Equal dense SPD blocks; each is a whole subtree in phase A. With
    # first_pivot, the third block's first pivot is that tiny value and its
    # first row and column hold ones: 1e-300 gives L entries near 1e300 and a
    # negative second pivot, a subnormal 1e-310 makes L overflow to infinity.
    blocks = []
    for index in range(count):
        block = np.eye(size) + 0.01 * np.ones((size, size))
        if first_pivot is not None and index == 2:
            block[0, 1:] = block[1:, 0] = 1.0
            block[0, 0] = first_pivot
        blocks.append(csc_array(block))
    return csc_array(block_diag(blocks, format="csc"))


def pattern_digest(factor):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(factor.indptr, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(factor.indices, dtype="<i8").tobytes())
    return digest.hexdigest()[:16]


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
            matrix = dense_boundary(size)
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
        matrix = csc_array(block_diag((csc_array(first), csc_array(last)), format="csc"))
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
            matrix = chain_and_block(chain, width)
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

    def assert_same_symbolic_stats(self, serial, parallel):
        for key in ("numeric_engine", "supernodes", "max_supernode_width", "largest_front",
                    "dense_update_flops", "spd_panel_entries"):
            self.assertEqual(serial.stats[key], parallel.stats[key], key)

    def test_thread_sweep_is_bitwise_identical(self):
        cases = (
            ("grid-20", poisson_3d(20), "MMD_AT_PLUS_A"),
            ("cliques", overlapping_cliques()[0], "NATURAL"),
        )
        for name, matrix, order in cases:
            with self.subTest(case=name):
                serial = splu(matrix, backend="native", permc_spec=order, num_threads=1)
                self.assertEqual(serial.stats["numeric_engine"], "symmetric-supernodal-ldlt")
                self.assertEqual(serial.stats["spd_subtrees"], 0)
                for threads in (2, 3, 8):
                    parallel = splu(matrix, backend="native", permc_spec=order, num_threads=threads)
                    self.assert_same_factors(serial, parallel)
                    self.assert_same_symbolic_stats(serial, parallel)
                self.check_factor(matrix, serial, tolerance=3e-10)

    def test_forest_factors_subtrees_concurrently(self):
        matrix = spd_forest()
        for order in ("NATURAL", "MMD_AT_PLUS_A"):
            with self.subTest(order=order):
                serial = splu(matrix, backend="native", permc_spec=order, num_threads=1)
                self.assertEqual(serial.stats["numeric_engine"], "symmetric-supernodal-ldlt")
                self.assertEqual(serial.stats["parallel_steps"], 0)
                self.assertEqual(serial.stats["spd_subtrees"], 0)
                self.assertEqual(serial.stats["spd_subtree_seconds"], 0.0)
                self.check_factor(matrix, serial)
                for threads in (2, 4, 8):
                    parallel = splu(matrix, backend="native", permc_spec=order, num_threads=threads)
                    if parallel.stats["num_threads"] > 1:
                        self.assertGreater(parallel.stats["parallel_steps"], 0)
                        self.assertEqual(parallel.stats["threading_backend"], "std::thread")
                        # With AMD the grid splits into many small subtrees
                        # and the four trees are comparable, so each tree is
                        # factored on its own worker. In natural order the
                        # grid is a long chain of nodes with about 140
                        # contributors each that dominates the work, and
                        # phase A may not run at all.
                        if order == "MMD_AT_PLUS_A":
                            self.assertEqual(parallel.stats["spd_subtrees"], 4)
                            self.assertGreater(parallel.stats["spd_subtree_seconds"], 0.0)
                    self.assert_same_factors(serial, parallel)
                    self.assert_same_symbolic_stats(serial, parallel)

    def test_failure_inside_or_after_concurrent_subtrees_falls_back(self):
        cases = (
            ("one-subtree-indefinite", leading_blocks, "one"),
            ("all-subtrees-indefinite", leading_blocks, "all"),
            ("top-node-indefinite", leading_blocks, "late"),
            ("tiny-first-pivot", block_forest, 1e-300),
            # Subnormal: under flush-to-zero the zero pivot is rejected
            # instead, which also falls back.
            ("overflow-in-l", block_forest, 1e-310),
        )
        for name, build, defect in cases:
            with self.subTest(case=name):
                # The same structure without the defect runs subtrees
                # concurrently, so the failure below happens in or after them.
                control = build()
                factor = splu(control, backend="native", permc_spec="NATURAL", num_threads=4)
                self.assertEqual(factor.stats["numeric_engine"], "symmetric-supernodal-ldlt")
                if factor.stats["num_threads"] > 1:
                    self.assertGreaterEqual(factor.stats["spd_subtrees"], 2)

                matrix = build(defect)
                serial = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
                for threads in (2, 4, 8):
                    parallel = splu(matrix, backend="native", permc_spec="NATURAL",
                                    num_threads=threads)
                    for factor in (serial, parallel):
                        self.assertNotIn("ldlt", factor.stats["numeric_engine"])
                        self.assertEqual(factor.stats["parallel_panels"], 0)
                        self.assertEqual(factor.stats["spd_panel_entries"], 0)
                        self.assertEqual(factor.stats["spd_subtrees"], 0)
                        self.assertGreater(factor.stats["failed_spd_seconds"], 0.0)
                    self.assert_same_factors(serial, parallel)
                self.check_factor(matrix, serial)

    def test_symbolic_structure_is_pinned(self):
        # Engine, node statistics and the exact filled L pattern (SHA-256 of
        # the int64 CSC indptr and indices, first 16 hex digits). The values
        # were recorded with the row-subtree symbolic phase (find_nodes with
        # an explicit tail comparison) before the supernodal symbolic phase
        # replaced it; both must describe the same filled graph.
        cases = (
            ("grid-12", lambda: poisson_3d(12), (
                ("NATURAL", "symmetric-ldlt-csc", 1584, 145, 145, 231419, "3b346fe1412ee595"),
                ("COLAMD", "symmetric-ldlt-csc", 785, 225, 288, 145731, "a5cf7b2092d39ab7"),
                ("MMD_AT_PLUS_A", "symmetric-ldlt-csc", 1196, 181, 206, 76038, "eb086cedd2666c1c"),
            )),
            ("grid-16", lambda: poisson_3d(16), (
                ("COLAMD", "symmetric-ldlt-csc", 1784, 141, 576, 615656, "f65ae3331757ac49"),
            )),
            ("grid-20", lambda: poisson_3d(20), (
                ("COLAMD", "symmetric-supernodal-ldlt", 3312, 518, 939, 1858270, "58831466e9d8e959"),
                ("MMD_AT_PLUS_A", "symmetric-supernodal-ldlt", 5437, 608, 708, 842282, "e69bc39d0e282b80"),
            )),
            ("cliques", lambda: overlapping_cliques()[0], (
                ("NATURAL", "symmetric-supernodal-ldlt", 7, 350, 350, 108191, "0c8fbb50d1d36b34"),
            )),
            ("dense-319", lambda: dense_boundary(319), (
                ("NATURAL", "symmetric-ldlt-csc", 1, 319, 319, 51040, "4ab75b618838b0e8"),
            )),
            ("dense-320", lambda: dense_boundary(320), (
                ("NATURAL", "symmetric-supernodal-ldlt", 1, 320, 320, 51360, "2206f56f3977acef"),
            )),
            ("dense-352", lambda: dense_boundary(352), (
                ("MMD_AT_PLUS_A", "symmetric-supernodal-ldlt", 1, 352, 352, 62128, "e5da9014c404c910"),
            )),
            ("fan-in-scattered", lambda: fan_in_rejection(False, overflow=False), (
                ("NATURAL", "symmetric-supernodal-ldlt", 12, 400, 400, 105063, "6a8aca3676be2bff"),
            )),
            ("fan-in-contiguous", lambda: fan_in_rejection(True, overflow=False), (
                ("NATURAL", "symmetric-supernodal-ldlt", 12, 400, 400, 105063, "67e7510ba740e78e"),
            )),
            ("forest", spd_forest, (
                ("NATURAL", "symmetric-supernodal-ldlt", 1587, 450, 450, 467709, "c434ef4fccf3bb26"),
                ("MMD_AT_PLUS_A", "symmetric-supernodal-ldlt", 1199, 450, 450, 312328, "0fffc3d2f7fa3031"),
            )),
            ("chain-and-block", lambda: chain_and_block(20_000, 1000), (
                ("NATURAL", "symmetric-supernodal-ldlt", 20000, 1000, 1000, 540499, "58b5dd9c27438a0e"),
            )),
        )
        for name, build, expectations in cases:
            matrix = build()
            for order, engine, supernodes, width, front, lower_nnz, digest in expectations:
                with self.subTest(case=name, order=order):
                    factor = splu(matrix, backend="native", permc_spec=order, num_threads=1)
                    stats = factor.stats
                    self.assertEqual(stats["numeric_engine"], engine)
                    self.assertEqual(stats["supernodes"], supernodes)
                    self.assertEqual(stats["max_supernode_width"], width)
                    self.assertEqual(stats["largest_front"], front)
                    self.assertEqual(factor.L.nnz, lower_nnz)
                    self.assertEqual(pattern_digest(factor.L), digest)


if __name__ == "__main__":
    unittest.main()
