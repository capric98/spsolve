import unittest
from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import hadamard
from scipy.sparse import block_diag, csc_array, diags, kron, random_array, tril

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


def with_row_engine_block(values, size=200, diagonal=4.0, border=0.01):
    """A dense array followed, block-diagonally, by a size-row SPD arrow block.

    The block's last column holds size > 192 entries, so NATURAL skips the
    left-looking probe and factors the result in the row engine, while the
    leading block keeps its own pivots and scales. Small NATURAL fixtures
    use it to keep the row engine's branches covered, since NATURAL factors
    them alone in the CSC engine. The block is exactly symmetric and
    diagonally dominant, so a symmetric input stays symmetric and still
    tries the SPD path first.
    """
    block = diagonal * np.eye(size)
    block[-1, :-1] = block[:-1, -1] = border
    return block_diag((csc_array(np.asarray(values)), csc_array(block))).toarray()


def exact_partial_pivoting(matrix, factor, diag_pivot_thresh=1.0):
    """Row order of threshold partial pivoting in exact arithmetic.

    Eliminates the scaled matrix in the factor's column order with the
    documented rule: the largest magnitude, replaced by the preferred row
    perm_c[k] when it reaches the threshold, with exact ties among the largest
    broken by the smallest current row position (rows start at their labels,
    and each pivot swaps positions with the row at position k). Returns the
    row order, the number of columns a position tie decided, and whether every
    value was dyadic, in which case floating-point elimination is exact too.
    """
    dense = matrix.toarray()
    n = dense.shape[0]
    position = list(range(n))
    row_at = list(range(n))
    pivoted = set()
    columns = []
    ties = 0
    dyadic = True
    for k in range(n):
        col = int(factor.perm_c[k])
        values = {
            int(row): Fraction(float(
                dense[row, col] * factor.row_scale[row] * factor.col_scale[col]))
            for row in np.flatnonzero(dense[:, col])
        }
        for pivot_row, multipliers in columns:
            coefficient = values.get(pivot_row, 0)
            if coefficient:
                for row, multiplier in multipliers.items():
                    values[row] = values.get(row, Fraction(0)) - multiplier * coefficient
        dyadic = dyadic and all(
            value.denominator & (value.denominator - 1) == 0 for value in values.values()
        )
        candidates = {row: value for row, value in values.items() if row not in pivoted}
        maximum = max(abs(value) for value in candidates.values())
        largest = [row for row, value in candidates.items() if abs(value) == maximum]
        pivot = min(largest, key=position.__getitem__)
        preferred = candidates.get(col, 0)
        if preferred != 0 and abs(preferred) >= diag_pivot_thresh * maximum:
            pivot = col
        elif len(largest) > 1:
            ties += 1
        old, displaced = position[pivot], row_at[k]
        row_at[k], row_at[old] = pivot, displaced
        position[pivot], position[displaced] = k, old
        pivoted.add(pivot)
        value = candidates[pivot]
        columns.append((pivot, {
            row: entry / value for row, entry in candidates.items()
            if row != pivot and entry != 0
        }))
    return row_at, ties, dyadic


class SparseLUTests(unittest.TestCase):
    def assert_same_factors(self, first, second):
        for name in ("perm_r", "perm_c", "row_scale", "col_scale"):
            assert_array_equal(getattr(first, name), getattr(second, name))
        for name in ("L", "U"):
            for field in ("indptr", "indices", "data"):
                assert_array_equal(getattr(getattr(first, name), field),
                                   getattr(getattr(second, name), field))

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
                # block_diag stores every entry of the dense np.eye(40), zeros
                # included; the sparse second block makes it return an array.
                matrix = csc_array(block_diag((np.eye(40), csc_array(dense)), format="csc"))
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

    def test_left_looking_reports_nonfinite_candidates_and_singular_rows(self):
        n = 40
        band = diags(
            [4.0 * np.ones(n), -np.ones(n - 1), 0.25 * np.ones(n - 9)],
            [0, 1, 9], shape=(n, n), format="csc",
        ).toarray()
        orderings = ("COLAMD", "MMD_AT_PLUS_A")
        for ordering in orderings:
            factor = splu(csc_array(band), backend="native", permc_spec=ordering)
            self.assertEqual(factor.stats["numeric_engine"], "left-looking-csc")
        # Each planted value is, in some ordering here, the only candidate of
        # its column: a NaN magnitude never wins the maximum comparison, so it
        # must be reported as non-finite rather than as an empty column.
        for value in (np.nan, np.inf):
            for position in (1, 2, 5, 12):
                planted = band.copy()
                planted[position, position] = value
                for ordering in orderings:
                    for threads in (1, 4):
                        with self.subTest(value=value, position=position,
                                          ordering=ordering, threads=threads):
                            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                                splu(csc_array(planted), backend="native",
                                     permc_spec=ordering, num_threads=threads,
                                     check_finite=False)
        # A duplicated row stays exactly singular; a perturbed copy with the
        # same pattern shows that this pattern runs in the CSC engine.
        duplicated = band.copy()
        duplicated[21, :] = duplicated[20, :]
        perturbed = duplicated.copy()
        perturbed[21, 21] = 3.0
        for ordering in orderings:
            factor = splu(csc_array(perturbed), backend="native", permc_spec=ordering)
            self.assertEqual(factor.stats["numeric_engine"], "left-looking-csc")
            for threads in (1, 4):
                with self.subTest(ordering=ordering, threads=threads, case="duplicated"):
                    with self.assertRaisesRegex(np.linalg.LinAlgError, "singular"):
                        splu(csc_array(duplicated), backend="native",
                             permc_spec=ordering, num_threads=threads)

    def test_left_looking_random_bands_are_accurate_and_thread_invariant(self):
        # Narrow bands with uniform, +-1 and small-integer values (the last
        # two full of ties), half of them row-permuted and some with entries
        # outside the band. A column's search order depends only on the
        # matrix and the pivots so far, so every thread count gives the same
        # factors, and threshold 1 keeps every multiplier at most 1.
        rng = np.random.default_rng(6061)
        samplers = (
            lambda size: rng.uniform(-1.0, 1.0, size),
            lambda size: rng.choice([-1.0, 1.0], size=size),
            lambda size: rng.choice([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], size=size),
        )
        n = 300
        expected = rng.normal(size=n)
        for case in range(6):
            values = samplers[case % 3]
            # Draw until the matrix is well conditioned (deterministic).
            while True:
                offsets = sorted({0, *rng.integers(-6, 7, size=4).tolist()})
                matrix = diags([values(n - abs(offset)) for offset in offsets], offsets,
                               shape=(n, n), format="csc")
                if case >= 3:
                    rows = rng.integers(0, n, size=12)
                    cols = np.clip(rows + rng.integers(-40, 41, size=12), 0, n - 1)
                    matrix = matrix + csc_array((values(12), (rows, cols)), shape=(n, n))
                matrix = csc_array(matrix)
                matrix.eliminate_zeros()
                if np.linalg.cond(matrix.toarray()) < 1e8:
                    break
            permuted = case % 2 == 0
            if permuted:
                matrix = csc_array(matrix[rng.permutation(n), :])
            for ordering in ("COLAMD",) if permuted else ("COLAMD", "MMD_AT_PLUS_A"):
                with self.subTest(case=case, ordering=ordering):
                    factors = [
                        splu(matrix, backend="native", permc_spec=ordering, num_threads=threads)
                        for threads in (1, 2, 3, 8)
                    ]
                    reference = factors[0]
                    self.assertEqual(reference.stats["numeric_engine"], "left-looking-csc")
                    self.assertLessEqual(abs(reference.L).max(), 1.0)
                    self.assert_factor_identity(matrix, reference)
                    assert_allclose(
                        reference.solve(matrix @ expected, iterative_refinement=False),
                        expected, rtol=1e-9, atol=1e-9,
                    )
                    for factor in factors[1:]:
                        self.assert_same_factors(reference, factor)

    def test_left_looking_breaks_exact_ties_by_row_position(self):
        # Sign-flipped 8x8 Hadamard blocks under a random row permutation:
        # every elimination value is dyadic, so floating-point elimination is
        # exact in whatever order updates reach an entry, and most columns
        # hold several largest candidates of equal magnitude. The pivots must
        # be those of the documented rule.
        rng = np.random.default_rng(0)
        block = hadamard(8).astype(np.float64)
        matrix = csc_array(block_diag(
            [csc_array(block * rng.choice([-1.0, 1.0], size=(8, 1))) for _ in range(16)],
            format="csc",
        ))
        matrix = csc_array(matrix[rng.permutation(matrix.shape[0]), :])
        for ordering in ("COLAMD", "MMD_AT_PLUS_A"):
            factors = []
            for threads in (1, 4):
                with self.subTest(ordering=ordering, threads=threads):
                    factor = splu(matrix, backend="native", permc_spec=ordering,
                                  num_threads=threads)
                    factors.append(factor)
                    self.assertEqual(factor.stats["numeric_engine"], "left-looking-csc")
                    order, ties, dyadic = exact_partial_pivoting(matrix, factor)
                    self.assertTrue(dyadic)
                    self.assertGreater(ties, 64)
                    assert_array_equal(factor.perm_r, order)
                    self.assert_factor_identity(matrix, factor, atol=0.0)
            self.assert_same_factors(*factors)

    def test_row_engine_breaks_exact_ties_by_row_position(self):
        # The Hadamard blocks of the test above, factored by the row engine:
        # with NATURAL and a dyadic arrow block whose last column is wider
        # than the probe's reach limit, every value stays dyadic.
        rng = np.random.default_rng(0)
        block = hadamard(8).astype(np.float64)
        blocks = block_diag(
            [csc_array(block * rng.choice([-1.0, 1.0], size=(8, 1))) for _ in range(16)],
            format="csc",
        ).toarray()
        blocks = blocks[rng.permutation(blocks.shape[0]), :]
        matrix = csc_array(with_row_engine_block(blocks, diagonal=64.0, border=0.25))
        factors = []
        for threads in (1, 4):
            with self.subTest(threads=threads):
                factor = splu(matrix, backend="native", permc_spec="NATURAL",
                              num_threads=threads)
                factors.append(factor)
                self.assertEqual(factor.stats["numeric_engine"], "row-hybrid")
                order, ties, dyadic = exact_partial_pivoting(matrix, factor)
                self.assertTrue(dyadic)
                self.assertGreater(ties, 64)
                assert_array_equal(factor.perm_r, order)
                self.assert_factor_identity(matrix, factor, atol=0.0)
        self.assert_same_factors(*factors)

    def test_preferred_row_at_the_threshold_is_kept(self):
        # Unit row and column maxima keep the scaled values exact: the
        # preferred row 0 holds half the column maximum, which threshold 0.5
        # accepts and threshold 0.75 rejects, in either engine.
        small = np.array([[0.5, 1.0], [1.0, 0.25]])
        for engine, values in (("left-looking-csc", small),
                               ("row-hybrid", with_row_engine_block(small))):
            matrix = csc_array(values)
            for threshold, swaps in ((0.5, 0), (0.75, 1)):
                with self.subTest(engine=engine, threshold=threshold):
                    factor = splu(matrix, backend="native", permc_spec="NATURAL",
                                  diag_pivot_thresh=threshold)
                    self.assertEqual(factor.stats["numeric_engine"], engine)
                    self.assertEqual(factor.stats["pivot_swaps"], swaps)
                    self.assert_factor_identity(matrix, factor)

    def test_left_looking_duplicated_row_of_permuted_band_is_singular(self):
        # Two equal rows receive the same updates in the same order: each
        # update comes from a column that both rows depend on, and the search
        # places such columns before the one that pivots the first copy. The
        # second copy therefore cancels exactly in any search order.
        rng = np.random.default_rng(7171)
        n = 400
        offsets = [-7, -1, 0, 2, 9]
        band = diags(
            [rng.choice([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], size=n - abs(offset))
             for offset in offsets],
            offsets, shape=(n, n), format="csc",
        ).toarray()
        band[211, :] = band[173, :]
        perturbed = band.copy()
        perturbed[211, 173] += 0.5
        for ordering, rows in (("COLAMD", rng.permutation(n)),
                               ("MMD_AT_PLUS_A", np.arange(n))):
            factor = splu(csc_array(perturbed[rows, :]), backend="native",
                          permc_spec=ordering)
            self.assertEqual(factor.stats["numeric_engine"], "left-looking-csc")
            for threads in (1, 4):
                with self.subTest(ordering=ordering, threads=threads):
                    with self.assertRaisesRegex(np.linalg.LinAlgError, "singular"):
                        splu(csc_array(band[rows, :]), backend="native",
                             permc_spec=ordering, num_threads=threads)

    def test_colamd_probe_restart_bounds_discarded_reservations(self):
        # A restarted probe's discarded CSC arrays count as workspace at their
        # capacity. Each of the four arrays reserves at most a dense triangle,
        # and a probe that must restart (an input column wider than the
        # 192-row reach limit) reserves only O(n) entries.
        for dense_size in (140, 260):
            with self.subTest(dense_size=dense_size):
                rng = np.random.default_rng(555)
                dense = rng.uniform(-0.5, 0.5, size=(dense_size, dense_size))
                dense += 300.0 * np.eye(dense_size)
                # block_diag stores every entry of the dense np.eye(40), zeros
                # included; the sparse second block makes it return an array.
                matrix = csc_array(block_diag((np.eye(40), csc_array(dense)), format="csc"))
                n = matrix.shape[0]
                matrix = csc_array(matrix[rng.permutation(n), :])
                factor = splu(matrix, backend="native", permc_spec="COLAMD")
                self.assertTrue(factor.stats["numeric_probe_fallback"])
                workspace = factor.stats["peak_workspace_bytes"]
                if dense_size > 192:
                    self.assertLess(workspace, 256 * n)
                else:
                    # Probe vectors, restart snapshot and column pointers,
                    # then four arrays of 8-byte entries.
                    fixed = 8 * (11 * n + 2 * (n + 1))
                    self.assertLessEqual(workspace, fixed + 32 * (n * (n + 1) // 2))
                    self.assertGreater(workspace, 256 * n)

    def test_colamd_probe_restart_after_staged_reserve_reports_its_arrays(self):
        # A narrow band (7 entries per column) beside a full band of
        # half-width 95 (191 entries per column, so the restart is not certain
        # from the input): the probe passes the reserve's second stage after
        # max(256, n / 16) columns, then restarts in the wide part. The
        # discarded arrays it reports are sized from the factor so far, a few
        # entries per row, not from nnz(A), which the wide part dominates.
        rng = np.random.default_rng(91)

        def band(size, half):
            offsets = list(range(-half, half + 1))
            return diags([rng.uniform(-1.0, 1.0, size - abs(offset)) for offset in offsets],
                         offsets, shape=(size, size), format="csc")

        matrix = csc_array(block_diag((band(3000, 3), band(600, 95)), format="csc"))
        n = matrix.shape[0]
        matrix = csc_array(matrix[rng.permutation(n), :])
        factor = splu(matrix, backend="native", permc_spec="COLAMD")
        stats = factor.stats
        self.assertEqual(stats["numeric_engine"], "row-hybrid")
        self.assertTrue(stats["numeric_probe_fallback"])
        self.assertGreater(stats["numeric_probe_columns"], max(256, n // 16))
        fixed = 8 * (11 * n + 2 * (n + 1))
        self.assertLess(stats["peak_workspace_bytes"], fixed + 32 * 16 * n)
        # The bound is well below a reserve of 2 * nnz(A) entries per array.
        self.assertLess(16 * n, matrix.nnz // 4 * 2)
        self.assert_factor_identity(matrix, factor)

    def test_natural_probe_keeps_narrow_bands_in_csc(self):
        # NATURAL runs the bounded left-looking probe as well. A narrow
        # nonsymmetric band, with its rows in order or permuted, completes in
        # the CSC engine in the identity column order, with the same factors
        # at every thread count.
        n = 1000
        base = csc_array(
            diags(
                [-0.2 * np.ones(n - 7), -np.ones(n - 1), 4.0 * np.ones(n),
                 0.5 * np.ones(n - 2), 0.1 * np.ones(n - 9)],
                [-7, -1, 0, 2, 9], format="csc",
            )
        )
        expected = np.random.default_rng(323).normal(size=(n, 3))
        for label, rows in (("in-order", np.arange(n)),
                            ("permuted", np.random.default_rng(321).permutation(n))):
            matrix = csc_array(base[rows, :])
            saved = matrix.data.copy()
            factors = []
            for threads in (1, 4):
                with self.subTest(rows=label, threads=threads):
                    factor = splu(matrix, backend="native", permc_spec="NATURAL",
                                  num_threads=threads)
                    factors.append(factor)
                    stats = factor.stats
                    self.assertEqual(stats["numeric_engine"], "left-looking-csc")
                    self.assertEqual(stats["ordering_backend"], "builtin")
                    assert_array_equal(factor.perm_c, np.arange(n))
                    self.assertFalse(stats["numeric_probe_fallback"])
                    self.assertEqual(stats["numeric_probe_columns"], n)
                    self.assertLess(stats["numeric_probe_max_front"], 40)
                    self.assertGreater(stats["numeric_probe_seconds"], 0.0)
                    self.assert_factor_identity(matrix, factor)
                    for trans, operator in (("N", matrix), ("T", matrix.T)):
                        assert_allclose(
                            factor.solve(operator @ expected, trans=trans,
                                         iterative_refinement=False),
                            expected, rtol=2e-13, atol=2e-13,
                        )
            self.assert_same_factors(*factors)
            assert_array_equal(matrix.data, saved)

    def test_natural_probe_skips_or_restarts_where_the_row_engine_is_faster(self):
        # NATURAL does not start the probe when the row engine would store a
        # row dense from its setup (at least 64 entries filling a quarter of
        # its span) or when an input column is wider than the 192-row reach
        # limit; it reports a fallback with zero counters. A probe that
        # starts restarts after a column with more than 64 multipliers, once
        # its updates so far project beyond 64 * nnz(A), or at the bounds it
        # shares with COLAMD. Rows of 63 entries, below the dense test, stay
        # in the CSC engine.
        rng = np.random.default_rng(97)

        def dense(size):
            return csc_array(rng.uniform(-1.0, 1.0, (size, size)) + 2.0 * np.eye(size) * size)

        def band(size, half):
            offsets = list(range(-half, half + 1))
            values = diags([rng.uniform(-1.0, 1.0, size - abs(offset)) for offset in offsets],
                           offsets, format="csc")
            return csc_array(values + diags(np.full(size, float(half)), format="csc"))

        def schur(size, split=96):
            # Sparse rows (21 entries) whose elimination fills a dense
            # size-by-size block; the probe restarts in that block.
            values = 8.0 * np.eye(split + size)
            for row in range(split):
                cols = split + rng.choice(size, 20, replace=False)
                values[row, cols] = rng.uniform(-0.15, 0.15, 20)
            for row in range(split, split + size):
                cols = rng.choice(split, 20, replace=False)
                values[row, cols] = rng.uniform(-0.15, 0.15, 20)
            return csc_array(values)

        def grid(side):
            # 2-D convection-diffusion in natural order: five entries per
            # row, but fill as wide as the grid side.
            one = diags([-1.5 * np.ones(side - 1), 2.0 * np.ones(side), -0.5 * np.ones(side - 1)],
                        [-1, 0, 1], format="csc")
            identity = diags(np.ones(side), format="csc")
            return csc_array(kron(identity, one) + kron(one, identity))

        # Sparse rows, but one input column wider than the reach limit.
        bordered = diags([-np.ones(299), 4.0 * np.ones(300), 0.5 * np.ones(299)],
                         [-1, 0, 1]).toarray()
        bordered[:, -1] = 0.25
        bordered[-1, -1] = 4.0
        cases = (
            # name, matrix, engine, how the probe ended
            ("dense-63", dense(63), "left-looking-csc", "completed"),
            ("dense-64", dense(64), "row-hybrid", "skipped"),
            ("band-30", band(400, 30), "left-looking-csc", "completed"),
            ("band-32", band(400, 32), "row-hybrid", "skipped"),
            ("wide-column", csc_array(bordered), "row-hybrid", "skipped"),
            ("fill-induced-block", schur(128), "row-hybrid", "restarted"),
            ("random-fill", general_matrix(320, 0.025, 14), "row-hybrid", "restarted"),
            ("grid-17", grid(17), "left-looking-csc", "completed"),
            ("grid-24", grid(24), "row-hybrid", "restarted"),
        )
        for name, matrix, engine, probe in cases:
            n = matrix.shape[0]
            expected = np.random.default_rng(98).normal(size=(n, 2))
            factors = []
            for threads in (1, 4):
                with self.subTest(case=name, threads=threads):
                    factor = splu(matrix, backend="native", permc_spec="NATURAL",
                                  num_threads=threads)
                    factors.append(factor)
                    stats = factor.stats
                    self.assertEqual(stats["numeric_engine"], engine)
                    self.assertEqual(stats["numeric_probe_fallback"], probe != "completed")
                    if probe == "skipped":
                        self.assertEqual(stats["numeric_probe_columns"], 0)
                        self.assertEqual(stats["numeric_probe_max_front"], 0)
                        self.assertEqual(stats["numeric_probe_updates"], 0)
                        # Only the pattern check, which numeric time includes.
                        self.assertLessEqual(stats["numeric_probe_seconds"],
                                             stats["numeric_seconds"])
                    elif probe == "restarted":
                        self.assertGreater(stats["numeric_probe_columns"], 0)
                        self.assertLess(stats["numeric_probe_columns"], n)
                        if name == "fill-induced-block":
                            # The 96 sparse columns complete; the block's
                            # first column has over 64 multipliers, and the
                            # rest of the block would cost far more than the
                            # few updates done so far.
                            self.assertEqual(stats["numeric_probe_columns"], 97)
                            self.assertLessEqual(stats["numeric_probe_max_front"], 192)
                        if name == "grid-24":
                            # The work so far projects beyond 64 * nnz(A)
                            # updates long before the probe would reach it
                            # (after about 336 of the 576 columns).
                            self.assertLess(stats["numeric_probe_columns"], 100)
                            self.assertLess(stats["numeric_probe_updates"], 8 * matrix.nnz)
                        self.assertGreater(stats["numeric_probe_seconds"], 0.0)
                        self.assertGreaterEqual(stats["numeric_seconds"],
                                                stats["numeric_probe_seconds"])
                    else:
                        self.assertEqual(stats["numeric_probe_columns"], n)
                    self.assert_factor_identity(matrix, factor)
                    for trans, operator in (("N", matrix), ("T", matrix.T)):
                        assert_allclose(
                            factor.solve(operator @ expected, trans=trans,
                                         iterative_refinement=False),
                            expected, rtol=1e-11, atol=1e-11,
                        )
            self.assert_same_factors(*factors)

    def test_natural_probe_rules_hold_at_their_thresholds(self):
        # Each NATURAL rule on both sides of its threshold:
        # - the pattern check: one row of 64 entries spanning 256 columns is
        #   stored dense by the row engine (64 * 4 >= 256), spanning 257 it
        #   is not;
        # - the multiplier limit: a fill-induced block of 65 rows gives 64
        #   multipliers and completes, one of 66 rows restarts after the 96
        #   sparse columns and the block's first column;
        # - natural_restart_pays after the multiplier limit: behind a band,
        #   the 66-row block's first column (65 multipliers) suggests
        #   65^2 * 65 updates for its 65 remaining columns; the probe
        #   restarts while that exceeds four times the updates done (band of
        #   1,000 columns: 62,745 updates) and finishes otherwise (1,100
        #   columns: 69,045, above 65^3 / 4 = 68,656 but below
        #   65^2 * 66 / 4 = 69,712, so one more remaining column would
        #   restart it);
        # - its horizon cap: with a band of 300 columns after the block, 365
        #   columns remain but only 192 count, 65^2 * 192 = 811,200, so the
        #   probe restarts below 202,800 updates done (band of 3,215 columns:
        #   202,290) and finishes above (3,232 columns: 203,361); horizons of
        #   191 or 193 columns would move the threshold to 201,744 or
        #   203,856 and flip one of the two;
        # - natural_restart_pays after the per-column limit: in `chain`, 129
        #   columns with at most 64 multipliers each feed one column whose
        #   updates pass 8,192 (8,253 at its last source); with 16,444
        #   updates done (the triggering source's 62 included) and 61
        #   columns remaining the restart pays, 8,253 * 61 = 503,433 >
        #   4 * 16,444; behind a band of 1,746 columns and a tridiagonal
        #   block of 37, 125,889 updates are done and the probe finishes,
        #   503,433 <= 4 * 125,889 = 503,556, a margin smaller than one more
        #   remaining column (8,253) and than the source's updates (4 * 62);
        #   behind a band of 8,000 columns a 128-row block also passes the
        #   per-column limit late in the block without restarting;
        # - the projected work: the 2-D grid of side 32 (1,024 rows) restarts
        #   at column 64 = 1024 / 16, the first column checked.
        def schur(size, seed=11, split=96):
            rng = np.random.default_rng(seed)
            values = 8.0 * np.eye(split + size)
            for row in range(split):
                cols = split + rng.choice(size, 20, replace=False)
                values[row, cols] = rng.uniform(-0.15, 0.15, 20)
            for row in range(split, split + size):
                cols = rng.choice(split, 20, replace=False)
                values[row, cols] = rng.uniform(-0.15, 0.15, 20)
            return csc_array(values)

        def band(n):
            return csc_array(diags(
                [-0.2 * np.ones(n - 7), -np.ones(n - 1), 4.0 * np.ones(n),
                 0.5 * np.ones(n - 2), 0.1 * np.ones(n - 9)],
                [-7, -1, 0, 2, 9], format="csc"))

        def tridiagonal(n):
            return csc_array(diags([-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)],
                                   [-1, 0, 1], format="csc"))

        def then(*parts):
            return csc_array(block_diag(parts, format="csc"))

        def chain(sources=129, shared=62):
            # Column j < 129 updates column j + 1 (superdiagonal 1.0), so
            # every source carries the 62 shared rows of column 0 plus two
            # subdiagonal rows (64 multipliers; 63 and 62 for the last two).
            # Column 129 couples to all 129 sources; its reach is 191 rows.
            n = sources + shared
            values = 4.0 * np.eye(n)
            values[sources:, 0] = 0.01
            for j in range(sources):
                for d in (1, 2):
                    if j + d < sources:
                        values[j + d, j] = 0.01
                if j > 0:
                    values[j - 1, j] = 1.0
            values[:sources, sources] = 0.01
            return csc_array(values)

        def grid(side):
            one = diags([-1.5 * np.ones(side - 1), 2.0 * np.ones(side),
                         -0.5 * np.ones(side - 1)], [-1, 0, 1], format="csc")
            return csc_array(kron(diags(np.ones(side)), one) + kron(one, diags(np.ones(side))))

        def spread_row(span, n=300, row=10, count=64):
            values = 5.0 * np.eye(n)
            cols = np.round(np.linspace(row, row + span - 1, count)).astype(int)
            self.assertEqual(np.unique(cols).size, count)
            values[row, cols[1:]] = 0.01
            return csc_array(values)

        cases = (
            # name, matrix, completed probe columns (None: completes in CSC)
            ("span-256", spread_row(256), 0),
            ("span-257", spread_row(257), None),
            ("block-65", schur(65), None),
            ("block-66", schur(66), 97),
            ("band-1000+block-66", then(band(1000), schur(66)), 1097),
            ("band-1100+block-66", then(band(1100), schur(66)), None),
            ("band-3215+block-66+band-300", then(band(3215), schur(66), band(300)), 3312),
            ("band-3232+block-66+band-300", then(band(3232), schur(66), band(300)), None),
            ("chain", chain(), 129),
            ("band-1746+tridiagonal-37+chain", then(band(1746), tridiagonal(37), chain()), None),
            ("band-8000+block-128", then(band(8000), schur(128)), None),
            ("grid-32", grid(32), 64),
        )
        for name, matrix, restart in cases:
            n = matrix.shape[0]
            factors = []
            for threads in (1, 4):
                with self.subTest(case=name, threads=threads):
                    factor = splu(matrix, backend="native", permc_spec="NATURAL",
                                  num_threads=threads)
                    factors.append(factor)
                    stats = factor.stats
                    if restart is None:
                        self.assertEqual(stats["numeric_engine"], "left-looking-csc")
                        self.assertEqual(stats["numeric_probe_columns"], n)
                    else:
                        self.assertEqual(stats["numeric_engine"], "row-hybrid")
                        self.assertTrue(stats["numeric_probe_fallback"])
                        self.assertEqual(stats["numeric_probe_columns"], restart)
                    scaled = csc_array(matrix.multiply(factor.row_scale[:, None])
                                       .multiply(factor.col_scale[None, :]))
                    error = scaled[factor.perm_r, :][:, factor.perm_c] - factor.L @ factor.U
                    self.assertLess(np.max(abs(error.data), initial=0.0), 1e-13)
            self.assert_same_factors(*factors)

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
            for kind, values, row_engine in (
                ("nonsymmetric", nonsymmetric, False),
                ("near_symmetric", near_symmetric, False),
                # The same systems factored by the row engine (see
                # with_row_engine_block), which NATURAL no longer selects for
                # them alone.
                ("nonsymmetric+block", with_row_engine_block(nonsymmetric), True),
                ("near_symmetric+block", with_row_engine_block(near_symmetric), True),
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
                        if row_engine and ordering == "NATURAL":
                            self.assertEqual(factor.stats["numeric_engine"], "row-hybrid")
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
        for scale in (1e-16, 1.0, 1e16):
            small = scale * np.array([[0.0, 1.0], [1.0, 0.0]])
            # The second variant stays exactly symmetric (the appended block
            # is SPD), fails the SPD attempt in the small block, and is
            # factored by the row engine.
            for row_engine, values in ((False, small), (True, with_row_engine_block(small))):
                matrix = csc_array(values)
                expected = np.arange(matrix.shape[0], dtype=np.float64) + 1.0
                expected[:2] = (2.0, -3.0)
                for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
                    with self.subTest(scale=scale, ordering=ordering, row_engine=row_engine):
                        factor = splu(
                            matrix,
                            backend="native",
                            permc_spec=ordering,
                            num_threads=4,
                        )
                        self.assertNotEqual(
                            factor.stats["numeric_engine"], "symmetric-ldlt-csc"
                        )
                        if row_engine:
                            self.assertGreater(factor.stats["failed_spd_seconds"], 0.0)
                            if ordering == "NATURAL":
                                self.assertEqual(factor.stats["numeric_engine"], "row-hybrid")
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
        small = general_matrix(40, 0.12, 10)
        for row_engine, matrix in ((False, small),
                                   (True, csc_array(with_row_engine_block(small.toarray())))):
            for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
                with self.subTest(ordering=ordering, row_engine=row_engine):
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
                    if row_engine and ordering == "NATURAL":
                        self.assertEqual(factor.stats["numeric_engine"], "row-hybrid")

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
        # A zero preferred pivot is never taken, even with threshold zero.
        # The second variant runs the row engine's preferred-row test.
        small = np.array([[0.0, 1.0], [1.0, 1.0]])
        for engine, values in (("left-looking-csc", small),
                               ("row-hybrid", with_row_engine_block(small))):
            with self.subTest(engine=engine):
                matrix = csc_array(values)
                expected = np.arange(matrix.shape[0], dtype=np.float64) + 1.0
                expected[:2] = (2.0, -3.0)
                factor = splu(
                    matrix,
                    backend="native",
                    permc_spec="NATURAL",
                    diag_pivot_thresh=0.0,
                )
                self.assertEqual(factor.stats["numeric_engine"], engine)
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
