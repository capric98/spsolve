import unittest
from unittest.mock import patch
from warnings import catch_warnings, simplefilter

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array

from spsolve import splu


class IterativeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.matrix = csc_array(
            [[4.0, 1.0, 0.0], [-2.0, 5.0, 0.5], [0.0, 1.0, 3.0]]
        )
        self.factor = splu(self.matrix, backend="native", permc_spec="NATURAL")
        self.expected = np.array([1.0, -2.0, 3.0])

    def inaccurate_solves(self, correction_weights):
        native_solve = self.factor._solve_once
        calls = 0

        def solve(rhs, **kwargs):
            nonlocal calls
            solved = native_solve(rhs, **kwargs)
            call = calls
            calls += 1
            if call == 0:
                return solved + 0.5
            weight = correction_weights[call - 1]
            if not np.isfinite(weight):
                return np.full_like(solved, weight)
            return weight * solved

        return solve

    def test_convergence_on_final_correction_preserves_solve_semantics(self):
        for trans in ("N", "T", "H"):
            operator = self.matrix if trans == "N" else self.matrix.T
            for multiple in (False, True):
                expected = (
                    np.column_stack((self.expected, self.expected[::-1]))
                    if multiple
                    else self.expected
                )
                for overwrite in (False, True):
                    with self.subTest(trans=trans, multiple=multiple, overwrite=overwrite):
                        rhs = operator @ expected
                        original = rhs.copy()
                        with (
                            patch.object(
                                self.factor,
                                "_solve_once",
                                side_effect=self.inaccurate_solves((0.5, 1.0)),
                            ) as solve,
                            catch_warnings(record=True) as warnings,
                        ):
                            simplefilter("always", RuntimeWarning)
                            result = self.factor.solve(
                                rhs, trans=trans, overwrite_b=overwrite, num_threads=2
                            )
                        self.assertEqual(warnings, [])
                        self.assertEqual(solve.call_count, 3)
                        for call in solve.call_args_list:
                            self.assertEqual(call.kwargs, {"trans": trans, "num_threads": 2})
                        self.assertEqual(result.shape, expected.shape)
                        assert_allclose(result, expected, rtol=2e-14, atol=2e-14)
                        assert_allclose(operator @ result, original, rtol=2e-14, atol=2e-14)
                        if overwrite:
                            self.assertIs(result, rhs)
                        else:
                            assert_array_equal(rhs, original)

    def test_exhausted_correction_budget_warns_after_final_solution(self):
        rhs = self.matrix @ self.expected
        with (
            patch.object(
                self.factor,
                "_solve_once",
                side_effect=self.inaccurate_solves((0.5, 0.5)),
            ) as solve,
            self.assertWarnsRegex(RuntimeWarning, "iterative refinement did not converge"),
        ):
            result = self.factor.solve(rhs)
        self.assertEqual(solve.call_count, 3)
        assert_allclose(result, self.expected + 0.125, rtol=2e-14, atol=2e-14)
        self.assertGreater(np.max(np.abs(rhs - self.matrix @ result)), 0.1)

    def test_stalled_or_worsening_correction_retains_best_iterate(self):
        for weights, retained_offset in (((0.0,), 0.5), ((-1.0,), 0.5), ((0.5, -1.0), 0.25)):
            with self.subTest(correction_weights=weights):
                rhs = self.matrix @ self.expected
                with (
                    patch.object(
                        self.factor,
                        "_solve_once",
                        side_effect=self.inaccurate_solves(weights),
                    ) as solve,
                    self.assertWarnsRegex(RuntimeWarning, "iterative refinement did not converge"),
                ):
                    result = self.factor.solve(rhs, overwrite_b=True)
                self.assertEqual(solve.call_count, 1 + len(weights))
                self.assertIs(result, rhs)
                assert_allclose(result, self.expected + retained_offset, rtol=2e-14, atol=2e-14)

    def test_nonfinite_final_correction_warns_and_retains_best_iterate(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(correction=invalid):
                with (
                    patch.object(
                        self.factor,
                        "_solve_once",
                        side_effect=self.inaccurate_solves((0.5, invalid)),
                    ) as solve,
                    catch_warnings(record=True) as warnings,
                ):
                    simplefilter("always", RuntimeWarning)
                    result = self.factor.solve(self.matrix @ self.expected)
                self.assertEqual(solve.call_count, 3)
                self.assertEqual(len(warnings), 1)
                self.assertIn("iterative refinement did not converge", str(warnings[0].message))
                assert_allclose(result, self.expected + 0.25, rtol=2e-14, atol=2e-14)

    def test_nonfinite_error_cannot_be_reported_as_converged(self):
        factor = splu(csc_array(np.eye(2)), backend="native")
        for rhs in (
            np.array([np.nan, 1.0]),
            np.array([np.inf, 1.0]),
            np.full(2, 0.75 * np.finfo(np.float64).max),
        ):
            with self.subTest(rhs=rhs):
                with (
                    patch.object(factor, "_solve_once", wraps=factor._solve_once) as solve,
                    catch_warnings(record=True) as warnings,
                ):
                    simplefilter("always", RuntimeWarning)
                    factor.solve(rhs)
                self.assertEqual(solve.call_count, 1)
                self.assertEqual(len(warnings), 1)
                self.assertIn("iterative refinement did not converge", str(warnings[0].message))

    def test_accurate_native_solves_and_zero_rhs_need_no_correction(self):
        for trans in ("N", "T", "H"):
            operator = self.matrix if trans == "N" else self.matrix.T
            for expected in (self.expected, np.zeros((3, 2)), np.empty((3, 0))):
                with self.subTest(trans=trans, shape=expected.shape):
                    with (
                        patch.object(
                            self.factor, "_solve_once", wraps=self.factor._solve_once
                        ) as solve,
                        catch_warnings(record=True) as warnings,
                    ):
                        simplefilter("always", RuntimeWarning)
                        result = self.factor.solve(operator @ expected, trans=trans)
                    self.assertEqual(warnings, [])
                    self.assertEqual(solve.call_count, 1)
                    assert_allclose(result, expected, rtol=2e-14, atol=2e-14)

    def test_wilkinson_growth_reports_actual_final_backward_error(self):
        for n in (30, 50, 60, 80):
            with self.subTest(n=n):
                values = np.eye(n) - np.tril(np.ones((n, n)), -1)
                values[:, -1] = 1.0
                matrix = csc_array(values)
                expected = np.random.default_rng(47).normal(size=n)
                rhs = matrix @ expected
                factor = splu(
                    matrix, backend="native", permc_spec="NATURAL", num_threads=1
                )
                with catch_warnings(record=True) as warnings:
                    simplefilter("always", RuntimeWarning)
                    result = factor.solve(rhs)
                denominator = abs(matrix) @ np.abs(result) + np.abs(rhs)
                error = np.max(np.abs(rhs - matrix @ result) / denominator)
                self.assertTrue(np.isfinite(error))
                above_target = error > 10.0 * np.finfo(np.float64).eps
                self.assertEqual(len(warnings), int(above_target))
                if above_target:
                    self.assertTrue(issubclass(warnings[0].category, RuntimeWarning))
                    self.assertIn(
                        "iterative refinement did not converge", str(warnings[0].message)
                    )
                assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_disabled_refinement_skips_error_checks_and_corrections(self):
        with (
            patch.object(
                self.factor, "_solve_once", side_effect=self.inaccurate_solves(())
            ) as solve,
            catch_warnings(record=True) as warnings,
        ):
            simplefilter("always", RuntimeWarning)
            result = self.factor.solve(
                self.matrix @ self.expected, iterative_refinement=False
            )
        self.assertEqual(warnings, [])
        self.assertEqual(solve.call_count, 1)
        assert_allclose(result, self.expected + 0.5, rtol=2e-14, atol=2e-14)


if __name__ == "__main__":
    unittest.main()
