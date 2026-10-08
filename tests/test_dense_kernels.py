import os
import platform
import subprocess
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array

from spsolve import splu
from spsolve.spparms import available_dense_kernels, get_dense_kernel, set_dense_kernel
from tests.test_blocked_lu import dense_pivoted, sparse_to_dense
from tests.test_supernodal_ldlt import overlapping_cliques


AVX2_AVAILABLE = "avx2-fma" in available_dense_kernels()


def spd_block(size):
    dense = 0.125 * np.ones((size, size))
    dense[np.diag_indices(size)] = 4.0
    return csc_array(dense)


def random_spd_block(size, seed=352):
    # Exactly symmetric with unstructured values; spd_block's repeated values
    # give products that a fused multiply-add rounds no differently.
    rng = np.random.default_rng(seed)
    factor = rng.uniform(-1.0, 1.0, (size, size))
    gram = factor @ factor.T / size
    return csc_array((gram + gram.T) / 2 + np.eye(size))


def normalized(name):
    """Map a forced or selected kernel name to the name stats report."""
    key = name.strip().lower()
    if key == "auto":
        return available_dense_kernels()[0]
    return "avx2-fma" if key == "avx2" else key


class DenseKernelTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_dense_kernel, None)

    def check_factor(self, matrix, factor, tolerance):
        scaled = csc_array(matrix.multiply(factor.row_scale[:, None])
                           .multiply(factor.col_scale[None, :]))
        error = scaled[factor.perm_r, :][:, factor.perm_c] - factor.L @ factor.U
        self.assertLess(np.max(abs(error.data), initial=0.0), 3e-13)
        expected = np.random.default_rng(529).normal(size=(matrix.shape[0], 2))
        for trans, operator in (("N", matrix), ("T", matrix.T)):
            assert_allclose(factor.solve(operator @ expected, trans=trans,
                                         iterative_refinement=False),
                            expected, rtol=tolerance, atol=tolerance)

    def assert_same_factors(self, first, second):
        for name in ("perm_r", "perm_c", "row_scale", "col_scale"):
            assert_array_equal(getattr(first, name), getattr(second, name))
        for name in ("L", "U"):
            for field in ("indptr", "indices", "data"):
                assert_array_equal(getattr(getattr(first, name), field),
                                   getattr(getattr(second, name), field))

    def assert_close_factors(self, factor, reference):
        # Same pivots and patterns; values may differ only by kernel rounding.
        for name in ("perm_r", "perm_c", "row_scale", "col_scale"):
            assert_array_equal(getattr(factor, name), getattr(reference, name))
        for name in ("L", "U"):
            first, second = getattr(factor, name), getattr(reference, name)
            assert_array_equal(first.indptr, second.indptr)
            assert_array_equal(first.indices, second.indices)
            scale = np.max(np.abs(second.data), initial=0.0)
            assert_allclose(first.data, second.data, rtol=1e-12, atol=1e-14 * scale)

    def test_selection_api(self):
        names = available_dense_kernels()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-1], "scalar")
        self.assertTrue(set(names) <= {"avx2-fma", "sse2", "scalar"})
        if platform.machine().lower() in {"amd64", "x86_64", "x64"}:
            self.assertIn("sse2", names)
        if "avx2-fma" in names:
            # The fused kernel is the best one whenever the CPU supports it.
            self.assertEqual(names[0], "avx2-fma")
        for name in names:
            set_dense_kernel(name.upper())
            self.assertEqual(get_dense_kernel(), name)
        set_dense_kernel("auto")
        self.assertEqual(get_dense_kernel(), names[0])
        if "avx2-fma" in names:
            set_dense_kernel("scalar")
            set_dense_kernel("avx2")
            self.assertEqual(get_dense_kernel(), "avx2-fma")
        set_dense_kernel("scalar")
        rejected = ["bogus", "", "sse2\0", "avx512"]
        if "avx2-fma" not in names:
            rejected += ["avx2", "AVX2-FMA"]
        for name in rejected:
            with self.subTest(name=name), self.assertRaises(ValueError):
                set_dense_kernel(name)
            # A rejected name leaves the selection unchanged.
            self.assertEqual(get_dense_kernel(), "scalar")
        with self.assertRaises(TypeError):
            set_dense_kernel(2)

    def test_forced_kernel_environment(self):
        forced = os.environ.get("SPSOLVE_FORCE_KERNEL")
        set_dense_kernel(None)
        if forced is not None and forced.strip().lower() != "auto":
            # An unusable forced value is ignored by the library; under a
            # forced run that is a configuration error, so fail loudly.
            self.assertIn(normalized(forced), available_dense_kernels(),
                          f"SPSOLVE_FORCE_KERNEL={forced!r} is not usable here")
            self.assertEqual(get_dense_kernel(), normalized(forced))
        elif forced is None:
            self.assertEqual(get_dense_kernel(), available_dense_kernels()[0])
        code = ("from spsolve.spparms import get_dense_kernel, set_dense_kernel; "
                "print(get_dense_kernel()); set_dense_kernel('scalar'); "
                "set_dense_kernel(None); print(get_dense_kernel())")
        requests = [name.upper() for name in available_dense_kernels()] + ["auto", "bogus"]
        if "avx2-fma" in available_dense_kernels():
            requests.append("avx2")
        for request in requests:
            with self.subTest(request=request):
                environment = dict(os.environ, SPSOLVE_FORCE_KERNEL=request)
                output = subprocess.run([sys.executable, "-c", code], env=environment,
                                        capture_output=True, text=True, check=True).stdout.split()
                expected = (available_dense_kernels()[0] if request == "bogus"
                            else normalized(request))
                self.assertEqual(output, [expected, expected])

    def test_kernels_agree_across_threads_and_tile_paths(self):
        cases = []
        for n in (158, 159, 160, 161, 162):
            # Panel and tile row/column remainders in the LU flush; the flush
            # target counts cover every row-group remainder modulo 4.
            cases.append((f"dense-{n}", dense_pivoted(n), "NATURAL", "row-hybrid", 3e-12))
        cases.append(("sparse-to-dense", sparse_to_dense(), "NATURAL", "row-hybrid", 3e-12))
        # Lower-triangular trailing and contiguous intersection updates.
        cases.append(("spd-352", spd_block(352), "NATURAL", "symmetric-supernodal-ldlt", 3e-12))
        # Partial, noncontiguous contributor intersections.
        cases.append(("cliques", overlapping_cliques()[0], "NATURAL",
                      "symmetric-supernodal-ldlt", 3e-10))
        names = available_dense_kernels()
        for label, matrix, order, engine, tolerance in cases:
            set_dense_kernel("scalar")
            reference = splu(matrix, backend="native", permc_spec=order, num_threads=1)
            for name in names:
                set_dense_kernel(name)
                with self.subTest(case=label, kernel=name):
                    serial = splu(matrix, backend="native", permc_spec=order, num_threads=1)
                    parallel = splu(matrix, backend="native", permc_spec=order, num_threads=4)
                    for factor in (serial, parallel):
                        self.assertEqual(factor.stats["dense_kernel"], name)
                        self.assertEqual(factor.stats["numeric_engine"], engine)
                        self.check_factor(matrix, factor, tolerance)
                    if engine == "row-hybrid":
                        self.assertGreater(serial.stats["lu_dense_panels"], 0)
                    else:
                        self.assertGreater(serial.stats["dense_update_flops"], 0)
                    self.assert_same_factors(serial, parallel)
                    self.assert_close_factors(serial, reference)

    @unittest.skipUnless(AVX2_AVAILABLE, "avx2-fma kernel not available on this build or CPU")
    def test_avx2_kernel_matches_scalar(self):
        cases = (
            # An odd size leaves row and column remainders for the 4x8
            # microkernel edges in the LU flush tiles.
            ("dense-225", dense_pivoted(225), "row-hybrid"),
            ("spd-352", random_spd_block(352), "symmetric-supernodal-ldlt"),
        )
        for label, matrix, engine in cases:
            set_dense_kernel("scalar")
            reference = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=1)
            set_dense_kernel("avx2-fma")
            factors = []
            for threads in (1, 4):
                with self.subTest(case=label, threads=threads):
                    factor = splu(matrix, backend="native", permc_spec="NATURAL",
                                  num_threads=threads)
                    self.assertEqual(factor.stats["dense_kernel"], "avx2-fma")
                    self.assertEqual(factor.stats["numeric_engine"], engine)
                    self.check_factor(matrix, factor, 3e-12)
                    if engine == "row-hybrid":
                        self.assertGreater(factor.stats["lu_dense_panels"], 0)
                        # The LU flush (running order) does not fuse: it must
                        # match the scalar pivot-tail materialization, so its
                        # factors equal the scalar kernel's bit for bit.
                        self.assert_same_factors(factor, reference)
                    else:
                        self.assertGreater(factor.stats["dense_update_flops"], 0)
                        self.assert_close_factors(factor, reference)
                    factors.append(factor)
            with self.subTest(case=label, check="threads"):
                self.assert_same_factors(factors[0], factors[1])
            if engine != "row-hybrid":
                with self.subTest(case=label, check="fused"):
                    # The LDL.T update rounds once per term, so some last bits
                    # differ, which shows that the fused kernel actually ran.
                    self.assertFalse(np.array_equal(factors[0].L.data, reference.L.data)
                                     and np.array_equal(factors[0].U.data, reference.U.data))
        # A duplicated dense row cancels exactly whichever path (flush or
        # pivot-tail materialization) applies each panel term.
        duplicated = dense_pivoted(193).toarray()
        duplicated[150, :] = duplicated[149, :]
        for threads in (1, 4):
            with self.subTest(case="duplicated-row", threads=threads):
                with self.assertRaisesRegex(np.linalg.LinAlgError, "singular"):
                    splu(csc_array(duplicated), backend="native", permc_spec="NATURAL",
                         num_threads=threads)

    def test_every_kernel_reports_nonfinite_updates(self):
        n = 1030
        growth = np.eye(n) - np.tril(np.ones((n, n)), -1)
        growth[:, -1] = 1.0
        matrix = csc_array(growth)
        for name in available_dense_kernels():
            set_dense_kernel(name)
            for threads in (1, 4):
                with self.subTest(kernel=name, threads=threads):
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        splu(matrix, backend="native", permc_spec="NATURAL",
                             num_threads=threads)

    def test_factor_keeps_its_kernel_snapshot(self):
        matrix = dense_pivoted(160)
        for name in available_dense_kernels():
            set_dense_kernel(name)
            factor = splu(matrix, backend="native", permc_spec="NATURAL", num_threads=2)
            set_dense_kernel("scalar")
            self.assertEqual(factor.stats["dense_kernel"], name)


if __name__ == "__main__":
    unittest.main()
