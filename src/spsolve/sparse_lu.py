from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import SparseEfficiencyWarning, csc_array, csr_array, issparse, sparray
from scipy.sparse.linalg import SuperLU as ScipySuperLU
from scipy.sparse.linalg import splu as scipy_splu

from ._spsolve import _NativeSparseLU, _splu_factorize  # type: ignore
from .spparms import get_max_threads

Backend = Literal["auto", "native", "scipy"]
PermcSpec = Literal["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]

# Promotion is deliberately benchmark-gated. Set to True only after the
# reproducible gate in benchmarks/factorization.py passes.
NATIVE_DEFAULT = False


def _thread_count(value: int | None) -> int:
    threads = get_max_threads() if value is None else int(value)
    if threads < 1:
        raise ValueError("num_threads must be at least one")
    return threads


def _canonical_csc(A: sparray, *, dtype: np.dtype[Any] | None = None) -> csc_array:
    if not issparse(A):
        raise TypeError(f"expected a SciPy sparse matrix or array, got {type(A)!r}")
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square sparse matrix, got shape={A.shape}")

    if getattr(A, "format", None) != "csc":
        warn(
            "native sparse LU uses CSC input; converting a copy to CSC",
            SparseEfficiencyWarning,
            stacklevel=3,
        )
    converted = csc_array(A, dtype=dtype, copy=True)
    converted.sum_duplicates()
    converted.eliminate_zeros()
    converted.sort_indices()
    converted.data = np.ascontiguousarray(converted.data)
    if (
        converted.indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or converted.indices.dtype != converted.indptr.dtype
    ):
        converted.indices = np.ascontiguousarray(converted.indices, dtype=np.int64)
        converted.indptr = np.ascontiguousarray(converted.indptr, dtype=np.int64)
    converted.indices = np.ascontiguousarray(converted.indices)
    converted.indptr = np.ascontiguousarray(converted.indptr)
    return converted


def _validate_backend(backend: str) -> Backend:
    normalized = backend.lower()
    if normalized not in {"auto", "native", "scipy"}:
        raise ValueError("backend must be 'auto', 'native', or 'scipy'")
    return normalized  # type: ignore[return-value]


def _copy_solution_back(original: Any, solution: NDArray[np.float64], overwrite: bool):
    if (
        overwrite
        and isinstance(original, np.ndarray)
        and original.dtype == np.float64
        and original.flags.writeable
        and original.shape == solution.shape
    ):
        np.copyto(original, solution)
        return original
    return solution


class SparseLU:
    """Reusable native sparse LU factorization.

    The stored factors satisfy ``Pr @ Dr @ A @ Dc @ Pc == L @ U``. The
    permutation arrays map factor row/column positions to original indices.
    Scaling and permutations are applied automatically by :meth:`solve`.
    """

    def __init__(self, native: _NativeSparseLU, A: csc_array):
        self._native = native
        self._A = A
        self._L: csc_array | None = None
        self._U: csc_array | None = None
        self._perm_r: NDArray[np.int64] | None = None
        self._perm_c: NDArray[np.int64] | None = None
        self._row_scale: NDArray[np.float64] | None = None
        self._col_scale: NDArray[np.float64] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._A.shape

    @property
    def dtype(self) -> np.dtype[np.float64]:
        return np.dtype(np.float64)

    @property
    def nnz(self) -> int:
        return int(self._native.nnz)

    @property
    def perm_r(self) -> NDArray[np.int64]:
        if self._perm_r is None:
            self._perm_r = np.asarray(self._native._perm_r(), dtype=np.int64)
            self._perm_r.flags.writeable = False
        return self._perm_r

    @property
    def perm_c(self) -> NDArray[np.int64]:
        if self._perm_c is None:
            self._perm_c = np.asarray(self._native._perm_c(), dtype=np.int64)
            self._perm_c.flags.writeable = False
        return self._perm_c

    @property
    def row_scale(self) -> NDArray[np.float64]:
        if self._row_scale is None:
            self._row_scale = np.asarray(self._native._row_scale(), dtype=np.float64)
            self._row_scale.flags.writeable = False
        return self._row_scale

    @property
    def col_scale(self) -> NDArray[np.float64]:
        if self._col_scale is None:
            self._col_scale = np.asarray(self._native._col_scale(), dtype=np.float64)
            self._col_scale.flags.writeable = False
        return self._col_scale

    @property
    def L(self) -> csc_array:
        if self._L is None:
            data, indices, indptr = self._native._export_l_csr()
            factor = csr_array(
                (
                    np.asarray(data, dtype=np.float64),
                    np.asarray(indices, dtype=np.int64),
                    np.asarray(indptr, dtype=np.int64),
                ),
                shape=self.shape,
            )
            self._L = factor.tocsc()
        return self._L

    @property
    def U(self) -> csc_array:
        if self._U is None:
            data, indices, indptr = self._native._export_u_csr()
            factor = csr_array(
                (
                    np.asarray(data, dtype=np.float64),
                    np.asarray(indices, dtype=np.int64),
                    np.asarray(indptr, dtype=np.int64),
                ),
                shape=self.shape,
            )
            self._U = factor.tocsc()
        return self._U

    @property
    def stats(self) -> dict[str, Any]:
        result = dict(self._native._stats())
        result["backend"] = "native"
        result["shape"] = self.shape
        return result

    def _solve_once(
        self,
        rhs: NDArray[np.float64],
        *,
        trans: str,
        num_threads: int | None,
    ) -> NDArray[np.float64]:
        work = np.array(rhs, dtype=np.float64, order="C", copy=True, ndmin=2)
        if rhs.ndim == 1:
            work = work.reshape(self.shape[0], 1)
        self._native._solve_inplace(
            work,
            trans,
            _thread_count(num_threads),
        )
        return work

    def solve(
        self,
        b: NDArray[Any],
        trans: Literal["N", "T", "H"] = "N",
        overwrite_b: bool = False,
        *,
        num_threads: int | None = None,
        iterative_refinement: bool = True,
    ) -> NDArray[np.float64]:
        """Solve with the stored factors, preserving vector/matrix RHS shape.

        Refinement applies at most two corrections and checks the final
        componentwise backward error. Nonconvergence emits a RuntimeWarning;
        a correction that worsens or invalidates the error is discarded.
        """
        rhs = np.asarray(b)
        if rhs.ndim not in (1, 2):
            raise ValueError("b must be a one- or two-dimensional dense array")
        if rhs.shape[0] != self.shape[0]:
            raise ValueError(
                f"b has {rhs.shape[0]} rows, expected {self.shape[0]}"
            )
        if np.iscomplexobj(rhs):
            raise TypeError("native float64 LU does not support a complex RHS")
        if not np.issubdtype(rhs.dtype, np.number):
            raise TypeError(f"b must have a numeric dtype, got {rhs.dtype}")

        operation = trans.upper()
        if operation not in {"N", "T", "H"}:
            raise ValueError("trans must be 'N', 'T', or 'H'")
        vector_rhs = rhs.ndim == 1
        rhs_2d = np.asarray(rhs, dtype=np.float64)
        if vector_rhs:
            rhs_2d = rhs_2d.reshape(self.shape[0], 1)

        solution = self._solve_once(
            rhs_2d, trans=operation, num_threads=num_threads
        )

        if iterative_refinement and self.shape[0] and rhs_2d.shape[1]:
            matrix = self._A if operation == "N" else self._A.T
            abs_matrix = abs(matrix)
            target = 10.0 * np.finfo(np.float64).eps
            previous_solution = solution
            previous_error = np.inf
            converged = False
            failure_reason = "correction limit reached"
            # Inspect the initial solve and both corrected solutions.
            for correction_count in range(3):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    residual = rhs_2d - matrix @ solution
                    denominator = abs_matrix @ np.abs(solution) + np.abs(rhs_2d)
                    if (
                        np.all(np.isfinite(solution))
                        and np.all(np.isfinite(residual))
                        and np.all(np.isfinite(denominator))
                    ):
                        ratios = np.divide(
                            np.abs(residual),
                            denominator,
                            out=np.full_like(residual, np.inf),
                            where=denominator > 0,
                        )
                        # A zero denominator certifies only an exact zero residual.
                        ratios[(denominator == 0) & (residual == 0)] = 0.0
                        backward_error = float(np.max(ratios, initial=0.0))
                    else:
                        backward_error = np.inf
                if not np.isfinite(backward_error):
                    solution = previous_solution
                    failure_reason = "non-finite backward error"
                    break
                if backward_error <= target:
                    converged = True
                    break
                if backward_error >= previous_error:
                    solution = previous_solution
                    failure_reason = "backward error did not decrease"
                    break
                if correction_count == 2:
                    break
                previous_solution = solution
                previous_error = backward_error
                correction = self._solve_once(
                    np.asarray(residual), trans=operation, num_threads=num_threads
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    solution = solution + correction
            if not converged:
                warn(
                    "iterative refinement did not converge to the backward-error "
                    f"tolerance ({failure_reason})",
                    RuntimeWarning,
                    stacklevel=2,
                )

        shaped = solution[:, 0] if vector_rhs else solution
        return _copy_solution_back(b, shaped, overwrite_b)


def splu(
    A: sparray,
    permc_spec: PermcSpec | None = "COLAMD",
    diag_pivot_thresh: float | None = 1.0,
    *,
    num_threads: int | None = None,
    backend: Backend = "auto",
    check_finite: bool = True,
) -> SparseLU | ScipySuperLU:
    """Factorize a sparse square matrix using the native or SciPy backend."""
    selected = _validate_backend(backend)
    if not issparse(A):
        raise TypeError(f"expected a SciPy sparse matrix or array, got {type(A)!r}")

    spec = "COLAMD" if permc_spec is None else permc_spec.upper()
    threshold = 1.0 if diag_pivot_thresh is None else float(diag_pivot_thresh)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("diag_pivot_thresh must be between 0 and 1")
    native_supported = A.dtype == np.float64 and spec in {
        "COLAMD",
        "MMD_AT_PLUS_A",
        "NATURAL",
    }
    if selected == "auto":
        selected = "native" if NATIVE_DEFAULT and native_supported else "scipy"

    if selected == "scipy":
        matrix = _canonical_csc(A)
        if check_finite and not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix contains NaN or infinity")
        return scipy_splu(
            matrix,
            permc_spec=spec,
            diag_pivot_thresh=threshold,
        )

    if A.dtype != np.float64:
        raise TypeError(
            f"native sparse LU currently supports float64 A, got {A.dtype}"
        )
    if spec not in {"COLAMD", "MMD_AT_PLUS_A", "NATURAL"}:
        raise ValueError(
            "native permc_spec must be 'COLAMD', 'MMD_AT_PLUS_A', or 'NATURAL'"
        )
    matrix = _canonical_csc(A, dtype=np.dtype(np.float64))
    try:
        native = _splu_factorize(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            matrix.shape[0],
            spec,
            threshold,
            _thread_count(num_threads),
            bool(check_finite),
        )
    except RuntimeError as error:
        if "singular" in str(error).lower():
            raise np.linalg.LinAlgError(str(error)) from error
        raise
    return SparseLU(native, matrix)


def factorized(A: sparray, **factor_options: Any) -> Callable[[NDArray[Any]], NDArray[Any]]:
    """Return a solve callable holding a reusable factorization of ``A``."""
    factor = splu(A, **factor_options)
    return factor.solve
