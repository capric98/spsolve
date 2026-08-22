from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import SparseEfficiencyWarning, csr_array, issparse

from ._spsolve import spsolve_triangular as _spsolve_triangular  # type: ignore
from .spparms import get_max_threads, get_prefer_order


def _prepare_csr(A: Any) -> csr_array:
    if getattr(A, "format", None) != "csr":
        warn(
            "CSR format is required; converting a copy to CSR",
            SparseEfficiencyWarning,
            stacklevel=3,
        )
    if np.issubdtype(A.dtype, np.complexfloating):
        raise TypeError("native triangular solve currently supports real A only")
    if not (np.issubdtype(A.dtype, np.number) or A.dtype == np.bool_):
        raise TypeError(f"A must have a numeric dtype, got {A.dtype}")

    is_csr = getattr(A, "format", None) == "csr"
    index_dtype_ok = is_csr and A.indices.dtype in (
        np.dtype(np.int32),
        np.dtype(np.int64),
    )
    needs_copy = (
        not is_csr
        or A.dtype != np.float64
        or not A.data.flags.c_contiguous
        or not A.indices.flags.c_contiguous
        or not A.indptr.flags.c_contiguous
        or not index_dtype_ok
        or A.indices.dtype != A.indptr.dtype
    )
    matrix = csr_array(A, dtype=np.float64, copy=needs_copy)
    if not matrix.data.flags.c_contiguous:
        matrix.data = np.ascontiguousarray(matrix.data)
    if (
        matrix.indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or matrix.indices.dtype != matrix.indptr.dtype
    ):
        matrix.indices = np.ascontiguousarray(matrix.indices, dtype=np.int64)
        matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int64)
    if not matrix.indices.flags.c_contiguous:
        matrix.indices = np.ascontiguousarray(matrix.indices)
    if not matrix.indptr.flags.c_contiguous:
        matrix.indptr = np.ascontiguousarray(matrix.indptr)
    return matrix


def _work_order(rhs: NDArray[Any]) -> str:
    if rhs.ndim == 2 and rhs.flags.f_contiguous and not rhs.flags.c_contiguous:
        return "F"
    return "F" if get_prefer_order() == "F" else "C"


def spsolve_triangular(
    A: Any,
    b: ArrayLike,
    lower: bool = True,
    overwrite_b: bool = False,
    overwrite_A: bool = False,
    unit_diagonal: bool = False,
    *,
    num_threads: int | None = None,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Solve a real sparse triangular system for one or many dense RHSs.

    Work is parallelized over independent RHS blocks. A one-dimensional RHS is
    accepted and the returned shape always matches ``b``. With
    ``unit_diagonal=True``, stored diagonal entries are ignored and missing
    diagonal entries are treated as one.
    """
    if not issparse(A):
        raise TypeError(f"expected a SciPy sparse matrix or array, got {type(A)!r}")
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square sparse matrix, got shape={A.shape}")

    rhs = np.asarray(b)
    if rhs.ndim not in (1, 2):
        raise ValueError("b must be a one- or two-dimensional dense array")
    if rhs.shape[0] != A.shape[0]:
        raise ValueError(f"b has {rhs.shape[0]} rows, expected {A.shape[0]}")
    if not (np.issubdtype(rhs.dtype, np.number) or rhs.dtype == np.bool_):
        raise TypeError(f"b must have a numeric dtype, got {rhs.dtype}")

    # The native kernel never changes A; overwrite_A remains an accepted SciPy
    # compatibility argument. A compatible CSR is borrowed read-only, while
    # conversion or dtype/layout repair uses a private copy.
    del overwrite_A
    matrix = _prepare_csr(A)
    vector_rhs = rhs.ndim == 1
    rhs_columns = 1 if vector_rhs else rhs.shape[1]
    complex_rhs = np.iscomplexobj(rhs)
    result_dtype = np.dtype(np.complex128 if complex_rhs else np.float64)
    threads = get_max_threads() if num_threads is None else int(num_threads)
    if threads < 1:
        raise ValueError("num_threads must be at least one")

    can_overwrite = (
        overwrite_b
        and isinstance(b, np.ndarray)
        and b.dtype == result_dtype
        and b.flags.writeable
        and (b.flags.c_contiguous or b.flags.f_contiguous)
    )
    # A multi-column Fortran complex array cannot be reinterpreted as a
    # two-dimensional float64 RHS without copying.
    if (
        complex_rhs
        and rhs.ndim == 2
        and rhs.flags.f_contiguous
        and not rhs.flags.c_contiguous
    ):
        can_overwrite = False

    if can_overwrite:
        work = b
    else:
        order = "C" if complex_rhs else _work_order(rhs)
        work = np.array(rhs, dtype=result_dtype, order=order, copy=True)
    work_2d = work.reshape(A.shape[0], rhs_columns) if vector_rhs else work

    if complex_rhs:
        if not work_2d.flags.c_contiguous:
            work_2d = np.array(work_2d, dtype=np.complex128, order="C", copy=True)
            can_overwrite = False
        native_rhs = work_2d.view(np.float64).reshape(A.shape[0], 2 * rhs_columns)
    else:
        native_rhs = work_2d

    try:
        _spsolve_triangular(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            native_rhs,
            bool(lower),
            bool(unit_diagonal),
            threads,
        )
    except RuntimeError as error:
        if "singular" in str(error).lower():
            raise np.linalg.LinAlgError(str(error)) from error
        raise

    result = work_2d[:, 0] if vector_rhs else work_2d
    if can_overwrite:
        return b
    return result
