from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse, sparray
from scipy.sparse.linalg import is_sptriangular

from .sparse_lu import Backend, PermcSpec, SparseLU, _validate_backend, splu
from .spsolve_triangular import spsolve_triangular


def spsolve(
    A: sparray,
    b: NDArray[Any],
    overwrite_b: bool = False,
    permc_spec: PermcSpec | None = "COLAMD",
    *,
    num_threads: int | None = None,
    backend: Backend = "auto",
    check_finite: bool = True,
    iterative_refinement: bool = True,
) -> NDArray[Any]:
    """Solve ``A @ x = b`` with a sparse LU factorization."""
    if not issparse(A):
        raise TypeError(f"expected a SciPy sparse matrix or array, got {type(A)!r}")
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square sparse matrix, got shape={A.shape}")

    rhs = np.asarray(b)
    if rhs.ndim not in (1, 2):
        raise ValueError("b must be a one- or two-dimensional dense array")
    if rhs.shape[0] != A.shape[0]:
        raise ValueError(f"b has {rhs.shape[0]} rows, expected {A.shape[0]}")

    selected_backend = _validate_backend(backend)

    # Preserve the established fast path for triangular float64 systems.
    if selected_backend != "scipy" and A.dtype == np.float64:
        is_lower, is_upper = is_sptriangular(A)
        if is_lower or is_upper:
            return spsolve_triangular(
                A,
                rhs,
                lower=bool(is_lower),
                overwrite_b=overwrite_b,
                num_threads=num_threads,
            )

    factor = splu(
        A,
        permc_spec=permc_spec,
        num_threads=num_threads,
        backend=selected_backend,
        check_finite=check_finite,
    )
    if isinstance(factor, SparseLU):
        return factor.solve(
            rhs,
            overwrite_b=overwrite_b,
            num_threads=num_threads,
            iterative_refinement=iterative_refinement,
        )

    solution = np.asarray(factor.solve(rhs))
    if (
        overwrite_b
        and isinstance(b, np.ndarray)
        and b.flags.writeable
        and b.shape == solution.shape
        and b.dtype == solution.dtype
    ):
        np.copyto(b, solution)
        return b
    return solution
