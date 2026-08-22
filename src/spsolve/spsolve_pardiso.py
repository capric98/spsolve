from __future__ import annotations

from typing import Any
from warnings import warn


def solve(A: Any, b: Any, **kwargs: Any):
    """Deprecated compatibility alias for the native general solver."""
    warn(
        "spsolve_pardiso no longer uses MKL and is deprecated; use spsolve instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from .spsolve import spsolve

    kwargs.pop("backend", None)
    return spsolve(A, b, backend="native", **kwargs)
