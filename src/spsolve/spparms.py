from __future__ import annotations

import os
from typing import Literal

from ._spsolve import (  # type: ignore
    _available_dense_kernels,
    _dense_kernel,
    _set_dense_kernel,
)


def _initial_thread_count() -> int:
    configured = os.getenv("OMP_NUM_THREADS")
    if configured is not None:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


_PREFER_ORDER: Literal["C", "F", "A", "K"] = "C"
_OMP_NUM_THREADS = _initial_thread_count()


def set_num_threads(n: int) -> None:
    global _OMP_NUM_THREADS
    value = int(n)
    if value < 1:
        raise ValueError("thread count must be at least one")
    _OMP_NUM_THREADS = value


def get_max_threads() -> int:
    return _OMP_NUM_THREADS


def get_dense_kernel() -> str:
    """Return the dense tile kernel that new native factorizations use."""
    return str(_dense_kernel())


def available_dense_kernels() -> list[str]:
    """Return the dense kernels this build and CPU support, best first."""
    return [str(name) for name in _available_dense_kernels()]


def set_dense_kernel(name: str | None) -> None:
    """Select the dense tile kernel for subsequent native factorizations.

    ``name`` is ``"auto"`` (the best available kernel), ``"avx2"`` or
    ``"avx2-fma"``, ``"sse2"``, or ``"scalar"``, case-insensitively. ``None``
    restores the startup selection: ``SPSOLVE_FORCE_KERNEL`` when it names a
    usable kernel, otherwise ``"auto"``. Unknown or unsupported names raise
    ``ValueError``.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("dense kernel name must be a string or None")
    _set_dense_kernel(name)


def get_prefer_order() -> Literal["C", "F", "A", "K"]:
    return _PREFER_ORDER


def set_prefer_order(order: Literal["C", "F", "A", "K"]) -> None:
    if order not in {"C", "F", "A", "K"}:
        raise ValueError("order must be 'C', 'F', 'A', or 'K'")
    global _PREFER_ORDER
    _PREFER_ORDER = order
