import os
from typing import Literal


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


def get_prefer_order() -> Literal["C", "F", "A", "K"]:
    return _PREFER_ORDER


def set_prefer_order(order: Literal["C", "F", "A", "K"]) -> None:
    if order not in {"C", "F", "A", "K"}:
        raise ValueError("order must be 'C', 'F', 'A', or 'K'")
    global _PREFER_ORDER
    _PREFER_ORDER = order
