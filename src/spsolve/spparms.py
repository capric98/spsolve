import os

from typing import Literal

try:
    import psutil
    _PS_CPU_AFFINITY = psutil.Process().cpu_affinity()
    _PS_CPU_AFFINITY = [0] if not _PS_CPU_AFFINITY else _PS_CPU_AFFINITY
    _PS_CPU_AFF_NUMS = len(_PS_CPU_AFFINITY)
except Exception as _:
    _PS_CPU_AFF_NUMS = 0


_PREFER_ORDER    = 'C'
_OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", _PS_CPU_AFF_NUMS))


def set_num_threads(n: int):
    global _OMP_NUM_THREADS
    _OMP_NUM_THREADS = int(n)

def get_max_threads() -> int:
    return _OMP_NUM_THREADS

def get_prefer_order() -> Literal['C', 'F', 'A', 'K']:
    return _PREFER_ORDER

def set_prefer_order(order: Literal['C', 'F', 'A', 'K']):
    global _PREFER_ORDER
    _PREFER_ORDER = order