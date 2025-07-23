_PREFER_ORDER    = "C"
_OMP_NUM_THREADS = 0 # use auto detect in C++ by default


try:
    import os
    import psutil
    _PS_CPU_AFFINITY = psutil.Process().cpu_affinity()
    _PS_CPU_AFFINITY = [0] if not _PS_CPU_AFFINITY else _PS_CPU_AFFINITY
    _OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", len(_PS_CPU_AFFINITY)))
except Exception as _:
    pass


def set_num_threads(n: int):
    global _OMP_NUM_THREADS
    _OMP_NUM_THREADS = int(n)

def get_max_threads() -> int:
    return _OMP_NUM_THREADS