import os
import psutil

_SPSOLVE_ORDER   = "C"
_CONTIG_FLAG_STR = f"{_SPSOLVE_ORDER.lower()}_contiguous"
_PS_CPU_AFFINITY = psutil.Process().cpu_affinity()
_PS_CPU_AFFINITY = [0] if not _PS_CPU_AFFINITY else _PS_CPU_AFFINITY
OMP_NUM_THREADS  = int(os.getenv("OMP_NUM_THREADS", len(_PS_CPU_AFFINITY)))