import os
import warnings

import psutil
import scipy
import numpy as np

from . import _spsolve


_SPSOLVE_ORDER   = "C"
_CONTIG_FLAG_STR = f"{_SPSOLVE_ORDER.lower()}_contiguous"
_PS_CPU_AFFINITY = psutil.Process().cpu_affinity()
_PS_CPU_AFFINITY = [0] if not _PS_CPU_AFFINITY else _PS_CPU_AFFINITY
OMP_NUM_THREADS  = int(os.getenv("OMP_NUM_THREADS", len(_PS_CPU_AFFINITY)))


def spsolve_triangular(A, b: np.ndarray, lower: bool=True, overwrite_b: bool=False, overwrite_A: bool=False, unit_diagonal: bool=False) -> np.ndarray:
    if not scipy.sparse.isspmatrix_coo(A):
        # convert to coo_matrix to guarantee the ascending order of coords
        A = scipy.sparse.coo_matrix(A)

    rows, cols = A.coords
    vals: np.ndarray = A.data

    # sanity check
    A_shape = A.shape
    assert(A_shape[0] == A_shape[1])
    assert(A_shape[0] == b.shape[0])
    if overwrite_A: warnings.warn("overwrite_A has no effect here", stacklevel=2)
    if unit_diagonal: warnings.warn("unit_diagonal has no effect here", stacklevel=2)


    if vals.dtype != np.float64:
        if np.iscomplexobj(vals):
            raise Exception("complex A is not supported currently")

        overwrite_b = True
        warnings.warn(f"explictly made a copy of A from dtype='{vals.dtype}' to dtype='{np.dtype(np.float64)}'", stacklevel=2)
        vals = vals.astype(np.float64, copy=True, order=_SPSOLVE_ORDER)


    flag_C128_as_F64 = False
    if (b_dtype:=b.dtype) != np.float64:
        if np.iscomplexobj(b):
            if np.iscomplexobj(vals):
                raise Exception("complex A\\b is not supported currently")
            else:
                if b_dtype == np.complex128:
                    flag_C128_as_F64 = True
                    b = b.view(dtype=np.float64)
                else:
                    raise Exception(f"'{np.dtype(b_dtype)}' b is not supported currently")

        else:
            warnings.warn(f"'b' has dtype='{b.dtype}', explicitly make a copy of dtype='{np.dtype(np.float64)}'", stacklevel=2)
            overwrite_b = True
            b = b.astype(np.float64, copy=True, order=_SPSOLVE_ORDER)


    ans: np.ndarray = b if overwrite_b else b.copy(order=_SPSOLVE_ORDER)

    if overwrite_b:
        if not getattr(ans.data, _CONTIG_FLAG_STR):
            warnings.warn(f"overwrite_b in enabled but 'b' is not {_SPSOLVE_ORDER} CONTIGUOUS", stacklevel=2)
            ans = b.copy(order=_SPSOLVE_ORDER)
    else:
        ans = b.copy(order=_SPSOLVE_ORDER)

    ## Will it run faster by force lower? Introduce an overhead of flip but may be friendlier to the cache...
    # if not lower:
    #     lower = True
    #     rows = np.flip(rows)
    #     cols = np.flip(cols)
    #     vals = np.flip(vals)

    # solve Ax=b in place where b is already copied into ans or overwrite_b is True
    _spsolve.spsolve_triangular(rows, cols, vals, ans, A.nnz, lower, OMP_NUM_THREADS)

    if flag_C128_as_F64: ans = ans.view(dtype=b_dtype)

    return ans


if __name__ == "__main__":
    pass