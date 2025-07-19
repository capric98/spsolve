import os
import warnings

import psutil
import scipy
import numpy as np
import scipy.sparse

from . import _spsolve


_SPSOLVE_ORDER   = "C"
_CONTIG_FLAG_STR = f"{_SPSOLVE_ORDER.lower()}_contiguous"
_PS_CPU_AFFINITY = psutil.Process().cpu_affinity()
_PS_CPU_AFFINITY = [0] if not _PS_CPU_AFFINITY else _PS_CPU_AFFINITY
OMP_NUM_THREADS  = int(os.getenv("OMP_NUM_THREADS", len(_PS_CPU_AFFINITY)))


def spsolve_triangular(A, b: np.ndarray, lower: bool=True, overwrite_b: bool=False, overwrite_A: bool=False, unit_diagonal: bool=False) -> np.ndarray:
    # Make sure CSR to ensure memory contiguous
    if not scipy.sparse.isspmatrix_csr(A):
        warnings.warn("use CSR matrix for better performance")
        A = scipy.sparse.csr_matrix(A)

    if not A.has_sorted_indices:
        A.sort_indices()


    data:    np.ndarray = A.data
    indices: np.ndarray = A.indices
    indptr:  np.ndarray = A.indptr
    nnz:            int = A.size


    # sanity check
    assert(nnz == data.size)
    assert(nnz == indices.size)

    A_shape: tuple = A.shape # type: ignore
    assert(A_shape[0] == A_shape[1])
    assert(A_shape[0] == b.shape[0])

    if overwrite_A: warnings.warn("overwrite_A has no effect here", stacklevel=2)
    if unit_diagonal: warnings.warn("unit_diagonal has no effect here", stacklevel=2)


    if data.dtype != np.float64:
        if np.iscomplexobj(data):
            raise Exception("complex A is not supported currently")

        overwrite_b = True
        warnings.warn(f"explictly made a copy of A from dtype='{data.dtype}' to dtype='{np.dtype(np.float64)}'", stacklevel=2)
        data = data.astype(np.float64, copy=True, order=_SPSOLVE_ORDER)


    flag_C128_as_F64 = False
    if (b_dtype:=b.dtype) != np.float64:
        if np.iscomplexobj(b):
            if np.iscomplexobj(data):
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
    #     data = np.flip(data)

    # solve Ax=b in place where b is already copied into ans or overwrite_b is True
    _spsolve.spsolve_triangular(data, indices, indptr, ans, lower, OMP_NUM_THREADS)

    if flag_C128_as_F64: ans = ans.view(dtype=b_dtype)

    return ans


if __name__ == "__main__":
    pass