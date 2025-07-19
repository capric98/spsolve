import os

from warnings import warn

import psutil

from numpy import iscomplexobj, ndarray, dtype, float64, complex128
from scipy.sparse import isspmatrix_csr, csr_matrix, SparseEfficiencyWarning

from . import _spsolve


_SPSOLVE_ORDER   = "C"
_CONTIG_FLAG_STR = f"{_SPSOLVE_ORDER.lower()}_contiguous"
_PS_CPU_AFFINITY = psutil.Process().cpu_affinity()
_PS_CPU_AFFINITY = [0] if not _PS_CPU_AFFINITY else _PS_CPU_AFFINITY
OMP_NUM_THREADS  = int(os.getenv("OMP_NUM_THREADS", len(_PS_CPU_AFFINITY)))


def spsolve_triangular(A, b: ndarray, lower: bool=True, overwrite_b: bool=False, overwrite_A: bool=False, unit_diagonal: bool=False) -> ndarray:
    # Make sure CSR to ensure memory contiguous
    if not isspmatrix_csr(A):
        warn("CSR matrix format is required. Converting to CSR matrix.", SparseEfficiencyWarning, stacklevel=2)
        A = csr_matrix(A)

    if not A.has_sorted_indices:
        A.sort_indices()


    data:    ndarray = A.data
    indices: ndarray = A.indices
    indptr:  ndarray = A.indptr
    nnz:         int = A.size


    # sanity check
    assert(nnz == data.size)
    assert(nnz == indices.size)

    A_shape: tuple = A.shape # type: ignore
    assert(A_shape[0] == A_shape[1])
    assert(A_shape[0] == b.shape[0])

    if overwrite_A: warn("overwrite_A has no effect here", stacklevel=2)
    if unit_diagonal: warn("unit_diagonal has no effect here", stacklevel=2)


    if data.dtype != float64:
        if iscomplexobj(data):
            raise Exception("complex A is not supported currently")

        overwrite_b = True
        warn(f"explictly made a copy of A from dtype='{data.dtype}' to dtype='{dtype(float64)}'", stacklevel=2)
        data = data.astype(float64, copy=True, order=_SPSOLVE_ORDER)


    flag_C128_as_F64 = False
    if (b_dtype:=b.dtype) != float64:
        if iscomplexobj(b):
            if iscomplexobj(data):
                raise Exception("complex A\\b is not supported currently")
            else:
                if b_dtype == complex128:
                    flag_C128_as_F64 = True
                    b = b.view(dtype=float64)
                else:
                    raise Exception(f"'{dtype(b_dtype)}' b is not supported currently")

        else:
            warn(f"'b' has dtype='{b.dtype}', explicitly make a copy of dtype='{dtype(float64)}'", stacklevel=2)
            overwrite_b = True
            b = b.astype(float64, copy=True, order=_SPSOLVE_ORDER)


    ans: ndarray = b if overwrite_b else b.copy(order=_SPSOLVE_ORDER)

    if overwrite_b:
        if not getattr(ans.data, _CONTIG_FLAG_STR):
            warn(f"overwrite_b in enabled but 'b' is not {_SPSOLVE_ORDER} CONTIGUOUS", stacklevel=2)
            ans = b.copy(order=_SPSOLVE_ORDER)
    else:
        ans = b.copy(order=_SPSOLVE_ORDER)

    ## Will it run faster by force lower? Introduce an overhead of flip but may be friendlier to the cache...
    # if not lower:
    #     lower = True
    #     rows = flip(rows)
    #     cols = flip(cols)
    #     data = flip(data)

    # solve Ax=b in place where b is already copied into ans or overwrite_b is True
    _spsolve.spsolve_triangular(data, indices, indptr, ans, lower, OMP_NUM_THREADS)

    if flag_C128_as_F64: ans = ans.view(dtype=b_dtype)

    return ans


if __name__ == "__main__":
    pass