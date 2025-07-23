from warnings import warn

from numpy import iscomplexobj, ndarray, dtype, float64, complex128
from scipy.sparse import issparse, isspmatrix_csr, spmatrix, csr_matrix, SparseEfficiencyWarning

from .spparms import get_max_threads, get_prefer_order
from .assure_contiguous import assure_contiguous
from ._spsolve import spsolve_triangular as _spsolve_triangular # type: ignore


def spsolve_triangular(A: spmatrix, b: ndarray, lower: bool=True, overwrite_b: bool=False, overwrite_A: bool=False, unit_diagonal: bool=False) -> ndarray:
    if not issparse(A): raise ValueError(f"expect a scipy sparse matrix but got '{type(A)}'")
    if not isspmatrix_csr(A):
        warn("CSR matrix format is required. Converting to CSR matrix.", SparseEfficiencyWarning, stacklevel=2)
        A = csr_matrix(A)
    if not A.has_sorted_indices: A.sort_indices() # type: ignore


    data:    ndarray = assure_contiguous(A.data)    # type: ignore
    indices: ndarray = assure_contiguous(A.indices) # type: ignore
    indptr:  ndarray = assure_contiguous(A.indptr)  # type: ignore
    nnz:         int = A.size                       # type: ignore
    _PREFER_ORDER    = get_prefer_order()


    # sanity check
    assert(nnz == data.size)
    assert(nnz == indices.size)

    A_shape: tuple = A.shape # type: ignore
    assert(A_shape[0] == A_shape[1])
    assert(A_shape[0] == b.shape[0])
    if overwrite_A: warn("overwrite_A has no effect here", stacklevel=2)


    if not b.data.contiguous:
        b = b.copy(order=_PREFER_ORDER)
        overwrite_b = True


    if data.dtype != float64:
        if iscomplexobj(data):
            raise Exception("complex A is not supported currently")

        overwrite_b = True
        warn(f"explictly made a copy of A from dtype='{data.dtype}' to dtype='{dtype(float64)}'", stacklevel=2)
        data = data.astype(float64, copy=True, order=_PREFER_ORDER)


    flag_C128_as_F64 = False
    if (b_dtype:=b.dtype) != float64:
        if iscomplexobj(b):
            if iscomplexobj(data):
                raise Exception("complex A\\b is not supported currently")
            else:
                if b_dtype == complex128:
                    flag_C128_as_F64 = True
                    if b.data.f_contiguous:
                        b = b.copy(order=_PREFER_ORDER) # cannot use float64 logic to solve complex F contiguous b
                        overwrite_b = True
                    b = b.view(dtype=float64)
                else:
                    raise Exception(f"'{dtype(b_dtype)}' b is not supported currently")

        else:
            warn(f"'b' has dtype='{b.dtype}', explicitly make a copy of dtype='{dtype(float64)}'", stacklevel=2)
            overwrite_b = True
            b = b.astype(float64, copy=True, order=_PREFER_ORDER)


    ans: ndarray = b if overwrite_b else b.copy(order=_PREFER_ORDER) # b.copy(order="A")


    ## Will it run faster by force lower? Introduce an overhead of flip but may be friendlier to the cache...
    # if not lower:
    #     lower = True
    #     rows = flip(rows)
    #     cols = flip(cols)
    #     data = flip(data)

    # solve Ax=b in place where b is already copied into ans or overwrite_b is True
    _spsolve_triangular(data, indices, indptr, ans, lower, unit_diagonal, get_max_threads())


    if flag_C128_as_F64: ans = ans.view(dtype=b_dtype)

    return ans


if __name__ == "__main__":
    pass