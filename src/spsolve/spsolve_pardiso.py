from warnings import warn

from numpy import empty_like, zeros, ndarray, int32, int64
from scipy.sparse import issparse, sparray, csr_array, SparseEfficiencyWarning

from .spparms import get_max_threads
from .assure_contiguous import assure_contiguous

from ._spsolve import is_built_with_mkl # type: ignore


def _black_hole(*args, **kwargs) -> int:
    return -99

_spsolve_pardiso = _black_hole

if is_built_with_mkl():
    from ._spsolve import spsolve_pardiso # type: ignore
    _spsolve_pardiso = spsolve_pardiso


def solve(A: sparray, b: ndarray) -> ndarray:
    if not issparse(A): raise ValueError(f"expect a scipy sparse array but got '{type(A)}'")
    if A.format != "csr": # type: ignore
        warn("CSR array format is required. Converting to CSR array.", SparseEfficiencyWarning, stacklevel=2)
        A = csr_array(A)
    if not A.has_sorted_indices: A.sort_indices() # type: ignore

    data:    ndarray = assure_contiguous(A.data)    # type: ignore
    indices: ndarray = assure_contiguous(A.indices) # type: ignore
    indptr:  ndarray = assure_contiguous(A.indptr)  # type: ignore

    # sanity check
    M, N = A.shape # type: ignore
    nrhs = b.shape[1]
    assert(M == N)
    assert(M == b.shape[0])

    if not b.data.f_contiguous:
        b = b.copy(order="F")

    ans = empty_like(b, order="F")

    mtype = 11
    iparm = zeros(64, dtype=int64) # nanobind will cast to MKL_INT

    iparm[0 ] = 1   # manually setting
    iparm[1 ] = 103 # OMP version of fill-in reduce, maximum level of optimization (L=10)
    iparm[5 ] = 0   # set to 1 if overwrite_b
    iparm[7 ] = 2   # default 2-steps refinement
    iparm[9 ] = 13  # set to 8 if symmetric
    iparm[10] = 1   # set to 0 if symmetric
    iparm[11] = 0   # Ax=b, no transpose
    iparm[12] = 1   # enable matching for nonsymmetric
    iparm[17] = 0   # disable nnz report
    iparm[20] = 1   # default pivoting
    iparm[23] = 0   # cannot use 10, idk why
    iparm[34] = 1   # zero-based indexing
    iparm[36] = 0   # CSR format

    status = _spsolve_pardiso(
        data, indices, indptr,
        b, ans, iparm,
        M, N, nrhs, mtype,
        get_max_threads(),
    )

    if status!=0:
        raise

    return ans


if __name__ == "__main__":
    pass