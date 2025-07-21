import time

from warnings import warn

from numpy import arange, ones, ndarray
from scipy.sparse import issparse, spmatrix, csr_matrix, SparseEfficiencyWarning
from scipy.sparse.linalg import spbandwidth, is_sptriangular, splu, SuperLU

from .spsolve_triangular import spsolve_triangular


def spsolve(A: spmatrix, b: ndarray, overwrite_b: bool=False, permc_spec: str="COLAMD", use_umfpack: bool=True) -> ndarray:
    warn("unfinished spsolve function", stacklevel=2)

    # sanity check
    # non-square matrix can utilize QR solver, but I didn't find a sparse QR decomposition available,
    # let it throw exception currently...
    assert(issparse(A))
    A_shape = A.shape
    assert(A_shape[0] == A_shape[1])
    assert(A_shape[0] == b.shape[0])

    # lo_bandwidth, hi_bandwidth = spbandwidth(A)

    # if lo_bandwidth == hi_bandwidth == 0:
    #     # diagonal solver
    #     warn("diagonal solver is not implemented, use triangular solver instead", stacklevel=2)
    #     return spsolve_triangular(A, b, overwrite_b=overwrite_b, lower=True)

    # if lo_bandwidth == 0 or hi_bandwidth == 0:
    #     return spsolve_triangular(A, b, overwrite_b=overwrite_b, lower=(hi_bandwidth==0))

    is_lo_tri, is_up_tri = is_sptriangular(A)
    if is_lo_tri or is_up_tri:
        return spsolve_triangular(A, b, lower=bool(is_lo_tri), overwrite_b=overwrite_b)

    # if is permuted triangular

    # if symmetric or hermitian: chol? LDL?

    # LU solver
    t_start = time.perf_counter()
    lu: SuperLU = splu(A, permc_spec=permc_spec)
    t_elapsed = time.perf_counter() - t_start
    print(f"LU decomposition finished in {1000*t_elapsed:.2f} ms")

    L = csr_matrix(lu.L)
    U = csr_matrix(lu.U)

    Pr = csr_matrix((ones(A_shape[0]), (lu.perm_r, arange(A_shape[0]))))
    Pc = csr_matrix((ones(A_shape[0]), (arange(A_shape[0]), lu.perm_c)))

    ans = Pr @ b
    spsolve_triangular(L, ans, lower=True, overwrite_b=True)
    spsolve_triangular(U, ans, lower=False, overwrite_b=True)

    return Pc @ ans


if __name__ == "__main__":
    pass