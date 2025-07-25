from warnings import warn

from numpy import arange, ones, ndarray
from scipy.sparse import issparse, sparray, csr_array, SparseEfficiencyWarning
from scipy.sparse.linalg import spbandwidth, is_sptriangular, splu, SuperLU

from .matmul import _matmul_csr
from .spsolve_triangular import spsolve_triangular
from .spsolve_pardiso import solve

from ._spsolve import is_built_with_mkl # type: ignore



def spsolve(A: sparray, b: ndarray, overwrite_b: bool=False, permc_spec: str="COLAMD", **kwargs) -> ndarray:

    if is_built_with_mkl():
        return solve(A, b, **kwargs)

    warn("experimental spsolve, for general usage try pypardiso", SparseEfficiencyWarning, stacklevel=2)

    # sanity check
    # non-square matrix can utilize QR solver, but I didn't find a sparse QR decomposition available,
    # let it throw exception currently...
    assert(issparse(A))
    A_shape = A.shape # type: ignore
    assert(A_shape[0] == A_shape[1])
    assert(A_shape[0] == b.shape[0])

    # lo_bandwidth, hi_bandwidth = spbandwidth(A)

    # if lo_bandwidth == hi_bandwidth == 0:
    #     # diagonal solver
    #     warn("diagonal solver is not implemented, use triangular solver instead", stacklevel=2)
    #     return spsolve_triangular(A, b, overwrite_b=overwrite_b, lower=True)

    # if lo_bandwidth == 0 or hi_bandwidth == 0:
    #     return spsolve_triangular(A, b, overwrite_b=overwrite_b, lower=(hi_bandwidth==0))

    is_lower_triangular, is_upper_triangular = is_sptriangular(A)
    if is_lower_triangular or is_upper_triangular:
        return spsolve_triangular(A, b, lower=bool(is_lower_triangular), overwrite_b=overwrite_b)

    # if is permuted triangular

    # if symmetric or hermitian: chol? LDL?

    # LU solver
    # t_start = time.perf_counter()
    lu: SuperLU = splu(A, permc_spec=permc_spec)
    # t_elapsed = time.perf_counter() - t_start
    # print(f"LU decomposition finished in {1000*t_elapsed:.2f} ms")

    L = csr_array(lu.L)
    U = csr_array(lu.U)

    Pr = csr_array((ones(A_shape[0]), (lu.perm_r, arange(A_shape[0]))))
    Pc = csr_array((ones(A_shape[0]), (arange(A_shape[0]), lu.perm_c)))

    # solve Ax=b via x = Pc @ ( U \ ( L \ (Pr@b) ) )
    ans = Pr @ b
    spsolve_triangular(L, ans, lower=True, overwrite_b=True, unit_diagonal=True) # LU decomposition should give unit_diagonal L?
    spsolve_triangular(U, ans, lower=False, overwrite_b=True)

    if overwrite_b:
        _matmul_csr(Pc, ans, b) # write Pc @ ans into b
        ans = b
    else:
        ans = Pc @ ans

    return ans


if __name__ == "__main__":
    pass