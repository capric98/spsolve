from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.sparse._sparsetools import csr_matvecs

def _matmul_csr(A: csr_matrix, b: ndarray, ans: ndarray):
    M, N = A._shape_as_2d # type: ignore
    nrhs = b.shape[-1]

    csr_matvecs(M, N, nrhs, A.indptr, A.indices, A.data,
                b.ravel(), ans.ravel()) # type: ignore