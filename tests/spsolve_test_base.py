import numpy as np
import scipy
import scipy.sparse


__REPEAT__ = 10;


def reset_rng():
    np.random.seed(0)


def random_A(size, density: float = 0.01, lower: bool=True, sp_type: str="csr") -> scipy.sparse.sparray:
    # diagonal dominant
    A = scipy.sparse.spdiags(5+np.random.rand(size), 0, size, size) + scipy.sparse.random(size, size, density)

    A = scipy.sparse.tril(A) if lower else scipy.sparse.triu(A)
    A = scipy.sparse.csr_array(A) if sp_type=="csr" else scipy.sparse.csc_array(A)

    return A

def random_b(M, N) -> np.ndarray:
    return np.random.rand(M, N)

def allclose(a, b) -> bool:
    return np.allclose(a, b)


if __name__ == "__main__":
    pass