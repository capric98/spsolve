import numpy as np
import scipy
import scipy.sparse

from spsolve import spsolve_triangular
from .spsolve_test_base import allclose, random_A, random_b, reset_rng, __REPEAT__


def test_unit_diagonal():
    reset_rng()
    for size in [10, 100, 900]:
        for density in [0.001, 0.01, 0.1, 1]:
            for lower in [True, False]:
                for nrhs in [1, 2, 3, 4, 6, 8, 32, 100]:
                    for _ in range(__REPEAT__):
                        print(f"testing size={size}, density={density}, lower={lower}, nrhs={nrhs}", end="...")
                        A: scipy.sparse.csr_matrix = random_A(size, density, lower) # type: ignore
                        A.data = A.data / size # make sure it's still diagonal dominant

                        for k in A.indptr:
                            if lower:
                                if k!=0: A.data[k-1] = 1
                            else:
                                if k!=A.indptr[-1]: A.data[k] = 1

                        assert(np.all(A.diagonal() == 1))

                        x = random_b(size, nrhs)
                        b = A @ x

                        ans = spsolve_triangular(A, b, lower, overwrite_b=True, unit_diagonal=True)
                        assert(allclose(x, ans))
                        print("pass")


if __name__ == "__main__":
    pass