from spsolve import spsolve_triangular
from .spsolve_test_base import allclose, random_A, random_b, reset_rng, __REPEAT__

def test_c_contiguous_inplace():
    reset_rng()
    for size in [10, 100, 1000]:
        for density in [0.001, 0.01, 0.1, 1]:
            for lower in [True, False]:
                for nrhs in [1, 2, 3, 4, 6, 8, 32, 100]:
                    for _ in range(__REPEAT__):
                        A = random_A(size, density, lower)
                        x = random_b(size, nrhs)
                        b = A @ x

                        spsolve_triangular(A, b, lower, overwrite_b=True)
                        assert(allclose(x, b))


if __name__ == "__main__":
    pass