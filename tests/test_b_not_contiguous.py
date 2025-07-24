from scipy import sparse
from numpy import empty

from spsolve import spsolve_triangular
from .spsolve_test_base import allclose, random_A, random_b, reset_rng, __REPEAT__


def test_b_not_contiguous_c():
    reset_rng()

    size = 1000
    density = 0.01
    nrhs = 1000
    factor = 3

    for _ in range(__REPEAT__):
        for lower in [True, False]:
            A: sparse.csr_array = random_A(size, density, lower) # type: ignore
            x = random_b(size, factor*nrhs)
            b = A @ x
            x = x[:, nrhs:(2*nrhs)]
            b = b[:, nrhs:(2*nrhs)]

            assert(not(x.data.contiguous))
            assert(not(b.data.contiguous))
            assert(allclose(A@x, b))

            ans = spsolve_triangular(A, b, lower, overwrite_b=True)
            assert(not(b.data.contiguous))
            assert(ans.data.c_contiguous)
            assert(allclose(ans, x))


def test_b_not_contiguous_f():
    reset_rng()

    size = 1000
    density = 0.01
    nrhs = 1000
    factor = 3

    for _ in range(__REPEAT__):
        for lower in [True, False]:
            A: sparse.csr_array = random_A(size, density, lower) # type: ignore
            x = random_b(factor*size, nrhs).copy(order="F")
            b = empty(x.shape, order="F")

            for k in range(factor):
                b[k*size:(k+1)*size, :] = A @ x[k*size:(k+1)*size, :]

            print(x.shape)

            x = x[size:2*size, :]
            b = b[size:2*size, :]

            cb = b.copy(order="F")

            assert(not(x.data.contiguous))
            assert(not(b.data.contiguous))
            assert(allclose(A@x, b))

            ans = spsolve_triangular(A, b, lower, overwrite_b=True)
            assert(not(b.data.contiguous))
            assert(ans.data.c_contiguous)
            assert(allclose(ans, x))
            assert(allclose(b, cb))


if __name__ == "__main__":
    pass