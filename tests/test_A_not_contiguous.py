from scipy import sparse

from spsolve import spsolve_triangular
from .spsolve_test_base import allclose, random_A, random_b, reset_rng, __REPEAT__


def test_A_not_contiguous():
    reset_rng()

    size = 1000
    density = 0.01
    nrhs = 10
    factor = 3

    for _ in range(__REPEAT__):
        for lower in [True, False]:
            A: sparse.csr_array = random_A(size, density, lower) # type: ignore
            x = random_b(size, nrhs)

            big_data = random_b(A.data.size, factor)
            nc_data  = big_data[:, 1]

            for k in A.indptr:
                if lower:
                    if k!=0: nc_data[k-1] += 5
                else:
                    if k!=A.indptr[-1]: nc_data[k] += 5

            assert(nc_data.size == A.data.size)

            assert(not(nc_data.data.contiguous))
            A.data = nc_data
            cA = A.copy()
            assert(cA.data.data.contiguous)
            assert(not(A.data.data.contiguous))

            # print(cA.toarray())

            b = A @ x
            assert(allclose(cA@x, b))

            ans = spsolve_triangular(A, b, overwrite_b=True, lower=lower)
            print("error = ", (x-ans).max())
            assert(allclose(x, ans))


if __name__ == "__main__":
    pass