import time

import scipy
import numpy as np

from spsolve import spsolve_triangular
from scipy.sparse.linalg import spsolve_triangular as scipy_spsolve_triangular


if __name__ == "__main__":

    print(time.strftime("Current time: %H:%M:%S", time.localtime()))

    unit = "ms"
    tscale = 1e3
    n_ave = 1000

    size = 10000
    nrhs = 1000
    density = 0.1

    np.random.seed(0)

    LUA = scipy.sparse.spdiags(5+np.random.rand(size), 0, size, size) + scipy.sparse.random(size, size, density)

    for lower in [False, True]:
        print(f"==========lower={lower}==========")
        # A: size * size (sparse)
        # b: size * nrhs
        A = scipy.sparse.csr_matrix(scipy.sparse.tril(LUA) if lower else scipy.sparse.triu(LUA))
        x = np.random.randn(size, nrhs) #+ 1j*np.random.randn(size, nrhs)
        b = A @ x


        usr_ans = spsolve_triangular(A, b, lower=lower)
        diff    = abs(usr_ans - x)

        # usr_ans = usr_ans.copy(order="F") # test mismatch of order

        if not np.allclose(usr_ans, x):
            print("AssertionError: usr_ans != x")
            dnnz = np.count_nonzero(diff)
            print(f" nnz = {dnnz}, ratio = {100*dnnz/(size*nrhs):.2f}%")
            print(f" max = {np.max(diff):.2e}")
            if size * nrhs < 100:
                print("correct answer:")
                print(x)
                print("user function got:")
                print(usr_ans)
            continue
        else:
            print(f"ans == x, max_abs_err = {np.max(diff):.2e}")



        def tfunc():
            np.copyto(usr_ans, b)
            spsolve_triangular(A, usr_ans, overwrite_b=True, lower=lower)

        def tfunc_scipy():
            np.copyto(usr_ans, b)
            scipy_spsolve_triangular(A, usr_ans, overwrite_b=True, lower=lower)


        # t_start = time.perf_counter()
        # for _ in range(n_ave): tfunc_scipy()
        # t_elapsed = time.perf_counter() - t_start

        # print(f"single call of scipy: {tscale*t_elapsed/n_ave:.6f} {unit} (average of {n_ave} calls)")

        t_start = time.perf_counter()
        for _ in range(n_ave): tfunc()
        t_elapsed = time.perf_counter() - t_start

        print(f"single call of spsolve: {tscale*t_elapsed/n_ave:.6f} {unit} (average of {n_ave} calls)")