import timeit

import scipy
import numpy as np

from spsolve import spsolve
# from scipy.sparse.linalg import spsolve


if __name__ == "__main__":

    bnum = 2
    rnum = 10
    size = 1000
    nrhs = 1000
    density = 0.1

    np.random.seed(0)

    # A: size * size (sparse)
    # b: size * nrhs
    A = scipy.sparse.csc_array(scipy.sparse.spdiags(5+np.random.rand(size), 0, size, size) + scipy.sparse.random(size, size, density))
    x = np.random.randn(size, nrhs) #+ 1j*np.random.randn(size, nrhs)
    b = A @ x

    # print(A.toarray())

    usr_ans = spsolve(A, b)
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
        exit(1)
    else:
        print(f"ans == x, max_abs_err = {np.max(diff):.2e}")



    def tfunc():
        np.copyto(usr_ans, b)
        spsolve(A, usr_ans, overwrite_b=True)

    execution_times = timeit.repeat(stmt=tfunc, number=bnum, repeat=rnum)
    times_per_call  = 1000 * np.array(execution_times) / bnum

    avg_time = np.mean(times_per_call)
    std_dev = np.std(times_per_call)
    best_time = np.min(times_per_call)
    worst_time = np.max(times_per_call)


    print(f"bench \"spsolve_triangular\" {rnum} times, each time average {bnum} calls.")
    print("-" * 30)
    print(f"Average Time:  {avg_time:.6f} ms")
    print(f"Std Deviation: {std_dev:.6f} ms")
    print(f"Best Time:     {best_time:.6f} ms")
    print(f"Worst Time:    {worst_time:.6f} ms")
