import functools
import timeit

import scipy
import numpy as np

from spsolve import spsolve_triangular


if __name__ == "__main__":

    bnum = 10
    rnum = 1000
    size = 1000
    nrhs = 1000 # 20
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

        # b = b.astype(np.intc, copy=True) # test data type
        # b = b.copy(order="F")

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

        # if size <= 10000:
        #     npy_ans = np.linalg.solve(A.todense(), b)
        #     spy_ans = scipy.sparse.linalg.spsolve_triangular(csrA, b, lower=lower)

        #     if size * nrhs < 100:
        #         print(npy_ans)
        #         print(usr_ans)
        #         print(usr_ans - npy_ans)

        #     print("numpy==user:", np.allclose(npy_ans, usr_ans))
        #     print("scipy==user:", np.allclose(spy_ans, usr_ans))
        #     # print(A@npy_ans-b)
        #     # print(A@usr_ans-b)


        # start_time = time.time()
        # for _ in range(bnum):
        #     scipy.sparse.linalg.spsolve_triangular(csrA, b, lower=lower)
        # end_time = time.time()
        # print(f"scipy(spsolve_tri) elapsed: {(end_time-start_time):.2f} s")

        def tfunc():
            np.copyto(usr_ans, b)
            spsolve_triangular(A, usr_ans, overwrite_b=True, lower=lower)

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

    # t_wait = 10
    # print(f"wait {t_wait}s to start run nrhs figure using {__THREADS__} threads...")
    # time.sleep(t_wait)

    # bnum = 100
    # size = 10000
    # nrhs = 20 # 20
    # density = 0.1

    # LUA = scipy.sparse.spdiags(5+np.random.rand(size), 0, size, size) + scipy.sparse.random(size, size, density)

    # max_nrhs  = 100
    # nrhs_step = 1
    # nrhs_arr  = list(range(nrhs_step, max_nrhs+1, nrhs_step))
    # time_data = np.zeros((len(nrhs_arr), 2))

    # for one in range(2):
    #     lower = (one==1)
    #     for k in range(len(nrhs_arr)):
    #         nrhs = nrhs_arr[k]

    #         A = scipy.sparse.tril(LUA) if lower else scipy.sparse.triu(LUA)
    #         x = np.random.randn(size, nrhs)
    #         b = A @ x

    #         start_time = time.time()
    #         for _ in range(bnum):
    #             spsolve_triangular(A, b, overwrite_b=True, lower=lower)
    #         end_time = time.time()

    #         time_data[k, one] = end_time-start_time
    #         print(f"{'L' if lower else 'U'} nrhs={nrhs} t={time_data[k, one]:.2f}s")

    # import matplotlib.pyplot as plt

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Times New Roman"],
    # })
    # plt.plot(nrhs_arr, time_data)
    # plt.xlim((min(nrhs_arr), max(nrhs_arr)))
    # plt.xlabel("$n_\\mathrm{rhs}$")
    # plt.ylabel("Time (s)")
    # plt.legend(["Upper", "Lower"])

    # plt.show()