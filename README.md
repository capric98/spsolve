# spsolve

`spsolve` is a sparse linear equations solver that is compatible with `scipy.sparse.linalg.spsolve_triangular`.

It implements a naïve forward / backward substitution solver which:

1. Use AVX2 instructions to vectorize the calculation.
2. Use OpenMP to parallel when $\mathbf{b}$ has many columns. Achieve the best performance when $n_\text{RHS} \ge 4 \times n_\text{cores}$

## Usage

```bash
pip install -U git+https://github.com/capric98/spsolve
```

And then replace `scipy.sparse.linalg.solve_triangular()` to `spsolve.solve_triangular()` in your code.

A `spsolve.spsolve()` function is available for replacement of `scipy.sparse.linalg.solve()`, currently it simply uses `scipy.sparse.linalg.splu` and then solve `Ax=b` via `x = Pc @ { U \ [ L \ (Pr@b) ] }`, while `splu` almost runs in a single thread. Please let me know if you has a better idea to do LU decomposition. Try [PyPardiso](https://github.com/haasad/PyPardiso), [SLEPc](https://slepc.upv.es/), [Trilinos](https://github.com/trilinos/Trilinos), etc., for general using cases.

An experimental Intel MKL PARDISO based `spsolve()` can be used if user has Intel MKL installed before hand, and build the project with explicitly flag set:

```
pip install -U git+https://github.com/capric98/spsolve --config-setting=cmake.args="-DSP_USE_MKL=ON"
```

## Limitations

1. Currently **only** support CPUs with AVX2 instructions.

2. Native support only for `scipy.sparse.csr_array`, other sparse array will be converted to CSR format.

3. Slight performance degradation when $n_\text{RHS}$  is not fourfold.

4. Not fully parallel when $n_\text{RHS}$ is small.

5. Limited data type supported:

   |       |       $A$       |  \   |        $b$        |  =   |        $x$        |                                                              |
   | ----: | :-------------: | :--: | :---------------: | :--: | :---------------: | :----------------------------------------------------------- |
   | dtype |  `np.float64`   |      |   `np.float64`    |      |   `np.float64`    | ✅                                                            |
   | dtype |  `np.float64`   |      | ``np.complex128`` |      | ``np.complex128`` | ✅ View `b` as double and solve a $2\times n_\text{RHS}$ problem. |
   | dtype | `np.complex128` |      |   `np.float64`    |      | ``np.complex128`` | ❌ Will support later.                                        |
   | dtype | `np.complex128` |      |  `np.complex128`  |      |  `np.complex128`  | ❌ Will support later.                                        |

   All other data types will be cast to `np.float64` or `np.complex128`. For experts who benefit from low precision or require higher precision, it should be easy to modify this project.

## Performance
* Environment:
  * Intel Core i5-13600K, P-Cores @5.2GHz, Windows 10 LTSC
  * Python 3.13.5: `OMP_NUM_THREADS=6`, manually bind to physical P-Cores, `overwrite_b=True`
  * MATLAB R2025a: `maxNumCompThreads=6` (default)

* Comparison between SciPy, MATLAB and spsolve:

  * $\mathbf{A}$: $10000\times10000$ sparse array with density of 10%, non-zero main diagonal dominant, then use `scipy.sparse.(tril|triu)` to get a triangular sparse array, stored in CSR format

  * $\mathbf{b}$: $10000\times n_\text{RHS}$ dense `np.ndarray`

  * Each function runs 1000 times and uses the averaged time of single solve.

![](./benchmarks/static/speedup.png)

## TODO

- [x] Add tests.
- [ ] Implement `spsolve_triangular` for `np.complex128`.
- [ ] Implement `solve` for general `scipy.sparse` matrices.

## Acknowledgments

* [MATLAB: mldivide, \\](https://www.mathworks.com/help/matlab/ref/double.mldivide.html)
* [NumPy](https://numpy.org/)
* [OpenMP](https://www.openmp.org/)
* [SciPy](https://scipy.org/)
* [nanobind](https://github.com/wjakob/nanobind)
* [psutil](https://github.com/giampaolo/psutil)
