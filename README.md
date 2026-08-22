# spsolve

`spsolve` is a lightweight sparse direct solver for Python. It provides a
clean-room, reusable, multithreaded sparse LU implementation for general real
matrices and a fast native triangular solver. The native extension does not
link against Intel MKL or another external solver library; SciPy remains the
sparse-container compatibility layer and explicit fallback.

The native LU backend is currently opt-in. The default `"auto"` backend uses
SciPy SuperLU for general matrices until the documented performance promotion
gate passes consistently; triangular float64 systems continue to use the
native fast path.

## Install

```bash
pip install -U git+https://github.com/capric98/spsolve
```

The build needs a C++20 compiler. OpenMP is detected automatically; without it,
triangular kernels fall back to serial while LU still uses `std::thread`.
Wheels and ordinary builds target baseline x64 CPUs; use
`-DSP_ARCH_NATIVE=ON` only for a machine-local build.

## General sparse LU

```python
import numpy as np
from scipy.sparse import csc_array
from spsolve import factorized, splu, spsolve

A = csc_array([
    [4.0, 1.0, 0.0],
    [2.0, 3.0, 1.0],
    [1.0, 0.0, 2.0],
])
b = np.array([1.0, 2.0, 3.0])

# Stable default: SciPy SuperLU for a general matrix.
x = spsolve(A, b)

# Experimental native float64 factorization, reusable for many RHSs.
lu = splu(
    A,
    backend="native",
    permc_spec="MMD_AT_PLUS_A",
    num_threads=8,
)
x = lu.solve(b)
x_transpose = lu.solve(b, trans="T")

solve_A = factorized(A, backend="native", num_threads=8)
x_again = solve_A(b)
```

Native factors expose `shape`, `dtype`, `nnz`, `perm_r`, `perm_c`, lazy `L`
and `U` sparse arrays, equilibration scales, and timing/fill statistics. Row and
column scaling, threshold partial pivoting, exact singular detection, and up to
two iterative-refinement steps are enabled by default.

## Triangular solve

```python
from spsolve import spsolve_triangular

x = spsolve_triangular(A_triangular, b, lower=True, num_threads=8)
```

The triangular solver accepts vector or matrix RHSs, C- or Fortran-contiguous
layouts, real or complex RHSs with a real matrix, and implicit unit diagonals.
Parallel work is across independent RHS blocks.

## Backend policy and limitations

- `backend="auto"`: SciPy for general matrices; native for the established
  triangular float64 fast path.
- `backend="native"`: experimental native LU for float64 matrices. Ordering
  choices are `NATURAL`, clean-room approximate `COLAMD`, and
  `MMD_AT_PLUS_A`.
- `backend="scipy"`: explicit SciPy SuperLU fallback, including complex and
  non-float64 problems supported by SciPy.

Sparse LU exposes limited parallelism for narrow-front or already triangular
matrices because pivots are dependency ordered. Irregular matrices with wide
fronts benefit most. The native implementation remains opt-in because it does
not yet consistently match SuperLU's mature supernodal performance.

See [the implementation notes](docs/native-lu.md), [API notes](docs/api.md),
and [benchmark gate](docs/benchmarking.md) for details.

## Development

```bash
python -m unittest discover -v -s tests -t .
python benchmarks/factorization.py --quick
python benchmarks/factorization.py --threads 8 --gate
```

The project is BSD-licensed. NumPy, SciPy, nanobind, CMake, the platform C++
runtime, and optional OpenMP are the only runtime/build foundations; no solver
source is vendored.
