# spsolve

`spsolve` is a lightweight sparse direct solver for Python. It provides a
reusable, adaptive, multithreaded sparse LU implementation for general real
matrices and a fast native triangular solver. The extension has no external
factorization or BLAS runtime dependency; SciPy remains the sparse-container
compatibility layer and explicit fallback. A small BSD-licensed AMD/COLAMD
ordering subset is compiled into the extension.

The native LU backend is currently opt-in. The default `"auto"` backend uses
SciPy SuperLU for general matrices until the documented performance promotion
gate passes consistently; triangular float64 systems continue to use the
native fast path.

## Install

Python 3.14 or newer is required.

```bash
pip install -U git+https://github.com/capric98/spsolve
```

The build needs a C++20 compiler. OpenMP is detected automatically; without it,
triangular kernels fall back to serial while LU still uses `std::thread`.
Wheels and ordinary builds target baseline x64 CPUs; AVX2+FMA dense kernels
are selected at runtime when the CPU supports them; use `-DSP_ARCH_NATIVE=ON`
only for a machine-local build.

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
  choices are `NATURAL`, SuiteSparse `COLAMD`, and SuiteSparse AMD-backed
  `MMD_AT_PLUS_A`.
- `backend="scipy"`: explicit SciPy SuperLU fallback, including complex and
  non-float64 problems supported by SciPy.

Sparse LU exposes limited parallelism for narrow-front or already triangular
matrices because pivots are dependency ordered. Irregular matrices with wide
fronts benefit most. Structured symmetric positive-definite and narrow-front
problems use lower-overhead serial CSC engines instead. On one Windows x64
machine (i5-13600K, `avx2-fma`) the native backend passes the checked-in
performance gate (`--extended --dense-lu`) with both the promotion ordering
`MMD_AT_PLUS_A` and `COLAMD`, the default `permc_spec` (three consecutive
runs each). The backend remains opt-in until the gate passes consistently on
both Windows and Linux release builds.

See [the implementation notes](docs/native-lu.md), [API notes](docs/api.md),
and [benchmark gate](docs/benchmarking.md) for details.

## Development

CI runs on Python 3.14 on Windows and Linux, covering automatic dense-kernel
selection, forced SSE2, and a Linux build with AVX2 kernels disabled.

```bash
python -m unittest discover -v -s tests -t .
python benchmarks/factorization.py --quick
python benchmarks/factorization.py --threads 8 --gate
```

Set `SPSOLVE_FORCE_KERNEL` (`auto`, `avx2`/`avx2-fma`, `sse2`, or `scalar`) to
run the native factorizations on a specific dense kernel, for example
`SPSOLVE_FORCE_KERNEL=sse2` to test the baseline kernel on an AVX2 machine or
to reproduce factors across machines. Build with
`-Ccmake.define.SP_ENABLE_AVX2=OFF` to omit the AVX2 kernel entirely.

The project is BSD-licensed. NumPy, SciPy, nanobind, CMake, the platform C++
runtime, and optional OpenMP are the runtime/build foundations. BSD-licensed
AMD/COLAMD ordering source is vendored; no third-party factorization solver or
BLAS implementation is included. See [third-party notices](THIRD_PARTY_NOTICES.md).
