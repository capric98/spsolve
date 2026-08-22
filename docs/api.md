# API notes

## `splu`

```python
splu(
    A,
    permc_spec="COLAMD",
    diag_pivot_thresh=1.0,
    *,
    num_threads=None,
    backend="auto",
    check_finite=True,
)
```

`A` must be a square SciPy sparse matrix or array. General `auto` calls return
a SciPy `SuperLU` factor while the native performance gate is pending. Explicit
`native` calls return `SparseLU` and currently require `float64` plus one of
`NATURAL`, `COLAMD`, or `MMD_AT_PLUS_A`.

The native object provides:

- `solve(b, trans="N", overwrite_b=False, num_threads=None,
  iterative_refinement=True)`;
- `shape`, `dtype`, and factor `nnz`;
- immutable `perm_r`, `perm_c`, `row_scale`, and `col_scale` arrays;
- lazily materialized CSC `L` and `U`;
- a `stats` dictionary with ordering/factor times, fill, pivot and threading
  counters.

`trans` accepts `N`, `T`, and `H`; `H` equals `T` for a real factor. A native
factor accepts real numeric vector or matrix RHSs and returns `float64`.

## `factorized`

`factorized(A, **factor_options)` retains a factor and returns its bound
`solve` callable. Pass `backend="native"` to select the experimental backend.

## `spsolve`

`spsolve(A, b, ...)` performs a one-shot factor and solve. It accepts vector and
dense matrix RHSs. `backend="auto"` is the stable default; use explicit
`native` only when the current float64 scope and performance profile fit the
problem.

## `spsolve_triangular`

The native triangular API is compatible with the main SciPy arguments and adds
`num_threads`. The matrix is canonicalized as CSR and must actually match the
requested lower/upper triangle. With `unit_diagonal=True`, stored diagonal
values are ignored and absent diagonal entries mean one.

Real numeric matrices are converted to `float64`. RHSs are returned as
`float64` or `complex128`, preserve vector versus matrix shape, and may be
overwritten only when their dtype, writability, and memory layout permit it.

## Global thread setting

```python
from spsolve.spparms import set_num_threads

set_num_threads(8)
```

An explicit per-call `num_threads` takes precedence. Otherwise the initial
value comes from `OMP_NUM_THREADS` or `os.cpu_count()`.
