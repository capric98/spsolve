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
- a `stats` dictionary with ordering, symbolic, numeric, assembly, and total
  factor times; ordering and numeric-engine names; fill and AMD estimates;
  pivot/threading counters; and workspace/front diagnostics.

`trans` accepts `N`, `T`, and `H`; `H` equals `T` for a real factor. A native
factor accepts real numeric vector or matrix RHSs and returns `float64`.

Native solves enable iterative refinement by default, with at most two
correction solves. The componentwise backward error is checked after every
correction against `10 * float64 epsilon`. A `RuntimeWarning` reports failure
to reach that tolerance, stalled improvement, or a non-finite error. If a
correction worsens or invalidates the error, the preceding better solution is
returned. Set `iterative_refinement=False` to use only the initial solve.

## `factorized`

`factorized(A, **factor_options)` retains a factor and returns its bound
`solve` callable. Pass `backend="native"` to select the experimental backend.

## `spsolve`

`spsolve(A, b, ...)` performs a one-shot factor and solve. It accepts vector and
dense matrix RHSs. `backend="auto"` is the stable default; use explicit
`native` only when the current float64 scope and performance profile fit the
problem.

For general `auto` and `scipy` solves, `A` and `b` are jointly promoted to a
shared floating or complex dtype before factorization. This supports a real
matrix with a complex RHS and preserves the precision of a wider RHS. Matrix
values and the original RHS remain unchanged unless `overwrite_b=True` and
the RHS is a writable NumPy array with the solution's shape and dtype.
Explicit general native LU still requires a real `float64` matrix and a real
RHS; the float64 triangular fast path also accepts complex RHSs.

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

## Dense kernel selection

```python
from spsolve.spparms import available_dense_kernels, get_dense_kernel, set_dense_kernel

available_dense_kernels()   # usable kernels, best first, e.g. ['sse2', 'scalar']
set_dense_kernel("scalar")  # case-insensitive: 'auto', 'avx2'/'avx2-fma', 'sse2', 'scalar'
get_dense_kernel()          # 'scalar'
set_dense_kernel(None)      # restore the startup selection
```

Native blocked LU panel flushes and supernodal LDL^T updates run on a dense
tile kernel. The startup selection is the best kernel this build and CPU
support, unless the environment variable `SPSOLVE_FORCE_KERNEL` (`auto`,
`avx2` or `avx2-fma`, `sse2`, or `scalar`, case-insensitive) names a usable
kernel when the extension first selects one; an unknown or unsupported value
is ignored.
`set_dense_kernel` raises `ValueError` for an unknown or unsupported name and
`TypeError` for a non-string; `'auto'` selects the best available kernel and
`None` restores the startup selection.

The setting is process-wide. Each factorization takes one snapshot when it
starts, so a concurrent change never mixes kernels within a factor, and
`stats["dense_kernel"]` reports the kernel used. Within one kernel, factors are
bitwise identical at every thread count. Different kernels may differ in the
last bits (a fused multiply-add kernel rounds once per term instead of twice);
select a kernel explicitly to reproduce factors across machines.
