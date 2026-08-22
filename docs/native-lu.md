# Native sparse LU design

## Scope

The native backend factors square, nonsymmetric `float64` SciPy sparse arrays
without MKL, SuiteSparse, or another external factorization package. It is a
clean-room implementation intended to establish a portable correctness and
threading baseline while retaining SciPy as a mature fallback.

The stored relationship is

```text
Pr @ Dr @ A @ Dc @ Pc = L @ U
```

`perm_r[k]` and `perm_c[k]` are the original row and column at factor position
`k`. `Dr` and `Dc` are diagonal infinity-norm equilibration scales. `solve`
applies all permutations and scales internally for normal and transposed
systems.

## Factorization pipeline

1. Python converts the input to a copied, canonical CSC array, sums duplicate
   entries, removes explicit zeros, sorts indices, and checks shape/dtype.
2. C++ validates CSC bounds and optional finiteness.
3. Rows and columns are equilibrated by infinity norms. Structural zero rows or
   columns are reported as singular before numerical elimination.
4. A column ordering is selected:
   - `NATURAL` preserves input order.
   - `COLAMD` is a clean-room approximate minimum-degree ordering of the
     column-intersection graph. Dense rows and graph growth are bounded.
   - `MMD_AT_PLUS_A` performs minimum-degree elimination on the structural
     graph of `A + A.T`; it is generally preferable for PDE-like or nearly
     symmetric patterns.
5. Sparse Gaussian elimination stores sorted active rows. Rows whose remaining
   front becomes at least 25% dense switch to a contiguous dense-tail
   representation, avoiding repeated sparse merges and allocator contention.
   At every column the factor chooses the largest available pivot unless the
   original diagonal candidate meets `diag_pivot_thresh * column_max`.
6. Rows below a pivot are independent. A persistent `std::thread` worker pool
   dynamically schedules sufficiently large row-update fronts; small fronts
   stay serial to avoid synchronization overhead.
7. `L` and `U` are retained in row form. Their public CSC objects are built
   lazily. Solves traverse these factors and parallelize blocks of independent
   right-hand sides.
8. Python optionally performs two componentwise backward-error refinement
   steps using the original matrix.

All arithmetic within a row follows a deterministic order. A threaded factor
therefore produces the same permutations and factor entries as a one-thread
factor for the same input and options.

## Threading behavior

Sparse LU has a sequential pivot dependency. The native backend parallelizes
the expensive work below each selected pivot, not the pivot selection itself.
Statistics expose `parallel_steps`, `row_updates`, and `largest_front` so users
can tell whether a problem supplied meaningful parallel work.

Good candidates have wide, irregular fronts and substantial fill. Banded,
triangular, or very small problems usually remain serial and should not be
expected to speed up. Factor solves parallelize across RHS blocks; a single RHS
solve is serial.

The factor worker pool uses only the C++ standard library. OpenMP is optional
and currently accelerates the established multi-RHS triangular kernels. A
build without OpenMP remains functional.

## Numerical safeguards

- row and column infinity-norm equilibration;
- threshold partial pivoting in every active column;
- structural and exact numerical singular detection;
- rejection of non-finite inputs by default;
- overflow checks during scaling and row updates;
- componentwise iterative refinement with stall detection.

This is not a rank-revealing factorization and does not estimate condition
numbers. Very ill-conditioned matrices can still lose forward accuracy even
when their backward residual is small.

## Current limitations and next optimization target

- native matrix values are `float64` only;
- no complex native factors yet (the SciPy backend handles them);
- ordering and pivot selection are single-threaded;
- numerical elimination is row-oriented rather than supernodal;
- there is no out-of-core mode or symbolic-factor reuse across matrices with
  the same pattern.

The key remaining performance step is a supernodal/panel representation with
dense blocked updates and a dependency scheduler. That would improve cache
reuse and vectorization and reduce allocator pressure on large fronts. The
current backend is intentionally not promoted to the default until the
reproducible gate passes.
