# Native sparse LU design

## Scope

The native backend factors square `float64` SciPy sparse arrays without an
external factorization or BLAS runtime. Its numerical factorization kernels
are implemented in this project. A small BSD-licensed subset of SuiteSparse
AMD 3.3.4 and COLAMD 3.3.5 is compiled in for fill-reducing orderings; SciPy is
retained as the mature compatibility fallback.

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
3. A column ordering is selected:
   - `NATURAL` preserves input order.
   - `COLAMD` uses the vendored COLAMD routine on the input pattern. Engine
     selection uses actual numeric work in this permutation; there is no
     second ordering pass.
   - `MMD_AT_PLUS_A` uses the vendored AMD routine on the structural graph of
     `A + A.T`. AMD's fill and maximum-front estimates also guide numeric
     engine selection.
4. Rows and columns are equilibrated by infinity norms. Structural zero rows or
   columns are reported as singular before numerical elimination.
5. The factorizer chooses one of three numerical engines:
   - An exactly symmetric matrix first attempts a sparse simplicial `LDL.T`
     factorization with symmetric equilibration, using whichever column
     permutation the caller requested as a symmetric permutation. Symmetry
     requires matching transpose entries with exactly equal values in the
     canonical input. Unequal entries, even one floating-point step apart,
     use general LU so that neither triangle of the input is discarded.
     Positive pivots certify the SPD path. Its factors are exposed through the
     common LU contract as `L` and `D @ L.T`; internally it retains only the
     strict lower triangle and diagonal, constructing public `L` and `U`
     lazily. A non-positive pivot falls back to pivoted LU.
   - A nonsymmetric matrix with `MMD_AT_PLUS_A` uses a Gilbert-Peierls-style
     left-looking CSC LU when AMD estimates a maximum front at most 192 and
     fill at most 20. `COLAMD` instead starts in CSC with its actual pivots:
     a column reach above 192, more than 8,192 scalar updates in a column,
     more than `64 * nnz(A)` total updates, or factor storage above
     `20 * nnz(A)` triggers a restart in the row engine. The discarded probe
     is bounded, and restart preserves the requested column permutation and
     equilibration. Both engines perform threshold partial pivoting.
   - Wider or fill-heavy patterns use the row-hybrid LU. Sorted sparse rows
     convert to dense tails when their remaining front is at least 25% dense.
     A persistent `std::thread` team dynamically schedules sufficiently large
     independent row-update fronts.
6. Factors remain in their engine-native CSC or row storage. Public CSC `L`
   and `U` objects are materialized lazily. Solves operate directly on the
   retained storage and parallelize blocks of independent right-hand sides.
7. Python optionally performs at most two componentwise backward-error
   refinement steps using the original matrix. It checks the error after each
   correction, retains the better iterate if a correction worsens or
   invalidates the error, and warns when the tolerance is not reached.

All arithmetic within a row follows a deterministic order. A threaded factor
therefore produces the same permutations and factor entries as a one-thread
factor for the same input and options.

## Threading behavior

Sparse LU has a sequential pivot dependency. The row-hybrid engine parallelizes
the expensive work below each selected pivot, not pivot selection itself.
Statistics expose `numeric_engine`, `parallel_steps`, `row_updates`, and
`largest_front` so users can tell which path ran and whether a problem supplied
meaningful parallel work.

`ordering_estimated_l_nnz` and `ordering_estimated_max_front` report AMD
estimates for `MMD_AT_PLUS_A`; zero means unavailable for other orderings.
COLAMD exposes `numeric_probe_columns`, `numeric_probe_max_front`,
`numeric_probe_updates`, `numeric_probe_seconds`, and `numeric_probe_fallback`.
The probe counts completed columns and scalar updates, and records the largest
observed reach, including the column that triggers a restart. Probe time is
included in numeric/factor time; on success it is the CSC factorization itself.
Workspace diagnostics include the temporary restart snapshot and discarded
CSC buffers on fallback.

Good threading candidates have wide, irregular fronts and substantial fill.
Banded and PDE-like problems usually select a low-overhead serial CSC engine;
their goal is parity through efficient sparse traversal rather than parallel
speedup. Factor solves parallelize across RHS blocks; a single RHS solve is
serial.

The factor worker pool uses only the C++ standard library. OpenMP is optional
and currently accelerates the established multi-RHS triangular kernels. A
build without OpenMP remains functional.

## Numerical safeguards

- row and column infinity-norm equilibration;
- threshold partial pivoting in every active column;
- positive-pivot validation before accepting the SPD specialization;
- structural and exact numerical singular detection;
- rejection of non-finite inputs by default;
- overflow checks during scaling and row updates;
- componentwise iterative refinement with final-error and convergence checks.

This is not a rank-revealing factorization and does not estimate condition
numbers. Very ill-conditioned matrices can still lose forward accuracy even
when their backward residual is small.

## Current limitations and next optimization target

- native matrix values are `float64` only;
- no complex native factors yet (the SciPy backend handles them);
- ordering and pivot selection are single-threaded;
- the CSC engines are simplicial rather than blocked supernodal kernels;
- there is no out-of-core mode or symbolic-factor reuse across matrices with
  the same pattern.

Fundamental supernodes are detected for diagnostics on the SPD path, but the
current implementation does not yet execute blocked panel updates. A future
supernodal path could improve cache reuse on larger fronts. Default promotion
also remains contingent on repeated release-build results on both Windows and
Linux, as described in the benchmark notes.
