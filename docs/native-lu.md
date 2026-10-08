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
   - An exactly symmetric matrix first attempts a sparse `LDL.T`
     factorization with symmetric equilibration, using whichever column
     permutation the caller requested as a symmetric permutation. Symmetry
     requires matching transpose entries with exactly equal values in the
     canonical input. Unequal entries, even one floating-point step apart,
     use general LU so that neither triangle of the input is discarded.
     Positive pivots certify the SPD path. Its factors are exposed through the
     common LU contract as `L` and `D @ L.T`; internally it retains only the
     strict lower triangle and diagonal, constructing public `L` and `U`
     lazily. Narrow fronts use the simplicial CSC kernel. Adjacent elimination
     tree parents and column counts give upper bounds on fundamental-node
     width and work. An exact pattern pass requires possible nodes at least
     320 columns wide with sufficient combined work; exact nodes must pass the
     same guard to enable the blocked path described below. Ten 32-column
     panels amortize packing, while the work guard prevents a small wide node
     from routing overwhelmingly narrow matrices through the panel driver.
     A non-positive pivot or non-finite
     intermediate falls back to pivoted LU with the original equilibration.
   - A nonsymmetric matrix with `MMD_AT_PLUS_A` uses a Gilbert-Peierls-style
     left-looking CSC LU when AMD estimates a maximum front at most 192 and
     fill at most 20. `COLAMD` instead starts in CSC with its actual pivots:
     a column reach above 192, more than 8,192 scalar updates in a column,
     more than `64 * nnz(A)` total updates, or factor storage above
     `20 * nnz(A)` triggers a restart in the row engine. The discarded probe
     is bounded, and restart preserves the requested column permutation and
     equilibration. Both engines perform threshold partial pivoting.
   - Wider or fill-heavy patterns use the row-hybrid LU. Sorted sparse rows
     with at least 64 entries convert to dense tails when at least 25% of
     their own remaining span, from first to last remaining nonzero column,
     is nonzero. The tail is allocated for that span with its end rounded up
     to 64 columns, and grows on demand; columns outside it are exact zeros.
     Later blocks of a block-diagonal matrix therefore carry no leading zero
     columns, and pivot candidate scans skip dense rows whose storage
     excludes the pivot column.
     Sparse rows receive eager updates and retain their fill adjacency. When
     at least 32 active rows are dense with at least 128 columns remaining,
     dense rows use panels of up to 32 pivot columns. A persistent `std::thread`
     team schedules large sparse row updates and dense output tiles.
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

## Blocked SPD panels

The `symmetric-supernodal-ldlt` engine processes one exact fundamental
supernode at a time. Its dense workspace contains only that node's columns
and filled rows. It scatters original matrix entries, then subtracts prior
node contributions `L[R,Q] @ D[Q] @ L[J,Q].T`. Sparse row incidence discovers
contributors without scanning all prior nodes. Contributors are deduplicated
and visited in factor order; only their actual row and column intersections
are packed, including noncontiguous and partial overlaps.

The diagonal block is factored in panels of at most 32 columns. The below-panel
solve applies `inv(L_panel.T)` and `inv(D_panel)` once, followed by a tiled
rank-k update of the remaining lower triangle and rectangular tail. Independent
16-by-16 output tiles share the persistent worker team and call the shared
dense kernels described below. A tile whose output rows and columns are both
contiguous node positions (always true for trailing updates) is updated in
place in the node rectangle; a tile that straddles the diagonal writes only its
lower entries, and tiles entirely above it are skipped. An irregular
intersection tile is gathered into a 16-by-16 scratch block, updated, and
scattered back. Because the output columns are a prefix of the ascending
output rows, a tile that straddles the diagonal is a diagonal tile, and only
its lower triangle is gathered, computed, and scattered. Tiny intersections
retain sparse outer products.

For node width `W` and height `R`, predicted internal update terms are
`W*(W-1)*(3*R-2*W+1)/6`. Only nodes at least 320 columns wide contribute useful
work; their total must be at least `3072*n`, where `n` is the whole matrix
dimension. The cheap parent/count runs bound this work before the extra exact
pattern pass, and actual fundamental nodes enforce the final decision.
The measured guard accounts for per-node driver overhead without rejecting
an expensive large node solely because it covers few matrix columns.

The filled CSC pattern, including entries that cancel numerically to zero,
remains the public storage contract. No factors, symmetric scales, or blocked
work counters are published until every positive pivot and finite intermediate
has passed validation. Each output entry visits contributors in the same order
at every thread count.

## Dense LU panels

Pivoting remains sequential. Sparse rows receive each complete U-row update
immediately. Dense rows keep every coefficient inside the current panel
current, while deferring updates beyond its end. Before a dense row becomes
a pivot, its deferred tail is materialized from its L multipliers and retained
U rows in ascending pivot order. U is therefore complete before it supplies
any subsequent update, even when row swaps cross panel boundaries.

At a panel boundary, surviving affected dense rows receive a tiled
`L_panel * U_panel` update. Packing uses the union of actual U tail columns;
it does not sweep unrelated zero tails. Every output starts from its current
value and subtracts each source term in pivot order. Independent 16-by-64
tiles call the shared dense kernels: when a tile's packed columns are
contiguous, the kernel updates the rows' dense tails in place; otherwise the
tile is gathered into a 16-by-64 scratch block and scattered back. Both paths
compute identical bits, and the choice depends only on column positions.
Sparse-to-dense conversions occur only after this flush, so conversion
neither omits nor duplicates an update. After activation, short trailing
panels retain the same recurrence.

The activation thresholds are independent of the requested thread count.
Purely sparse phases keep their existing update scheduler and do not allocate
panel packing buffers. Selected pivot rows release their dense storage only
after U has been retained; eliminated rows never participate in a later flush.

## Dense kernels

Both blocked paths call one kernel table, `dense_kernels::Kernels`
(`src/dense_kernels.hpp`): `gemm_sub` (`C -= A @ B`), `gemm_sub_rows` (the
same with per-row output pointers), `gemm_sub_lower` (writes only entries with
`i + shift >= j`), and `axpy_sub` (`y -= alpha * x`, not yet used by the
engines). The GEMM entry points take an accumulation order, each with a
separate multiply per term (no fused multiply-add):

- `running`: the output starts from its current value and subtracts each
  term in ascending `k`, `c = (c - a0*b0) - a1*b1 - ...`;
- `sum_then_subtract`: the terms are summed from zero in ascending `k`, and
  the sum is subtracted from the output once, `c = c - (a0*b0 + a1*b1 + ...)`.

The LU flush uses `running`. It is the recurrence of the eager in-panel
updates and of pivot-tail materialization, so a dense row's entries receive
the same arithmetic whichever of those paths applies a term, and the flush
stays bitwise identical to its previous implementation. The supernodal LDL^T
update uses `sum_then_subtract`, its previous association: after symmetric
scaling the diagonal and nearby entries it updates are often much larger than
the individual terms, and summing first rounds at the scale of the terms with
one rounding at the scale of the entry. With `running` the LDL^T backward
error `max|P^T S A S P - L D L^T|` grew about fourfold (for example from
6.7e-16 to 2.6e-15 on four dense 600-wide SPD blocks).

In either order an element's result depends only on its own operands, never
on its tile, path, or thread. The `sse2` kernel generalizes the earlier 4-by-4
microkernel to 1-4 rows by 4, 2, or 1 columns, so edges use the same
arithmetic, and reproduces the previous LU and LDL^T results bit for bit. The
portable `scalar` kernel performs the same operations and, on x86-64,
produces the same bits.

Every kernel reports whether any element it wrote is non-finite. The SIMD
kernels fold `x - x` of every stored register into one witness per microtile
(zero for finite values, NaN otherwise) and test it once, so every written
element is covered. A kernel may stop at its first non-finite block, leaving
later elements uncomputed, so its flag is authoritative on every path: a
false flag raises the usual LU non-finite error or rejects the SPD attempt
before any gathered tile is scattered back, and no path scatters an element
the kernel was not asked to compute.

A factorization snapshots the active kernel once, before ordering, and passes
it to both engines; `stats["dense_kernel"]` names it (`avx2-fma`, `sse2`, or
`scalar`) even when the selected engine performs no tile updates. The default
is the best kernel the build and CPU support; `SPSOLVE_FORCE_KERNEL` and
`spsolve.spparms.set_dense_kernel` override it (see `docs/api.md`). Wider
kernels belong in separately compiled translation units behind a CPUID and
XCR0 check (`cpu_supports_avx2_fma`); the shared header holds declarations
only, so no inline code compiled for a wider target can be linked into the
baseline paths. This build has no AVX2 translation unit yet
(`SPSOLVE_HAVE_AVX2_KERNELS=0`).

## Threading behavior

Sparse LU has a sequential pivot dependency. The row-hybrid engine parallelizes
the expensive work below each selected pivot and at dense panel boundaries,
while pivot selection remains serial.
Statistics expose `numeric_engine`, `parallel_steps`, `row_updates`, and
`largest_front` so users can tell which path ran and whether a problem supplied
meaningful parallel work.

`lu_dense_panels` counts nonempty dense tail flushes, and
`lu_parallel_panels` counts flushes that used the worker team; the latter is
included in `parallel_steps`. `lu_panel_columns` is the maximum packed U-tail
support size. `lu_dense_update_flops` counts two operations per packed
multiply/subtract term, including zeros in packed operands; eager panel
coefficients and pivot-tail materialization are excluded.
`lu_panel_workspace_bytes` estimates peak panel-owned vector capacities,
including discovery marks, row/column lists, and packed L/U operands. It
excludes retained factors, row-engine sparse/dense storage, and worker runtime
storage, and must not be interpreted as total row-engine peak memory.
The SPD-specific `parallel_panels`, `dense_update_flops`, and `spd_*` fields
retain their existing meaning. `dense_kernel` names the dense tile kernel the
factorization snapshotted (`avx2-fma`, `sse2`, or `scalar`; see "Dense
kernels"), whether or not the selected engine used it.

`ordering_estimated_l_nnz` and `ordering_estimated_max_front` report AMD
estimates for `MMD_AT_PLUS_A`; zero means unavailable for other orderings.
COLAMD exposes `numeric_probe_columns`, `numeric_probe_max_front`,
`numeric_probe_updates`, `numeric_probe_seconds`, and `numeric_probe_fallback`.
The probe counts completed columns and scalar updates, and records the largest
observed reach, including the column that triggers a restart. Probe time is
included in numeric/factor time; on success it is the CSC factorization itself.
Workspace diagnostics include the temporary restart snapshot and discarded
CSC buffers on fallback.

For blocked SPD, `parallel_panels` counts actual parallel tile phases and is
also included in `parallel_steps`. `dense_update_flops` counts two operations
per executed multiply/subtract term, including small panel and sparse
intersection work; diagonal weighting, divisions, and operand packing are
excluded.
`largest_front` is the actual largest filled node height on either SPD path;
`spd_panel_entries` is the largest allocated dense node rectangle. A rectangle
uses less than twice its node's lower-triangular CSC entries, and does not
densify unrelated matrix rows or columns. `peak_workspace_bytes` estimates
simultaneously live numeric and symbolic vector buffers, including CSC factors,
row incidence, packed operands, and the node rectangle; thread runtime storage
is excluded.

`spd_source_seconds` measures prior-node updates including intersection lookup
and packing; `spd_panel_seconds` measures node factorization and trailing
updates. `spd_pack_seconds` is the dense-buffer packing portion of those two
times, so it must not be added to them. `failed_spd_seconds` records a discarded
SPD attempt, and is included in the subsequent LU numeric/factor time. Peak
workspace also includes that attempt's maximum live buffers.

Good threading candidates have wide, irregular fronts and substantial fill.
Banded and narrow PDE-like problems usually select a low-overhead serial CSC
engine. Wider SPD PDE fronts can use blocked parallel updates. Factor solves
parallelize across RHS blocks; a single RHS solve is serial.

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
- general LU retains serial pivot selection and pivot-tail materialization;
  dense-row panels still retain each dense row's span-wide tail in the row
  engine;
- dense-row classification measures density over a row's whole remaining
  span, so a row that is dense in a prefix but has one far column (for
  example, a bordered system ordered border-last) has a wide span: it stays
  sparse, or its dense tail covers that whole span;
- there is no out-of-core mode or symbolic-factor reuse across matrices with
  the same pattern.

Default promotion remains contingent on repeated release-build results on both
Windows and Linux, as described in the benchmark notes. The native backend
remains opt-in.
