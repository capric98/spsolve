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
`[first, last)` current, while deferring updates at or beyond `last`. The
panel path (`DensePanel` in `src/sparse_lu_panels.hpp`) maintains these
invariants:

- **Retained pivot rows.** While a panel is active, a dense row selected as
  pivot keeps its dense storage until the panel's flush and is not modified
  after its pivot step. Later eager in-panel updates read its coefficients in
  place with one `axpy_sub` per dense target, over
  `[pivot + 1, min(last, end of its storage))`; a sparse pivot supplies its U
  entries as before.
- **Contiguous materialization.** Before a dense row becomes a pivot, its
  deferred tail is completed from its L multipliers in ascending pivot
  order: one `axpy_sub` over `[last, tail_end)` for each retained dense
  source, where `tail_end` is one past the source's last nonzero, and a walk
  over the U entries of a sparse source. U is therefore complete before it
  supplies any subsequent update, even when row swaps cross panel
  boundaries.
- **Lazy U rows.** The U row (`Entry` vector) of a retained dense pivot is
  built at the flush, or earlier, serially and before the workers start,
  when a sparse row needs it as an update source. Sparse pivots publish
  their U rows at once, and dense pivots outside a panel build theirs at the
  pivot step. The flush finalizes the panel even when nothing is left to
  update (the last panel never has targets): it builds the remaining U rows
  and releases the retained storage.
- **Segment packing.** The packed U support is the union of the panel
  pivots' tails at or after `last`, as sorted disjoint column segments. A
  dense pivot contributes `[last, tail_end)`, found by a backward scan of its
  retained row, so the dense pivots together form one segment starting at
  `last`; a sparse pivot contributes its tail columns. Gaps narrower than 16
  columns are bridged with zero U columns. Dense pivots are copied into the
  packed U rows segment by segment, sparse pivots scattered. Before packing,
  every target is extended serially to the end of the union, so each segment
  is contiguous in every target.
- **Tiling.** Independent tiles of 16 target rows by at most 64 columns of
  one segment call `gemm_sub_rows` in place on the targets' dense tails.
  Every output starts from its current value and subtracts its `rank` terms
  in ascending pivot order (`running`). Segments and tiles depend only on
  column positions, never on the thread count. The flush uses the worker
  team when more than one thread is requested, there are at least four
  tiles, and the counted work is at least 262,144 flops.
- **One rounding rule.** Every path that updates a dense entry (eager
  in-panel updates, materialization, the flush, and dense-pivot updates
  outside panels) computes `c = c - round(l*u)` with separate rounding, in
  ascending pivot order, so two identical rows stay bitwise identical
  whichever path their updates take; exact singular detection of a
  duplicated row depends on it. Columns whose packed U values are all zero
  (bridged gaps, zeros inside a dense tail) can change at most the sign of a
  zero entry, which is never stored as a factor entry or accepted as a
  candidate.

Finiteness is checked for every candidate value and pivot, every L
multiplier, every value a sparse row update writes, every kernel call
(`axpy_sub` and the flush GEMM report whether any element they wrote is
non-finite, and a false flag raises the usual error), and every U row built
from dense storage. The scalar U-entry walks of sparse sources during
materialization and of sparse pivots during eager panel updates are not
checked per element. The checks are complete: suppose elimination stores a
non-finite value in column `c` of row `r`, which is later pivoted at position
`pos(r)`. If `c >= pos(r)`, the value belongs to `r`'s U row, which is
checked when it is built, before any factor is published. If `c < pos(r)`,
the value is still present at step `c`: non-finite operands never produce
finite results, columns at or before the current pivot are only written to
clear an already checked coefficient, and deferred updates to column `c` are
applied before step `c` (a flush precedes the next panel, and a panel keeps
its own columns current). A non-finite value is nonzero, so `r` is a
candidate at step `c` and the candidate check rejects it. Every row is
pivoted, so no value escapes both checks. Detection can come later than the
arithmetic step that produced the value; if an exactly singular column is
reached in between, the singular error is reported instead. In either case
no factor is published. Input values are validated separately when
`check_finite` is set.

Sparse-to-dense conversions occur only after the flush, so conversion
neither omits nor duplicates an update. After activation, short trailing
panels retain the same recurrence. The activation thresholds are independent
of the requested thread count. Purely sparse phases keep their existing
update scheduler and allocate no panel state. Eliminated rows never
participate in a later flush.

## Dense kernels

Both blocked paths call one kernel table, `dense_kernels::Kernels`
(`src/dense_kernels.hpp`): `gemm_sub` (`C -= A @ B`), `gemm_sub_rows` (the
same with per-row output pointers), `gemm_sub_lower` (writes only entries with
`i + shift >= j`), and `axpy_sub` (`y -= alpha * x`, one `running` term per
element; the row-hybrid LU uses it for every update whose source is a dense
pivot row: eager in-panel updates, pivot-tail materialization, and
dense-row updates outside panels). The GEMM entry points take an
accumulation order:

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
on its tile, path, or thread. Three kernels implement the table:

- `scalar`: portable C++, one output row at a time.
- `sse2`: the baseline x86-64 kernel. It generalizes the earlier 4-by-4
  microkernel to 1-4 rows by 4, 2, or 1 columns, so edges use the same
  arithmetic, and reproduces the earlier LU and LDL^T results bit for bit.
  `scalar` performs the same operations and, on x86-64, produces the same
  bits. Both round each product and each addition or subtraction separately
  (two roundings per term).
- `avx2-fma` (`src/dense_kernels_avx2.cpp`): a 4-by-8 microkernel with eight
  YMM accumulators. Per `k` it loads two YMM of `B`, broadcasts four elements
  of `A`, and issues eight multiply-subtract pairs (`running`) or eight fused
  multiply-adds (`sum_then_subtract`). Edges use 1-4 rows by 8, 4, 2, or 1
  columns with the same per-element operations and read nothing outside the
  requested rows and columns. `gemm_sub_lower` computes a microtile that the
  diagonal crosses in an aligned 4-by-8 scratch block holding only the
  entries on or below the diagonal, stores back only those entries, and
  tests only those for finiteness. On an i5-13600K, 16-by-64 rank-32 LU flush
  tiles run at about 52 GFLOPS (`sse2`: 21) and off-diagonal 16-by-16 LDL^T
  tiles at about 60-64 GFLOPS (`sse2`: 18-22) for ranks 32-320. Diagonal
  16-by-16 tiles (`gemm_sub_lower`) reach about 35-48 GFLOPS (`sse2`: 16),
  counting only the entries written, because of the scratch gather and
  scalar scatter of their diagonal-crossing microtiles.

`avx2-fma` fuses only the `sum_then_subtract` order: `s = fma(a_k, b_k, s)`
rounds once per term, followed by the single subtraction `c - s`. Supernodal
LDL^T factors therefore differ from the `sse2` and `scalar` ones in the last
bits (normwise about 1e-16 on the benchmark SPD cases) with the same backward
error (for example 6.7e-16 on four dense 600-wide SPD blocks with every
kernel). The `running` order, and `axpy_sub`, deliberately round twice per
term, so LU factors are bitwise identical under all three kernels. The reason
is the row-hybrid recurrence above: a dense row's tail receives each panel
term either from the flush or, when the row becomes a pivot first, from
pivot-tail materialization (`DensePanel::materialize_pivot`), and both must
round identically. With a fused flush, a dense row duplicated from a pivot
row kept a residue near 1e-18 instead of cancelling to zero, and the exactly
singular matrix factored instead of raising. Materialization and eager
updates from dense pivot rows call `axpy_sub`, but terms from sparse pivot
rows are applied by scalar U-entry walks, so fusing `running` would require
those walks to round the same way as every kernel.

Every kernel reports whether any element it wrote is non-finite. The SIMD
kernels fold `x - x` of every stored register into one witness per microtile
(zero for finite values, NaN otherwise) and test it once, so every written
element is covered. A kernel may stop at its first non-finite block, leaving
later elements uncomputed, so its flag is authoritative on every path: a
false flag raises the usual LU non-finite error or rejects the SPD attempt
before any gathered tile is scattered back, and no path scatters an element
the kernel was not asked to compute. A kernel never reports an element it
did not write: the diagonal-crossing blocks of `avx2-fma` compute entries
above the diagonal in scratch but neither store nor test them.

A factorization snapshots the active kernel once, before ordering, and passes
it to both engines; `stats["dense_kernel"]` names it (`avx2-fma`, `sse2`, or
`scalar`) even when the selected engine performs no tile updates. The default
is the best kernel the build and CPU support: `avx2-fma` when
`cpu_supports_avx2_fma` finds AVX, FMA, AVX2, and OSXSAVE in CPUID and XMM
plus YMM state enabled in XCR0, otherwise `sse2` on x86-64 and `scalar`
elsewhere. `SPSOLVE_FORCE_KERNEL` and `spsolve.spparms.set_dense_kernel`
override it (see `docs/api.md`). Within one kernel, factors are bitwise
identical at every thread count; across machines they are bitwise identical
only when the same kernel runs, so force a kernel (for example `sse2`, which
every x86-64 CPU supports) to reproduce LDL^T factors across machines.

Wider kernels live in separately compiled translation units reached only
through the kernel table after the CPU check. CMake compiles
`src/dense_kernels_avx2.cpp`, and only that file, with `/arch:AVX2` (MSVC) or
`-mavx2 -mfma` (GCC/Clang; `-Xarch_x86_64` in an Apple universal build) when `SP_ENABLE_AVX2` (default `ON`) is set and the target
is x86-64; it then defines `SPSOLVE_HAVE_AVX2_KERNELS=1`. The rest of the
module, including wheels, targets baseline x86-64. Hygiene rules for such a
translation unit, also stated in its header comment:

- it includes only `<immintrin.h>`, typedef-only standard headers, and the
  declarations-only `dense_kernels.hpp`: an inline function or template
  instantiated there would be emitted as a COMDAT with VEX encoding, and the
  linker could keep that copy for baseline callers;
- every helper has internal linkage, and only the kernel table and its entry
  points are external;
- the table is `constinit`, so loading the module runs none of its code;
- link-time code generation (`/GL`, `-flto`) stays off, and `vzeroupper`
  stays enabled (never `-mno-vzeroupper`);
- it relies on the module-wide `-ffp-contract=off` (below) to keep GCC from
  fusing the separate multiply and subtract of the `running` order into FMA;
  the intended FMA intrinsics are unaffected.

The whole module is built without floating-point contraction
(`-ffp-contract=off` target-wide on GCC/Clang; MSVC does not contract under
its default `/fp:precise`), so every scalar and vector path, including the
eager row updates and pivot-tail materialization in `sparse_lu.cpp` and
`sparse_lu_panels.hpp`, rounds `a*b` and the following subtraction
separately, even with `SP_ARCH_NATIVE` on an FMA host. The exact cancellation
of duplicated rows described above depends on it.

Disassembling a Windows release module (`dumpbin /disasm` with a linker map)
showed YMM and VEX-encoded instructions only in functions from
`dense_kernels_avx2.cpp`, apart from Microsoft's STL vector algorithms
(`msvcprt:vector_algorithms.obj`, called from `sparse_lu.cpp` and
`spsolve.cpp` before this kernel existed), which check the CPU themselves.

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
support size: the total width of a flush's column segments, including
bridged gap columns and zero columns inside dense pivots' tails (see "Dense
LU panels"). `lu_dense_update_flops` counts two operations per packed
multiply/subtract term, including zeros in packed operands (bridged and zero
columns as well); eager panel coefficients and pivot-tail materialization
are excluded. `lu_panel_workspace_bytes` estimates peak panel-owned vector
capacities, including discovery marks, row lists, per-pivot panel state,
segment and tile lists, and packed L/U operands. It excludes retained
factors, row-engine sparse/dense storage (including pivot rows retained
until a flush), and worker runtime storage, and must not be interpreted as
total row-engine peak memory.
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
- overflow checks during scaling and row updates (the dense panel path checks
  candidates, multipliers, kernel results, and retained U rows; see "Dense
  LU panels");
- componentwise iterative refinement with final-error and convergence checks.

This is not a rank-revealing factorization and does not estimate condition
numbers. Very ill-conditioned matrices can still lose forward accuracy even
when their backward residual is small.

## Current limitations and next optimization target

- native matrix values are `float64` only;
- no complex native factors yet (the SciPy backend handles them);
- ordering and pivot selection are single-threaded;
- general LU retains serial pivot selection, eager in-panel updates, and
  pivot-tail materialization, each touching separately allocated dense rows;
  dense-row panels still retain each dense row's span-wide tail in the row
  engine, and a dense pivot's packed U range runs from the panel end to its
  last nonzero, interior zero runs included;
- dense-row classification measures density over a row's whole remaining
  span, so a row that is dense in a prefix but has one far column (for
  example, a bordered system ordered border-last) has a wide span: it stays
  sparse, or its dense tail covers that whole span;
- the `avx2-fma` kernel does not fuse the LU flush (`running` order): the
  scalar U-entry walks of sparse pivot rows (eager panel updates and
  pivot-tail materialization) apply the same terms outside the kernel table
  and must round identically, so dense LU gains only from the wider vectors
  (see "Dense kernels");
- there is no out-of-core mode or symbolic-factor reuse across matrices with
  the same pattern.

Default promotion remains contingent on repeated release-build results on both
Windows and Linux, as described in the benchmark notes. The native backend
remains opt-in.
