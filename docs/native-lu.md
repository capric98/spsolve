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
2. C++ validates CSC bounds and optional finiteness, and in the same pass
   accumulates each row's largest magnitude for equilibration.
3. A column ordering is selected:
   - `NATURAL` preserves input order.
   - `COLAMD` uses the vendored COLAMD routine on the input pattern. Engine
     selection uses actual numeric work in this permutation; there is no
     second ordering pass.
   - `MMD_AT_PLUS_A` uses the vendored AMD routine on the structural graph of
     `A + A.T`. AMD's fill and maximum-front estimates also guide numeric
     engine selection.
4. Rows and columns are equilibrated by infinity norms. Structural zero rows or
   columns are reported as singular before numerical elimination; rows are
   checked first, then columns, each in natural order. Column maxima use
   four interleaved running maxima per column, which is exact (the
   maximum ignores NaN terms either way, so the scales are bitwise those of
   one running maximum).
5. The factorizer chooses one of three numerical engines:
   - An exactly symmetric matrix first attempts a sparse `LDL.T`
     factorization with symmetric equilibration, using whichever column
     permutation the caller requested as a symmetric permutation. Symmetry
     requires matching transpose entries with exactly equal values in the
     canonical input. Unequal entries, even one floating-point step apart,
     use general LU so that neither triangle of the input is discarded. The
     check is one O(nnz) transpose walk: visiting columns in order, each
     entry `(i, j)` must match the next unconsumed entry of column `i`,
     which must hold row `j` and an equal value; it stops at the first
     mismatch.
     Positive pivots certify the SPD path. Its factors are exposed through the
     common LU contract as `L` and `D @ L.T`; internally it retains only the
     strict lower triangle and diagonal, constructing public `L` and `U`
     lazily. Narrow fronts use the simplicial CSC kernel. Adjacent elimination
     tree parents and column counts give the exact fundamental nodes, their
     widths and their work, before any row index is formed. The blocked path
     described below requires nodes at least 320 columns wide with
     sufficient combined work. Ten 32-column panels amortize packing, while
     the work guard prevents a small wide node from routing overwhelmingly
     narrow matrices through the panel driver.
     A non-positive pivot or non-finite
     intermediate falls back to pivoted LU with the original equilibration.
   - A nonsymmetric matrix with `MMD_AT_PLUS_A` uses a Gilbert-Peierls-style
     left-looking CSC LU when AMD estimates a maximum front at most 192 and
     fill at most 20. `COLAMD` and `NATURAL` instead start in CSC with their
     actual pivots (the probe): a column reach above 192, more than 8,192
     scalar updates in a column, more than `64 * nnz(A)` total updates, or
     factor storage above `20 * nnz(A)` triggers a restart in the row
     engine. The discarded probe is bounded, and restart preserves the
     requested column permutation and equilibration. `NATURAL` does not
     start the probe when an input column is wider than 192 rows or a row
     would be stored dense by the row engine's setup; its probe also
     restarts after a column with more than 64 multipliers, and its
     multiplier and per-column limits apply only while the work they
     predict for the next `min(n - k - 1, 192)` columns exceeds four times
     the updates done so far; and after `max(32, n/16)` columns it
     restarts once its updates, extrapolated to all columns, exceed
     `64 * nnz(A)` (see "Engine selection for `NATURAL`" below). Both
     engines perform threshold partial
     pivoting. See "Left-looking CSC LU" below for the search order and the
     storage reserve of the CSC engine.
   - Wider or fill-heavy patterns use the row-hybrid LU. Sorted sparse rows
     with at least 64 entries convert to dense tails when at least 25% of
     their own remaining span, from first to last remaining nonzero column,
     is nonzero. The tail is allocated for that span with its end rounded up
     to 64 columns, and grows on demand; columns outside it are exact zeros.
     Later blocks of a block-diagonal matrix therefore carry no leading zero
     columns, and pivot candidate scans skip dense rows whose storage
     excludes the pivot column. Setup counts each row's scaled nonzeros and
     their first and last factor columns in one pass over the permuted
     input, then applies the same dense test to those counts: a dense row's
     tail is allocated and filled directly, and a sparse row's sorted entry
     list is allocated at its exact size. Rows dense at setup never enter the
     per-column candidate lists; they are found through the dense row list,
     or the panel block inside a panel.
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

## Left-looking CSC LU

The `left-looking-csc` engine is serial and factors one column at a time
(Section 4 of `docs/algorithms.md`). For factor column `k` it scatters the
scaled input column; finds the reach of the column's input rows through the
columns of `L` factored so far by a depth-first search, which writes the
reached rows in topological order; applies the update of each reached
pivoted row in that order; picks the pivot among the reached unpivoted rows;
and appends `U`'s column (the reached pivoted rows with nonzero values, in
reach order, then the pivot) and `L`'s column (the pivot row with 1, then
the nonzero multipliers, in reach order). Exact zeros are not stored in
either. The scatter can precede the search because the dense work vector is
zero between columns: only rows of a column's reach are written, and the end
of every column resets its whole reach (earlier builds also cleared the
reach before scattering).

**Search order.** Each column's search starts from the column's pivoted row
of smallest pivot position, then from the column's rows in stored order,
skipping rows already reached (an unpivoted row is a leaf). Unpivoted rows
hold position -1, so the start is the minimum of the positions as unsigned
values, found without a branch in the pass that scatters the column, before
any search. Both kinds of start go through one loop with a single call of
the search (an extra first iteration stands for the first root). A first
version of this change called the search separately for the first root, and
that build was 14-28% slower in the numeric phase on a nonsymmetric
tridiagonal matrix of 200,000 rows than earlier builds; with one loop (and
the scatter moved into the same pass) the difference is 0-7% (below). The
work is the same in both versions, so most of that cost came from the
generated code.
In a band that first search typically reaches every other pivoted row of the
column, so the traversal is the same from column to column whatever the row
labels are. Earlier builds started from the rows in stored order, which follows the
row labels: on the row-permuted 20,000-row band of the benchmark, an
instrumented one-thread build spent 5.9 ms in the search against 2.8 ms for
the same band with its rows in order, with the same reach and the same
1,259,452 edges examined (one per update). Starting from the smallest
position took the search to 3.4 ms, with the same edges and 4% more search
steps, so the gain is in how regularly the work runs, not in its amount.
The order depends only on the matrix (its stored row order) and the pivots
so far.

The scratch prototype that preceded this change started from every pivoted
row in increasing position (a sort of the column's pivoted positions) and
then from the unpivoted rows. In a scratch build that switches between the
orders (one thread, medians of three interleaved rounds), that was as fast
as the smallest-position start on the row-permuted band (numeric phase 7.0
against 6.8 ms, both with the earlier search formulation), but on random
full bands with pivoting (half-width 3, 8 and 12; 50,000, 20,000 and 10,000
rows; rows permuted) it was 16-18% slower than the stored order: the sort
and the second pass cost more there than the order saves. With the search
formulation below the sorted start was 14-23% slower than the stored order
on the full bands of half-width 3 and 8 (permuted or not) and 11% slower on
`banded-nonsym` with `MMD_AT_PLUS_A`, whose stored order already follows the
pivots. How the smallest-position start compares with the stored order on
such inputs is within the code-generation variation between builds (5-10%
either way in this and an independent set of measurements). The start order
does cost on columns with a single pivoted row in a pattern that the
smallest-position start rearranges: a nonsymmetric tridiagonal matrix of
200,000 rows, which `MMD_AT_PLUS_A` orders in reverse, takes 6-7% longer in
its numeric phase than in earlier builds and 1.5-3% longer end to end
(interleaved runs, including an independent review's), while with `COLAMD`
it is unchanged. In a scratch build
with switches the start order alone cost 3-5% on that matrix with either
ordering, and arranging the first root's search, the minimum and the
scatter differently (separate passes, a separate call, scattering after the
search) moved it by at most 3%. Every other left-looking family measured is
faster than in earlier builds (see "Row-permuted band" in
`docs/benchmarking.md`).

**Search formulation.** The search keeps the current row's factor column and
edge range in locals and pushes only suspended pivoted rows (their factor
column and next edge). An unpivoted row reached by an edge has no edges of
its own, so it is written to the reach at once instead of being pushed,
reloaded and popped. This writes every row exactly where the earlier
formulation (one stack entry per reached row, whose state was reloaded from
the row on every resume) wrote it: on 130 stress matrices with both
orderings the factors were bitwise identical to those of the earlier
formulation with the same start order (both the stored and the
smallest-position start). It cut the one-thread numeric phase
by 11-24% across the families measured with the stored start order (for
example 10.2 to 8.3 ms on the row-permuted band, 6.0 to 4.6 ms on the same
band with rows in order, and 13.1 to 10.8 ms on the full band of half-width
8).

**Effect on the factors.** The search order decides the order in which
updates reach an entry, so `L` and `U` differ from earlier builds in the
last bits (on the benchmark bands by at most 2.2e-16 of the largest `U`
entry, with identical permutations and patterns), and a pivot can change
where two candidates are equal or within rounding error of each other,
including the preferred row against the threshold: ties are still broken by
current row position, but rounding may now separate the candidates the
other way. Exact singularity is still an exactly zero computed pivot (see
Section 3 of `docs/algorithms.md`), so for a matrix that is singular to
working precision, whether the factorization meets an exact zero can change
in either direction. Two equal rows still cancel exactly in any order
(Section 4 of `docs/algorithms.md`), so duplicated rows are always reported
singular; duplicated columns and other integer combinations of rows or
columns are not covered by that argument. Non-finite detection is unchanged:
every candidate and update is still tested. Since the probe's pivots can
change, so can its decision and its counters.

Two stress sets measured these effects against `c46bda9`, with both
orderings at several thread counts: the comparisons ran at one and four
threads, and this change was also run at 1, 2, 3, 4 and 8 threads on 65
matrices of the first set and at 1, 2, 3 and 8 threads on the whole second
set, with identical factors at every thread count. Their counts describe
those sets, not bounds. The first has 260 matrices (narrow random bands with uniform,
`+-1` and small-integer values, near-bands, near-singular matrices and
matrices with a duplicated row or column or a row equal to a sum of two
rows; 356 left-looking factorizations). In it 7 row permutations changed,
all in `+-1` or `+-1, +-2` bands, each at a column whose largest candidates
are exactly equal in exact rational arithmetic; 21 factors kept their
permutations but stored a different set of entries, because a value
cancelled to exactly zero in one order and not in the other (exact zeros are
not stored, and a dropped entry can also remove later fill), with factor
sizes within 0.5%; one duplicated-column matrix with `MMD_AT_PLUS_A` that was
reported singular now factors (the componentwise backward error of an
unrefined solve is 1.6e-10), and none became singular; the probe decision
was the same on every matrix, and 1 of 14 restarts reported different probe
counters (`numeric_probe_max_front` 318 instead of 304,
`numeric_probe_updates` 18,547 instead of 18,668). The second, an
independent set of 355 matrices (similar bands, convection-diffusion grids,
duplicated rows and columns, integer combinations of rows or columns,
planted NaN or infinite entries with `check_finite=False`, and columns and
full bands around the probe's limits; 383 left-looking factorizations), had
25 permutation changes and 54 pattern changes (factor sizes within 1.1%). An
independent review recomputed its permutation changes in 120-digit
arithmetic: they were at exact ties (one of them between the preferred row
and the threshold), at a near tie (relative gap 2.6e-17), and in numerically
singular matrices among candidates of size about 1e-32. Ten factorizations
that completed before now
report singular (six with a duplicated column, three with a row combination,
and a random band with `MMD_AT_PLUS_A` whose earlier factors had a backward
error of 0.21), and five reported singular before now complete; all fifteen
matrices are singular to working precision (smallest singular value at most
6.2e-17 of the largest). Duplicated rows were reported singular in all 64
cases. Probe decisions were the same, 2 of 40 restarts reported different
probe counters, and the planted NaN and infinite entries raised the same
errors and messages as before.

**Storage reserve.** The four factor arrays (`L` and `U` row indices and
values) are reserved in two stages. Each first gets `1.15 * E + n` entries,
where `E` is the larger of `2n` and AMD's estimate of `nnz(L)`
(`MMD_AT_PLUS_A` only), as in earlier builds. After the first
`max(256, n / 16)` columns, the sizes of `L` and `U` so far are extrapolated
to `n` columns, and an array whose extrapolated size exceeds its capacity is
reserved for 1.15 times that size plus `n`, but never beyond the ceiling
`1.15 * max(E, 2 * nnz(A)) + n` entries. No reserve exceeds a dense triangle,
`n * (n + 1) / 2` entries, which no array of `L` or `U` can exceed. Past its
reserve an array grows geometrically as before (1.5-fold with MSVC, doubling
with libstdc++ and libc++). The second stage depends only on `n`, the factor
so far and the current capacity; through the capacity, the standard
library's growth policy can decide whether the extension happens, which
affects memory only, never the factors. A probe that is certain to restart
(an input column alone is wider than the 192-row reach limit) skips the
second stage, since its arrays are discarded (with `NATURAL` such a probe
does not start at all); the ceiling stays below the
probe's factor limit of `20 * nnz(A)` entries. With `COLAMD` (no estimate)
earlier builds started each array at `3.3 * n` entries and grew it 1.5-fold,
three times for each array of the row-permuted band (159,953 and 199,925
entries in `L` and `U`), touching fresh pages at every step. A first
version of this change reserved the ceiling at once, which avoided that but
left factors that need less with most of it unused, and inflated the
discarded arrays of probes that restart.

Bound: since `n <= nnz(A)` and no reserve exceeds the ceiling, an array
that ends within its reserve holds, without an AMD estimate, at most
`3.3 * nnz(A)` unused entries: at most `4 * 8 * 3.3`, about 106 bytes per
input entry, for the four arrays of 8-byte entries. That is the same bound
as for the `3.3 * n` initial reserve of earlier builds. An array that
outgrows its reserve holds instead the slack of its last geometric step, at
most half its entries with MSVC's 1.5-fold growth (all of them with the
doubling of libstdc++ and libc++), as before. With `MMD_AT_PLUS_A` the
initial reserve from AMD's estimate dominates as before (the engine is
selected only for an estimate of at most `10 * nnz(A)`). Unused capacity is
never touched, so it is committed on Windows but not resident (address
space only on Linux).

Measured with `COLAMD`, one thread, each case in a fresh process (committed
memory held by the factor after the call, the peak of committed memory
during it, page faults):

| matrix | earlier builds | ceiling reserved at once | this change |
|---|---|---|---|
| nonsymmetric tridiagonal, n=1,000,000 (`L`, `U` 2,000,000 entries each) | 185 MB held, 262 MB peak | 326 MB, 402 MB | 185 MB, 262 MB |
| upper triangular with 10 entries per column, n=1,000,000 (`L` the unit diagonal) | 385 MB, 519 MB | 898 MB, 975 MB | 406 MB, 508 MB |
| full band of half-width 95, n=50,000 (the probe restarts; row engine) | 336 MB, 342 MB | 336 MB, 789 MB | 336 MB, 342 MB |
| row-permuted band, n=1,000,000 | 448 MB, 581 MB, 267,000 faults | 489 MB, 566 MB, 144,000 faults | 455 MB, 557 MB, 149,000 faults |

The upper triangular factor holds 21 MB more than in earlier builds because
`U` is reserved for 1.15 times its extrapolated size instead of ending after
1.5-fold growth slightly above it; its numeric phase took 105 instead of
143-181 ms. `peak_workspace_bytes` does not count the factor arrays of a
completed left-looking factorization (they are the result) and is unchanged
there. On a probe restart it counts the discarded arrays at their capacity,
as before. Their capacity now differs from earlier builds only when the
probe passed the second stage, where it follows the extrapolation rather
than 1.5-fold growth from `3.3 * n`: on the benchmark cases only pivot-heavy
with `COLAMD` changed (0.50 instead of 0.42 MB), and over the 54 restarts of
the two stress sets above it stayed the same on 24, rose on 22 (at most 1.54
times) and fell on 8 (to 0.86 times), against up to 36 times with the
ceiling reserved at once; the full band above reports 10.0 MB in both
builds, against 676 MB with the ceiling reserved at once. The restart
releases the discarded arrays before the row engine starts, as before.

## Engine selection for `NATURAL`

`NATURAL` runs the bounded probe of `COLAMD` in the identity column order:
the same reach, per-column, total-update and factor-storage bounds, the same
restart from the pristine permutation and equilibration, and the same
pivoting rule (threshold partial pivoting with `diag_pivot_thresh`,
preferred row `perm_c[k] = k`, ties by current row position). Exactly
symmetric input still tries the SPD path first, and `MMD_AT_PLUS_A` keeps
its estimate-based selection. Earlier builds always used the row engine for
`NATURAL`, which keeps sparse rows in sorted entry lists and pays a merge per
update: narrow nonsymmetric bands took 2.2-2.4 times SuperLU's time
(`banded-nonsym` and the row-permuted band of the benchmark, which the CSC
engine now factors at 0.61-0.63 times; see `docs/benchmarking.md`). The
measurements in this section were taken with Python 3.15.0, NumPy 2.5.3 and
SciPy 1.18.1 unless stated otherwise.

The scalar CSC loop is not faster than the row engine everywhere. The row
engine stores a row densely once it has at least 64 entries filling a
quarter of its span, and then updates it with the dense kernels and,
inside panels, on the worker team. Measured with `NATURAL` (one thread; a
scratch build that runs the CSC engine to completion, without the pattern
check, the `NATURAL` rules and the update budgets, against the row engine
of `4f1885e`; medians of three interleaved rounds):

| matrix | CSC engine / row engine |
|---|---:|
| full band, half-width 30 (61 entries per row), n=20,000, pivoting | 0.52 |
| same, half-width 32 (65 entries per row) | 1.24 |
| same, half-width 64, n=6,000 | 2.74 |
| diagonally dominant band, half-width 30 / 32 / 40, n=20,000 / 20,000 / 6,000 | 0.48 / 1.16 / 1.65 |
| dense block of 63 / 64 / 96 / 128 rows | 0.61 / 1.40 / 1.94 / 3.00 |
| 100 dense blocks of 56 / 60 blocks of 64 / 20 blocks of 128 | 0.55 / 1.59 / 2.89 |
| 96 sparse rows (21 entries) whose elimination fills a dense block of 64 / 96 / 128 / 160 rows | 0.60 / 0.89 / 1.14 / 1.58 |
| 2-D convection-diffusion grid, side 20 / 24 / 32 / 45 | 0.46 / 0.45 / 0.64 / 0.71 |

The crossover is the row engine's dense-row threshold, not the size of
the front: a band whose rows have 61 entries runs at 0.52, one with 65
entries at 1.24. The grids' rows stay below it up to side 45, and the probe
bounds stop them anyway (below). `NATURAL` therefore adds rules of its own
to the probe, none of which applies to `COLAMD` (whose results stay bitwise
those of earlier builds):

- **Pattern check before the probe.** The probe does not start when an
  input column has more than 192 entries (its reach would exceed the
  limit, so the probe would certainly restart; an `O(n)` scan of the
  column pointers), or when some row has at least 64 stored entries filling
  at least a quarter of the span from its first to its last column, the
  row engine's setup dense test (one pass over the row indices, about
  0.6-0.8 ns per entry: 0.8 ms for the 1.3 million entries of the band of
  half-width 32). The row engine counts scaled nonzeros where this check
  counts stored entries; the two differ where a scaled product underflows
  to zero, and for explicitly stored zeros, which the public `splu`
  removes but a direct call of the private `_splu_factorize` can pass. The
  dense LU families and `sparse-dense-border` of the benchmark take the
  first branch (under 0.01 ms); bands with rows of 64 or more entries and
  dense blocks of 64 to 192 rows the second. A skipped probe reports
  `numeric_probe_fallback` with zero columns, front and updates, and the
  check's time as `numeric_probe_seconds`.
- **Multiplier limit, and when a restart pays.** A column with more than 64
  multipliers (entries of `L` below the diagonal) signals fill that makes
  the row engine's rows dense during the factorization, where its dense
  kernels win (the filled blocks of 96 rows and more above). Such a column
  `k`, or one that passes the per-column limit of 8,192 updates shared
  with `COLAMD`, restarts the probe only when the work it suggests for the
  next `h = min(n - k - 1, 192)` columns exceeds four times the updates
  done so far, `D`, which a restart discards: `m^2 * h` for `m`
  multipliers, or `c * h` for the column's `c` updates up to and including
  the source that passes the limit (which `D` counts too). At the start of
  a block that fills, `D` is small and the probe restarts at once: the
  blocks of 96 to 160 rows above restart at their first column, after 97
  columns, at 0.94-1.04 times their earlier time (with the shared bounds
  alone their probe ran almost to the end of the block, 1.35-1.71 times,
  measured with Python 3.14). Behind a long accepted band, `D` is large
  and the probe finishes the block in CSC. An absolute limit restarted
  there and discarded the whole band: the benchmark band of 20,000 rows
  followed by 96 sparse rows filling a block of 96, 128 or 160 rows, and
  full bands of half-width 16 and 30 followed by the 96- and 128-row
  blocks, took 1.25-1.59 times their earlier time in one-process scans
  (1.27-1.57 in the review's fresh-process runs), and now take 0.28-0.60
  times (fresh processes, five rounds). The factor 4: for a trailing dense
  block of `b <= 193` rows the estimate at its first column, `(b - 1)^3`,
  is about three times the block's remaining work of about `b^3 / 3`
  updates (a 63-row block takes 81,375; `63^3 / 3` is 83,349), so the probe
  finishes such a block while its remaining work stays below about 4/3 of
  `D`. A restart redoes the prefix in the row engine, which takes 2-4 times
  longer per update on the band's sparse rows than this loop. The factor
  was chosen by measurement: with factor 1 the benchmark band followed by
  blocks of 112 to 160 rows still restarted (1.26-1.36 times); factors 4
  and 16 finish all of them (0.34-0.40).

  The horizon `h` stops at the reach limit: it assumes the fill is a block
  no wider than the fronts the probe accepts. Counting all `n - k - 1`
  remaining columns, as the rule first did, treated every column after a
  block in the middle of the matrix as fill, and those inputs restarted
  late. A band of half-width 30 (20,000 rows), the 96-row filled block and
  15,000 tridiagonal rows took 1.53-1.56 times the earlier time; with the
  benchmark band (20,000 rows) after the block instead, 1.45-1.47; with
  half-width-30 bands of 10,000 rows on both sides, 1.29; and with the
  benchmark band on both sides (10,000 or 20,000 rows each), 1.16-1.18.
  With the horizon they finish in CSC at 0.53-0.60 and 0.28-0.30 times
  (fresh processes, five rounds, one and eight threads). The cost falls on
  fill that lasts longer than 192 columns behind a long cheap prefix. The
  estimate no longer sees it, so the probe continues until the
  projected-work rule stops it, and discards more. Behind the benchmark
  band (20,000 rows), each of these took 1.2-1.6 times the earlier time
  (best of five fresh-process rounds) against 1.1-1.2 with the full count:
  a band with entries at offsets ±1 and ±66 or ±80 (1,000 to 8,000 rows,
  more than 64 multipliers per column once it fills); 16 filled blocks of
  128 rows or 8 of 160 rows (each behind its 96 sparse rows); and 2-D grids
  of side 66 and 80. In rounds where the row engine ran in its slower
  allocation mode, the ratio reached 1.7 with either form and 2.0 with the
  cap (the same row-engine work takes 43 or 60 ms in different processes
  for some of these; see `docs/benchmarking.md`).

  The per-column part matters behind a band of half-width 16 or 30 and a
  128-row block, whose late columns pass 8,192 updates after the multiplier
  test let the block continue. Its restart side decides only where a
  column passes 8,192 updates while no column before it had more than 64
  multipliers. The tests construct one (129 columns of 64 multipliers
  feeding one column). A build in which that side never restarts gave
  identical factors and probe statistics on the benchmark matrices, the
  284 stress matrices below and 286 scratch families of this section
  (bands, blocks, grids, triangular and random patterns); for them, the
  per-column limit only relaxes the shared bound.
- **Projected work.** After `max(32, n/16)` columns, the probe restarts once
  its scalar updates so far, extrapolated to all `n` columns at their
  average per column, exceed the total budget of `64 * nnz(A)`. A grid's
  work per column is steady after its first rows, while the budget itself
  is reached only after most of its columns: the 2-D grid of side 24 with
  the shared bounds discarded 336 of its 576 columns (178,155 updates,
  0.4 ms of a 1.9 ms factorization), and now 59 columns (18,435 updates,
  0.04 ms). To stop a probe that would have completed within the budget,
  a prefix must cover at least `n/16` columns and do more than
  `4 * nnz(A)` updates there.

`tests/test_sparse_lu.py` pins each rule at its threshold:

- a row of 64 entries spanning 256 or 257 columns;
- fill-induced blocks of 65 and 66 rows;
- the 66-row block behind bands of 1,000 and 1,100 columns, about 0.91 and
  1.006 times the restart threshold (one more remaining column would
  restart the second);
- the same block between a band of 3,215 or 3,232 columns and a band of
  300 columns, where the horizon is capped (0.9975 and 1.0028 times the
  threshold; a cap of 191 or 193 columns flips one of them);
- a column whose updates pass 8,192 after 129 columns of 64 multipliers,
  alone (restarts) and behind 125,889 updates, where it continues by less
  than one more remaining column or the passing source's updates would
  change;
- a 128-row block behind 8,000 columns;
- the grid of side 32, which restarts at column 64 = 1024 / 16.

All
rules depend only on the matrix and the options (the pattern, and for the
others the values and `diag_pivot_thresh` through the pivots and the
factor so far), never on the thread count. The check runs in its own,
non-inlined function, and the row engine counts its rows again. The row
engine is part of `factorize_typed`, and MSVC's code for it moves with any
change to that function: a version that handed the check's counts to the
row engine factored `sparse-dense-border` with `NATURAL` 8% slower with one
thread (interleaved fresh-process runs, identical factors, Python 3.14).
The `COLAMD` and `MMD_AT_PLUS_A` paths call an instantiation of the CSC
engine without the `NATURAL` rules, whose per-column loops compile to the
same instructions as in earlier builds (MSVC `/FAs` listings; only the
one-time scan for a certain restart is laid out differently). Even so, the
timing of unchanged paths moved by a few percent either way. In
interleaved fresh-process runs (five rounds) the `COLAMD` and
`MMD_AT_PLUS_A` benchmark cases took 0.96-1.03 times their earlier time;
`MMD_AT_PLUS_A` `sparse-dense-border` with one thread measured 1.03 here
and 1.048 in the review's runs, and single gate runs against the review's
baseline 1.03 here and 1.07 in the review's, with 1.08 under Python 3.14
(199.2 to 214.9 ms). `NATURAL` inputs that skip the probe and run only the
row engine took 1.01-1.07 times their earlier time (`sparse-dense-border`
1.03-1.04 here and 1.065 in the review's runs, a 2-D grid followed by a
wide column 1.01-1.07, the 20,000-row band with a dense block of 150 or
180 rows 1.04-1.07). Factors are bitwise identical in all of these. The
functions that do that work, such as the row engine's sparse row update,
compile to the same instructions as before but sit at other offsets within
64-byte lines (the sparse row update at 48 instead of 16, linker map), and
other variants of this change, with the same source for those functions,
measured `sparse-dense-border` at 1.00 to 1.09, so these differences follow
code placement rather than the work done. Moving the row engine out of
`factorize_typed` (or into its own translation unit) would decouple it from
such changes; it is a candidate follow-up, not part of this change.

The probe's errors are final, as with `COLAMD`: an exactly singular column
or a non-finite value found by the probe is reported, not retried in the
row engine. Over 284 stress matrices with `NATURAL` (narrow random bands
with uniform, `+-1` and small-integer values, rows permuted or not, some
with entries outside the band; full bands of half-width 40 to 97;
random sparse; small dense and half-dense matrices; `+-1` Hadamard blocks;
singular matrices with a duplicated row or column or a combination of rows
or columns; planted NaN and infinite entries with `check_finite=False`;
triangular patterns; small arrows; and wide columns), 135 completed in the
CSC engine, 39 were skipped, 38 restarted, and the errors matched earlier
builds except on 4 matrices, all singular to working precision (smallest
singular value at most 4.8e-17 of the largest): one band reported singular
that the row engine factored (backward error 3.2e-12), and three singular
matrices with a duplicated or combined column factored (backward errors
1.3e-15 to 1.6e-14). The planted NaN and infinite entries gave the same
outcome as before in all 40 matrices (37 non-finite errors, two singular
columns reached first, and one NaN that reaches `U` without entering any
update and is stored by both engines), and all ten duplicated rows were
reported singular at the same column. The worst unrefined componentwise
backward error of a completed CSC factorization was 6.4e-9 (a band with
condition number 2e16), against 4.9e-7 from the row engine on the same
matrix. Factors were bitwise identical at 1, 2, 3, 4 and 8 threads for
every matrix. These results were the same under Python 3.14 and 3.15, and
before and after the cost-aware restart rule and its horizon cap, which
changed none of them (the cap left every factor and probe statistic of
the stress matrices, the test fixtures and the benchmark bitwise
unchanged).

Trade-offs that remain:

- inputs that the pattern check sends to the row engine although the CSC
  engine would be faster, because a few rows are dense: a band of 20,000
  rows with a dense 100-row block (0.28 times the row engine's time when
  the probe ran to completion) and small arrows with dense last rows and
  columns (0.60-0.66; both measured with Python 3.14). They run as in
  earlier builds;
- the pattern pass costs up to 2% of the row engine's time where it sends
  an input without a column wider than 192 rows to the row engine (bands
  with rows of 64 or more entries, dense blocks of 64 to 192 rows), which
  earlier builds started at once;
- late restarts that the rules cannot avoid, each still discarding an
  accepted prefix (fresh processes, five rounds, one and eight threads):
  - fill whose reach, including coupled rows, exceeds 192. Behind the
    benchmark band, the 96 sparse rows with a 176- or 193-row block took
    1.24-1.29 times the earlier time (the coupled sparse rows already take
    the reach to 193 at 176 rows). Behind a band of half-width 30, a band
    with entries at offsets ±1 and ±96 took 1.50-1.54.
  - a long fill region that the projected-work rule stops late, because
    the prefix's average keeps the projection low. The benchmark band
    followed by a 2-D grid of side 64 restarted at column 21,438 and took
    1.31-1.36. A band of half-width 30 (20,000 rows) followed by a grid of
    side 86 or 88 restarted at column 27,178 of 27,396 or 26,489 of 27,744
    and took 1.83-1.93, the largest ratio measured; it ran to the same
    columns before the horizon cap.
  - fill that lasts longer than the 192-column horizon behind a long cheap
    prefix: 1.2-1.6, up to 2.0 in slow rounds (see the multiplier limit
    above);
- grids still fall back after a short probe: 2-D convection-diffusion
  grids of side 20 to 64 took 1.05-1.10 times their earlier time and 3-D
  grids of side 6, 8 and 10 1.00-1.08 (fresh processes, five rounds; the
  review measured up to 1.18 and 1.12; after the horizon cap, which does
  not change their probe, side 20 measured 1.12-1.13 and sides 32 and 64
  1.05-1.07), where the probe is 1-8% of the
  time and the rest is the code-placement effect above, although the CSC
  engine would factor the 2-D grids up to side 45 in 0.45-0.71 times the
  row engine's time if their probe were allowed more than `64 * nnz(A)`
  updates;
- a factorization in a process that has just run large CSC factorizations
  can take longer in the row engine than in a fresh process (up to 1.4
  times in one sequence of benchmark families), as with other allocator
  effects noted in `docs/benchmarking.md`.

## Blocked SPD panels

The symbolic phase works on supernodes. Its first pass (elimination tree and
column counts, shared with the simplicial kernel) already gives the exact
fundamental nodes: column `j + 1` joins column `j`'s node when
`parent(j) = j + 1` and `count(j) = count(j + 1) + 1`, because the strict
rows of column `j` other than `j + 1` are rows of its parent, so equal counts
make the two row sets equal. The node widths and heights, the work guard
below, `supernodes`, `max_supernode_width` and `largest_front` follow
without any row index. Earlier builds formed every row index of `L` in a
second row-subtree pass (one step per entry of `L`), compared the strict rows
of adjacent columns explicitly (which never split a node), and built the row
incidence of `L` (another `nnz(L)` indices) for contributor discovery at
numeric time. The supernodal pass instead computes each node's tail, the
strict rows of its last column, as the union of the original entries of the
node's columns and of its child nodes' tails, keeping rows below the node, in
increasing node order with a marker array, and sorts it: `O(nnz(A) + sum of
tails)` plus the sorts (see `docs/algorithms.md`, Section 7). The first
pass's counts predict every tail's size, and a mismatch raises
`logic_error`. One walk over the sorted tails lists every node's contributors
in increasing node order, with the run of their tail rows that falls in the
node's columns, and each node's fan-in work. The CSC row indices of `L` are
written only at assembly: column `first + c` of a node holds the rows
`first + c + 1, ..., last - 1` and then the node's tail, which is exactly its
filled pattern (by the node test above), and assembly checks each column's
count against the first pass. The filled pattern, including entries that
cancel, is unchanged: `test_symbolic_structure_is_pinned` holds the stored
`L` pattern and the node statistics of the previous build on the test
fixtures, for both SPD engines. On 3-D Poisson of
side 20 with `COLAMD` the symbolic phase fell from 17.5 to 5.2 ms: the
row-subtree pass, the explicit comparison, the row incidence and the
allocation of the CSC arrays (now made at assembly or during phase A, below)
took 3.8, 0.6, 5.9 and 4.1 ms, while the first pass stays at 2.9 ms and the
tails and contributor lists take 2.2 ms (1.6 ms of it for the tails and
their sorting).

The `symmetric-supernodal-ldlt` engine factors one exact fundamental
supernode at a time. Each node owns a dense row-major block of
`height x width` entries, its own columns first and then its tail rows in
ascending order, and all blocks share one array allocated for the numeric
phase: the symbolic phase computes `block_ptr`, the prefix sums of the node
rectangles (with overflow checks), which gives each block's offset. A node
zero-fills its block, scatters original matrix entries into it (a per-thread
map gives the rectangle position of each tail row), then subtracts the
contributions `L[R,Q] @ D[Q] @ L[J,Q].T` of the nodes the symbolic phase
listed for it, in increasing node order. The rows of a contributor `Q` that
fall inside the target are a contiguous suffix of `Q`'s rows
(elimination-tree containment; the code checks that every one has a target
position), so `L[R,Q]` is a run of full rows of `Q`'s finished block and is
read in place with row stride `|Q|`. Only the right operand `D[Q] @ L[J,Q].T`, from those of the rows that
are target columns, is packed, as a `|Q|`-by-columns block with
`right[k][j] = d[k] * l[j][k]`. The rows' target positions may be
noncontiguous and the overlap partial. Each contributor is applied as one
rank-`|Q|` update: every target entry sums all of that node's terms
`l[i][k] * (d[k] * l[j][k])` from zero in ascending column order and subtracts
the sum once (see "Dense kernels"). The packed operand holds at most `|Q|`
times the contributor's height, the size of that contributor's block. A
finished block is never written again, only read by later nodes.

Earlier builds kept one node rectangle at a time, copied each finished node
into the CSC values, and packed contributors from there, including
`L[R,Q] @ D[Q]` as a rows-by-`|Q|` left operand, so each term was rounded as
`(l[i][k] * d[k]) * l[j][k]`. Reading `L` in place moves the diagonal factor
to the right operand. This changed LDL^T factors only in the last bits
(largest entry change 2.2e-16 of the largest `U` entry, relative Frobenius
change about 1e-17, on 3-D Poisson of side 20 under `avx2-fma` and `sse2`)
and left the backward error `max|P^T S A S P - L D L^T|` unchanged (1.89e-15
with `COLAMD` and 1.22e-15 with `MMD_AT_PLUS_A`; Frobenius norms within
0.1%). Nodes without contributors (such as disconnected dense blocks) and
updates through tiny intersections are bitwise unchanged. Before that, builds
packed and subtracted one sum per 32 columns of a contributor; the single sum
changed LDL^T factors only in the last bits (largest entry change about 8e-16
of the largest `U` entry, relative Frobenius change about 1e-16) and did not
increase the backward error (3-D Poisson of side 20: 1.44e-15 to 1.22e-15
with `MMD_AT_PLUS_A`, 1.89e-15 unchanged with `COLAMD`).

All contributions to a node are applied in one fan-in phase. For each
contributor the node records its target positions (the positions of its rows
in the node rectangle, ascending; those that are node columns come first, so
output entry `(i, j)` lies on or below the diagonal exactly when `i >= j`),
its first and last target rows, and, unless the intersection is tiny, the
offset of its packed operand. The node rectangle is split into fixed strips of
16 rows (at multiples of 16, spanning every column), so the partition depends
on the node's shape only. A strip visits the contributions in ascending node
order, finds by binary search the output rows of each that fall in the strip,
and applies them together with the output columns up to the last of those rows
(later columns are above the diagonal), in parts of at most 16 rows by 128
columns:

- Tiny intersections (fewer than 256 output rows times columns in the whole
  contribution) retain sparse outer products, reading the contributor's block
  in place: per source column `k`, `f[i][j] -= (l[j][k] * d[k]) * l[i][k]`.
- Otherwise a part whose output columns are contiguous node positions is
  updated in place in the node rectangle: its rows that are also its columns
  are then contiguous too and `gemm_sub_lower` updates them with the matching
  diagonal shift, and its rows below every column are updated by `gemm_sub`
  when contiguous or by `gemm_sub_rows` through one pointer per row. A part
  with noncontiguous columns is gathered into the worker's 16-by-128 scratch
  block (only its entries on or below the diagonal), updated by `gemm_sub` or
  `gemm_sub_lower`, and scattered back.

The choice between a sparse and a kernel update depends on the whole
contribution's size, never on its part in a strip, and a kernel element's
result depends only on its own operands, so every entry receives the same
operations, in the same contributor order, as with one update per contributor:
the fan-in left `LDL^T` factors bitwise identical to the previous build (one
tiled update per contributor) under all three kernels. A node whose fan-in
meets the parallel threshold below (counting its strips as items) packs every
operand into one buffer first and runs its strips on the worker team, claimed
from the bottom up because lower rows reach more columns. Otherwise the
thread that factors the node (the calling thread, or a phase A worker; see
below) applies one whole contribution after another, as one strip of all
rows, packing each operand just before it is used, so the buffer only needs
room for the largest one. Strips write disjoint rows and only read the
finished contributor blocks, the operands, and the target positions, so
neither the thread that runs a strip nor the order of strips can change a
result. The packed operand buffer, which the panels' shared right operand also
uses, and the gather blocks start on 64-byte boundaries: the allocator only
guarantees 16 bytes, and a shared panel operand that started 16 bytes into a
cache line made the strip phase of four dense 600-wide SPD blocks about 30%
slower at one thread (22 instead of 17 million cycles; starts 0 or 32 bytes
into a line were unaffected, 48 bytes at most 4% slower) in some processes and
not in others, depending only on where the heap placed the buffer.

The strip shape was chosen by interleaved runs (three rounds of seven
batched samples, `avx2-fma`, i5-13600K) of builds whose shape was set at run
time, measuring `spd_source_seconds` at eight threads, for two-dimensional
tiles (rows by columns, aligned at multiples of the tile shape; a part is at
most one tile) and for full-width strips (parts of at most 128 columns):

| shape | 3-D Poisson 20, `COLAMD` | 3-D Poisson 20, `MMD_AT_PLUS_A` | random SPD, n=3,000, `MMD_AT_PLUS_A` |
|---|---:|---:|---:|
| 2-D tiles 32 x 128 | 20.8-21.2 ms | 5.9-6.0 ms | 8.6-9.0 ms |
| 2-D tiles 16 x 16 | 27.4 ms | 7.7 ms | 52.8 ms |
| 2-D tiles 64 x 64 | 26.9 ms | 5.9 ms | 9.6 ms |
| 2-D tiles 32 x 256 | 21.2 ms | 5.8 ms | 6.7 ms |
| strips of 32 rows | 19.8 ms | 5.9 ms | 5.0 ms |
| strips of 24 rows | 20.1 ms | 5.7 ms | 4.4 ms |
| strips of 16 rows | 20.5 ms | 5.6 ms | 4.5 ms |
| strips of 64 rows | 35.0 ms | 8.1 ms | 5.4 ms |

The random SPD matrix (a symmetric random pattern of density 0.004 plus a
dominant diagonal) ends in one dense root node of width 2,158 that receives
842 rank-1 contributions with irregular columns; with two-dimensional tiles
every tile repeats its binary searches per contribution and column range, and
the small parts make many short kernel calls, so full-width strips cost about
half as much there. Taller tiles or strips leave fewer items and fall below
the two-items-per-thread rule on more nodes (64-row strips: 26 instead of 55
parallel phases on 3-D Poisson with `COLAMD`); 16 x 16 tiles multiply the
per-part overhead. Strips of 16 rows also keep the 350-column root of the
`overlapping_cliques` test matrix parallel at eight threads (1.76 instead of
2.03 ms numeric time with 32-row strips). Parts of 256 columns instead of
128 were slower in five of the six strip comparisons, by up to 20% (the random
matrix with 16-row strips). At one thread every node takes the whole-node
path, where the part shape (16 to 64 rows, 128 to 512 columns) moved
`spd_source_seconds` by at most 4%.

The node's columns are then factored in panels `P = [begin, end)` of at most
32 columns. For each panel:

1. The diagonal block is factored right-looking. Each pivot must be positive
   and finite. Column `p` is divided by its pivot and copied to a contiguous
   buffer, so its rank-1 update of each later block row `i` is an AXPY over
   `j`, `f[i][j] -= w * l[j][p]` with `w = l[i][p] * d[p]`: the same
   per-element sequence as the scalar loop it replaced, so an entry that
   cancels exactly still stores an exact zero.
2. Rows below the block are solved in AXPY form, `L[i,P] = F[i,P]
   inv(L_PP.T) inv(D_P)`. The panel forms `S[k][p] = d[k] * l[p][k]` once
   (only when rows below the block exist); each row then runs, for `k`
   ascending, `x[k] /= d[k]` followed by `x[p] -= x[k] * S[k][p]` for every
   `p > k`, contiguous in `p`. Four rows are interleaved to overlap their
   division chains. Each entry subtracts its terms in ascending `k`, each a
   rounded product and a rounded subtraction, and then divides by its pivot.
   Earlier builds formed each term as `(l[i][k] * d[k]) * l[p][k]` inside a
   dot-product loop; the new association `l[i][k] * (d[k] * l[p][k])`
   changed LDL^T factors only in the last bits (largest change about 2e-16 of
   the largest `U` entry, relative Frobenius change about 1e-17) and left the
   backward error unchanged (3-D Poisson of side 20: 1.89e-15 with `COLAMD`
   and 1.22e-15 with `MMD_AT_PLUS_A`, Frobenius norms within 0.2%).
3. The trailing update `F[R,J] -= (L[R,P] D_P) L[J,P].T` covers the remaining
   node columns `J = [end, width)`, lower part only, and the rows `R` below the
   block. Rows `[end, width)` are solved first, in order, on the calling
   thread, because their L entries form the shared right operand
   `L[J,P].T`, which is packed once. One phase then covers rows
   `[end, height)` in fixed strips of 16 node rows, claimed bottom-up because
   lower rows update more columns. A strip solves its own rows at or below
   `width`, forms `L D` for its rows, and applies one rank-`|P|` kernel call
   (`sum_then_subtract`) to its rows' entries in columns `[end, min(row,
   width - 1)]`. A strip reads only its own rows and the shared, read-only
   pivots, `S`, and right operand, and writes only its own rows, so neither
   the strip partition (fixed by position) nor the thread that runs a strip
   can change a result.

A node's fan-in or a panel's strip phase runs on the persistent worker team
when it has at least two threads, at least 2 MFLOP (2^21 counted operations,
as in `dense_update_flops`: a fan-in counts all of its contributions, sparse
ones included, and a strip phase counts its solves and its update), and at
least two items per thread (the node's 16-row fan-in strips, or the panel's
strips), counting at most eight threads. Otherwise the thread that factors
the node does the work: a panel's strips in order, or a fan-in one whole
contribution after another (see above); a node inside a phase A subtree
always does. A dispatch costs about 20 microseconds at four to eight
threads and 2 MFLOP is about 35 microseconds of `avx2-fma` tile work, so
smaller phases ran slower in parallel. Capping the counted threads keeps a
team larger than eight from running serially an update that eight threads
would share; its extra workers only cost the dispatch. The rule is an
efficiency heuristic that depends only on the update's size and the thread
count, and it decides only which thread computes which entries, never their
arithmetic (the fan-in rule was kept unchanged; its measured strip shape is
discussed above). Panel strips are 16 rows rather than 32 so that, under this
rule, panels with 256 to 512 rows below them also run in parallel at eight
threads; the kernel runs at the same speed on either strip height.

Nodes are scheduled by subtree. A node reads only the blocks and pivots of
its contributors, which are its descendants in the elimination tree, so
disjoint subtrees can be factored concurrently. The symbolic phase gives each
node a work estimate: its fan-in and panel operations as counted in
`dense_update_flops`, plus 8 per block entry, 256 per contributor and 2,048
per node for the memory-bound zero-fill, scatter, target positions and
bookkeeping (allowances measured on small nodes; the estimate only balances
work). With `W(s)` the estimate summed over node `s` and its descendants and
`W` the total, a node with `W(s) <= W / (2 * threads)` whose parent exceeds
that bound, or that is a root, heads a subtree. When at least two threads and
two subtrees exist:

- phase A runs one `parallel_for` over the subtrees, largest `W(s)` first;
  each worker factors a whole subtree serially, its nodes in increasing order,
  with its own row map, operand buffer, fan-in lists and gather block, all
  sized on the calling thread before the phase from the symbolic lists, so
  no subtree item allocates. A subtree's nodes never use the worker
  team themselves (no nested parallelism): their fan-ins and panel strips run
  on the worker's thread, which `worth_parallel` with one thread selects;
- phase B then factors the remaining top nodes, whose descendants are all
  finished, in increasing order on the calling thread, with the parallel
  fan-ins and panel strip phases described above.

Otherwise (one thread, or fewer than two subtrees) every node is a top node,
as in earlier builds. The bound follows Graham's list-scheduling argument: a
greedy schedule finishes within the average load plus its largest item, so
subtrees of at most half the per-thread share keep the workers within 1.5
times the ideal time. Alternating runs at four and eight threads of a build
whose bound was set at run time (before the CSC allocation items below)
found `W / threads` and `2 * W / threads` slower on 3-D Poisson of side 20
(`COLAMD`: 40.0-40.6 and 42 ms instead of 39.3 ms at eight threads, and
`MMD_AT_PLUS_A` 17.0-17.7 instead of 16.5-16.8 ms), because fewer, larger
subtrees balance worse while their top nodes still run in phase B.
A forest of comparable trees (disconnected components) is the exception: when
there are at least two trees, at least half as many as the counted threads
(`min(threads, 8)`), and no tree holds more than half of the work, every tree
is a subtree. Phase B's top nodes are often dense and scale poorly (four dense
600-wide SPD blocks: 9.3 ms in phase B at eight threads, 11.3 ms on one
thread), while four workers factor the four blocks in about 3.5-5 ms. The
rule waives the bound above: a tree holding up to half of the work runs on
one worker, which caps phase A's speedup at about 2x for such a forest
(adversarial forests of that kind still ran at 0.48-0.85 times the previous
build's time in an independent review). A
forest dominated by one tree keeps the default bound (a 20,000-row chain
beside a 1,000-wide block runs that block in phase B), and so does the
`spd_forest` test matrix in natural order, whose 12^3 grid becomes a chain of
nodes with about 140 contributors each that holds most of the work.

Phase A also allocates the CSC arrays of the factor, which assembly fills,
as two more items ordered with the subtrees by estimated cost (8 per entry):
zero-filling fresh pages costs about 1.8 ms per array on 3-D Poisson of side
20 with `COLAMD` (1.86 million entries) and otherwise runs on the calling
thread at assembly, while the workers that finish their subtrees early absorb
it here. An allocation failure on a worker is ignored and the allocation is
repeated at assembly on the calling thread, so it is reported only once
every pivot has passed. Without phase A the two arrays are allocated
concurrently at assembly when there are at least two threads. This changes
one behavior: earlier builds allocated both arrays in the symbolic phase, so
running out of memory for them raised `MemoryError` even for a matrix that
then turned out indefinite; such a matrix now falls back to LU (which needs
its own memory).

A worker checks the team's failure flag (`FactorExecutor::failed`) before
each node and abandons its subtree once any item has thrown; the executor
starts no further item, waits for every worker, and rethrows the first
exception on the calling thread, where a `NonPositiveDefinite` rejects the
attempt as before. Several subtrees may fail at once; any of their failures
leads to the same fallback, and the fallback LU does not depend on which one
was reported.

The partition depends on the thread count (through the bound, the forest rule
and `worth_parallel`), which cannot change a result: each node's block is a
function of B's entries in its columns and of its contributors' finished
blocks and pivots only, and its contributors are in its own subtree,
finished earlier by the same worker, or are top nodes with smaller numbers
or subtree nodes, all finished before it in phase B. Each node applies its
contributions in the fixed increasing order of the symbolic phase, every
output entry receiving the same operations whether its fan-in runs whole on
one thread or in strips (the kernel contract, see "Threading behavior"), and
factors its panels in position-fixed strips. Nodes write only their own
blocks and pivots, and the maps, operand buffers, fan-in lists and gather
blocks are per worker. Factors were bitwise identical to the previous build
under all three kernels and at 1, 2, 3, 4, 8 and 20 threads.

Finiteness is tested where values become final rather than where they are
produced. Original entries are tested when scattered. Afterwards every
node-rectangle entry on or below the diagonal ends as a pivot or as an L
entry: each pivot is tested (positive and finite) before it is used, and each
panel's L entries are scanned once (`fabs(v) <= DBL_MAX`) after their last
write, the diagonal block's after its factorization, rows `[end, width)` after
their serial solve, and each strip's rows after its solve. Dense kernel calls
report through their flag, which is authoritative: a kernel may stop early, so
a false flag rejects the attempt at once. A node's L entries are therefore all
tested before any later node reads its block. This is complete because entries
combine only through addition, subtraction, multiplication, and division by a
tested pivot, which keep an Inf or NaN operand non-finite (`Inf*0` and
`Inf-Inf` are NaN), and because every auxiliary product reaches an entry or a
flag: a sparse update's `l * d` updates its own diagonal position with the
nonzero `l`; the diagonal block's `l[i][p] * d[p]` updates the future pivot
`(i, i)`; `S[k][p]` enters every lower row's `l[i][p]`; and a contributor's
packed `d[k] * l[j][k]` enters output entry `(j, j)` with the nonzero
`l[j][k]` (a zero `l` gives a zero product), a diagonal entry that the
contribution's part in the strip holding row `j` (or the whole contribution,
on the thread that factors the node) always includes and a kernel computes. The
contributor's in-place left operand holds only tested L entries. Every fan-in
strip, and every node of a phase A subtree, runs unless an earlier item has
already failed, in which case the attempt is rejected anyway. A rejected
attempt assembles and publishes nothing and falls back to pivoted LU. Matrices whose denormal pivot
makes L overflow in the serially solved rows or in strip-solved tail rows are
rejected by those scans; overflow inside the diagonal block reaches the next
pivot test first.

After the last node has passed, one assembly pass writes the CSC row indices
of `L` (column `first + c` of a node holds the rows `first + c + 1, ...,
last - 1` and then the node's tail) and copies the strict lower entries of
every block into the CSC values (block rows `c + 1` to `height - 1` of block
column `c`, in order), and then releases the blocks; `assembly_seconds`
includes it, and the allocation of the CSC arrays when phase A did not make
it. Entries that cancelled to zero are copied as stored zeros. A failed
attempt returns before assembly. The copy runs in fixed items of eight factor
columns, split at node boundaries; it uses the worker team when there are at
least 2^18 CSC entries and two items per thread (counting at most eight
threads), and since it only copies, neither the partition nor the thread can
change a value. A node of width `W` and
height `R` stores `R*W` block entries for its `R*W - W*(W+1)/2` strict L
entries, so the blocks hold at most `2*nnz(L) + n` entries, with equality only
for nodes whose height equals their width (dense diagonal blocks). On 3-D
Poisson of side 20 they hold 2.11 million entries (16.9 MB, 1.14 times the L
entries) with `COLAMD` and 1.08 million (8.7 MB, 1.30 times) with
`MMD_AT_PLUS_A`; four dense SPD blocks of size 600 hold 1.44 million (11.5 MB,
twice the L entries). The CSC arrays (row indices and values) are allocated
during phase A or at assembly, so they coexist with the blocks from then on
(and are discarded with a failed attempt only if phase A had allocated them);
the blocked path's peak memory is the factor plus all node blocks, where
builds before the block storage held the factor plus one node rectangle and
its packed operands. Earlier builds allocated both CSC arrays in the
symbolic phase.

For node width `W` and height `R`, predicted internal update terms are
`W*(W-1)*(3*R-2*W+1)/6`. Only nodes at least 320 columns wide contribute useful
work; their total must be at least `3072*n`, where `n` is the whole matrix
dimension. The exact fundamental nodes from the parent/count test decide this
before any row index is formed (earlier builds evaluated the guard on the
same runs as an upper bound and again after an exact pattern pass, which
always agreed). The measured guard accounts for per-node driver overhead
without rejecting an expensive large node solely because it covers few
matrix columns.

The filled CSC pattern, including entries that cancel numerically to zero,
remains the public storage contract. No factors, symmetric scales, or blocked
work counters are published until every positive pivot and finite intermediate
has passed validation, and the CSC values are assembled from the blocks only
then. Each output entry visits contributors in the same order at every thread
count.

## Dense LU panels

Pivoting remains sequential. Sparse rows receive each complete U-row update
immediately. Dense rows keep every coefficient inside the current panel
`[first, last)` current, while deferring updates at or beyond `last`. The
panel path (`DensePanel` in `src/sparse_lu_panels.hpp`) maintains these
invariants:

- **Panel block.** When a panel starts, after the pending dense rows that
  begin before `last` are admitted, its members are the dense rows not yet
  pivoted whose storage meets `[first, last)`. Their panel columns are
  copied into one column-major `members x 32` block. Inside a panel every
  dense candidate is a member: pending rows start at or after `last`,
  conversions wait for the flush, and every other dense row is an exact zero
  in the panel's columns, which only a candidate's update could change. The
  block is then factored in place, dgetf2-style: per pivot column one
  contiguous scan of the block column finds the members' candidates (each
  nonzero of an unpivoted member, checked for finiteness), their largest
  magnitude (ties to the smaller position), and their count for
  `row_updates`; a member's position follows every row swap. Sparse rows
  stay candidates through the column lists, and the two maxima merge by the
  same rule (larger magnitude, then smaller position), so the pivot choice
  and the preferred-row test are those of a single candidate list. After the
  pivot step, the block column is divided by the pivot and tested for
  finiteness as a whole, then every later block column with a nonzero pivot
  row value `u` takes one `axpy_sub` with `u`. Members that are not
  candidates have multiplier `+-0`, and a pivot's block row is cleared at
  its pivot step, so no target list is needed. Each member records, as a
  32-bit mask, the panel columns at which it was a candidate: those are its
  L entries for the panel.
- **Retained pivot rows.** While a panel is active, a dense row selected as
  pivot appends its panel L entries, writes its block row back to its
  storage as its in-panel U values, and keeps that storage, unmodified, until
  the panel's flush. A sparse pivot supplies its in-panel U entries from its
  U row.
- **Contiguous materialization.** Before a dense row becomes a pivot, its
  deferred tail is completed from its L multipliers in ascending pivot
  order: one `axpy_sub` over `[last, tail_end)` for each retained dense
  source, where `tail_end` is one past the source's last nonzero, and a walk
  over the U entries of a sparse source. U is therefore complete before it
  supplies any subsequent update, even when row swaps cross panel
  boundaries.
- **Storage extents.** A member's storage grows exactly as in-place updates
  would have grown it (to `min(last, end of that pivot's row)` for each
  panel step at which it was a candidate), so storage ends, and the work
  estimates derived from them, do not depend on the block. While a panel is
  active, a member's storage inside the panel is stale; like every storage
  column before a row's current pivot column, it is never read.
- **Lazy U rows.** The U row (`Entry` vector) of a retained dense pivot is
  built once: serially, before the workers start, when a sparse row needs it
  as an update source, otherwise during the flush. Sparse pivots publish
  their U rows at once, and dense pivots outside a panel build theirs at the
  pivot step. The flush finalizes the panel even when nothing is left to
  update (the last panel never has targets): it builds the remaining U rows
  and releases the retained storage.
- **Segment packing.** The packed U support covers the panel pivots'
  nonzeros at or after `last`, as sorted disjoint column segments. The
  retained dense pivots' tails `[last, tail_end)` are OR-ed into a column
  mask (`tail_end` comes from a backward scan of the retained row), and the
  mask's columns are merged with the sparse pivots' tail columns. Gaps
  narrower than 16 columns are bridged with zero U columns, wider gaps start
  a new segment, so a dense block plus a distant column packs two segments
  rather than the whole range between them. Dense pivots are copied into the
  packed U rows segment by segment, sparse pivots scattered; the targets'
  packed L rows are the block's multiplier columns, transposed, with exact
  zeros outside each target's candidate mask. Before packing, every target
  is extended serially to the end of the union, so each segment is
  contiguous in every target.
- **Flush pass.** The flush targets are the members still unpivoted that were
  candidates in the panel. One pass runs three kinds of independent items:
  the U rows still to be built (each also releases its retained storage),
  the targets' panel L entries (groups of 16 targets), and tiles of 16
  target rows by at most 64 columns of one segment that call
  `gemm_sub_rows` in place on the targets' dense tails. Every output starts
  from its current value and subtracts its `rank` terms in ascending pivot
  order (`running`). Each item writes only its own rows, so segments, items
  and results depend only on column positions, never on the thread count.
  The pass uses the worker team when more than one thread is requested,
  there are at least four tiles, and the counted tile work is at least
  262,144 flops.
- **One rounding rule.** Every path that updates a dense entry (block
  updates, materialization, the flush, and dense-row updates outside
  panels) computes `c = c - round(l*u)` with separate rounding, in ascending
  pivot order, so two identical rows stay bitwise identical whichever path
  their updates take; exact singular detection of a duplicated row depends
  on it. A block entry receives exactly the operations that in-place updates
  of its row would apply: multiplier `value / pivot` (division), then one
  term per later pivot with the pivot row's current value. Terms with a zero
  multiplier or a zero U value, and packed columns whose U values are all
  zero (bridged gaps), can change at most the sign of a zero entry, which is
  never stored as a factor entry or accepted as a candidate.

Finiteness is checked for every candidate value (in the block scan for
members) and pivot, every L multiplier (one test per block column), every
value a sparse row update writes, every kernel call (`axpy_sub` and the flush
GEMM report whether any element they wrote is non-finite, and a false flag
raises the usual error), and every U row built from dense storage. Only the
scalar U-entry walks of sparse sources during materialization are not
checked per element. The checks are complete: suppose elimination stores a
non-finite value in column `c` of row `r` (in its storage, or in its block
row while `c` is inside the active panel), and `r` is later pivoted at
position `pos(r)`. If `c >= pos(r)`, the value belongs to `r`'s U row (a
pivot's block row is written back to its storage), which is checked when it
is built, before any factor is published. If `c < pos(r)`, the value is
still present at step `c`: non-finite operands never produce finite results,
columns at or before the current pivot are only written to clear an already
checked coefficient or replace it by its checked multiplier, and deferred updates
to column `c` are applied before step `c` (a flush precedes the next panel,
and the block keeps the panel's columns current and is scanned directly). A
non-finite value is nonzero, so `r` is a candidate at step `c` and the
candidate check rejects it. Every row is pivoted, so no value escapes both
checks. Detection can come later than the arithmetic step that produced the
value; if an exactly singular column is reached in between, the singular
error is reported instead. Because a block elimination tests all multipliers
of a step before any update, a step may raise "non-finite multiplier" where
updating row by row raised "non-finite value" for an earlier target. In
either case no factor is published. Input values are validated separately
when `check_finite` is set.

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
element; the row-hybrid LU uses it for the panel block's column updates,
for pivot-tail materialization from retained dense pivots, and for dense-row
updates from dense pivots outside panels). The GEMM entry points take an
accumulation order:

- `running`: the output starts from its current value and subtracts each
  term in ascending `k`, `c = (c - a0*b0) - a1*b1 - ...`;
- `sum_then_subtract`: the terms are summed from zero in ascending `k`, and
  the sum is subtracted from the output once, `c = c - (a0*b0 + a1*b1 + ...)`.

The LU flush uses `running`. It is the recurrence of the panel block's
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
  tiles run at about 52 GFLOPS (`sse2`: 21). Measured on the 16-by-16 output
  tiles that earlier builds used for each LDL^T contributor update (the
  fan-in now issues parts of up to 16 by 128; no isolated throughput figure
  was taken for them), off-diagonal tiles ran at about 60-64 GFLOPS (`sse2`:
  18-22) for ranks 32-320, and diagonal tiles (`gemm_sub_lower`) at about
  35-48 GFLOPS (`sse2`: 16), counting only the entries written, because of the
  scratch gather and scalar scatter of their diagonal-crossing microtiles.

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
singular matrix factored instead of raising. The panel block and
materialization from dense pivot rows call `axpy_sub`, but some terms from
sparse pivot rows (pivot-tail materialization and dense-row updates outside
panels) are applied by scalar U-entry walks, so fusing `running` would
require those walks to round the same way as every kernel.

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
scalar walks (sparse-row updates and dense-row updates from sparse pivots
outside panels in `sparse_lu.cpp`, and pivot-tail materialization from
sparse sources in `sparse_lu_panels.hpp`), rounds `a*b` and the following
subtraction separately, even with `SP_ARCH_NATIVE` on an FMA host. The exact cancellation
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

Partitions elsewhere depend on sizes and positions only. The exception is
the supernodal LDL^T engine, in two places. Through `worth_parallel`, the
thread count decides whether a node's contributions are applied whole on one
thread (parts cut from each contribution's own rows and columns) or in 16-row
strips on the worker team (parts cut at node-row positions). Through the
subtree bound `W / (2 * threads)` and the forest rule (see "Blocked SPD
panels"), it decides which nodes a phase A worker factors serially and which
top nodes run in phase B with the team. Its factors stay bitwise identical at
every thread count only because of the kernel contract in
`src/dense_kernels.hpp`: an element's result depends only on its own operands,
never on the tile shape, its position in a tile, or the call path (exercised
by `test_kernels_agree_across_threads_and_tile_paths` and the thread-count
comparisons of the factor tests), and because each node's block depends only
on its inputs and on its contributors' finished blocks, which are its
descendants and are finished before it in either phase. Any future kernel
must keep that contract; for example, blocking the rank (`k`) dimension into
shape-dependent partial sums would make LDL^T factors depend on the thread
count.

`lu_dense_panels` counts nonempty dense tail flushes, and
`lu_parallel_panels` counts flushes that used the worker team; the latter is
included in `parallel_steps`. `lu_panel_columns` is the maximum packed U-tail
support size: the total width of a flush's column segments, including
bridged gap columns (see "Dense LU panels"). `lu_dense_update_flops` counts
two operations per packed multiply/subtract term, including zeros in packed
operands (bridged columns and zeros inside segments as well); the panel
block's own updates and pivot-tail materialization are excluded.
`lu_panel_workspace_bytes` estimates peak panel-owned vector capacities,
including membership and discovery marks, member and row lists, the panel
block, per-pivot panel state, the segment column mask, segment and tile
lists, and packed L/U operands. It excludes retained factors, row-engine
sparse/dense storage (including pivot rows retained until a flush), and
worker runtime storage, and must not be interpreted as total row-engine peak
memory. At two or more threads its value depends on worker scheduling (it
sums per-worker vector capacities, which grow with the items each worker
happened to run), so it is stable only at one thread; it is a diagnostic,
not part of the thread-count determinism contract.
The SPD-specific `parallel_panels`, `dense_update_flops`, and `spd_*` fields
retain their existing meaning; with concurrent subtrees, the `spd_*` timers
add the slowest phase A worker's time to phase B's, and `spd_subtrees` and
`spd_subtree_seconds` describe that phase (see below). `dense_kernel` names the dense tile kernel the
factorization snapshotted (`avx2-fma`, `sse2`, or `scalar`; see "Dense
kernels"), whether or not the selected engine used it.

`ordering_estimated_l_nnz` and `ordering_estimated_max_front` report AMD
estimates for `MMD_AT_PLUS_A`; zero means unavailable for other orderings.
COLAMD and NATURAL expose `numeric_probe_columns`, `numeric_probe_max_front`,
`numeric_probe_updates`, `numeric_probe_seconds`, and `numeric_probe_fallback`
(all zero or false with `MMD_AT_PLUS_A`, and for any ordering when the SPD
path succeeds). The probe counts completed columns and scalar updates, and
records the largest observed reach, including the column that triggers a
restart. A column that triggers the factor-storage limit, or the `NATURAL`
multiplier limit or projected-work bound, is factored first and counts as
completed; one that triggers the reach, per-column or total update limit
does not. Probe time is included in numeric/factor time; on success it is
the CSC factorization itself. For `NATURAL` it also covers the pattern check
that decides whether the probe starts; a probe that check skips reports
`numeric_probe_fallback` true, zero columns, front and updates, and the
check's time as `numeric_probe_seconds` (earlier builds reported false and
zeros for every `NATURAL` factor). Workspace diagnostics include the
temporary restart snapshot (the requested permutation and both scale
vectors; the probe's counters and empty factor buffers are restored
alongside) and discarded CSC buffers on fallback, at their capacity, which
includes the storage reserve described in "Left-looking CSC LU". The
`NATURAL` check's per-row extents (24 bytes per row) are released before
the probe, whose own vectors are larger, and are not counted.

For blocked SPD, `parallel_panels` counts actual parallel phases (a node's
contributor fan-in or a panel's strip phase that met the threshold in
"Blocked SPD panels"); nodes factored in phase A never contribute to it.
`parallel_steps` is `parallel_panels`, plus one when phase A factored at
least two subtrees concurrently. `spd_subtrees` is the number of phase A
subtrees (zero when phase A did not run, including on one thread), and
`spd_subtree_seconds` is phase A's wall time, which includes the CSC array
allocation items. `threading_backend` reports `std::thread` for a column
engine whenever `parallel_steps` is positive (earlier builds tested
`parallel_panels`, identical before phase A existed). A fan-in is one phase
per node; earlier builds counted one phase per contributor update that met
the threshold, so the count fell, for example from 98 to 55 on 3-D Poisson of
side 20 with `COLAMD` and from 31 to 25 with `MMD_AT_PLUS_A` at eight
threads. With phase A, the nodes of the subtrees no longer count: at eight
threads the same cases report 45 and 21 parallel panels plus one subtree
step (17 and 9 subtrees), and four dense 600-wide SPD blocks 0 instead of 40
(four subtrees).
`dense_update_flops` counts two operations per executed multiply/subtract
term, including the diagonal-block, below-panel solve, and sparse
intersection work; diagonal weighting (including the panel's `S`, the
strips' `L D` operands, and the contributors' `D L.T` operands), divisions,
and operand packing are excluded.
`largest_front` is the actual largest filled node height on either SPD path;
`spd_panel_entries` is the largest allocated dense node rectangle. A rectangle
uses less than twice its node's lower-triangular CSC entries, and does not
densify unrelated matrix rows or columns. `peak_workspace_bytes` estimates
simultaneously live numeric and symbolic vector buffers, including CSC factors
and packed operands. On the blocked SPD path it counts the supernodal
symbolic structure (node bounds, column-to-node map, tails, contributor
lists, per-node work), the whole node block array once (every node
rectangle, allocated for the numeric phase, at most `2*nnz(L) + n` doubles;
see "Blocked SPD panels"), and the CSC arrays from their allocation, during
phase A or at assembly, on; it is the larger of the numeric phase's peak and
the assembly's (symbolic structure, blocks and CSC arrays together). Builds
before the supernodal symbolic phase allocated the CSC arrays before the
numeric phase and also held the row incidence of `L` (`nnz(L) + n + 1`
indices), so the estimate fell, for example on 3-D Poisson of side 20 with
`COLAMD` from 62.9, 66.0, 66.1 and 66.3 MB to 49.2, 53.2, 53.3 and 53.6 MB at
one, four, eight and twenty threads, with `MMD_AT_PLUS_A` from 30.7 and
32.4-32.6 MB to 24.1 and 27.1-27.4 MB, on four dense 600-wide SPD blocks from
29.1-29.5 MB to 23.2-24.5 MB, and on a random SPD matrix (n=3,000, density
0.004, `COLAMD`) from 105.9 and 115.1-115.3 MB to 82.7 and 99.2-100.5 MB.
Phase A adds, per worker, a row map of `n` 4-byte entries (8 bytes when
`n >= 2^31`), one 16-by-128 gather block, and an operand buffer, contribution
list and target positions sized for the largest subtree node, all released
after phase A; the team is limited so that these never exceed half the bytes
of the node block array (at least two workers), which bounds the per-thread
growth. A million-row matrix at twenty threads would need 80 MB of row maps,
which the limit allows only next to at least 160 MB of node blocks. On the
random matrix above the limit holds phase A to fewer workers than threads
(without it, 118 and 155 MB at eight and twenty threads). Earlier builds counted one node
rectangle instead, so the estimate rose with the block storage, for example
from 49.3 to 62.6 MB on 3-D Poisson of side 20 with `COLAMD`, 25.5 to
30.3 MB with `MMD_AT_PLUS_A`, and 20.5 to 29.1 MB on four dense SPD blocks of
size 600. It also counts the fan-in's contribution list and target positions,
one 16-by-128 gather block per thread (16 KB each, allocated once per
factorization), and the packed operand buffer, which the fan-ins and the
panels share. That buffer only grows. A fan-in on the worker team packs all
of its node's operands at once: the sum over its kernel contributors `Q` of
`|Q|` times the contribution's output columns doubles, which is at most the
sum of those contributors' blocks (each operand holds at most `|Q|` times the
contributor's height); the target positions add one index per output row of
every contribution. On the calling thread the buffer needs room for the
largest operand only. On the blocked SPD path the estimate therefore depends
on the thread count, through the parallel fan-ins and the phase A workers.
When the parallel fan-in was introduced (before the supernodal symbolic
phase), 3-D Poisson of side 20 with `COLAMD` reported 62.9 MB with one thread
and 66.0, 66.1, and 66.3 MB with four, eight, and twenty (62.6 MB at every
thread count before the fan-in), 30.7 and 32.4-32.6 MB with `MMD_AT_PLUS_A`
(30.3 MB before), and the random SPD matrix of the strip measurements above
(n=3,000, `COLAMD`) 141.7 MB with one thread and 152.2-152.5 MB with four to
twenty (137.3 MB before); four dense SPD blocks of size 600, which have no
contributors, reported 29.1-29.3 MB (29.1 MB before).
Packing a parallel fan-in's operands in batches would bound the buffer
further; this build does not. Thread runtime storage is excluded, including
the fixed per-call stack blocks of the panel steps (the 32-by-32 `S`, a
32-entry column, and each strip's 16-by-32 `L D` operand) and the fan-in's 16
row pointers per part.
The estimate sums vector capacities, which grow geometrically, so a change in
which buffers are resized, and in what order, can move it by a few hundred KB
either way between builds: removing the trailing-update operand buffers
lowered it on most SPD cases but raised 3-D Poisson with `MMD_AT_PLUS_A` by
about 53 KB.

`spd_source_seconds` measures prior-node updates: building each node's
contribution list and target positions from the symbolic lists, packing, and
the fan-in itself, on the calling thread or the worker team (the scatter of
original entries before it is not included); `spd_panel_seconds` measures
node factorization, below-panel solves, and trailing updates. With phase A,
each of the three `spd_*` timers is the phase B time plus the largest time of
any phase A worker for that timer (the maxima may come from different
workers), so they still overlap and must not be added, and they no longer sum
the work of all nodes: the sum over phase A workers would exceed the wall
time.
`spd_pack_seconds` is the dense-buffer packing portion of those two times, so
it must not be added to them: it covers each contributor's `D L.T` right
operand (its left operand is read in place and not packed) and each panel's
shared right operand, which are packed by the thread that factors the node
(the calling thread, or a phase A worker), all of a node's operands before a
parallel fan-in or each one just before its update otherwise. Each strip
forms its own `L D` operand inside its work item, possibly on a worker, so
that formation is counted only in `spd_panel_seconds`.
`symbolic_seconds` of the blocked path covers the symmetry test, the first
pass, the node test and the supernodal pass (tails, contributor lists, block
offsets); earlier builds also counted the row-subtree pass, the row incidence
and the allocation of the CSC arrays there. `numeric_seconds` of the blocked
path ends when the last node is factored and includes phase A, with any CSC
array allocation it made; `assembly_seconds` then covers allocating the CSC
arrays phase A did not, writing the CSC row indices, copying the node blocks
into the CSC values, releasing the blocks, and stopping the worker team.
`failed_spd_seconds` records a discarded
SPD attempt, and is included in the subsequent LU numeric/factor time. Peak
workspace also includes that attempt's maximum live buffers, including CSC
arrays that a phase A allocation item made before the attempt was rejected.
Whether that item ran before the failure depends on scheduling, so after a
rejected attempt with phase A the estimate can vary between runs and thread
counts (no test compares it).

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
- positive-pivot validation before accepting the SPD specialization; the
  supernodal path tests each pivot and scans each panel's L entries once
  instead of testing every intermediate, with the completeness argument in
  "Blocked SPD panels";
- structural and exact numerical singular detection;
- rejection of non-finite inputs by default;
- overflow checks during scaling and row updates (the dense panel path checks
  candidates, multipliers, kernel results, and retained U rows; see "Dense
  LU panels");
- both LU engines test every pivot candidate for finiteness, so a non-finite
  candidate raises the non-finite error even when it is the only candidate
  of its column (a NaN never wins the maximum comparison, and skipping it
  would report the column as singular);
- componentwise iterative refinement with final-error and convergence checks.

This is not a rank-revealing factorization and does not estimate condition
numbers. Very ill-conditioned matrices can still lose forward accuracy even
when their backward residual is small.

## Current limitations and next optimization target

- native matrix values are `float64` only;
- no complex native factors yet (the SciPy backend handles them);
- ordering and pivot selection are single-threaded;
- the left-looking CSC engine is a serial, scalar Gilbert-Peierls loop
  without supernodes or symmetric pruning. With the search order, search
  formulation and storage reserve of "Left-looking CSC LU", the row-permuted
  20,000-row band with `COLAMD` took 0.71-0.76 times SuperLU's time with
  eight threads (0.74-0.76 with one) in three gate runs, against 1.14-1.17
  before, and 0.80-0.82 times when run alone in a fresh process (1.23-1.28
  before, interleaved runs); its numeric phase takes about 5 ms against
  SuperLU's 9.5 ms for its whole factorization, so COLAMD (about 8 ms, as
  for SuperLU) is now most of its time. The search start order is a
  heuristic for bands, and it costs on short columns in some orders: a
  nonsymmetric tridiagonal matrix of 200,000 rows with `MMD_AT_PLUS_A` takes
  6-7% longer in its numeric phase than in earlier builds (1.5-3% end to
  end; see "Left-looking CSC LU");
- the left-looking storage reserve extrapolates each factor array's size
  from the first `max(256, n / 16)` columns, so a factor whose later columns
  are much sparser than its first ones keeps unused (committed, not
  resident) capacity up to the ceiling `1.15 * max(2n, AMD estimate,
  2 * nnz(A)) + n` entries per array, and one whose later columns are denser
  grows 1.5-fold (doubling with libstdc++ and libc++) past it, as in earlier
  builds;
- `NATURAL` now runs the probe (see "Engine selection for `NATURAL`"), and
  the narrow bands of the benchmark take 0.61-0.63 times SuperLU's time
  instead of 2.2-2.4. What remains: inputs with a few dense rows (a band
  with a dense block of up to 192 rows, small arrows) stay in the row
  engine although the CSC engine would be faster; the skipped probe's
  pattern pass costs up to 2% where earlier builds started the row engine
  at once; fill whose reach exceeds the 192-row limit, a long fill region
  that the projected-work rule stops late, or fill lasting longer than
  the 192-column horizon of the cost-aware rule after a long cheap
  prefix still restarts the probe late and discards an accepted prefix
  (1.2-1.9 times the earlier time in the cases measured; the worst, a
  band of half-width 30 followed by a 2-D grid of side 86, restarts 218
  columns before the end); and 2-D
  and small 3-D grids restart after a short probe (0.04-1.3 ms) although
  2-D grids up to side 45 would factor in the CSC engine at 0.45-0.71
  times the row engine's time with a larger work budget;
- `COLAMD` does not apply the `NATURAL` rules (the pattern check, the
  multiplier limit with its cost-aware restart, and the projected work),
  so its probe still runs until the wide column when an input column is
  wider than 192 rows, and still accepts inputs whose rows the row engine
  would hold dense (with `NATURAL`, completing such probes took 1.2-3.0
  times the row engine's time on wide bands and dense blocks of 64 to 128
  rows); applying the rules to `COLAMD` needs its own measurements;
- the row engine is part of `factorize_typed`, so any change to that
  function, or elsewhere in `sparse_lu.cpp`, can move the row engine's
  timing by a few percent through code placement alone (1.01-1.08 on
  unchanged paths with `NATURAL` and `MMD_AT_PLUS_A`; see "Engine selection
  for `NATURAL`"). Moving the row engine into its own function or
  translation unit is a candidate follow-up;
- one-thread efficiency depends on the `avx2-fma` kernel: with
  `SPSOLVE_FORCE_KERNEL=sse2` (the path of x86-64 CPUs without AVX2 and FMA)
  the `COLAMD` gate run failed the 1.5 one-thread criterion on
  `sparse-dense-border` (1.54) and the four- and eight-block dense LU cases
  (1.53 and 1.52), and connected and five-block dense LU sat at 1.48, while
  the `MMD_AT_PLUS_A` run passed with dense LU at 1.37-1.45 (single runs
  under Python 3.14; see "Baseline kernel" in `docs/benchmarking.md`). A
  `COLAMD` re-run at `4b9b8ef` under Python 3.15 and SciPy 1.18.1 failed it
  on five cases (`sparse-dense-border` 1.59, row-pivoted dense LU 1.56, the
  four-, five- and eight-block cases 1.54, 1.53 and 1.52; connected dense LU
  1.47), every eight-thread ratio staying at or below 0.92 (see "Re-run at
  `4b9b8ef`" there);
- general LU retains serial per-pivot panel work: the block column scan, the
  right-looking block update (one `axpy_sub` per later panel column; a
  recursively blocked panel would touch the block less often), and
  pivot-tail materialization (one `axpy_sub` per earlier panel pivot);
  dense rows still keep span-wide tails in separately allocated storage.
  End to end, eight threads take 0.72-0.80 times the one-thread time on the
  dense LU benchmark cases (connected dense LU of size 600 with
  `MMD_AT_PLUS_A`: 7.7 instead of 10.0 ms; medians of the final gate runs in
  `docs/benchmarking.md`);
- a row that becomes dense during the factorization stays listed in the
  sparse column lists by its earlier sparse entries, which the candidate
  scan still visits (rows dense at setup are no longer listed);
- dense-row classification measures density over a row's whole remaining
  span, so a row that is dense in a prefix but has one far column (for
  example, a bordered system ordered border-last) has a wide span: it stays
  sparse, or its dense tail covers that whole span;
- the supernodal LDL^T symbolic phase is serial, and its first pass
  (elimination tree and column counts by row-subtree walks, shared with the
  simplicial kernel) still takes one step per entry of `L`: 2.9 of the 5.3 ms
  symbolic time on 3-D Poisson of side 20 with `COLAMD` (as measured when the
  supernodal symbolic phase was introduced; the final gate runs report
  5.1-5.3 ms in all at eight threads), where the node tails and their sorting
  take 1.6 ms and the contributor lists 0.7 ms. A column count algorithm in
  nearly `O(nnz(A))` time (Gilbert, Ng and Peyton) would remove most of the
  first pass;
- supernodal LDL^T keeps serial work around its parallel phases in phase B:
  the top nodes are factored one at a time, each panel solves the node's own
  rows `[end, width)` on the calling thread before its strip phase (for a
  dense node, that is the whole below-panel solve), in baseline (SSE2)
  arithmetic outside the kernel table, a node whose fan-in falls below the
  parallel threshold applies its contributions on the calling thread, and
  target positions and operand packing run on the calling thread even before
  a parallel fan-in. A dense top node therefore scales poorly (four dense
  600-wide SPD blocks took 9.3 ms in phase B at eight threads and 11.3 ms on
  one thread, before the forest rule made them subtrees). On 3-D Poisson of
  side 20 with `COLAMD` at eight threads the numeric phase takes about 22 ms
  (20.6-23.1 ms in the final gate runs), about 6 ms of it in phase A (17
  subtrees and the CSC allocations) and the rest for the 38 top nodes; when
  phase A was introduced (23 ms in all), the top nodes' 17 ms were 8.0 ms for
  the strips of 19 parallel fan-ins, 1.7 ms for the other 19 fan-ins, 1.9 ms
  for block setup and target positions, and mostly panel work otherwise.
  Strips cut each contribution's rows by node position, so a sparse
  contribution is applied in short parts, and every strip scans
  the node's contribution list; a panel's rank-32 update also reads and writes
  the whole trailing block once per panel, which limits its eight-thread
  scaling on large dense nodes; the workers of the i5-13600K also land on its
  efficiency cores. Phase A balances subtrees by an estimate and starts them
  largest first, but a subtree is never split between workers, and the forest
  rule is a heuristic for this code's phase B efficiency;
- the blocked LDL^T path keeps every node block until assembly, up to
  `2*nnz(L) + n` doubles beside the CSC factor (1.14 to 2 times the L entries
  on the benchmark cases), and fresh block memory costs page faults and a
  release on every factorization: four dense 600-wide SPD blocks, whose nodes
  have no contributors to read in place, ran 1-2% slower (1.5% to 2.2% with
  one thread, 0.6% to 1.4% with eight, in two sets of interleaved runs when
  the block storage was introduced) than with a single reused node
  rectangle. A failed SPD attempt on a symmetric indefinite matrix also
  allocates the whole node block array, and touches its pages up to the
  failing node (in phase A, up to the failing nodes of every subtree
  started), and, when phase A ran, may have allocated the CSC arrays too,
  before falling back to LU (earlier builds held one node
  rectangle and allocated the CSC arrays before every attempt);
  `peak_workspace_bytes` reports it (with phase A, the count of those CSC
  arrays depends on scheduling). Allocating the CSC arrays and releasing
  the blocks still fault in and free fresh pages on every factorization
  (about 3.6 and 0.8 ms on 3-D Poisson of side 20 with `COLAMD` when phase A
  was introduced, the former hidden in phase A when it runs; the final build
  reports `assembly_seconds` of about 7 ms with one thread and 3 ms with
  eight: medians of 21 factorizations of 7.3 and 3.0 ms here and 7.07 and
  2.89 ms in an independent review, with 2.9-4.0 ms recorded at eight threads
  by the final gate runs);
- the `avx2-fma` kernel does not fuse the LU flush (`running` order): the
  scalar U-entry walks of sparse pivot rows (pivot-tail materialization and
  dense-row updates outside panels) apply the same terms outside the kernel
  table and must round identically, so dense LU gains only from the wider
  vectors (see "Dense kernels");
- there is no out-of-core mode or symbolic-factor reuse across matrices with
  the same pattern.

Default promotion remains contingent on repeated release-build results on both
Windows and Linux, as described in the benchmark notes. The native backend
remains opt-in. At the end of the performance roadmap (`105fe2c`, 2026-10-09;
Windows x64 release build on an i5-13600K with `avx2-fma`), the
`MMD_AT_PLUS_A` gate with `--extended --dense-lu` passed three consecutive
runs, and the `COLAMD` runs (the default `permc_spec`) failed only on the
row-permuted band. With the left-looking search order, search formulation
and staged storage reserve of "Left-looking CSC LU", three consecutive
`COLAMD` gate runs and three consecutive `MMD_AT_PLUS_A` gate runs passed
every case on the same machine (see "Current status" in
`docs/benchmarking.md`). A re-run at `4b9b8ef` (2026-10-10, Python 3.15.0,
NumPy 2.5.3, SciPy 1.18.1) repeated that: three consecutive passing runs of
each gate, every ratio within -0.03 to +0.05 of the Python 3.14 runs.
Linux x64 release builds have not been measured since these changes (CI runs
the tests and the quick benchmark without `--gate`), so `NATIVE_DEFAULT` in
`src/spsolve/sparse_lu.py` stays `False`. `NATURAL` now runs the probe (see
"Engine selection for `NATURAL`"), which does not change either gate. The
next target is a Linux gate run.
