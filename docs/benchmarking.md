# Factorization benchmark and promotion gate

Run the quick smoke benchmark with:

```bash
python benchmarks/factorization.py --quick --repeats 3
```

Run the default eight-thread promotion gate with:

```bash
python benchmarks/factorization.py --threads 8 --repeats 5 --gate
```

Exercise additional structural families with:

```bash
python benchmarks/factorization.py --threads 8 --extended
```

`--extended` adds a 3-D Poisson matrix, a narrow nonsymmetric band, and
`sparse-dense-border`: a 20,000-row nonsymmetric band (2,500 with `--quick`)
bordered by a 600-row dense block (200 with `--quick`) in the trailing rows and
columns, with 4 random couplings of magnitude below 0.01 per border row and
per border column into the band. The few couplings keep band fill modest, so
the case measures a dense block embedded in a large sparse matrix rather than
coupling-driven fill.

Add the dense LU families with:

```bash
python benchmarks/factorization.py --threads 8 --dense-lu
```

This adds a connected dense general matrix (`dense-lu`), its row-pivoted copy
(`dense-pivoted`), four, five, and eight disconnected copies of it
(`dense-lu-blocks`, `dense-lu-blocks-5`, `dense-lu-blocks-8`), and four exactly
symmetric, diagonally dominant dense SPD blocks (`dense-spd-blocks`) for the
LDL.T path. Blocks are 600 rows (200 with `--quick`).

With `--extended --permc-spec COLAMD`, the suite also includes a 20,000-row
banded matrix whose rows are randomly permuted (`--quick` uses 2,500 rows).
This checks that row labels do not turn narrow COLAMD factors into a wide-front
classification. The case is specific to COLAMD: MMD's `A + A.T` graph changes
under independent row permutations and can produce a very different ordering.

The promotion ordering is `MMD_AT_PLUS_A`. Use `--permc-spec COLAMD` or
`--permc-spec NATURAL` to compare another public ordering without editing the
script.

Add `--json build/factorization.json` to retain machine-readable environment,
criteria, timings, fill, residual, engine selection, and pass/fail data. Fast
cases are automatically batched to at least 0.10 seconds per timing sample so
timer noise and garbage-collection jitter do not dominate; override this with
`--min-sample-seconds` when needed. Previous factors are released outside each
timed interval, so object-destruction cost is not attributed to the next
factorization.

JSON results also include COLAMD probe work and restart diagnostics. Numeric
time includes any discarded probe, while ordering time contains only the
requested ordering. The promotion criteria also apply to the permuted-band
case when it is included.

## Criteria

Every included case must satisfy all of the following on median timings:

- native factorization at the requested thread count (`--threads`, default 8)
  takes no more than 1.05× SciPy SuperLU time (`--match-tolerance`) using the
  same ordering request (`MMD_AT_PLUS_A` unless `--permc-spec` says otherwise);
- native one-thread factorization takes no more than 1.5× SciPy SuperLU time
  (`--serial-tolerance`). SuperLU is serial, so this ratio certifies algorithmic
  efficiency independently of core count and keeps the gate meaningful on
  4-vCPU CI runners;
- relative backward residual is at most `1e-12`;
- native one-thread and multithread factors produce identical row and column
  permutations.

Thread scaling is reported but no longer gated. The table and JSON still carry
`speedup` (native one-thread over native multithread time) and `eligible` (one
thread takes at least `--minimum-serial-seconds`, default 0.20 seconds, and the
factor reports parallel pivot steps), but neither affects `passed`, and the
gate no longer requires at least one eligible case. Serial optimizations such
as the dense-row panels below cut one-thread time more than multithread time,
so the speedup ratio fell below the former 2.0× requirement while absolute
times improved. A ratio that penalizes faster serial code is not a useful
promotion criterion; the one-thread SuperLU ratio checks single-core
efficiency instead. JSON records this as `criteria.speedup_enforced: false`
alongside the historical `criteria.minimum_speedup`.

The table's `N/S` and `N1/S` columns are the multithread and one-thread
native/SuperLU ratios. `kernel` (JSON `dense_kernel`) names the dense kernel a
factor reports, or `unknown` when the backend does not report one; JSON
`environment.dense_kernel` and `environment.forced_kernel` record the package
setting and the `SPSOLVE_FORCE_KERNEL` override, when available.

The script constrains optional BLAS implementations to one thread before
loading NumPy/SciPy. This keeps the comparison focused on factorization rather
than hidden nested BLAS teams. Inputs are canonicalized once before timing
(duplicates summed, indices sorted), so neither backend pays sorting inside
the timed region.

### Dense kernel selection

Blocked LU flushes and supernodal LDL.T updates run on the dense kernel the
extension selects at startup: `avx2-fma` on CPUs with AVX2 and FMA, otherwise
`sse2` on x86-64 (see [the design notes](native-lu.md#dense-kernels)). Timings
therefore depend on the CPU's kernel, and the `kernel` column shows which one
ran. Set `SPSOLVE_FORCE_KERNEL` before starting the script to pin one, for
example to compare a CI runner with a workstation or to measure the kernel's
own contribution:

```bash
SPSOLVE_FORCE_KERNEL=sse2 python benchmarks/factorization.py --threads 8 --repeats 5 --extended --dense-lu
```

Forcing a kernel also makes factors reproducible across machines: general LU
factors are bitwise identical under every kernel on x86-64, while LDL.T
factors from `avx2-fma` differ from `sse2` and `scalar` in the last bits.

On a Windows x64 release build (i5-13600K, five batched samples,
`--extended --dense-lu`, `MMD_AT_PLUS_A`), the default `avx2-fma` kernel
reduced one-thread medians against forced `sse2` in the same build by 14-24%
for irregular, pivot-heavy, 3-D Poisson, dense LU (connected, pivoted, and
blocked) and dense SPD cases; eight-thread medians moved within about ±4%,
because their remaining time is serial pivoting, pivot-tail materialization,
and setup rather than tile updates. Cases without blocked dense work were
unchanged.

## Poisson bottleneck and resolution

Profiling the original 8,100-row 2-D Poisson case separated two costs that the
old aggregate timer obscured:

- the hash-graph minimum-degree implementation spent about 0.066 seconds in
  ordering, versus roughly 0.001 seconds after switching to the compact AMD
  kernel;
- after that ordering fix, the general row-hybrid engine still spent about
  0.041 seconds on per-row sparse merges, active-column bookkeeping, and
  synchronization for a matrix with narrow, predictable SPD structure.

Adding more worker threads could not remove those serial setup and dependency
costs. The accepted fix was therefore adaptive: use sparse simplicial LDL.T
for exactly symmetric positive-definite matrices, left-looking CSC LU for
narrow nonsymmetric fronts, and reserve threaded row updates for genuinely
wide fronts. The current 2-D Poisson factor has about 7% more stored entries
than SuperLU, but its approximately 0.0051-second factorization time is now
comparable to the measured 0.0077-second SuperLU time.

## Current status

The tables below were measured before the `avx2-fma` kernel existed, on the
`sse2` kernel or its predecessor; runs on an AVX2 machine now default to
`avx2-fma` and are faster on dense-update-heavy cases (see "Dense kernel
selection" above). Force `SPSOLVE_FORCE_KERNEL=sse2` to compare with them.

Before the dense LU panel change, the strict gate passed more than three
consecutive times on checked Windows
x64 release builds with Python 3.14. The final run produced:

| case | native 1 thread | native 8 threads | SuperLU | speedup | native/SuperLU | engine |
|---|---:|---:|---:|---:|---:|---|
| Poisson 2-D, n=8,100 | 0.0050 s | 0.0051 s | 0.0077 s | 0.99x | 0.66x | symmetric LDL.T CSC |
| irregular, n=2,500 | 0.4200 s | 0.1037 s | 0.3118 s | 4.05x | 0.33x | row hybrid |
| pivot-heavy, n=2,000 | 1.1724 s | 0.2406 s | 0.3956 s | 4.87x | 0.61x | row hybrid |

An extended five-repeat run also passed for a 3-D Poisson matrix (0.34x
SuperLU time) and a narrow nonsymmetric banded matrix (0.75x SuperLU time).
These timings are illustrative, not cross-platform guarantees; CPU scheduling,
compiler, allocator, and SciPy build all matter.

These results use `MMD_AT_PLUS_A`. An explicitly requested `COLAMD` permutation
also passed the three representative cases. The blocked SPD kernel improves
larger fronts. It trailed SuperLU on the `COLAMD` 3-D Poisson case and on
dense SPD blocks until the contributor and panel steps described under
"Blocked SPD comparison" below; since then, the only case that still fails
in the full `--extended --dense-lu` runs is the `COLAMD` row-permuted band,
a general LU case (1.14-1.18x).

### Dense LU panel comparison

The optional `--dense-lu` flag adds connected dense general matrices, row-pivoted
dense matrices, four, five, and eight disconnected dense general blocks
(`dense-lu-blocks`, `dense-lu-blocks-5`, `dense-lu-blocks-8`), and four exactly
symmetric dense SPD blocks (`dense-spd-blocks`). It respects the requested
ordering and runs alongside the existing families; `--quick` uses blocks of
size 200, while the full run uses size 600. JSON results include the LU panel
flush, support, executed-work, and workspace counters described in
[the design notes](native-lu.md).

Sequential Windows x64 release comparisons with five batched samples,
one-thread BLAS, and identical matrix/order requests measured these rounded
medians for the dense-row panel change:

| case | ordering | previous 1 thread | panel 1 thread | previous 8 threads | panel 8 threads | SuperLU |
|---|---|---:|---:|---:|---:|---:|
| four dense general blocks, size 600 | COLAMD | 0.580 s | 0.127 s | 0.189 s | 0.106 s | 0.039 s |
| connected dense general, size 600 | COLAMD | 0.071 s | 0.025 s | 0.030 s | 0.018 s | 0.010 s |
| row-pivoted dense general, size 600 | MMD_AT_PLUS_A | 0.071 s | 0.024 s | 0.030 s | 0.018 s | 0.011 s |
| irregular, n=2,500 | COLAMD | 0.729 s | 0.245 s | 0.166 s | 0.124 s | 0.271 s |
| pivot-heavy, n=2,000 | MMD_AT_PLUS_A | 0.665 s | 0.212 s | 0.147 s | 0.116 s | 0.241 s |

Sparse fill followed by dense conversion improved in a 289-row bordered
system, and a 20,000-row nonsymmetric band followed by a 193-row dense block
improved from about 36 to 30 ms with eight threads. Narrow NATURAL band
controls improved after removing an unused input-pattern copy; CSC and SPD
controls retain their existing engines. Upper triangular families with only
8, 31, 32, 40, or 64 initially dense rows also retained or improved timings,
including cases with no nonempty tail flush.

These are absolute-time improvements, with substantially larger serial gains.
Some main-family runs now fail the then-current 2.0x thread-scaling requirement
(since dropped; see Criteria):
COLAMD irregular measured 1.96–1.98x across two comparisons. MMD pivot-heavy
varied from 1.82x to roughly 2.00x, making that result sensitive to timing
variation. Dense cases still exceed the 1.05x SuperLU requirement;
the native backend remains opt-in. Dense panels reduce repeated row-update
barriers, but sequential pivots, pivot-tail materialization, and retained row
storage limit further speedup. No eligibility or promotion threshold was
relaxed at that time.

The four-block COLAMD example packs at most 592 tail columns, with about
0.51 MB of panel-owned buffers; the connected 600-row example uses about
0.32 MB. These diagnostics exclude the existing row storage and factors.
A representative four-block factor reported about 58 ms in its numeric
loop against about 106 ms end to end, leaving substantial setup and retained
storage costs beyond the blocked update itself.

The later contiguous panel path (retained dense pivot rows, `axpy_sub`
materialization, segment packing, and lazy U rows; see
[the design notes](native-lu.md#dense-lu-panels)) lowered row-hybrid times
on the same machine with the `avx2-fma` kernel, with bitwise identical
factors: connected dense LU of size 600 from about 17.7 to 14.4 ms at eight
threads (numeric phase 10.0 to 6.3 ms), irregular and pivot-heavy by about
20% at one thread, and the four-, five-, and eight-block dense cases by about
20%. The tables above remain historical.

The panel block that followed (column-major `members x 32` block factored
in place, gappy dense tails split into column segments, and the flush's L
and U row assembly run as work items of its parallel pass; see
[the design notes](native-lu.md#dense-lu-panels)) again kept the factors
bitwise identical. Interleaved runs against the previous build on the same
machine (`avx2-fma`, medians of five rounds) measured connected dense LU of
size 600 (`MMD_AT_PLUS_A`) at 15.4 ms instead of 16.6 ms with one thread
(numeric phase 7.8 to 6.5 ms) and 14.0 ms instead of 15.5 ms with eight
(numeric 6.7 to 4.6 ms); five 600-wide blocks at 67.6 instead of 78.6 ms
with eight threads; irregular (n=2,500) and pivot-heavy (n=2,000) with
`MMD_AT_PLUS_A` 27% and 23% faster with eight threads, and 11-14% with
one. A dense 400-wide block whose rows also share a column 800 columns to
its right (NATURAL) packs 369 instead of 1,169 columns per flush and runs
29% faster with one thread. In the full benchmark, `sparse-dense-border`
packs at most 611 (`MMD_AT_PLUS_A`) and 1,913 (`COLAMD`) columns per flush,
down from 643 and 2,327.

The setup rework that followed (row maxima accumulated during validation,
column maxima with four running maxima, an O(nnz) symmetry walk instead of a
binary search per entry, and a count-then-fill row-engine setup that fills
dense tails directly; see [the design notes](native-lu.md#factorization-pipeline))
kept every factor, statistic, and error message bitwise identical. Interleaved
runs against the previous build (`avx2-fma`, medians of five rounds) reduced
the time outside `ordering_seconds` and `factor_seconds` for connected dense
LU of size 600 from 8.2 to 3.0 ms (`MMD_AT_PLUS_A`; 8.0 to 2.9 ms with
`COLAMD`) and for five 600-wide blocks from 40.1 to 15.5 ms; end to end, the
connected case ran in 11.3 instead of 15.9 ms with one thread and 8.6
instead of 14.2 ms with eight, and the five blocks in 56.7 instead of 78.4 ms
and 41.9 instead of 68.1 ms. Four exactly symmetric dense SPD blocks of size
600 ran 19% faster (53.5 to 43.3 ms), mostly from the symmetry test, and
`sparse-dense-border` 3% faster with eight threads; sparse controls
(irregular, pivot-heavy, banded, row-permuted band, 2-D and 3-D Poisson)
stayed within 1%. One-thread `numeric_seconds` of the dense cases moved
within a few percent either way in interleaved A/B (one sequential run
showed up to 7%), because the factor loop now first-touches its L and U
storage instead of reusing memory freed by the discarded setup lists: page
faults per connected factorization fell from about 3,400 to 2,300 overall
but rose from about 540 to 1,330 inside the loop. In full benchmark runs
every `MMD_AT_PLUS_A` case except `dense-spd-blocks` passed the gate,
including all dense general LU families; `dense-spd-blocks` sits at the
1.05x boundary (1.05x and 1.07x in two runs). The `COLAMD` run still failed
on 3-D Poisson (1.07x, within 1% of the previous build in interleaved
runs), the row-permuted band (1.14x), and the dense SPD blocks (1.15x, down
from 1.44x). One run on one platform does not meet the promotion rule
below.

### Blocked SPD comparison

Two sequential Windows x64 release comparisons used five batched samples per
case, one-thread BLAS, and identical matrix/order requests before and after the
blocked SPD change. The rounded eight-thread medians were:

| case | ordering | previous native | blocked native | SuperLU |
|---|---|---:|---:|---:|
| Poisson 3-D, side 20 | COLAMD | 0.226 s | 0.147 s | 0.134 s |
| Poisson 3-D, side 20 | MMD_AT_PLUS_A | 0.075 s | 0.045 s | 0.188 s |
| four dense SPD blocks, size 600 | MMD_AT_PLUS_A | 0.085 s | 0.052 s | 0.039 s |
| Poisson 2-D, side 90 | COLAMD | 0.0082 s | 0.0084 s | 0.0105 s |
| Poisson 2-D, side 90 | MMD_AT_PLUS_A | 0.0044 s | 0.0044 s | 0.0073 s |

One-thread Poisson 3-D COLAMD improved from about 0.224 to 0.175 seconds, and
dense SPD blocks from 0.085 to 0.064 seconds. The largest node must be at least
320 columns wide to select blocked work: measured nodes of width 193–256 did
not amortize the extra symbolic and packing costs, while width 320 and larger
improved serial and threaded timings. A cheap tree/count width bound keeps
narrow cases on the original symbolic/numeric path. A second structural guard
requires at least 3,072 useful internal update terms per whole-matrix column,
using the node formula in [the design notes](native-lu.md). The cheap tree/count
runs bound this work before exact pattern construction; exact nodes decide
final selection.

Mixed-component comparisons exposed why maximum width alone was insufficient.
A 320-column dense block beside a 100,000-row tridiagonal component previously
routed every narrow node through the panel driver, roughly doubling native
factor time despite unchanged 502,718 factor entries. The work guard retains
simplicial processing for this family. A 600-column block with 10,000 narrow
rows supplies enough useful work to retain measured serial/threaded gains;
a 2,000-column block beside 100,000 narrow rows remains eligible because its
cubic work dominates. A fixed wide-column fraction would reject useful cases
of this kind.

The COLAMD 3-D case adds roughly 15 ms of pre-numeric symbolic work and spends
about 24 ms packing operands within its numeric phase. Its executed update
terms are within 0.5% of the simplicial count; actual intersections avoid
padding sparse contributors with structural zeros. Estimated simultaneous
vector buffers increase from 30.3 to 48.3 MB, including a largest node rectangle
of 335,146 doubles (2.68 MB). Dense-block buffers increase from 11.7 to 20.6 MB,
with a 360,000-double node rectangle (2.88 MB). These are local node workspaces,
with sparse CSC factors retained throughout.

Unrefined normal/transpose solves and identical one/eight-thread factors were
checked alongside timing. At that time, before the contributor change below,
the COLAMD 3-D and dense-block cases still exceeded the unchanged 1.05x
SuperLU criterion, so these gains did not establish universal parity or
authorize default promotion. Linux results remain required.

Packing each prior supernode once and applying it as one rank-`|Q|` update,
with a dispatch-aware parallel threshold (at least 2 MFLOP and two tiles per
thread, counting at most eight threads; see
[the design notes](native-lu.md#blocked-spd-panels)), cut the
eight-thread parallel tile phases of 3-D Poisson (side 20) from 579 to 95
with `COLAMD` and from 124 to 31 with `MMD_AT_PLUS_A`; `dense_update_flops`
is unchanged. Interleaved runs against the previous build (`avx2-fma`,
medians of five rounds) measured the `COLAMD` case at 143.6 instead of
151.9 ms with one thread and 138.4 instead of 147.8 ms with eight (prior-node
update time 83.9 to 76.2 ms), and the `MMD_AT_PLUS_A` case at 45.4 instead of
47.9 ms and 43.1 instead of 45.0 ms. Four dense SPD blocks of size 600, whose
nodes have no contributors, moved within 1.5% either way; their parallel
trailing updates fell from 60 to 40 because updates below 2 MFLOP now run on
the calling thread. Wider packed operands raised estimated peak workspace
from 48.3 to 49.3 MB (`COLAMD`) and from 24.1 to 25.4 MB (`MMD_AT_PLUS_A`).
Across three full `COLAMD` runs 3-D Poisson took 0.89-1.03x SuperLU time with
eight threads (1.07x before), so the case now sits just inside the gate;
general LU cases were unchanged within noise.

The panel step that followed (contiguous AXPY updates in the diagonal block,
an AXPY-form below-panel solve over four interleaved rows, finiteness scans of
each panel's L entries instead of per-element tests, and one parallel phase
per panel over 16-row strips that solve their own tail rows and apply the
rank-32 update; see [the design notes](native-lu.md#blocked-spd-panels))
changed LDL^T factors only in the last bits, with unchanged backward error.
Interleaved runs against the previous build (`avx2-fma`, medians of five
rounds) measured four dense SPD blocks of size 600 (`MMD_AT_PLUS_A`) at 32.4
instead of 44.0 ms with one thread and 31.0 instead of 43.1 ms with eight
(`spd_panel_seconds` 20.1 to 8.4 and 19.1 to 6.3 ms), 3-D Poisson of side 20
with `COLAMD` at 113.6 instead of 144.2 ms and 105.9 instead of 138.4 ms, and
with `MMD_AT_PLUS_A` at 33.9 instead of 45.7 ms and 31.5 instead of 43.7 ms.
A 1,000-wide dense block beside a 20,000-row chain ran in 29.0 instead of
38.8 ms with eight threads. In the full benchmark the dense SPD blocks took
0.76x (`MMD_AT_PLUS_A`) and 0.86x (`COLAMD`) SuperLU time with eight threads
and now pass the gate; 3-D Poisson took 0.17x and 0.78x. Simplicial and
general LU cases were unchanged within noise in interleaved runs. That
`MMD_AT_PLUS_A` run passed the gate for every case of
`--extended --dense-lu`; the `COLAMD` run failed only on the row-permuted
band (1.14x). One run on one platform does not meet the promotion rule below.

Keeping every supernode in one contiguous block array, reading contributors'
L rows in place, packing only `D L.T`, and copying the blocks into the CSC
factor once at the end (see [the design notes](native-lu.md#blocked-spd-panels))
changed LDL^T factors only in the last bits, with unchanged backward error;
`dense_update_flops`, `parallel_panels`, and `spd_panel_entries` are
unchanged. Interleaved runs against the previous build (`avx2-fma`, medians of
five rounds) measured 3-D Poisson of side 20 with `COLAMD` at 96.6 instead of
112.8 ms with one thread and 88.9 instead of 106.4 ms with eight (numeric
phase 83.0 to 64.1 ms, prior-node updates 69.4 to 51.5 ms,
`spd_pack_seconds` 15.8 to 1.1 ms; `assembly_seconds` rose from 0.03 to
2.3 ms for the block copy and release), and with `MMD_AT_PLUS_A` at 32.3
instead of 33.9 ms and 29.3 instead of 31.3 ms. A 1,000-wide dense block
beside a 20,000-row chain ran in 23.2 instead of 23.9 ms with eight threads.
Four dense SPD blocks of size 600 have no contributors to read in place and
pay for fresh block memory (page faults on 11.5 MB, and its release): 1-2%
slower (31.4 instead of 31.0 ms with one thread and 29.4 instead of 29.2 ms
with eight in these runs; an independent review measured 2.2% and 1.4%).
Estimated peak workspace rose from 49.3 to 62.6 MB (`COLAMD`), 25.5 to
30.3 MB (`MMD_AT_PLUS_A`), and 20.5 to 29.1 MB (dense blocks). In the full
benchmark 3-D Poisson took 0.65x (`COLAMD`, 0.78x before) and 0.15x
(`MMD_AT_PLUS_A`) SuperLU time with eight threads, and the dense SPD blocks
0.82x and 0.74x; the `MMD_AT_PLUS_A` run passed the gate for every case, and
the `COLAMD` run failed only on the row-permuted band (1.15x). Simplicial and
general LU controls stayed within 1% in interleaved runs except
`sparse-dense-border` with `MMD_AT_PLUS_A` and one thread (4-5% slower),
whose row-hybrid code did not change: builds that differ only in the SPD
code measured it at 0.98x to 1.05x of the previous build, a code-placement
effect in the shared translation unit.

Applying all contributions to a node in one fan-in phase (fixed 16-row strips
of the node rectangle run on the worker team when the node's update is large
enough, otherwise one whole contribution after another on the calling thread;
see [the design notes](native-lu.md#blocked-spd-panels)) kept every factor,
permutation, and scale bitwise identical to the previous build under all
three kernels and at 1, 2, 3, 4, 8, and 20 threads; `dense_update_flops` is
unchanged. Interleaved runs against the previous build (`avx2-fma`, medians of
five rounds) measured 3-D Poisson of side 20 with `COLAMD` at 74.6 instead of
96.8 ms with one thread and 57.1 instead of 89.7 ms with eight (numeric phase
49.3 instead of 70.8 and 32.3 instead of 64.4 ms; prior-node updates 34.4
instead of 56.1 and 19.5 instead of 51.3 ms), and with `MMD_AT_PLUS_A` at
28.3 instead of 32.3 ms and 24.6 instead of 29.5 ms (numeric 12.7 instead of
17.3 ms at eight threads). A random sparse SPD matrix (n=3,000, density 0.004,
`MMD_AT_PLUS_A`), whose dense 2,158-wide root receives 842 rank-1
contributions, ran in 71.5 instead of 105.5 ms with eight threads (prior-node
updates 5.1 instead of 39.8 ms). The one-thread gain comes from the
whole-node path itself: kernel parts of up to 16 by 128 entries instead of
16-by-16 tiles, in-place updates (through row pointers when the rows are not
contiguous) of every part whose columns are contiguous, each operand packed
just before its use, and less bookkeeping per contribution. Four dense SPD
blocks of size 600, which have no
contributors, moved by +0.5% (one thread) and -1.7% (eight threads), and a
1,000-wide dense block beside a 20,000-row chain by +1.7% and 0.0% (its
20,000 one-contributor nodes each pay about 12 ns more of bookkeeping).
General LU controls (irregular n=2,500 with both orderings, dense LU of size
600, `sparse-dense-border`, pivot-heavy, 2-D Poisson) moved between -5% and
+3% across two sets of runs, without a consistent sign; every function
outside the SPD path compiles to identical machine code (MSVC `/FAs`
listings), so these are placement effects and noise. `parallel_panels` fell
from 98 to 55 (`COLAMD`) and from 31 to 25 (`MMD_AT_PLUS_A`) at eight
threads, since a fan-in counts once per node. Estimated peak workspace rose
from 62.6 to 62.9 MB with one thread and to 66.0-66.1 MB with four or eight
(`COLAMD`), and from 30.3 to 30.7 and 32.4 MB (`MMD_AT_PLUS_A`). In the full
benchmark 3-D Poisson took 0.40-0.43x (`COLAMD`, 0.65x before) and 0.13x
(`MMD_AT_PLUS_A`) SuperLU time with eight threads, and 0.54-0.55x and 0.15x
with one thread; the `MMD_AT_PLUS_A` runs passed the gate for every case, and
the `COLAMD` runs failed only on the row-permuted band (1.14x and 1.19x).
One run on one platform does not meet the promotion rule below.

The supernodal symbolic phase and subtree scheduling that followed (node
tails and contributor lists computed from the exact parent/count node test
instead of a second row-subtree pass, an explicit tail comparison and a row
incidence; subtrees of at most `W / (2 * threads)` estimated work, or the
trees of a forest of comparable trees, factored concurrently before the top
nodes, with the CSC arrays allocated during that phase; see
[the design notes](native-lu.md#blocked-spd-panels)) kept every factor,
permutation, and scale, the stored `L` pattern, `supernodes`,
`max_supernode_width`, `largest_front`, and `dense_update_flops` bitwise
identical to the previous build under all three kernels and at 1, 2, 3, 4, 8,
and 20 threads. Interleaved runs against the previous build (`avx2-fma`,
medians of five rounds, times in ms) measured:

| case | ordering | 1 thread before | after | 8 threads before | after | symbolic before | after |
|---|---|---:|---:|---:|---:|---:|---:|
| 3-D Poisson, side 20 | COLAMD | 76.8 | 65.9 | 57.1 | 35.5 | 17.1 | 5.3 |
| 3-D Poisson, side 20 | MMD_AT_PLUS_A | 29.4 | 25.4 | 26.2 | 16.5 | 8.1 | 3.2 |
| four dense SPD blocks, size 600 | MMD_AT_PLUS_A | 33.4 | 29.8 | 31.3 | 20.4 | 9.1 | 4.2 |
| four dense SPD blocks, size 600 | COLAMD | 35.8 | 32.4 | 33.8 | 23.6 | 9.3 | 4.3 |
| 1,000-wide block beside a 20,000-row chain | NATURAL | 28.6 | 25.7 | 24.7 | 21.7 | 8.8 | 5.6 |
| SPD blocks 330, 400, 450 and a 12^3 grid | MMD_AT_PLUS_A | 13.6 | 11.6 | 13.5 | 8.3 | 3.5 | 1.8 |
| random SPD, n=3,000, density 0.004 | MMD_AT_PLUS_A | 72.0 | 65.6 | 43.1 | 32.0 | 14.2 | 3.4 |

With eight threads the numeric phase of 3-D Poisson took 22.0 instead of
32.2 ms with `COLAMD` (17 subtrees in about 5.7 ms) and 8.7 instead of
13.4 ms with `MMD_AT_PLUS_A` (9 subtrees), and the dense SPD blocks 4.5
instead of 10.1 ms (each block on its own worker). With one thread
`assembly_seconds` rose (3-D Poisson with `COLAMD` 2.3 to 7.5 ms) because the
CSC arrays are now allocated there rather than in the symbolic phase. The
one-thread numeric phase of the blocked cases moved by 3-5% between two
builds whose blocked-path machine code was identical (MSVC `/FAs` listings;
they differed only in the simplicial loop below), a code-placement effect;
the table uses the final build. The simplicial kernel's loop now reads
through local base pointers: with the restructured driver function, MSVC
reloaded the data pointers of the factor vectors after every store of its
innermost loop, which made simplicial factors 3-7% slower (2-D Poisson with
`COLAMD`, 3-D Poisson of side 16 with `COLAMD`, a 320-wide block beside a
100,000-row chain); with the hoisted pointers they ran at 0.97-1.00x of the
previous build. The LU controls (dense LU of size 600, `sparse-dense-border`,
irregular and pivot-heavy with both orderings) moved by -9% to +5% with
overlapping per-round ranges; their machine code differs only in member
offsets and the executor's worker-limit test. In the full benchmark 3-D
Poisson took 0.08x (`MMD_AT_PLUS_A`, 15-16 ms) and 0.26x (`COLAMD`, 36 ms)
SuperLU time with eight threads and 0.12x and 0.47-0.48x with one, and the
dense SPD blocks 0.48-0.49x and 0.55-0.56x (0.73-0.74x and 0.75-0.81x with
one thread); the `MMD_AT_PLUS_A` runs passed the gate for every case, and the
`COLAMD` runs failed only on the row-permuted band (1.15x and 1.18x).
Estimated peak workspace fell
on every blocked case (3-D Poisson with `COLAMD` from 66.1 to 53.3 MB at
eight threads, dense SPD blocks from 29.3 to 24.3 MB), because the row
incidence is gone and the CSC arrays are allocated later.

The row-permuted 20,000-row COLAMD band exposes a separate classifier problem.
Using an unrelated AMD analysis previously reported a front of 11,309 despite
only 359,878 factor entries and narrow actual fronts. Replacing that analysis
with bounded numeric probing reduced native eight-thread factorization from
0.0846 seconds to 0.0218 seconds in a three-repeat Windows release comparison.
The probe observed a maximum column reach of 31 and retained the CSC engine;
factor fill and residual were unchanged. SuperLU took 0.0167 seconds, so the
new extended case still fails the unchanged 1.05x gate at about 1.30x. The
misclassification and extra ordering pass are fixed; the remaining CSC cost
still needs optimization. Native remains opt-in.

Do not switch `NATIVE_DEFAULT` in `src/spsolve/sparse_lu.py` until the full gate
passes in at least three consecutive release-build runs on both Windows and
Linux x64, correctness tests pass, and no matrix family shows a material
regression. The backend therefore remains opt-in after this Windows result.
