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

A case is eligible for the scaling criterion when native one-thread
factorization takes at least 0.20 seconds and the factor reports parallel pivot
steps. Medians must satisfy all of the following:

- every scaling-eligible case reaches at least 2.0× over native one-thread;
- every representative case, including cases too small for the scaling test,
  takes no more than 1.05× SciPy SuperLU time using the same
  `MMD_AT_PLUS_A` ordering request;
- every case has relative backward residual at most `1e-12`;
- native one-thread and multithread factors produce identical row and column
  permutations;
- at least one case is large enough for the scaling criterion.

The script constrains optional BLAS implementations to one thread before
loading NumPy/SciPy. This keeps the comparison focused on factorization rather
than hidden nested BLAS teams.

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

The strict gate passed more than three consecutive times on checked Windows
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
also passes the three representative cases. The blocked SPD kernel improves
larger fronts, but still trails SuperLU for some high-fill permutations and
dense blocks, as the focused comparison below shows.

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
checked alongside timing. The COLAMD 3-D and dense-block cases still exceed
the unchanged 1.05x SuperLU criterion, so these gains do not establish universal
parity or authorize default promotion. Linux results remain required.

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
