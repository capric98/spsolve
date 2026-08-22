# Factorization benchmark and promotion gate

Run the quick smoke benchmark with:

```bash
python benchmarks/factorization.py --quick --repeats 3
```

Run the default eight-thread promotion gate with:

```bash
python benchmarks/factorization.py --threads 8 --repeats 5 --gate
```

Exercise two additional structural families with:

```bash
python benchmarks/factorization.py --threads 8 --extended
```

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
for numerically symmetric positive-definite matrices, left-looking CSC LU for
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
also passes the three representative cases, but the current simplicial SPD
kernel remains slower than SuperLU on the extended 3-D Poisson case with that
ordering. A blocked supernodal kernel is still the likely next improvement for
such high-fill permutations.

Do not switch `NATIVE_DEFAULT` in `src/spsolve/sparse_lu.py` until the full gate
passes in at least three consecutive release-build runs on both Windows and
Linux x64, correctness tests pass, and no matrix family shows a material
regression. The backend therefore remains opt-in after this Windows result.
