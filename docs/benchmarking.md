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
blocked) and dense SPD cases when it was introduced; eight-thread medians
moved within about ±4%, because their remaining time was serial pivoting,
pivot-tail materialization, and setup rather than tile updates. Cases without
blocked dense work were unchanged. After the later panel, setup, and LDL^T
steps more of the time is kernel work: in the final roadmap build the same
comparison (one forced `sse2` run against the median of three `avx2-fma`
runs) gave one-thread medians 23-37% lower and eight-thread medians 6-16%
lower with `avx2-fma` on those cases, and within 2% on the others (see
"Baseline kernel" under "Current status").

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
wide fronts. The 2-D Poisson factor has about 7% more stored entries than
SuperLU, and in the final roadmap gate runs it factored in 4.6 ms against
SuperLU's 7.4 ms (`MMD_AT_PLUS_A`, eight threads).

## Current status

### State after the left-looking search change

Measured on 2026-10-09 with this change applied to `c46bda9` (the
left-looking search start order, search formulation and staged storage
reserve; see "Left-looking CSC LU" in
[the design notes](native-lu.md#left-looking-csc-lu)), on the machine and
software of the next section, otherwise idle. Three consecutive `COLAMD`
gate runs, then three consecutive `MMD_AT_PLUS_A` gate runs, one at a time
(`build/review/step6a_r2_gate_{colamd,mmd}_{1,2,3}.json`; not tracked):

```bash
python benchmarks/factorization.py --threads 8 --repeats 5 --gate --extended --dense-lu --permc-spec COLAMD
python benchmarks/factorization.py --threads 8 --repeats 5 --gate --extended --dense-lu
```

All six runs passed every case. Every relative backward residual was at most
5.7e-16, and every case produced identical one- and eight-thread
permutations. Medians of the three runs in milliseconds, ratios as in the
next section (the median of the per-run ratios, with the range); `COLAMD`:

| case | native 1 thread | native 8 threads | SuperLU | N/S | N1/S | engine |
|---|---:|---:|---:|---:|---:|---|
| 2-D Poisson, n=8,100 | 8.04 | 7.97 | 10.4 | 0.76 (0.76-0.79) | 0.77 (0.77-0.82) | simplicial LDL^T |
| irregular, n=2,500 | 125 | 69.2 | 270 | 0.25 (0.25-0.26) | 0.46 (0.46-0.47) | row hybrid |
| pivot-heavy, n=2,000 | 85.6 | 53.8 | 201 | 0.27 | 0.43 (0.42-0.45) | row hybrid |
| 3-D Poisson, n=8,000 | 64.5 | 34.8 | 137 | 0.25 (0.25-0.26) | 0.47 (0.46-0.47) | supernodal LDL^T |
| nonsymmetric band, n=8,000 | 3.33 | 3.32 | 4.61 | 0.72 (0.71-0.73) | 0.72 (0.72-0.74) | left-looking CSC |
| `sparse-dense-border`, n=20,600 | 722 | 368 | 562 | 0.65 (0.65-0.66) | 1.29 (1.28-1.31) | row hybrid |
| row-permuted band, n=20,000 | 12.9 | 12.9 | 17.3 | 0.75 (0.71-0.76) | 0.75 (0.74-0.76) | left-looking CSC |
| dense LU, 600 | 10.3 | 7.78 | 9.88 | 0.77 (0.76-0.83) | 1.04 (1.01-1.06) | row hybrid |
| row-pivoted dense LU, 600 | 10.7 | 8.15 | 10.3 | 0.79 (0.77-0.82) | 1.05 (1.04-1.06) | row hybrid |
| four dense LU blocks of 600 | 41.2 | 31.2 | 39.8 | 0.78 (0.76-0.80) | 1.04 (1.01-1.05) | row hybrid |
| five dense LU blocks of 600 | 51.3 | 39.7 | 49.7 | 0.80 (0.75-0.81) | 1.03 (1.01-1.04) | row hybrid |
| eight dense LU blocks of 600 | 87.4 | 67.8 | 83.7 | 0.81 (0.81-0.83) | 1.05 (1.04-1.05) | row hybrid |
| four dense SPD blocks of 600 | 31.4 | 22.3 | 39.8 | 0.56 (0.56-0.58) | 0.78 (0.78-0.81) | supernodal LDL^T |

`MMD_AT_PLUS_A`:

| case | native 1 thread | native 8 threads | SuperLU | N/S | N1/S | engine |
|---|---:|---:|---:|---:|---:|---|
| 2-D Poisson, n=8,100 | 4.55 | 4.51 | 7.47 | 0.61 (0.59-0.61) | 0.61 (0.59-0.62) | simplicial LDL^T |
| irregular, n=2,500 | 71.4 | 40.6 | 190 | 0.21 (0.21-0.23) | 0.37 (0.37-0.39) | row hybrid |
| pivot-heavy, n=2,000 | 101 | 64.4 | 243 | 0.26 (0.26-0.28) | 0.41 (0.41-0.45) | row hybrid |
| 3-D Poisson, n=8,000 | 23.4 | 14.2 | 191 | 0.07 | 0.12 | supernodal LDL^T |
| nonsymmetric band, n=8,000 | 2.37 | 2.38 | 3.56 | 0.66 (0.65-0.67) | 0.66 (0.64-0.67) | left-looking CSC |
| `sparse-dense-border`, n=20,600 | 199 | 163 | 292 | 0.56 (0.55-0.57) | 0.68 (0.68-0.69) | row hybrid |
| dense LU, 600 | 9.72 | 7.47 | 10.5 | 0.70 (0.68-0.74) | 0.96 (0.91-0.98) | row hybrid |
| row-pivoted dense LU, 600 | 10.4 | 7.52 | 10.5 | 0.72 (0.71-0.72) | 1.00 (0.98-1.00) | row hybrid |
| four dense LU blocks of 600 | 39.6 | 28.4 | 40.9 | 0.69 (0.68-0.75) | 0.97 (0.96-0.97) | row hybrid |
| five dense LU blocks of 600 | 49.3 | 36.7 | 51.1 | 0.72 (0.71-0.73) | 0.96 (0.96-0.97) | row hybrid |
| eight dense LU blocks of 600 | 81.3 | 56.6 | 81.0 | 0.71 (0.69-0.72) | 1.00 (0.98-1.05) | row hybrid |
| four dense SPD blocks of 600 | 28.7 | 20.4 | 40.2 | 0.51 (0.50-0.52) | 0.72 (0.71-0.72) | supernodal LDL^T |

Only the two left-looking cases changed: the row-permuted band from 1.16
(1.14-1.17) to 0.75 times SuperLU's time with eight threads, and the
nonsymmetric band from 0.89 to 0.72 (`COLAMD`) and from 0.77 to 0.66
(`MMD_AT_PLUS_A`); see "Row-permuted band" below. Every other case gave
bitwise identical permutations, scales, `L` and `U` to `c46bda9` at one and
eight threads, with the same engines and probe decisions. Their median times
moved by -8% to +4% against the final roadmap runs below, without a
consistent sign (every function outside `factorize_left_looking` compiles
to identical machine code; MSVC `/FAs` listings). The probe of a restart
takes the same time as before (0.1 ms on pivot-heavy and irregular); its
`peak_workspace_bytes` is unchanged except for pivot-heavy with `COLAMD`,
0.50 instead of 0.42 MB, whose probe passed the second stage of the storage
reserve before restarting (see the design notes).

### State after the roadmap

These tables are superseded for the two left-looking cases by the section
above; the other cases factor identically.

Measured on 2026-10-09 at `105fe2c`, the last commit of the performance
roadmap whose Step 0 baseline is `4fce891`: Windows 10 x64 (build 19044),
Intel Core i5-13600K (6 performance and 8 efficiency cores), MSVC 19.51
release build from `python -m pip install .`, Python 3.14.8, NumPy 2.3.5,
SciPy 1.16.3 SuperLU with BLAS pinned to one thread, and the default
`avx2-fma` kernel. The machine was otherwise idle (under one core of
background load before and after every run). Each command ran three
consecutive times, one run at a time:

```bash
python benchmarks/factorization.py --threads 8 --repeats 5 --gate --extended --dense-lu
python benchmarks/factorization.py --threads 8 --repeats 5 --gate --extended --dense-lu --permc-spec COLAMD
```

All three `MMD_AT_PLUS_A` runs passed every case. All three `COLAMD` runs
failed one case only, the row-permuted band (`left-looking-csc`): eight
threads took 1.14-1.17 times SuperLU's time, while its one-thread ratio
(1.14-1.15) is within the 1.5 limit; see "Row-permuted band" below. In all six
runs every relative backward residual was at most 5.7e-16, and every case
produced identical one- and eight-thread permutations. The tables give the
medians of the three runs, times in milliseconds; each ratio is the median
of the three per-run ratios (so it can differ in the last digit from the
quotient of the median times), with the range of the three runs in
parentheses when the rounded values differ, and bold marks a failed
criterion.

`MMD_AT_PLUS_A`:

| case | native 1 thread | native 8 threads | SuperLU | N/S | N1/S | engine |
|---|---:|---:|---:|---:|---:|---|
| 2-D Poisson, n=8,100 | 4.58 | 4.59 | 7.43 | 0.61 (0.61-0.62) | 0.62 (0.60-0.63) | simplicial LDL^T |
| irregular, n=2,500 | 73.0 | 41.0 | 189 | 0.22 | 0.39 (0.38-0.39) | row hybrid |
| pivot-heavy, n=2,000 | 103 | 65.2 | 244 | 0.27 (0.26-0.27) | 0.42 | row hybrid |
| 3-D Poisson, n=8,000 | 23.7 | 15.4 | 192 | 0.08 | 0.12 | supernodal LDL^T |
| nonsymmetric band, n=8,000 | 2.79 | 2.79 | 3.59 | 0.77 (0.77-0.78) | 0.78 (0.77-0.78) | left-looking CSC |
| `sparse-dense-border`, n=20,600 | 203 | 172 | 296 | 0.58 (0.56-0.58) | 0.68 (0.68-0.69) | row hybrid |
| dense LU, 600 | 9.96 | 7.67 | 10.2 | 0.76 (0.72-0.78) | 0.97 (0.96-0.99) | row hybrid |
| row-pivoted dense LU, 600 | 10.3 | 7.47 | 10.4 | 0.71 (0.70-0.73) | 0.99 (0.97-1.00) | row hybrid |
| four dense LU blocks of 600 | 39.3 | 28.4 | 40.2 | 0.70 (0.70-0.71) | 0.98 (0.97-0.99) | row hybrid |
| five dense LU blocks of 600 | 48.8 | 35.4 | 51.5 | 0.69 (0.68-0.71) | 0.95 (0.94-0.98) | row hybrid |
| eight dense LU blocks of 600 | 81.1 | 60.4 | 79.9 | 0.76 (0.74-0.78) | 1.02 (1.01-1.02) | row hybrid |
| four dense SPD blocks of 600 | 28.5 | 19.5 | 40.5 | 0.49 (0.47-0.50) | 0.70 (0.70-0.73) | supernodal LDL^T |

`COLAMD`:

| case | native 1 thread | native 8 threads | SuperLU | N/S | N1/S | engine |
|---|---:|---:|---:|---:|---:|---|
| 2-D Poisson, n=8,100 | 8.10 | 8.11 | 10.6 | 0.77 (0.76-0.77) | 0.77 | simplicial LDL^T |
| irregular, n=2,500 | 129 | 70.8 | 274 | 0.26 (0.25-0.27) | 0.47 | row hybrid |
| pivot-heavy, n=2,000 | 84.3 | 53.6 | 202 | 0.27 | 0.42 | row hybrid |
| 3-D Poisson, n=8,000 | 65.4 | 34.6 | 139 | 0.25 (0.25-0.26) | 0.47 | supernodal LDL^T |
| nonsymmetric band, n=8,000 | 4.15 | 4.14 | 4.65 | 0.89 (0.89-0.90) | 0.89 (0.89-0.90) | left-looking CSC |
| `sparse-dense-border`, n=20,600 | 731 | 371 | 565 | 0.66 (0.65-0.66) | 1.29 (1.28-1.30) | row hybrid |
| row-permuted band, n=20,000 | 19.8 | 20.0 | 17.2 | **1.16 (1.14-1.17)** | 1.15 (1.14-1.15) | left-looking CSC |
| dense LU, 600 | 10.4 | 8.29 | 9.83 | 0.82 (0.80-0.85) | 1.05 (1.05-1.06) | row hybrid |
| row-pivoted dense LU, 600 | 10.9 | 8.22 | 10.3 | 0.79 (0.78-0.80) | 1.06 (1.03-1.08) | row hybrid |
| four dense LU blocks of 600 | 42.0 | 31.9 | 40.0 | 0.80 (0.77-0.85) | 1.05 (1.05-1.07) | row hybrid |
| five dense LU blocks of 600 | 51.9 | 39.8 | 49.8 | 0.80 (0.77-0.81) | 1.04 (1.04-1.07) | row hybrid |
| eight dense LU blocks of 600 | 87.9 | 67.6 | 83.9 | 0.81 (0.80-0.82) | 1.03 (1.03-1.06) | row hybrid |
| four dense SPD blocks of 600 | 30.9 | 22.8 | 39.5 | 0.58 (0.55-0.60) | 0.79 (0.78-0.79) | supernodal LDL^T |

These timings are illustrative, not cross-platform guarantees; CPU
scheduling, compiler, allocator, and SciPy build all matter. The JSON of each
run is `build/review/final_{mmd,colamd}_{1,2,3}.json` (not tracked).

The promotion ordering, `MMD_AT_PLUS_A`, thus passed three consecutive runs
on this Windows build; `COLAMD` did not, because of the row-permuted band
(`COLAMD` is the default `permc_spec` of `splu` and `spsolve`), until the
left-looking search change above. No Linux x64
release-build gate run has been made since these changes (CI runs
the test suite and the `--quick` smoke benchmark without `--gate`), so the
promotion rule below is not met and `NATIVE_DEFAULT` stays `False`.

### Effect of the roadmap

The Step 0 baseline (`build/review/baseline_{mmd,colamd}.json`: one run of
the same commands at `4fce891` on the same machine, before the dense-kernel
layer) failed six `MMD_AT_PLUS_A` cases (every dense LU and dense SPD family)
and nine `COLAMD` cases (also 3-D Poisson, `sparse-dense-border` on its
one-thread ratio, and the row-permuted band). Times in milliseconds, Step 0
against the medians above; the baseline is a single run, so differences of a
few percent are noise:

| case | ordering | 1 thread | 8 threads | N/S | N1/S |
|---|---|---:|---:|---:|---:|
| 2-D Poisson | MMD_AT_PLUS_A | 4.77 → 4.58 | 4.84 → 4.59 | 0.64 → 0.61 | 0.64 → 0.62 |
| irregular | MMD_AT_PLUS_A | 146 → 73.0 | 74.6 → 41.0 | 0.39 → 0.22 | 0.77 → 0.39 |
| pivot-heavy | MMD_AT_PLUS_A | 218 → 103 | 115 → 65.2 | 0.47 → 0.27 | 0.89 → 0.42 |
| 3-D Poisson | MMD_AT_PLUS_A | 56.8 → 23.7 | 44.9 → 15.4 | 0.23 → 0.08 | 0.29 → 0.12 |
| nonsymmetric band | MMD_AT_PLUS_A | 2.81 → 2.79 | 2.81 → 2.79 | 0.78 → 0.77 | 0.78 → 0.78 |
| `sparse-dense-border` | MMD_AT_PLUS_A | 236 → 203 | 194 → 172 | 0.65 → 0.58 | 0.80 → 0.68 |
| dense LU | MMD_AT_PLUS_A | 24.4 → 9.96 | 18.3 → 7.67 | 1.73 → 0.76 | 2.31 → 0.97 |
| row-pivoted dense LU | MMD_AT_PLUS_A | 24.8 → 10.3 | 19.2 → 7.47 | 1.81 → 0.71 | 2.34 → 0.99 |
| four dense LU blocks | MMD_AT_PLUS_A | 125 → 39.3 | 95.1 → 28.4 | 2.38 → 0.70 | 3.13 → 0.98 |
| five dense LU blocks | MMD_AT_PLUS_A | 305 → 48.8 | 137 → 35.4 | 2.69 → 0.69 | 5.99 → 0.95 |
| eight dense LU blocks | MMD_AT_PLUS_A | 631 → 81.1 | 259 → 60.4 | 3.22 → 0.76 | 7.85 → 1.02 |
| four dense SPD blocks | MMD_AT_PLUS_A | 66.1 → 28.5 | 54.2 → 19.5 | 1.35 → 0.49 | 1.65 → 0.70 |
| 2-D Poisson | COLAMD | 8.53 → 8.10 | 8.31 → 8.11 | 0.79 → 0.77 | 0.81 → 0.77 |
| irregular | COLAMD | 246 → 129 | 135 → 70.8 | 0.49 → 0.26 | 0.90 → 0.47 |
| pivot-heavy | COLAMD | 178 → 84.3 | 103 → 53.6 | 0.51 → 0.27 | 0.88 → 0.42 |
| 3-D Poisson | COLAMD | 177 → 65.4 | 148 → 34.6 | 1.08 → 0.25 | 1.29 → 0.47 |
| nonsymmetric band | COLAMD | 4.12 → 4.15 | 4.11 → 4.14 | 0.89 → 0.89 | 0.90 → 0.89 |
| `sparse-dense-border` | COLAMD | 1123 → 731 | 530 → 371 | 0.94 → 0.66 | 2.00 → 1.29 |
| row-permuted band | COLAMD | 20.0 → 19.8 | 19.9 → 20.0 | 1.16 → 1.16 | 1.17 → 1.15 |
| dense LU | COLAMD | 24.3 → 10.4 | 17.7 → 8.29 | 1.79 → 0.82 | 2.46 → 1.05 |
| row-pivoted dense LU | COLAMD | 25.2 → 10.9 | 20.0 → 8.22 | 1.97 → 0.79 | 2.49 → 1.06 |
| four dense LU blocks | COLAMD | 123 → 42.0 | 95.5 → 31.9 | 2.38 → 0.80 | 3.06 → 1.05 |
| five dense LU blocks | COLAMD | 308 → 51.9 | 132 → 39.8 | 2.66 → 0.80 | 6.20 → 1.04 |
| eight dense LU blocks | COLAMD | 635 → 87.9 | 275 → 67.6 | 3.38 → 0.81 | 7.79 → 1.03 |
| four dense SPD blocks | COLAMD | 68.1 → 30.9 | 57.7 → 22.8 | 1.45 → 0.58 | 1.71 → 0.79 |

The dense general LU families ran 2.3-7.8 times faster with one thread and
2.1-4.3 times with eight, the dense SPD blocks 2.2-2.3 and 2.5-2.8 times,
3-D Poisson 2.4 and 2.9 times (`MMD_AT_PLUS_A`) and 2.7 and 4.3 times
(`COLAMD`), irregular and pivot-heavy 1.9-2.1 and 1.8-1.9 times, and
`sparse-dense-border` 1.16 and 1.13 times (`MMD_AT_PLUS_A`) and 1.54 and 1.43
times (`COLAMD`). 2-D Poisson moved by 2-5% and the two bands by at most 1.4%:
the roadmap did not target the simplicial LDL^T and left-looking CSC engines
that factor them. Part of the dense gains is the `avx2-fma` kernel, which the
baseline did not have (see the next section).

### Baseline kernel (`sse2`)

One run of each ordering with `SPSOLVE_FORCE_KERNEL=sse2` in the same build
(otherwise as above) measures the baseline x86-64 path that CPUs without AVX2
and FMA take (N/S at eight threads):

| case | `MMD_AT_PLUS_A` N/S | `MMD_AT_PLUS_A` N1/S | `COLAMD` N/S | `COLAMD` N1/S |
|---|---:|---:|---:|---:|
| 2-D Poisson | 0.60 | 0.60 | 0.78 | 0.77 |
| irregular | 0.23 | 0.53 | 0.29 | 0.68 |
| pivot-heavy | 0.30 | 0.67 | 0.30 | 0.64 |
| 3-D Poisson | 0.09 | 0.17 | 0.29 | 0.66 |
| nonsymmetric band | 0.78 | 0.78 | 0.89 | 0.89 |
| `sparse-dense-border` | 0.59 | 0.70 | 0.70 | **1.54** |
| row-permuted band | n/a | n/a | **1.16** | 1.14 |
| dense LU | 0.83 | 1.37 | 0.93 | 1.48 |
| row-pivoted dense LU | 0.81 | 1.37 | 0.90 | 1.44 |
| four dense LU blocks | 0.85 | 1.43 | 0.94 | **1.53** |
| five dense LU blocks | 0.80 | 1.42 | 0.88 | 1.48 |
| eight dense LU blocks | 0.83 | 1.45 | 0.89 | **1.52** |
| four dense SPD blocks | 0.54 | 0.90 | 0.61 | 0.98 |

The `MMD_AT_PLUS_A` run passed. The `COLAMD` run failed the one-thread
criterion on `sparse-dense-border` and the four- and eight-block dense LU
cases, with the connected and five-block cases at 1.48, besides the
row-permuted band. Every eight-thread ratio except the row-permuted band's
stayed at or below 0.94. On a CPU without AVX2 and FMA the one-thread
criterion is therefore marginal for dense general LU with `COLAMD`; these are
single runs, and a runner's own CPU decides which kernel it uses. The
left-looking cases do not use the dense kernel; for their current ratios see
the first section.

### Row-permuted band (`COLAMD`)

The case is the nonsymmetric band of `banded-nonsym` at 20,000 rows (99,981
entries) with randomly permuted rows. Before the roadmap, replacing an
unrelated AMD analysis (which reported a front of 11,309) with the bounded
COLAMD probe cut native eight-thread time from 84.6 to 21.8 ms; the probe
keeps the left-looking CSC engine (maximum column reach 31) with 359,878
factor entries, the same as SuperLU. The roadmap did not optimize this
engine's numeric loop (`02ae72b` added a finiteness test of every pivot
candidate, and `1cc9dd5` reduced the probe's restart snapshot to the fields
the probe writes and sped up the validation and equilibration shared by all
engines, both with bitwise identical factors), and at its end (`105fe2c` and
`c46bda9`) the case failed the eight-thread criterion. In those gate runs
native spent 8.0-8.2 ms in COLAMD
(`ordering_seconds`), 10.8-11.3 ms in the left-looking factorization
(`numeric_seconds`, all of it the successful probe, including the copy of
its restart snapshot: 20,000 columns, 1,259,452 scalar updates), and under
1 ms outside both timers (validation, equilibration, the symmetry test,
releasing the snapshot, and the Python input copy and wrapper: about 0.45
and 0.23 ms with one thread in a separate fresh-process measurement); it
reports `symbolic_seconds` and `assembly_seconds` of zero and no parallel
steps, since the engine is serial. SuperLU takes 17.2 ms; factoring
SuperLU's own column-permuted matrix with `NATURAL` takes 9.5 ms, so its
COLAMD ordering takes about 7.7 ms (the `NATURAL` run still computes the
column elimination tree and postorder). The ordering costs are therefore
close (on a copy whose rows are relabeled by their first appearance in the
column order, native COLAMD takes 4.8 ms instead of 8.0-8.3 ms in the same
runs, and SuperLU's ordering 4.6 instead of 7.7 ms), and the gap was the
left-looking numeric phase: about 11 ms against SuperLU's 9.5 ms for its
whole factorization.

At `c46bda9` the case was sensitive to the allocator state of the process.
Run alone in a fresh process with eight threads it took 21.6-22.1 ms
(1.25-1.28 times the SuperLU time of the same runs), with about 5,100 page
faults per one-thread factorization; after the larger `sparse-dense-border`
factorizations that precede it in the benchmark it took 19.8-20.0 ms and
about 3,000 page faults. The `L` and `U` arrays started with room for
`3.3 * n` entries when no AMD estimate exists (as with `COLAMD`) and were
reallocated three times each on the way to their 159,953 and 199,925
entries, touching fresh pages at every step. SuperLU's time does not depend
on the process state.

Scratch prototypes (not part of `c46bda9`) located the remaining cost:
reserving each factor array for `2.3 * nnz(A) + n` entries, and starting
each column's depth-first search from its entries in increasing pivot
position. Relabeling the rows for memory locality without changing the
traversal order gave nothing, so the traversal order, not the locality of
the row labels, cost the time. This change (see "Left-looking CSC LU" in
[the design notes](native-lu.md#left-looking-csc-lu)) reserves the factor
arrays in two stages, extrapolating each one's size after the first
`max(256, n / 16)` columns (bitwise identical factors; here `L` and `U`
reach 159,953 and 199,925 entries without further growth), starts each
search from the column's pivoted row of smallest pivot position and then
from its rows in stored order (cheaper than the prototype's sort on bands
with pivoting), scatters the column in the pass that finds that row, and
runs the search without pushing unpivoted rows (bitwise identical to the
earlier search for the same start order). The start order changes the
factors in the last bits: on this band by at most 2.2e-16 of the largest `U`
entry, with identical permutations and pattern.

Interleaved runs against `c46bda9` (`avx2-fma`, eight threads, medians of
five rounds, six for the fresh-process band; ratios against the SuperLU time
of the same runs; times in ms):

| case | process | `c46bda9` | this change | N/S before | N/S after | numeric before | numeric after | page faults per one-thread factorization |
|---|---|---:|---:|---:|---:|---:|---:|---|
| row-permuted band | fresh | 21.6 | 14.0 | 1.26 (1.23-1.28) | 0.81 (0.80-0.82) | 12.4 | 5.0 | 5,093 to 2,770 |
| row-permuted band | after `sparse-dense-border` | 19.7 | 13.2 | 1.15 (1.13-1.16) | 0.76 (0.68-0.78) | 10.9 | 4.6 | 3,038 to 1,874 |
| nonsymmetric band, `COLAMD` | fresh | 5.34 | 4.17 | 1.16 (1.12-1.18) | 0.89 (0.88-0.92) | 3.12 | 1.95 | 1,425 to 1,045 |
| nonsymmetric band, `MMD_AT_PLUS_A` | fresh | 3.10 | 2.66 | 0.85 (0.83-0.92) | 0.76 (0.74-0.79) | 1.86 | 1.44 | 272 to 273 |

With one thread (the engine is serial) the row-permuted band took 13.9
instead of 21.7 ms in a fresh process (N1/S 1.27 to 0.80) and 12.9 instead
of 19.7 ms after `sparse-dense-border` (1.14 to 0.75). The fresh-process
nonsymmetric band with `COLAMD` also exceeded 1.05 before (1.16; 0.89 in the
benchmark order). Row-engine controls with `COLAMD` in the same kind of runs
stayed within 1% (connected dense LU of size 600: 11.2 instead of 11.3 ms
with one thread, 8.02 ms with eight in both; pivot-heavy: 95.1 instead of
94.5 ms and 55.5 instead of 55.7 ms).

A scratch build with a switch for each part of the first version of this
change (which reserved the ceiling at once) measured the fresh-process
eight-thread numeric phase of the row-permuted band at 13.1 ms with none of
them, 10.3 ms with the reserve alone, 10.0 ms with the start order alone,
11.0 ms with the search formulation alone and 5.4 ms with all three (five
rounds). On other left-looking families the one-thread numeric phase
(medians of five interleaved rounds of 13 factorizations each, or six of 21
for the last four rows, where the end-to-end time is given as well) changed
as follows:

| matrix | ordering | n | `c46bda9` | this change |
|---|---|---:|---:|---:|
| nonsymmetric band, rows in order | `COLAMD` | 20,000 | 6.47 | 3.87 |
| band of the benchmark offsets, random values, rows permuted | `COLAMD` | 20,000 | 11.3 | 7.23 |
| same, diagonally dominant | `COLAMD` | 20,000 | 10.5 | 4.52 |
| band with offsets -12, -5, -1, 0, 3, 8, 15, rows permuted | `COLAMD` | 20,000 | 22.0 | 13.8 |
| full band of half-width 3, rows permuted | `COLAMD` | 50,000 | 12.5 | 8.81 |
| same, rows in order | `COLAMD` | 50,000 | 10.0 | 6.93 |
| full band of half-width 8, rows permuted | `COLAMD` | 20,000 | 14.3 | 9.03 |
| half-width-2 band with 400 entries within 30 of the diagonal, rows permuted | `COLAMD` | 20,000 | 2.80 | 2.23 |
| 2-D convection-diffusion, 60 by 60 | `MMD_AT_PLUS_A` | 3,600 | 6.65 | 4.01 |
| `banded-nonsym` | `MMD_AT_PLUS_A` | 8,000 | 1.62 | 1.24 |
| upper triangular, 10 entries per column (end to end) | `COLAMD` | 200,000 | 29.8 (84.5) | 21.0 (72.8) |
| nonsymmetric tridiagonal (end to end) | `COLAMD` | 200,000 | 10.9 (29.1) | 10.9 (29.1) |
| same (end to end) | `MMD_AT_PLUS_A` | 200,000 | 8.71 (23.0) | 9.32 (23.6) |

The full bands have uniform random values in (-1, 1), so they pivot. The
tridiagonal matrix with `MMD_AT_PLUS_A`, which that ordering eliminates in
reverse, is the one regression: 6-7% in the numeric phase and 1.5-3% end
to end in interleaved runs, including an independent review's (see the
design notes). Where the
probe restarts, its discarded work got cheaper as well: the 2-D
convection-diffusion matrix with `COLAMD` took 16.5 instead of 17.7 ms in
its numeric phase, and a random sparse matrix (n=4,000, about three
entries per column) 52.5 instead of 54.7 ms, with overlapping per-round
ranges.

Other orderings: with `MMD_AT_PLUS_A` the row permutation destroys the
structure of `A + A.T`; native uses the row-hybrid engine (175 ms with one
thread, 147 ms with eight, 1.10 million factor entries) while SuperLU takes
5.48 seconds (4.77 million entries). With `NATURAL` native never uses the
left-looking engine, and the row-hybrid engine takes 19-22 ms against
SuperLU's 8.9-9.1 ms (2.1-2.5 times in three runs here, 2.3-2.6 times in an
independent review; the unpermuted 8,000-row band also takes 2.4 times
SuperLU's time with `NATURAL`; measured at `c46bda9`, unaffected by the
change above). A scratch prototype that also runs the `COLAMD` probe for
`NATURAL`, with the prototype's reserve and sorted search start, took
7.1-8.2 ms on the
row-permuted band (0.81-0.89 times the SuperLU time of the same runs,
8.7-9.2 ms) and 2.4 ms on the unpermuted band (0.86-0.87 times the SuperLU
time of the same run). See the limitations in
[the design notes](native-lu.md#current-limitations-and-next-optimization-target).

## History

The sections below record the per-step measurements of earlier changes, each
against the build before it, so their absolute times predate later steps.

Before the dense LU panel change, the strict gate, which then also required a
2.0x thread speedup on eligible cases, passed more than three consecutive
times on Windows x64 release builds with Python 3.14 for 2-D Poisson (0.66x
SuperLU time with eight threads), irregular (0.33x), and pivot-heavy
(0.61x), and an extended run for 3-D Poisson (0.34x) and the nonsymmetric
band (0.75x), all with `MMD_AT_PLUS_A`.

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
20%. The comparison table above remains historical.

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

## Promotion rule

Do not switch `NATIVE_DEFAULT` in `src/spsolve/sparse_lu.py` until the full gate
passes in at least three consecutive release-build runs on both Windows and
Linux x64, correctness tests pass, and no matrix family shows a material
regression. The backend therefore remains opt-in after this Windows result.
