# Working in this repository

`spsolve` is a BSD-3 Python package with a C++20 nanobind extension: a native
sparse LU / LDL^T factorization and a native sparse triangular solver, with no
BLAS or external factorization library. SciPy is the container layer and the
`backend="scipy"` fallback. This file is for coding agents and humans who want
to make a change without re-deriving the project's constraints.

Read first, in this order: `README.md`, `docs/native-lu.md` (engine design and
`stats` semantics), `docs/algorithms.md` (the mathematics each engine
implements), `docs/benchmarking.md` (the promotion gate), `docs/api.md`.

## Layout

| Path | What lives there |
|---|---|
| `src/spsolve/*.py` | Public API: `splu`, `spsolve`, `factorized`, `spsolve_triangular`, `spparms` (thread/order/dense-kernel settings). Canonicalizes input, maps native errors to `LinAlgError`, runs iterative refinement. |
| `src/spsolve.cpp` | nanobind module `_spsolve`: binds the triangular kernels and the dense-kernel selection, and calls `bind_sparse_lu`. |
| `src/sparse_lu.cpp`, `src/sparse_lu.hpp` | `NativeSparseLU` and `factorize_typed`: validation, equilibration, ordering calls, engine selection, the SPD attempt, the left-looking CSC LU, the row-hybrid LU, `solve_inplace`, CSC export, `stats`. Also `FactorExecutor`, the persistent `std::thread` team (per-call worker limit, `failed()` polling). |
| `src/sparse_ldlt.hpp` | Supernodal LDL^T kernels: the exact fundamental-node test, the supernodal symbolic phase (node tails, contributor lists, block offsets, work estimates), subtree scheduling (phase A subtrees and the CSC array allocation on single workers, phase B top nodes with the team), the per-node contribution fan-in (fixed 16-row strips; in-place `gemm_sub`/`gemm_sub_lower`/`gemm_sub_rows` parts, gather/scatter for noncontiguous columns), per-node panel factorization, and CSC assembly (row indices and values). |
| `src/sparse_lu_panels.hpp` | `DensePanel`: deferred dense-row updates and the tiled `L_panel * U_panel` flush of the row-hybrid LU. |
| `src/dense_kernels.hpp` | Dense tile kernel interface (`dense_kernels::Kernels`: `gemm_sub`, `gemm_sub_rows`, `gemm_sub_lower`, `axpy_sub`) and selection API. Declarations only: it is included by wider-ISA translation units. |
| `src/dense_kernels.cpp` | Scalar and SSE2 kernels, CPUID/XCR0 detection, `SPSOLVE_FORCE_KERNEL` and runtime selection behind an atomic pointer. |
| `src/dense_kernels_avx2.cpp` | `avx2-fma` kernel (4x8 microkernel; FMA only in the LDL^T `sum_then_subtract` order). The only file compiled with `/arch:AVX2` / `-mavx2 -mfma`; see its header comment for the hygiene rules. Built when `SP_ENABLE_AVX2` (default ON) and the target is x86-64. |
| `src/spsolve_triangular*.hpp` | CSR triangular solve for C- and Fortran-ordered right-hand sides. |
| `third_party/suitesparse_ordering/` | Vendored SuiteSparse AMD 3.3.4 and COLAMD 3.3.5 (BSD-3) plus a minimal config shim. Do not edit; upgrade wholesale and update `THIRD_PARTY_NOTICES.md`. |
| `tests/` | `unittest` suites. `tests/__init__.py` is empty; tests import the installed package. |
| `benchmarks/factorization.py` | Reproducible benchmark and promotion gate against SciPy SuperLU. |
| `docs/` | Design, algorithms, API, benchmark notes. Update them in the same commit as a behavior change. |
| `build/` | Gitignored scratch for local CMake trees and benchmark JSON/log output (`build/review/` by convention). |

## Build, test, benchmark

```bash
python -m pip install . -v                       # scikit-build-core + CMake + C++20 compiler
python -m unittest discover -v -s tests -t .     # must pass before any commit
python benchmarks/factorization.py --quick       # smoke run, also used by CI
python benchmarks/factorization.py --threads 8 --repeats 5 --extended --dense-lu --json build/review/run.json
python benchmarks/factorization.py --threads 8 --repeats 5 --gate   # promotion gate, exits nonzero on failure
```

- Tests and benchmarks import `spsolve` from site-packages, so rebuild with
  `pip install .` after every C++ change. A stale install silently tests old
  code.
- CMake options: `SP_USE_OPENMP` (default ON, only the triangular kernels and
  multi-RHS solves use OpenMP), `SP_ARCH_NATIVE` (default OFF; wheels target
  baseline x64, so never make correctness depend on it), `SP_ENABLE_AVX2`
  (default ON; builds the runtime-dispatched AVX2+FMA dense kernel on x86-64
  targets). Pass options through pip as `-Ccmake.define.SP_ARCH_NATIVE=ON`.
  `SPSOLVE_FORCE_KERNEL=sse2` (or `scalar`, `avx2`) runs the tests or
  benchmarks on another dense kernel without rebuilding.
- On Windows the build uses MSVC; a manual `cmake` run needs `vcvars64.bat`.
- Benchmarks pin BLAS libraries to one thread before importing NumPy so the
  SuperLU comparison is factorization versus factorization. Run them on an
  otherwise idle machine; medians of batched samples are still noisy at a few
  percent, so re-run before calling a change a regression.
- Small allocation-heavy cases depend on what ran earlier in the process (the
  row-permuted band is about 10% faster after `sparse-dense-border`); compare
  runs with the same case order.
- Keep a JSON of the state before your change and compare `native_1_seconds`,
  `native_n_seconds`, `numeric_seconds`, `dense_rows`, `lu_*`, `parallel_*`,
  `spd_*`, `residual`, and `stable_permutations` per case afterwards. The
  gate criteria are in `docs/benchmarking.md`.

## Invariants (do not break; tests pin them)

- **Thread-count determinism.** The same input and options give bitwise
  identical `perm_r`, `perm_c`, scales, `L` and `U` at every thread count.
  Tiling, segmentation, path selection and remainder handling may depend on
  sizes and positions only, never on the thread count or on which worker ran.
  Every output entry accumulates its terms in one fixed order. Checked by
  `assert_same_factors` in `tests/test_blocked_lu.py` and
  `tests/test_supernodal_ldlt.py` and by exact comparisons in
  `tests/test_sparse_lu.py`. The one exception is the supernodal LDL^T
  engine: its fan-in partition (whole-node parts on one thread vs 16-row
  strips on the worker team, through `worth_parallel`) and its subtree
  partition (which nodes a phase A worker factors serially, through the
  bound `W / (2 * threads)` and the forest rule; which worker takes which
  subtree is decided at run time) depend on the thread count. Its factors
  stay invariant only because a dense-kernel element's result is independent
  of tile shape and call path (`src/dense_kernels.hpp`,
  `test_kernels_agree_across_threads_and_tile_paths`) and because a node's
  block depends only on its input entries and its contributors' finished
  blocks (its descendants, finished before it in either phase), applied in
  the fixed order of the symbolic contributor lists with all scratch state
  per worker, so every kernel must keep that contract (for example, no
  shape-dependent K-blocking).
- **Pivoting semantics.** Threshold partial pivoting with `diag_pivot_thresh`
  and the preferred row `perm_c[k]`; ties break by current row position.
  Exact singularity raises an error containing "singular" (Python maps it to
  `numpy.linalg.LinAlgError`). Any non-finite intermediate raises a
  `RuntimeError` containing "non-finite" before factors are published, unless
  an exactly singular column is reached before a hoisted (deferred) check
  sees it, in which case "singular" is reported first; no factors are
  published either way. Hoisting a per-element finiteness check needs a
  written completeness argument in the code and in `docs/native-lu.md`.
- **SPD path.** Factors are published only after every pivot is positive and
  finite; otherwise fall back to pivoted LU with `failed_spd_seconds > 0`,
  `parallel_panels == 0`, `spd_panel_entries == 0`. Engine names and selection
  boundaries (`symmetric-ldlt-csc` vs `symmetric-supernodal-ldlt` at width 320,
  the `3072 * n` work guard, `left-looking-csc` vs `row-hybrid` limits) are
  pinned by tests; change them only together with the tests and docs.
  Phase A workers poll `FactorExecutor::failed()` between nodes and never
  call `parallel_for` (it is not reentrant). Subtree items allocate nothing
  (their workspaces are sized on the calling thread); the two CSC allocation
  items swallow an allocation failure, which assembly repeats on the calling
  thread. A failed attempt also reports `spd_subtrees == 0`.
- **Public contracts.** `Pr @ Dr @ A @ Dc @ Pc == L @ U`; lazy CSC `L`/`U`
  via `export_l_csr`/`export_u_csr`; `solve_inplace`; every existing `stats`
  key keeps its meaning (new keys may be added); the filled pattern including
  numerically cancelled entries stays stored.
- **No input mutation.** The native code only reads `const` pointers; tests
  assert the caller's arrays are unchanged.
- **Licensing.** BSD-3 only. Do not copy code from SuperLU, CHOLMOD, UMFPACK,
  MUMPS, PaStiX, Eigen's sparse modules, or MKL, and do not add a BLAS
  dependency. Kernels are written in-house; ideas from papers are fine and
  belong in `docs/algorithms.md` with a citation.

## Conventions

- C++20, `index_t = std::int64_t`, exceptions for errors (workers capture and
  rethrow through `FactorExecutor`), no global mutable state except the
  executor's own and the dense-kernel selection. Baseline translation units
  use SSE2 intrinsics only, guarded by the existing `__SSE2__ || _M_X64` test
  with a scalar fallback; dense tile arithmetic goes through the
  `dense_kernels::Kernels` snapshot a factorization takes once. Wider ISAs
  live only in separately compiled translation units (today
  `src/dense_kernels_avx2.cpp`) reached through the `dense_kernels` table
  after the CPUID/XCR0 check, so baseline wheels keep running. Such a unit
  includes only `<immintrin.h>`, typedef-only standard headers and the
  declarations-only `dense_kernels.hpp` (no STL inline code, which could be
  COMDAT-folded into baseline callers), keeps helpers in an anonymous
  namespace, exports only its kernel table (`constinit`) and entry points,
  and is never built with LTO (`/GL`, `-flto`) or `-mno-vzeroupper`. A kernel
  order may fuse multiply-adds only if no other code path applies the same
  terms with separate roundings (see "Dense kernels" in `docs/native-lu.md`).
- `.clang-format` has `DisableFormat: true`: match the surrounding style by
  hand (4-space indent, `static_cast`, `[[nodiscard]]`, braces on the same
  line).
- Python: NumPy/SciPy only, `from __future__ import annotations`, type hints,
  warnings for format conversions (`SparseEfficiencyWarning`).
- Every behavior or threshold change updates `docs/native-lu.md` (and
  `docs/benchmarking.md` if a benchmark number or criterion moves). Timing
  fields in `stats` document exactly what they include.
- Commit messages: `feat:` / `perf:` / `fix:` / `bench:` / `docs:` prefix, an
  imperative subject under 72 characters, and a wrapped body that states what
  changed and the measured effect with numbers (see `git log`). One logical
  change per commit. Never push unless asked.

## How to make a change safely

1. Measure first. Reproduce the case with `benchmarks/factorization.py` or a
   small script, read `stats` (engine, `numeric_seconds`, `parallel_steps`,
   `dense_rows`, `spd_*`) and locate the phase, before touching the code.
2. Keep the engine selection and the public contract unchanged unless the
   change is about them; most speedups live inside one engine or kernel.
3. Rebuild, run the whole test suite, then the benchmark with `--json`, and
   compare against your pre-change JSON for every case, not only the target.
4. Run a thread sweep (1, 2, 3, 4, 8) on the affected matrices and confirm
   bitwise identical factors, not just small residuals.
5. Update the docs in the same change, including the "Current limitations"
   section of `docs/native-lu.md` when a limitation appears or disappears.
6. Prefer an independent review of the diff (different person or agent) that
   re-runs tests and benchmarks before committing.

## Known gaps

The native backend stays opt-in (`backend="native"`) until the gate passes on
Windows and Linux release builds. As of `105fe2c` the `MMD_AT_PLUS_A` gate
(`--extended --dense-lu`) passes three consecutive Windows runs; the `COLAMD`
runs fail only on `permuted-band` (left-looking CSC), and no Linux gate run
exists. Current performance gaps and the planned
fixes are summarized at the end of `docs/native-lu.md` and in the status tables
of `docs/benchmarking.md`.
