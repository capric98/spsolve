# Factorization benchmark and promotion gate

Run the quick smoke benchmark with:

```bash
python benchmarks/factorization.py --quick --repeats 3
```

Run the default eight-thread promotion gate with:

```bash
python benchmarks/factorization.py --threads 8 --repeats 5 --gate
```

Add `--json build/factorization.json` to retain machine-readable environment,
criteria, timings, fill, residual, and pass/fail data.

## Criteria

A case is eligible for the scaling criterion when native one-thread
factorization takes at least 0.20 seconds and the factor reports parallel pivot
steps. Medians must satisfy all of the following:

- every scaling-eligible case reaches at least 2.0× over native one-thread;
- every representative case, including cases too small for the scaling test,
  takes no more than 1.05× SciPy SuperLU time using the same
  `MMD_AT_PLUS_A` ordering request;
- every case has relative backward residual at most `1e-12`;
- at least one case is large enough for the scaling criterion.

The script constrains optional BLAS implementations to one thread before
loading NumPy/SciPy. This keeps the comparison focused on factorization rather
than hidden nested BLAS teams.

## Current status

The backend remains opt-in. After adding the hybrid dense-front path, a
development run on Windows x64 with Python 3.14 showed the 2,500-row irregular
case improving from approximately 0.41 seconds at one thread to 0.14 seconds at
eight threads (about 3.0×), ahead of SciPy SuperLU at approximately 0.19
seconds. A pivot-heavy 2,000-row case also reached more than 3× scaling and ran
ahead of SuperLU. The 8,100-row structured Poisson case took approximately 0.11
seconds natively versus 0.008 seconds in SuperLU, so the
representative-family gate still fails. Timings are illustrative only; use the
included script on each release platform for a promotion decision.

Do not switch `NATIVE_DEFAULT` in `src/spsolve/sparse_lu.py` until the full gate
passes consistently on Windows and Linux x64, correctness tests pass, and no
matrix family shows a material regression.
