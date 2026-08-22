# Vendored sparse ordering subset

This directory contains the minimum source subset needed for the 64-bit AMD
3.3.4 and COLAMD 3.3.5 entry points from SuiteSparse 7.12.3. The upstream
sources were taken from the `v7.12.3` release without algorithm changes.

Only ordering code is compiled into the extension. No SuiteSparse solver,
BLAS, shared library, runtime package, or upstream build system is included.
`suitesparse_config_shim.c` supplies the two allocation functions used by AMD.

Upstream: <https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/v7.12.3>

The AMD, COLAMD, and SuiteSparse_config BSD-3-Clause license texts are retained
alongside the sources as `LICENSE-*.txt`.
