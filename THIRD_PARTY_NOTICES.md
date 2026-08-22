# Third-party notices

## SuiteSparse AMD and COLAMD

`third_party/suitesparse_ordering` contains the source subset used for sparse
column ordering:

- AMD 3.3.4 from SuiteSparse 7.12.3;
- COLAMD 3.3.5 from SuiteSparse 7.12.3;
- the required declarations from SuiteSparse_config 7.12.3 and a local minimal
  allocation shim.

AMD, COLAMD, and SuiteSparse_config are distributed under BSD-3-Clause
licenses. Their complete license texts are retained under
`third_party/suitesparse_ordering/LICENSE-*.txt` and are included in binary
wheels.

Only ordering routines are included. No SuiteSparse factorization package,
external BLAS implementation, shared library, or runtime package is bundled.

Upstream source: <https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/v7.12.3>
