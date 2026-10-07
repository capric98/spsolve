"""Reproducible sparse-LU benchmark and native-backend promotion gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

# Keep optional BLAS threading out of the SuperLU comparison. These variables
# must be set before importing NumPy/SciPy.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy
from scipy.sparse import csc_array, diags, eye, kron, random_array
from scipy.sparse.linalg import splu as scipy_splu

from spsolve import SparseLU, splu


@dataclass
class Result:
    name: str
    n: int
    input_nnz: int
    factor_nnz: int
    scipy_factor_nnz: int
    fill_ratio: float
    fill_over_scipy: float
    native_1_seconds: float
    native_n_seconds: float
    scipy_seconds: float
    speedup: float
    native_to_scipy: float
    parallel_steps: int
    stable_permutations: bool
    ordering_backend: str
    ordering_seconds: float
    symbolic_seconds: float
    numeric_seconds: float
    assembly_seconds: float
    factor_seconds: float
    numeric_engine: str
    largest_front: int
    supernodes: int
    max_supernode_width: int
    peak_workspace_bytes: int
    numeric_probe_columns: int
    numeric_probe_max_front: int
    numeric_probe_updates: int
    numeric_probe_seconds: float
    numeric_probe_fallback: bool
    residual: float
    eligible: bool
    passed: bool


def poisson_2d(side: int) -> csc_array:
    one_d = diags(
        [-np.ones(side - 1), 4.0 * np.ones(side), -np.ones(side - 1)],
        [-1, 0, 1],
        format="csc",
    )
    identity = eye(side, format="csc")
    coupling = diags(
        [-np.ones(side - 1), -np.ones(side - 1)], [-1, 1], format="csc"
    )
    return csc_array(kron(identity, one_d) + kron(coupling, identity))


def poisson_3d(side: int) -> csc_array:
    one_d = diags(
        [-np.ones(side - 1), 2.0 * np.ones(side), -np.ones(side - 1)],
        [-1, 0, 1],
        format="csc",
    )
    identity = eye(side, format="csc")
    return csc_array(
        kron(kron(identity, identity), one_d)
        + kron(kron(identity, one_d), identity)
        + kron(kron(one_d, identity), identity)
    )


def banded_nonsymmetric(n: int) -> csc_array:
    return csc_array(
        diags(
            [
                -0.2 * np.ones(n - 7),
                -np.ones(n - 1),
                4.0 * np.ones(n),
                0.5 * np.ones(n - 2),
                0.1 * np.ones(n - 9),
            ],
            [-7, -1, 0, 2, 9],
            shape=(n, n),
            format="csc",
        )
    )


def irregular(n: int, density: float, seed: int) -> csc_array:
    rng = np.random.default_rng(seed)
    matrix = random_array(
        (n, n),
        density=density,
        format="csc",
        rng=rng,
        data_sampler=lambda size: rng.uniform(-1.0, 1.0, size),
    )
    row_norm = np.asarray(abs(matrix).sum(axis=1)).ravel()
    return csc_array(matrix + diags(row_norm + 1.0, format="csc"))


def row_permuted_banded(n: int, seed: int = 321) -> csc_array:
    matrix = banded_nonsymmetric(n)
    return csc_array(matrix[np.random.default_rng(seed).permutation(n), :])


def pivot_heavy(n: int, density: float, seed: int) -> csc_array:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    dominant = csc_array(
        (5.0 * np.ones(n), (np.arange(n), permutation)), shape=(n, n)
    )
    noise = random_array(
        (n, n),
        density=density,
        format="csc",
        rng=rng,
        data_sampler=lambda size: rng.uniform(-0.2, 0.2, size),
    )
    return csc_array(dominant + noise + diags(0.05 * np.ones(n)))


def median_runtime(
    factory: Callable[[], object],
    repeats: int,
    warmups: int,
    minimum_sample_seconds: float = 0.0,
) -> tuple[float, object]:
    result: object | None = None
    for _ in range(warmups):
        result = factory()
    batch_size = 1
    if minimum_sample_seconds > 0.0:
        gc.collect()
        start = time.perf_counter()
        result = factory()
        probe_seconds = time.perf_counter() - start
        if probe_seconds > 0.0:
            batch_size = min(
                128,
                max(1, math.ceil(minimum_sample_seconds / probe_seconds)),
            )

    timings = []
    for _ in range(repeats):
        # Destroy each previous factor before starting the next timer so
        # allocator-specific destruction is not charged to factorization and
        # batches do not retain many large factors simultaneously.
        result = None
        gc.collect()
        elapsed = 0.0
        for _ in range(batch_size):
            result = None
            start = time.perf_counter()
            result = factory()
            elapsed += time.perf_counter() - start
        timings.append(elapsed / batch_size)
    assert result is not None
    return statistics.median(timings), result


def relative_residual(matrix: csc_array, factor: object) -> float:
    rng = np.random.default_rng(20260821)
    expected = rng.normal(size=matrix.shape[0])
    rhs = matrix @ expected
    solution = factor.solve(rhs)
    numerator = np.linalg.norm(matrix @ solution - rhs, ord=np.inf)
    denominator = (
        np.linalg.norm(abs(matrix) @ np.abs(solution), ord=np.inf)
        + np.linalg.norm(rhs, ord=np.inf)
    )
    return float(numerator / denominator) if denominator else 0.0


def benchmark_case(
    name: str,
    matrix: csc_array,
    *,
    threads: int,
    repeats: int,
    warmups: int,
    minimum_serial_seconds: float,
    match_tolerance: float,
    minimum_sample_seconds: float = 0.0,
    permc_spec: str = "MMD_AT_PLUS_A",
) -> Result:
    native_options = {"backend": "native", "permc_spec": permc_spec}
    serial_time, serial = median_runtime(
        lambda: splu(matrix, num_threads=1, **native_options),
        repeats,
        warmups,
        minimum_sample_seconds,
    )
    parallel_time, parallel = median_runtime(
        lambda: splu(matrix, num_threads=threads, **native_options),
        repeats,
        warmups,
        minimum_sample_seconds,
    )
    scipy_time, scipy_factor = median_runtime(
        lambda: scipy_splu(matrix, permc_spec=permc_spec),
        repeats,
        warmups,
        minimum_sample_seconds,
    )
    assert isinstance(serial, SparseLU) and isinstance(parallel, SparseLU)

    speedup = serial_time / parallel_time
    native_to_scipy = parallel_time / scipy_time
    residual = relative_residual(matrix, parallel)
    scipy_factor_nnz = scipy_factor.L.nnz + scipy_factor.U.nnz
    stable_permutations = bool(
        np.array_equal(serial.perm_r, parallel.perm_r)
        and np.array_equal(serial.perm_c, parallel.perm_c)
    )
    eligible = serial_time >= minimum_serial_seconds and parallel.stats["parallel_steps"] > 0
    passed = (
        (not eligible or speedup >= 2.0)
        and native_to_scipy <= match_tolerance
        and residual <= 1e-12
        and stable_permutations
    )
    return Result(
        name=name,
        n=matrix.shape[0],
        input_nnz=matrix.nnz,
        factor_nnz=parallel.nnz,
        scipy_factor_nnz=scipy_factor_nnz,
        fill_ratio=parallel.stats["fill_ratio"],
        fill_over_scipy=parallel.nnz / scipy_factor_nnz,
        native_1_seconds=serial_time,
        native_n_seconds=parallel_time,
        scipy_seconds=scipy_time,
        speedup=speedup,
        native_to_scipy=native_to_scipy,
        parallel_steps=parallel.stats["parallel_steps"],
        stable_permutations=stable_permutations,
        ordering_backend=parallel.stats["ordering_backend"],
        ordering_seconds=parallel.stats["ordering_seconds"],
        symbolic_seconds=parallel.stats["symbolic_seconds"],
        numeric_seconds=parallel.stats["numeric_seconds"],
        assembly_seconds=parallel.stats["assembly_seconds"],
        factor_seconds=parallel.stats["factor_seconds"],
        numeric_engine=parallel.stats["numeric_engine"],
        largest_front=parallel.stats["largest_front"],
        supernodes=parallel.stats["supernodes"],
        max_supernode_width=parallel.stats["max_supernode_width"],
        peak_workspace_bytes=parallel.stats["peak_workspace_bytes"],
        numeric_probe_columns=parallel.stats.get("numeric_probe_columns", 0),
        numeric_probe_max_front=parallel.stats.get("numeric_probe_max_front", 0),
        numeric_probe_updates=parallel.stats.get("numeric_probe_updates", 0),
        numeric_probe_seconds=parallel.stats.get("numeric_probe_seconds", 0.0),
        numeric_probe_fallback=parallel.stats.get("numeric_probe_fallback", False),
        residual=residual,
        eligible=eligible,
        passed=passed,
    )


def print_results(results: list[Result], threads: int) -> None:
    print(
        f"{'case':<16} {'n':>7} {'nnz(A)':>10} {'fill':>7} "
        f"{'native-1':>10} {f'native-{threads}':>10} {'SciPy':>10} "
        f"{'speedup':>8} {'N/S':>7} {'fill/S':>7} {'engine':>20} {'gate':>7}"
    )
    for result in results:
        gate = "PASS" if result.passed else "FAIL"
        print(
            f"{result.name:<16} {result.n:7d} {result.input_nnz:10d} "
            f"{result.fill_ratio:7.1f} {result.native_1_seconds:10.4f} "
            f"{result.native_n_seconds:10.4f} {result.scipy_seconds:10.4f} "
            f"{result.speedup:8.2f} {result.native_to_scipy:7.2f} "
            f"{result.fill_over_scipy:7.2f} {result.numeric_engine:>20} {gate:>7}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--minimum-serial-seconds", type=float, default=0.20)
    parser.add_argument("--match-tolerance", type=float, default=1.05)
    parser.add_argument("--min-sample-seconds", type=float, default=0.10)
    parser.add_argument(
        "--permc-spec",
        choices=("NATURAL", "COLAMD", "MMD_AT_PLUS_A"),
        default="MMD_AT_PLUS_A",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="include 3-D Poisson and banded cases; COLAMD also tests row-permuted bands",
    )
    parser.add_argument("--gate", action="store_true", help="exit nonzero when the gate fails")
    parser.add_argument("--json", type=Path, help="write machine-readable results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.threads < 2
        or args.repeats < 1
        or args.warmups < 0
        or args.min_sample_seconds < 0.0
    ):
        raise SystemExit(
            "threads >= 2, repeats >= 1, warmups >= 0, and "
            "min-sample-seconds >= 0 are required"
        )

    if args.quick:
        matrices = [
            ("poisson", poisson_2d(45)),
            ("irregular", irregular(900, 0.006, 7)),
            ("pivot-heavy", pivot_heavy(650, 0.008, 23)),
        ]
    else:
        matrices = [
            ("poisson", poisson_2d(90)),
            ("irregular", irregular(2500, 0.002, 7)),
            ("pivot-heavy", pivot_heavy(2000, 0.0025, 23)),
        ]
    if args.extended:
        matrices.extend(
            [
                ("poisson-3d", poisson_3d(10 if args.quick else 20)),
                ("banded-nonsym", banded_nonsymmetric(2500 if args.quick else 8000)),
            ]
        )
        if args.permc_spec == "COLAMD":
            matrices.append(
                ("permuted-band", row_permuted_banded(2500 if args.quick else 20000))
            )

    results = [
        benchmark_case(
            name,
            matrix,
            threads=args.threads,
            repeats=args.repeats,
            warmups=args.warmups,
            minimum_serial_seconds=args.minimum_serial_seconds,
            match_tolerance=args.match_tolerance,
            minimum_sample_seconds=args.min_sample_seconds,
            permc_spec=args.permc_spec,
        )
        for name, matrix in matrices
    ]
    print_results(results, args.threads)

    eligible = [result for result in results if result.eligible]
    gate_passed = bool(eligible) and all(result.passed for result in results)
    print(f"promotion gate: {'PASS' if gate_passed else 'FAIL'}")

    if args.json:
        payload = {
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "threads": args.threads,
            },
            "criteria": {
                "minimum_serial_seconds": args.minimum_serial_seconds,
                "minimum_speedup": 2.0,
                "maximum_native_to_scipy": args.match_tolerance,
                "maximum_residual": 1e-12,
                "stable_permutations": True,
                "minimum_sample_seconds": args.min_sample_seconds,
                "permc_spec": args.permc_spec,
            },
            "passed": gate_passed,
            "results": [asdict(result) for result in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return int(args.gate and not gate_passed)


if __name__ == "__main__":
    raise SystemExit(main())
