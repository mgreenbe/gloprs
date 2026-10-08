#!/usr/bin/env python3
"""Differentially compare pinned GLOP and gloprs sparse LU traces."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def encode(
    matrix: list[list[float]],
    pivot_threshold: float = 0.01,
    zlatev_parameter: int = 3,
    singularity_threshold: float = 1e-15,
) -> str:
    entries = [
        (row, column, value)
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != 0.0
    ]
    lines = [f"{len(matrix)} {len(entries)}"]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    lines.append(
        f"{pivot_threshold:.17g} {zlatev_parameter} {singularity_threshold:.17g}"
    )
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> dict[str, list[str]]:
    completed = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    )
    result: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        result[fields[0]] = fields[1:]
    return result


def close(left: str, right: str) -> bool:
    a = float(left)
    b = float(right)
    # The two optimized compilers can differ by a few ulps in long triangular
    # solves even when pivot order and fill are identical.
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def compare(case: int, native: dict[str, list[str]], rust: dict[str, list[str]]) -> None:
    if native.keys() != rust.keys():
        raise AssertionError(f"case {case}: fields differ: {native.keys()} != {rust.keys()}")
    if "singular" in native:
        return
    for field in (
        "row_perm",
        "inverse_col_perm",
        "upper_entries",
        "entries",
        "sparse_right_positions",
        "sparse_left_positions",
        "stats_hex",
    ):
        if native[field] != rust[field]:
            raise AssertionError(f"case {case}: {field}: {native[field]} != {rust[field]}")
    for field in (
        "determinant",
        "deterministic_time",
        "right",
        "left",
        "sparse_right",
        "sparse_left",
    ):
        if len(native[field]) != len(rust[field]) or not all(
            close(a, b) for a, b in zip(native[field], rust[field], strict=True)
        ):
            raise AssertionError(f"case {case}: {field}: {native[field]} != {rust[field]}")


def generated_matrix(
    generator: random.Random, n: int, density: float = 0.18
) -> list[list[float]]:
    matrix = [[0.0] * n for _ in range(n)]
    for row in range(n):
        absolute_sum = 0.0
        for column in range(n):
            if row != column and generator.random() < density:
                value = generator.uniform(-2.0, 2.0)
                matrix[row][column] = value
                absolute_sum += abs(value)
        matrix[row][row] = absolute_sum + generator.uniform(0.25, 2.0)
    # Column shuffling prevents the diagonal from predetermining pivot order.
    order = list(range(n))
    generator.shuffle(order)
    return [[row[column] for column in order] for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0x5EED5EED)
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=24)
    parser.add_argument("--density", type=float, default=0.18)
    parser.add_argument("--pivot-threshold", type=float, default=0.01)
    parser.add_argument("--zlatev-parameter", type=int, default=3)
    parser.add_argument("--singularity-threshold", type=float, default=1e-15)
    parser.add_argument(
        "--native", type=Path, default=ROOT / "target/native/lu_reference_adapter"
    )
    parser.add_argument(
        "--rust", type=Path, default=ROOT / "target/debug/examples/lu_trace"
    )
    args = parser.parse_args()

    generator = random.Random(args.seed)
    for case in range(args.cases):
        n = generator.randint(args.min_n, args.max_n)
        data = encode(
            generated_matrix(generator, n, args.density),
            args.pivot_threshold,
            args.zlatev_parameter,
            args.singularity_threshold,
        )
        try:
            compare(case, run(args.native, data), run(args.rust, data))
        except AssertionError:
            print(data, end="")
            raise
    print(f"{args.cases} LU traces agree")


if __name__ == "__main__":
    main()
