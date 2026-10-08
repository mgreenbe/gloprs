#!/usr/bin/env python3
"""Compare general sparse-matrix operations with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries(generator: random.Random, rows: int, columns: int) -> list[tuple[int, int, float]]:
    result = []
    for column in range(columns):
        for row in range(rows):
            if generator.random() < 0.18:
                result.append((row, column, generator.uniform(-5.0, 5.0)))
    return result


def generate(generator: random.Random) -> str:
    m = generator.randint(1, 25)
    k = generator.randint(1, 25)
    n = generator.randint(0, 25)
    a = entries(generator, m, k)
    b = entries(generator, k, n)
    alpha = generator.uniform(-3.0, 3.0)
    beta = generator.uniform(-3.0, 3.0)
    fields = [f"{m} {k} {n} {len(a)} {len(b)} {alpha:.17g} {beta:.17g}"]
    fields.extend(f"{row} {column} {value:.17g}" for row, column, value in a + b)
    return "\n".join(fields) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    return [
        line.split()
        for line in subprocess.run(
            [executable], input=data, text=True, capture_output=True, check=True
        ).stdout.splitlines()
    ]


def equal(left: list[str], right: list[str]) -> bool:
    if left[0] != right[0] or len(left) != len(right):
        return False
    if left[0] == "magnitudes":
        start, stride = 1, 1
    else:
        if left[1:4] != right[1:4]:
            return False
        start, stride = 4, 3
    for position in range(start, len(left), stride):
        if stride == 3 and left[position : position + 2] != right[position : position + 2]:
            return False
        value_position = position if stride == 1 else position + 2
        if not math.isclose(
            float(left[value_position]),
            float(right[value_position]),
            rel_tol=2e-15,
            abs_tol=0.0,
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x5A45E)
    native = ROOT / "target/native/sparse_matrix_reference_adapter"
    rust = ROOT / "target/debug/examples/sparse_matrix_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            not equal(left, right) for left, right in zip(expected, actual, strict=True)
        ):
            raise AssertionError(f"case {case}: {expected} != {actual}")
    print(f"{args.cases} sparse-matrix traces agree")


if __name__ == "__main__":
    main()
