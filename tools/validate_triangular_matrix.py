#!/usr/bin/env python3
"""Compare triangular storage, builders, copies, and solves with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    n = generator.randint(40, 120)
    entries: list[tuple[int, int, float]] = []
    identity_prefix = generator.randrange(n + 1)
    for column in range(n):
        diagonal = (
            1.0
            if column < identity_prefix
            else generator.choice([-1.0, 1.0]) * generator.uniform(0.5, 2.0)
        )
        entries.append((column, column, diagonal))
        if column >= identity_prefix:
            for row in range(column + 1, n):
                if generator.random() < 0.02:
                    entries.append((row, column, generator.uniform(-0.2, 0.2)))
    root = min(identity_prefix, n - 1)
    lines = [f"{n} {len(entries)} {root}"]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return [line.split() for line in output.splitlines()]


def equal(left: list[str], right: list[str]) -> bool:
    if left[0] != right[0] or len(left) != len(right):
        return False
    for a, b in zip(left[1:], right[1:], strict=True):
        try:
            if not math.isclose(float(a), float(b), rel_tol=3e-14, abs_tol=5e-15):
                return False
        except ValueError:
            if a != b:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x7A1A6)
    native = ROOT / "target/native/triangular_matrix_reference_adapter"
    rust = ROOT / "target/debug/examples/triangular_matrix_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            not equal(left, right) for left, right in zip(expected, actual, strict=True)
        ):
            raise AssertionError(f"case {case}: {expected} != {actual}")
    print(f"{args.cases} triangular-matrix traces agree")


if __name__ == "__main__":
    main()
