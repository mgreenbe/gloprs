#!/usr/bin/env python3
"""Compare sparse equilibration with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random, case: int) -> str:
    rows = generator.randint(0, 40)
    columns = generator.randint(0, 40)
    possible = [(r, c) for c in range(columns) for r in range(rows)]
    count = generator.randint(0, min(len(possible), 4 * (rows + columns)))
    if case % 8 == 0:
        count = 0
    positions = generator.sample(possible, count)
    entries = []
    for row, column in positions:
        if case % 8 == 1:
            value = generator.choice([-1.0, 1.0])
        elif case % 8 in (2, 3):
            exponent = generator.uniform(-5.0, 5.0)
            value = math.copysign(10.0**exponent, generator.choice([-1.0, 1.0]))
        elif case % 8 == 4 and entries:
            exponent = generator.choice([-40.0, 40.0])
            value = math.copysign(10.0**exponent, generator.choice([-1.0, 1.0]))
        elif case % 8 == 5:
            value = 0.0
        else:
            exponent = generator.uniform(-40.0, 40.0)
            value = math.copysign(10.0**exponent, generator.choice([-1.0, 1.0]))
        entries.append((row, column, value))
    lines = [f"{rows} {columns} {len(entries)}"]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return [line.split() for line in output.splitlines()]


def close(left: str, right: str) -> bool:
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return left == right
    if "e" not in left.lower() and "e" not in right.lower():
        return left == right
    return a == b or math.isclose(a, b, rel_tol=3e-15, abs_tol=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x5CA1E)
    native = ROOT / "target/native/matrix_scaler_reference_adapter"
    rust = ROOT / "target/debug/examples/matrix_scaler_trace"
    for case in range(args.cases):
        data = generate(generator, case)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            len(left) != len(right)
            or any(not close(a, b) for a, b in zip(left, right))
            for left, right in zip(expected, actual)
        ):
            raise AssertionError(f"case {case}:\n{expected}\n!=\n{actual}\n{data}")
    print(f"{args.cases} matrix-scaler traces agree")


if __name__ == "__main__":
    main()
