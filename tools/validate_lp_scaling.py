#!/usr/bin/env python3
"""Compare whole-model default scaling with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random, case: int) -> str:
    rows = generator.randint(0, 25)
    columns = generator.randint(0, 25)
    possible = [(r, c) for c in range(columns) for r in range(rows)]
    count = generator.randint(0, min(len(possible), 3 * (rows + columns)))
    entries = []
    for row, col in generator.sample(possible, count):
        entries.append((row, col, math.copysign(10.0 ** generator.uniform(-25, 25), generator.choice([-1, 1]))))
    lines = [f"{rows} {columns} {len(entries)}"]
    lines.extend(f"{row} {col} {value:.17g}" for row, col, value in entries)
    for _ in range(columns):
        objective = generator.choice([0.0, generator.uniform(-100, 100)])
        lower = generator.uniform(-100, 0)
        upper = generator.uniform(0, 100)
        lines.append(f"{objective:.17g} {lower:.17g} {upper:.17g}")
    for _ in range(rows):
        lower = generator.uniform(-100, 0)
        upper = generator.uniform(0, 100)
        lines.append(f"{lower:.17g} {upper:.17g}")
    lines.append(str(case % 4))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    return [line.split() for line in subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout.splitlines()]


def close(left: str, right: str) -> bool:
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return left == right
    if "e" not in left.lower() and "e" not in right.lower():
        return left == right
    return a == b or math.isclose(a, b, rel_tol=4e-15, abs_tol=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    args = parser.parse_args()
    generator = random.Random(0x1A5CA1E)
    native = ROOT / "target/native/lp_scaling_reference_adapter"
    rust = ROOT / "target/debug/examples/lp_scaling_trace"
    for case in range(args.cases):
        data = generate(generator, case)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            len(left) != len(right) or any(not close(a, b) for a, b in zip(left, right))
            for left, right in zip(expected, actual)
        ):
            raise AssertionError(f"case {case}:\n{expected}\n!=\n{actual}\n{data}")
    print(f"{args.cases} whole-LP scaling traces agree")


if __name__ == "__main__":
    main()
