#!/usr/bin/env python3
"""Compare whole-model scaling, including parameterized methods, with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random, case: int, parameterized: bool) -> str:
    rows = generator.randint(0, 25)
    columns = generator.randint(0, 25)
    possible = [(r, c) for c in range(columns) for r in range(rows)]
    count = generator.randint(0, min(len(possible), 3 * (rows + columns)))
    entries = []
    for row, col in generator.sample(possible, count):
        entries.append((row, col, math.copysign(10.0 ** generator.uniform(-25, 25), generator.choice([-1, 1]))))
    if parameterized and case % 3 == 2 and columns:
        occupied_rows = {row for row, _, _ in entries}
        for row in range(rows):
            if row not in occupied_rows:
                col = generator.randrange(columns)
                value = math.copysign(10.0 ** generator.uniform(-25, 25), generator.choice([-1, 1]))
                entries.append((row, col, value))
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
    if parameterized:
        lines.append(str(case % 3))
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
    parser.add_argument("--parameterized", action="store_true")
    args = parser.parse_args()
    generator = random.Random(0x1A5CA1E)
    native = ROOT / "target/native/lp_scaling_reference_adapter"
    rust = ROOT / "target/debug/examples" / (
        "lp_scaling_parameters_trace" if args.parameterized else "lp_scaling_trace"
    )
    for case in range(args.cases):
        data = generate(generator, case, args.parameterized)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual):
            raise AssertionError(f"case {case}: output line count differs\n{data}")
        for left, right in zip(expected, actual):
            if len(left) != len(right):
                raise AssertionError(f"case {case}, {left[0]}: length differs\n{data}")
            for position, (a, b) in enumerate(zip(left, right)):
                if not close(a, b):
                    raise AssertionError(
                        f"case {case}, {left[0]}[{position}]: {a} != {b}\n{data}"
                    )
    print(f"{args.cases} whole-LP scaling traces agree")


if __name__ == "__main__":
    main()
