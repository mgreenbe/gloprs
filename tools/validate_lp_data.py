#!/usr/bin/env python3
"""Compare core LinearProgram behavior with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    rows = generator.randint(1, 20)
    columns = generator.randint(1, 25)
    entries: list[tuple[int, int, float]] = []
    variable_data = []
    for column in range(columns):
        variable_type = generator.randrange(3)
        if variable_type and generator.random() < 0.5:
            lower, upper = 0.0, 1.0
        else:
            lower = generator.randint(-4, 1)
            upper = generator.randint(max(int(lower), 1), 7)
        variable_data.append((variable_type, lower, upper, generator.uniform(-3.0, 3.0)))
    constraint_data = []
    for row in range(rows):
        lower = generator.randint(-8, 0)
        upper = generator.randint(0, 8)
        constraint_data.append((lower, upper))
    for column in range(columns):
        integer = variable_data[column][0] != 0
        for row in range(rows):
            if generator.random() < 0.18:
                value = generator.randint(-3, 3) if integer else generator.uniform(-3.0, 3.0)
                if value != 0:
                    entries.append((row, column, value))
    solution = [generator.uniform(data[1] - 1.0, data[2] + 1.0) for data in variable_data]
    lines = [
        f"{rows} {columns} {len(entries)} {generator.randrange(2)} "
        f"{generator.uniform(-2, 2):.17g} {generator.uniform(.2, 3):.17g} 1e-8"
    ]
    lines.extend(f"{kind} {lower:.17g} {upper:.17g} {cost:.17g}" for kind, lower, upper, cost in variable_data)
    lines.extend(f"{lower:.17g} {upper:.17g}" for lower, upper in constraint_data)
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    lines.append(" ".join(f"{value:.17g}" for value in solution))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    output = subprocess.run([executable], input=data, text=True, capture_output=True, check=True).stdout
    return [line.split() for line in output.splitlines()]


def equal(left: list[str], right: list[str]) -> bool:
    if left[0] != right[0] or len(left) != len(right):
        return False
    for a, b in zip(left[1:], right[1:], strict=True):
        if ":" in a or ":" in b:
            a_parts = a.split(":")
            b_parts = b.split(":")
            if a_parts[:2] != b_parts[:2] or not math.isclose(
                float(a_parts[2]), float(b_parts[2]), rel_tol=2e-14, abs_tol=2e-13
            ):
                return False
        else:
            try:
                numbers_equal = math.isclose(
                    float(a), float(b), rel_tol=2e-14, abs_tol=2e-13
                )
            except ValueError:
                numbers_equal = a == b
            if not numbers_equal:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    args = parser.parse_args()
    generator = random.Random(0x1DADA)
    native = ROOT / "target/native/lp_data_reference_adapter"
    rust = ROOT / "target/debug/examples/lp_data_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            not equal(left, right) for left, right in zip(expected, actual, strict=True)
        ):
            raise AssertionError(f"case {case}: {expected} != {actual}\n{data}")
    print(f"{args.cases} lp_data traces agree")


if __name__ == "__main__":
    main()
