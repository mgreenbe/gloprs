#!/usr/bin/env python3
"""Compare LpScalingHelper with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def number(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.17g}"


def generate(generator: random.Random, case: int) -> str:
    n = generator.randint(1, 60)
    positive = lambda: 10.0 ** generator.uniform(-12.0, 12.0)
    row_factors = [positive() for _ in range(n)]
    col_factors = [positive() for _ in range(n)]
    objective = [generator.choice([0.0, generator.uniform(-100.0, 100.0)]) for _ in range(n)]
    if case % 10 == 0:
        objective = [0.0] * n
    bound_pool = [0.0, -float("inf"), float("inf")]
    lower = [generator.choice(bound_pool + [generator.uniform(-100.0, 100.0)]) for _ in range(n)]
    upper = [generator.choice(bound_pool + [generator.uniform(-100.0, 100.0)]) for _ in range(n)]
    if case % 10 == 1:
        lower = [-generator.uniform(2.0, 100.0) for _ in range(n)]
        upper = [generator.uniform(2.0, 100.0) for _ in range(n)]
    elif case % 10 == 2:
        lower = [-generator.uniform(1e-6, 0.5) for _ in range(n)]
        upper = [generator.uniform(1e-6, 0.5) for _ in range(n)]
    values = [generator.uniform(-100.0, 100.0) for _ in range(n)]
    if generator.random() < 0.7:
        pattern = generator.sample(range(n), generator.randint(1, n))
    else:
        pattern = []
    basis = list(range(n))
    generator.shuffle(basis)
    selected = generator.randrange(n)
    lines = [str(n)]
    for vector in (row_factors, col_factors, objective, lower, upper, values):
        lines.append(" ".join(number(value) for value in vector))
    lines.append(f"{len(pattern)} " + " ".join(map(str, pattern)))
    lines.append(" ".join(map(str, basis)))
    lines.append(str(selected))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    return [
        line.split()
        for line in subprocess.run(
            [executable], input=data, text=True, capture_output=True, check=True
        ).stdout.splitlines()
    ]


def close(left: str, right: str) -> bool:
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return left == right
    return a == b or math.isclose(a, b, rel_tol=3e-15, abs_tol=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x5CA11A6)
    native = ROOT / "target/native/lp_data_utils_scaling_reference_adapter"
    rust = ROOT / "target/debug/examples/lp_data_utils_scaling_trace"
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
    print(f"{args.cases} LP-scaling-helper traces agree")


if __name__ == "__main__":
    main()
