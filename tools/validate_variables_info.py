#!/usr/bin/env python3
"""Differentially compare pinned GLOP and gloprs VariablesInfo traces."""

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


def generate(generator: random.Random) -> str:
    rows = generator.randint(1, 12)
    columns = generator.randint(rows, rows + 15)
    entries: list[tuple[int, int, float]] = []
    for column in range(columns):
        for row in range(rows):
            if generator.random() < 0.2:
                entries.append((row, column, generator.uniform(-3.0, 3.0)))
    lines = [f"{rows} {columns} {len(entries)}"]
    lines.extend(f"{r} {c} {number(v)}" for r, c, v in entries)
    for _ in range(columns):
        kind = generator.randrange(5)
        if kind == 0:
            lower, upper = -float("inf"), float("inf")
        elif kind == 1:
            lower, upper = generator.uniform(-5.0, 5.0), float("inf")
        elif kind == 2:
            lower, upper = -float("inf"), generator.uniform(-5.0, 5.0)
        elif kind == 3:
            lower = generator.uniform(-5.0, 5.0)
            upper = lower + generator.uniform(0.01, 5.0)
        else:
            lower = upper = generator.uniform(-5.0, 5.0)
        lines.append(f"{number(lower)} {number(upper)}")
    reduced = [generator.choice([0.0, 1e-8, -1e-8, 1e-3, -1e-3]) for _ in range(columns)]
    lines.append(" ".join(number(value) for value in reduced))
    basic = generator.sample(range(columns), generator.randint(0, min(rows, columns)))
    lines.append(" ".join([str(len(basic)), *(str(value) for value in basic)]))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def compare(native: str, rust: str) -> bool:
    native_lines = {line.split()[0]: line.split()[1:] for line in native.splitlines()}
    rust_lines = {line.split()[0]: line.split()[1:] for line in rust.splitlines()}
    if native_lines.keys() != rust_lines.keys():
        return False
    for field, native_values in native_lines.items():
        rust_values = rust_lines[field]
        if len(native_values) != len(rust_values):
            return False
        if field.endswith(("_lower", "_upper")):
            if not all(
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
                for left, right in zip(native_values, rust_values, strict=True)
            ):
                return False
        elif native_values != rust_values:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0xA11CE5)
    parser.add_argument(
        "--native",
        type=Path,
        default=ROOT / "target/native/variables_info_reference_adapter",
    )
    parser.add_argument(
        "--rust",
        type=Path,
        default=ROOT / "target/debug/examples/variables_info_trace",
    )
    args = parser.parse_args()
    generator = random.Random(args.seed)
    for case in range(args.cases):
        data = generate(generator)
        native = run(args.native, data)
        rust = run(args.rust, data)
        if not compare(native, rust):
            print(data, end="")
            raise AssertionError(f"case {case}:\nGLOP:\n{native}\ngloprs:\n{rust}")
    print(f"{args.cases} VariablesInfo traces agree")


if __name__ == "__main__":
    main()
