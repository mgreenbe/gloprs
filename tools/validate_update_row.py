#!/usr/bin/env python3
"""Differentially compare pinned GLOP and gloprs update-row kernels."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    rows = generator.randint(1, 30)
    columns = generator.randint(1, 35)
    entries = []
    for column in range(columns):
        for row in range(rows):
            if generator.random() < 0.12:
                entries.append((row, column, generator.uniform(-3.0, 3.0)))
    lines = [f"{rows} {columns} {len(entries)}"]
    lines.extend(f"{r} {c} {v:.17g}" for r, c, v in entries)
    lhs = [0.0 if generator.random() < 0.7 else generator.uniform(-2.0, 2.0) for _ in range(rows)]
    if not any(lhs):
        lhs[generator.randrange(rows)] = generator.uniform(0.1, 2.0)
    lines.append(" ".join(f"{value:.17g}" for value in lhs))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> dict[str, list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return {line.split()[0]: line.split()[1:] for line in output.splitlines()}


def compare(native: dict[str, list[str]], rust: dict[str, list[str]]) -> bool:
    if native.keys() != rust.keys():
        return False
    for field, native_values in native.items():
        rust_values = rust[field]
        if len(native_values) != len(rust_values):
            return False
        if field.endswith("_positions"):
            if native_values != rust_values:
                return False
        elif not all(
            math.isclose(float(left), float(right), rel_tol=1e-14, abs_tol=1e-14)
            for left, right in zip(native_values, rust_values, strict=True)
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0xC011A2)
    parser.add_argument(
        "--native", type=Path, default=ROOT / "target/native/update_row_reference_adapter"
    )
    parser.add_argument(
        "--rust", type=Path, default=ROOT / "target/debug/examples/update_row_trace"
    )
    args = parser.parse_args()
    generator = random.Random(args.seed)
    for case in range(args.cases):
        data = generate(generator)
        native = run(args.native, data)
        rust = run(args.rust, data)
        if not compare(native, rust):
            print(data, end="")
            raise AssertionError(f"case {case}:\nGLOP: {native}\ngloprs: {rust}")
    print(f"{args.cases} UpdateRow traces agree")


if __name__ == "__main__":
    main()
