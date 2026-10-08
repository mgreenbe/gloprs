#!/usr/bin/env python3
"""Compare compact sparse-matrix behavior with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    rows = generator.randint(1, 40)
    columns = generator.randint(0, 40)
    entries = []
    for column in range(columns):
        for row in range(rows):
            if generator.random() < 0.15:
                entries.append((row, column, generator.randint(-20, 20) / 4.0))
                if generator.random() < 0.05:
                    entries.append((row, column, generator.randint(-20, 20) / 4.0))
    fields = [f"{rows} {columns} {len(entries)}"]
    fields.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    return "\n".join(fields) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return [line.split() for line in output.splitlines()]


def equal(left: list[str], right: list[str]) -> bool:
    if left[0] != right[0] or len(left) != len(right):
        return False
    if left[0] == "products":
        return all(
            math.isclose(float(a), float(b), rel_tol=2e-15, abs_tol=0.0)
            for a, b in zip(left[1:], right[1:], strict=True)
        )
    if left[0] == "sparse_view":
        return left[1:4] == right[1:4] and all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=0.0)
            for a, b in zip(left[4:], right[4:], strict=True)
        )
    if left[0] == "compact_view":
        if left[1:4] != right[1:4] or any(
            float(a) != float(b)
            for a, b in zip(left[4:6], right[4:6], strict=True)
        ):
            return False
        for position in range(6, len(left), 3):
            if left[position : position + 2] != right[position : position + 2]:
                return False
            if float(left[position + 2]) != float(right[position + 2]):
                return False
        return True
    if left[1:4] != right[1:4]:
        return False
    for position in range(4, len(left), 3):
        if left[position : position + 2] != right[position : position + 2]:
            return False
        if not math.isclose(
            float(left[position + 2]), float(right[position + 2]), rel_tol=0.0, abs_tol=0.0
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0xC05C)
    native = ROOT / "target/native/compact_sparse_reference_adapter"
    rust = ROOT / "target/debug/examples/compact_sparse_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            not equal(left, right) for left, right in zip(expected, actual, strict=True)
        ):
            raise AssertionError(f"case {case}: {expected} != {actual}")
    print(f"{args.cases} compact sparse-matrix traces agree")


if __name__ == "__main__":
    main()
