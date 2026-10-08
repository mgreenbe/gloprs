#!/usr/bin/env python3
"""Compare matrix utility behavior with the pinned GLOP implementation."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    rows = generator.randint(1, 20)
    columns = generator.randint(rows, 35)
    dense = [[0.0] * rows for _ in range(columns)]
    for column in range(columns):
        for row in range(rows):
            if generator.random() < 0.2:
                dense[column][row] = generator.randint(-8, 8) / 2.0
    for column in range(1, columns):
        if generator.random() < 0.25:
            source = generator.randrange(column)
            multiplier = generator.choice([-3.0, -2.0, 0.5, 2.0, 3.0])
            dense[column] = [multiplier * value for value in dense[source]]
    entries = [
        (row, column, value)
        for column in range(columns)
        for row, value in enumerate(dense[column])
        if value != 0.0
    ]
    fields = [f"{rows} {columns} {len(entries)} 1e-12"]
    fields.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    return "\n".join(fields) + "\n"


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x6A7A1C)
    native_path = ROOT / "target/native/matrix_utils_reference_adapter"
    rust_path = ROOT / "target/debug/examples/matrix_utils_trace"
    for case in range(args.cases):
        data = generate(generator)
        native = run(native_path, data)
        rust = run(rust_path, data)
        if native != rust:
            raise AssertionError(f"case {case}:\n{native}\n!=\n{rust}")
    print(f"{args.cases} matrix-utils traces agree")


if __name__ == "__main__":
    main()
