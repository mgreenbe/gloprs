#!/usr/bin/env python3
"""Compare pinned GLOP and gloprs initial-basis construction."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def encode(
    rows: int,
    columns: list[list[tuple[int, float]]],
    candidates: list[int],
) -> str:
    entries = [
        (row, column, value)
        for column, values in enumerate(columns)
        for row, value in values
    ]
    lines = [f"{rows} {len(columns)} {len(entries)} {len(candidates)}"]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    lines.append(" ".join(map(str, candidates)))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def generated_case(generator: random.Random) -> tuple[int, list[list[tuple[int, float]]], list[int]]:
    rows = generator.randint(1, 35)
    structural = generator.randint(0, 2 * rows + 8)
    columns: list[list[tuple[int, float]]] = []
    for _ in range(structural):
        values = [
            (row, generator.uniform(-2.0, 2.0))
            for row in range(rows)
            if generator.random() < 0.18
        ]
        columns.append(values)
    # GLOP's matrix includes the trailing identity/slack block used to fill
    # rows not covered by the stable independent candidate subset.
    for row in range(rows):
        columns.append([(row, 1.0)])
    candidates = list(range(structural))
    generator.shuffle(candidates)
    candidates = candidates[: generator.randint(0, len(candidates))]
    return rows, columns, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0xBA515)
    parser.add_argument(
        "--native",
        type=Path,
        default=ROOT / "target/native/initial_basis_reference_adapter",
    )
    parser.add_argument(
        "--rust",
        type=Path,
        default=ROOT / "target/debug/examples/initial_basis_trace",
    )
    args = parser.parse_args()
    generator = random.Random(args.seed)
    for case in range(args.cases):
        data = encode(*generated_case(generator))
        native = run(args.native, data)
        rust = run(args.rust, data)
        if native != rust:
            print(data, end="")
            raise AssertionError(f"case {case}: {native!r} != {rust!r}")
    print(f"{args.cases} initial-basis traces agree")


if __name__ == "__main__":
    main()
