#!/usr/bin/env python3
"""Compare GLOP and gloprs dual steepest-edge update traces."""

from __future__ import annotations

import argparse
import math
import random
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def stats_equivalent(left: list[str], right: list[str]) -> bool:
    if len(left) != 1 or len(right) != 1:
        return False
    texts = [bytes.fromhex(value[0]).decode() for value in (left, right)]
    pattern = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:e[-+]\d+)?")
    numbers = [[float(value) for value in pattern.findall(text)] for text in texts]
    skeletons = [re.sub(r"\s+", "", pattern.sub("#", text)) for text in texts]
    return skeletons[0] == skeletons[1] and len(numbers[0]) == len(numbers[1]) and all(
        math.isclose(a, b, rel_tol=3e-11, abs_tol=5e-12)
        for a, b in zip(numbers[0], numbers[1], strict=True)
    )


def generate(generator: random.Random) -> str:
    n = generator.randint(1, 64)
    leaving = generator.randrange(n)
    # Keep the triangular bases reasonably conditioned. The purpose of this
    # differential trace is to expose behavioral differences, not to amplify
    # last-bit factorization differences by many orders of magnitude.
    diagonal = [0.0] * n
    row_permutation = list(range(n))
    column_permutation = list(range(n))
    generator.shuffle(row_permutation)
    generator.shuffle(column_permutation)
    off_diagonal = []
    for column in range(n):
        off_diagonal.append(
            (
                row_permutation[column],
                column_permutation[column],
                math.copysign(generator.uniform(0.75, 2.0), generator.uniform(-1, 1)),
            )
        )
    for column in range(1, n):
        column_entries = [
            (
                row_permutation[row],
                column_permutation[column],
                generator.uniform(-0.2, 0.2),
            )
            for row in range(column)
            if generator.random() < 0.18
        ]
        if not column_entries:
            column_entries.append(
                (
                    row_permutation[generator.randrange(column)],
                    column_permutation[column],
                    generator.uniform(-0.2, 0.2),
                )
            )
        off_diagonal.extend(column_entries)
    entering = [
        math.copysign(generator.uniform(0.25, 4.0), generator.uniform(-1.0, 1.0))
        for _ in range(n)
    ]
    lines = [f"{n} {leaving} {len(off_diagonal)} -1"]
    lines.append(" ".join(f"{value:.17g}" for value in diagonal))
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in off_diagonal)
    lines.append(" ".join(f"{value:.17g}" for value in entering))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> dict[str, list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return {line.split()[0]: line.split()[1:] for line in output.splitlines()}


def compare(native: dict[str, list[str]], rust: dict[str, list[str]]) -> bool:
    if (
        native.keys() != rust.keys()
        or native["precise"] != rust["precise"]
        or not stats_equivalent(native["stats_hex"], rust["stats_hex"])
    ):
        return False
    for field in ("entries", "initial", "direction", "left", "updated"):
        if len(native[field]) != len(rust[field]) or not all(
            math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=2e-13)
            for a, b in zip(native[field], rust[field], strict=True)
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0xD0A1ED6E)
    parser.add_argument(
        "--native",
        type=Path,
        default=ROOT / "target/native/dual_edge_norms_reference_adapter",
    )
    parser.add_argument(
        "--rust",
        type=Path,
        default=ROOT / "target/debug/examples/dual_edge_norms_trace",
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
    n = 160
    off_diagonal = [(row, column, 0.001 * ((row + column) % 7 - 3))
                    for column in range(n) for row in range(column + 1, n)]
    lines = [f"{n} 0 {len(off_diagonal)} 0", " ".join("1" for _ in range(n))]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in off_diagonal)
    lines.append(" ".join("1" for _ in range(n)))
    data = "\n".join(lines) + "\n"
    native = run(args.native, data)
    rust = run(args.rust, data)
    assert int(native["entries"][0]) > 10_000
    if not compare(native, rust):
        raise AssertionError(f"limited case:\nGLOP: {native}\ngloprs: {rust}")
    print(f"{args.cases} ordinary plus one limited dual-edge-norm traces agree")


if __name__ == "__main__":
    main()
