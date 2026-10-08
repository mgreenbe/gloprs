#!/usr/bin/env python3
"""Compare GLOP and gloprs primal steepest-edge and Devex traces."""

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
    parsed = []
    for text in texts:
        parsed.append({
            line.split(":", 1)[0].strip(): [float(value) for value in pattern.findall(line)]
            for line in text.splitlines()
            if ":" in line
        })
    if parsed[0].keys() != parsed[1].keys():
        return False
    for name in parsed[0]:
        expected = parsed[0][name]
        actual = parsed[1][name]
        if len(expected) != len(actual) or expected[0] != actual[0]:
            return False
        tolerance = 1.0 if name == "lower_bounded_norms" else 5e-12
        if not all(
            math.isclose(a, b, rel_tol=3e-11, abs_tol=tolerance)
            for a, b in zip(expected[1:], actual[1:], strict=True)
        ):
            return False
    return True


def generate(generator: random.Random) -> str:
    n = generator.randint(1, 48)
    nonbasic = generator.randint(1, 12)
    leaving = generator.randrange(n)
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
    values: list[float] = []
    for column in range(nonbasic):
        current = [
            generator.uniform(-3.0, 3.0)
            if column == 0 or generator.random() >= 0.5
            else 0.0
            for _ in range(n)
        ]
        values.extend(current)
    lines = [f"{n} {nonbasic} {leaving} {len(off_diagonal)} -1"]
    lines.append(" ".join(f"{value:.17g}" for value in diagonal))
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in off_diagonal)
    lines.append(" ".join(f"{value:.17g}" for value in values))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> dict[str, list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return {line.split()[0]: line.split()[1:] for line in output.splitlines()}


def compare(native: dict[str, list[str]], rust: dict[str, list[str]]) -> bool:
    scalar_fields = {"precise", "precision_watcher", "clear_watcher"}
    if native.keys() != rust.keys() or any(
        native[field] != rust[field] for field in scalar_fields
    ) or not stats_equivalent(native["stats_hex"], rust["stats_hex"]):
        return False
    for field in native:
        if field in scalar_fields or field == "stats_hex":
            continue
        if len(native[field]) != len(rust[field]) or not all(
            math.isclose(float(a), float(b), rel_tol=3e-11, abs_tol=3e-12)
            for a, b in zip(native[field], rust[field], strict=True)
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0xA11CEED6E)
    parser.add_argument(
        "--native",
        type=Path,
        default=ROOT / "target/native/primal_edge_norms_reference_adapter",
    )
    parser.add_argument(
        "--rust",
        type=Path,
        default=ROOT / "target/debug/examples/primal_edge_norms_trace",
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
    nonbasic = 2
    off_diagonal = [
        (row, column, 0.001 * ((row + column) % 7 - 3))
        for column in range(n)
        for row in range(column + 1, n)
    ]
    lines = [f"{n} {nonbasic} 0 {len(off_diagonal)} 0", " ".join("1" for _ in range(n))]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in off_diagonal)
    lines.append(" ".join("1" for _ in range(n * nonbasic)))
    data = "\n".join(lines) + "\n"
    native = run(args.native, data)
    rust = run(args.rust, data)
    assert int(native["entries"][0]) > 10_000
    if not compare(native, rust):
        raise AssertionError(f"limited case:\nGLOP: {native}\ngloprs: {rust}")
    print(f"{args.cases} ordinary plus one limited primal-edge-norm traces agree")


if __name__ == "__main__":
    main()
