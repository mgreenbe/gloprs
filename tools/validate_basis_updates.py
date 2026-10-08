#!/usr/bin/env python3
"""Differentially compare pinned GLOP and gloprs middle-product updates."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(
    generator: random.Random, refactorization: bool, dynamic: bool, eta: bool
) -> str:
    n = generator.randint(1, 16)
    sets = 3
    structural_columns = sets * n
    entries = []
    for set_index in range(sets):
        for column in range(n):
            for row in range(n):
                if row == column:
                    value = generator.uniform(1.0, 2.0)
                elif not refactorization and generator.random() < 0.15:
                    value = generator.uniform(-0.05, 0.05)
                else:
                    continue
                entries.append((row, set_index * n + column, value))
    update_count = generator.randint(n, 3 * n)
    updates = []
    generation = [0] * n
    for _ in range(update_count):
        leaving = generator.randrange(n)
        generation[leaving] = (generation[leaving] + 1) % sets
        entering = generation[leaving] * n + leaving
        updates.append((entering, leaving))
    period = generator.randint(1, max(1, n // 2)) if refactorization else 10000
    lines = [
        f"{n} {structural_columns} {len(entries)} {len(updates)} {period} "
        f"{int(dynamic)} {int(not eta)}"
    ]
    lines.extend(f"{r} {c} {v:.17g}" for r, c, v in entries)
    lines.extend(f"{entering} {leaving}" for entering, leaving in updates)
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> dict[str, list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return {line.split()[0]: line.split()[1:] for line in output.splitlines()}


def compare(native: dict[str, list[str]], rust: dict[str, list[str]]) -> bool:
    if native.keys() != rust.keys():
        return False
    for field, values in native.items():
        other = rust[field]
        if field == "stats_hex":
            if values != other:
                return False
            continue
        if len(values) != len(other) or not all(
            math.isclose(float(left), float(right), rel_tol=2e-10, abs_tol=2e-11)
            for left, right in zip(values, other, strict=True)
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0xBA515)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--refactorization", action="store_true")
    parser.add_argument("--eta", action="store_true")
    parser.add_argument(
        "--native",
        type=Path,
        default=ROOT / "target/native/basis_update_reference_adapter",
    )
    parser.add_argument(
        "--rust", type=Path, default=ROOT / "target/debug/examples/basis_update_trace"
    )
    args = parser.parse_args()
    generator = random.Random(args.seed)
    for case in range(args.cases):
        data = generate(
            generator, args.refactorization or args.dynamic, args.dynamic, args.eta
        )
        native = run(args.native, data)
        rust = run(args.rust, data)
        if not compare(native, rust):
            print(data, end="")
            raise AssertionError(f"case {case}:\nGLOP: {native}\ngloprs: {rust}")
    print(f"{args.cases} basis-update traces agree")


if __name__ == "__main__":
    main()
