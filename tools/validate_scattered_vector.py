#!/usr/bin/env python3
"""Compare scattered-vector state transitions with pinned GLOP."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    size = generator.randint(0, 160)
    count = generator.randint(0, 2 * size) if size else 0
    ratio = generator.choice([0.0, 0.025, 0.05, 0.5, 0.8, 1.0])
    lines = [f"{size} {count} {ratio:.17g}"]
    for _ in range(count):
        row = generator.randrange(size)
        value = generator.choice([0.0, generator.randint(-16, 16) / 4.0])
        lines.append(f"{row} {value:.17g}")
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def normalized(output: str) -> list[list[str]]:
    result: list[list[str]] = []
    for line in output.splitlines():
        fields: list[str] = []
        for field in line.split():
            try:
                fields.append(float(field).hex())
            except ValueError:
                fields.append(field)
        result.append(fields)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x5CA77E2)
    args = parser.parse_args()
    generator = random.Random(args.seed)
    native = ROOT / "target/native/scattered_vector_reference_adapter"
    rust = ROOT / "target/debug/examples/scattered_vector_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if normalized(expected) != normalized(actual):
            raise AssertionError(f"case {case}:\n{expected}\n!=\n{actual}")
    print(f"{args.cases} scattered-vector traces agree")


if __name__ == "__main__":
    main()
