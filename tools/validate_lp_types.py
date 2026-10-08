#!/usr/bin/env python3
"""Compare lp_types status, scalar, and bit-vector behavior with pinned GLOP."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    size = generator.randint(1, 200)
    other_size = generator.randint(0, 200)
    positions = [i for i in range(size) if generator.random() < 0.35]
    other_positions = [i for i in range(other_size) if generator.random() < 0.35]
    query = generator.randrange(size)
    return "\n".join(
        [
            f"{size} {len(positions)} {other_size} {len(other_positions)} {query}",
            " ".join(map(str, positions)),
            " ".join(map(str, other_positions)),
        ]
    ) + "\n"


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x1F7A9E5)
    args = parser.parse_args()
    generator = random.Random(args.seed)
    native = ROOT / "target/native/lp_types_reference_adapter"
    rust = ROOT / "target/debug/examples/lp_types_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if expected != actual:
            raise AssertionError(f"case {case}:\n{expected}\n!=\n{actual}")
    print(f"{args.cases} lp_types traces agree")


if __name__ == "__main__":
    main()
