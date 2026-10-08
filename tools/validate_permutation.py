#!/usr/bin/env python3
"""Compare typed permutation behavior with the pinned GLOP implementation."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    size = generator.randint(0, 100)
    permutation = list(range(size))
    generator.shuffle(permutation)
    if size and generator.random() < 0.2:
        if generator.random() < 0.5:
            permutation[generator.randrange(size)] = -1
        else:
            permutation[generator.randrange(size)] = size
    values = [generator.randint(-(2**40), 2**40) for _ in range(size)]
    return (
        f"{size}\n"
        + " ".join(map(str, permutation))
        + "\n"
        + " ".join(map(str, values))
        + "\n"
    )


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0xFEA47E)
    args = parser.parse_args()
    generator = random.Random(args.seed)
    native = ROOT / "target/native/permutation_reference_adapter"
    rust = ROOT / "target/debug/examples/permutation_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if expected != actual:
            raise AssertionError(f"case {case}:\n{expected}\n!=\n{actual}")
    print(f"{args.cases} permutation traces agree")


if __name__ == "__main__":
    main()
