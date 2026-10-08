#!/usr/bin/env python3
"""Compare isolated rank-one update kernels with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    dimension = generator.randint(1, 80)
    num_updates = generator.randint(0, 12)
    ratio = generator.choice([0.0, 0.025, 0.05, 0.5, 1.0])
    lines = [f"{dimension} {num_updates} {ratio:.17g}"]
    for _ in range(num_updates):
        u_positions = generator.sample(range(dimension), generator.randint(0, dimension))
        v_positions = generator.sample(range(dimension), generator.randint(0, dimension))
        u = [(i, generator.uniform(-2.0, 2.0)) for i in u_positions]
        v = [(i, generator.uniform(-2.0, 2.0)) for i in v_positions]
        dot = generator.uniform(-0.9, 2.0)
        lines.append(f"{len(u)} {len(v)} {dot:.17g}")
        lines.extend(f"{i} {value:.17g}" for i, value in u)
        lines.extend(f"{i} {value:.17g}" for i, value in v)
    rhs = [generator.choice([0.0, generator.uniform(-3.0, 3.0)]) for _ in range(dimension)]
    lines.append(" ".join(f"{value:.17g}" for value in rhs))
    nonzeros = [i for i, value in enumerate(rhs) if value != 0.0]
    if nonzeros and generator.random() < 0.85:
        extras = [i for i, value in enumerate(rhs) if value == 0.0 and generator.random() < 0.1]
        pattern = nonzeros + extras
        generator.shuffle(pattern)
    else:
        pattern = []
    lines.append(f"{len(pattern)} " + " ".join(map(str, pattern)))
    return "\n".join(lines) + "\n"


def run(executable: Path, data: str) -> list[list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return [line.split() for line in output.splitlines()]


def close(left: str, right: str) -> bool:
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return left == right
    if a.is_integer() and b.is_integer() and "e" not in left.lower() and "e" not in right.lower():
        return left == right
    return a == b or math.isclose(a, b, rel_tol=2e-14, abs_tol=1e-300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0xA11CE55)
    args = parser.parse_args()
    generator = random.Random(args.seed)
    native = ROOT / "target/native/rank_one_update_reference_adapter"
    rust = ROOT / "target/debug/examples/rank_one_update_trace"
    for case in range(args.cases):
        data = generate(generator)
        expected = run(native, data)
        actual = run(rust, data)
        if len(expected) != len(actual) or any(
            len(a) != len(b) or any(not close(x, y) for x, y in zip(a, b))
            for a, b in zip(expected, actual)
        ):
            raise AssertionError(f"case {case}:\n{expected}\n!=\n{actual}")
    print(f"{args.cases} rank-one-update traces agree")


if __name__ == "__main__":
    main()
