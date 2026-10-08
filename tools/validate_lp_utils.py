#!/usr/bin/env python3
"""Compare numerically ordered lp_utils reductions with pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    n = generator.randint(0, 257)
    values = []
    for _ in range(2 * n):
        if generator.random() < 0.15:
            values.append(0.0)
        else:
            values.append(
                math.copysign(
                    10.0 ** generator.uniform(-100.0, 100.0),
                    generator.uniform(-1.0, 1.0),
                )
            )
    return f"{n}\n" + " ".join(f"{value:.17g}" for value in values) + "\n"


def run(executable: Path, data: str) -> dict[str, list[str]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    return {line.split()[0]: line.split()[1:] for line in output.splitlines()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x1F0015)
    native_path = ROOT / "target/native/lp_utils_reference_adapter"
    rust_path = ROOT / "target/debug/examples/lp_utils_trace"
    for case in range(args.cases):
        data = generate(generator)
        native = run(native_path, data)
        rust = run(rust_path, data)
        if native.keys() != rust.keys():
            raise AssertionError(f"case {case}: fields differ")
        for field in native:
            if len(native[field]) != len(rust[field]):
                raise AssertionError(f"case {case}: {field} lengths differ")
            exact = field in {
                "reset_values",
                "support",
                "permuted",
                "known_negated",
                "clear_protocol",
            }
            if any(
                float(left) != float(right)
                if exact
                else not math.isclose(
                    float(left), float(right), rel_tol=2e-15, abs_tol=0.0
                )
                for left, right in zip(native[field], rust[field], strict=True)
            ):
                raise AssertionError(
                    f"case {case} {field}: {native[field]} != {rust[field]}"
                )
    print(f"{args.cases} lp_utils traces agree (reset within cross-compiler ulps)")


if __name__ == "__main__":
    main()
