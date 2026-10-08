#!/usr/bin/env python3
"""Compare sparse-vector transformations with the pinned GLOP implementation."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    n = generator.randint(1, 80)
    count = generator.randint(0, 3 * n)
    entries = [
        (generator.randrange(n), generator.choice([0.0, generator.uniform(-10.0, 10.0)]))
        for _ in range(count)
    ]
    weights = [generator.uniform(0.0, 3.0) for _ in range(n)]
    threshold = generator.uniform(0.0, 2.0)
    destinations = list(range(n))
    generator.shuffle(destinations)
    partial = [destination if generator.random() < 0.7 else -1 for destination in destinations]
    tags = [generator.randrange(n) if generator.random() < 0.35 else -1 for _ in range(n)]
    fields = [str(n), str(count)]
    fields.extend(f"{row} {value:.17g}" for row, value in entries)
    fields.extend(f"{value:.17g}" for value in weights)
    fields.append(f"{threshold:.17g}")
    fields.extend(map(str, partial))
    fields.extend(map(str, tags))
    return "\n".join(fields) + "\n"


def run(executable: Path, data: str) -> dict[str, list[tuple[int, float]]]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout
    result = {}
    for line in output.splitlines():
        fields = line.split()
        count = int(fields[1])
        entries = [(int(fields[i]), float(fields[i + 1])) for i in range(2, len(fields), 2)]
        assert len(entries) == count
        result[fields[0]] = entries
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x5A4E5E)
    native_path = ROOT / "target/native/sparse_vector_reference_adapter"
    rust_path = ROOT / "target/debug/examples/sparse_vector_trace"
    for case in range(args.cases):
        data = generate(generator)
        native = run(native_path, data)
        rust = run(rust_path, data)
        if native.keys() != rust.keys():
            raise AssertionError(f"case {case}: fields differ")
        for field in native:
            if len(native[field]) != len(rust[field]):
                raise AssertionError(f"case {case} {field}: lengths differ")
            for left, right in zip(native[field], rust[field], strict=True):
                if left[0] != right[0] or not math.isclose(
                    left[1], right[1], rel_tol=2e-15, abs_tol=0.0
                ):
                    raise AssertionError(f"case {case} {field}: {native[field]} != {rust[field]}")
    print(f"{args.cases} sparse-vector traces agree")


if __name__ == "__main__":
    main()
