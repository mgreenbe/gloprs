#!/usr/bin/env python3
"""Run reproducible in-process LU timing comparisons against pinned GLOP."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import subprocess
from pathlib import Path

from validate_lu import generated_matrix


ROOT = Path(__file__).resolve().parents[1]


def encode(matrix: list[list[float]], repetitions: int) -> str:
    entries = [
        (row, column, value)
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != 0.0
    ]
    lines = [f"{len(matrix)} {len(entries)} {repetitions}"]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    return "\n".join(lines) + "\n"


def time(executable: Path, data: str) -> tuple[float, float]:
    output = subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout.split()
    return float(output[0]), float(output[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500])
    parser.add_argument("--density", type=float, default=0.02)
    args = parser.parse_args()
    generator = random.Random(0x1AB3C4)
    for n in args.sizes:
        data = encode(generated_matrix(generator, n, args.density), args.repetitions)
        native = [time(ROOT / "target/native/lu_benchmark_reference_adapter", data) for _ in range(args.trials)]
        rust = [time(ROOT / "target/release/examples/lu_benchmark", data) for _ in range(args.trials)]
        native_checksum = native[0][1]
        rust_checksum = rust[0][1]
        if not all(value[1] == native_checksum for value in native) or not all(
            value[1] == rust_checksum for value in rust
        ):
            raise AssertionError("benchmark checksum changed between trials")
        if not math.isclose(native_checksum, rust_checksum, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(
                f"benchmark checksums differ: {native_checksum} != {rust_checksum}"
            )
        native_time = statistics.median(value[0] for value in native)
        rust_time = statistics.median(value[0] for value in rust)
        print(
            f"n={n} native={native_time:.6f}s rust={rust_time:.6f}s "
            f"ratio={rust_time / native_time:.3f}"
        )


if __name__ == "__main__":
    main()
