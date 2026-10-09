#!/usr/bin/env python3
"""Time identical dense lower-transpose solves in pinned GLOP and gloprs.

Build the Rust example in release mode and the C++ adapter with
build_glop_reference_adapter.py before running this script. Construction,
parsing, and output are excluded from each executable's internal timer.
"""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/lower_transpose_benchmark_reference_adapter"
RUST = ROOT / "target/release/examples/lower_transpose_benchmark"


def factor_input(n: int, density: float, repetitions: int) -> str:
    generator = random.Random(0x713A5E)
    entries = [(column, column, 1.0) for column in range(n)]
    for column in range(n):
        for row in range(column + 1, n):
            if generator.random() < density:
                coefficient = generator.uniform(-0.01, 0.01)
                entries.append((row, column, coefficient))
    lines = [f"{n} {len(entries)} {repetitions}"]
    lines.extend(f"{row} {column} {value:.17g}" for row, column, value in entries)
    return "\n".join(lines) + "\n"


def run(binary: Path, data: str) -> tuple[float, float]:
    completed = subprocess.run(
        [binary], input=data, text=True, capture_output=True, check=True
    )
    elapsed, checksum = completed.stdout.split()
    return float(elapsed), float(checksum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--density", type=float, default=0.03)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    for n in args.sizes:
        data = factor_input(n, args.density, args.repetitions)
        native_times: list[float] = []
        rust_times: list[float] = []
        for trial in range(args.trials + 1):
            order = [(NATIVE, native_times), (RUST, rust_times)]
            if trial % 2:
                order.reverse()
            checksums = []
            for binary, times in order:
                elapsed, checksum = run(binary, data)
                checksums.append(checksum)
                if trial:
                    times.append(elapsed)
            if abs(checksums[0] - checksums[1]) > 1e-10 * abs(checksums[0]):
                raise RuntimeError(f"n={n}: checksums disagree: {checksums}")
        native = statistics.median(native_times)
        rust = statistics.median(rust_times)
        print(
            f"n={n} density={args.density} repetitions={args.repetitions} "
            f"native={native:.1f} ns rust={rust:.1f} ns ratio={rust/native:.3f} "
            f"ranges=[{min(native_times):.1f},{max(native_times):.1f}]/"
            f"[{min(rust_times):.1f},{max(rust_times):.1f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
