#!/usr/bin/env python3
"""Compare GLOP's compact-column dot product with the Rust port in isolation.

Build the Rust release example and the matching native adapter first. Matrix
construction and input parsing are excluded from both internal timers.
"""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/compact_dot_benchmark_reference_adapter"
RUST = ROOT / "target/release/examples/compact_dot_benchmark"


def matrix_input(rows: int, columns: int, entries: int, repetitions: int) -> str:
    generator = random.Random(0xC0A1C7)
    lines = [f"{rows} {columns} {entries} {repetitions}"]
    for _ in range(columns):
        for _ in range(entries):
            row = generator.randrange(rows)
            coefficient = generator.uniform(-1.0, 1.0)
            lines.append(f"{row} {coefficient:.17g}")
    return "\n".join(lines) + "\n"


def run(binary: Path, data: str, mode: str | None = None) -> tuple[float, float]:
    process = subprocess.run(
        [binary, *([mode] if mode else [])],
        input=data, text=True, capture_output=True, check=True
    )
    nanoseconds, checksum = process.stdout.split()
    return float(nanoseconds), float(checksum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--entries", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--rust", type=Path, default=RUST)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.repetitions, args.trials, *args.entries) < 1:
        parser.error("all sizes and counts must be positive")
    for entries in args.entries:
        data = matrix_input(args.rows, args.columns, entries, args.repetitions)
        times: dict[str, list[float]] = {
            "native": [], "rust": [], "validated": [], "bound": []
        }
        for trial in range(args.trials + 1):
            order = [
                ("native", NATIVE), ("rust", args.rust),
                ("validated", args.rust), ("bound", args.rust),
            ]
            if trial % 2:
                order.reverse()
            checksums = []
            for name, binary in order:
                nanoseconds, checksum = run(
                    binary, data, name if name in ("validated", "bound") else None
                )
                checksums.append(checksum)
                if trial:
                    times[name].append(nanoseconds)
            if any(abs(checksums[0] - value) > 1e-10 * max(1.0, abs(checksums[0]))
                   for value in checksums[1:]):
                raise RuntimeError(f"{entries} entries: checksums disagree: {checksums}")
        native = statistics.median(times["native"])
        rust = statistics.median(times["rust"])
        validated = statistics.median(times["validated"])
        bound = statistics.median(times["bound"])
        print(
            f"rows={args.rows} columns={args.columns} entries={entries} "
            f"native={native:.2f} ns rust={rust:.2f} ns "
            f"validated={validated:.2f} ns bound={bound:.2f} ns "
            f"ratios={rust/native:.3f}/{validated/native:.3f}/{bound/native:.3f} "
            f"bound/validated={bound/validated:.3f} "
            f"ranges=[{min(times['native']):.2f},{max(times['native']):.2f}]/"
            f"[{min(times['rust']):.2f},{max(times['rust']):.2f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
