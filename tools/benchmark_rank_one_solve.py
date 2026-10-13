#!/usr/bin/env python3
"""Compare isolated dense rank-one solves, excluding construction and reset."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/rank_one_solve_benchmark_reference_adapter"
RUST = ROOT / "target/release/examples/rank_one_solve_benchmark"


def run(binary: Path, parameters: list[int]) -> tuple[float, float]:
    output = subprocess.run(
        [binary, *map(str, parameters)], check=True, capture_output=True, text=True
    ).stdout.split()
    return float(output[0]), float(output[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--updates", type=int, nargs="+", default=[8, 32, 96])
    parser.add_argument("--entries", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if min(args.dimension, args.repetitions, args.trials, *args.updates, *args.entries) < 1:
        parser.error("sizes, repetitions, and trials must be positive")
    for updates in args.updates:
        for entries in args.entries:
            for left in (0, 1):
                parameters = [args.dimension, updates, entries, args.repetitions, left]
                times: dict[str, list[float]] = {"native": [], "rust": []}
                for trial in range(args.trials + 1):
                    order = [("native", NATIVE), ("rust", RUST)]
                    if trial % 2:
                        order.reverse()
                    results = {}
                    for label, binary in order:
                        nanos, checksum = run(binary, parameters)
                        results[label] = checksum
                        if trial:
                            times[label].append(nanos)
                    if not math.isclose(
                        results["native"], results["rust"], rel_tol=1e-12, abs_tol=1e-10
                    ):
                        raise RuntimeError(f"checksums differ: {parameters} {results}")
                native = statistics.median(times["native"])
                rust = statistics.median(times["rust"])
                print(
                    f"updates={updates} entries={entries} left={left} "
                    f"native={native:.1f}ns rust={rust:.1f}ns ratio={rust/native:.3f} "
                    f"native_range=[{min(times['native']):.1f},{max(times['native']):.1f}] "
                    f"rust_range=[{min(times['rust']):.1f},{max(times['rust']):.1f}]",
                    flush=True,
                )


if __name__ == "__main__":
    main()
