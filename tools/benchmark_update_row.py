#!/usr/bin/env python3
"""Compare GLOP and gloprs column-wise update-row benchmark hooks."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/update_row_benchmark_reference_adapter"
RUST = ROOT / "target/release/examples/update_row_benchmark"


def run(binary: Path, parameters: list[int]) -> tuple[float, float]:
    fields = subprocess.run(
        [binary, *map(str, parameters)], check=True, capture_output=True, text=True
    ).stdout.split()
    return float(fields[0]), float(fields[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--entries", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.repetitions, args.trials) < 1 or min(args.entries) < 0:
        parser.error("sizes, repetitions, and trials must be positive; entries nonnegative")
    for entries in args.entries:
        for validated in (0, 1):
            parameters = [args.rows, args.columns, entries, args.repetitions]
            times: dict[str, list[float]] = {"native": [], "rust": []}
            for trial in range(args.trials + 1):
                order = [("native", NATIVE), ("rust", RUST)]
                if trial % 2:
                    order.reverse()
                checksums = {}
                for label, binary in order:
                    call = parameters if label == "native" else [*parameters, validated]
                    micros, checksum = run(binary, call)
                    checksums[label] = checksum
                    if trial:
                        times[label].append(micros)
                if not math.isclose(
                    checksums["native"], checksums["rust"], rel_tol=1e-12, abs_tol=1e-10
                ):
                    raise RuntimeError(f"checksums differ: {parameters} {checksums}")
            native = statistics.median(times["native"])
            rust = statistics.median(times["rust"])
            print(
                f"rows={args.rows} structural={args.columns} entries={entries} "
                f"validated={validated} native={native:.2f}us rust={rust:.2f}us "
                f"ratio={rust/native:.3f} "
                f"ranges=[{min(times['native']):.2f},{max(times['native']):.2f}]/"
                f"[{min(times['rust']):.2f},{max(times['rust']):.2f}]",
                flush=True,
            )


if __name__ == "__main__":
    main()
