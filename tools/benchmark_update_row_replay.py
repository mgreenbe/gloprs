#!/usr/bin/env python3
"""Replay captured production update rows through native and Rust kernels."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/update_row_replay_reference_adapter"
RUST = ROOT / "target/release/examples/update_row_replay"


def run(binary: Path, fixture: Path, mode: str, repetitions: int) -> tuple[float, float, int, int]:
    fields = subprocess.run(
        [binary, fixture, mode, str(repetitions)],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return float(fields[0]), float(fields[1]), int(fields[2]), int(fields[3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("dot", "output", "count", "coeffs", "positions", "full"),
                        default=["dot", "output", "count", "coeffs", "positions", "full"])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.repetitions < 1 or args.trials < 1:
        parser.error("repetitions and trials must be positive")
    for fixture in args.fixtures:
        for mode in args.modes:
            times: dict[str, list[float]] = {"native": [], "rust": []}
            metadata = None
            for trial in range(args.trials + 1):
                order = [("native", NATIVE), ("rust", RUST)]
                if trial % 2:
                    order.reverse()
                results = {}
                for label, binary in order:
                    micros, checksum, relevant, nonzeros = run(
                        binary, fixture, mode, args.repetitions
                    )
                    results[label] = checksum
                    metadata = (relevant, nonzeros)
                    if trial:
                        times[label].append(micros)
                if not math.isclose(results["native"], results["rust"],
                                    rel_tol=1e-11, abs_tol=1e-8):
                    raise RuntimeError(f"checksums disagree: {fixture} {mode} {results}")
            native = statistics.median(times["native"])
            rust = statistics.median(times["rust"])
            print(f"{fixture.parent.name}/{fixture.stem} {mode} "
                  f"relevant={metadata[0]} nnz={metadata[1]} "
                  f"native={native:.2f}us rust={rust:.2f}us ratio={rust/native:.3f} "
                  f"ranges=[{min(times['native']):.2f},{max(times['native']):.2f}]/"
                  f"[{min(times['rust']):.2f},{max(times['rust']):.2f}]", flush=True)


if __name__ == "__main__":
    main()
