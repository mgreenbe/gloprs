#!/usr/bin/env python3
"""Alternate two release direct-solve binaries on selected Netlib models."""

from __future__ import annotations

import argparse
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"


def run(binary: Path, model: str) -> tuple[float, str, int]:
    fields = subprocess.run(
        [binary, MODELS / f"{model}.mps"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.split()
    return float(fields[0]), fields[1], int(fields[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["dfl001", "fit2d"])
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    for model in args.models:
        times: dict[str, list[float]] = {"a": [], "b": []}
        for trial in range(args.trials + 1):
            order = [("a", args.a), ("b", args.b)]
            if trial % 2:
                order.reverse()
            outcomes = []
            for label, binary in order:
                seconds, status, iterations = run(binary, model)
                outcomes.append((status, iterations))
                if trial:
                    times[label].append(seconds)
            if outcomes[0] != outcomes[1]:
                raise RuntimeError(f"{model}: {outcomes}")
        a = statistics.median(times["a"])
        b = statistics.median(times["b"])
        print(f"{model}: A={a:.6f}s B={b:.6f}s B/A={b/a:.3f} "
              f"trials A={times['a']} B={times['b']}", flush=True)


if __name__ == "__main__":
    main()
