#!/usr/bin/env python3
"""Alternate two release direct-solve binaries on selected Netlib models."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"
FAST = ROOT / "baselines/netlib-dual-trajectories.json.gz"


def run(binary: Path, model: str, mode: str) -> tuple[float, str, int]:
    command = [binary, MODELS / f"{model}.mps"]
    if mode == "presolved":
        command.extend(("presolved_solve_timing", "1"))
    fields = subprocess.run(
        command,
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.split()
    if mode == "presolved":
        result = dict(zip(fields[::2], fields[1::2], strict=True))
        return float(result["solve_seconds"]), result["status"], int(result["iterations"])
    return float(fields[0]), fields[1], int(fields[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["dfl001", "fit2d"])
    parser.add_argument("--fast96", action="store_true", help="use the full fast-96 fixture")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--mode", choices=("direct", "presolved"), default="direct")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    if args.fast96:
        with gzip.open(FAST, "rt", encoding="utf-8") as stream:
            models = [case["name"] for case in json.load(stream)["results"]]
    else:
        models = args.models
    total_a = 0.0
    total_b = 0.0
    ratios = []
    for model in models:
        times: dict[str, list[float]] = {"a": [], "b": []}
        for trial in range(args.trials + 1):
            order = [("a", args.a), ("b", args.b)]
            if trial % 2:
                order.reverse()
            outcomes = []
            for label, binary in order:
                seconds, status, iterations = run(binary, model, args.mode)
                outcomes.append((status, iterations))
                if trial:
                    times[label].append(seconds)
            if outcomes[0] != outcomes[1]:
                raise RuntimeError(f"{model}: {outcomes}")
        a = statistics.median(times["a"])
        b = statistics.median(times["b"])
        total_a += a
        total_b += b
        ratios.append(b / a)
        print(f"{model}: A={a:.6f}s B={b:.6f}s B/A={b/a:.3f} "
              f"trials A={times['a']} B={times['b']}", flush=True)
    print(f"aggregate: models={len(models)} A={total_a:.6f}s B={total_b:.6f}s "
          f"B/A={total_b/total_a:.3f} median_model={statistics.median(ratios):.3f}")


if __name__ == "__main__":
    main()
