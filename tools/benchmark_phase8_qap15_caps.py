#!/usr/bin/env python3
"""Serial, paired direct qap15 timings at matched simplex iteration caps.

Each adapter parses the MPS before starting its solve timer. The difference
between cap timings estimates an interval cost, not a per-stage breakdown.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "datasets/netlib/mps/qap15.mps"
NATIVE = ROOT / "target/native/netlib_timing_reference_adapter"
RUST = ROOT / "target/release/examples/netlib_timing"


def run(binary: Path, cap: int, wall_limit: float) -> dict:
    output = subprocess.run(
        [binary, MODEL, "1", "dual", str(cap), str(wall_limit)],
        check=True,
        capture_output=True,
        text=True,
        timeout=wall_limit + 30,
    ).stdout.split()
    if len(output) != 4:
        raise RuntimeError(f"unexpected timing output from {binary}: {output}")
    return {
        "seconds": float(output[0]),
        "status": output[1],
        "iterations": int(output[2]),
        "objective": float(output[3]),
    }


def median_and_mad(values: list[float]) -> dict:
    median = statistics.median(values)
    return {
        "median_seconds": median,
        "mad_seconds": statistics.median(abs(value - median) for value in values),
        "trials_seconds": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", type=int, nargs="+", default=[25_000, 30_000])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--wall-limit", type=float, default=240.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.warmups < 0 or any(cap <= 0 for cap in args.caps):
        parser.error("trials must be positive, warmups nonnegative, and caps positive")
    if args.caps != sorted(set(args.caps)):
        parser.error("caps must be strictly increasing")

    results: list[dict] = []
    for cap in args.caps:
        for _ in range(args.warmups):
            for binary in (NATIVE, RUST):
                run(binary, cap, args.wall_limit)
        times: dict[str, list[float]] = {"native": [], "rust": []}
        for trial in range(args.trials):
            order = (("native", NATIVE), ("rust", RUST))
            if trial % 2:
                order = tuple(reversed(order))
            outcomes = {}
            for label, binary in order:
                result = run(binary, cap, args.wall_limit)
                outcomes[label] = result
                times[label].append(result["seconds"])
            native, rust = outcomes["native"], outcomes["rust"]
            if (native["status"], native["iterations"]) != (
                rust["status"], rust["iterations"]
            ) or not math.isclose(
                native["objective"], rust["objective"], rel_tol=1e-8, abs_tol=1e-8
            ):
                raise RuntimeError(f"cap {cap}: divergent outcomes: {outcomes}")
            print(f"cap={cap} trial={trial + 1} native={native} rust={rust}", flush=True)
        summary = {label: median_and_mad(values) for label, values in times.items()}
        summary["cap"] = cap
        summary["ratio"] = (
            summary["rust"]["median_seconds"] / summary["native"]["median_seconds"]
        )
        results.append(summary)

    intervals = []
    for before, after in zip(results, results[1:]):
        native = after["native"]["median_seconds"] - before["native"]["median_seconds"]
        rust = after["rust"]["median_seconds"] - before["rust"]["median_seconds"]
        intervals.append({
            "start_cap": before["cap"],
            "end_cap": after["cap"],
            "native_seconds": native,
            "rust_seconds": rust,
            "ratio": rust / native,
            "caveat": "difference of independent capped solves, not a stage timer",
        })
    report = {
        "machine": platform.platform(),
        "gloprs_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "trials": args.trials,
        "warmups": args.warmups,
        "results": results,
        "intervals": intervals,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(intervals, indent=2), flush=True)


if __name__ == "__main__":
    main()
