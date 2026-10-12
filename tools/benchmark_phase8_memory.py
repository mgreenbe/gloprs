#!/usr/bin/env python3
"""Compare peak resident memory for matched direct Netlib solves on macOS."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "datasets/netlib"
FAST = ROOT / "baselines/netlib-dual-trajectories.json.gz"
NATIVE = ROOT / "target/native/netlib_timing_reference_adapter"
RUST = ROOT / "target/release/examples/netlib_timing"


def run(binary: Path, model: Path) -> tuple[int, int, str, int]:
    process = subprocess.run(
        ["/usr/bin/time", "-l", binary, model],
        check=True, capture_output=True, text=True, timeout=30,
    )
    status, iterations = process.stdout.split()[1:3]
    resident = re.search(r"^\s*(\d+)\s+maximum resident set size$",
                         process.stderr, re.MULTILINE)
    footprint = re.search(r"^\s*(\d+)\s+peak memory footprint$",
                          process.stderr, re.MULTILINE)
    if resident is None or footprint is None:
        raise RuntimeError(f"unrecognized macOS time -l output: {process.stderr}")
    return int(resident[1]), int(footprint[1]), status, int(iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.trials < 1:
        parser.error("trials must be positive")
    with gzip.open(FAST, "rt", encoding="utf-8") as stream:
        cases = json.load(stream)["results"]
    if options.models is not None:
        selected = set(options.models)
        cases = [case for case in cases if case["name"] in selected]
        unknown = selected - {case["name"] for case in cases}
        if unknown:
            parser.error(f"models outside fast-96 fixture: {sorted(unknown)}")
    report = []
    for number, case in enumerate(cases, start=1):
        model = DATASET / case["input"]
        measurements: dict[str, list[tuple[int, int]]] = {"native": [], "rust": []}
        for trial in range(options.trials):
            order = (("native", NATIVE), ("rust", RUST))
            if trial % 2:
                order = tuple(reversed(order))
            for name, executable in order:
                resident, footprint, status, iterations = run(executable, model)
                if (status, iterations) != (case["status"], case["iterations"]):
                    raise RuntimeError(f"{case['name']} {name}: {status}/{iterations}")
                measurements[name].append((resident, footprint))
        native_rss = statistics.median(value[0] for value in measurements["native"])
        rust_rss = statistics.median(value[0] for value in measurements["rust"])
        result = {
            "name": case["name"],
            "native_peak_rss_bytes": native_rss,
            "rust_peak_rss_bytes": rust_rss,
            "rss_ratio": rust_rss / native_rss,
            "native_peak_footprint_bytes": statistics.median(
                value[1] for value in measurements["native"]),
            "rust_peak_footprint_bytes": statistics.median(
                value[1] for value in measurements["rust"]),
            "native_trials": measurements["native"],
            "rust_trials": measurements["rust"],
        }
        report.append(result)
        print(f"[{number:02d}/{len(cases)}] {case['name']:<10} "
              f"RSS native={native_rss/1048576:.1f} MiB "
              f"Rust={rust_rss/1048576:.1f} MiB ratio={rust_rss/native_rss:.3f}", flush=True)
    output = {
        "machine": platform.platform(),
        "timing_scope": "whole process, including MPS parsing; macOS /usr/bin/time -l",
        "upstream_commit": "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5",
        "trials": options.trials,
        "summary": {
            "median_model_rss_ratio": statistics.median(item["rss_ratio"] for item in report),
            "maximum_native_peak_rss_bytes": max(item["native_peak_rss_bytes"] for item in report),
            "maximum_rust_peak_rss_bytes": max(item["rust_peak_rss_bytes"] for item in report),
        },
        "results": report,
    }
    print(json.dumps(output["summary"], indent=2))
    if options.output is not None:
        options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
