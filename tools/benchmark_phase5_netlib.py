#!/usr/bin/env python3
"""Serial native/Rust public-solver timings with presolve on and off."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"


def run(executable: Path, model: Path, mode: str) -> tuple[str, int, float]:
    command = [str(executable), str(model)]
    if mode == "raw":
        command.append("raw")
    output = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=25
    ).stdout.splitlines()
    status, iterations = output[0].split()[:2]
    elapsed = float(output[-1].split()[1])
    return status, int(iterations), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--models", nargs="+", required=True)
    options = parser.parse_args()
    for name in options.models:
        model = MODELS / f"{name}.mps"
        results = {}
        for mode in ("raw", "default"):
            times: dict[str, list[float]] = {"native": [], "rust": []}
            outcomes = {}
            for trial in range(options.trials):
                order = (("native", NATIVE), ("rust", RUST))
                if trial % 2:
                    order = tuple(reversed(order))
                for label, executable in order:
                    status, iterations, elapsed = run(executable, model, mode)
                    outcomes[label] = (status, iterations)
                    times[label].append(elapsed)
            if mode == "default" and outcomes["native"] != outcomes["rust"]:
                raise AssertionError(f"{name} {mode}: {outcomes}")
            results[mode] = {
                "native": statistics.median(times["native"]),
                "rust": statistics.median(times["rust"]),
                "native_status": outcomes["native"][0],
                "rust_status": outcomes["rust"][0],
                "native_iterations": outcomes["native"][1],
                "rust_iterations": outcomes["rust"][1],
            }
        raw, default = results["raw"], results["default"]
        print(json.dumps({
            "model": name,
            "raw": raw,
            "default": default,
            "native_presolve_speedup": raw["native"] / default["native"],
            "rust_presolve_speedup": raw["rust"] / default["rust"],
            "default_rust_native_ratio": default["rust"] / default["native"],
        }))


if __name__ == "__main__":
    main()
