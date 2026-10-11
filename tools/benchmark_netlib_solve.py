#!/usr/bin/env python3
"""Compare serial in-solver time for pinned GLOP and gloprs on Netlib."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "netlib"
FIXTURE = PROJECT_ROOT / "baselines" / "netlib-dual-trajectories.json.gz"
NATIVE = PROJECT_ROOT / "target" / "native" / "netlib_timing_reference_adapter"
RUST = PROJECT_ROOT / "target" / "release" / "examples" / "netlib_timing"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mode", choices=("dual", "primal", "advanced"), default="dual")
    parser.add_argument("--iteration-limit", type=int)
    parser.add_argument("--models", nargs="*", help="benchmark only these Netlib names")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(
    binary: Path, model: Path, mode: str, iteration_limit: int | None
) -> tuple[float, str, int, float]:
    command = [str(binary), str(model)]
    if mode != "dual" or iteration_limit is not None:
        command.extend(("1", mode))
    if iteration_limit is not None:
        command.append(str(iteration_limit))
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30.0
    )
    elapsed, status, iterations, objective = completed.stdout.split()
    return float(elapsed), status, int(iterations), float(objective)


def main() -> None:
    options = arguments()
    if options.iteration_limit is not None and options.iteration_limit < 0:
        raise ValueError("iteration limit must be nonnegative")
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as stream:
        fixture = json.load(stream)
    expected_results = fixture["results"]
    if options.models is not None:
        selected = set(options.models)
        expected_results = [item for item in expected_results if item["name"] in selected]
        missing = selected - {item["name"] for item in expected_results}
        if missing:
            raise ValueError(f"unknown Netlib models: {sorted(missing)}")

    results = []
    for number, expected in enumerate(expected_results, start=1):
        model = DATASET_ROOT / expected["input"]
        for _ in range(options.warmups):
            run(NATIVE, model, options.mode, options.iteration_limit)
            run(RUST, model, options.mode, options.iteration_limit)

        native_times = []
        rust_times = []
        terminal = None
        for trial in range(options.trials):
            order = ((NATIVE, native_times), (RUST, rust_times))
            if trial % 2:
                order = tuple(reversed(order))
            trial_objectives = {}
            trial_outcomes = {}
            for binary, times in order:
                elapsed, status, iterations, objective = run(
                    binary, model, options.mode, options.iteration_limit
                )
                if options.iteration_limit is None and (
                    status != expected["status"] or iterations != expected["iterations"]
                ):
                    raise RuntimeError(
                        f"{expected['name']}: {binary.name} produced "
                        f"{status}/{iterations}, expected "
                        f"{expected['status']}/{expected['iterations']}"
                    )
                trial_objectives[binary] = objective
                trial_outcomes[binary] = (status, iterations)
                times.append(elapsed)
                terminal = {
                    "status": status,
                    "iterations": iterations,
                    "objective": objective,
                }
            if trial_outcomes[RUST] != trial_outcomes[NATIVE]:
                raise RuntimeError(
                    f"{expected['name']}: Rust outcome {trial_outcomes[RUST]} "
                    f"differs from native {trial_outcomes[NATIVE]}"
                )
            rust_objective = trial_objectives[RUST]
            native_objective = trial_objectives[NATIVE]
            if not (
                math.isnan(rust_objective) and math.isnan(native_objective)
                or math.isclose(
                    rust_objective, native_objective, rel_tol=1e-8, abs_tol=1e-8
                )
            ):
                raise RuntimeError(
                    f"{expected['name']}: Rust objective {rust_objective} "
                    f"differs from native {native_objective}"
                )

        native = statistics.median(native_times)
        rust = statistics.median(rust_times)
        result = {
            "name": expected["name"],
            "native_seconds": native,
            "rust_seconds": rust,
            "ratio": rust / native,
            **terminal,
        }
        results.append(result)
        print(
            f"[{number:02d}/{len(expected_results)}] {expected['name']:<10} "
            f"native={native:9.6f}s rust={rust:9.6f}s ratio={rust / native:7.3f}",
            flush=True,
        )

    native_total = sum(result["native_seconds"] for result in results)
    rust_total = sum(result["rust_seconds"] for result in results)
    ratios = [result["ratio"] for result in results]
    summary = {
        "mode": options.mode,
        "iteration_limit": options.iteration_limit,
        "trials": options.trials,
        "warmups": options.warmups,
        "native_total_seconds": native_total,
        "rust_total_seconds": rust_total,
        "aggregate_ratio": rust_total / native_total,
        "median_model_ratio": statistics.median(ratios),
        "results": results,
    }
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"}, indent=2
        )
    )
    if options.output is not None:
        options.output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
