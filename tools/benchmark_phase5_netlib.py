#!/usr/bin/env python3
"""Serial native/Rust public-solver timings with presolve on and off."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import struct
import subprocess
from pathlib import Path

from lp_corpus import NMM_CORPORA, models as corpus_models


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"
PHASE7 = ROOT / "baselines/phase7-netlib-report.json"


def run(executable: Path, model: Path, mode: str,
        wall_seconds: float = 20.0) -> tuple[str, int, int, float]:
    command = [str(executable), str(model)]
    if mode == "raw":
        command.append("raw")
    elif wall_seconds != 20.0:
        command.extend(("phase7", str(wall_seconds)))
    output = subprocess.run(
        command, check=True, capture_output=True, text=True,
        timeout=wall_seconds + 30
    ).stdout.splitlines()
    status, iterations, objective_bits = output[0].split()[:3]
    elapsed = float(next(line.split()[1] for line in output if line.startswith("elapsed ")))
    return status, int(iterations), int(objective_bits), elapsed


def version(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--corpus", choices=("netlib-fast96", "miplib2017",
                                      "mittelmann-lpopt", "extra", *NMM_CORPORA),
                        default="netlib-fast96")
    parser.add_argument("--mode", choices=("default", "raw", "both"), default="default")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.trials < 1 or options.warmups < 0:
        parser.error("trials must be positive and warmups nonnegative")
    if not 0 < options.wall_seconds <= 3600:
        parser.error("wall-seconds must be positive and at most 3600")
    if options.wall_seconds != 20.0 and options.mode != "default":
        parser.error("nondefault wall limit is only supported for default pipeline")
    phase7 = json.loads(PHASE7.read_text(encoding="utf-8"))
    expected = {case["name"]: case for case in phase7["results"]}
    selected = corpus_models(options.corpus)
    by_key = {model.key: model for model in selected}
    by_name = {model.name: model for model in selected
               if sum(other.name == model.name for other in selected) == 1}
    requested = options.models or list(by_key)
    unknown = set(requested) - set(by_key) - set(by_name)
    if unknown:
        parser.error(f"models outside {options.corpus}: {sorted(unknown)}")
    cases = [by_key[name] if name in by_key else by_name[name] for name in requested]
    modes = ("raw", "default") if options.mode == "both" else (options.mode,)
    report = []
    for number, case in enumerate(cases, start=1):
        name, model = case.key, case.path
        if not model.is_file():
            raise FileNotFoundError(model)
        results = {}
        for mode in modes:
            times: dict[str, list[float]] = {"native": [], "rust": []}
            trial_outcomes: list[dict[str, tuple[str, int, int]]] = []
            limited_trials: list[bool] = []
            for _ in range(options.warmups):
                run(NATIVE, model, mode, options.wall_seconds)
                run(RUST, model, mode, options.wall_seconds)
            for trial in range(options.trials):
                order = (("native", NATIVE), ("rust", RUST))
                if trial % 2:
                    order = tuple(reversed(order))
                outcomes: dict[str, tuple[str, int, int]] = {}
                trial_times = {}
                for label, executable in order:
                    status, iterations, objective_bits, elapsed = run(
                        executable, model, mode, options.wall_seconds)
                    outcomes[label] = (status, iterations, objective_bits)
                    times[label].append(elapsed)
                    trial_times[label] = elapsed
                limited = (options.corpus != "netlib-fast96"
                           and max(trial_times.values()) >= 0.95 * options.wall_seconds
                           and (outcomes["native"][0] != "OPTIMAL"
                                or outcomes["rust"][0] != "OPTIMAL"))
                if mode == "default" and not limited:
                    native, rust = outcomes["native"], outcomes["rust"]
                    if native[:2] != rust[:2] or not math.isclose(
                        from_bits(native[2]), from_bits(rust[2]), rel_tol=1e-8, abs_tol=1e-8
                    ):
                        raise AssertionError(f"{name} {mode} trial {trial}: {outcomes}")
                trial_outcomes.append(outcomes)
                limited_trials.append(limited)
                if limited:
                    break
            outcomes = trial_outcomes[-1]
            wall_limited = any(limited_trials)
            if mode == "default" and case.corpus == "netlib":
                reference = expected[case.name]
                assert outcomes["native"][:2] == (
                    reference["native_status"], reference["native_iterations"]
                ), (name, outcomes, reference)
            results[mode] = {
                "native_seconds": statistics.median(times["native"]),
                "rust_seconds": statistics.median(times["rust"]),
                "native_trials_seconds": times["native"],
                "rust_trials_seconds": times["rust"],
                "native_status": outcomes["native"][0],
                "rust_status": outcomes["rust"][0],
                "native_iterations": outcomes["native"][1],
                "rust_iterations": outcomes["rust"][1],
                "native_objective_bits": outcomes["native"][2],
                "rust_objective_bits": outcomes["rust"][2],
                "trial_outcomes": trial_outcomes,
                "wall_limited_trials": limited_trials,
                "same_path_length": outcomes["native"][:2] == outcomes["rust"][:2],
                "wall_limited": wall_limited,
            }
            results[mode]["ratio"] = (
                results[mode]["rust_seconds"] / results[mode]["native_seconds"]
            )
        result = {"name": case.name, "corpus": case.corpus,
                  "nonzeros": case.nonzeros, **results}
        report.append(result)
        print(f"[{number:02d}/{len(cases)}] {name:<10} " + " ".join(
            f"{mode}=" + ("wall-limited" if results[mode]["wall_limited"]
                        else f"{results[mode]['ratio']:.3f}x") for mode in modes
        ), flush=True)

    summary = {}
    for mode in modes:
        comparable = [result for result in report if not result[mode]["wall_limited"]]
        cases = [result[mode] for result in comparable]
        native_total = sum(case["native_seconds"] for case in cases)
        rust_total = sum(case["rust_seconds"] for case in cases)
        summary[mode] = {
            "comparable_models": len(comparable),
            "wall_limited_models": len(report) - len(comparable),
            "native_total_seconds": native_total,
            "rust_total_seconds": rust_total,
            "aggregate_ratio": rust_total / native_total if native_total else None,
            "median_model_ratio": (statistics.median(case["ratio"] for case in cases)
                                   if cases else None),
            "sqrt_nnz_weighted_geometric_ratio": math.exp(
                sum(math.sqrt(item["nonzeros"]) * math.log(item[mode]["ratio"])
                    for item in comparable)
                / sum(math.sqrt(item["nonzeros"]) for item in comparable)
            ) if comparable else None,
            "same_path_length_count": sum(case["same_path_length"] for case in cases),
        }
    output = {
        "machine": platform.platform(),
        "processor": version(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "rustc": version(["rustc", "--version"]),
        "clang": version(["clang++", "--version"]).splitlines()[0],
        "gloprs_commit": version(["git", "rev-parse", "HEAD"]),
        "upstream_commit": phase7["upstream_commit"],
        "timing_scope": "serial public LPSolver solve; MPS parsing and output excluded",
        "trials": options.trials,
        "warmups": options.warmups,
        "wall_seconds": options.wall_seconds,
        "modes": modes,
        "corpus": options.corpus,
        "summary": summary,
        "results": report,
    }
    print(json.dumps(summary, indent=2))
    if options.output is not None:
        options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
