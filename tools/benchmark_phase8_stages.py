#!/usr/bin/env python3
"""Serial native/Rust stage timings on the fast Netlib corpus.

The parse timer includes conversion to LinearProgram. The presolve timer starts
from an already-parsed model and includes the upstream-equivalent model copy,
near-zero cleanup, and MainLpPreprocessor run, but not parsing or output. The
solve timer starts from a matched presolved model and excludes both preceding
stages and postsolve.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import statistics
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"
MANIFEST = ROOT.parent / "datasets/netlib/manifest.json"
FAST = ROOT / "baselines/netlib-dual-trajectories.json.gz"
NATIVE = ROOT / "target/native/full_presolve_netlib_reference_adapter"
RUST = ROOT / "target/release/examples/full_presolve_netlib"


def version(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def run(executable: Path, model: Path, stage: str, repetitions: int) -> dict:
    mode = "presolved_solve_timing" if stage == "solve" else f"{stage}_timing"
    command = [executable, model, mode]
    if stage != "parse":
        command.append(str(repetitions))
    process = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    )
    fields = process.stdout.split()
    result = dict(zip(fields[::2], fields[1::2], strict=True))
    assert len(result) == 4 + (stage == "presolve"), process.stdout
    seconds = float(result.pop(f"{stage}_seconds"))
    return {"seconds": seconds, **result}


def same_outcome(native: dict, rust: dict, stage: str) -> bool:
    if stage != "solve":
        return native == rust
    if native["status"] != rust["status"] or native["iterations"] != rust["iterations"]:
        return False
    def value(bits: str) -> float:
        return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]
    return math.isclose(value(native["objective_bits"]),
                        value(rust["objective_bits"]), rel_tol=1e-8, abs_tol=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("parse", "presolve", "solve", "both", "all"),
                        default="all")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--presolve-repetitions", type=int, default=3)
    parser.add_argument("--solve-repetitions", type=int, default=1)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if (options.trials < 1 or options.warmups < 0
            or options.presolve_repetitions < 1 or options.solve_repetitions < 1):
        parser.error("invalid trial, warmup, or repetition count")
    with gzip.open(FAST, "rt", encoding="utf-8") as stream:
        fast_names = [case["name"] for case in json.load(stream)["results"]]
    names = fast_names if options.models is None else options.models
    unknown = set(names) - set(fast_names)
    if unknown:
        parser.error(f"models outside the fast-96 fixture: {sorted(unknown)}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nonzeros = {case["name"]: case["nonzeros"] for case in manifest["problems"]}
    stages = {"both": ("parse", "presolve"),
              "all": ("parse", "presolve", "solve")}.get(options.stage, (options.stage,))
    report: list[dict] = []
    for number, name in enumerate(names, start=1):
        model = MODELS / f"{name}.mps"
        result: dict = {"name": name, "nonzeros": nonzeros[name]}
        for stage in stages:
            repetitions = (options.presolve_repetitions if stage == "presolve"
                           else options.solve_repetitions)
            for _ in range(options.warmups):
                run(NATIVE, model, stage, repetitions)
                run(RUST, model, stage, repetitions)
            times: dict[str, list[float]] = {"native": [], "rust": []}
            for trial in range(options.trials):
                order = (("native", NATIVE), ("rust", RUST))
                if trial % 2:
                    order = tuple(reversed(order))
                outcomes = {}
                for label, executable in order:
                    measured = run(executable, model, stage, repetitions)
                    times[label].append(measured.pop("seconds"))
                    outcomes[label] = measured
                if not same_outcome(outcomes["native"], outcomes["rust"], stage):
                    raise RuntimeError(f"{name} {stage}: {outcomes}")
            native = statistics.median(times["native"])
            rust = statistics.median(times["rust"])
            result[stage] = {
                "native_seconds": native,
                "rust_seconds": rust,
                "ratio": rust / native,
                "native_trials_seconds": times["native"],
                "rust_trials_seconds": times["rust"],
                "native_median_absolute_deviation_seconds": statistics.median(
                    abs(value - native) for value in times["native"]
                ),
                "rust_median_absolute_deviation_seconds": statistics.median(
                    abs(value - rust) for value in times["rust"]
                ),
                "outcome": outcomes["native"],
            }
        report.append(result)
        print(f"[{number:02d}/{len(names)}] {name:<10} " + " ".join(
            f"{stage}={result[stage]['ratio']:.3f}x" for stage in stages
        ), flush=True)
    summary = {}
    for stage in stages:
        native = sum(item[stage]["native_seconds"] for item in report)
        rust = sum(item[stage]["rust_seconds"] for item in report)
        summary[stage] = {
            "native_total_seconds": native,
            "rust_total_seconds": rust,
            "aggregate_ratio": rust / native,
            "median_model_ratio": statistics.median(item[stage]["ratio"] for item in report),
            "sqrt_nnz_weighted_geometric_ratio": math.exp(
                sum(math.sqrt(item["nonzeros"]) * math.log(item[stage]["ratio"])
                    for item in report)
                / sum(math.sqrt(item["nonzeros"]) for item in report)
            ),
        }
    output = {
        "machine": platform.platform(),
        "processor": version(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "rustc": version(["rustc", "--version"]),
        "clang": version(["clang++", "--version"]).splitlines()[0],
        "gloprs_commit": version(["git", "rev-parse", "HEAD"]),
        "upstream_commit": version(["git", "-C", str(ROOT.parent / "or-tools"),
                                    "rev-parse", "HEAD"]),
        "trials": options.trials,
        "warmups": options.warmups,
        "presolve_repetitions": options.presolve_repetitions,
        "solve_repetitions": options.solve_repetitions,
        "summary": summary,
        "results": report,
    }
    print(json.dumps(summary, indent=2))
    if options.output is not None:
        options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
