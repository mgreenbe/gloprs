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
import json
import math
import platform
import statistics
import struct
import subprocess
from pathlib import Path

from lp_corpus import NMM_CORPORA, models as corpus_models


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/full_presolve_netlib_reference_adapter"
RUST = ROOT / "target/release/examples/full_presolve_netlib"


def version(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def run(executable: Path, model: Path, stage: str, repetitions: int,
        wall_seconds: float = 20.0) -> dict:
    mode = "presolved_solve_timing" if stage == "solve" else f"{stage}_timing"
    command = [executable, model, mode]
    if stage != "parse":
        command.append(str(repetitions))
    if stage == "solve":
        command.append(str(wall_seconds))
    process = subprocess.run(
        command, check=True, capture_output=True, text=True,
        timeout=max(60, wall_seconds * repetitions + 30)
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
    parser.add_argument("--wall-seconds", type=float, default=20.0,
                        help="per-solve wall limit for matched-presolved solves")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--corpus", choices=("netlib-fast96", "miplib2017",
                                      "mittelmann-lpopt", "extra", *NMM_CORPORA),
                        default="netlib-fast96")
    parser.add_argument("--rust", type=Path, default=RUST,
                        help="release Rust adapter to compare with native GLOP")
    parser.add_argument("--native", type=Path, default=NATIVE,
                        help="reference adapter (also permits a saved Rust A/B binary)")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if (options.trials < 1 or options.warmups < 0
            or options.presolve_repetitions < 1 or options.solve_repetitions < 1
            or not 0 < options.wall_seconds <= 3600):
        parser.error("invalid trial, warmup, or repetition count")
    selected = corpus_models(options.corpus)
    by_key = {model.key: model for model in selected}
    by_name = {model.name: model for model in selected
               if sum(other.name == model.name for other in selected) == 1}
    requested = options.models or list(by_key)
    unknown = set(requested) - set(by_key) - set(by_name)
    if unknown:
        parser.error(f"models outside {options.corpus}: {sorted(unknown)}")
    cases = [by_key[name] if name in by_key else by_name[name] for name in requested]
    stages = {"both": ("parse", "presolve"),
              "all": ("parse", "presolve", "solve")}.get(options.stage, (options.stage,))
    report: list[dict] = []
    for number, case in enumerate(cases, start=1):
        name, model = case.key, case.path
        if not model.is_file():
            raise FileNotFoundError(model)
        result: dict = {"name": case.name, "corpus": case.corpus,
                        "nonzeros": case.nonzeros}
        for stage in stages:
            if stage == "solve":
                presolve_native = run(options.native, model, "presolve", 1)
                presolve_rust = run(options.rust, model, "presolve", 1)
                presolve_native.pop("seconds")
                presolve_rust.pop("seconds")
                if presolve_native != presolve_rust:
                    raise RuntimeError(f"{name} presolve: {presolve_native} != {presolve_rust}")
                if presolve_native["status"] != "INIT":
                    result[stage] = {"skipped": "presolve terminated",
                                     "outcome": presolve_native}
                    continue
            repetitions = (options.presolve_repetitions if stage == "presolve"
                           else options.solve_repetitions)
            for _ in range(options.warmups):
                run(options.native, model, stage, repetitions, options.wall_seconds)
                run(options.rust, model, stage, repetitions, options.wall_seconds)
            times: dict[str, list[float]] = {"native": [], "rust": []}
            for trial in range(options.trials):
                order = (("native", options.native), ("rust", options.rust))
                if trial % 2:
                    order = tuple(reversed(order))
                outcomes = {}
                for label, executable in order:
                    measured = run(executable, model, stage, repetitions,
                                   options.wall_seconds)
                    times[label].append(measured.pop("seconds"))
                    outcomes[label] = measured
                if (stage == "solve" and options.corpus != "netlib-fast96"
                        and max(times["native"][-1], times["rust"][-1])
                        >= 0.95 * options.wall_seconds
                        and (outcomes["native"]["status"] != "OPTIMAL"
                             or outcomes["rust"]["status"] != "OPTIMAL")):
                    result[stage] = {"skipped": "wall-limited, inconclusive",
                                     "outcomes": outcomes}
                    break
                if not same_outcome(outcomes["native"], outcomes["rust"], stage):
                    raise RuntimeError(f"{name} {stage}: {outcomes}")
            if "skipped" in result.get(stage, {}):
                continue
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
        print(f"[{number:02d}/{len(cases)}] {name:<10} " + " ".join(
            f"{stage}=" + ("skipped" if "skipped" in result[stage]
                          else f"{result[stage]['ratio']:.3f}x") for stage in stages
        ), flush=True)
    summary = {}
    for stage in stages:
        timed = [item for item in report if "skipped" not in item[stage]]
        native = sum(item[stage]["native_seconds"] for item in timed)
        rust = sum(item[stage]["rust_seconds"] for item in timed)
        summary[stage] = {
            "timed_models": len(timed),
            "skipped_models": len(report) - len(timed),
            "native_total_seconds": native,
            "rust_total_seconds": rust,
            "aggregate_ratio": rust / native if native else None,
            "median_model_ratio": (statistics.median(item[stage]["ratio"] for item in timed)
                                   if timed else None),
            "sqrt_nnz_weighted_geometric_ratio": math.exp(
                sum(math.sqrt(item["nonzeros"]) * math.log(item[stage]["ratio"])
                    for item in timed)
                / sum(math.sqrt(item["nonzeros"]) for item in timed)
            ) if timed else None,
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
        "wall_seconds": options.wall_seconds,
        "corpus": options.corpus,
        "summary": summary,
        "results": report,
    }
    print(json.dumps(summary, indent=2))
    if options.output is not None:
        options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
