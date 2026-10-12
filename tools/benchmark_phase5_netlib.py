#!/usr/bin/env python3
"""Serial native/Rust public-solver timings with presolve on and off."""

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
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"
FAST = ROOT / "baselines/netlib-dual-trajectories.json.gz"
PHASE7 = ROOT / "baselines/phase7-netlib-report.json"
MANIFEST = ROOT.parent / "datasets/netlib/manifest.json"


def run(executable: Path, model: Path, mode: str) -> tuple[str, int, int, float]:
    command = [str(executable), str(model)]
    if mode == "raw":
        command.append("raw")
    output = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=25
    ).stdout.splitlines()
    status, iterations, objective_bits = output[0].split()[:3]
    elapsed = float(output[-1].split()[1])
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
    parser.add_argument("--mode", choices=("default", "raw", "both"), default="default")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.trials < 1 or options.warmups < 0:
        parser.error("trials must be positive and warmups nonnegative")
    with gzip.open(FAST, "rt", encoding="utf-8") as stream:
        fast_names = [case["name"] for case in json.load(stream)["results"]]
    phase7 = json.loads(PHASE7.read_text(encoding="utf-8"))
    expected = {case["name"]: case for case in phase7["results"]}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nonzeros = {case["name"]: case["nonzeros"] for case in manifest["problems"]}
    names = fast_names if options.models is None else options.models
    unknown = set(names) - set(fast_names)
    if unknown:
        parser.error(f"models outside the fast-96 fixture: {sorted(unknown)}")
    modes = ("raw", "default") if options.mode == "both" else (options.mode,)
    report = []
    for number, name in enumerate(names, start=1):
        model = MODELS / f"{name}.mps"
        results = {}
        for mode in modes:
            times: dict[str, list[float]] = {"native": [], "rust": []}
            outcomes: dict[str, tuple[str, int, int]] = {}
            for _ in range(options.warmups):
                run(NATIVE, model, mode)
                run(RUST, model, mode)
            for trial in range(options.trials):
                order = (("native", NATIVE), ("rust", RUST))
                if trial % 2:
                    order = tuple(reversed(order))
                for label, executable in order:
                    status, iterations, objective_bits, elapsed = run(executable, model, mode)
                    outcomes[label] = (status, iterations, objective_bits)
                    times[label].append(elapsed)
            if mode == "default":
                native, rust = outcomes["native"], outcomes["rust"]
                if native[:2] != rust[:2] or not math.isclose(
                    from_bits(native[2]), from_bits(rust[2]), rel_tol=1e-8, abs_tol=1e-8
                ):
                    raise AssertionError(f"{name} {mode}: {outcomes}")
            if mode == "default":
                reference = expected[name]
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
                "same_path_length": outcomes["native"][:2] == outcomes["rust"][:2],
            }
            results[mode]["ratio"] = (
                results[mode]["rust_seconds"] / results[mode]["native_seconds"]
            )
        result = {"name": name, "nonzeros": nonzeros[name], **results}
        report.append(result)
        print(f"[{number:02d}/{len(names)}] {name:<10} " + " ".join(
            f"{mode}={results[mode]['ratio']:.3f}x" for mode in modes
        ), flush=True)

    summary = {}
    for mode in modes:
        cases = [result[mode] for result in report]
        native_total = sum(case["native_seconds"] for case in cases)
        rust_total = sum(case["rust_seconds"] for case in cases)
        summary[mode] = {
            "native_total_seconds": native_total,
            "rust_total_seconds": rust_total,
            "aggregate_ratio": rust_total / native_total,
            "median_model_ratio": statistics.median(case["ratio"] for case in cases),
            "sqrt_nnz_weighted_geometric_ratio": math.exp(
                sum(math.sqrt(item["nonzeros"]) * math.log(item[mode]["ratio"])
                    for item in report)
                / sum(math.sqrt(item["nonzeros"]) for item in report)
            ),
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
        "modes": modes,
        "summary": summary,
        "results": report,
    }
    print(json.dumps(summary, indent=2))
    if options.output is not None:
        options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
