#!/usr/bin/env python3
"""Serial, reproducible public-solver differential audit of all Netlib LPs.

The 96 fast native traces and qap15's completed native trace are pinned
fixtures. qap12 is run live on both implementations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "datasets/netlib"
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"
PRESOLVE = ROOT / "target/release/examples/full_presolve_netlib"
NATIVE_PRESOLVE = ROOT / "target/native/full_presolve_netlib_reference_adapter"
CLI = ROOT / "target/release/gloprs"
BASELINE = ROOT / "baselines/phase5-netlib-solution-native.json.gz"
QAP15_BASELINE = ROOT / "baselines/phase7-qap15-terminal-native.json.gz"
PRESOLVE_BASELINE = ROOT / "baselines/main-preprocessor-netlib-native.json"
MODEL_BASELINE = ROOT / "baselines/netlib-glop.json"
REPORT = ROOT / "target/netlib-audit.json"
PIN = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def run(executable: Path, model: Path, timeout: float,
        mode: str | None = None,
        wall_seconds: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), str(model), *([mode] if mode else []),
         *([str(wall_seconds)] if wall_seconds is not None else [])],
        capture_output=True, text=True,
        check=False, timeout=timeout,
    )


def number(bits: str) -> float:
    return struct.unpack("=d", struct.pack("=Q", int(bits)))[0]


def close(left: float, right: float) -> bool:
    return left == right or math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def compare(native: list[str], rust: list[str]) -> tuple[list[str], dict[str, object]]:
    issues: list[str] = []
    if len(native) < 8 or len(rust) < 8:
        return ["trace_shape"], {}
    n = [line.split() for line in native[:8]]
    r = [line.split() for line in rust[:8]]
    metrics: dict[str, object] = {
        "native_status": n[0][0], "rust_status": r[0][0],
        "native_iterations": int(n[0][1]), "rust_iterations": int(r[0][1]),
        "objective": number(r[0][2]),
        "maximum_primal_infeasibility": number(r[0][3]),
        "maximum_dual_infeasibility": number(r[0][4]),
    }
    if n[0][0] != r[0][0]:
        issues.append("incorrect_status")
    if n[0][1] != r[0][1]:
        issues.append("pivot_path_or_termination")
    if n[1][:2] != r[1][:2]:
        issues.append("model_dimensions")
    for index, label in enumerate(("objective", "primal_infeasibility", "dual_infeasibility"), 2):
        if not close(number(n[0][index]), number(r[0][index])):
            issues.append(label)
    for index, label in ((6, "variable_basis"), (7, "constraint_basis")):
        if n[index] != r[index]:
            issues.append(label)
    exact_bits = 0
    checked = 0
    worst = 0.0
    for index, label in ((2, "primal"), (3, "reduced_costs"), (4, "dual"), (5, "activities")):
        if n[index][0] != r[index][0] or len(n[index]) != len(r[index]):
            issues.append(f"{label}_shape")
            continue
        for left, right in zip(n[index][1:], r[index][1:]):
            checked += 1
            if left == right:
                exact_bits += 1
                continue
            expected, actual = number(left), number(right)
            difference = abs(expected - actual) / max(1.0, abs(expected), abs(actual))
            worst = max(worst, difference)
            if not close(expected, actual):
                issues.append(label)
    metrics.update(exact_vector_bits=exact_bits, checked_vector_bits=checked,
                   worst_relative_vector_error=worst)
    return sorted(set(issues)), metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall-timeout", type=float, default=25.0)
    parser.add_argument("--qap15-wall-seconds", type=float, default=180.0)
    parser.add_argument("--skip-presolve", action="store_true")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    names: list[str] = json.loads((DATASET / "subsets/all.json").read_text())
    if args.models is not None:
        unknown = set(args.models) - set(names)
        if unknown:
            parser.error(f"unknown models: {sorted(unknown)}")
        names = [name for name in names if name in set(args.models)]
    fixture = json.loads(gzip.decompress(BASELINE.read_bytes()))
    if fixture["upstream_commit"] != PIN:
        raise ValueError("native fixture does not match pinned OR-Tools commit")
    native_cases = {case["name"]: case["lines"] for case in fixture["results"]}
    qap15_fixture = json.loads(gzip.decompress(QAP15_BASELINE.read_bytes()))
    if qap15_fixture["upstream_commit"] != PIN:
        raise ValueError("qap15 native fixture does not match pinned OR-Tools commit")
    presolve_fixture = json.loads(PRESOLVE_BASELINE.read_text())
    if presolve_fixture["upstream_commit"] != PIN:
        raise ValueError("presolve fixture does not match pinned OR-Tools commit")
    presolve_cases = {case["name"]: case for case in presolve_fixture["results"]}
    native_model_cases = {case["name"]: case for case in json.loads(MODEL_BASELINE.read_text())["results"]}
    manifest = json.loads((DATASET / "manifest.json").read_text())
    provenance = {case["name"]: case for case in manifest["problems"]}
    results = []
    for name in names:
        model = DATASET / "mps" / f"{name}.mps"
        case: dict[str, object] = {"name": name, "mps_sha256": provenance[name]["mps_sha256"]}
        if hashlib.sha256(model.read_bytes()).hexdigest() != case["mps_sha256"]:
            raise ValueError(f"{name}: dataset checksum changed")
        issues: list[str] = []
        inspection = subprocess.run(
            [str(CLI), "inspect", "--tsv", str(model)],
            capture_output=True, text=True, check=False, timeout=args.wall_timeout,
        )
        if inspection.returncode:
            issues.append("model_parse")
        else:
            fields = inspection.stdout.rstrip("\n").split("\t")
            stats = dict(rows=int(fields[1]), columns=int(fields[2]),
                         nonzeros=int(fields[3]), data_fingerprint=fields[4])
            case["model"] = stats
            if stats != native_model_cases[name]["model"]:
                issues.append("parser_model_mismatch")
        if name in presolve_cases and not args.skip_presolve:
            try:
                process = run(PRESOLVE, model, args.wall_timeout)
                digest = hashlib.sha256(process.stderr.encode()).hexdigest()
                case["presolved_sha256"] = digest
                if process.returncode != 0 or digest != presolve_cases[name]["presolved_sha256"]:
                    issues.append("presolve_model")
            except subprocess.TimeoutExpired:
                issues.append("presolve_timeout")
        elif name in {"qap12", "qap15"} and not args.skip_presolve:
            try:
                native_presolve = run(NATIVE_PRESOLVE, model, args.wall_timeout, "presolve_only")
                rust_presolve = run(PRESOLVE, model, args.wall_timeout, "presolve_only")
                if native_presolve.returncode or rust_presolve.returncode:
                    issues.append("presolve_error")
                else:
                    native_hash = hashlib.sha256(native_presolve.stderr.encode()).hexdigest()
                    rust_hash = hashlib.sha256(rust_presolve.stderr.encode()).hexdigest()
                    case["presolved_sha256"] = rust_hash
                    if native_hash != rust_hash:
                        issues.append("presolve_model")
            except subprocess.TimeoutExpired:
                issues.append("presolve_timeout")
            if name == "qap15":
                try:
                    prefixes = []
                    for executable in (NATIVE_PRESOLVE, PRESOLVE):
                        result = subprocess.run(
                            [executable, model, "5000", "fingerprint"],
                            capture_output=True, text=True, check=True,
                            timeout=args.wall_timeout,
                        )
                        prefixes.append(result.stdout.strip())
                    case["deterministic_prefix_iterations"] = 5000
                    case["deterministic_prefix_fingerprint"] = prefixes[1]
                    if prefixes[0] != prefixes[1]:
                        issues.append("deterministic_prefix")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    issues.append("deterministic_prefix_error")
        native = native_cases.get(name)
        native_refactorizations: int | None = None
        if name == "qap15":
            if qap15_fixture["model_sha256"] != case["mps_sha256"]:
                raise ValueError("qap15 terminal fixture has the wrong model checksum")
            native = qap15_fixture["lines"][:8]
            native_refactorizations = int(qap15_fixture["lines"][10].split()[1])
            case["native_refactorizations"] = native_refactorizations
        else:
            try:
                process = run(NATIVE, model, args.wall_timeout, "phase7")
                if process.returncode == 0:
                    live_native = process.stdout.splitlines()
                    if native is not None and live_native[:8] != native:
                        issues.append("native_fixture_drift")
                    if native is None:
                        native = live_native[:8]
                    if len(live_native) >= 11 and live_native[10].startswith("refactorizations "):
                        native_refactorizations = int(live_native[10].split()[1])
                        case["native_refactorizations"] = native_refactorizations
                    else:
                        issues.append("native_refactorization_trace_shape")
                else:
                    issues.append("native_error")
            except subprocess.TimeoutExpired:
                issues.append("native_timeout")
        try:
            rust_wall = args.qap15_wall_seconds if name == "qap15" else None
            timeout = max(args.wall_timeout, rust_wall + 30.0) if rust_wall else args.wall_timeout
            process = run(RUST, model, timeout, "phase7", rust_wall)
            rust = process.stdout.splitlines()
            if process.returncode != 0:
                issues.append("rust_error")
            elif native is not None:
                if len(rust) < 12 or not rust[10].startswith("independent_primal_violation "):
                    issues.append("independent_primal_trace_shape")
                else:
                    violation = float(rust[10].split()[1])
                    case["independent_primal_violation"] = violation
                    if not math.isfinite(violation) or violation > 1e-6:
                        issues.append("invalid_primal_solution")
                if len(rust) >= 12 and rust[11].startswith("refactorizations "):
                    rust_refactorizations = int(rust[11].split()[1])
                    case["rust_refactorizations"] = rust_refactorizations
                else:
                    issues.append("rust_refactorization_trace_shape")
                    rust_refactorizations = None
                native_status = native[0].split()[0]
                rust_status = rust[0].split()[0]
                if native_status == "PRIMAL_FEASIBLE" or rust_status == "PRIMAL_FEASIBLE":
                    issues.append("time_limited_incomparable")
                else:
                    differences, metrics = compare(native, rust)
                    issues.extend(differences)
                    case.update(metrics)
                    if (native_refactorizations is not None
                            and rust_refactorizations is not None
                            and native_refactorizations != rust_refactorizations):
                        issues.append("refactorization_count")
                    if len(rust) < 10 or not rust[9].startswith("deterministic "):
                        issues.append("deterministic_trace_shape")
                    else:
                        actual = number(rust[9].split()[1])
                        expected = (number(qap15_fixture["lines"][9].split()[1])
                                    if name == "qap15" else native_model_cases[name]["deterministic_time"])
                        case.update(native_deterministic_time=expected,
                                    rust_deterministic_time=actual)
                        if not close(expected, actual):
                            issues.append("deterministic_time")
            elif rust:
                fields = rust[0].split()
                case.update(rust_status=fields[0], rust_iterations=int(fields[1]))
        except subprocess.TimeoutExpired:
            issues.append("rust_timeout")
        case["classifications"] = sorted(set(issues)) or ["match"]
        results.append(case)
        print(f"{name}: {','.join(case['classifications'])}", flush=True)
    report = {
        "upstream_commit": PIN,
        "native_fixture": str(BASELINE.relative_to(ROOT)),
        "qap15_native_fixture": str(QAP15_BASELINE.relative_to(ROOT)),
        "parameters": "pinned defaults; 20-second internal limit except qap15",
        "qap15_internal_wall_seconds": args.qap15_wall_seconds,
        "wall_timeout_seconds": args.wall_timeout,
        "presolve_checked": not args.skip_presolve,
        "models_requested": len(names),
        "summary": {
            "classifications": dict(sorted(Counter(
                label for case in results for label in case["classifications"]
            ).items())),
            "presolve_models_checked": sum("presolved_sha256" in case for case in results),
            "exact_vector_bits": sum(int(case.get("exact_vector_bits", 0)) for case in results),
            "checked_vector_bits": sum(int(case.get("checked_vector_bits", 0)) for case in results),
        },
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report: {args.report}")
    return 0 if all(case["classifications"] == ["match"] for case in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
