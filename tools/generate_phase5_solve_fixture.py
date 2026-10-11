#!/usr/bin/env python3
"""Capture native full-presolve solves and isolated capped shift/scale states."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "baselines/phase5-solve-cases.json"
OUTPUT = ROOT / "baselines/phase5-solve-native.json"
ADAPTER = ROOT / "target/native/phase5_solve_reference_adapter"


def main() -> None:
    case_bytes = CASES.read_bytes()
    specification = json.loads(case_bytes)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "or-tools",
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if commit != specification["upstream_commit"]:
        raise RuntimeError(f"native checkout {commit} is not pinned {specification['upstream_commit']}")
    results = []
    for case in specification["cases"]:
        shortcut = int(case.get("singleton_shortcut", False))
        initial_basis = case.get("initial_basis", "none")
        def run_at(iteration_limit: int) -> dict[str, list[str]]:
            lines = [f"{case['mode']} {initial_basis} {len(case['constraints'])} {len(case['variables'])} {len(case['matrix'])} {shortcut} {case.get('scaling_method', 0)} {iteration_limit}"]
            lines.extend(" ".join(map(str, row)) for row in case["matrix"])
            lines.extend(" ".join(map(str, row)) for row in case["variables"])
            lines.extend(" ".join(map(str, row)) for row in case["constraints"])
            output = subprocess.run(
                [ADAPTER], input="\n".join(lines) + "\n", text=True,
                capture_output=True, check=True, timeout=20,
            ).stdout
            return {parts[0]: parts[1:] for line in output.splitlines() if (parts := line.split())}

        fields = run_at(-1)
        iterations = int(fields["iterations"][0])
        capped = []
        for cap in range(iterations + 1):
            snapshot = run_at(cap)
            observed_iterations = int(snapshot["iterations"][0])
            if observed_iterations != cap:
                raise RuntimeError(f"{case['name']}: cap {cap} produced {observed_iterations} iterations")
            capped.append({
                "limit": cap,
                "status": snapshot["status"][0],
                "basis": [int(value) for value in snapshot["basis"]],
                "primal_bits": [int(value) for value in snapshot["primal"]],
                "dual_bits": [int(value) for value in snapshot["dual"]],
            })
        results.append({
            "name": case["name"],
            "status": fields["status"][0],
            "iterations": iterations,
            "basis": [int(value) for value in fields["basis"]],
            "capped_trajectory": capped,
            "primal_bits": [int(value) for value in fields["primal"]],
            "dual_bits": [int(value) for value in fields["dual"]],
            "variable_status": fields["variable_status"],
            "constraint_status": fields["constraint_status"],
        })
    OUTPUT.write_text(json.dumps({
        "upstream_commit": commit,
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "results": results,
    }, indent=2) + "\n")
    print(f"wrote {len(results)} native solves to {OUTPUT}")


if __name__ == "__main__":
    main()
