#!/usr/bin/env python3
"""Generate small-branch Phase-4 outcomes from the pinned native GLOP build."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "or-tools"
CASES = ROOT / "baselines/phase4-cases.json"
OUTPUT = ROOT / "baselines/phase4-native.json"
ADAPTER = ROOT / "target/native/phase4_reference_adapter"


def input_text(case: dict[str, object]) -> str:
    matrix = case["matrix"]
    variables = case["variables"]
    constraints = case["constraints"]
    assert isinstance(matrix, list)
    assert isinstance(variables, list)
    assert isinstance(constraints, list)
    lines = [
        f"{case['mode']} {len(constraints)} {len(variables)} {len(matrix)} {case['iterations']}"
    ]
    lines.extend(" ".join(map(str, row)) for row in matrix)
    lines.extend(" ".join(map(str, row)) for row in variables)
    lines.extend(" ".join(map(str, row)) for row in constraints)
    return "\n".join(lines) + "\n"


def main() -> None:
    case_bytes = CASES.read_bytes()
    specification = json.loads(case_bytes)
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=UPSTREAM,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if checkout_commit != specification["upstream_commit"]:
        raise RuntimeError(
            f"native checkout {checkout_commit} does not match pinned "
            f"{specification['upstream_commit']}"
        )
    results = []
    for case in specification["cases"]:
        output = subprocess.run(
            [ADAPTER],
            input=input_text(case),
            text=True,
            capture_output=True,
            check=True,
            timeout=20.0,
        ).stdout
        fields = {
            parts[0]: parts[1:]
            for line in output.splitlines()
            if (parts := line.split())
        }
        required = {
            "status", "iterations", "objective_bits", "basis", "value_bits",
            "reduced_bits", "primal_ray_bits", "dual_ray_bits",
        }
        if not required <= fields.keys():
            raise RuntimeError(f"{case['name']}: native adapter returned {output!r}")
        results.append(
            {
                "name": case["name"],
                "status": fields["status"][0],
                "iterations": int(fields["iterations"][0]),
                "objective_bits": int(fields["objective_bits"][0]),
                "basis": list(map(int, fields["basis"])),
                "value_bits": list(map(int, fields["value_bits"])),
                "reduced_bits": list(map(int, fields["reduced_bits"])),
                "primal_ray_bits": list(map(int, fields["primal_ray_bits"])),
                "dual_ray_bits": list(map(int, fields["dual_ray_bits"])),
            }
        )
        print(f"{case['name']}: {fields['status'][0]} {fields['iterations'][0]}")
    fixture = {
        "upstream_commit": specification["upstream_commit"],
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
