#!/usr/bin/env python3
"""Pin bound-shift/scaling traces from the selected upstream GLOP commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "baselines/preprocessor-cases.json"
OUTPUT = ROOT / "baselines/preprocessor-native.json"
ADAPTER = ROOT / "target/native/preprocessor_reference_adapter"


def main() -> None:
    case_bytes = CASES.read_bytes()
    specification = json.loads(case_bytes)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "or-tools",
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != specification["upstream_commit"]:
        raise RuntimeError(f"expected {specification['upstream_commit']}, got {commit}")

    results = []
    for case in specification["cases"]:
        rows = len(case["constraints"])
        columns = len(case["variables"])
        lines = [f"{case['mode']} {rows} {columns} {len(case['matrix'])} {case['cost_scaling']} {case.get('scaling_method', 0)}"]
        lines.extend(" ".join(map(str, entry)) for entry in case["matrix"])
        lines.extend(" ".join(map(str, variable[:3])) for variable in case["variables"])
        lines.extend(" ".join(map(str, constraint[:2])) for constraint in case["constraints"])
        lines.extend(" ".join(map(str, variable[3:])) for variable in case["variables"])
        lines.extend(str(constraint[2]) for constraint in case["constraints"])
        output = subprocess.run(
            [ADAPTER], input="\n".join(lines) + "\n", text=True,
            capture_output=True, check=True, timeout=20,
        ).stdout
        fields = {
            parts[0]: [int(item) for item in parts[1:]]
            for line in output.splitlines() if (parts := line.split())
        }
        results.append({"name": case["name"], "trace": fields})

    OUTPUT.write_text(json.dumps({
        "upstream_commit": commit,
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "results": results,
    }, indent=2) + "\n")
    print(f"wrote {len(results)} native cases to {OUTPUT}")


if __name__ == "__main__":
    main()
