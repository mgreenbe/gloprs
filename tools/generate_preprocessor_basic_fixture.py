#!/usr/bin/env python3
"""Pin early/terminal preprocessor transforms and recovery against GLOP."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "baselines/preprocessor-basic-cases.json"
OUTPUT = ROOT / "baselines/preprocessor-basic-native.json"
ADAPTER = ROOT / "target/native/preprocessor_basic_reference_adapter"


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
        lines = [f"{case['mode']} {len(case['constraints'])} {len(case['variables'])} {len(case['matrix'])}"]
        lines.extend(" ".join(map(str, entry)) for entry in case["matrix"])
        lines.extend(" ".join(map(str, variable)) for variable in case["variables"])
        lines.extend(" ".join(map(str, constraint)) for constraint in case["constraints"])
        lines.append(case.get("post_status", "INIT"))
        lines.extend(" ".join(map(str, variable)) for variable in case["post_variables"])
        lines.extend(" ".join(map(str, constraint)) for constraint in case["post_constraints"])
        output = subprocess.run(
            [ADAPTER], input="\n".join(lines) + "\n", text=True,
            capture_output=True, check=True, timeout=20,
        ).stdout
        fields = {}
        for line in output.splitlines():
            parts = line.split()
            if parts:
                fields[parts[0]] = [int(item) if item.lstrip('-').isdigit() else item for item in parts[1:]]
        results.append({"name": case["name"], "trace": fields})
    OUTPUT.write_text(json.dumps({
        "upstream_commit": commit,
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "results": results,
    }, indent=2) + "\n")
    print(f"wrote {len(results)} native basic-preprocessor cases to {OUTPUT}")


if __name__ == "__main__":
    main()
