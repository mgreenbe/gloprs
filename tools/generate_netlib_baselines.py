#!/usr/bin/env python3
"""Run pinned native GLOP over the complete shared Netlib manifest."""

from __future__ import annotations

import json
import hashlib
import math
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "netlib"
RUNNER = PROJECT_ROOT / "tools" / "run_glop_reference.py"
OUTPUT = PROJECT_ROOT / "baselines" / "netlib-glop.json"
UPSTREAM_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def main() -> None:
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for number, problem in enumerate(manifest["problems"], start=1):
        name = problem["name"]
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--summary",
                str(DATASET_ROOT / problem["path"]),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = {
                "exit_code": completed.returncode,
                "status": None,
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
        result["name"] = name
        result["input"] = problem["path"]
        result["mps_sha256"] = problem["mps_sha256"]
        published = problem["known_optimal_objective"]
        objective = result.get("objective")
        result["published_objective"] = published
        result["published_objective_relative_error"] = (
            abs(objective - published) / max(1.0, abs(published))
            if published is not None
            and objective is not None
            and math.isfinite(objective)
            else None
        )
        results.append(result)
        print(f"[{number:02d}/{len(manifest['problems'])}] {name}: {result.get('status')}")

    baseline = {
        "solver": "Google OR-Tools GLOP",
        "upstream_commit": UPSTREAM_COMMIT,
        "generated": date.today().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset_manifest_sha256": hashlib.sha256(
            (DATASET_ROOT / "manifest.json").read_bytes()
        ).hexdigest(),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    failures = [result for result in results if result.get("exit_code") != 0]
    print(f"wrote {len(results)} results to {OUTPUT}; {len(failures)} failed to parse or solve")


if __name__ == "__main__":
    main()
