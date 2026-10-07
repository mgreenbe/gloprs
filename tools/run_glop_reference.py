#!/usr/bin/env python3
"""Run the pinned native GLOP solver and emit a normalized JSON result."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVER = PROJECT_ROOT.parent / "or-tools" / "build-gloprs-solve" / "bin" / "solve"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def last_match(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None


def optional_float(pattern: str, text: str) -> float | None:
    value = last_match(pattern, text)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mps", type=Path)
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="gloprs-reference-") as temporary:
        response_stem = Path(temporary) / "response"
        completed = subprocess.run(
            [
                str(args.solver),
                f"--input={args.mps.resolve()}",
                "--solver=glop",
                f"--dump_response={response_stem}",
                "--dump_format=json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        log = completed.stdout + completed.stderr
        response_path = response_stem.with_suffix(".json")
        response = (
            json.loads(response_path.read_text(encoding="utf-8"))
            if response_path.exists()
            else None
        )

    result = {
        "input": str(args.mps),
        "exit_code": completed.returncode,
        "status": response.get("status") if response else None,
        "objective": response.get("objective_value") if response else None,
        "best_bound": response.get("best_objective_bound") if response else None,
        "iterations": (
            int(value) if (value := last_match(r"^iterations:\s*(\d+)\s*$", log)) else None
        ),
        "solve_time_seconds": optional_float(rf"^time:\s*({FLOAT})\s*$", log),
        "deterministic_time": optional_float(
            rf"^deterministic_time:\s*({FLOAT})\s*$", log
        ),
        "maximum_primal_infeasibility": optional_float(
            rf"^Max\. primal infeasibility = ({FLOAT})\s*$", log
        ),
        "maximum_dual_infeasibility": optional_float(
            rf"^Max\. dual infeasibility = ({FLOAT})\s*$", log
        ),
        "variable_values": response.get("variable_value") if response else None,
        "dual_values": response.get("dual_value") if response else None,
        "reduced_costs": response.get("reduced_cost") if response else None,
        "basis": None,
        "basis_note": "OR-Tools solve does not expose GLOP basis statuses in its response",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
