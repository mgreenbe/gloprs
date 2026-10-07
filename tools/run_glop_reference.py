#!/usr/bin/env python3
"""Run the pinned native GLOP solver and emit a normalized JSON result."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = PROJECT_ROOT / "target" / "native" / "glop_reference_adapter"
MPSOLVER_STATUS = {
    "OPTIMAL": "MPSOLVER_OPTIMAL",
    "PRIMAL_INFEASIBLE": "MPSOLVER_INFEASIBLE",
    "INFEASIBLE_OR_UNBOUNDED": "MPSOLVER_INFEASIBLE",
    "PRIMAL_UNBOUNDED": "MPSOLVER_UNBOUNDED",
    "INVALID_PROBLEM": "MPSOLVER_MODEL_INVALID",
    "ABNORMAL": "MPSOLVER_ABNORMAL",
    "IMPRECISE": "MPSOLVER_ABNORMAL",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mps", type=Path)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="omit primal, dual, and reduced-cost vectors",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    completed = subprocess.run(
        [str(args.adapter), str(args.mps.resolve())],
        check=False,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout) if completed.returncode == 0 else None
    wall_time = time.perf_counter() - start
    maximum_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_resident_set_bytes = (
        maximum_rss if platform.system() == "Darwin" else maximum_rss * 1024
    )

    result = {
        "input": str(args.mps),
        "exit_code": completed.returncode,
        "status": (
            MPSOLVER_STATUS.get(response["status"], response["status"])
            if response
            else None
        ),
        "model": response.get("model") if response else None,
        "objective": response.get("objective") if response else None,
        "best_bound": None,
        "iterations": response.get("iterations") if response else None,
        "solve_time_seconds": response.get("solve_time_seconds") if response else None,
        "wall_time_seconds": wall_time,
        "peak_resident_set_bytes": peak_resident_set_bytes,
        "deterministic_time": response.get("deterministic_time") if response else None,
        "maximum_primal_infeasibility": (
            response.get("maximum_primal_infeasibility") if response else None
        ),
        "maximum_dual_infeasibility": (
            response.get("maximum_dual_infeasibility") if response else None
        ),
        "basis": response.get("basis") if response else None,
    }
    if not args.summary:
        result.update(
            variable_values=response.get("variable_values") if response else None,
            dual_values=response.get("dual_values") if response else None,
            reduced_costs=response.get("reduced_costs") if response else None,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
