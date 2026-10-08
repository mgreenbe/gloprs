#!/usr/bin/env python3
"""Validate gloprs solve results against the pinned native GLOP baseline."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "netlib"
DEFAULT_BASELINE = PROJECT_ROOT / "baselines" / "netlib-glop.json"
DEFAULT_BINARY = PROJECT_ROOT / "target" / "release" / "gloprs"
RESULT = re.compile(
    r"status=(?P<status>\S+) objective=(?P<objective>\S+) "
    r"iterations=(?P<iterations>\d+) "
    r"primal_infeasibility=(?P<primal>\S+) "
    r"dual_infeasibility=(?P<dual>\S+)"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="smallest-10")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--objective-tolerance", type=float, default=1e-7)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="maximum wall-clock seconds per model",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    names = json.loads(
        (DATASET_ROOT / "subsets" / f"{args.subset}.json").read_text()
    )
    native = {
        result["name"]: result
        for result in json.loads(args.baseline.read_text())["results"]
    }
    failures: list[str] = []
    print(
        "model\tstatus\tobjective_error\titerations\tnative_iterations\tprimal\tdual",
        flush=True,
    )
    for name in names:
        try:
            completed = subprocess.run(
                [str(args.binary), "solve", str(DATASET_ROOT / "mps" / f"{name}.mps")],
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{name}: timed out after {args.timeout:g} seconds")
            print(f"{name}\tTIMEOUT\t-\t-\t{native[name]['iterations']}\t-\t-", flush=True)
            continue
        match = RESULT.fullmatch(completed.stdout.strip())
        if completed.returncode != 0 or match is None:
            failures.append(f"{name}: solver failed: {completed.stderr.strip()}")
            continue
        fields = match.groupdict()
        reference = native[name]
        status = fields["status"]
        expected_status = reference["status"].removeprefix("MPSOLVER_")
        objective = float(fields["objective"])
        objective_error = abs(objective - reference["objective"])
        objective_scale = max(1.0, abs(reference["objective"]))
        primal = float(fields["primal"])
        dual = float(fields["dual"])
        if status != expected_status:
            failures.append(f"{name}: status {status}, expected {expected_status}")
        if not math.isfinite(objective) or objective_error > args.objective_tolerance * objective_scale:
            failures.append(
                f"{name}: objective error {objective_error:.3e} exceeds tolerance"
            )
        if primal > args.feasibility_tolerance or dual > args.feasibility_tolerance:
            failures.append(
                f"{name}: infeasibilities primal={primal:.3e}, dual={dual:.3e}"
            )
        print(
            f"{name}\t{status}\t{objective_error:.3e}\t{fields['iterations']}\t"
            f"{reference['iterations']}\t{primal:.3e}\t{dual:.3e}",
            flush=True,
        )
    if failures:
        print("\nFAILURES")
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
