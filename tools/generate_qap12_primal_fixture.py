#!/usr/bin/env python3
"""Generate qap12 primal snapshots from pinned GLOP with a pivot trace hook.

The native binary must emit `NATIVE_PIVOT_DIAG iteration entering leaving row`
on stderr for each pivot. Use an instrumented copy of the pinned upstream
source in an ignored build directory; do not edit the upstream checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "datasets/netlib/mps/qap12.mps"
UPSTREAM = ROOT.parent / "or-tools"
OUTPUT = ROOT / "baselines/qap12-primal.json"
PINNED_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def digest(values: list[str]) -> str:
    return hashlib.sha256(" ".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=UPSTREAM,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"native checkout {commit} is not pinned {PINNED_COMMIT}")
    snapshots = []
    for limit in (21502, 1000000):
        result = subprocess.run(
            [args.binary.resolve(), MODEL, str(limit), "primal"],
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        fields = {
            items[0]: items[1:]
            for line in result.stdout.splitlines()
            if (items := line.split())
        }
        pivots = []
        for line in result.stderr.splitlines():
            items = line.split()
            if items and items[0] == "NATIVE_PIVOT_DIAG":
                iteration, entering, leaving, row = map(int, items[1:])
                pivots.append(f"{entering}:{leaving}:{row}:{iteration}")
        iterations = int(fields["iterations"][0])
        if len(pivots) != iterations:
            raise RuntimeError(f"limit {limit}: {len(pivots)} pivots for {iterations} iterations")
        snapshots.append(
            {
                "limit": limit,
                "status": fields["status"][0],
                "iterations": iterations,
                "pivot_sha256": digest(pivots),
                "basis_sha256": digest(fields["basis"]),
                "reduced_bits_sha256": digest(fields["reduced_bits"]),
            }
        )
        print(f"limit {limit}: {snapshots[-1]['status']} after {iterations} pivots")
    fixture = {
        "upstream_commit": PINNED_COMMIT,
        "mps_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "parameters": {"use_dual_simplex": False, "use_scaling": False},
        "snapshots": snapshots,
    }
    OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
