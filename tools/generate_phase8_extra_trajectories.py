#!/usr/bin/env python3
"""Pin direct dual-simplex paths for selected extra-corpus performance models.

Requires a temporary pinned-native build emitting
``NATIVE_PIVOT_DIAG iteration entering leaving row`` on stderr. The upstream
checkout itself must remain unmodified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT.parent / "datasets"
UPSTREAM = ROOT.parent / "or-tools"
PINNED_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"
MODELS = ("neos-1445765", "graph20-20-1rand", "milo-v12-6-r2-40-1",
          "cvs16r128-89", "neos-1171737", "neos-3216931-puriri")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "baselines/phase8-extra-trajectories.json.gz")
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                            check=True, text=True, capture_output=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"native checkout {commit} is not {PINNED_COMMIT}")
    by_name = {}
    for band in ("nmm0-50k", "nmm50k-100k"):
        manifest = json.loads((DATASETS / band / "manifest.json").read_text())
        for case in manifest["instances"]:
            if case["name"] in MODELS:
                if case["name"] in by_name:
                    raise RuntimeError(f"duplicate fixture model {case['name']}")
                by_name[case["name"]] = case
    results = []
    for name in MODELS:
        case = by_name[name]
        path = DATASETS / case["path"]
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != case["sha256"]:
            raise RuntimeError(f"{name}: dataset checksum mismatch")
        process = subprocess.run([args.binary.resolve(), path, "1000000"],
                                 check=True, text=True, capture_output=True, timeout=120)
        fields = {parts[0]: parts[1:] for line in process.stdout.splitlines()
                  if (parts := line.split())}
        pivots = []
        for line in process.stderr.splitlines():
            parts = line.split()
            if parts and parts[0] == "NATIVE_PIVOT_DIAG":
                iteration, entering, leaving, row = map(int, parts[1:])
                pivots.append([entering, leaving, row, iteration])
        iterations = int(fields["iterations"][0])
        if len(pivots) != iterations:
            raise RuntimeError(f"{name}: {len(pivots)} pivots for {iterations} iterations")
        results.append({"name": name, "input": case["path"], "mps_sha256": checksum,
                        "status": fields["status"][0], "iterations": iterations,
                        "basis": [int(value) for value in fields["basis"]],
                        "pivots": pivots})
        print(f"{name}: {iterations} pivots, {results[-1]['status']}")
    fixture = {"upstream_commit": PINNED_COMMIT,
               "parameters": {"use_scaling": False, "use_dual_simplex": True},
               "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contents = (json.dumps(fixture, separators=(",", ":")) + "\n").encode()
    args.output.write_bytes(gzip.compress(contents, compresslevel=9, mtime=0))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
