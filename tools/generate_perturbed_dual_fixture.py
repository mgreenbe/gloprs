#!/usr/bin/env python3
"""Capture pinned GLOP's perturbed-dual terminal states for fast Netlib.

Build the native dual_netlib_trace_reference_adapter first. Each native solve
has its own 20-second wall-clock limit. The resulting gzip JSON is deterministic
apart from changes to the pinned native build or the shared Netlib manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "datasets/netlib"
NATIVE = ROOT / "target/native/dual_netlib_trace_reference_adapter"
OUTPUT = ROOT / "baselines/netlib-perturbed-dual.json.gz"
UPSTREAM_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"
OMITTED = {"qap12", "qap15"}
VECTOR_FIELDS = {"basis", "value_bits", "reduced_bits", "norm_bits"}
SCALAR_FIELDS = {"status", "iterations", "updates"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=NATIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()

    manifest_path = DATASET / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    problems = [
        item
        for item in json.loads(manifest_bytes)["problems"]
        if item["name"] not in OMITTED
    ]
    if len(problems) != 96:
        raise RuntimeError(f"expected 96 fast Netlib models, got {len(problems)}")

    results = []
    for number, problem in enumerate(problems, 1):
        completed = subprocess.run(
            [arguments.binary, DATASET / problem["path"], "1000000", "perturb"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        result: dict[str, object] = {}
        for line in completed.stdout.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] in VECTOR_FIELDS:
                result[fields[0]] = [int(value) for value in fields[1:]]
            elif fields[0] in {"iterations", "updates"}:
                result[fields[0]] = int(fields[1])
            elif fields[0] == "status":
                result[fields[0]] = fields[1]
        missing = (VECTOR_FIELDS | SCALAR_FIELDS) - result.keys()
        if missing:
            raise RuntimeError(f"{problem['name']}: missing native fields {missing}")
        result.update(
            name=problem["name"],
            input=problem["path"],
            mps_sha256=problem["mps_sha256"],
        )
        results.append(result)
        print(f"[{number:02d}/96] {problem['name']}: {result['iterations']} iterations", flush=True)

    fixture = {
        "solver": "Google OR-Tools GLOP",
        "upstream_commit": UPSTREAM_COMMIT,
        "parameters": {
            "use_dual_simplex": True,
            "use_scaling": False,
            "perturb_costs_in_dual_simplex": True,
            "max_number_of_iterations": 1000000,
        },
        "omitted_models": sorted(OMITTED),
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(fixture, separators=(",", ":")) + "\n").encode()
    arguments.output.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    print(f"wrote {len(results)} native results to {arguments.output}")


if __name__ == "__main__":
    main()
