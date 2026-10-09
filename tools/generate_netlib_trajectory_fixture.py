#!/usr/bin/env python3
"""Generate the pinned native-GLOP Netlib pivot-trajectory fixture.

The native adapter must emit one ``NATIVE_PIVOT entering leaving row iteration``
line on stderr after every dual-simplex pivot. This hook is intentionally kept
out of the pinned upstream checkout; use a temporary diagnostic build when
regenerating the fixture.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "netlib"
DEFAULT_BINARY = (
    PROJECT_ROOT / "target" / "native" / "dual_netlib_trace_reference_adapter"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "baselines" / "netlib-dual-trajectories.json.gz"
UPSTREAM_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"
OMITTED_MODELS = {"qap12", "qap15"}
VECTOR_FIELDS = {"basis", "value_bits", "reduced_bits", "norm_bits"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_snapshot(stdout: str) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name in VECTOR_FIELDS:
            snapshot[name] = [int(value) for value in fields[1:]]
        elif name in {"iterations", "updates"}:
            snapshot[name] = int(fields[1])
        elif name == "status":
            snapshot[name] = fields[1]
    missing = {"status", "iterations", "updates", *VECTOR_FIELDS} - snapshot.keys()
    if missing:
        raise RuntimeError(f"native adapter omitted fields: {sorted(missing)}")
    return snapshot


def parse_pivots(stderr: str) -> list[list[int]]:
    pivots = []
    for line in stderr.splitlines():
        fields = line.split()
        if fields and fields[0] == "NATIVE_PIVOT":
            if len(fields) != 5:
                raise RuntimeError(f"malformed native pivot line: {line}")
            pivots.append([int(value) for value in fields[1:]])
    return pivots


def main() -> None:
    arguments = parse_arguments()
    manifest_path = DATASET_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = [
        problem
        for problem in manifest["problems"]
        if problem["name"] not in OMITTED_MODELS
    ]
    results = []
    for number, problem in enumerate(problems, start=1):
        completed = subprocess.run(
            [arguments.binary, DATASET_ROOT / problem["path"], "1000000"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = parse_snapshot(completed.stdout)
        result["name"] = problem["name"]
        result["input"] = problem["path"]
        result["mps_sha256"] = problem["mps_sha256"]
        result["pivots"] = parse_pivots(completed.stderr)
        if len(result["pivots"]) != result["iterations"]:
            raise RuntimeError(
                f"{problem['name']}: captured {len(result['pivots'])} pivots "
                f"for {result['iterations']} iterations"
            )
        results.append(result)
        print(
            f"[{number:02d}/{len(problems)}] {problem['name']}: {result['iterations']} pivots"
        )

    fixture = {
        "solver": "Google OR-Tools GLOP",
        "upstream_commit": UPSTREAM_COMMIT,
        "generated": datetime.now(tz=UTC).date().isoformat(),
        "parameters": {"use_dual_simplex": True, "use_scaling": False},
        "omitted_models": sorted(OMITTED_MODELS),
        "dataset_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    contents = (json.dumps(fixture, separators=(",", ":")) + "\n").encode()
    arguments.output.write_bytes(gzip.compress(contents, compresslevel=9, mtime=0))
    print(f"wrote {len(results)} trajectories to {arguments.output}")


if __name__ == "__main__":
    main()
