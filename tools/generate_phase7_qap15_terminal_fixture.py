#!/usr/bin/env python3
"""Capture pinned GLOP's completed default-pipeline qap15 solve."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "datasets/netlib/mps/qap15.mps"
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
OUTPUT = ROOT / "baselines/phase7-qap15-terminal-native.json.gz"
PIN = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def main() -> None:
    manifest = json.loads((ROOT.parent / "datasets/netlib/manifest.json").read_text())
    case = next(problem for problem in manifest["problems"] if problem["name"] == "qap15")
    checksum = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert checksum == case["mps_sha256"]
    process = subprocess.run(
        [NATIVE, MODEL, "phase7", "180"],
        capture_output=True, text=True, check=True, timeout=210,
    )
    lines = process.stdout.splitlines()
    assert len(lines) == 11, len(lines)
    assert lines[0].startswith("OPTIMAL "), lines[0]
    assert lines[9].startswith("deterministic "), lines[9]
    assert lines[10].startswith("refactorizations "), lines[10]
    fixture = {
        "upstream_commit": PIN,
        "model_sha256": checksum,
        "internal_wall_seconds": 180,
        "lines": lines,
    }
    OUTPUT.write_bytes(gzip.compress((json.dumps(fixture, sort_keys=True) + "\n").encode(), mtime=0))
    print(f"{OUTPUT}: {lines[0]}, {lines[10]}")


if __name__ == "__main__":
    main()
