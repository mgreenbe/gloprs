#!/usr/bin/env python3
"""Compare the optional transformed dual Phase-I path with pinned GLOP.

Build the native and release Rust ``dual_netlib_trace`` adapters first. Each
solve has an independent 20-second wall limit, as in the strict trajectory
fixture. The test compares terminal status, iteration count, ordered basis,
and all reduced-cost and dual-norm bits.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "datasets/netlib"
FIXTURE = ROOT / "baselines/netlib-dual-trajectories.json.gz"
NATIVE = ROOT / "target/native/dual_netlib_trace_reference_adapter"
RUST = ROOT / "target/release/examples/dual_netlib_trace"
FIELDS = ("status", "iterations", "basis", "reduced_bits", "norm_bits")


def solve(binary: Path, model: Path) -> dict[str, list[str]]:
    output = subprocess.run(
        [binary, model, "100000", "transformed"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20.0,
    ).stdout
    return {
        tokens[0]: tokens[1:]
        for line in output.splitlines()
        if (tokens := line.split())
    }


def main() -> None:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as stream:
        models = json.load(stream)["results"]
    for model in models:
        path = DATASET / model["input"]
        native = solve(NATIVE, path)
        rust = solve(RUST, path)
        for field in FIELDS:
            if native[field] != rust[field]:
                raise AssertionError(f"{model['name']}: {field} differs")
    print(f"{len(models)} transformed-dual solves agree bit-for-bit")


if __name__ == "__main__":
    main()
