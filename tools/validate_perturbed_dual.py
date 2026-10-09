#!/usr/bin/env python3
"""Audit perturbed dual solves against pinned GLOP on the 96 fast Netlib LPs.

Build the native and release Rust ``dual_netlib_trace`` adapters first. Each
solve has an independent 20-second wall limit.
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
        [binary, model, "100000", "perturb"],
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
    disagreements: dict[str, set[str]] = {}
    for model in models:
        path = DATASET / model["input"]
        native = solve(NATIVE, path)
        rust = solve(RUST, path)
        differing = {field for field in FIELDS if native[field] != rust[field]}
        if differing:
            disagreements[model["name"]] = differing
    if disagreements:
        raise AssertionError(f"Perturbed-dual differences: {disagreements}")
    print(f"{len(models)} perturbed-dual solves agree bit-for-bit")


if __name__ == "__main__":
    main()
