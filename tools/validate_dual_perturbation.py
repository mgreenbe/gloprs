#!/usr/bin/env python3
"""Compare GLOP and gloprs immediately after dual cost perturbation.

Build the two ``dual_netlib_trace`` adapters first. The iteration limit of zero
isolates initialization and perturbation from later pivot-path differences.
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
FIELDS = ("basis", "reduced_bits", "norm_bits")


def snapshot(binary: Path, model: Path) -> dict[str, list[str]]:
    output = subprocess.run(
        [binary, model, "0", "perturb"],
        check=True,
        capture_output=True,
        text=True,
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
        native = snapshot(NATIVE, path)
        rust = snapshot(RUST, path)
        for field in FIELDS:
            if native[field] != rust[field]:
                raise AssertionError(f"{model['name']}: {field} differs")
    print(f"{len(models)} perturbed initial states agree bit-for-bit")


if __name__ == "__main__":
    main()
