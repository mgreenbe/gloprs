#!/usr/bin/env python3
"""Generate Matrix Market numerical-kernel probes for Netlib LPs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT.parent / "datasets"
MANIFEST = DATASETS / "netlib/manifest.json"
MODELS = DATASETS / "netlib/mps"
OUTPUT = DATASETS / "netlib-simplex-probes"
BINARY = ROOT / "target/release/examples/netlib_simplex_probe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Netlib instance names (without .mps)")
    parser.add_argument("--all", action="store_true", help="run all Netlib instances")
    parser.add_argument("--wall-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args()
    if options.wall_seconds <= 0 or not BINARY.is_file():
        parser.error("positive --wall-seconds and a release-built probe example are required")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {problem["name"] for problem in manifest["problems"]}
    if options.all == bool(options.models):
        parser.error("specify either --all or one or more model names")
    selected = [problem["name"] for problem in manifest["problems"]] if options.all else options.models
    unknown = set(selected) - known
    if unknown:
        parser.error(f"unknown Netlib instances: {sorted(unknown)}")
    for name in selected:
        model = MODELS / f"{name}.mps"
        print(f"Probing {name}...", flush=True)
        subprocess.run(
            [BINARY, model, str(options.wall_seconds), options.output],
            check=True,
        )


if __name__ == "__main__":
    main()
