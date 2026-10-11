#!/usr/bin/env python3
"""Pin native public-solver postsolve states for the 96 fast Netlib models."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
NAMES = ROOT / "baselines/main-preprocessor-netlib-native.json"
OUTPUT = ROOT / "baselines/phase5-netlib-solution-native.json.gz"
UPSTREAM_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def main() -> None:
    names = [case["name"] for case in json.loads(NAMES.read_text())["results"]]
    results = []
    for name in names:
        process = subprocess.run(
            [NATIVE, MODELS / f"{name}.mps"],
            check=True, capture_output=True, text=True, timeout=25,
        )
        lines = process.stdout.splitlines()
        if len(lines) != 9:
            raise ValueError(f"{name}: unexpected trace length {len(lines)}")
        results.append({"name": name, "lines": lines[:8]})
        print(name, flush=True)
    payload = json.dumps({
        "upstream_commit": UPSTREAM_COMMIT,
        "time_limit_seconds": 20,
        "results": results,
    }, separators=(",", ":")).encode()
    with OUTPUT.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    print(f"{len(results)} native solutions: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
