#!/usr/bin/env python3
"""Pin exact default-presolve states for 96 Netlib models to native GLOP."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ROOT.parent / "datasets/netlib/subsets/all.json"
MPS = ROOT.parent / "datasets/netlib/mps"
ADAPTER = ROOT / "target/native/full_presolve_netlib_reference_adapter"
OUTPUT = ROOT / "baselines/main-preprocessor-netlib-native.json"


def main() -> None:
    names = [name for name in json.loads(NAMES.read_text()) if name not in {"qap12", "qap15"}]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "or-tools",
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    results = []
    for name in names:
        result = subprocess.run(
            [ADAPTER, MPS / f"{name}.mps"], check=True,
            capture_output=True, text=True, timeout=25,
        )
        state = result.stderr
        if not state.startswith("presolved "):
            raise RuntimeError(f"missing presolve dump for {name}")
        status, iterations, *_ = result.stdout.split()
        results.append({
            "name": name,
            "presolved_sha256": hashlib.sha256(state.encode()).hexdigest(),
            "presolved_summary": state.splitlines()[0],
            "solve_status": status,
            "solve_iterations": int(iterations),
        })
    OUTPUT.write_text(json.dumps({"upstream_commit": commit, "results": results}, indent=2) + "\n")
    print(f"wrote {len(results)} native default-presolve states to {OUTPUT}")


if __name__ == "__main__":
    main()
