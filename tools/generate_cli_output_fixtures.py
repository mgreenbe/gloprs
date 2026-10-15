#!/usr/bin/env python3
"""Capture CLI layout fixtures using the pinned native GLOP API.

Build tools/cli_output_reference_adapter.cc first. Solve labels/precision use
ortools/linear_solver/solve.cc conventions with GLOP's API status names and
iteration/feasibility fields; inspect is GetPrettyProblemStats() verbatim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without overwriting")
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "target/native/cli_output_reference_adapter")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT.parent / "or-tools"), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError("native checkout does not match pinned upstream")
    output = ROOT / "tests/fixtures/cli"
    cases = [(output / "model.mps", "model"),
             (ROOT.parent / "datasets/netlib/mps/afiro.mps", "afiro")]
    metadata = {"upstream_commit": PIN,
                "formatting_reference": "ortools/linear_solver/solve.cc (labels/precision), "
                                        "ortools/lp_data/lp_data.cc (inspect)",
                "solve_scope": "selected summary fields; GLOP API status names; no timing/logs",
                "models": {}}
    for model, name in cases:
        metadata["models"][name] = hashlib.sha256(model.read_bytes()).hexdigest()
        for mode in ("solve", "inspect"):
            actual = subprocess.check_output(
                [str(args.binary.resolve()), model.name, mode], cwd=model.parent)
            fixture = output / f"{name}.{mode}.txt"
            if args.check:
                if fixture.read_bytes() != actual:
                    raise AssertionError(f"native output changed: {fixture}")
            else:
                fixture.write_bytes(actual)
    text = json.dumps(metadata, indent=2) + "\n"
    manifest = output / "native.json"
    if args.check:
        if manifest.read_text() != text:
            raise AssertionError("fixture provenance changed")
    else:
        manifest.write_text(text)
    print("CLI fixtures match pinned native output" if args.check else "Captured native CLI fixtures")


if __name__ == "__main__":
    main()
