#!/usr/bin/env python3
"""Compare default-pipeline recovered Netlib solutions with pinned GLOP."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"
NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"
FIXTURE = ROOT / "baselines/main-preprocessor-netlib-native.json"


def value(bits: str) -> float:
    return struct.unpack("=d", struct.pack("=Q", int(bits)))[0]


def run(executable: Path, model: Path) -> list[list[str]]:
    output = subprocess.run(
        [executable, model], check=True, capture_output=True, text=True, timeout=25
    ).stdout
    return [line.split() for line in output.splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*")
    options = parser.parse_args()
    cases = json.loads(FIXTURE.read_text())["results"]
    if options.models is not None:
        selected = set(options.models)
        cases = [case for case in cases if case["name"] in selected]
        missing = selected - {case["name"] for case in cases}
        if missing:
            raise ValueError(f"unknown models: {sorted(missing)}")
    exact = 0
    checked = 0
    worst = (0.0, "", "", 0, 0.0, 0.0)
    for case in cases:
        name = case["name"]
        model = MODELS / f"{name}.mps"
        native = run(NATIVE, model)
        rust = run(RUST, model)
        if len(native) != 9 or len(rust) != 9:
            raise AssertionError(f"{name}: trace shape")
        if native[0][:2] != rust[0][:2] or native[1][:2] != rust[1][:2]:
            raise AssertionError(f"{name}: status, iterations, or dimensions differ")
        for expected, actual in zip(native[6:8], rust[6:8]):
            if expected != actual:
                differences = [i for i, (a, b) in enumerate(zip(expected[1:], actual[1:])) if a != b]
                raise AssertionError(f"{name}: {expected[0]} differs at {differences[:20]} ({len(differences)} total)")
        for label, left, right in zip(
            ("objective", "primal_infeasibility", "dual_infeasibility"),
            native[0][2:], rust[0][2:],
        ):
            a, b = value(left), value(right)
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(f"{name}: {label} {a} != {b}")
        for expected, actual in zip(native[2:6], rust[2:6]):
            if expected[0] != actual[0] or len(expected) != len(actual):
                raise AssertionError(f"{name}: vector shape differs")
            for index, (left, right) in enumerate(zip(expected[1:], actual[1:])):
                checked += 1
                if left == right:
                    exact += 1
                    continue
                a, b = value(left), value(right)
                error = abs(a - b) / max(1.0, abs(a), abs(b))
                if error > worst[0]:
                    worst = (error, name, expected[0], index, a, b)
                if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                    raise AssertionError(f"{name}: {expected[0]}[{index}] {a} != {b}")
        print(f"{name}: status, iterations, statuses, and solution agree", flush=True)
    print(f"{len(cases)} models agree; {exact}/{checked} solution bits exact; worst {worst}")


if __name__ == "__main__":
    main()
