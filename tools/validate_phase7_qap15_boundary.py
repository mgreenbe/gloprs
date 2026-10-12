#!/usr/bin/env python3
"""Opt-in regression for the narrowest tested qap15 capped-state split.

An iteration cap triggers solver cleanup. Consequently this is a boundary
between *capped solve results*, not proof of the first divergent live pivot.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "datasets/netlib/mps/qap15.mps"
NATIVE = ROOT / "target/native/full_presolve_netlib_reference_adapter"
RUST = ROOT / "target/release/examples/full_presolve_netlib"

EXPECTED = {
    ("native", 67016): "fingerprint PRIMAL_FEASIBLE 67016 517183055442211656 8460763652137331920 12468359482389918763 4644158713029490075 4643795558761367181 460",
    ("rust", 67016): "fingerprint PRIMAL_FEASIBLE 67016 517183055442211656 8460763652137331920 12468359482389918763 4644158713029490075 4643795558761367181 460",
    ("native", 67017): "fingerprint PRIMAL_FEASIBLE 67017 12980704661235042970 12378309574913995901 17407790522513068365 4644158689225321349 4643795534858330369 460",
    ("rust", 67017): "fingerprint PRIMAL_FEASIBLE 67017 13675232656710914380 12378309574913995901 10894796018542025057 4644158689120636569 4643795534818666309 460",
}


def fingerprint(executable: Path, cap: int) -> str:
    process = subprocess.run(
        [executable, MODEL, str(cap), "fingerprint", "180"],
        capture_output=True, text=True, check=True, timeout=210,
    )
    return process.stdout.strip()


def main() -> None:
    for cap in (67016, 67017):
        for name, executable in (("native", NATIVE), ("rust", RUST)):
            actual = fingerprint(executable, cap)
            expected = EXPECTED[(name, cap)]
            assert actual == expected, (name, cap, expected, actual)
            print(f"{name} qap15 capped state {cap}: pinned fingerprint matches")
    assert EXPECTED[("native", 67016)] == EXPECTED[("rust", 67016)]
    assert EXPECTED[("native", 67017)] != EXPECTED[("rust", 67017)]


if __name__ == "__main__":
    main()
