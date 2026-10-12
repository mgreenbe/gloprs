#!/usr/bin/env python3
"""Regress the formerly divergent default-pipeline clock prefixes."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPS = ROOT.parent / "datasets/netlib/mps"
NATIVE = ROOT / "target/native/full_presolve_netlib_reference_adapter"
RUST = ROOT / "target/release/examples/full_presolve_netlib"


def clock(executable: Path, name: str, cap: int) -> tuple[int, int, int, int]:
    result = subprocess.run(
        [executable, MPS / f"{name}.mps", str(cap), "clock"],
        capture_output=True, text=True, check=True, timeout=25,
    )
    fields = result.stdout.split()
    assert fields[::2] == ["clock", "updates", "basis_time", "refactorizations"], fields
    return int(fields[1]), int(fields[3]), int(fields[5]), int(fields[7])


def solve_stages(executable: Path, name: str, cap: int) -> dict[str, list[str]]:
    result = subprocess.run(
        [executable, MPS / f"{name}.mps", str(cap), "stages"],
        capture_output=True, text=True, check=True, timeout=25,
    )
    lines = {fields[0]: fields[1:] for line in result.stdout.splitlines()
             if (fields := line.split())}
    return {name: lines[name] for name in (
        "stage_upper", "stage_upper_order", "stage_updates",
        "stage_updates_order", "stage_lower", "stage_lower_order")}


def main() -> None:
    cases = [("d2q06c", 31, 32), ("sierra", 228, 229)]
    for name, previous, first in cases:
        before_native = clock(NATIVE, name, previous)
        before_rust = clock(RUST, name, previous)
        after_native = clock(NATIVE, name, first)
        after_rust = clock(RUST, name, first)
        assert before_native == before_rust, (name, previous)
        assert after_native == after_rust, (name, first, after_native, after_rust)
        print(f"{name}: native/Rust basis clocks agree through iteration {first}")
    native_stages = solve_stages(NATIVE, "d2q06c", 31)
    rust_stages = solve_stages(RUST, "d2q06c", 31)
    assert native_stages == rust_stages, "d2q06c transpose-solve sparse support"
    print("d2q06c: LU upper, rank-one update, and LU lower stages agree")


if __name__ == "__main__":
    main()
