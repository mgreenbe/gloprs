#!/usr/bin/env python3
"""Compare deterministic, external, merge, and wall-limit GLOP traces."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def trace(executable: Path) -> str:
    return subprocess.run(
        [executable], text=True, capture_output=True, check=True
    ).stdout


def equivalent(native: str, rust: str) -> bool:
    native_lines = [line.split() for line in native.splitlines()]
    rust_lines = [line.split() for line in rust.splitlines()]
    if len(native_lines) != len(rust_lines):
        return False
    for expected, actual in zip(native_lines, rust_lines, strict=True):
        if expected[0] != actual[0] or len(expected) != len(actual):
            return False
        for left, right in zip(expected[1:], actual[1:], strict=True):
            if left in {"true", "false"} or right in {"true", "false"}:
                if left != right:
                    return False
            elif float(left) != float(right):
                return False
    return True


def main() -> None:
    native = trace(ROOT / "target/native/time_limit_reference_adapter")
    rust = trace(ROOT / "target/debug/examples/time_limit_trace")
    if not equivalent(native, rust):
        raise AssertionError(f"GLOP:\n{native}\ngloprs:\n{rust}")
    print("GLOP and gloprs time-limit traces agree")


if __name__ == "__main__":
    main()
