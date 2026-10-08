#!/usr/bin/env python3
"""Compare GLOP and gloprs distribution formatting and ordering."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    outputs = []
    for executable in [
        ROOT / "target/native/stats_reference_adapter",
        ROOT / "target/debug/examples/stats_trace",
    ]:
        outputs.append(subprocess.run([executable], text=True, capture_output=True, check=True).stdout)
    if outputs[0] != outputs[1]:
        raise AssertionError(f"GLOP:\n{outputs[0]}\ngloprs:\n{outputs[1]}")
    print("GLOP and gloprs statistics traces agree")


if __name__ == "__main__":
    main()
