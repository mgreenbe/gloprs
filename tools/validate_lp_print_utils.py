#!/usr/bin/env python3
"""Compare decimal and monomial formatting with pinned GLOP."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=10_000)
    args = parser.parse_args()
    generator = random.Random(0x5052494E54)
    bits = [
        0,
        1 << 63,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
    ]
    while len(bits) < args.cases:
        candidate = generator.getrandbits(64)
        if candidate & 0x7FF0000000000000 != 0x7FF0000000000000:
            bits.append(candidate)
    source = f"{len(bits)}\n" + " ".join(f"{value:x}" for value in bits) + "\n"

    outputs = []
    for executable in (
        ROOT / "target/native/lp_print_utils_reference_adapter",
        ROOT / "target/debug/examples/lp_print_utils_trace",
    ):
        outputs.append(
            subprocess.run(
                [executable], input=source, text=True, capture_output=True, check=True
            ).stdout
        )
    if outputs[0] != outputs[1]:
        native = outputs[0].splitlines()
        rust = outputs[1].splitlines()
        mismatch = next(
            index
            for index, pair in enumerate(zip(native, rust, strict=False))
            if pair[0] != pair[1]
        )
        raise AssertionError(
            f"case {mismatch // 4}, line {mismatch}: "
            f"native={native[mismatch]!r}, rust={rust[mismatch]!r}"
        )
    print(f"{len(bits)} lp_print_utils traces agree exactly")


if __name__ == "__main__":
    main()
