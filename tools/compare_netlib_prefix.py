#!/usr/bin/env python3
"""Locate a native/Rust revised-simplex trajectory split by iteration prefix.

Both adapters solve with identical parameters and stop after the requested
number of pivots. This is slower than an in-process pivot hook but requires no
modification to the pinned upstream checkout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/dual_netlib_trace_reference_adapter"
RUST = ROOT / "target/release/examples/dual_netlib_trace"


def snapshot(binary: Path, model: Path, iteration: int, mode: str) -> dict[str, list[str]]:
    result = subprocess.run(
        [binary, model, str(iteration), mode],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        fields[0]: fields[1:]
        for line in result.stdout.splitlines()
        if (fields := line.split())
    }


def compare(model: Path, iteration: int, mode: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        native = pool.submit(snapshot, NATIVE, model, iteration, mode)
        rust = pool.submit(snapshot, RUST, model, iteration, mode)
        return native.result(), rust.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--mode", choices=("primal", "perturb", "transformed"), default="primal")
    parser.add_argument("--matching", type=int, required=True)
    parser.add_argument("--different", type=int, required=True)
    parser.add_argument("--basis-set", action="store_true", help="compare basic-column sets only")
    args = parser.parse_args()

    lower, upper = args.matching, args.different
    while upper - lower > 1:
        middle = (lower + upper) // 2
        native, rust = compare(args.model, middle, args.mode)
        same = (
            set(native["basis"]) == set(rust["basis"])
            if args.basis_set
            else native["basis"] == rust["basis"]
            and native["reduced_bits"] == rust["reduced_bits"]
        )
        print(f"{middle}: {'same' if same else 'different'}", flush=True)
        if same:
            lower = middle
        else:
            upper = middle

    print(f"first differing prefix: {upper}")
    previous_native, previous_rust = compare(args.model, lower, args.mode)
    native, rust = compare(args.model, upper, args.mode)
    print(f"previous prefix {lower}: basis equal={previous_native['basis'] == previous_rust['basis']}")
    for name, before, after in (
        ("native", previous_native, native),
        ("rust", previous_rust, rust),
    ):
        changes = [
            (row, old, new)
            for row, (old, new) in enumerate(zip(before["basis"], after["basis"]))
            if old != new
        ]
        removed = set(before["basis"]) - set(after["basis"])
        added = set(after["basis"]) - set(before["basis"])
        print(
            f"{name}: status={after['status'][0]}, basis changes={len(changes)}, "
            f"first changes={changes[:5]}, removed={sorted(removed)}, added={sorted(added)}"
        )
    print(f"rust last pivot: {rust.get('pivot', [])}")
    for field in ("basis", "reduced_bits", "value_bits"):
        indices = [
            i for i, (a, b) in enumerate(zip(native[field], rust[field])) if a != b
        ]
        print(f"{field}: {len(indices)} differences, first indices={indices[:10]}")


if __name__ == "__main__":
    main()
