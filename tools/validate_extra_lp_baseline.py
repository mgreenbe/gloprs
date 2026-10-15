#!/usr/bin/env python3
"""Opt-in Rust regression gate against pinned native extra-corpus outcomes."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess

from lp_corpus import ROOT, models


BASELINE = ROOT / "baselines/extra-native-outcomes.json"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"


def number(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="corpus/name identifiers")
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_text())
    wall_seconds = baseline["wall_seconds"]
    if baseline["corpus"] != "extra":
        raise ValueError("unexpected extra-corpus baseline metadata")
    by_key = {case.key: case for case in models("extra")}
    selected = []
    for entry in baseline["results"]:
        key = f"{entry['corpus']}/{entry['name']}"
        if args.models is None or key in args.models:
            selected.append((key, by_key[key], entry))
    if args.models is not None and {key for key, _, _ in selected} != set(args.models):
        parser.error("some requested models are unknown or wall-limited")
    for number_, (key, case, expected) in enumerate(selected, 1):
        native = expected["native"]
        result = subprocess.run([RUST, case.path, "phase7", str(wall_seconds)],
                                check=True, capture_output=True, text=True,
                                timeout=wall_seconds + 30)
        actual = result.stdout.splitlines()[0].split()
        if (actual[0] != native[0] or int(actual[1]) != native[1]
                or not math.isclose(number(int(actual[2])), number(native[2]),
                                    rel_tol=1e-8, abs_tol=1e-8)):
            raise AssertionError(f"{key}: native={native}, Rust={actual[:3]}")
        print(f"[{number_}/{len(selected)}] {key}: {actual[0]} {actual[1]}", flush=True)
    print(f"Validated {len(selected)} completed models against pinned native outcomes")


if __name__ == "__main__":
    main()
