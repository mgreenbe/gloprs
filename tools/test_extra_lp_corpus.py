#!/usr/bin/env python3
"""Differential public-solver gate for small MIPLIB and Mittelmann LP models.

Run after building the release Rust and pinned native phase5 solution adapters.
MIPLIB models are solved as LP relaxations; integrality is not enforced by GLOP.
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess

from lp_corpus import ROOT, models


NATIVE = ROOT / "target/native/phase5_netlib_solution_reference_adapter"
RUST = ROOT / "target/release/examples/phase5_netlib_solution_trace"


def number(bits: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


def check(native: list[str], rust: list[str]) -> None:
    assert len(native) == len(rust) == 9
    native_header, rust_header = native[0].split(), rust[0].split()
    assert native_header[:2] == rust_header[:2], (native_header, rust_header)
    assert math.isclose(number(native_header[2]), number(rust_header[2]),
                        rel_tol=1e-8, abs_tol=1e-8)
    assert native[1].split()[:2] == rust[1].split()[:2]
    for line in range(2, 6):
        left, right = native[line].split(), rust[line].split()
        assert left[0] == right[0] and len(left) == len(right)
        for position, (a, b) in enumerate(zip(left[1:], right[1:])):
            assert math.isclose(number(a), number(b), rel_tol=1e-9, abs_tol=1e-9), (
                line, position, number(a), number(b))
    assert native[6:8] == rust[6:8], "basis variable/constraint status differs"


def trace(executable, path, wall_seconds: float, timeout: float) -> list[str]:
    result = subprocess.run([executable, path, "phase7", str(wall_seconds)],
                            capture_output=True, text=True, check=True, timeout=timeout)
    return result.stdout.splitlines()[:9]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("miplib2017", "mittelmann-lpopt", "extra"),
                        default="extra")
    parser.add_argument("--models", nargs="+", help="corpus/name or unique model name")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 3600:
        parser.error("wall-seconds must be positive and at most 3600")
    selected = models(args.corpus)
    if args.models:
        keys = set(args.models)
        selected = [case for case in selected if case.key in keys or case.name in keys]
        if len(selected) != len(keys):
            parser.error("some requested model names are unknown or ambiguous")
    inconclusive = []
    for number_, case in enumerate(selected, 1):
        if not case.path.is_file():
            raise FileNotFoundError(case.path)
        timeout = args.wall_seconds + 30
        native = trace(NATIVE, case.path, args.wall_seconds, timeout)
        rust = trace(RUST, case.path, args.wall_seconds, timeout)
        native_status = native[0].split()[0]
        rust_status = rust[0].split()[0]
        if (max(float(native[8].split()[1]), float(rust[8].split()[1]))
                >= 0.95 * args.wall_seconds
                and (native_status != "OPTIMAL" or rust_status != "OPTIMAL")):
            inconclusive.append(case.key)
            print(f"[{number_}/{len(selected)}] {case.key}: wall-limited, inconclusive",
                  flush=True)
            continue
        try:
            check(native, rust)
        except AssertionError as error:
            raise AssertionError(f"{case.key}: {error}") from error
        print(f"[{number_}/{len(selected)}] {case.key}: "
              f"{native[0].split()[0]} {native[0].split()[1]} iterations", flush=True)
    print(f"Validated {len(selected) - len(inconclusive)} models against pinned native GLOP; "
          f"{len(inconclusive)} wall-limited/inconclusive: {inconclusive}")


if __name__ == "__main__":
    main()
