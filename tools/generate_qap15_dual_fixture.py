#!/usr/bin/env python3
"""Pin direct, unscaled qap15 dual snapshots from the upstream GLOP adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "or-tools"
MODEL = ROOT.parent / "datasets/netlib/mps/qap15.mps"
ADAPTER = ROOT / "target/native/dual_netlib_trace_reference_adapter"
OUTPUT = ROOT / "baselines/qap15-dual.json"
PINNED_COMMIT = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5"


def digest(values: list[str]) -> str:
    return hashlib.sha256(" ".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default=[0, 10_000, 20_000, 30_000, 40_000],
        help="iteration caps to snapshot (default: 0 10000 20000 30000 40000)",
    )
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="also solve qap15 without an effective iteration cap",
    )
    options = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=UPSTREAM,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"native checkout {commit} != {PINNED_COMMIT}")

    snapshots = []
    limits = list(options.limits)
    if options.include_terminal:
        limits.append(1_000_000)
    for limit in limits:
        output = subprocess.run(
            [ADAPTER, MODEL, str(limit)],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        ).stdout
        fields = {
            tokens[0]: tokens[1:]
            for line in output.splitlines()
            if (tokens := line.split())
        }
        snapshot = {
            "limit": limit,
            "status": fields["status"][0],
            "iterations": int(fields["iterations"][0]),
            "basis_sha256": digest(fields["basis"]),
            "reduced_bits_sha256": digest(fields["reduced_bits"]),
            "norm_bits_sha256": digest(fields["norm_bits"]),
        }
        snapshots.append(snapshot)
        print(f"{limit}: {snapshot['status']} after {snapshot['iterations']} pivots", flush=True)

    fixture = {
        "upstream_commit": PINNED_COMMIT,
        "mps_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "parameters": {"use_dual_simplex": True, "use_scaling": False},
        "snapshots": snapshots,
    }
    OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
