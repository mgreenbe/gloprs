#!/usr/bin/env python3
"""Serial late-window stack samples of capped native and Rust qap15 solves.

Run after a paired 25k/30k cap timing establishes the pivot-25k wall offset.
Sampling is diagnostic and deliberately separate from uncontended timings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "datasets/netlib/mps/qap15.mps"
BINARIES = {
    "native": ROOT / "target/native/netlib_timing_reference_adapter",
    "rust": ROOT / "target/release/examples/netlib_timing",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=30_000)
    parser.add_argument("--delay", type=float, default=18.0)
    parser.add_argument("--duration", type=int, default=65)
    parser.add_argument("--interval-ms", type=int, default=10)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.cap <= 0 or args.delay < 0 or args.duration <= 0 or args.interval_ms <= 0:
        parser.error("invalid cap, delay, duration, or sampling interval")

    results = []
    for label, binary in BINARIES.items():
        print(f"sampling {label}", flush=True)
        started = time.monotonic()
        process = subprocess.Popen(
            [binary, MODEL, "1", "dual", str(args.cap), "240"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(args.delay)
        if process.poll() is not None:
            raise RuntimeError(f"{label} completed before the requested sample window")
        sample_path = args.output_prefix.with_name(
            args.output_prefix.name + f"-{label}.sample.txt"
        )
        sampled = subprocess.run(
            [
                "sample", str(process.pid), str(args.duration),
                str(args.interval_ms), "-mayDie", "-file", str(sample_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        stdout, stderr = process.communicate(timeout=270)
        if process.returncode != 0 or sampled.returncode != 0:
            raise RuntimeError(
                f"{label}: solve={process.returncode}, sample={sampled.returncode}; "
                f"solve stderr={stderr}; sample stderr={sampled.stderr}"
            )
        result = {
            "label": label,
            "wall_seconds_with_sampler": time.monotonic() - started,
            "solve_output": stdout.strip(),
            "sample_file": str(sample_path),
        }
        results.append(result)
        print(result, flush=True)
    output = args.output_prefix.with_suffix(".json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
