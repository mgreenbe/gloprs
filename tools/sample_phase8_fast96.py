#!/usr/bin/env python3
"""Serial paired stack samples of repeated LP pipeline stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "datasets/netlib/mps"
BINARIES = {
    "native": ROOT / "target/native/full_presolve_netlib_reference_adapter",
    "rust": ROOT / "target/release/examples/full_presolve_netlib",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Netlib model name or path to an MPS model")
    parser.add_argument("--stage", choices=("presolve", "solve"), default="solve")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--interval-ms", type=int, default=1)
    parser.add_argument("--rust", type=Path, default=BINARIES["rust"],
                        help="Rust executable to sample (for controlled compiler builds)")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.delay < 0 or args.duration < 1 or args.interval_ms < 1:
        parser.error("invalid repetitions or sample interval")
    model = Path(args.model) if args.model.endswith(".mps") else MODELS / f"{args.model}.mps"
    if not model.is_file():
        parser.error(f"missing model: {model}")

    results = []
    for label, binary in (("native", BINARIES["native"]), ("rust", args.rust)):
        print(f"sampling {args.model} {label}", flush=True)
        process = subprocess.Popen(
            [binary, model,
             "presolved_solve_timing" if args.stage == "solve" else "presolve_timing",
             str(args.repetitions)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(args.delay)
        if process.poll() is not None:
            raise RuntimeError(f"{label} completed before sampling began")
        sample_file = args.output_prefix.with_name(
            args.output_prefix.name + f"-{label}.sample.txt"
        )
        sample = subprocess.run(
            ["sample", str(process.pid), str(args.duration), str(args.interval_ms),
             "-mayDie", "-file", str(sample_file)],
            capture_output=True, text=True, check=False,
        )
        stdout, stderr = process.communicate(timeout=180)
        if process.returncode != 0 or sample.returncode != 0:
            raise RuntimeError(
                f"{label}: solve={process.returncode} sample={sample.returncode}; "
                f"stderr={stderr}; sampler={sample.stderr}"
            )
        result = {"label": label, "solve_output": stdout.strip(),
                  "sample_file": str(sample_file)}
        results.append(result)
        print(result, flush=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
