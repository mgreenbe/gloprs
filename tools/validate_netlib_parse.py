#!/usr/bin/env python3
"""Compare gloprs MPS model statistics with the pinned native GLOP adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT.parent / "datasets" / "netlib"
DEFAULT_CLI = PROJECT_ROOT / "target" / "debug" / "gloprs"
DEFAULT_RUNNER = PROJECT_ROOT / "tools" / "run_glop_reference.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for number, problem in enumerate(manifest["problems"], start=1):
        model_path = args.dataset / problem["path"]
        rust = subprocess.run(
            [args.cli, "inspect", "--tsv", model_path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.rstrip("\n").split("\t")
        rust_stats = {
            "rows": int(rust[1]),
            "columns": int(rust[2]),
            "nonzeros": int(rust[3]),
            "data_fingerprint": rust[4],
        }
        native = json.loads(
            subprocess.run(
                [sys.executable, args.runner, "--summary", model_path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )["model"]
        if rust_stats != native:
            failures.append(f"{problem['name']}: Rust {rust_stats}, GLOP {native}")
        print(f"[{number:02d}/{len(manifest['problems'])}] {problem['name']}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"all {len(manifest['problems'])} models agree")


if __name__ == "__main__":
    main()
