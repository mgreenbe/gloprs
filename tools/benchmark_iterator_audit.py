#!/usr/bin/env python3
"""Alternate saved release binaries for the counted-loop audit; retain raw trials."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path
from benchmark_lower_transpose import factor_input
from benchmark_phase8_stages import run as run_stage
from lp_corpus import models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("triangular", "rank", "solve", "pricing"), required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--entries", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 7, 16, 17, 64])
    parser.add_argument("--models", nargs="+",
                        default=["graph20-20-1rand", "neos-1171737", "traininstance2"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or min(args.entries) < 1 or max(args.entries) * 13 >= 2048:
        parser.error("trials must be positive and entries must be between 1 and 157")
    cases = []
    if args.kind == "triangular":
        for n in (256, 1024):
            for density in (0.01, 0.1):
                for unit in (True, False):
                    data = factor_input(n, density, 2000, not unit)
                    for mode in ("lower", "upper", "transpose-lower", "transpose-upper"):
                        cases.append((dict(n=n, density=density, unit=unit, mode=mode),
                                      [mode], data))
    elif args.kind == "rank":
        for updates in (8, 96):
            for entries in args.entries:
                for left in (0, 1):
                    params = [4096, updates, entries, 10000, left]
                    cases.append((dict(updates=updates, entries=entries, left=left),
                                  list(map(str, params)), None))
    elif args.kind == "pricing":
        for size in (31, 64, 1024, 4096):
            cases.append((dict(size=size), [str(size), "10000"], None))
    else:
        by_name = {m.name: m.path for m in models("extra") + models("netlib-fast96")}
        for name in args.models:
            cases.append((dict(model=name), [str(by_name[name])], None))
    results = []
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    metadata = dict(
        kind=args.kind, trials=args.trials, warmups=1, machine=platform.platform(),
        rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
        binary_sha256={label: hashlib.sha256(path.read_bytes()).hexdigest()
                       for label, path in binaries.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case, parameters, data in cases:
        times = {label: [] for label in binaries}
        for trial in range(args.trials + 1):
            order = list(binaries)
            if trial % 2:
                order.reverse()
            checksums = []
            for label in order:
                if args.kind == "solve":
                    outcome = run_stage(binaries[label], Path(parameters[0]), "solve", 1)
                    elapsed = outcome.pop("seconds")
                    checksums.append(outcome)
                else:
                    fields = subprocess.run([binaries[label], *parameters], input=data,
                                            text=True, capture_output=True, check=True).stdout.split()
                    elapsed, checksum = map(float, fields)
                    checksums.append(checksum)
                if trial:
                    times[label].append(elapsed)
            # These are Rust A/B builds: require exact status, pivot count,
            # and objective bits, rather than native-relative tolerances.
            agrees = (checksums[0] == checksums[1] if args.kind == "solve"
                      else math.isclose(*checksums, rel_tol=1e-12, abs_tol=1e-10))
            if not agrees:
                raise RuntimeError(f"checksum mismatch: {case} {checksums}")
        ratio = statistics.median(times["after"]) / statistics.median(times["before"])
        results.append(dict(case=case, times=times,
                            unit="seconds" if args.kind == "solve" else "nanoseconds",
                            ratio=ratio, outcome=checksums[0]))
        print(f"{case}: after/before={ratio:.4f}", flush=True)
        # Checkpoint every finished case so interrupted campaigns retain data.
        args.output.write_text(json.dumps(dict(metadata, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
