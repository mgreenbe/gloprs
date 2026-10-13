#!/usr/bin/env python3
"""Serial GLOP/gloprs LU timing on the 2,593 saved Netlib basis matrices."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT.parent / "datasets/netlib-simplex-probes"
NATIVE = ROOT / "target/native/netlib_factorization_reference_adapter"
RUST = ROOT / "target/release/examples/netlib_factorization_benchmark"


def run(binary: Path, paths: list[Path], repetitions: int, warmups: int,
        selected_source: Path | None, evict_mib: int,
        retain_factorization: bool, pivot_threshold: float | None) -> dict[Path, tuple[int, int, float, float]]:
    command = [binary, str(repetitions), str(warmups)]
    if evict_mib:
        command.extend(("--evict-mib", str(evict_mib)))
    if retain_factorization:
        command.append("--retain-factorization")
    if pivot_threshold is not None:
        command.extend(("--pivot-threshold", str(pivot_threshold)))
    if selected_source is not None:
        command.extend(("--selected-source", str(selected_source)))
    command.extend(str(path) for path in paths)
    output = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    results = {}
    for line in output.splitlines():
        path, dimension, nonzeros, count, seconds, determinant = line.split("\t")
        key = Path(path)
        if key in results or int(count) != repetitions:
            raise ValueError(f"duplicate or malformed result: {line}")
        results[key] = (int(dimension), int(nonzeros), float(seconds), float(determinant))
    if set(results) != set(paths):
        raise ValueError(f"{binary}: missing or unexpected benchmark results")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROBES)
    parser.add_argument("--selected-source", type=Path,
                        help="benchmark selection-*.txt views into this original compact matrix")
    parser.add_argument("--native", type=Path, default=NATIVE)
    parser.add_argument("--rust", type=Path, default=RUST)
    parser.add_argument("--output", type=Path, default=ROOT / "target/netlib-factorization-timings.csv")
    parser.add_argument("--raw-output", type=Path, help="per-probe, per-trial timings (defaults beside --output)")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--evict-mib", type=int, default=0,
                        help="touch this many MiB outside each factorization timer")
    parser.add_argument("--retain-factorization", action="store_true",
                        help="reuse one LU object across all selected inputs")
    parser.add_argument("--chronological", action="store_true",
                        help="sort by numeric pivot, then selection filename")
    parser.add_argument("--pivot-threshold", type=float,
                        help="override the native/Rust LU pivot threshold")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--limit", type=int, help="use the first N paths for a smoke test")
    parser.add_argument("--models", nargs="+", help="restrict probes to these Netlib models")
    parser.add_argument("--pivot-min", type=int, help="minimum refactorization pivot (inclusive)")
    parser.add_argument("--pivot-max", type=int, help="maximum refactorization pivot (inclusive)")
    options = parser.parse_args()
    if options.repetitions < 1 or options.warmups < 0 or options.trials < 1 or not 0 <= options.evict_mib <= 1024:
        parser.error("repetitions and trials must be positive; warmups must be nonnegative")
    if options.pivot_threshold is not None and not 0 <= options.pivot_threshold <= 1:
        parser.error("pivot threshold must lie in [0, 1]")
    for binary in (options.native, options.rust):
        if not binary.is_file():
            parser.error(f"build the benchmark executable first: {binary}")
    if options.selected_source is not None and not options.selected_source.is_file():
        parser.error(f"selected source does not exist: {options.selected_source}")
    pattern = "*-*/selection-*.txt" if options.selected_source else "*-*/factorization-*.mtx"
    paths = sorted(options.root.glob(pattern))
    if options.chronological:
        paths.sort(key=lambda path: (path.parent.name.rsplit("-", 1)[0],
                                     int(path.parent.name.rsplit("-", 1)[1]), path.name))
    if options.models:
        models = set(options.models)
        paths = [path for path in paths if path.parent.name.rsplit("-", 1)[0] in models]
    if options.pivot_min is not None:
        paths = [path for path in paths
                 if int(path.parent.name.rsplit("-", 1)[1]) >= options.pivot_min]
    if options.pivot_max is not None:
        paths = [path for path in paths
                 if int(path.parent.name.rsplit("-", 1)[1]) <= options.pivot_max]
    if options.limit is not None:
        if options.limit < 1:
            parser.error("--limit must be positive")
        paths = paths[: options.limit]
    if not paths:
        parser.error(f"no factorization probes under {options.root}")
    raw_output = options.raw_output or options.output.with_name(options.output.stem + "-raw.csv")
    native_times: dict[Path, list[float]] = {path: [] for path in paths}
    rust_times: dict[Path, list[float]] = {path: [] for path in paths}
    metadata: dict[Path, tuple[int, int]] = {}
    # Each native batch completes before its Rust counterpart starts. Neither
    # solver shares a CPU with the other, and matrix I/O is outside both clocks.
    # Retained-LU mode must use one process for the whole chronological stream.
    batch_size = len(paths) if options.retain_factorization else 50
    for trial in range(options.trials):
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            if trial % 2 == 0:
                native = run(options.native, batch, options.repetitions, options.warmups, options.selected_source, options.evict_mib, options.retain_factorization, options.pivot_threshold)
                rust = run(options.rust, batch, options.repetitions, options.warmups, options.selected_source, options.evict_mib, options.retain_factorization, options.pivot_threshold)
            else:
                rust = run(options.rust, batch, options.repetitions, options.warmups, options.selected_source, options.evict_mib, options.retain_factorization, options.pivot_threshold)
                native = run(options.native, batch, options.repetitions, options.warmups, options.selected_source, options.evict_mib, options.retain_factorization, options.pivot_threshold)
            for path in batch:
                n, nnz, native_seconds, native_det = native[path]
                rust_n, rust_nnz, rust_seconds, rust_det = rust[path]
                if (n, nnz) != (rust_n, rust_nnz):
                    raise ValueError(f"different matrix dimensions/nnz: {path}")
                if math.isfinite(native_det) != math.isfinite(rust_det):
                    raise ValueError(f"different determinant finiteness: {path}")
                if math.isinf(native_det) and native_det != rust_det:
                    raise ValueError(f"different infinite determinants: {path}")
                if math.isfinite(native_det) and not math.isclose(
                    native_det, rust_det, rel_tol=1e-8, abs_tol=1e-12
                ):
                    raise ValueError(f"different determinants: {path}: {native_det}, {rust_det}")
                metadata[path] = (n, nnz)
                native_times[path].append(native_seconds / options.repetitions)
                rust_times[path].append(rust_seconds / options.repetitions)
            print(f"trial {trial + 1}/{options.trials}: {min(start + len(batch), len(paths))}/{len(paths)}", flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["probe", "dimension", "nonzeros", "trial", "repetitions", "warmups", "evict_mib", "pivot_threshold", "native_seconds", "rust_seconds", "ratio"])
        for path in paths:
            n, nnz = metadata[path]
            for trial, (native, rust) in enumerate(zip(native_times[path], rust_times[path]), start=1):
                writer.writerow([path.relative_to(options.root), n, nnz, trial, options.repetitions, options.warmups, options.evict_mib, options.pivot_threshold if options.pivot_threshold is not None else "default", f"{native:.12g}", f"{rust:.12g}", f"{rust/native:.9g}"])
    total_native = 0.0
    total_rust = 0.0
    with options.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["probe", "dimension", "nonzeros", "native_seconds", "rust_seconds", "ratio"])
        for path in paths:
            native = statistics.median(native_times[path])
            rust = statistics.median(rust_times[path])
            total_native += native
            total_rust += rust
            n, nnz = metadata[path]
            writer.writerow([path.relative_to(options.root), n, nnz, f"{native:.9g}", f"{rust:.9g}", f"{rust/native:.6g}"])
    trial_totals = [(sum(native_times[path][trial] for path in paths), sum(rust_times[path][trial] for path in paths)) for trial in range(options.trials)]
    for trial, (native, rust) in enumerate(trial_totals, start=1):
        print(f"trial {trial}: native={native:.3f}s rust={rust:.3f}s ratio={rust/native:.3f}")
    model_paths: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        model_paths[path.parent.name.rsplit("-", 1)[0]].append(path)
    model_output = options.output.with_name(options.output.stem + "-models.csv")
    with model_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "matrices", "native_seconds", "rust_seconds", "ratio", "min_trial_ratio", "max_trial_ratio"])
        for model, members in sorted(model_paths.items()):
            native = sum(statistics.median(native_times[path]) for path in members)
            rust = sum(statistics.median(rust_times[path]) for path in members)
            ratios = [sum(rust_times[path][trial] for path in members) / sum(native_times[path][trial] for path in members) for trial in range(options.trials)]
            writer.writerow([model, len(members), f"{native:.9g}", f"{rust:.9g}", f"{rust/native:.6g}", f"{min(ratios):.6g}", f"{max(ratios):.6g}"])
    print(f"{len(paths)} matrices: native={total_native:.3f}s rust={total_rust:.3f}s ratio={total_rust/total_native:.3f}")
    print(options.output)
    print(raw_output)
    print(model_output)


if __name__ == "__main__":
    main()
