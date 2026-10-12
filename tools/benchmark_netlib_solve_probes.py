#!/usr/bin/env python3
"""Serial native/Rust sparse FTRAN/BTRAN timing on saved Netlib probes.

Both adapters preload a basis, reproduce LU permutation absorption, and load
each RHS before timing the two sparse solve stages. Output is a per-RHS CSV;
the printed totals sum median per-RHS solve times, never file I/O or LU time.
"""

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
NATIVE = ROOT / "target/native/netlib_solve_probe_reference_adapter"
RUST = ROOT / "target/release/examples/netlib_solve_probe_benchmark"


def groups(root: Path, models: set[str] | None) -> list[tuple[str, Path, Path, list[Path]]]:
    result = []
    for basis in sorted(root.glob("*-*/basis-*.mtx")):
        model = basis.parent.name.rsplit("-", 1)[0]
        if models is not None and model not in models:
            continue
        serial = basis.stem.split("-")[-1]
        factorization = basis.with_name(f"factorization-{serial}.mtx")
        if not factorization.exists():
            raise FileNotFoundError(factorization)
        rhs = sorted(basis.parent.glob(f"ftran-{serial}-*.mtx"))
        rhs += sorted(basis.parent.glob(f"btran-{serial}-*.mtx"))
        if rhs:
            result.append((model, factorization, basis, rhs))
    return result


def run(binary: Path, repetitions: int, warmups: int, factorization: Path,
        basis: Path, rhs: list[Path]) -> dict[str, tuple[str, int, int, float, float, float, float]]:
    command = [str(binary), str(repetitions), str(warmups), str(factorization),
               str(basis), "-"]
    completed = subprocess.run(
        command, input="\n".join(map(str, rhs)) + "\n", text=True,
        capture_output=True, check=True,
    )
    rows = {}
    for line in completed.stdout.splitlines():
        path, kind, n, nnz, reps, nanos, l1, linf, weighted = line.split("\t")
        if int(reps) != repetitions or path in rows:
            raise ValueError(f"duplicate or invalid adapter output: {line}")
        rows[path] = (kind, int(n), int(nnz), float(nanos) / repetitions,
                      float(l1), float(linf), float(weighted))
    if len(rows) != len(rhs) or set(rows) != set(map(str, rhs)):
        raise ValueError(f"{binary}: incomplete RHS output for {basis}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROBES)
    parser.add_argument("--native", type=Path, default=NATIVE)
    parser.add_argument("--rust", type=Path, default=RUST)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--rhs-limit", type=int)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "target/netlib-solve-probe-timings.csv")
    options = parser.parse_args()
    if options.trials < 1 or options.repetitions < 1 or options.warmups < 0:
        parser.error("trials/repetitions must be positive; warmups nonnegative")
    selected = groups(options.root, set(options.models) if options.models else None)
    if options.limit_groups is not None:
        selected = selected[:options.limit_groups]
    if not selected:
        parser.error("no basis/RHS probe groups selected")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    totals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    mismatch_count = 0
    with options.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "basis", "rhs", "kind", "dimension",
                         "rhs_nnz", "native_ns", "rust_ns", "ratio"])
        for number, (model, factorization, basis, rhs) in enumerate(selected, 1):
            if options.rhs_limit is not None:
                rhs = rhs[:options.rhs_limit]
            if not rhs:
                continue
            samples: dict[str, dict[str, list[float]]] = {
                "native": defaultdict(list), "rust": defaultdict(list)
            }
            metadata = {}
            for trial in range(options.trials):
                order = (("native", options.native), ("rust", options.rust))
                if (number + trial) % 2:
                    order = tuple(reversed(order))
                results = {}
                for label, binary in order:
                    results[label] = run(binary, options.repetitions,
                                         options.warmups, factorization, basis, rhs)
                for path in map(str, rhs):
                    native = results["native"][path]
                    rust = results["rust"][path]
                    if native[:3] != rust[:3]:
                        raise ValueError(f"{path}: native/Rust kind, dimension or nnz differ")
                    for a, b in zip(native[4:], rust[4:]):
                        if not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-9):
                            mismatch_count += 1
                            if mismatch_count <= 5:
                                print(f"checksum mismatch: {path}: {a} vs {b}", flush=True)
                    samples["native"][path].append(native[3])
                    samples["rust"][path].append(rust[3])
                    metadata[path] = native[:3]
            for path in map(str, rhs):
                native_ns = statistics.median(samples["native"][path])
                rust_ns = statistics.median(samples["rust"][path])
                kind, dimension, nnz = metadata[path]
                ratio = rust_ns / native_ns if native_ns else math.nan
                writer.writerow([model, str(basis), path, kind, dimension, nnz,
                                 native_ns, rust_ns, ratio])
                totals[kind]["native"].append(native_ns)
                totals[kind]["rust"].append(rust_ns)
                if nnz:
                    totals[kind + "_nonzero"]["native"].append(native_ns)
                    totals[kind + "_nonzero"]["rust"].append(rust_ns)
            if number % 100 == 0 or number == len(selected):
                print(f"[{number}/{len(selected)}] {model}: {len(rhs)} RHS", flush=True)
    for kind, times in sorted(totals.items()):
        native = sum(times["native"])
        rust = sum(times["rust"])
        print(f"{kind}: {len(times['native'])} RHS; native={native / 1e9:.6f}s "
              f"rust={rust / 1e9:.6f}s ratio={rust / native:.3f}")
    print(f"checksum field mismatches={mismatch_count}; CSV={options.output}")
    if mismatch_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
