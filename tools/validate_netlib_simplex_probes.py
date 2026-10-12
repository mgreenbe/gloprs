#!/usr/bin/env python3
"""Validate Matrix Market syntax and basis/RHS grouping in generated probes."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1].parent / "datasets/netlib-simplex-probes"
FILE = re.compile(r"(factorization|basis|ftran|btran)-(\d{4})(?:-(\d{6}))?\.mtx")


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open(encoding="ascii") as stream:
        assert stream.readline() == "%%MatrixMarket matrix coordinate real general\n", path
        rows, columns, count = map(int, stream.readline().split())
        assert rows > 0 and columns > 0 and count >= 0, path
        for line in range(count):
            row, column, value = stream.readline().split()
            assert 1 <= int(row) <= rows and 1 <= int(column) <= columns, (path, line)
            assert math.isfinite(float(value)), (path, line)
        assert not stream.readline(), (path, "extra entries")
    return rows, columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    options = parser.parse_args()
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for directory in sorted(options.root.iterdir()):
        if not directory.is_dir():
            continue
        name, pivot = directory.name.rsplit("-", 1)
        assert pivot.isdecimal(), directory
        groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
        for path in directory.iterdir():
            match = FILE.fullmatch(path.name)
            assert match is not None, path
            kind, serial, ordinal = match.groups()
            assert (ordinal is None) == (kind in {"factorization", "basis"}), path
            groups[serial][kind].append(path)
        for serial, files in groups.items():
            assert len(files["factorization"]) == 1, (directory, serial)
            n, columns = matrix_shape(files["factorization"][0])
            assert n == columns, files["factorization"][0]
            rhs_files = files["ftran"] + files["btran"]
            assert len(files["basis"]) == (1 if rhs_files else 0), (directory, serial)
            if rhs_files:
                assert matrix_shape(files["basis"][0]) == (n, n), files["basis"][0]
            for rhs in rhs_files:
                assert matrix_shape(rhs) == (n, 1), rhs
            for index, kind in enumerate(("factorization", "basis", "ftran", "btran")):
                totals[name][index] += len(files[kind])
    for name, (factorizations, bases, ftran, btran) in sorted(totals.items()):
        print(f"{name}: factorization={factorizations} basis={bases} ftran={ftran} btran={btran}")


if __name__ == "__main__":
    main()
