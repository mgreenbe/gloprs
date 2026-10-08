#!/usr/bin/env python3
"""Compare SparseRow's specialized API with the pinned GLOP implementation."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(generator: random.Random) -> str:
    n = generator.randint(1, 100)
    columns = generator.sample(range(n), generator.randint(1, n))
    entries = [(column, generator.uniform(-100.0, 100.0)) for column in columns]
    complete = list(range(n))
    generator.shuffle(complete)
    partial = [destination if generator.random() < 0.7 else -1 for destination in complete]
    # Keep at least one stored entry so the first/last wrapper accessors remain defined.
    retained_column = columns[generator.randrange(len(columns))]
    partial[retained_column] = complete[retained_column]
    fields = [str(n), str(len(entries))]
    fields.extend(f"{column} {value:.17g}" for column, value in entries)
    fields.extend(map(str, complete))
    fields.extend(map(str, partial))
    return "\n".join(fields) + "\n"


def run(executable: Path, data: str) -> str:
    return subprocess.run(
        [executable], input=data, text=True, capture_output=True, check=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    generator = random.Random(0x5A0A)
    native = ROOT / "target/native/sparse_row_reference_adapter"
    rust = ROOT / "target/debug/examples/sparse_row_trace"
    for case in range(args.cases):
        data = generate(generator)
        native_output = run(native, data)
        rust_output = run(rust, data)
        if native_output != rust_output:
            raise AssertionError(
                f"case {case}:\ninput:\n{data}GLOP:\n{native_output}gloprs:\n{rust_output}"
            )
    print(f"{args.cases} sparse-row traces agree exactly")


if __name__ == "__main__":
    main()
