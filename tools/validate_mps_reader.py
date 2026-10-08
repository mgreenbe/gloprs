#!/usr/bin/env python3
"""Compare focused MPS parsing behavior with the pinned GLOP reader."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def number(generator: random.Random) -> str:
    return f"{generator.uniform(-20.0, 20.0):.17g}"


def generate(generator: random.Random, case: int) -> str:
    num_rows = generator.randint(1, 8)
    num_columns = generator.randint(1, 8)
    row_kinds = [generator.choice("LEG") for _ in range(num_rows)]
    lines = [f"NAME MODEL{case}"]
    if generator.random() < 0.5:
        if generator.random() < 0.5:
            lines.append(f"OBJSENSE {generator.choice(['MIN', 'MAX'])}")
        else:
            lines.extend(["OBJSENSE", f" {generator.choice(['MIN', 'MAX'])}"])
    lines.append("ROWS")
    has_objective = generator.random() < 0.85
    if has_objective:
        lines.append(" N OBJ")
    if generator.random() < 0.2:
        lines.append(" N FREE")
    lines.extend(f" {kind} R{i}" for i, kind in enumerate(row_kinds))
    lines.append("COLUMNS")
    integer_start = generator.randrange(num_columns + 1)
    integer_end = generator.randrange(integer_start, num_columns + 1)
    for column in range(num_columns):
        if column == integer_start and integer_start != integer_end:
            lines.append(" MARK0000 'MARKER' 'INTORG'")
        pairs: list[str] = []
        candidates = [f"R{i}" for i in range(num_rows)]
        if has_objective:
            candidates.append("OBJ")
        if " N FREE" in lines:
            candidates.append("FREE")
        for row in generator.sample(candidates, generator.randint(1, min(2, len(candidates)))):
            pairs.extend([row, number(generator)])
        lines.append(f" X{column} " + " ".join(pairs))
        if column + 1 == integer_end and integer_start != integer_end:
            lines.append(" MARK0001 'MARKER' 'INTEND'")
    lines.append("RHS")
    for row in range(num_rows):
        lines.append(f" RHS{row % 2} R{row} {number(generator)}")
    if has_objective and generator.random() < 0.5:
        lines.append(f" RHS9 OBJ {number(generator)}")
    if generator.random() < 0.5:
        lines.append("RANGES")
        for row in generator.sample(range(num_rows), generator.randint(0, num_rows)):
            lines.append(f" RNG{row % 2} R{row} {number(generator)}")
    lines.append("BOUNDS")
    for column in range(num_columns):
        kind = generator.choice(["LO", "UP", "FX", "FR", "MI", "PL", "BV", "LI", "UI"])
        if kind in {"LO", "UP", "FX", "LI", "UI"}:
            lines.append(f" {kind} BND{column % 2} X{column} {number(generator)}")
        else:
            lines.append(f" {kind} BND{column % 2} X{column}")
    if generator.random() < 0.9:
        lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def fixed_card(*fields: str) -> str:
    starts = [1, 4, 14, 24, 39, 49]
    widths = [2, 8, 8, 12, 8, 12]
    card = [" "] * 61
    for value, start, width in zip(fields, starts, widths):
        rendered = value[:width]
        card[start : start + len(rendered)] = rendered
    return "".join(card).rstrip()


def generate_fixed(generator: random.Random, case: int) -> str:
    num_rows = generator.randint(1, 5)
    num_columns = generator.randint(1, 5)
    row_kinds = [generator.choice("LEG") for _ in range(num_rows)]
    name = f"F{case}"[:8]
    lines = [f"NAME          {name}", "ROWS", fixed_card("N", "OBJ")]
    lines.extend(fixed_card(kind, f"R{i}") for i, kind in enumerate(row_kinds))
    lines.append("COLUMNS")
    for column in range(num_columns):
        first = generator.randrange(num_rows)
        second = generator.randrange(num_rows) if num_rows > 1 and generator.random() < 0.5 else None
        fields = ["", f"X{column}", f"R{first}", str(generator.randint(-9, 9) or 1)]
        if second is not None:
            fields.extend([f"R{second}", str(generator.randint(-9, 9) or 1)])
        lines.append(fixed_card(*fields))
    lines.append("RHS")
    for row in range(num_rows):
        lines.append(fixed_card("", "RHS", f"R{row}", str(generator.randint(-9, 9))))
    lines.append("BOUNDS")
    for column in range(num_columns):
        kind = generator.choice(["LO", "UP", "FX", "FR", "MI", "PL", "BV"])
        fields = [kind, "BND", f"X{column}"]
        if kind in {"LO", "UP", "FX"}:
            fields.append(str(generator.randint(-9, 9)))
        lines.append(fixed_card(*fields))
    lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def run(executable: Path, source: str, form: str = "auto") -> str:
    return subprocess.run(
        [executable, form], input=source, text=True, capture_output=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x4D5053)
    args = parser.parse_args()
    native = ROOT / "target/native/mps_reader_reference_adapter"
    rust = ROOT / "target/debug/examples/mps_reader_trace"
    generator = random.Random(args.seed)
    focused = [
        "name LOWER\nROWS\n N OBJ\nENDATA\n",
        "NAME OBJNAME\nOBJNAME\n OBJ\nROWS\n N OBJ\nENDATA\n",
        "NAME SENSE\nOBJSENSE\n MINIMIZE\nROWS\n N OBJ\nENDATA\n",
        "NAME SENSE\nOBJSENSE\n max\nROWS\n N OBJ\nENDATA\n",
        "NAME COLUMN\nROWS\n N OBJ\n E R\nCOLUMNS\nX R 1\nENDATA\n",
        "NAME END\nROWS\n N OBJ\n ENDATA\n",
        "NAME BOUND\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n UP B X\nENDATA\n",
        "NAME IGNORED\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n FR B X nonsense\nENDATA\n",
        "NAME NUMBER\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1D2\nENDATA\n",
        "NAME UNKNOWN_ROW\nROWS\n N OBJ\nCOLUMNS\n X NEW_ROW 2\nRHS\n RHS NEW_ROW 3\nENDATA\n",
        "NAME LAZY\nROWS\n N OBJ\nLAZYCONS\n L CUT\nCOLUMNS\n X CUT 2\nENDATA\n",
        "NAME EMPTY_INDICATORS\nROWS\n N OBJ\nINDICATORS\nENDATA\n",
        "NAME INDICATOR\nROWS\n N OBJ\n E R\nCOLUMNS\n X R 1\nINDICATORS\n IF R X 1\nENDATA\n",
        "NAME          A B\nROWS\n N  OBJ\nENDATA\n",
        "NAME NEGZERO\nROWS\n N OBJ\n E R\nCOLUMNS\n X OBJ -0 R -0\nRHS\n RHS R -0\nENDATA\n",
        "NAME UNDERFLOW\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1e-999\nENDATA\n",
        "NAME INFINITY\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1e309\nENDATA\n",
    ]
    for case, source in enumerate(focused):
        expected = run(native, source)
        actual = run(rust, source)
        if expected != actual:
            raise AssertionError(
                f"focused case {case}: expected {expected!r}, got {actual!r}\n{source}"
            )
    for case in range(args.cases):
        for form, source in (
            ("free", generate(generator, case)),
            ("fixed", generate_fixed(generator, case)),
        ):
            for requested in ("auto", "free", "fixed"):
                expected = run(native, source, requested)
                actual = run(rust, source, requested)
                if expected != actual:
                    raise AssertionError(
                        f"{form}/{requested} case {case}: expected {expected!r}, "
                        f"got {actual!r}\n{source}"
                    )
    print(f"{args.cases} free and {args.cases} fixed MPS-reader traces agree")


if __name__ == "__main__":
    main()
