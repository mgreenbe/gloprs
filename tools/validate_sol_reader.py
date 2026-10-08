#!/usr/bin/env python3
"""Differentially compare SOL parsing with pinned GLOP."""
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/sol_reader_reference_adapter"
RUST = ROOT / "target/debug/examples/sol_reader_trace"

def run(exe: Path, data: str) -> str:
    return subprocess.run([exe], input=data, text=True, capture_output=True, check=True).stdout

def compare(names: list[str], solution: str) -> None:
    data = f"{len(names)}\n" + "\n".join(name or "<empty>" for name in names) + "\n" + solution
    expected, actual = run(NATIVE, data), run(RUST, data)
    if expected != actual: raise AssertionError(f"{data!r}: {expected!r} != {actual!r}")

def main() -> None:
    count = 0
    fixed = [
        "", "# comment\n", "x 1\n", "x 1junk\n", "x\n", "x 1 2\n",
        "unknown 1\n", "=obj= 5\nx -0\n", "x inf\n", "x -inf\n",
        "x nan\n", "x .5\ny 2e+2 # c\n", "x nope\n", "x 1\nx 2\n",
    ]
    for solution in fixed:
        compare(["x", "y"], solution)
        count += 1
    compare(["", ""], "c0 1\nc1 2\n")
    count += 1
    rng = random.Random(0x501)
    tokens = ["0", "-0", "1", "-2.5", ".25", "3e2", "4e-2junk", "nan", "inf", "bad", "0x1.8p2junk", "1e-400"]
    for _ in range(1000):
        names = [f"x{i}" for i in range(rng.randrange(1, 12))]
        lines = []
        for _ in range(rng.randrange(0, 25)):
            name = rng.choice(names + ["=obj=", "unknown"])
            value = rng.choice(tokens)
            suffix = rng.choice(["", " # comment", " extra"])
            lines.append(f"{name}\t{value}{suffix}")
        compare(names, "\n".join(lines) + "\n")
        count += 1
    for _ in range(250):
        integer = "".join(rng.choice("0123456789abcdef") for _ in range(rng.randrange(1, 25)))
        fraction = "".join(rng.choice("0123456789abcdef") for _ in range(rng.randrange(0, 25)))
        exponent = rng.randrange(-1100, 1100)
        compare(["x"], f"x 0x{integer}.{fraction}p{exponent}\n")
        count += 1
    print(f"{count} SOL-reader cases agree")

if __name__ == "__main__": main()
