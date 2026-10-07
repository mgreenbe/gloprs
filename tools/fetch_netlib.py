#!/usr/bin/env python3
"""Download and expand the official Netlib LP benchmark corpus."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import urllib.request
from datetime import date
from pathlib import Path


BASE_URL = "https://www.netlib.org/lp/data"
DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "netlib"
TABLE_ROW = re.compile(
    r"^(?P<name>[A-Z0-9.-]+)\s+"
    r"(?P<rows>\d+)\s+"
    r"(?P<columns>\d+)\s+"
    r"(?P<nonzeros>\d+)\s+"
    r"(?P<bytes>\d+|\(see NOTES\))"
    r"(?P<remainder>.*)$"
)


def download(url: str) -> bytes:
    """Return one URL as bytes, with a descriptive user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "gloprs-netlib/0.1"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_catalog(catalog: str) -> list[dict[str, object]]:
    """Parse directly downloadable problems from Netlib's summary table."""
    problems: list[dict[str, object]] = []
    in_table = False

    for line in catalog.splitlines():
        if line.strip() == "PROBLEM SUMMARY TABLE":
            in_table = True
            continue
        if in_table and line.strip() == "BOUND-TYPE TABLE":
            break
        if not in_table:
            continue

        match = TABLE_ROW.match(line.strip())
        if match is None or match["bytes"] == "(see NOTES)":
            continue

        remainder = match["remainder"].strip()
        flags = ""
        if remainder.startswith("BR"):
            flags, remainder = "BR", remainder[2:].strip()
        elif remainder.startswith("B") or remainder.startswith("R"):
            flags, remainder = remainder[0], remainder[1:].strip()

        objective_text = remainder.removesuffix("**").strip()
        objective = None
        if objective_text and objective_text != "(see NOTES)":
            objective = float(objective_text)

        problems.append(
            {
                "name": match["name"].lower(),
                "rows": int(match["rows"]),
                "columns": int(match["columns"]),
                "nonzeros": int(match["nonzeros"]),
                "catalog_compressed_bytes": int(match["bytes"]),
                "has_bounds": "B" in flags,
                "has_ranges": "R" in flags,
                "known_optimal_objective": objective,
            }
        )

    if not problems:
        raise RuntimeError("No Netlib problems found in the official catalog")
    return problems


def write_new_or_verify(path: Path, data: bytes) -> None:
    """Create a dataset file, refusing to replace different existing bytes."""
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"refusing to replace changed dataset file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    catalog_bytes = download(f"{BASE_URL}/readme")
    emps_source = download(f"{BASE_URL}/emps.c")
    problems = parse_catalog(catalog_bytes.decode("ascii"))

    provenance = DATASET_ROOT / "provenance"
    write_new_or_verify(provenance / "netlib-readme.txt", catalog_bytes)
    write_new_or_verify(provenance / "emps.c", emps_source)

    old_manifest_path = DATASET_ROOT / "manifest.json"
    old_manifest: dict[str, dict[str, object]] = {}
    if old_manifest_path.exists():
        old_entries = json.loads(old_manifest_path.read_text(encoding="utf-8"))[
            "problems"
        ]
        old_manifest = {entry["name"]: entry for entry in old_entries}

    entries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="gloprs-netlib-") as temporary:
        temporary_path = Path(temporary)
        source_path = temporary_path / "emps.c"
        executable_path = temporary_path / "emps"
        source_path.write_bytes(emps_source)
        subprocess.run(
            ["cc", "-O2", str(source_path), "-o", str(executable_path)],
            check=True,
        )

        for problem in problems:
            name = str(problem["name"])
            source_url = f"{BASE_URL}/{name}"
            compressed = download(source_url)
            expanded = subprocess.run(
                [str(executable_path)],
                input=compressed,
                check=True,
                capture_output=True,
            ).stdout

            previous = old_manifest.get(name)
            if previous is not None and previous["source_sha256"] != sha256(compressed):
                raise RuntimeError(f"upstream source changed for {name}")

            relative_path = Path("mps") / f"{name}.mps"
            write_new_or_verify(DATASET_ROOT / relative_path, expanded)
            entries.append(
                {
                    **problem,
                    "path": relative_path.as_posix(),
                    "format": "MPS",
                    "source_url": source_url,
                    "source_sha256": sha256(compressed),
                    "source_bytes": len(compressed),
                    "mps_sha256": sha256(expanded),
                    "mps_bytes": len(expanded),
                    "provenance": "Netlib LP/DATA; expanded with official emps.c",
                    "license": "No explicit dataset license stated by Netlib",
                }
            )

    entries.sort(key=lambda entry: str(entry["name"]))
    manifest = {
        "dataset": "Netlib LP test problems",
        "source": f"{BASE_URL}/",
        "retrieved": date.today().isoformat(),
        "catalog_sha256": sha256(catalog_bytes),
        "decompressor_sha256": sha256(emps_source),
        "excluded_generated_problems": ["qap8", "qap12", "qap15", "stocfor3", "truss"],
        "problems": entries,
    }
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    old_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    by_size = sorted(entries, key=lambda entry: (int(entry["mps_bytes"]), entry["name"]))
    subsets = DATASET_ROOT / "subsets"
    subsets.mkdir(parents=True, exist_ok=True)
    for count in (10, 25, 50):
        names = [entry["name"] for entry in by_size[:count]]
        (subsets / f"smallest-{count}.json").write_text(
            json.dumps(names, indent=2) + "\n", encoding="utf-8"
        )
    (subsets / "all.json").write_text(
        json.dumps([entry["name"] for entry in entries], indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"prepared {len(entries)} Netlib problems in {DATASET_ROOT}")


if __name__ == "__main__":
    main()
