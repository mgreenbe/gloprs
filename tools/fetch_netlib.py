#!/usr/bin/env python3
"""Download, generate, index, and verify the official Netlib LP corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

BASE_URL = "https://www.netlib.org/lp/data"
QAP_URL = "https://www.netlib.org/lp/generators/qap"
DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "netlib"
TABLE_ROW = re.compile(
    r"^(?P<name>[A-Z0-9.-]+)\s+(?P<rows>\d+)\s+(?P<columns>\d+)\s+"
    r"(?P<nonzeros>\d+)\s+(?P<bytes>\d+|\(see NOTES\))(?P<remainder>.*)$"
)
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?$")
EMPTY_RHS = {"bore3d", "cycle", "greenbea", "greenbeb", "kb2", "recipe", "tuff"}
EXTRA_RHS_REMOVED = {"beaconfd", "brandy", "fffff800", "israel"}
EXTRA_BOUNDS_REMOVED = {"greenbea", "greenbeb", "grow15", "grow22", "grow7", "recipe"}
EXTRA_FREE_ROWS_REMOVED = {
    "80bau3b", "boeing1", "bore3d", "e226", "fffff800", "finnis",
    "forplan", "ganges", "greenbea", "greenbeb", "maros", "pilot",
    "pilot87", "recipe", "sctap1", "sctap2", "sctap3", "share2b",
    "ship04l", "ship04s", "ship08l", "ship08s", "ship12l", "ship12s",
}
EXPLICIT_ZEROS_REMOVED = {
    "grow15", "grow22", "grow7", "nesm", "scorpion", "scrs8", "seba",
    "sierra", "stair",
}
NEGATED_OBJECTIVE = {
    "boeing1", "boeing2", "degen2", "degen3", "etamacro", "fit1d",
    "fit2d", "ganges", "grow15", "grow22", "grow7", "lotfi", "maros",
    "pilot", "pilot.ja", "pilot.we", "pilotnov", "sc105", "sc50a",
    "sc50b", "stair",
}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "gloprs-netlib/0.1"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_catalog(catalog: str) -> list[dict[str, object]]:
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
        if match is None:
            continue
        remainder = match["remainder"].strip()
        flags = ""
        if remainder.startswith("BR"):
            flags, remainder = "BR", remainder[2:].strip()
        elif remainder.startswith(("B", "R")):
            flags, remainder = remainder[0], remainder[1:].strip()
        objective_text = remainder.removesuffix("**").strip()
        objective = None if not objective_text or objective_text == "(see NOTES)" else float(objective_text)
        byte_text = match["bytes"]
        problems.append(
            {
                "name": match["name"].lower(),
                "rows": int(match["rows"]),
                "columns": int(match["columns"]),
                "nonzeros": int(match["nonzeros"]),
                "catalog_compressed_bytes": None if byte_text == "(see NOTES)" else int(byte_text),
                "has_bounds": "B" in flags,
                "has_ranges": "R" in flags,
                "objective_sense": "minimize",
                "published_status": "optimal" if objective is not None else "not_reported",
                "known_optimal_objective": objective,
            }
        )
    if not problems:
        raise RuntimeError("No Netlib problems found in the official catalog")
    return problems


def write_new_or_verify(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to replace changed dataset file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_shar(bundle: bytes, destination: Path) -> None:
    """Extract Netlib's historical shell archive without executing it."""
    lines = bundle.decode("ascii").splitlines(keepends=True)
    start = re.compile(r"^sed >([^ ]+) <<'([^']+)' 's/\^-//'$")
    position = 0
    while position < len(lines):
        match = start.match(lines[position].rstrip("\n"))
        if match is None:
            position += 1
            continue
        filename, marker = match.groups()
        if Path(filename).name != filename:
            raise RuntimeError(f"unsafe filename in Netlib bundle: {filename}")
        position += 1
        content: list[str] = []
        while position < len(lines) and lines[position].rstrip("\n") != marker:
            line = lines[position]
            content.append(line[1:] if line.startswith("-") else line)
            position += 1
        if position == len(lines):
            raise RuntimeError(f"unterminated file in Netlib bundle: {filename}")
        (destination / filename).write_text("".join(content), encoding="ascii")
        position += 1


def run(command: list[str], cwd: Path, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, cwd=cwd, check=True, **kwargs)  # type: ignore[arg-type]


def generate_problems(
    temporary: Path, emps: Path, provenance: Path
) -> dict[str, tuple[bytes, list[dict[str, object]], str]]:
    generated: dict[str, tuple[bytes, list[dict[str, object]], str]] = {}
    qap = temporary / "qap"
    qap.mkdir()
    qap_sources = []
    for filename in ("newlp.f", "data.8", "data.12", "data.15"):
        url = f"{QAP_URL}/{filename}"
        data = download(url)
        write_new_or_verify(provenance / "qap" / filename, data)
        (qap / filename).write_bytes(data)
        qap_sources.append({"url": url, "sha256": sha256(data), "bytes": len(data)})
    run(["gfortran", "-std=legacy", "-O2", "newlp.f", "-o", "qapgen"], qap)
    for size in (8, 12, 15):
        result = run(
            [str(qap / "qapgen")], qap,
            input=(qap / f"data.{size}").read_bytes(), capture_output=True,
        )
        generated[f"qap{size}"] = (
            result.stdout, qap_sources,
            f"Generated by official Terri Johnson newlp.f using data.{size}",
        )

    for name in ("truss", "stocfor3"):
        url = f"{BASE_URL}/{name}"
        bundle = download(url)
        write_new_or_verify(provenance / f"{name}.shar", bundle)
        source = [{"url": url, "sha256": sha256(bundle), "bytes": len(bundle)}]
        directory = temporary / name
        directory.mkdir()
        extract_shar(bundle, directory)
        if name == "truss":
            run(["gfortran", "-std=legacy", "-O2", "truss.f", "-o", "trussgen"], directory)
            run([str(directory / "trussgen")], directory, capture_output=True)
            generated[name] = (
                (directory / "mps").read_bytes(), source,
                "Generated by official Michael Ferris truss.f and bundled data",
            )
            continue

        expanded_core = run(
            [str(emps)], directory, input=(directory / "core.mpc").read_bytes(), capture_output=True,
        ).stdout
        (directory / "core.mps").write_bytes(expanded_core)
        for filename in ("std2mps.f", "input.f"):
            path = directory / filename
            source_text = path.read_text(encoding="ascii")
            path.write_text(source_text.replace("IABS(", "ABS("), encoding="ascii")
        run(
            ["gfortran", "-std=legacy", "-fallow-argument-mismatch", "-O2",
             "std2mps.f", "input.f", "-o", "stocgen"],
            directory,
        )
        shutil.copyfile(directory / "time7.frs", directory / "fort.1")
        shutil.copyfile(directory / "core.mps", directory / "fort.2")
        shutil.copyfile(directory / "stoch3.frs", directory / "fort.3")
        run([str(directory / "stocgen")], directory, capture_output=True)
        mps = b"".join((directory / f"fort.{unit}").read_bytes() for unit in range(11, 16))
        generated[name] = (
            mps, source,
            "Generated by official Gassmann sources and stoch3 data; IABS calls "
            "changed to generic ABS for modern gfortran integer-kind checking",
        )
    return generated


def caveats(name: str, mps: bytes) -> list[str]:
    result = []
    for names, description in (
        (EMPTY_RHS, "empty RHS section"),
        (EXTRA_RHS_REMOVED, "additional RHS vectors omitted by Netlib"),
        (EXTRA_BOUNDS_REMOVED, "additional bound sets omitted by Netlib"),
        (EXTRA_FREE_ROWS_REMOVED, "additional free rows omitted by Netlib"),
        (EXPLICIT_ZEROS_REMOVED, "explicit zeros removed by Netlib"),
        (NEGATED_OBJECTIVE, "objective negated by Netlib to make a minimization problem"),
    ):
        if name in names:
            result.append(description)
    if name in {"boeing1", "boeing2"}:
        result.append("integer markers removed by Netlib; treated as a continuous LP")
    if b"'MARKER'" in mps or b" MARKER " in mps:
        result.append("contains integer markers")
    if name == "standgub":
        result.append("contains nonstandard GUB marker rows EGROUP and ENDX")
    if name == "lotfi":
        result.append("aggregates seven objectives into one objective")
    return result


def numerical_metrics(mps: bytes, rows: int, columns: int, nonzeros: int) -> dict[str, object]:
    magnitudes = []
    for token in mps.decode("ascii").split():
        if NUMBER.match(token):
            value = abs(float(token.replace("D", "E").replace("d", "e")))
            if value != 0.0 and math.isfinite(value):
                magnitudes.append(value)
    minimum = min(magnitudes) if magnitudes else None
    maximum = max(magnitudes) if magnitudes else None
    return {
        "structural_density": nonzeros / (rows * columns),
        "numeric_abs_min": minimum,
        "numeric_abs_max": maximum,
        "numeric_dynamic_range": maximum / minimum if minimum and maximum else None,
    }


def write_subsets(entries: list[dict[str, object]]) -> None:
    subsets = DATASET_ROOT / "subsets"
    subsets.mkdir(parents=True, exist_ok=True)
    by_size = sorted(entries, key=lambda entry: (int(entry["mps_bytes"]), str(entry["name"])))
    for count in (10, 25, 50):
        names = [entry["name"] for entry in by_size[:count]]
        (subsets / f"smallest-{count}.json").write_text(
            json.dumps(names, indent=2) + "\n", encoding="utf-8"
        )
    (subsets / "all.json").write_text(
        json.dumps([entry["name"] for entry in entries], indent=2) + "\n", encoding="utf-8"
    )

    candidates = by_size[:50]
    groups = [
        by_size[:5],
        sorted(candidates, key=lambda entry: (float(entry["structural_density"]), str(entry["name"])))[:3],
        sorted(candidates, key=lambda entry: (-float(entry["structural_density"]), str(entry["name"])))[:3],
        sorted(candidates, key=lambda entry: (-float(entry["numeric_dynamic_range"]), str(entry["name"])))[:3],
    ]
    selected: list[str] = []
    for group in groups:
        for entry in group:
            name = str(entry["name"])
            if name not in selected:
                selected.append(name)
    for predicate in (
        lambda entry: bool(entry["has_bounds"]),
        lambda entry: bool(entry["has_ranges"]),
        lambda entry: "empty RHS section" in entry["caveats"],
        lambda entry: "contains nonstandard GUB marker rows EGROUP and ENDX" in entry["caveats"],
        lambda entry: entry["generation"] is not None,
    ):
        match = next((entry for entry in by_size if predicate(entry)), None)
        if match is not None and match["name"] not in selected:
            selected.append(str(match["name"]))
    (subsets / "representative-small.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    catalog_bytes = download(f"{BASE_URL}/readme")
    emps_source = download(f"{BASE_URL}/emps.c")
    problems = parse_catalog(catalog_bytes.decode("ascii"))
    provenance = DATASET_ROOT / "provenance"
    write_new_or_verify(provenance / "netlib-readme.txt", catalog_bytes)
    write_new_or_verify(provenance / "emps.c", emps_source)
    old_manifest_path = DATASET_ROOT / "manifest.json"
    old_manifest = {}
    if old_manifest_path.exists():
        old_entries = json.loads(old_manifest_path.read_text(encoding="utf-8"))["problems"]
        old_manifest = {entry["name"]: entry for entry in old_entries}

    entries = []
    with tempfile.TemporaryDirectory(prefix="gloprs-netlib-") as temporary_name:
        temporary = Path(temporary_name)
        emps_source_path = temporary / "emps.c"
        emps = temporary / "emps"
        emps_source_path.write_bytes(emps_source)
        run(["cc", "-O2", str(emps_source_path), "-o", str(emps)], temporary)
        generated = generate_problems(temporary, emps, provenance)
        for problem in problems:
            name = str(problem["name"])
            generation = None
            if name in generated:
                expanded, sources, generation = generated[name]
                source_url = str(sources[0]["url"])
                source_hash = sha256(b"".join(bytes.fromhex(str(item["sha256"])) for item in sources))
                source_bytes = sum(int(item["bytes"]) for item in sources)
            else:
                source_url = f"{BASE_URL}/{name}"
                compressed = download(source_url)
                expanded = run([str(emps)], temporary, input=compressed, capture_output=True).stdout
                sources = [{"url": source_url, "sha256": sha256(compressed), "bytes": len(compressed)}]
                source_hash = sha256(compressed)
                source_bytes = len(compressed)
            previous = old_manifest.get(name)
            if previous is not None:
                if previous["source_sha256"] != source_hash:
                    raise RuntimeError(f"upstream source changed for {name}")
                if previous["mps_sha256"] != sha256(expanded):
                    raise RuntimeError(f"generated MPS changed for {name}")
            relative_path = Path("mps") / f"{name}.mps"
            write_new_or_verify(DATASET_ROOT / relative_path, expanded)
            entries.append(
                {
                    **problem,
                    **numerical_metrics(expanded, int(problem["rows"]), int(problem["columns"]), int(problem["nonzeros"])),
                    "path": relative_path.as_posix(), "format": "MPS",
                    "source_url": source_url, "source_sha256": source_hash,
                    "source_sha256_kind": (
                        "SHA-256 of source bytes" if generation is None
                        else "SHA-256 of concatenated binary source SHA-256 digests"
                    ),
                    "source_bytes": source_bytes,
                    "generator_sources": sources if generation is not None else None,
                    "generation": generation,
                    "mps_sha256": sha256(expanded), "mps_bytes": len(expanded),
                    "has_integer_markers": b"'MARKER'" in expanded or b" MARKER " in expanded,
                    "unsupported_constructs": ["GUB markers"] if name == "standgub" else [],
                    "caveats": caveats(name, expanded),
                    "provenance": "Netlib LP collection",
                    "license": "No explicit dataset license stated by Netlib",
                }
            )

    entries.sort(key=lambda entry: str(entry["name"]))
    manifest = {
        "dataset": "Netlib LP test problems", "source": f"{BASE_URL}/",
        "retrieved": date.today().isoformat(),
        "objective_sense_note": "Netlib states that all distributed forms are minimization problems",
        "catalog_sha256": sha256(catalog_bytes),
        "decompressor_sha256": sha256(emps_source),
        "generator_toolchain": subprocess.run(
            ["gfortran", "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "subset_definitions": {
            "smallest-10/25/50": "lowest expanded MPS byte size, then canonical name",
            "all": "canonical name order",
            "representative-small": (
                "union of five smallest; three lowest-density, three highest-density, "
                "and three widest-numeric-range members of the 50 smallest; then the "
                "smallest examples with bounds, ranges, empty RHS, GUB markers, and generation"
            ),
        },
        "problems": entries,
    }
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    old_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_subsets(entries)
    (DATASET_ROOT / "README.md").write_text(
        "# Netlib LP corpus\n\n"
        "This shared corpus is generated from the official Netlib LP collection by "
        "`gloprs/tools/fetch_netlib.py`. It contains expanded MPS files, original "
        "generator provenance, per-source and per-MPS SHA-256 checksums, catalog "
        "metadata, and mechanically derived subsets.\n\n"
        "All distributed models are minimization problems. Netlib states no explicit "
        "dataset license; consult `provenance/netlib-readme.txt` and each manifest "
        "entry before redistribution. Regeneration requires a C compiler and "
        "`gfortran`; the exact generator compiler is recorded in `manifest.json`.\n",
        encoding="utf-8",
    )
    print(f"prepared {len(entries)} Netlib problems in {DATASET_ROOT}")


if __name__ == "__main__":
    main()
