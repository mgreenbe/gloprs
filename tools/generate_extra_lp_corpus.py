#!/usr/bin/env python3
"""Pin parser-verified matrix nnz for the small non-Netlib LP corpora.

Dataset manifest nnz includes objective coefficients; the threshold here is
on the actual constraint matrix. The manifest value minus column count is a
safe lower bound used to avoid parsing giant ineligible models.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

from lp_corpus import DATASETS, EXTRA_CORPORA, MAX_NONZEROS, ROOT


NATIVE = ROOT / "target/native/full_presolve_netlib_reference_adapter"
RUST = ROOT / "target/release/examples/full_presolve_netlib"
OUTPUT = ROOT / "baselines/extra-lp-corpus.json"


def parse_stats(executable: Path, path: Path) -> dict[str, int]:
    output = subprocess.run([executable, path, "parse_timing"], check=True,
                            capture_output=True, text=True, timeout=60).stdout.split()
    fields = dict(zip(output[::2], output[1::2], strict=True))
    return {field: int(fields[field]) for field in ("rows", "columns", "nonzeros")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    netlib_manifest = json.loads((DATASETS / "netlib/manifest.json").read_text())
    netlib_names = {case["name"].casefold() for case in netlib_manifest["problems"]}
    selected = []
    for corpus in EXTRA_CORPORA:
        manifest = json.loads((DATASETS / corpus / "mps_manifest.json").read_text())
        for entry in manifest["files"]:
            path = DATASETS / entry["path"]
            name = path.stem
            if (name.casefold() in netlib_names
                    or entry["nonzeros"] - entry["columns"] >= MAX_NONZEROS):
                continue
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"dataset checksum differs from manifest: {path}")
            native = parse_stats(NATIVE, path)
            rust = parse_stats(RUST, path)
            if native != rust:
                raise RuntimeError(f"{corpus}/{name}: parser mismatch: {native} != {rust}")
            if native["nonzeros"] >= MAX_NONZEROS:
                continue
            selected.append({"corpus": corpus, "name": name, "path": entry["path"],
                             "sha256": entry["sha256"],
                             "manifest_nonzeros": entry["nonzeros"], **native})
    selected.sort(key=lambda case: (EXTRA_CORPORA.index(case["corpus"]),
                                    case["name"].casefold()))
    payload = {
        "description": "Non-Netlib MPS instances with fewer than 150000 constraint-matrix entries",
        "cutoff_exclusive": MAX_NONZEROS,
        "count": len(selected),
        "native_adapter": "tools/full_presolve_netlib_reference_adapter.cc",
        "models": selected,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned {len(selected)} models: {OUTPUT}")


if __name__ == "__main__":
    main()
