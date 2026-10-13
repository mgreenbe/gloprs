"""Manifest-backed model selection shared by differential and timing harnesses."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT.parent / "datasets"
FAST = ROOT / "baselines/netlib-dual-trajectories.json.gz"
MAX_NONZEROS = 150_000
EXTRA_CORPORA = ("miplib2017", "mittelmann-lpopt")
EXTRA_FIXTURE = ROOT / "baselines/extra-lp-corpus.json"


@dataclass(frozen=True)
class Model:
    corpus: str
    name: str
    path: Path
    nonzeros: int

    @property
    def key(self) -> str:
        return f"{self.corpus}/{self.name}"


def models(corpus: str = "netlib-fast96") -> list[Model]:
    """Select MPS files using parser-verified matrix nnz, not objective entries."""
    netlib_manifest = json.loads((DATASETS / "netlib/manifest.json").read_text())
    netlib = {case["name"].casefold() for case in netlib_manifest["problems"]}
    if corpus == "netlib-fast96":
        with gzip.open(FAST, "rt", encoding="utf-8") as stream:
            names = [case["name"] for case in json.load(stream)["results"]]
        nonzeros = {case["name"]: case["nonzeros"] for case in netlib_manifest["problems"]}
        return [Model("netlib", name, DATASETS / "netlib/mps" / f"{name}.mps",
                      nonzeros[name]) for name in names]
    if corpus == "extra":
        return [model for source in EXTRA_CORPORA for model in models(source)]
    if corpus not in EXTRA_CORPORA:
        raise ValueError(f"unknown corpus: {corpus}")
    manifest = json.loads((DATASETS / corpus / "mps_manifest.json").read_text())
    manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}
    fixture = json.loads(EXTRA_FIXTURE.read_text())
    if fixture["cutoff_exclusive"] != MAX_NONZEROS:
        raise ValueError("extra LP corpus cutoff differs from pinned fixture")
    selected = []
    for entry in fixture["models"]:
        if entry["corpus"] != corpus:
            continue
        source = manifest_by_path[entry["path"]]
        if source["sha256"] != entry["sha256"]:
            raise ValueError(f"dataset manifest changed: {entry['path']}")
        path = DATASETS / entry["path"]
        name = path.stem
        if entry["nonzeros"] >= MAX_NONZEROS or name.casefold() in netlib:
            raise ValueError(f"invalid pinned extra LP model: {entry['path']}")
        selected.append(Model(corpus, name, path, entry["nonzeros"]))
    return sorted(selected, key=lambda model: model.name.casefold())
