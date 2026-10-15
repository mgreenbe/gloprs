#!/usr/bin/env python3
"""Check the complete Rust parameter schema/defaults against pinned GLOP."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROTO = (ROOT.parent / "or-tools/ortools/glop/parameters.proto").read_text()
RUST = (ROOT / "src/parameters.rs").read_text()

FIELDS = re.findall(
    r"optional\s+(\w+)\s+(\w+)\s*=\s*\d+\s*\[default\s*=\s*([^]]+)\]",
    PROTO,
)
STRUCT = RUST.split("pub struct GlopParameters {", 1)[1].split("\n}", 1)[0]
DEFAULT = RUST.split("impl Default for GlopParameters {", 1)[1].split("\nimpl GlopParameters", 1)[0]
TYPES = dict(re.findall(r"pub (\w+): (\w+)", STRUCT))
VALUES = dict(re.findall(r"^\s+(\w+): ([^,]+),", DEFAULT, re.M))
assert len(FIELDS) == 59, len(FIELDS)
assert set(TYPES) == set(VALUES) == {name for _, name, _ in FIELDS}

for proto_enum, proto_body in re.findall(r"enum (\w+)\s*\{(.*?)\n\s*\}", PROTO, re.S):
    rust_body = RUST.split(f"pub enum {proto_enum} {{", 1)[1].split("\n}", 1)[0]
    proto_variants = {
        "".join(word.title() for word in name.lower().split("_"))
        for name in re.findall(r"^\s*(\w+)\s*=\s*\d+\s*;", proto_body, re.M)
    }
    rust_variants = set(re.findall(r"^\s*(\w+),\s*$", rust_body, re.M))
    assert proto_variants == rust_variants, (proto_enum, proto_variants, rust_variants)

for proto_type, name, proto_default in FIELDS:
    rust_type = TYPES[name]
    rust_default = VALUES[name]
    if proto_type == "double":
        assert rust_type == "f64", name
        translated = rust_default.replace("f64::INFINITY", "inf").replace("f64::NEG_INFINITY", "-inf")
        assert float(proto_default) == float(translated), (name, proto_default, rust_default)
    elif proto_type in ("int32", "int64"):
        assert rust_type == {"int32": "i32", "int64": "i64"}[proto_type], name
        assert int(proto_default) == int(rust_default), name
    elif proto_type == "bool":
        assert rust_type == "bool" and proto_default == rust_default, name
    else:
        assert rust_type == proto_type, name
        enum_name = "".join(word.title() for word in proto_default.lower().split("_"))
        assert rust_default == f"{proto_type}::{enum_name}", (name, proto_default, rust_default)

print(f"{len(FIELDS)} parameter fields and defaults match pinned GLOP")
