#!/usr/bin/env python3
"""Build the basis-aware adapter against the pinned OR-Tools checkout."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = PROJECT_ROOT.parent / "or-tools"
DEFAULT_BUILD = DEFAULT_UPSTREAM / "build-gloprs-solve"
DEFAULT_OUTPUT = PROJECT_ROOT / "target" / "native" / "glop_reference_adapter"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "tools" / "glop_reference_adapter.cc",
    )
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    args = parser.parse_args()

    source = args.source
    library_dir = args.build / "lib"
    # libortools' public headers instantiate some Abseil templates in the
    # adapter, so those symbols are direct link dependencies too.  Keep one
    # (normally the unversioned symlink) spelling for every built dylib.
    libraries_by_target: dict[Path, Path] = {}
    for library in library_dir.glob("lib*.dylib"):
        target = library.resolve()
        current = libraries_by_target.get(target)
        if current is None or len(library.name) < len(current.name):
            libraries_by_target[target] = library
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "c++",
        "-std=c++20",
        "-O2",
        # The pinned OR-Tools library is a release build.  This is also an ABI
        # requirement for public classes whose debug-only fields are guarded
        # by NDEBUG (notably TimeLimit).
        "-DNDEBUG",
        '-DOR_PROTO_DLL=',
        "-DPROTOBUF_USE_DLLS",
        f"-I{args.upstream}",
        f"-I{args.build}",
        f"-I{args.build / '_deps' / 'protobuf-src' / 'src'}",
        f"-I{args.build / '_deps' / 'protobuf-src' / 'third_party' / 'utf8_range'}",
        f"-I{args.build / '_deps' / 'absl-src'}",
        str(source),
        *(str(path) for path in args.extra_source),
        f"-L{library_dir}",
        "-lortools",
    ]
    command.extend(str(path) for path in sorted(libraries_by_target.values()))
    if platform.system() == "Darwin":
        command.extend([f"-Wl,-rpath,{library_dir}", "-framework", "CoreFoundation"])
    else:
        command.extend([f"-Wl,-rpath={library_dir}"])
    command.extend(["-o", str(args.output)])
    subprocess.run(command, check=True)
    print(args.output)


if __name__ == "__main__":
    main()
