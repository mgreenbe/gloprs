# Native reference

| Field | Value |
|---|---|
| Repository | <https://github.com/google/or-tools.git> |
| Commit | `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5` |
| OR-Tools version | 9.15 |
| Local checkout | `../or-tools` |
| Native adapter build | `../or-tools/build-gloprs-solve` |
| License | Apache-2.0 |

The sibling checkout is a read-only reference for this project. An upstream
update must explicitly change the pin, regenerate affected fixtures, and pass
differential validation. Preserve source attribution and upstream path/commit
references in translated modules.

## Building reference adapters

Use a release OR-Tools build with GLOP and MPS support. The existing
`build-gloprs-solve` configuration uses `BUILD_CXX=ON`, with other solver
backends disabled. Its Gurobi option satisfies generic model-builder symbols;
reference adapters explicitly invoke GLOP and do not load Gurobi.
The adapter builder defaults to that directory and accepts `--upstream`,
`--build`, `--source`, and `--output` overrides:

```text
python3 tools/build_glop_reference_adapter.py
python3 tools/run_glop_reference.py --summary ../datasets/netlib/mps/afiro.mps
```

The focused adapter calls `glop::LPSolver` directly and includes variable and
constraint basis statuses. Generic OR-Tools solver responses omit these statuses
and are insufficient for full differential validation.

For a minimal standalone GLOP build, the reference configuration is:

```text
cmake -S ../or-tools -B ../or-tools/build-gloprs-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_DEPS=ON -DBUILD_CXX=OFF \
  -DBUILD_GLOP=ON -DBUILD_SAMPLES=ON -DBUILD_TESTING=ON \
  -DINSTALL_BUILD_DEPS=OFF
```

This minimal configuration supports core examples but does not include
`mps_reader.cc`; use the MPS-capable build for the adapters above. Local native
setup uses CMake, Ninja and a C++ compiler; the original reference used Apple
Clang 17.0.0. Recheck numerical reduction and contraction behavior when changing
compiler or target architecture.

See [CONTRIBUTING.md](CONTRIBUTING.md) for public-solver, stage, trajectory and
KKT checks. The core correspondence is `ortools/glop` → `src` and
`ortools/lp_data` → `lp_data/src`; focused Rust support replaces protobuf,
Abseil, logging and build infrastructure.
