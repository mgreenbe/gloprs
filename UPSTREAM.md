# Upstream reference

`gloprs` targets the following immutable Google OR-Tools revision:

| Field | Value |
|---|---|
| Repository | <https://github.com/google/or-tools.git> |
| Local reference checkout | `../or-tools` |
| Branch at clone time | `stable` |
| Commit | `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5` |
| Commit date | 2026-09-17T15:02:24+02:00 |
| OR-Tools version | 9.15 |
| License | Apache-2.0 |

The reference checkout is intentionally a sibling of this repository. Do not
advance it while translating files. Any future upstream update must change the
commit above explicitly, regenerate the source inventory, and receive its own
differential-validation pass.

## Initial native toolchain

| Tool | Version |
|---|---|
| CMake | 4.4.3 |
| Ninja | 1.13.2 |
| C++ compiler | Apple clang 17.0.0 (`clang-1700.6.4.2`) |

## Native standalone build

The pinned checkout was configured with:

```text
cmake -S . -B build-gloprs-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_DEPS=ON \
  -DBUILD_CXX=OFF \
  -DBUILD_GLOP=ON \
  -DBUILD_SAMPLES=ON \
  -DBUILD_TESTING=ON \
  -DINSTALL_BUILD_DEPS=OFF
```

The MPS-capable generic `solve` reference binary is built separately in
`../or-tools/build-gloprs-solve` with `BUILD_CXX=ON`, all optional solver
backends except GLOP disabled, and `USE_GUROBI=ON` only to satisfy generic
OR-Tools model-builder symbols (the reference invocation explicitly selects
GLOP and does not load Gurobi).

The `simple_glop_program` target builds successfully. Its CTest entry passes
and the executable reports the expected optimal objective value `4`. The
standalone build does not include `mps_reader.cc`; the MPS-capable structured
reference runner therefore remains a separate Phase 0 deliverable.

## Scope notes

The core source lives in `ortools/glop`. GLOP also directly depends on
`ortools/lp_data` and selected facilities from `ortools/base`,
`ortools/algorithms`, `ortools/graph`, `ortools/util`, and generated protobuf
types. The Rust port will translate solver semantics while replacing Bazel,
protobuf, Abseil, logging, status, and flag infrastructure with focused Rust
equivalents.
