# Phase 8 performance comparison: fast Netlib 96

This is a serial, optimized comparison against pinned native GLOP. The corpus
is the 96-model Netlib trajectory fixture, omitting `qap12` and `qap15`.
Every timing case agreed on status and iteration count. The strict pivot-path
regression and the 96-model public-postsolve KKT gate also pass after the
changes described below. Numerical and path fidelity is a gate, not an
optimization variable.

## Environment and method

- Apple M2, macOS 15.6.1 arm64; `rustc 1.95.0`, Apple Clang 17.
- Native OR-Tools commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`;
  initial Rust revision `8f84b1418f6f39eee5f73f916374f094ae853bf6`.
- Release builds, one warm-up and three alternating, serial timed trials per
  solver and model. Each reported model time is the median of its trials.
  Aggregate totals sum those medians. The `final` artifacts were measured
  from the uncommitted working-tree edits described here; their JSON revision
  field identifies the parent commit, not a clean source tree.
- Direct-dual and presolved-solve timers exclude MPS parsing and output.
  Parsing and presolve are timed separately. Public-solver timings include its
  default pipeline. The memory timer instead measures the entire process,
  including parsing and runtime/library startup.

| Timed scope | Initial ratio | Workspace edits | Direct row conversion | Inline dot product |
| --- | ---: | ---: | ---: | ---: |
| Direct unscaled dual solve | 1.086 (23.80 / 25.84 s) | 1.104 (23.74 / 26.22 s) | 1.172 (24.04 / 28.17 s) | not rerun |
| Public default pipeline | 1.347 (9.94 / 13.38 s) | 1.346 (9.68 / 13.04 s) | 1.301 (10.01 / 13.02 s) | 1.264 (10.32 / 13.05 s) |
| Solve after matched presolve | 1.356 (9.51 / 12.90 s) | 1.340 (9.62 / 12.89 s) | 1.291 (9.84 / 12.70 s) | 1.254 (9.81 / 12.31 s) |
| Parse to `LinearProgram` | 1.753 (0.216 / 0.379 s) | 1.756 (0.210 / 0.370 s) | not rerun | not rerun |
| Copy, cleanup, and presolve | 1.467 (0.261 / 0.383 s) | 1.461 (0.255 / 0.372 s) | not rerun | not rerun |

The public gap is overwhelmingly in the presolved simplex solve. Before the
compact-dot edit, its largest absolute per-model excesses were `dfl001`
(+0.88 s), `fit2d` (+0.77 s), and `d6cube` (+0.23 s). Small-model ratios should
not dominate an aggregate: after inlining, the presolved-solve median model
ratio is 1.250, and its sqrt-nonzero-weighted geometric mean is 1.264. The
corresponding public figures are 1.271 and 1.289. The direct-dual total is
strongly influenced by a few large models, so its 1.172 total ratio does not
describe the typical case; its sqrt-nonzero-weighted geometric mean is 1.313.
The direct-dual total also varied much more than the other two scopes across
these runs, warranting a controlled check before attributing its movement to
an implementation change.

The 96-model whole-process memory run used three alternating launches each.
Rust had lower peak RSS in all 96; median per-model Rust/native RSS ratio was
0.194, and the largest observed peaks were 33.3 MiB Rust versus 41.5 MiB
native. The difference includes executable and shared-library/runtime
footprints; it does **not** establish that Rust uses one-fifth as much solver
heap on matched instances.

## Findings and changes

Sampling `stocfor3` in the presolved primal path exposed frequent LU scattered
permutations. GLOP swaps the solved vector with a persistent zero scratchpad;
Rust previously copied the entire dense array. Rust now swaps the underlying
`Vec<f64>` storage, zeros only the old sparse positions when applicable, and
preserves the dense-sentinel behavior. The two-output cached left solve
likewise swaps its dense values instead of copying them. Strongly typed
row/column position lists still need a linear conversion. Separately,
`RightSolveForTau()` now returns a borrow of its persistent workspace instead
of cloning it every dual pivot, matching GLOP's reference-return API.
Focused tests check storage identity and values; six 300-case native
basis-update suites and the fast-96 strict trajectory gate pass.

The workspace changes help the focused `stocfor3` presolved solve, but alone
did not show a decisive full-corpus gain. A safe slice/chunk rewrite of the
compact-column dot product was tried, timed, and reverted because it did not
improve the focused direct-dual benchmark. A fresh presolved `fit2d` profile
put roughly 73% of Rust samples in that scalar product, called from update-row
and primal-edge-norm work. The compact matrix already stores 32-bit row
indices like upstream. In an isolated matched benchmark, Rust was 2.25–2.38×
native at 4–64 entries per column. ARM64 disassembly showed a redundant
signedness check on each stored row in addition to the checked vector gather.
Converting that row directly, as GLOP's `RowToColIndex()` does, retains the
vector bounds check and the exact four-accumulator arithmetic. The isolated
ratio fell to 1.81–2.11×, and the full presolved and public ratios improved as
shown above. The strict 96-model trajectory test and an invalid-negative-row
panic test pass. GLOP declares this kernel inline in `sparse.h`; marking the
Rust counterpart for inlining reduced the isolated ratio further to
1.73–1.83× and produced the final full-corpus presolved/public results in the
table. The final strict trajectory and public KKT gates still pass.

The after-edit direct-dual run instead showed a large `dfl001` slowdown.
Alternating five trials of the old/new Rust binaries on the same machine
measured the new binary faster on `dfl001` (0.955× old), `fit2d` (0.905×), and
`stocfor3` (0.965×). This points to machine variation in the full direct run;
it does not explain or erase that recorded result. More direct-dual sampling is
needed before claiming a whole-corpus gain. The remaining dot-product gap
likely involves checked vector gathers and code generation. The
lower-transpose triangular-solve gap documented in `PLAN.md` remains open.
No performance-parity claim is made yet.

A further layout experiment changed `TriangularMatrix` non-diagonal row storage
from 64-bit `usize` to 32-bit indices, matching native GLOP's storage width.
In the isolated lower-transpose benchmark at dimension 1024, the Rust/native
ratio fell from 1.51 to 1.45 with the same 3% density and 5,000 repetitions.
The full 96-model serial presolved-solve timing, however, is 9.527 seconds
native and 11.941 seconds Rust (1.253×), effectively unchanged from the
preceding 1.254×. The layout improves one kernel but does not account for the
solver-level gap. All 1,000 native triangular and LU traces, strict 96-model pivot
trajectories, and 96-model public postsolve/KKT comparisons pass. Its raw
per-model timing is `phase8-presolved-solve-u32-fast96.json`.

The A/B harness is `tools/benchmark_phase8_ab.py`. For that check, `A` was a
release binary with the checked signed conversion, and `B` differed only by
the direct integer conversion in the compact dot-product kernel. Its printed
per-trial times are reproducible with `--a`, `--b`, and `--models`; the binary
copies are local ignored `target/` artifacts, not source-controlled fixtures.
A second five-trial A/B comparison of the direct-conversion binary with the
inlined binary found the latter 2.5% faster on `dfl001` and 3.6% faster on
`fit2d`, but 1.3% slower on `stocfor3`; therefore the inline change is supported
more strongly by the complete presolved/public suites than by these three
direct-dual cases alone.

## Reproduction and raw results

Build the Rust release examples and the native adapters as described in
`PLAN.md`, then run these serially on the same idle machine:

```sh
python3 tools/benchmark_netlib_solve.py --trials 3 --warmups 1 --output baselines/phase8-direct-dual-dot-fast96.json
python3 tools/benchmark_phase5_netlib.py --mode default --trials 3 --warmups 1 --output baselines/phase8-public-dot-fast96.json
python3 tools/benchmark_phase8_stages.py --stage both --trials 3 --warmups 1 --output baselines/phase8-stages-final-fast96.json
python3 tools/benchmark_phase8_stages.py --stage solve --trials 3 --warmups 1 --output baselines/phase8-presolved-solve-dot-fast96.json
python3 tools/benchmark_phase8_stages.py --stage solve --trials 3 --warmups 1 --output baselines/phase8-presolved-solve-inline-fast96.json
python3 tools/benchmark_phase8_stages.py --stage solve --trials 3 --warmups 1 --output baselines/phase8-presolved-solve-u32-fast96.json
python3 tools/benchmark_phase5_netlib.py --mode default --trials 3 --warmups 1 --output baselines/phase8-public-inline-fast96.json
python3 tools/benchmark_phase8_memory.py --trials 3 --output baselines/phase8-memory-fast96.json
python3 tools/benchmark_compact_dot.py
```

The raw artifacts are `phase8-direct-dual-fast96.json`,
`phase8-public-fast96.json`, `phase8-presolved-solve-fast96.json`, and
`phase8-stages-fast96.json` for initial measurements; the corresponding
`*-final-fast96.json` files for after-workspace-edit runs;
`*-dot-fast96.json` files for after-row-conversion runs;
`*-inline-fast96.json` files for after-inlining runs; and
`phase8-memory-fast96.json`. Each timing JSON includes per-model trial times
and median absolute deviations. The memory JSON includes all three peak-RSS
and footprint trials per model.
