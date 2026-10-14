# Phase 8 indexed-access audit (14 October 2026)

The source inventory covered all Rust modules in `glop/src` and `lp_data/src`
and compared numerical hot-loop indexing with pinned GLOP
(`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`). The search included
direct `[]` reads/writes, iterator-indexed positions, and existing
`get_unchecked`/`SAFETY` sites. A source-level `[]` does not necessarily imply
a runtime check: LLVM can prove many loop bounds. Conversely, upstream C++
`operator[]` is unchecked even at a public API boundary; translating that
literally into a *safe* Rust API could create undefined behavior for callers.

| Region | Decision |
|---|---|
| LU permutations | Already aligned in `69ef17b`: private factorization permutations and retained n-element scratch/output use documented unchecked access. Caller-mutable sparse positions still get one release-mode range check. |
| Triangular solves and compact dot products | Existing unchecked accesses are justified by entry-point dimension checks or validated compact/triangular storage. Retained as is. |
| Primal edge/Devex incremental updates | Removed checked indexing of solver-owned norm/weight vectors for positions returned by the same matrix's `UpdateRow`. The public pivot entry point now checks the leaving row, retained norm/weight lengths, and `UpdateRow` storage width once; this protects the private unchecked loop even if a caller supplies a mismatched `UpdateRow`. |
| Dual edge-norm incremental update | Removed repeated checks for `tau` and norm slots in the n-row loop. The entry point checks direction and leaving-row dimensions; the exact recomputation and tau solve size their vectors to the same basis dimension. |
| Column-wise update-row output | Removed checked writes to retained coefficient and position buffers. The two callers check the relevant bitset width, and each relevant column is visited at most once, bounding the output cursor. |
| Benchmark update-row entry point | Added a width check before the unchecked output writes. A regression test passes a different-width compact matrix and verifies a dimension error. |
| Harris ratio test, rank-one elementary update, pricing public methods | Kept checked access where safe public methods accept caller-supplied indices/slices. Upstream has unchecked `[]`, but Rust cannot make those APIs unsound merely to match syntax. A private, validated kernel could later separate the safe boundary from hot internal calls. |
| Markowitz, sparse matrix builders, parsers, preprocessors, public sparse/scattered-vector methods | Kept checked access for caller-provided or mutable indices and malformed input paths. Some internal loops may have compiler-elided checks; do not substitute unchecked access without a demonstrated dimension invariant and, where appropriate, one-time validation. |

Two serial seven-trial old/new Rust matched-presolved-solve comparisons on
`cvs16r128-89`, `graph20-20-1rand`, and `milo-v12-6-r2-40-1` measured
new/old ratios of 1.006/1.005, 1.005/1.001, and 1.010/1.006 respectively.
Thus these source-fidelity edits are neutral to slightly slower as a group;
they are not a demonstrated performance repair. Raw local measurements are
`target/phase8-bounds-audit-ab{,-repeat}.json`. The direct assertions and
comments are retained to preserve Rust safe-API behavior and document the
internal invariant at each unchecked block.

Validation: `cargo fmt --all --check`, strict workspace Clippy, full workspace
tests, release fast-96 strict native trajectories, and the release extra-model
strict pivot fixture pass. A separate one-pass 213-model public gate matched
all statuses, iteration counts, and objectives within the harness tolerance,
without timeouts. Its raw diagnostic is
`target/phase8-bounds-audit-public-gate.json`; its single-pass timing ratio
is not used as performance evidence.
