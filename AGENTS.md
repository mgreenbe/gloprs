# AGENTS.md

## Mission

`gloprs` is a Rust port of GLOP, the simplex-based linear-programming solver in
Google OR-Tools. The objective is a working, understandable port with
performance parity, not a new solver inspired by GLOP.

The canonical project repository is <https://github.com/mgreenbe/gloprs.git>.

Translate the pinned upstream source as faithfully as Rust permits:

- preserve file and module boundaries when they remain meaningful;
- preserve algorithms, invariants, parameter defaults, numerical tolerances,
  iteration order, and tie-breaking behavior;
- prefer a direct idiomatic Rust equivalent over a redesign;
- record deliberate divergences from upstream and explain why they are needed;
- remove Google-specific infrastructure such as Bazel, protobuf, Abseil, and
  internal logging/status utilities rather than reproducing it wholesale.

Read `PLAN.md` before beginning substantial work and update it when milestones,
scope, or evidence change.

Keep `PORTING.md` current as the file-level ledger for the port. Add upstream
files and dependencies when they enter scope; update each row's Rust
destination, status, test evidence, divergences, and notes in the same change
that alters the corresponding port. Do not remove completed rows: the table is
the durable correspondence between the pinned upstream tree and the Rust tree.
Use `validated` only when the relevant unit tests and, where solver behavior is
affected, differential evidence against pinned GLOP both exist.

## Repository layout

The intended Cargo workspace mirrors the relevant part of `ortools/`:

- `lp_data/`: LP models, sparse matrix primitives, MPS parsing, validation, and
  model transformations needed by GLOP;
- `glop/`: basis management, sparse linear algebra, revised simplex,
  preprocessing, scaling, parameters, status, and the public solver API;
- `cli/` or `tools/`: small binaries for solving models, differential testing,
  dataset maintenance, and benchmarks;
- `../datasets/netlib/`: shared monorepo-level Netlib LP instances, provenance
  manifest, and named smoke-test subsets;
- `benches/`: reproducible microbenchmarks and end-to-end comparisons;
- `prompts/`: project instructions and historical task prompts.

The sibling directory `../or-tools` is the reference checkout when present. It
is not part of the Rust workspace and must not be modified as part of the port.
Likewise, datasets belong under the shared monorepo directory `../datasets`,
not under `gloprs/`, because other packages may consume the same corpora.

## Upstream fidelity

Before porting a file:

1. Confirm the pinned OR-Tools commit recorded by the project.
2. Read the complete upstream header, implementation, and directly associated
   tests.
3. Identify dependencies outside `ortools/glop` and decide whether they belong
   in `lp_data`, a small local compatibility module, or should be replaced by a
   standard Rust facility.
4. Add the upstream path and commit to the Rust module documentation or the
   project porting manifest.
5. Port tests together with behavior. Do not mark a file complete merely
   because it compiles.

Keep Apache-2.0 attribution and license headers where required. Do not copy
code whose licensing or provenance is unclear.

Names should remain recognizable to someone reading GLOP. Rust naming rules
may change spelling, but avoid gratuitous renaming. Preserve comments that
explain mathematics, numerical safeguards, or non-obvious performance choices;
rewrite C++-specific comments when their Rust equivalent differs.

## Rust implementation policy

- Use stable Rust unless a measured performance requirement justifies a
  documented exception.
- Prefer contiguous `Vec<T>` storage, slices, and reusable workspaces in hot
  paths. Avoid allocation inside simplex iterations unless upstream also does
  comparable work.
- Represent row, column, variable, and constraint indices with distinct
  newtypes where GLOP relies on strong-index semantics. Make conversions
  explicit at API boundaries.
- Preserve sparse traversal order when it affects determinism or pivot choice.
- Do not introduce hashing into deterministic hot paths without controlling
  iteration order and benchmarking the result.
- Keep floating-point comparisons, tolerances, sentinels, and infinity
  semantics aligned with upstream. Never replace a numerical safeguard with a
  more elegant formula without differential evidence.
- Use `unsafe` only when a safe implementation has been measured and shown to
  miss an important target. Isolate it, state its invariants, and test it.
- Protobuf parameter definitions should become ordinary Rust enums and structs
  with the same defaults and semantics. Serialization is secondary to solver
  behavior.
- Replace Abseil/status/logging facilities with small Rust equivalents. Avoid a
  broad compatibility layer that obscures the translated algorithm.

## Testing and validation

Every translated component should have tests at the narrowest useful level:

- port relevant upstream unit tests;
- add invariant tests for sparse structures, permutations, bases, and
  factorization updates;
- use deterministic seeds for randomized tests;
- compare solver status, objective, primal and dual feasibility, reduced costs,
  basis status, and iteration counts against the pinned GLOP executable;
- treat iteration-count or pivot-sequence divergence as evidence to investigate,
  even when the final objective agrees;
- retain minimized regression cases for every discovered discrepancy.

Netlib is the primary end-to-end regression corpus. Smoke subsets must be
derived mechanically from the manifest, not maintained as unrelated copies or
handwritten lists.

Before handing off a change, run the applicable subset of:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Run differential Netlib tests and release benchmarks for changes to numerical
kernels, pivot selection, basis updates, preprocessing, or the simplex loop.

## Performance work

Correctness comes first, but architecture must not preclude parity. Establish a
correct baseline before optimizing, then profile. Record benchmark commands,
compiler versions, build modes, machine details, and upstream commit.

Compare release builds under equivalent conditions. Report distributions and
per-instance regressions, not only aggregate elapsed time. Track at least:

- parse and presolve time;
- solve time and simplex iterations;
- basis factorization and update time;
- peak resident memory;
- objective and residual quality.

Do not accept a faster result produced by different tolerances, disabled checks,
or weaker termination criteria as performance parity.

## Dataset hygiene

Every downloaded instance must be represented in the dataset manifest with its
source URL, checksum, byte size, format, dimensions, nonzero count, known
objective/status when available, and license/provenance notes. Dataset scripts
must be deterministic and safe to rerun. Never silently replace an instance
under an existing identifier.

Treat `../datasets` as shared infrastructure. Put Netlib-specific files under
`../datasets/netlib`, avoid assumptions that only `gloprs` consumes them, and
keep package-specific benchmark output in `gloprs` rather than mixing it with
the shared source corpus.

## Change discipline

- `gloprs` is an independent Git repository rooted at `sparse/gloprs`; do not
  commit sibling checkouts, shared datasets, or monorepo-level files into it.
- Use `https://github.com/mgreenbe/gloprs.git` as the `origin` remote.
- Keep commits small enough to compare meaningfully with upstream files.
- Do not combine a faithful port with an unrelated redesign.
- Preserve user changes and unrelated worktree state.
- Update `PLAN.md` when completing a milestone or discovering a dependency,
  incompatibility, or performance risk.
- Update the corresponding `PORTING.md` rows whenever port status, destination,
  dependencies, test evidence, or deliberate divergences change.
- State what was tested and what remains unverified in every handoff.
