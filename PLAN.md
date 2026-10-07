# gloprs implementation plan

## Goal and success criteria

Port the GLOP linear-programming solver from Google OR-Tools to Rust while
remaining close enough to the C++ implementation that the two codebases can be
read side by side. Remove Google-specific build and support infrastructure, but
preserve solver behavior and prioritize performance parity.

The project is successful when:

1. `gloprs` parses and solves the complete chosen Netlib corpus.
2. Its statuses and objectives agree with the pinned GLOP reference within the
   same feasibility and optimality tolerances.
3. Primal, dual, reduced-cost, and complementary-slackness residuals pass
   independent validation.
4. Warm starts, basis import/export, limits, and important parameter choices
   behave compatibly.
5. Release-build time and memory are measured reproducibly against GLOP, and
   remaining gaps are localized and documented.
6. The Rust source retains a documented file-level correspondence with the
   pinned upstream source.

## Guiding decisions

- Target the OR-Tools `stable` branch initially, but pin one exact commit before
  translating code. Upstream movement must not silently change the target.
- Mirror the relevant `ortools/` structure as Cargo crates, chiefly `lp_data`
  and `glop`.
- Port bottom-up along dependency boundaries rather than beginning with the
  top-level solver.
- Prefer direct translation first. Optimize only against profiles and preserve
  a differential test for every behavior-changing fix.
- Maintain a buildable, tested workspace at each milestone.
- Treat Netlib metadata, expected results, and benchmark outputs as versioned
  test infrastructure rather than ad hoc local files.
- Store reusable datasets at the monorepo level under `../datasets` rather than
  inside `gloprs`; other packages in `sparse` may consume the same corpora.

## Project bootstrap

Initialize `gloprs` as its own Git repository before downloading or generating
anything:

- run `git init` in `sparse/gloprs`;
- configure `origin` as `https://github.com/mgreenbe/gloprs.git` and verify the
  intended default branch before the first push;
- add an appropriate `.gitignore` before any build or download step;
- commit `AGENTS.md`, `PLAN.md`, the project prompt, licensing files, and the
  initial workspace scaffold as the reproducible project baseline;
- keep the sibling `../or-tools` checkout, shared `../datasets`, Cargo build
  output, local profiles, and benchmark scratch data out of this repository.

Exit criterion:

- `gloprs` is an independent Git repository with a clean working tree and an
  initial baseline commit.

## Phase 0: pin and inventory upstream

Status: in progress. OR-Tools 9.15 commit
`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5` is pinned, the standalone GLOP
sample builds and passes, the initial source inventory is in `PORTING.md`, and
`tools/run_glop_reference.py` provides normalized JSON results from a native
MPS solve. Basis extraction is not exposed by OR-Tools' generic `solve` binary
and remains to be added through a focused native adapter.

Deliverables:

- Create or clone `../or-tools` from the official Google OR-Tools repository.
- Record the remote URL, exact commit SHA, branch/tag, OR-Tools version, license,
  C++ compiler, and build flags in an upstream manifest.
- Build the native GLOP solver and run its relevant unit tests.
- Produce an inventory of `ortools/glop`, the required portion of
  `ortools/lp_data`, and any dependencies in `ortools/base` or utility modules.
- Create a porting table with one row per upstream source/test file and columns
  for Rust destination, dependency status, test status, divergences, and notes.
- Build a minimal reference CLI that accepts an MPS file and emits
  machine-readable status, objective, iterations, timings, residuals, and basis
  information.

Exit criteria:

- The upstream SHA is immutable and recorded.
- The reference CLI solves at least one small LP reproducibly.
- Every initially relevant GLOP file is accounted for in the porting table.

## Phase 1: establish the Rust workspace

Create a Cargo workspace with:

- `lp_data`: model types, typed indices, sparse vectors/matrices, and readers;
- `glop`: numerical kernels and solver implementation;
- `cli`: MPS solving and structured result output;
- `tools`: dataset and differential-test utilities if these do not fit the CLI;
- benchmark targets for kernels and end-to-end solves.

Set project-wide policy:

- pinned Rust toolchain and edition;
- `rustfmt` and strict Clippy;
- debug and release profiles, including an explicitly documented benchmark
  profile;
- CI commands and supported platforms;
- Apache-2.0 licensing and upstream attribution;
- deterministic test seeds and floating-point comparison helpers.

Avoid choosing broad frameworks prematurely. Add dependencies only for a clear
need such as CLI parsing, error reporting, serialization, checksums, or
benchmarking.

Exit criteria:

- All workspace checks pass in CI and locally.
- A placeholder CLI and library API compile.
- The porting table maps upstream modules to workspace crates.

## Phase 2: create the Netlib test corpus

Status: in progress. `tools/fetch_netlib.py` reproducibly downloads and expands
93 directly published Netlib problems into `../datasets/netlib`, records source
and expanded checksums plus catalog metadata, and generates the 10, 25, 50, and
full subsets. Generated-only instances, a representative numerical subset, and
full native GLOP baselines remain.

Write a deterministic downloader/indexer that places the Netlib `.mps` files
under the shared monorepo path `../datasets/netlib/` (that is,
`sparse/datasets/netlib/`) and produces a manifest there containing:

- canonical instance name and relative path;
- source URL and retrieval date;
- SHA-256 checksum and compressed/uncompressed byte sizes;
- row count, column count, and structural nonzero count;
- objective sense;
- published best-known status and objective when available;
- parsing caveats, ranges, integer markers, or unsupported constructs;
- license and provenance notes.

Generate named subsets from the manifest:

- 10 smallest instances;
- 25 smallest instances;
- 50 smallest instances;
- representative small instances by sparsity and numerical characteristics;
- full Netlib corpus.

Represent subsets as manifests or ordered identifier lists beneath
`../datasets/netlib/`; do not copy instances into `gloprs` or duplicate the MPS
files for each subset. Keep GLOP/gloprs-specific run results and performance
reports within the `gloprs` project, since those are not shared source data.

Define “smallest” explicitly, initially by uncompressed MPS byte size, and keep
the derivation script and resulting ordered IDs in version control. Do not make
duplicate physical copies of an instance merely to form a subset.

Run the pinned GLOP reference CLI over every instance and store normalized
baseline metadata: status, objective, iterations, residuals, timings, and peak
memory where practical.

Exit criteria:

- Downloads are checksum-verified and reproducible.
- The reference GLOP baseline exists for every parseable instance.
- Smoke subsets can be regenerated exactly from the manifest.

## Phase 3: port `lp_data` foundations

Port the minimum model and sparse-data layer needed by GLOP:

1. Strong row, column, variable, and constraint index types.
2. Dense typed vectors and permutations.
3. Sparse vectors and column-oriented sparse matrices.
4. Scattered-vector workspaces and nonzero-pattern tracking.
5. Linear-program model representation: bounds, objective, names, scaling
   metadata, and basis status.
6. Model validation and canonicalization.
7. Fixed/free MPS parsing, including bounds, ranges, objective sense, duplicate
   entries, and numerical edge cases used in Netlib.
8. Deterministic model and solution summaries for differential testing.

Port the associated upstream tests before adding solver behavior. Add parser
golden tests and round-trip or normalization tests where exact text emission is
not an upstream requirement.

Exit criteria:

- Every smoke-subset MPS file parses.
- Parsed dimensions, nonzeros, bounds, and objectives agree with GLOP.
- Sparse primitive tests and microbenchmarks are established.

## Phase 4: port numerical and basis kernels

Port the hot foundations in dependency order, preserving upstream file
boundaries where practical:

- permutations and sparse matrix views;
- dense/scattered vector operations;
- triangular solves and transpose solves;
- LU factorization, Markowitz choices, singularity handling, and refactorization;
- basis representation and basis-factorization updates;
- update-row computation;
- primal and dual edge norms;
- numerical accuracy checks and condition/error estimates.

For each kernel:

- port upstream unit tests;
- compare against dense reference calculations on small randomized problems;
- add adversarial singular, nearly singular, duplicate, and badly scaled cases;
- benchmark allocations, traversal counts, and runtime against the C++ kernel;
- verify deterministic pivot and permutation choices where upstream promises or
  relies on them.

Exit criteria:

- Randomized and adversarial residual tests pass.
- Basis solves and updates agree with fresh factorization.
- Kernel benchmarks identify no unexplained order-of-magnitude regression.

## Phase 5: port simplex state and pivot mechanics

Port the components that maintain revised-simplex state:

- variable information, bounds, statuses, and boxed-variable transitions;
- reduced costs and dual values;
- entering-variable selection and pricing rules;
- ratio tests and leaving-variable selection;
- primal and dual feasibility tracking;
- primal/dual edge-norm updates;
- basis changes, update rows, direction computation, and refactorization policy;
- initial basis construction and crash strategies;
- iteration limits, time limits, deterministic limits, and progress statistics.

Initially expose narrow component tests rather than forcing every piece through
the final solver. Differential tests should compare pivot candidates, leaving
variables, step lengths, basis status, and numerical state after one controlled
iteration.

Exit criteria:

- Small hand-constructed dictionaries perform the same pivot as GLOP.
- Multi-iteration traces agree on deterministic fixtures.
- Refactorization and incremental basis updates remain numerically consistent.

## Phase 6: port revised simplex end to end

Translate the revised-simplex driver and its immediate orchestration:

- primal simplex;
- dual simplex;
- phase I feasibility procedures;
- optimization phase and termination tests;
- perturbation, bound flipping, and degeneracy handling;
- unbounded and infeasible certificates where GLOP exposes them;
- warm starts and externally supplied bases;
- solution extraction and post-solve validation.

Begin with preprocessing and scaling disabled so discrepancies remain localized.
Add a trace mode to both implementations that emits normalized iteration events
for side-by-side comparison without affecting release performance.

Progress through dataset gates:

1. synthetic and hand-written LPs;
2. Netlib 10-smallest;
3. Netlib 25-smallest;
4. Netlib 50-smallest;
5. full Netlib without preprocessing;

Exit criteria:

- Status and objective agree throughout the smoke suites.
- Independent KKT/residual validation passes.
- Any iteration divergence is understood and recorded.

## Phase 7: port scaling and preprocessing

Port scaling and preprocessors incrementally, one transformation at a time.
Each transformation needs:

- direct unit tests;
- a reversible transformation/postsolve test;
- comparison of transformed model statistics with GLOP;
- solution recovery tests for optimal, infeasible, and unbounded cases;
- before/after performance measurements on affected Netlib problems.

Likely work includes:

- matrix equilibration and objective/bound scaling;
- singleton and empty row/column handling;
- fixed variables and implied bounds;
- forcing and dominated constraints;
- duplicate/proportional rows and columns where used upstream;
- unconstrained variables and free constraints;
- postsolve stack and status reconstruction.

Exit criteria:

- Enabling transformations one by one preserves independently validated
  solutions.
- The default preprocessing pipeline agrees with GLOP across Netlib.

## Phase 8: parameters and public API

Replace `parameters.proto` with ordinary Rust types while retaining:

- corresponding enum variants;
- field names recognizable from GLOP;
- identical defaults;
- validation rules and interactions;
- an explicit mapping from every supported upstream field.

Expose:

- a library API for constructing and solving an LP;
- MPS loading through `lp_data`;
- basis import/export and warm starts;
- structured result, status, statistics, primal/dual values, and certificates;
- a CLI suitable for regression and benchmark automation.

Fields that are intentionally unsupported must fail explicitly rather than be
silently ignored.

Exit criteria:

- Parameter-default comparison is automated.
- Public examples solve representative LPs.
- Warm-start behavior is covered by differential tests.

## Phase 9: full differential validation

Build a harness that runs GLOP and `gloprs` with matched parameters and compares:

- parse/model statistics;
- presolve result;
- final status and objective;
- primal and dual feasibility;
- reduced costs and complementary slackness;
- basis status and certificate validity;
- simplex iterations and refactorization counts;
- deterministic time or work limits.

Classify mismatches rather than weakening tolerances globally:

- parser/model mismatch;
- legitimate alternate optimum;
- pivot-order divergence;
- tolerance-boundary difference;
- numerical instability;
- incorrect status or solution;
- unsupported feature.

Minimize failures into permanent regression fixtures.

Exit criteria:

- Full Netlib results are summarized in a checked-in report generated by the
  harness.
- No unexplained incorrect status or invalid solution remains.
- Known divergences have issue references and minimized tests.

## Phase 10: performance parity campaign

Benchmark native GLOP and `gloprs` on the same machine with pinned compilers and
equivalent optimized settings. Use repeated runs, warm caches where appropriate,
and report medians plus dispersion.

Measure:

- parsing, presolve, simplex, and postsolve separately;
- iteration count and time per iteration;
- LU factorization, solves, and basis-update cost;
- pricing and ratio-test cost;
- allocations and peak resident memory;
- total solve time and throughput across Netlib.

Use profiling to address gaps in this order:

1. algorithmic or iteration-count differences;
2. avoidable allocations and data-layout problems;
3. sparse traversal and cache locality;
4. bounds-check and abstraction overhead;
5. compiler/code-generation issues.

Any optimization that makes the Rust code depart materially from upstream must
be benchmarked, documented in the porting table, and protected by differential
tests.

Exit criteria:

- A reproducible comparison report lists per-instance and aggregate performance.
- Major regressions are either fixed or explained with profiles and follow-up
  work.
- Numerical quality remains equivalent under benchmark settings.

## Cross-cutting risks

- **Numerical divergence:** mathematically equivalent reassociation can change
  pivot choices. Preserve operation order until evidence supports a change.
- **Hidden infrastructure dependencies:** Abseil and protobuf removal may mask
  semantics such as stable iteration, default values, or status propagation.
- **Strong-index mistakes:** row/column confusion may compile if represented as
  bare integers. Use distinct types.
- **Nondeterminism:** hash iteration and parallelism can change pivot order.
  Establish a deterministic baseline before parallel optimization.
- **Parser scope:** Netlib MPS syntax and extensions must be inventoried against
  actual files, not assumed from a minimal grammar.
- **Performance by architecture:** a literal C++ container translation may be
  unidiomatic and slow, while premature redesign can obscure correctness. Port
  behavior first, then change representation behind tested interfaces.
- **Benchmark unfairness:** mismatched presolve, tolerances, compiler flags, or
  termination criteria invalidate comparisons.

## Immediate next actions

1. Initialize `sparse/gloprs` as an independent Git repository and add its
   baseline `.gitignore`, licensing files, and initial commit.
2. Initialize the Cargo workspace.
3. Clone and pin the official OR-Tools reference in `../or-tools`.
4. Generate the upstream file/dependency inventory and porting table.
5. Build a small native GLOP reference runner with structured output.
6. Implement the Netlib downloader and manifest schema.
7. Port strong indices and sparse `lp_data` primitives with tests.
