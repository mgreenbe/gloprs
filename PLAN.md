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

## Phase 0: pin and inventory upstream

Status: complete. The pinned upstream configuration is recorded in
`UPSTREAM.md`, and `PORTING.md` contains the initial file inventory. The native
GLOP builds pass, and `tools/run_glop_reference.py` emits normalized JSON for
MPS solves, including variable and constraint basis statuses supplied by the
focused native adapter in `tools/glop_reference_adapter.cc`.

Exit criteria:

- Basis statuses are present in structured reference results.

## Phase 1: port `lp_data` foundations

Status: complete. Typed sparse primitives, permutations, scattered workspaces,
the linear-program model, validation, deterministic summaries, and fixed/free
MPS parsing are implemented. All 98 models in the shared Netlib manifest parse,
and dimensions, nonzeros, bounds, and objective data agree with the pinned
native GLOP model via `tools/validate_netlib_parse.py`. A dependency-free
sparse-primitive microbenchmark is available as `cargo bench -p
gloprs-lp-data --bench sparse_primitives`.

Port the minimum model and sparse-data layer needed by GLOP:

1. Complete the remaining `lp_types` facilities, including bit vectors and the
   sparse entry iterator.
2. Complete sparse-vector operations, column views, typed permutations, sparse
   rows, and column-oriented sparse matrices.
3. Add scattered-vector workspaces and nonzero-pattern tracking.
4. Add the linear-program model representation: bounds, objective, names,
   scaling metadata, and basis status.
5. Add model validation and canonicalization.
6. Port fixed/free MPS parsing, including bounds, ranges, objective sense,
   duplicate entries, and numerical edge cases used in Netlib.
7. Add deterministic model and solution summaries for differential testing.

Port the associated upstream tests before adding solver behavior. Add parser
golden tests and round-trip or normalization tests where exact text emission is
not an upstream requirement.

Exit criteria:

- Every smoke-subset MPS file parses.
- Parsed dimensions, nonzeros, bounds, and objectives agree with GLOP.
- Sparse primitive tests and microbenchmarks are established.

## Phase 2: port numerical and basis kernels

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

## Phase 3: port simplex state and pivot mechanics

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

## Phase 4: port revised simplex end to end

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

## Phase 5: port scaling and preprocessing

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

## Phase 6: parameters and public API

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

## Phase 7: full differential validation

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

## Phase 8: performance parity campaign

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

1. Begin Phase 2 with the dense/scattered vector operations and sparse matrix
   views required by the numerical kernels.
2. Port triangular solves and transpose solves with dense-reference tests.
3. Port sparse LU and Markowitz pivot selection with residual tests and
   microbenchmarks.
