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

Status: complete. The typed sparse primitives, model, readers, and supporting
utilities required by GLOP's numerical kernels have been reconciled with the
pinned implementations at the level of storage, traversal, mutation, and
numerical order. All 98 Netlib models agree on parse fingerprints, and the
component-level generated native traces recorded below cover the translated
surface. Protobuf-only wrappers and the revised-simplex-dependent
`LINEAR_PROGRAM` scaling path remain assigned to their later infrastructure and
scaling phases; they are not substitutes or hidden Phase-1 implementations.

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

Status: complete. The compact views, triangular solves, threshold-Markowitz
LU, direct and transpose solves, both basis-update forms, update rows, edge
norms, and numerical checks now use the corresponding sparse GLOP algorithms
and data structures and have focused native differential coverage. The
original packed-dense LU was replaced by a sparse implementation with
an incremental residual nonzero pattern, GLOP's two-stage singleton ordering, a
Zlatev degree queue, sparse candidate columns, DFS-restricted left-looking
solves, contiguous triangular storage, and both directions of each
permutation. A 1,000-case generated differential trace agrees on pivot
permutations, factor entry counts, deterministic operation time, determinants,
and solves. Another 300 dense 40–100-dimensional traces agree, including the
contracted multiply-add ordering that controls cancellation-sensitive pivot
ties. A 500-case rectangular native trace also agrees on initial-basis
completion and exact pivot sequences. A separate large, low-density trace exercises the scattered and
hypersparse direct and transpose paths against GLOP, including their reported
nonzero positions. The factorization-time permuted lower solve now lives on
`TriangularMatrix`, as it does upstream, and operates on Markowitz's actual
in-progress L rather than a parallel graph copy. Its DFS performs GLOP's
persistent dependency-graph pruning with the same LIFO root and edge traversal
order and the same in-place row/coefficient swaps used by later numerical
substitution and retained in the completed factor. Cached candidate columns use GLOP's
logical-to-physical reusable allocation pool, and LU/basis solves now carry
typed scattered nonzero sets through GLOP's 2.5%/5% hypersparse switching
rules and explicitly stored triangular transposes.
The Phase-2 numerical controls now originate in a shared `GlopParameters`
value containing the complete pinned 59-field protobuf schema and all five
enums with their upstream defaults and signed scalar types. Validation follows
`parameters_validation.cc` in source order, including its distinct finite,
not-NaN, nonnegative, integer, and magnitude checks; 124 dedicated native
boundary cases agree byte-for-byte on the first diagnostic. Five hundred
native LU traces also agree under nondefault pivot thresholds and Zlatev
candidate counts.
The Phase-1 numerical utility layer now includes GLOP's dense, sparse, and
scattered scalar products and norms, accurate reductions, resetting reductions,
threshold/support helpers, and restricted norms. A 1,000-case native trace
agrees within expected cross-compiler ulps and verifies exact reset side effects.
Sparse-vector cleanup, threshold filtering, weighted filtering, partial
permutation, and tagged-entry movement now agree entry-for-entry with GLOP on
1,000 generated native traces. The audit also corrected `Reserve()` to use
GLOP's total-capacity contract, restored the signed-sentinel
`IndexPermutation` API used by partial permutations and tagged moves, and
preserves duplicate-state knowledge across bijective and partial index
permutations. The row-specialized wrapper now exposes the complete naming and
permutation surface from `sparse_row.h`; a dedicated 1,000-case native trace
agrees exactly on its accessors, iterator, permutations, coefficient bits, and
typed row-major storage. Typed bit vectors now carry the
pinned bucket, paired-bit, resize, content-copy, intersection, and union
semantics used by GLOP's solver state.
The remaining `lp_types` span/vector surface is now present, including borrowed
typed immutable and mutable views, capacity/reserve/assign/zero/resize-down
operations, pinned enum representation, and scalar helpers. A 1,000-case native
trace validates statuses, constants, and `Bitset64` operations. It exposed and
corrected a subtle iterator mismatch: GLOP traverses populated padding bits in
allocated buckets after cross-size operations, so Rust now uses the same
bucket/trailing-zero iteration rather than scanning only the logical range.
Sparse-vector storage has also been changed from an array of entry structs to
GLOP's structure-of-arrays organization: indices and coefficients occupy
separate contiguous regions, and iteration constructs lightweight entry views.
The stable temporary-pair cleanup algorithm and entry order remain unchanged.
The remaining logical resize and indexed/mutable entry surface is now present;
the 1,000-case native transformation trace plus downstream matrix and LU traces
cover the resulting representation. Safe Rust retains two typed allocations
instead of GLOP's manually split untyped block without changing traversal or
complexity.
The `SparseColumn` specialization now includes GLOP's row-named accessors and
permutation methods, a borrowed parallel-slice `ColumnView`, and the touched-row
`RandomAccessSparseColumn`. The sparse-vector native trace covers view order,
last-duplicate lookup behavior, random-access mutation, explicit zeros, and
sparse extraction as well as the inherited transformations.
Scattered vectors now expose GLOP-style allocation-free entries with specialized
row/column names and a safe O(1) borrowed transpose view in place of C++'s
layout-dependent `reinterpret_cast`. A dedicated 1,000-case native trace agrees
on sparse-mask state transitions, position order and duplication, sorting,
density switching, estimates, values, and transpose traversal.
The previously absent `matrix_utils` module now includes both proportional-
column algorithms with GLOP's exact fingerprint hash and normalization order,
leading-rectangle equality, and rightmost-identity recognition. All recorded
outputs agree exactly on 1,000 generated native traces.
`CompactSparseMatrix` now follows GLOP's explicit-dimension representation and
supports its incremental builders, dense/nonzero builders, slack augmentation,
stable-order transpose, four-accumulator column product, scattered updates,
copy paths, and borrowed compact basis views. Borrowed sparse matrix views also
cover matrix pairs and basis subsets. Generated native traces agree on
dimensions, entry order, coefficients, transpose/slack storage, products,
selected-view storage, entry counts, and norms. Permutation validity also
returns false for negative destinations instead of panicking. Random
population, cross-strong-index forward/inverse application, and the specialized
column-permutation path are present; a dedicated 1,000-case native trace agrees
on deterministic permutation behavior, while Rust's uniform `rand` shuffle
replaces `absl::BitGen`.
Triangular matrices now expose GLOP's separate-diagonal incremental builders,
normalized-column construction, identity-prefix bookkeeping, row-permutation
and copy paths, lower solves from a known starting column, both symbolic
closure algorithms, all four ordered hypersparse solve variants, triangularity
checks, and exact and estimated inverse infinity norms. Dense transpose solves
use GLOP's four-term accumulation groups, reverse storage traversal for lower
factors, and the same trailing-zero skip (including signed-zero behavior). A
1,000-case native trace agrees on stored entries, symbolic order and filtering,
norms, and dense and hypersparse direct/transpose solves.
General sparse matrices now preserve GLOP's construction and traversal order
for reserved transpose, row/column permutations, linear combinations, and
matrix products; the port also includes partial deletion, row append,
duplicate-aware equality, and magnitude diagnostics. A further 1,000 native
traces agree on entry order and numerical results. In particular, row
permutation no longer performs the non-upstream per-column sort that had added
both behavioral and logarithmic-complexity differences.
The Phase-1 model layer now also carries GLOP's integer/binary predicates,
solution-feasibility checks, objective translations, bound intersections,
slack injection/removal and equation-form invariants, plus metadata-preserving
row and column deletion. Its transpose and integer-variable classifications
use GLOP's lazy caches with matching invalidation, adoption, and incremental
deletion behavior rather than rebuilding temporary data on every query. A
500-case native trace agrees on the resulting decisions, bounds, sparse entry
order, and slack values. The same trace now also covers constraint-block
append, full and variables-only copying, row/column-permuted population, and
primal-to-dual construction, including exact variable grouping, duplicate-row
mapping, bounds, and sparse entry order. All four objective-scaling algorithms,
bound scaling, and magnitude-limited model validation are covered by the same
native comparison.
The MPS reader now uses GLOP's whole-input fixed-then-free auto-detection
rather than a line-by-line hybrid, and exposes both forced formats. A direct
native harness compares the complete parsed LinearProgram, including names,
integrality, sparse entry order, and numeric bit patterns. One thousand free
and one thousand fixed generated models agree in all three format modes, as do
focused malformed/numerical cases and all 98 Netlib model fingerprints. This
audit corrected integer-marker default bounds, vector-name handling,
feasibility-only models, later free rows, model-validation timing, missing
ENDATA behavior, case-sensitive section parsing, and signed-zero suppression.
The SOL reader's LinearProgram entry points are now ported as well, including
GLOP's comment/token diagnostics, generated fallback names, overwrite rules,
and `strtod` leading-value behavior. Its hexadecimal path performs explicit
round-to-nearest-even conversion and reproduces the pinned platform's exact
subnormal/overflow decisions. A 1,265-case native trace compares result bits
and diagnostics, including randomized ordinary and hexadecimal inputs. The
protobuf-only overload remains an infrastructure omission.
Independent LP decomposition now uses the same union-by-size/root-tie
partitioning, first-node component numbering, sorted global-column clusters,
and first-touch sparse constraint ordering as GLOP. Local model extraction and
both assignment projection directions agree byte-for-byte with native GLOP on
1,000 randomized models. Rust's shared source-model borrow replaces the C++ raw
pointer and mutex while enforcing the same no-mutation lifetime contract.
The LP text parser now follows GLOP's lexer/state machine directly, including
its permissive sign and multiplication syntax, prefix-matched integer/binary
keywords, exact name patterns, bound intersection, duplicate rejection, and
partial-model-on-error behavior. A 2,024-case native trace agrees on complete
canonical model dumps, parsed constraint fields and coefficient bits, and exact
diagnostics; focused malformed, overflow, underflow, and keyword-prefix probes
also agree. The protobuf conversion wrapper remains an infrastructure omission.
The associated model diagnostics now include GLOP's dimension, coefficient,
bound, structural, and nonzero-statistics strings as well as complete LP and
solution text emission. Generated parser traces compare all of that text
byte-for-byte, and a separate 10,000-case floating-point bit-pattern trace
agrees exactly on decimal and monomial formatting.
The ordinary sparse matrix scaler is a direct port of GLOP's geometric passes
and final equilibration. A strengthened 1,000-case native trace covers every
public factor getter and vector operation, null and explicit-zero matrices,
identity magnitudes, geometric and dynamic-range-cutoff branches, empty rows
and columns, extreme ranges, clearing, and repeated-`Init()` state. That audit
exposed and corrected Rust's reset-on-`Init()` behavior to match the pinned
implementation's factor-retaining `resize()`. `LpScalingHelper` likewise
agrees on 1,000 traces covering every scalar conversion, dense and sparse solve
unscaling, ordinary and slack variables, clearing, infinities, and degenerate
cost/bound normalization. Whole-model equilibration agrees on a further 1,000
native traces spanning all four objective cost-scaling algorithms and covering
every model bound, objective coefficient, sparse matrix entry, and retained
helper factor. The parameterized helper API now delegates the matrix method to
`glop`, where the actual revised simplex is available without reversing crate
dependencies. Another 500 generated native comparisons cover default,
equilibration, and auxiliary-LP matrix scaling with all four objective-scaling
algorithms. The auxiliary LP's ordered model also matched native bit-for-bit
on the investigated case. These randomized LP-method cases require every row
to contain an entry: upstream creates an absent row's scaling variable only
after solving, so its empty-row factor is not a stable solution value. A
separate native fixture retains empty-row coverage. The error-to-equilibration
dispatch is tested with an injected auxiliary-solve error, and a whole-model
fallback fixture agrees with native equilibration output. A deterministic
search of 11,000 finite sparse auxiliary LPs (dimensions up to 100, including
subnormal and near-maximum coefficients) produced no native solver error.
The native-error branch remains defensive; no naturally failing case is known.
The basis now uses
GLOP's middle-product-form rank-one updates and the same identity-basis,
fixed-period, and deterministic-time-adjusted refactorization policy. The
ordering around a scheduled rebuild also matches upstream: the iteration's
entering/leaving solves are charged first, the new basis column is visible to
the rebuild, and no rank-one update is appended. Update-row has the cached
column/row/hypersparse kernel selection, and primal and dual edge norms have
their incremental GLOP recurrences and reset policies. Their regression tests
agree with fresh factorization or exact recomputation, and both primal and dual
edge-norm implementations have 500-case native traces. Their expensive full
recomputations now use GLOP's `TimeLimit` threshold and stop point; separate
large traces prove the greater-than-10,000-LU-entry branch and partial-result
semantics, while the primal trace also covers recomputation watchers and
deterministic-time accounting. The supporting time-limit implementation has a
native differential trace for deterministic, external, merge, history-reset,
and sticky wall-limit behavior. GLOP's always-enabled distribution statistics,
Welford accumulation, merge/reset behavior, ordering, and formatting are also
ported; byte-level native traces now cover Markowitz, repeated basis
refactorizations, edge norms, and the empty default-release `UpdateRow`
statistics surface. That audit exposed and corrected the dedicated
rightmost-slack scalar-product path in the primal edge update. Integration with
the future revised-simplex driver, including end-to-end implicit-slack traces,
belongs to Phase 3; the canonical implicit-slack kernel behavior itself is
already covered here.
`VariablesInfo` now supplies GLOP's bound,
status, movement, basic/nonbasic, relevance, boxed-variable, relevant-entry,
and dual phase-I state, so the update-row and norm kernels consume the same
typed relevance bitset used by the upstream architecture. Its advanced
zero-copy mutable-bound API and allocation-free incremental structural/slack
loader are also ported with upstream's changed-column type recomputation and
unchanged fast path. An expanded 500-case generated native trace agrees on all
recorded `VariablesInfo` transitions, including mutated-state reconstruction.
A separate 500-case native trace agrees on update-row positions and
coefficients for the column-wise, row-wise, and hypersparse row-wise kernels.
Dual steepest-edge recomputation, precision checks, adaptive cached tau solves,
and Koberstein updates agree on a 500-case native trace over independently
row- and column-permuted sparse triangular bases.
Primal matrix norms, steepest-edge recomputation and updates, precise-sum Devex
weights, update-row integration, and entering-edge precision checks likewise
agree on 500 native traces using GLOP's canonical structural-plus-trailing-basis
layout with independently row- and column-permuted sparse triangular bases.
The middle-product-form basis update is covered by 300 generated multi-pivot
native traces that compare right and left solves, update sparsity, norms,
condition estimates, and deterministic time after every update. Separate
300-case traces cover exact fixed-period and dynamically adjusted
refactorization decisions. This exposed and corrected a contraction-sensitive
structural difference in rank-one updates: explicit fused multiply-adds now
retain the same machine-scale nonzeros as the optimized native build.
The rank-one scalar products now also use the four independent accumulators and
remainder order of `CompactSparseMatrix::ColumnScalarProduct`, rather than a
generic sequential iterator sum.
An additional 1,000-case isolated native trace covers elementary rank-one
multiply/inverse operations and packed dense and scattered factorization
solves, including sparse-mask transitions, entry accounting, clearing, and
deterministic time. The optimized native build contracts the scalar-product
accumulations as well as dense vector updates, so the Rust kernels explicitly
retain those contractions. GLOP's scattered helper instead materializes each
product before `ScatteredVector::Add`; the Rust sparse path preserves that
separate rounding too.
The alternate product-form eta update selected when
`use_middle_product_form_update` is false is now ported as well. Three matching
300-case native suites cover its ordinary, fixed-period, and dynamically
adjusted policies. They also verify GLOP's subtle representation convention:
an empty sparse eta column means that the populated dense direction is
authoritative, rather than that the eta column has no off-diagonal entries.
All six suites also force a final refactorization and compare GLOP's
specialized primal and dual squared-norm solves, temporary unit-row solves, and
their exact deterministic-time charges. The specialized Rust paths retain
scattered/hypersparse traversal rather than falling back to dense solves.
Repeated unit-row solves also reuse their pre-update triangular result from a
compact sparse column pool with the same refactorization lifetime as GLOP.
Problem-column solves now retain the complementary pre-upper intermediate in a
second compact pool, and middle-product updates consume the two cached columns
through the same explicit solve-then-update protocol as the C++ implementation.
Sampling localized roughly 75% of Rust factorization time to sparse candidate-
column solves. Removing non-upstream candidate/pattern copies and duplicate
scattered-workspace clearing improved the in-process release LU benchmark.
Against pinned GLOP on Apple arm64, the latest three-trial Rust/native ratios
are 1.79x, 1.83x, and 1.98x at dimensions 500, 750, and 1000 for 2%-density generated matrices.
This is not yet performance parity, but it rules out an unexplained order-of-
magnitude regression and provides a reproducible optimization gate.

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

Status: complete. The remaining state modules now follow their pinned GLOP
counterparts directly: the top-31 dynamic pricing heap, primal variable values
and infeasibility prices, reduced costs and dual values, primal prices, both
dual entering-variable ratio tests, and all Bixby/triangular/Maros crash
strategies. The primal Harris leaving-row code is retained as a narrow
Phase-3 kernel extracted from `revised_simplex.cc`; the full driver remains in
Phase 4 and is explicitly still `in progress` in `PORTING.md`.

Differential coverage includes 500 generated pricing-operation traces, 100
generated reduced-cost/value/feasibility states, 100 generated dual phase-I and
phase-II ratio tests, and 375 initial-basis crashes spanning all five modes.
Twenty controlled three-pivot primal traces agree with native `RevisedSimplex`
on iteration counts, basic-variable sets, final variable values, and reduced
costs. Existing Phase-2 multi-update suites continue to verify that the basis,
update-row, direction, edge-norm, and refactorization machinery used by those
pivots agrees after every update. The crash comparison deliberately preserves
two implementation-sensitive upstream details discovered by the trace: Maros's
literal row-to-column availability lookup and the pinned libc++ heap behavior
for equivalent triangular candidates.

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

Status: the previously specified unscaled, unpreprocessed exit gates pass.
The singleton-column initial-basis shortcut is now ported and checked against
native GLOP by the Phase-5 partial-pipeline solve fixture, including the
default enabled path and zero-pivot one-row controls.
File-level Rust counterparts for `revised_simplex` and
`lp_solver` now exist. The current driver includes equation-form conversion,
the primal phase-I piecewise-linear feasibility objective and breakpoint ratio
test, primal optimization, Harris ratio tests, bound flips, product-form basis
updates, initial-basis crashes and condition rejection, warm-basis input,
solution/ray extraction, independent residual checks, and opt-in normalized
iteration events. The public CLI can run an unscaled, unpreprocessed solve.
The pivot path now also performs GLOP's independent BTRAN/update-row versus
FTRAN pivot comparison and refactorizes instead of retaining an imprecise
middle-product update. Basis-update refactorizations leave their LU column
permutation visible until the revised-simplex owner applies it to the external
basis mapping, matching `PermuteBasis()`. The optimality path also performs
GLOP's precise final check by refactorizing, recomputing reduced costs, and
pricing again before accepting an empty candidate set. Primal phase I and II
now select entering columns through the ported `PrimalPrices` heap with the
configured Dantzig, steepest-edge, or Devex norms, test entering-edge precision,
and maintain the norms through the shared `UpdateRow` before each basis pivot.
Optimization pivots now maintain reduced costs and heap prices incrementally in
GLOP's order (edge norms, reduced costs, prices), use the same update-row sparse
support, and retry after the precise entering reduced cost invalidates the
selected candidate. Imprecise edge norms and bound flips now use GLOP's local
heap maintenance rather than unconditional price rebuilds. Full recomputation
is retained after refactorization or a changed Phase-I objective. Phase I now
refreshes only the preceding direction's basic costs between refactorizations
and retains its incremental reduced costs when those costs do not change.
As in GLOP, a refactorized basis also triggers a residual check and recomputes
basic variable values when the Harris-scaled feasibility tolerance is exceeded.
Exact steepest-edge initialization calls the LU-specific sparse squared-norm
solve directly for every relevant problem column, rather than allocating a
dense right-hand side and routing through the general basis solve.
Dual steepest-edge initialization likewise calls GLOP's specialized
row-indexed LU squared-norm kernel instead of allocating a dense unit vector
and routing every row through the general transpose solve. Its incremental tau
update now also consumes the scattered BTRAN result already owned by
`UpdateRow`, avoiding reconstruction through a fresh allocation and dense
scan on every dual pivot.
Maximization objective coordinates and primal objective limits follow GLOP's
offset/scaling sign convention and strict stopping test.
Final objective values now use GLOP's compensated `AccurateSum` recurrence in
both `RevisedSimplex` and `LPSolver`.  The revised-simplex solution snapshot
also changes the signs of dual values and reduced costs for maximization before
exposing them, as upstream does, and final primal/dual residual classification
uses GLOP's coordinate-scaled allowed errors while retaining the unscaled
maximum residuals for reporting.
The public solution loader now enforces GLOP's complete status/basis
consistency contract, including exact nonbasic values, free-variable and
constraint requirements, and the required number of basic variables.  Its
optimality validation also computes the compensated dual objective and GLOP's
expected primal-objective error bound before accepting the primal/dual gap,
and multiple-solution detection covers both variable and constraint facets.
Primal-unbounded results now receive GLOP's final refactorization check before
a certificate is accepted.  The public solver clears certificates between
solves, exposes rays only for the matching terminal status, removes internal
slack coordinates from primal rays, and establishes the dual-certificate
identity `variable_bounds_ray = -A^T constraints_ray` with GLOP's
minimization/maximization sign convention.
The default dedicated dual Phase I and dual Phase II are now connected. They use
the ported `DynamicMaximum` dual prices, exact dual edge norms, sparse BTRAN
and update-row computation, Harris bound-flipping ratio test, incremental
reduced-cost and norm updates, middle-product basis updates, boxed-variable
flips, precision-triggered refactorization, and final optimality/unboundedness
checks. Degenerate dual pivots now use GLOP's cost shifts, including the
minimum reduced-cost displacement, incremental shifted reduced costs, removal
of all shifts at a candidate termination, and refactorized reoptimization.
A pinned 3×3 case now follows cost-shift removal through the dual-to-primal
cleanup switch and primal reoptimization, matching native state and clock.
A separate 3×3 fixture removes a cost perturbation after dual Phase II, then
switches to primal reoptimization with no cost shift; native state and clock
agree. Imprecise pivots also increase the LU pivot threshold before rebuilding the
basis when fewer than ten updates have accumulated, matching `UpdateAndPivot()`.
The dual objective limit follows GLOP's shifted/scaled external
coordinates and is tested separately. One- and two-pivot regressions cover
optimal solves, a dedicated-phase-I regression starts from a dual-infeasible
basis, and a dual-ray regression covers primal infeasibility. The CLI and Netlib validator expose an
opt-in dual mode; all smallest-50 models pass status, objective, and independent
primal/dual feasibility validation with a 10-second per-model limit using the
dual driver throughout. The nondefault transformed-problem dual Phase-I
alternative was still falling back to the primal driver at this stage; it is
now connected and validated separately below.
On the full 98-model corpus, 96 models pass the same validation with a
10-second wall limit; only `qap12` and `qap15` time out, and no model now
terminates abnormally. In particular, the cost-shift
port resolves the former `ABNORMAL` results on `perold`, `pilot`, and `pilot87`.
An iteration-prefix differential adapter now records initial and current basis
mappings, the initial LU column permutation, Phase-I prices, dual edge norms,
values, and reduced costs from both implementations. It proves that `perold`'s
crash basis, initial LU permutation, and post-permutation basis agree exactly.
The apparent post-first-pivot refactorization in the first prefix trace was
GLOP's iteration-limit final check, not a trajectory event. Instrumentation
instead localized the first numerical drift to the first dual-edge-norm
update: a dense-sentinel scattered row was treated as having no entries when
computing tau. Preserving those dense values removed the pivot-12 divergence.
Basis permutations update Rust's Phase-I pricing vector and dual edge norms as
`PermuteBasis()` does upstream. GLOP's exact
shared `std::mt19937_64` stream, libc++/Abseil distributions, and isolated tied
top-31 pricing sequence agree exactly; this is not a tolerance or RNG defect.
The former pivot-46 difference was an exact-price tie exposed by roundoff in
an incremental Phase-I pricing solve. Factor-stage traces localized it to a
middle-product denominator: GLOP uses the four-accumulator
`CompactSparseMatrix::ColumnScalarProduct`, while Rust had used a linear fold.
Using the ported four-accumulator kernel removed the former pivot-46
divergence. A subsequent audit found that Rust unconditionally forced a second
LU rebuild after dedicated dual Phase I where GLOP's conditional
`Refactorize()` is a no-op; preserving the existing factorization removes the
resulting basis-order and random-stream drift. A later file-for-file audit
found that the 132-pivot `perold` prefix used a numerically equivalent but
algorithmically different left solve. GLOP computes the symbolic closure from
the explicitly transposed factor, then performs numerical transpose
substitution on the original factor. The Rust port now preserves that split.
The faithful path now agrees for the first 47 `perold` pivots. The former
pivot-46 tie was caused by constructing cached unit-row partial solves through
the generic symbolic hypersparse path. GLOP instead uses
`LeftSolveUForUnitRow()`, whose unit-RHS starting-column solve retains tiny
numerically nonzero residual entries outside that symbolic closure. Porting
that specialized path makes the affected middle-product update and Phase-I
price bit-identical; a later exact pricing tie first changes pivot 48. The
trajectory audit also found two Phase-II orchestration mismatches. GLOP applies pending
boxed-variable flips before updating dual prices for the preceding direction
at the start of the next iteration; Rust now preserves that order instead of
updating at the end of the preceding iteration. More importantly, a dense
FTRAN result from a boxed-variable flip now takes GLOP's full
`RecomputeDualPrices()` path instead of incrementally updating every row. The
latter difference retained stale top-31 heap entries and gave exact ties the
wrong multiplicity. The incremental reduced-cost update now uses an explicit
fused multiply-add, matching the contraction emitted for GLOP's
`rc[col] += new_leaving_reduced_cost * update_coeffs[col]` in the optimized
native build. An audit of the other direct multiply-accumulate kernels now
also preserves native contraction in pivot value updates, eta solves, dense
row-wise update-row products, sparse-to-dense updates, and matrix residual/RHS
construction. It deliberately leaves separately materialized products and
multi-accumulator reductions unchanged. Full reduced-cost recomputation now
likewise uses GLOP's
four-accumulator compact-column scalar product for structural columns and its
direct dual-value subtraction for trailing slacks; the scalar-product
accumulators use the fused operations emitted by the optimized native build.
The basic-objective left inverse now goes through GLOP's scattered-row
dense-sentinel solve (`LeftSolveUWithNonZeros`, middle-product updates, then
`LeftSolveLWithNonZeros`) rather than the separate packed dense transpose
solve. This restores `blend`'s exact trajectory and preserves the sparse solve
architecture used upstream.
This moves `scorpion`'s first divergence from pivot 21 to pivot 163, `israel`'s
from pivot 21 to pivot 148, makes `capri`, `vtp.base`, and `boeing2` agree
completely, and moves `lotfi`'s first divergence from pivot 18 to pivot 175. A
newer smallest-25 audit after the subsequent solve-path fixes finds complete
pivot-sequence agreement for 22 models:
`afiro`, `sc50a`, `sc50b`, `kb2`, `sc105`, `adlittle`, `stocfor1`, `blend`,
`scagr7`, `sc205`, `share2b`, `recipe`, `lotfi`, `vtp.base`, `share1b`, `capri`,
`scagr25`, `boeing2`, `scorpion`, `sctap1`, `scfxm1`, and `bandm`. In `bore3d`,
the former pivot-39 row-coordinate split was caused by omitting GLOP's
pre-FTRAN `dual_small_pivot_threshold` retry. Porting that branch, separately
from the later direction-relative pivot check, extends exact agreement through
pivot 65; pivot 66 now chooses native column 250 versus Rust column 249 while
leaving the same row and basic column. The other first divergent pivot is 149
for `israel`. The former
`brandy` pivot-120 split was caused by retaining an exact numerical
cancellation as a structural zero in an L column during the routine LU rebuild
after pivot 64. GLOP's `AddAndNormalizeTriangularColumn()` removes such entries;
doing the same restores the native 167-iteration count and exact final basis.

A renewed `bore3d` audit proved that the basis, reduced costs, dual norms, next
update-row support, and every update-row coefficient were bit-identical through
pivot 65, while shared-RNG drift changed the pivot-66 tie. The missing draws
were caused by a Phase-II orchestration mismatch: GLOP retains
`direction_.non_zeros` across an iteration retry and reprices those rows again,
whereas Rust cleared its pending row copy immediately after the first update.
Retaining it until a successful pivot replaces the direction restores GLOP's
intentional duplicate `DynamicMaximum` entries and tie draws. `bore3d` now has
the same 128 iterations, final ordered basis, reduced costs, and dual norms as
native GLOP. The same audit localized `boeing1`'s first
basis split to pivot 145 (native enters column 134, Rust column 130); immediately
before it, the ordered basis, reduced costs, and dual norms are bit-identical
(apart from a signed zero in a variable value). Sparse basis right solves now
perform GLOP's final conditional nonzero sort, and breakpoint ordering uses
ordinary floating-point comparisons so signed zero is not distinguished from
zero as it would be by `total_cmp()`. A further audit found that Rust applied
stale boxed-variable flips and pending price updates after the routine basis
refactorization at pivot 114; GLOP instead recomputes values and prices and
skips both incremental updates. The Phase-II branch now matches upstream and
the resulting state again agrees through pivot 144. The remaining pivot-145
tie exposed the other missing half of the refactorized branch: GLOP moves all
dual-infeasible nonbasic boxed variables to their opposite bounds before
recomputing basic values. Porting that pass removes stale columns 301 and 350
from the following flip list, restores the row-42/row-234 pricing tie and
Bernoulli draw, and reconciles every subsequent ordered-basis prefix.
`boeing1` now terminates optimally after the native 507 iterations with the
same ordered basis, reduced costs, and dual norms; only two signed-zero value
bits differ.

The basis factorization now also matches upstream's matrix ownership
architecture. It keeps an immutable shared view of the original problem matrix
and a basis-column mapping, so a simplex pivot changes one index and subsequent
LU refactorizations traverse selected original columns directly. This removes
the former per-pivot sparse-column clone and cleanup and avoids materializing a
separate basis matrix.

Future API work may reconsider storing that shared matrix handle in the basis
representation. Passing an ordinary `&SparseMatrix` into the refactorization
and norm operations that need it would remove reference-counted ownership, but
would require threading the borrow through the solver API because safe Rust
cannot store a reference from one field of `RevisedSimplex` into another field
of the same movable struct. Keep the current persistent shared view for port
fidelity unless profiling justifies that broader, deliberately non-upstream API
change; preserve the selected-column view and do not reintroduce basis-column
cloning or basis-matrix materialization.

The next path audit found the right-solve counterpart of the earlier
left-solve representation mismatch. GLOP's `RightSolveUWithNonZeros()` uses
`U` to compute the structural closure, then performs numerical substitution as
a transpose solve on the explicitly stored `U^T`; Rust had performed both
parts directly on `U`. Preserving GLOP's split removes the first post-update
solve discrepancy in `scfxm1`, extends `lotfi` by another five common pivots,
and extends `sctap1` by fifty common pivots. The remaining `scfxm1` difference
is localized to the FTRAN at pivot 4. Its post-lower and post-middle-product
vectors agree exactly; the first mismatch was one ulp in the optimized
four-product block of the transpose-lower `U^T` solve. Disassembly of the
pinned ARM64 GLOP build shows one rounded multiply for entry `i - 1`, followed
by fused multiply-adds for entries `i`, `i - 2`, and `i - 3`, before subtracting
the block sum. Reproducing that exact contraction sequence makes all 367
`scfxm1`, 236 `sctap1`, and 369 `bandm` pivots identical while restoring
`israel`'s prior 147-pivot prefix.
The direct Rust translation of
`RightSolveLForColumnView()` is now enabled. Its initial `scagr25` failure was
caused by scattering through `ScatteredVector::set()`: a structurally reached
entry that solved to zero retained a Rust-only membership bit, so a later
rank-one update changed it to `1.2000000000000002` without adding it back to
the authoritative position list. Scattering directly into values and positions
and leaving the auxiliary mask cleared matches GLOP's temporary-cache protocol.
`scagr25` follows all 482 native pivots and the smallest-25 solve gate
passes status, objective, and primal and dual feasibility.
The next `israel` audit confirms that pivot 148 is an ordinary Harris-ratio
choice between columns 229 and 230: both update-row coefficients agree, but
accumulated reduced-cost rounding reverses their order. Tracing the preceding
exact recomputation localizes the first one-ulp difference to the dense
`U^{-T}` solve. ARM64 disassembly shows that its forward four-product block
starts with a rounded product for `i + 1`, followed by FMAs for `i`, `i + 2`,
and `i + 3`; Rust now preserves that contraction order too. The remaining
difference was storage-order, not algebra. A direct pre-conversion trace proves
that Rust's Markowitz `U` construction order already agrees with upstream
entry-for-entry and bit-for-bit, so the provisional sort has been removed and
triangular solves now observe GLOP's physical column order. This exposed an
earlier pivot-143 tie. Tracing BTRAN through `U^{-T}`, the rank-one middle
product, and `L^{-T}` localized the first drift to the middle-product solve:
GLOP's `GetColumnOfU()` does not expose physical storage directly, but copies
the requested column to a `SparseColumn` and calls `CleanUp()`, sorting that
temporary before constructing the rank-one update. Rust now likewise sorts
only the accessor copy while leaving the triangular factor untouched. The
affected rank-one vectors and multipliers become bit-identical, and `israel`
again selects native column 101 at pivot 143. A fresh audit then invalidated
the older attribution of pivot 148 to that exact solve: its input and output
are now bit-identical. The first incremental drift is at Phase-II iteration 97,
where upstream's column-wise update row uses its four-accumulator
`ColumnScalarProduct()` but Rust used a linear sum. Matching that reduction and
GLOP's policy of retaining incremental reduced costs across normal
update-count refactorizations makes pivot 148 exact. Pivot 149 is the next
target (native column 230, Rust column 236). A complete elementary-update
trace corrected its initial attribution to the rank-one portion of BTRAN: the
seven stored `u`/`v`/`mu` triples, input vector, scalar multipliers, every
intermediate vector, and the final rank-one output agree bit-for-bit. The
following `L^{-T}` result and every relevant update-row coefficient agree too,
as does the complete reduced-cost vector after pivot 148. A cache trace then
showed that pivot 149 computes the common leaving row 157 as a cache miss in
both implementations and produces the same `U^{-T}` vector; the complete
BTRAN and ratio-test inputs are exact. Columns 230 and 236 are an exact
entering tie. Native selects 230 and Rust selects 236 because Rust has consumed
two extra shared-RNG draws in `DynamicMaximum::UpdateTopK()`: earlier price
updates for rows 128, 105, and 90 meet Rust's heap threshold, while the native
heap sees only its row-162 threshold tie in the corresponding interval. The
remaining localization target is therefore the earlier dual-price/top-31 heap
state divergence, not BTRAN arithmetic. The native differential suite still
confirms the important path distinction that optimized GLOP contracts dense
rank-one updates while its sparse scattered path passes a separately rounded
product to `Add()`; Rust deliberately retains that behavior.
An operation-for-operation pricing trace now localizes the first price-value
difference to the first Phase-II dual-price recomputation. Its bounds and dual
edge norm are exact, but basic column 169 differs by one ulp. The complete
right-hand side passed to the post-Phase-I basic-value solve is bit-identical;
the one-ulp difference is introduced by the refactorized-basis right solve
itself. A stage trace proves that the lower solve and middle product are exact;
the dense upper solve was the mismatch. Upstream solves through its explicitly
stored transpose of `U`, while Rust directly traversed column-stored `U`.
Routing the dense path through `transpose_upper` makes all three stages exact,
restores native column 230 at pivot 149, and reconciles every remaining
`israel` basis through native termination at iteration 287. Rust also now
places the iteration-limit test after pricing and pivot validation, as
`DualMinimize()` does, so a model proved optimal exactly at the limit reports
`OPTIMAL` rather than `DUAL_FEASIBLE`. The boxed-variable update path now also
matches upstream's sparse-to-dense accumulation switch at 80% density; that
was a real algorithmic/performance discrepancy, although it does not change
this `israel` solve because the switch is not reached there.
The remaining `perold` pivot-139 tie was caused by a representation mismatch
in the optimized dual-edge `tau` cache. When the preceding lower solve became
dense, GLOP retained populated cached values with an empty nonzero list—the
dense-vector sentinel—whereas Rust rebuilt an exact sparse support while
copying the cache. The subsequent rank-one solve consequently selected a
different numerical kernel. Preserving the empty support makes the early tau
solve, incremental edge norms, pricing-heap Bernoulli draw at iteration 128,
and pivot-139 ratio-test draw agree. `perold` now follows native GLOP through
all 1,049 iterations and finishes with bit-identical ordered basis, reduced
costs, and dual norms; its remaining value-bit differences are signed zeros.
A fresh 10-second-per-model native audit has 96 exact terminal path
fingerprints—status, iteration count, and ordered basis—and two timeouts
(`qap12` and `qap15`). The same dense-sentinel correction also reconciles
`fit2p`. The `maros-r7` mismatch was caused by replacing the basis-factorization
object when its triangular crash basis failed the condition-number check. GLOP
reinitializes the same object with the all-slack basis and deliberately retains
the rejected crash factorization's deterministic-time estimate. Retaining that
estimate prevents a premature update-count refactorization at pivot 65;
`maros-r7` now matches native GLOP through all 4,954 pivots, with an identical
final ordered basis and bit-identical reduced costs and dual norms; its four
value-bit differences are signed zeros. The former `pilot`, `pilot.we`, and
`pilot87` mismatches were caused by rebuilding an already refactorized LU for
precision retries and by recomputing reduced costs after refactorizations for
which GLOP retains its incremental vector. Matching GLOP's conditional
refactorization state machine makes all three models exact in status,
iteration count, ordered basis, reduced costs, and dual norms; remaining value
differences are signed zeros.

The last five exact dual-norm discrepancies were caused by using the
non-clearing squared-norm reduction after inverse-row solves. GLOP calls
`SquaredNormAndResetToZero()` even though the workspace is temporary, and its
interleaved stores change optimized floating-point reduction rounding. The
faithful clearing kernel makes all 96 non-QAP models agree bit-for-bit in
status, iteration count, ordered basis, reduced costs, and dual norms in the
20-second trajectory audit; only signed-zero value differences remain.
These native results are now a durable regression oracle in
`baselines/netlib-dual-trajectories.json.gz`. The opt-in release test
`glop/tests/netlib_trajectories.rs` solves all 96 models with an independent
20-second wall-clock limit and checks status, iteration and basis-update counts,
ordered basis, primal values modulo signed zero, reduced-cost and dual-norm
bits, and every pivot tuple. A complete run passes in 38.56 seconds on the
October 2026 development machine.

The former `vtp.base` pivot-16 discrepancy exposed the symbolic/numerical split
in GLOP's hypersparse left solve. Matching that split and explicitly
contracting scalar tail updates makes the localized unit-row and tau stages
bit-identical and extends the common trajectory through 77 pivots. Pivot 78
then exposed an orchestration mismatch: after a pivot-triggered refactorization,
GLOP permutes and retains its incrementally updated dual norms, while Rust
permuted and then cleared them, forcing an exact recomputation. Retaining those
norms makes all 141 `vtp.base` pivots agree. Exact initialization still uses
GLOP's specialized transpose-factor norm solve rather than the general left
solve.

This is not yet a validated Phase-4 port. The primal Phase-I objective update
now follows GLOP's incremental `ReducedCosts` orchestration; dual
reoptimization after cleanup,
full termination/reoptimization checks, and complete incremental warm-start cases
remain to be translated. The reproducible `tools/validate_netlib_solve.py`
gate passes status, objective, and independent primal/dual feasibility checks
on the 50 smallest Netlib models. With a 60-second per-model limit, all 96
completed models in the full 98-model Netlib set pass the same checks;
`dfl001` and `qap15` time out. Native iteration counts still diverge, sometimes
substantially, so neither file is marked ported or validated in `PORTING.md`.
With the shorter 10-second development gate, 95 models complete and pass while
`dfl001`, `qap12`, and `qap15` time out. An audit of GLOP's degenerate-pivot
path also confirmed that faithful Phase-II bound shifts depend on the
post-optimization cleanup and dual-simplex reoptimization loop: enabling the
shift alone makes `grow7` primal-imprecise after cleanup. Until that loop is
ported, the provisional primal-only driver continues snapping the leaving
variable to its target bound; this deliberate scaffolding divergence must be
removed with the dual driver.

The optional initial dual cost perturbation is now connected before the first
dual solve. It shares the solver RNG with pricing, draws once for every
structural column (including variable types that do not retain a perturbation),
and uses the fused scale calculation emitted by the pinned optimized GLOP
build. A zero-iteration differential check of all 96 fast Netlib models now
matches native GLOP bit-for-bit in the ordered basis, every reduced cost, and
every dual edge norm. A focused solve also verifies that cleanup reports the
original objective. A full 96-model, 20-second-per-model perturbed audit
initially found 56 terminal-state discrepancies. The first `afiro` split was
caused by omitting GLOP's final Phase-I check when the basis was not yet
refactorized: GLOP clears cost perturbations and retries even without an
explicit cost shift. Restoring that check makes `afiro` match its native
14-iteration path, ordered basis, reduced-cost bits, and dual-norm bits.
Restoring GLOP's random selection among exact primal Harris ties fixes
`gfrd-pnc` too. The final `scsd6` discrepancy was downstream of another
unnecessary refactorization: when a candidate's precise reduced cost made it
invalid, GLOP's `MakeReducedCostsPrecise()` was a no-op on an already precise,
refactorized basis, whereas Rust forced a new LU factorization. That extra
factorization permuted the basis, reversing the traversal order of the two
tied leaving candidates. Rust now retains the current factorization in this
case and uses a conditional refactorization for the primal final check, as
upstream does. The full 96-model perturbed audit now matches native GLOP
exactly on status, iteration count, ordered basis, reduced-cost bits, and
dual-norm bits, with independent 20-second model limits. Focused `afiro` and
`scsd6` regressions guard the two resolved pivot-path defects.
The pinned native terminal states for all 96 perturbed solves are now stored in
`baselines/netlib-perturbed-dual.json.gz`; an opt-in release integration test
in `glop/tests/netlib_perturbed_dual.rs` checks input checksums, status,
iterations, basis updates, ordered basis,
primal values (modulo signed zero), reduced costs, and dual norms against it.
The native adapter does not expose pivot events in this mode, so this artifact
does not claim complete perturbed pivot-sequence coverage.

A new small-LP native branch fixture (`baselines/phase4-cases.json` and
`phase4-native.json`) now covers 130 targeted primal, dual, limit, ray, and
warm-start cases. Zero-pivot equality-row cases pin the structural bases
selected by triangular and Maros crashes in both simplex orientations;
separate cases pin Bixby's deliberate skip without scaling and triangular
rejection by the initial condition-number threshold. The Rust test requires
each case's diagnostic branch tags
to be visited and compares native status, iterations, objective, ordered basis,
values, reduced costs, and rays; it also rejects newly instrumented tags with
no case. The first fixture exposed two final-snapshot gaps: on primal
infeasibility, GLOP returns Phase-I reduced costs and objective rather than
the unsolved user objective; for unbounded outcomes it returns the appropriate
signed infinity. These are now aligned. `baselines/phase4-coverage.md` lists
the remaining uninstrumented and uncovered branches explicitly. Broad Netlib
agreement is not being treated as proof of complete Phase-4 branch coverage.
The new positive dual Phase-I iteration-limit case also validates native's
post-Phase-I refactorization, reduced-cost precision restoration, and basic
value recomputation even when Phase I stops at its limit. A dual pivot now
invalidates the retained reduced-cost precision flag, so this cleanup makes
the same 16 ns recomputation and reaches native's exact 220 ns total.
The latest pair exercises an actual residual-driven `IMPRECISE` exit and its
`change_status_to_imprecise = false` counterpart on the same small LP.
Additional cases now cover primal limits before and after Phase I, an already
optimal zero-limit dual basis, dual shifts and boxed flips, objective-changing
warm starts, and imported starting values with push-to-vertex disabled. The
primal-limit cases repaired two control-flow divergences: the iteration check
now follows precise entering-column pricing, and a positive limit exhausted
by Phase I prevents entering Phase II, as in pinned GLOP. An `INIT` result
after primal Phase I now restores the user objective before final checks.
Immediate deterministic-limit cases now exercise primal and dedicated-dual
Phase I. The dual case exposed an orchestration discrepancy: native GLOP
always enters dedicated dual Phase I, even with an initially dual-feasible
basis. Rust now enters it too, preserving native `INIT` status when the time
limit has already expired. Positive one-pivot primal and dual Phase-II
iteration limits now match native terminal state. Nonzero time limits and
other later-phase exits still need targeted fixtures. A two-row warm start also validates simultaneous bound
changes after loading the prior basis.
An externally supplied structural basis agrees with native GLOP. Two
added-column warm starts exercise both a structural and a slack basis from
the previous solve. The slack-basis case exposed incorrect alignment of saved
slack statuses after the new column shifted their indices; the status import
now applies GLOP's `num_new_cols` remapping when the prior structural matrix
is unchanged. The primal incremental path now retains the numerical
factorization, rebinds its matrix view, and shifts saved slack indices. Both
added-column fixtures match native status, trajectory, and operation clock.
Two added-row warm starts now also match native GLOP, starting from both
structural and slack bases. The dual incremental path now extends the saved
basis with new slacks and refactorizes it in the retained factorization object.
Both fixtures match native status, trajectory, and deterministic operation
time. The structural-basis case exposed an omitted condition-number check
after refactorization, which accounted for exactly 10 ns.
A third added-row fixture lowers the condition threshold between solves. The
incremental basis is rejected and Rust falls back to a fresh basis, matching
native status, iteration count, final state, and cumulative clock. The missing
68 ns was GLOP's `ComputeInitialBasis(candidates)` Markowitz pass; Rust now
executes that pass, uses its returned basis for the recovery attempt, and
charges its actual operation count.
Postsolve validation now follows GLOP's two unbounded-ray checks: the primal
ray is tested for a nearby blocking bound and weak objective gain, while the
dual ray's full row combination must prove infeasibility within the solution
tolerance. Six additional pinned-native cases now bracket both rejection
decisions: the weak primal ray becomes `OPTIMAL`, the weak dual certificate
becomes `IMPRECISE`, stronger rays retain unbounded statuses, and disabling
imprecise conversion preserves GLOP's corresponding status behavior.
Two more native cases bracket the optimal-cleanup residual check at nonzero
solution tolerances (`1e-18` rejects, `1e-16` accepts on the same LP). The
distinct residual-adjusted primal/dual infeasibility branches remain uncovered.
An exact two-row primal Harris tie now has an end-to-end native fixture in
addition to the existing isolated shared-RNG ratio-test unit test.
A one-row degenerate primal pivot also exercises GLOP's off-bound leaving
value: Rust now retains the resulting bound shift until cleanup instead of
unconditionally snapping the value to its target bound. The fixture matches
native final status, iterations, basis, values, and reduced costs. Its shift
does not force a shift-induced primal-to-dual cleanup switch. A separate 3×3
Harris-tolerance fixture now does force that switch and dual reoptimization;
its native status, basis, certificate, and exact clock agree. A tight-tolerance
one-row control still forbids the switch.
A two-pivot native fixture with zero reduced-cost recomputation threshold
also reaches the precision retry and forced basis refactorization; its final
state matches native GLOP. A zero pivot-refactorization threshold on that LP
did not reach the early imprecise-pivot branch. A separate 3×3 pinned-native
LP now does: the Rust branch visit and final LU pivot-threshold bits agree with
native GLOP, along with status, iterations, ordered basis, values, reduced
costs, and objective.

The cleanup loop now also follows GLOP's residual-aware terminal classification:
it checks primal equation residual and basic-column dual residual after removing
shifts, raises each requested feasibility tolerance to at least its residual
error, and marks an `OPTIMAL` result `IMPRECISE` when both primal and dual
infeasibility exceed those effective tolerances. The final status check applies
GLOP's `change_status_to_imprecise` guard after a time/iteration-limited solve.
Both 96-model release fixtures (ordinary pivot trajectories and perturbed
terminal states) still pass. This does not complete Phase 4: primal bound-shift
cleanup switching, full warm-start behavior, and the remaining
unbounded/infeasible termination checks still need direct porting and validation.
The Phase-I driver now retains its temporary objective, refreshes all basic
costs after refactorization or only the preceding direction's rows otherwise,
and zeroes the leaving variable's Phase-I cost after each pivot as GLOP does.
The coupled-row two-pivot native fixture exposed a separate Dantzig-pricing
gap: edge norms can skip update-row construction, but incremental reduced
costs require it. Rust now requests GLOP's update row before the reduced-cost
scatter. The full coupled-row solve and its one-pivot limit, an independent
two-pivot solve, and a positive Phase-I limit with infeasibility remaining
match pinned native status, iterations, ordered basis, values, reduced costs,
and objective.
The post-optimal starting-value push now has a Rust counterpart to GLOP's
`PrimalPush()`. Pinned native cases cover a free nonbasic variable moved to
zero and a constrained push that pivots the basis; both agree in status,
iteration count, ordered basis, values, and reduced costs. Warm-start
initialization now matches GLOP's snapping before push: an unused BASIC
candidate is made FREE, then snapped to its nearest finite bound when within
`crossover_bound_snapping_distance` (infinite by default). A pinned bounded
case checks that the push is skipped after snapping; with snapping disabled,
the bounded `PrimalPush()` arm moves starting values 0.5 and 0.75 to the lower
and upper bounds respectively, matching native state.
The solver also retains its existing basis factorization on a dual warm start
when only bounds change and the saved state is its own. One- and two-row native
cases exercise
that quick path, including the zero-pivot second solve. Added-row/column quick
paths are now covered; remaining push arms are tracked in the coverage ledger.
The unchanged-matrix quick path now also retains the ordered basis and LU for
primal warm starts when bounds are unchanged, including a changed objective.
It updates the objective and limit while keeping the primal edge-norm cache,
as upstream does. The same path handles a repeated dual solve with unchanged
bounds without rebuilding LU. Pinned-native cases for a repeated primal solve,
a changed-objective primal solve, and a repeated dual solve assert the reuse
branch and match status, iterations, ordered basis, values, reduced costs, and
objective. The dimension-changing quick-start cases are covered separately.
The quick path also refreshes both objective limits on every solve, like
GLOP's `InitializeObjectiveLimit()`. Two more native cases change the primal
and dual limits between solves and check their zero-pivot limit exits without
rebuilding the basis.
The saved basis state is now reused by default on a second `Solve()` call,
as GLOP does. Four native cases omit any `LoadStateForNextSolve()` call and
exercise repeated primal/dual solves, a changed primal objective, and changed
dual bounds; all match GLOP's status, iterations, ordered basis, values,
reduced costs, objective, and quick-path branch. An explicit state clear still
forces a fresh solve, checked by a fifth native case. A bounded BASIC warm
candidate at an upper-bound snapping-distance equality is also pinned.
An additional deterministic-limit fixture stops inside `PrimalPush()` before
its pending super-basic variable is moved. Native and Rust both return
`OPTIMAL` with zero new pivots, the same value, and the same operation clock;
the fixture requires the push-interruption branch and forbids a push pivot.
The state API now also retains GLOP's external-state distinction across
successive `LoadStateForNextSolve()` calls. Loading another basis and then
restoring the previously saved statuses must still take the external-basis
validation path, not the quick reuse path; a sixth native case verifies it.
The revised-simplex driver now charges deterministic work at upstream's
loop-entry and phase-exit boundaries, including basis factorization, update
rows, pricing, reduced costs, edge norms, and dual Phase-I price updates.
Two positive-limit native fixtures stop after finite work, and the fixture
test checks that the time limit receives the entire clock increment. Native
clock bits are pinned in the artifact for continued auditing. Exact operation
counts are **not yet fully faithful**. The audit reconciled lazy reduced-cost
recomputation, mandatory fresh dual solves, zero-iteration reoptimization,
cached update-row lifetime, post-optimization cleanup before validating
unbounded rays, PrimalPush's lazy reduced-cost invalidation, and the separate
"cached" versus "precise" reduced-cost states needed after an adaptive LU
rebuild. A pending full reduced-cost recomputation now also skips the
incremental `UpdateRow` work, matching the precision-retry path. Common
optimal primal/dual solves, zero-limit cases, repeated cleanup reoptimization,
precision refactorization, and ray branches now match native clock totals;
the fixture asserts exact totals for all 135 terminal small branch cases;
two more pin upstream LU error messages from cold and warm initialization.
The two singular-saved-basis recovery cases now also match the native clock:
upstream absorbs LU's column permutation before testing the initial condition
bound, so the permuted identity basis avoids an extra 8 ns norm charge. The
all-slack fallback now rechecks the condition-number
upper bound and returns an ill-conditioned LU error when it too is rejected,
matching upstream's error path; a focused test covers this case. The
remaining Phase-I positive-limit gaps came from eagerly computing the user-
objective reduced costs after Phase I stopped at `INIT`; upstream only
invalidates them and computes them lazily for the final snapshot. The weak-ray
fixture with `change_status_to_imprecise=false` now
skips native's guarded terminal residual check and lazily refreshes reduced
costs before its ray test, closing that case's final 4 ns gap. Exact clocks in
this corpus do not yet establish fidelity for every limit boundary. Four new
fixtures place deterministic limits at adjacent `f64` values around primal
Phase-I and dedicated-dual Phase-I loop transitions. They match native paths
and exact clocks: one versus two primal pivots and zero versus one dual pivot.
Two further adjacent-limit pairs bracket one-versus-two pivots in primal and
dual Phase II; both trajectories and exact clocks agree with native. The
first dual boundary exposed an eager reduced-cost solve before the first
dedicated-dual time check. Moving it inside the phase loop, as upstream does,
reconciled the boundary without changing the final operation total.
Two more deterministic-limit fixtures stop in cleanup after, respectively,
a primal bound shift and a dual cost shift. Both require the cleanup time-limit
branch and match native terminal state and exact operation clock.
Strong primal and tolerance-boundary dual rays are now also pinned with
imprecise-status conversion disabled. The strong primal case exposed the
unconditional dual-residual solve in native's ray validation. Rust now performs
that solve and, only for a weak ray, reuses the left inverse to refresh reduced
costs without a second transpose solve. Both new clocks agree exactly.
The dual objective-limit fixture is paired with an identical LP without a
limit. Both cold and warm early-limit clocks now match native: the current
dual optimization call performs cleanup before the objective limit prevents
another call, and terminal `GetReducedCosts()` reuses the left inverse from
`GetDualValues()` instead of solving again. A dedicated branch event pins the
cleanup. The primal objective-limit shortcut is now removed. A native debugger
trace confirmed that GLOP enters `RecomputeBasicVariableValues()` during
cleanup, then localized the apparent 8 ns gap to earlier Phase-II work:
Rust had recomputed reduced costs before checking the objective limit,
whereas upstream checks the limit first and returns without that BTRAN.
Moving the check before lazy pricing and restoring cleanup matches the native
clock, status, and final values. A same-LP no-limit control forbids the limit
and cleanup events and also matches native exactly.
Two additional native fixtures use a positive 1 µs wall limit that is reached
before primal and dedicated-dual Phase-I work. Both match Rust status and
operation clock. Wall limits reached after meaningful work remain untested.
End-to-end two-pivot fixtures now exercise both primal steepest-edge and Devex
pricing. A four-row steepest-edge fixture with zero norm-drift threshold also
forces exact norm recomputation and next-iteration refactorization. It exposed
an eager retry in Rust after `TestEnteringEdgeNormPrecision()`: upstream marks
norms/prices for recomputation but allows the current precise pivot to proceed.
Rust now does likewise, and its explicit pricing call sites honor the native
`PrimalPrices` watcher by returning before requesting norms when a full price
pass is pending. The case matches native status, basis, values, and exact
operation clock (830 ns), closing a 365.5 ns excess from premature work.
A separate feasible 4×4 dual fixture now takes the zero-threshold norm-drift
request through the next-iteration forced LU refactorization and optimal
cleanup. The same LP at the default threshold forbids both precision events;
both runs match native status, basis, numerical state, and exact clock.
A second 4×4 dual fixture covers early imprecise-pivot detection and adaptive
LU threshold escalation, with a matched default-threshold control. Both
agree with native on the final threshold bits, state, and operation clock.
The first optimization call is now explicitly tracked: when Phase I has already
reached a time or iteration limit, Rust skips the optimization cleanup that
native GLOP never enters. This removes one excess solve in the Phase-I
iteration-limit case. The remaining 4 ns gap was subsequently closed by
making the post-Phase-I objective reset lazy, as described above.
The transformed dual Phase-I path now uses the basic values recomputed during
`EndDualPhaseI()` rather than solving the same basis again during common dual
Phase-II setup. Its native clock now matches exactly.
The added-column and added-row quick warm starts now match native operation
accounting exactly. The row case needed GLOP's post-factorization
condition-number upper-bound check; both paths have explicit branch fixtures.
The condition-threshold fallback also matches its native clock after porting
the saved-basis Markowitz candidate pass. A two-super-basic-variable push
fixture now reaches the small-pivot refactorization request; a matched control
forbids that request. Both agree with native final states and two-pivot paths,
and now match native operation clocks. The apparent 28 ns gaps were in the
fixture adapter: it mistakenly enabled dual simplex for the new Rust cases
while the pinned native adapter used primal simplex. A matched no-push control
and an identical-LP no-start control make that configuration distinction
explicit. A separate duplicate final-snapshot BTRAN was removed from the Rust
solver, consistent with native's left-inverse reuse.
Two strict-tolerance native cases exercise cleanup's dual-to-primal switch:
one stops at the iteration limit after changing status, and the other enters
primal reoptimization and confirms optimality without a new pivot. The latter
exposed a retry loop in Rust's provisional primal final check. Rejecting a
tiny entering reduced cost invalidated the final check even though no pivot
changed the basis; the checked state now persists until an actual pivot.
A three-row strict-tolerance native case now exercises the converse
primal-to-dual cleanup switch and enters dual reoptimization, agreeing on the
three-pivot final state. Relaxing the internal tolerances to `1e-15` on the
same primal case and a corresponding dual case avoids their respective
switches; the fixture explicitly forbids those branch tags. Shift-induced
switches and the both-infeasible cleanup branch remain open.

The nondefault transformed dual Phase I now follows GLOP's auxiliary-bound
sequence: transform bounds and statuses using the current reduced costs,
recompute nonbasic and basic values, run dual optimization on the auxiliary
problem, restore the original bounds and values, and test dual feasibility
before ordinary Phase II. A focused dual-infeasible model exercises this branch.
The native/Rust `dual_netlib_trace` adapters accept a `transformed` mode, and
`tools/validate_transformed_dual.py` compares the 96 fast Netlib models with
independent 20-second limits. All 96 match exactly in terminal status,
iteration count, ordered basis, reduced-cost bits, and dual-norm bits. This
closes that nondefault Phase-I orchestration gap; the remaining Phase-4 items
above still prevent declaring the whole driver validated.

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

The optional upstream integrality-scale polishing path now has a direct Rust
implementation and nine native-pinned cases covering an improving bound flip,
an improving basis pivot, a randomized two-candidate rejection, a deterministic
limit before the first move, the no-scale control, and explicit scale clearing; terminal states and exact
operation clocks agree. The seventh case checks that polishing runs again after
primal-to-dual cleanup reoptimization; that call was missing from the first
Rust draft. A further two-candidate pair validates a post-pivot rejection and
a deterministic limit after that pivot. Its precision-request branch remains
to validate. The advanced
`MinimizeFromTransposedMatrixWithSlack()` entry point has seven native-pinned
functional cases: an initial solve, a changed-bound dual warm solve, a
changed-matrix rebuild, a nonzero slack cost with objective scale/offset, and
a zero-row case, a changed-objective warm solve, and an externally supplied
basis.
Status, basis, numerical state, and exact operation
clock agree. The advanced entry now supplies an equation-form view to shared
initialization, removing the cached synthetic `LinearProgram` and its
bound/objective update pass. Unchanged calls skip the sparse-matrix equality
scan, and full initialization keeps the compact numerical matrix transposed
directly from the caller. The basis representation now shares the same
`Rc<CompactSparseMatrix>` as numerical operations and factors selected
columns directly through a compact Markowitz/LU view. A 128-case generated
test matches sparse-view pivots, factors, and operation counts; `VariablesInfo`
also shares that compact allocation on ordinary and advanced initialization.
The seven pinned advanced cases and the ordinary and perturbed fast-96 Netlib
regressions pass. The advanced entry now supplies a borrowed compact equation-
form view, with no sparse-column compatibility matrix. Its changed-matrix
candidate is installed only after warm-start comparisons against the prior
compact source, preserving the upstream rebuild decision and clock.
`PrimalEdgeNorms` now shares the solver's immutable compact matrix, rather
than cloning sparse columns on initialization or rebind; the workspace tests
and strict lint pass.
`UpdateRow` now shares the advanced entry's compact transpose instead of
building a sparse transpose copy; ordinary full initialization builds a
compact transpose directly from the LP matrix. The pinned small-case and
numerical-kernel tests pass. Initialization and warm comparisons now accept
both the ordinary borrowed sparse LP input and the advanced compact input.
The solver's four direct update-row calls now consume the compact numerical
matrix; the column-wise update kernel uses its four-accumulator scalar product.
The focused numerical tests, pinned Phase 4 cases, and both fast-96 native
trajectory fixtures pass after this change.
Problem-column FTRAN now accepts compact columns directly through the shared
LU lower-solve kernel; the strict fast-96 native trajectory fixture still
passes. `PrimalEdgeNorms` now also shares the compact matrix and traverses its
columns for exact and incremental norms. Basis updates already retain a
compact problem-column index; they do not need a sparse replacement column.
The dual-unbounded certificate's full update row now also reads compact
columns. The former sparse-column compatibility copy is removed; ordinary LP
input remains borrowed only at the equation-form boundary.
The solver's residual, imprecise-pivot, and value-correction loops now use
compact columns, and `UpdateRow` constructs or rebinds its transpose directly
from the compact matrix. The equation-form adapter and warm-start comparisons
now read either borrowed sparse input or borrowed compact input directly.
The ordinary solver basis now uses the shared compact view as well. Added
columns rebind the view and shift slack indices without numerical rebuild;
added rows reinitialize from the compact source. The obsolete sparse-view
basis branch and its `Rc<SparseMatrix>` have been removed; selected sparse
factorizations remain covered only as isolated Markowitz test references.
The full workspace suite, strict lint, both
fast-96 Netlib trajectory fixtures, and the 100,000-pivot qap15 native-state
and clock fixture pass on this compact-view implementation.
The direct unscaled qap15 comparison now also passes a native-pinned
150,000-pivot cap, including exact ordered basis, reduced-cost and dual-norm
bits, and operation clock. Native and Rust each report `DUAL_FEASIBLE` at the
cap. This is deeper-prefix evidence, not a terminal solve.
The pinned Phase-4 branch suite now includes 159 small cases. Two additional
ray cases validate the opposite primal movement direction and the opposite
dual-certificate sign against native GLOP, including exact ray bits and
deterministic clocks. These extend, but do not complete, the outstanding
ray-configuration inventory in `baselines/phase4-coverage.md`.
Three native-pinned warm-start cases remove the first row, every row, or every
structural column after a solved two-row basis, validating reordered basis
coordinates, zero-dimensional reinitialization, and the slack-only
infeasibility certificate with exact operation clocks.
A zero-internal-tolerance cleanup case also matches native GLOP through its
dual-to-primal switch and reoptimization. The separate both-infeasible guard
has not been reached by a native-pinned LP; this is a non-gating branch-coverage
follow-up, not a missing implementation.
The alternate transposed-matrix entry now has a direct serial large-model
timing control. On `fit2p`, `maros-r7`, and `stocfor3`, its Rust/native median
solve-time ratios are 1.309, 0.958, and 1.546 (three trials and one warm-up
per solver); the ordinary-entry controls are 1.310, 0.957, and 1.547.
The advanced entry therefore has no measurable extra cost on this sample,
although the shared solver gap on `stocfor3` still needs profile-guided work.

Exit criteria:

- Status and objective agree throughout the smoke suites.
- Independent KKT/residual validation passes.
- Any iteration divergence is understood and recorded.

These criteria now pass for the unscaled Phase-4 scope: both fast-96 native
trajectory fixtures pass, the small native branch suite has 159 cases, and
the ordinary fast-96 regression independently recomputes objective, primal
equations and bounds, dual stationarity, and complementary signs from the MPS
input. The smallest-50 public-solver comparison passes status, objective, and
feasibility checks. This completion does not assert exhaustive branch
coverage, Phase-5 preprocessing/postsolve, or Phase-8 performance parity;
remaining hardening work is tracked in `baselines/phase4-coverage.md`.

The direct, unscaled `qap15` terminal solve and any prefix deeper than the
verified 150,000 pivots are **not** Phase-4 exit criteria. Keep their native
fixtures opt-in. Revisit `qap15` after Phase-5 scaling/preprocessing is in
place, first comparing the normal preprocessed solve and only then deciding
whether deeper direct, unscaled tracing is useful.

## Phase 5: port scaling and preprocessing

Status: complete for the continuous-LP preprocessing and default-solve pipeline.
`ShiftVariableBoundsPreprocessor` now ports GLOP's
first transformation: zero-domain detection, compensated row/objective
offsets, overflow status, and status-sensitive recovery. `ScalingPreprocessor`
ports the later equilibration and auxiliary-LP matrix methods, objective and
bound scaling order, and recovers
primal and dual values with exact original-bound snapping. Focused tests include
shifted, scaled, and combined solves checked against the original model. The
components now have thirteen pinned-native differential fixtures checking the
complete transformed model and recovered primal/dual values bit-for-bit,
including their composition, overflow, all four cost-scaling choices, and
four auxiliary-LP matrix-scaling cases, including an empty row and column.
Regenerate the fixture with the native adapter build script using
`--source tools/preprocessor_reference_adapter.cc --output
target/native/preprocessor_reference_adapter`, then run
`python3 tools/generate_preprocessor_fixture.py`.
They are now wired into `LPSolver` through the full ordered
`MainLpPreprocessor`: it copies the model, drops near-zero entries, runs
presolve and scaling, solves, unwinds postsolve in reverse, and validates in
original coordinates. Fifteen native-pinned complete solves through the full
pipeline agree on status, iteration count, recovered
primal/dual bits, and basis statuses. Six cases disable the singleton-column
initial-basis shortcut to isolate Phase-5 transformations; nine enable it,
including the default
initial-basis path and zero-pivot one-row controls.
Ten isolated bound-shift/scaling cases also have native-pinned capped-state trajectories:
independent solves at every iteration cap compare the ordered basis, status,
and recovered primal/dual bits (20 snapshots total, up to a three-pivot
combined shift-and-scale path). These snapshots are not presented as a direct
pivot-event trace, because a cap at a phase transition can reorder multiple
basis rows. A direct native pivot hook would strengthen this gate later.
Thirteen additional stand-alone passes now have pinned-native transformation and
postsolve fixtures: fixed variables, free constraints, empty constraints,
empty columns, singleton-column sign normalization, and forcing/implied-free
constraints, implied-free variables, doubleton-free columns, and doubleton-equality
rows, proportional rows, proportional columns, dualization, and unconstrained
variables. The empty-row infeasibility, empty-column unbounded, forcing
dual recovery, conflicting forcing bounds, and implied-free relaxation paths
are covered by forty-nine basic-pass cases. Doubleton-free cases cover both pivot
signs, a later column touching an already deleted row, and the no-change path.
Doubleton-equality cases cover induced and retained bounds, plus chained dual recovery.
Proportional-row cases cover coincident/split bound sources, negative factors,
dual transfer, and primal infeasibility.
Proportional-column cases cover merging, opposite factors, slope-based fixing,
infeasible slope intervals, and infinite-distance postsolve.
Dualization cases cover ranged-row duplication, shifted bounds, a one-sided row,
a free column, and recovered status conversion.
Unconstrained-variable cases cover finite-bound dominance, unbounded-cost
detection, and one- and two-row zero-cost postsolve.
The arithmetic fixtures distinguish
native fused objective/activity accumulation from the deliberately rounded
column-substitution delta. The singleton pass, including all four postsolve
undo operations, and the 20-pass main driver are now wired into `LPSolver`.
Forty-nine native basic-pass fixtures include singleton chains, complete driver
transformations, and adverse statuses. A pinned fixture now compares every bit
of the presolved model (bounds, objective, offset, and ordered sparse entries)
on all 96 fast Netlib models; all agree. Public solves on those models also
agree on status. Direct presolved-model comparisons localized an early
divergence to primal steepest-edge pricing. Using GLOP's scattered squared
norm, scattered BTRAN path, and four-accumulator fused compact-column dot
product reduced iteration-count differences from 29 to 16 of 96. A further
small-pivot retry correction reduced the count to 14: GLOP refactorizes the
basis without eagerly recomputing reduced costs or rebuilding the primal
pricing heap, whereas Rust had reset both. The reduced-cost reset changed
`grow15`'s cached cost for column
603 at pivot 249, suppressed its native precision-triggered refactorization
at pivot 297, and caused a later path split. `grow15` now matches native's
441 iterations. The eager heap rebuild consumed an extra tie draw at `grow22`
pivot 151 and changed its pivot-185 tie; `grow22` now matches native's 550
iterations. A further correction removed an eager primal-pricing reset after
`UpdateAndPivot()` refactorizes an imprecise pivot; native GLOP retains that
heap as well. A sparse precision-dot arithmetic difference remained: native
Clang rounds four products before adding them in
order and fuses the tail, while Rust used a uniform accumulation. Matching the
native grouping fixes `czprob`'s one-ulp discrepancy at pivot 264 and restores
native status and iteration counts on all 96 default-presolve Netlib models.
The release-only full-pipeline gate now enforces exact iteration counts rather
than allowing named exceptions. A further serial native/Rust public-solver
comparison over all 96 fast Netlib models checks status and iteration count,
both recovered basis-status vectors exactly, and objective, primal/dual
infeasibility, primal/reduced-cost/dual/activity vectors to relative or absolute
`1e-9`. All pass; 598,355 of 601,244 numerical vector entries also agree
bit-for-bit, and the largest normalized vector difference is `2.91e-11`.
The native outputs are pinned in
`baselines/phase5-netlib-solution-native.json.gz`; the release regression is
`cargo test --release -p gloprs-glop --test phase5_netlib_solution -- --ignored`.
Regenerate with `tools/generate_phase5_netlib_solution_fixture.py` after
building `tools/phase5_netlib_solution_reference_adapter.cc` against the
pinned upstream checkout. `tools/validate_phase5_netlib_solution.py` remains
the live side-by-side diagnostic.
`greenbeb` exposed an unfused sparse-vector dot product in forcing-constraint
postsolve; native-style fused accumulation restores its BASIC choices.
`pilot` exposed missing public-solver strong-optimal-guarantee snapping and
cost/rhs perturbation checks; those now follow upstream's pre-snap objective,
post-snap residual order. The native and Rust default pipelines agree on
`pilot`'s 3,567 iterations and recovered solution.
The main driver's retained pass stack now mirrors GLOP's one-allocation-per-pass
ownership instead of storing large concrete passes inline. Its non-destructive
`RecoverSolution()` counterpart is also present; native-pinned main cases check
that repeated recovery and destructive recovery produce the same solution.
The `MainLpPreprocessor` LP driver and Phase-5 exit criteria are validated by
the pinned pass fixtures, 96-model presolved-state gate, and 96-model public
postsolve check. Serial three-trial release measurements on Apple arm64
(`rustc 1.95.0`, Apple Clang 17) compare the public solver with preprocessing
and scaling enabled and disabled. On `greenbeb`, presolve speeds native by
2.07x and Rust by 2.05x; default-pipeline Rust/native time is 1.40x. On
`pilot`, the respective speedups are 1.39x and 1.40x, and the default
Rust/native ratio is 1.27x. Tiny `afiro` is dominated by overhead, and
`scsd6` slows with presolve on both implementations. Performance parity is a
Phase-8 concern, not an unmeasured Phase-5 claim. The former raw,
presolve-disabled `pilot` mismatch (4,427 native versus 4,680 Rust) came from
the primal singleton crash's equal-penalty `std::sort` traversal. A small
index-based port of pinned libc++ comparator-sort ordering now gives the same
initial basis and 4,427 iterations on raw `pilot`. The public objective and
infeasibility bits match, and recovered vectors are numerically identical
(remaining bit differences are signed zeros). Six native-pinned primal
iteration caps also match on ordered basis and reduced-cost bits.
Its native-ordering fixtures exercise insertion, partition, and heap-fallback
cases, and the 96-model default public-solver gate remains green. Normal
`qap15` with a 20-second wall limit returned
`PRIMAL_FEASIBLE` on both sides but 11,790 native versus 8,858 Rust iterations;
the time-limited result is not a deterministic trajectory gate and `qap15`
remains deferred to performance work.
As upstream, auxiliary-LP
scaling falls back to equilibration only when its internal simplex errors.
`LPSolver` retains and repopulates an owned model across calls, and objective
scaling allocates a magnitude vector only for the median method.
The pinned main order starts with bound shifting, then repeated presolve
passes, terminal empty/proportional/dualization steps, and finally scaling;
`use_scaling` is independent of `use_preprocessing`.

The transformations were validated incrementally with:

- direct unit tests;
- a reversible transformation/postsolve test;
- comparison of transformed model statistics with GLOP;
- solution recovery tests for optimal, infeasible, and unbounded cases;
- before/after performance measurements on affected Netlib problems.

Exit criteria (met):

- Enabling transformations one by one preserves independently validated
  solutions.
- The default preprocessing pipeline agrees with GLOP across Netlib.

## Phase 6: parameters and public API

Implementation status: complete. The 59-field `GlopParameters` schema has an
automated pinned-proto default/type check (`tools/validate_parameter_defaults.py`)
and the existing native validation-diagnostic comparison. The solver rejects
non-default settings for five runtime knobs without a faithful Rust behavior,
reporting the field name through `last_error()`. `LPSolver` exposes an
allocation-free borrowed `SolveResult`, owned basis export/import, cumulative
deterministic time, and parameter-preserving `clear()`; MPS loading remains in
`lp_data`. The CLI exposes raw/dual/default solve modes. Two runnable examples
cover constructed and MPS models. A native-pinned public-API fixture verifies
that basis import changes a two-iteration cold solve to a zero-iteration warm
solve with identical primal/status/objective values.

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

Exit criteria (met): all 59 defaults and five enum variant sets agree with the
pinned proto; constructed-LP and MPS examples run; and the public cold/warm
basis test agrees with the pinned native adapter on status, objective, primal
solution, basis statuses, and iteration counts.

## Phase 7: full differential validation

Audit implementation: `tools/phase7_netlib_audit.py` serially checks all 98
manifest-pinned Netlib MPS files against the native model fingerprints, checks
the 96 pinned and two live presolve hashes, and compares default-pipeline public results
against native traces: status, objective, iteration count, infeasibilities,
four solution vectors, both basis-status vectors, and deterministic time. It
also independently recomputes primal bound/equation violations from the input
matrix. `glop/tests/phase5_netlib_solution.rs` independently recomputes the
full original-model KKT conditions on the 96 fast terminal solves. The
checked-in report is `baselines/phase7-netlib-report.json`; non-matches have
stable IDs in `baselines/phase7-known-issues.md`. The first affected clock
prefixes on `d2q06c` and `sierra` are pinned by
`tools/validate_phase7_clock_prefix.py`.

The earlier 98-model terminal audit had 97 clean matches and one documented
`qap15` path discrepancy (P7-001). The former `d2q06c` and `sierra` clock
defects are repaired: Rust no longer marks the temporary sparse membership
mask while scattering the direction into the cached inverse. Their first
affected clock prefixes and the three-stage `d2q06c` transpose solve now
agree bit-for-bit. The audit also compares live native/Rust refactorization
counts on all 98 models; the 97 other terminal solves agree. The
96-model KKT gate passes, and the `qap15` terminal solution has a separate
opt-in original-model KKT gate. At that audit, native `qap15` took 74,317
pivots and 527 refactorizations versus Rust's 73,789 and 521. Fixed-cap
cleanup complicated localization; the 67,016/67,017 capped-state split was
not the first live divergence. P7-001 is now resolved: the first live
reduced-cost drift was at pivot 7,505 in the dense basic-objective/direction
dot product. Matching Apple Clang's contracted eight-block reduction and
maintaining GLOP's basis-row-ordered basic-objective workspace restores the
7,506 pricing heap, the 65,430 RNG state, and the full `qap15` terminal
fingerprints. Both implementations now reach `OPTIMAL` after 74,317 pivots
and 527 refactorizations, with identical ordered basis, variable values,
reduced costs, objective, and deterministic clock. The opt-in
`qap15_presolved_trajectory` gate pins these checkpoints.

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

Exit criteria (met): the full checked-in report and regression harness cover
all 98 models, no incorrect status or invalid solution is unexplained, and
P7-001 has a pinned terminal fixture and an opt-in presolved-trajectory
regression covering the former pricing/RNG splits and terminal state. Phase 7's
validation deliverable is complete, and the former `qap15` terminal fidelity
defect is resolved. Direct, unscaled qap15 dual termination is tracked
separately as a performance problem.

## Phase 8: performance parity campaign

An allocation-site audit across `glop/src` and `lp_data/src` found three
per-pivot/recomputation workspaces that Rust discarded while GLOP retained
them: dual entering-variable breakpoints, primal Harris candidate/tie lists,
and the scattered basic-objective left inverse. These now retain capacity
across calls; the dual entering wrapper also borrows reduced costs instead of
copying the full vector, standalone residual checks borrow their cached
dual/reduced-cost vectors, pending dual price rows are refilled in place, and
Devex-weight and reduced-cost-perturbation resets retain their buffers.
Constructors in LP parsing/presolve and in one-time simplex initialization
were separated from these iterative sites. Remaining candidates for later
profiling include objective clones at reduced-cost recomputation and the
packed middle-product update representation; these need ownership/algorithm
review rather than a mechanical `new()` replacement. No speedup is claimed
from the audit alone without paired timing measurements.

The subsequent serial fast-96 rerun (one warm-up, three alternating trials)
measures 0.923× Rust/native for parsing, 1.011× for presolve, 1.156× for
solving matched presolved models, and 1.153× for the public default pipeline.
Direct unscaled dual totals 0.967×, but this is dominated by `pilot87`; its
sqrt-nnz weighted geometric ratio is 1.185× and 90 of 96 models remain slower
in Rust. The closest prior full presolved-solve sample was 1.164×, so the
new 1.156× is not enough to attribute a meaningful gain to the allocation
changes alone. Raw trials and remaining outliers are recorded in
`baselines/phase8-performance.md` and ignored `target/phase8-*-workspace-audit-fast96.json`.

A paired kernel pass then found a stronger, reproducible `fit2d` signal in the
compact-column dot product: 1.7–1.8× native with safe entry-array reads.
Scoped unchecked reads of the matrix's internally paired row/coefficient
arrays, retaining checked RHS-vector indexing and the exact arithmetic order,
improved isolated new/old medians by 18–28% at 4–64 entries per column.
Focused same-code A/B solves improved `fit2d` by 3.2% and `dfl001` by 0.7%.
The fast-96 presolved-solve ratio fell from 1.156× to 1.129× and the public
pipeline from 1.153× to 1.122×, with unchanged status and iteration counts.
The strict native trajectory and public postsolve/KKT gates pass. Dense
lower-transpose solve is only about 1.11× native at dimension 1024 in the
existing isolated benchmark, so the compact dot was the clearer target.
Removing full-objective clones at reduced-cost recomputation brings ownership
closer to upstream but showed no convincing timing gain in a focused A/B.
On the final combined source, serial fast-96 ratios are 1.134× for the
presolved solve and 1.121× for the public pipeline; the small run-to-run
movement from the compact-dot-only sample is not assigned to the objective
move.
On all 271 saved `dfl001` refactorization probe groups, BTRAN is 0.991×
native across 563 sampled RHS while FTRAN is 1.107× across 275; all checksums
agree. Thus the synthetic lower-transpose gap does not currently explain
`dfl001`'s solver-level excess. Next profiling should examine its FTRAN,
factorization, and pivot-update shares before altering transpose substitution.
See `baselines/phase8-performance.md` for methods and raw-result paths.

The first serial, optimized, untraced comparison of the 96 fast Netlib models
(excluding `qap12` and `qap15`, three timed trials and one warm-up per solver)
found a 1.358 Rust/native aggregate solve-time ratio. The largest ratios were
`sierra` 2.833, `ship12l` 2.796, `80bau3b` 2.723, and `stocfor3` 2.447.
Profiling exposed several upstream correspondence gaps, now corrected: LU
edge-norm solves reuse sparse scratch space; hypersparse update rows iterate
set bits instead of scanning the entire column range; update-row density uses
the already-maintained relevant-entry count; dense triangular transpose solves
advance a contiguous cursor; direction workspaces persist across pivots; and
triangular transposes use a counting pass to fill contiguous storage directly.
The unchanged 96-model native trajectory fixture passes after these changes.

The same comparison rerun now totals 23.800 seconds for native and 29.175
seconds for Rust (1.226 aggregate ratio; median per-model ratio 1.408). The
largest absolute excesses are `dfl001` 1.748 seconds, `stocfor3` 0.580,
`truss` 0.498, and `pilot87` 0.405. The remaining gap is not yet explained or
accepted as parity: profiles of both `dfl001` and `stocfor3` point strongly to
repeated LU/Markowitz refactorization. At that baseline Rust constructed a new
`LuFactorization` and fresh Markowitz workspaces at each rebuild, whereas
upstream reused its factorization object and its allocated workspaces.

The factorization lifecycle now retains the `LuFactorization` object, the
Markowitz residual-pattern rows, candidate and degree-queue arrays, reusable
physical column pools, in-progress lower factor, and the four final triangular
factor/transpose buffers across refactorizations. A repeated A/B/A
factorization test and the 96-model strict native trajectory fixture pass.
Focused serial medians show only small gains: `stocfor3` remains about 1.61×
native and `truss` about 1.29×; `dfl001` is about 1.315× after the full
factor-buffer reuse (versus 1.335× after the first stage). Thus allocation
reuse was a fidelity fix, not the main explanation for the timing gap.

The factor-representation divergence is now resolved. Markowitz writes into
the retained `L` and `U` triangular matrices directly and applies the final
row permutation to their stored off-diagonal indices in place, matching
upstream `Markowitz::ComputeLU()`. The former sparse-column, tuple-column, and
factor-rebuild copies are gone; first-time candidate columns also fill their
reusable physical slot without allocating a replacement column. One thousand
native LU differential traces and the strict 96-model Netlib pivot-trajectory
fixture pass. The full three-trial serial Netlib timing rerun totals 23.777
seconds native and 26.225 seconds Rust, a 1.103 aggregate ratio (median model
ratio 1.318). `dfl001` is 1.246×, `stocfor3` 1.480×, and `truss` 1.293×;
remaining gaps require separate profiling rather than more factor conversion
or allocation changes.

A subsequent profile-guided pass matched two more upstream details without
changing arithmetic order: Markowitz's DFS uses a compact row bitset and
bucket clearing, and `TriangularMatrix::TransposeLowerSolveInternal` dispatches
unit versus non-unit diagonals once and handles the 1–3-entry tail with fixed
branches. The 96-model strict trajectory fixture and 1,000 native LU traces
still pass. A fresh full serial three-trial comparison totals 23.799 seconds
native and 26.100 seconds Rust, a 1.097 aggregate ratio; the median per-model
ratio is 1.325. The largest remaining absolute excesses are `dfl001` (1.21
seconds), `stocfor3` (0.41), and `truss` (0.34). Profiles of `dfl001` and five
repeated `fit2p` solves still put dense lower-transpose substitution at the top
of Rust's self-time. Both implementations now have the same traversal and
floating-point grouping, but a remaining code-generation/bounds-check cost is
only a hypothesis. An unsafe indexing experiment was rejected because
`lp_data` forbids unsafe code; a separate sparse-entry fetch rewrite showed no
timing benefit and was reverted. Further work should compare generated code
and isolate this kernel in a controlled microbenchmark before changing its
representation or safety policy.

The dense upper-transpose solve now also mirrors GLOP's one-time unit-diagonal
dispatch and forward cursor across contiguous columns, retaining the pinned
native four-product and tail arithmetic order. A targeted 0–4-entry column
test, 1,000 native triangular traces, and the strict 96-model release
trajectory fixture pass. A serial three-trial timing sample on `dfl001`,
`fit2p`, and `stocfor3` gives an aggregate Rust/native ratio of 1.279, versus
1.275 for those same models in the preceding full run; this control-flow
alignment has no discernible end-to-end timing effect.

An isolated lower-transpose benchmark now feeds identical generated factor
entries and right-hand sides to the pinned native kernel and release Rust
kernel, excluding construction and parsing from the timer. On this machine
(`Apple clang 17`, `rustc 1.95.0`, upstream `100f66e62`), the original safe
indexing loop took 1.48× native time at dimension 256 and 1.73× at dimension
1024 (3% lower-triangle density, 5,000 repeated solves, seven timed trials).
Native ARM64 disassembly has no per-entry bounds checks; Rust's original loop
had several for every four-entry block. Switching to checked per-column slices
and fixed-size reverse chunks preserves the floating-point order while
reducing the dimension-1024 ratio to 1.49×. The full 96-model serial
three-trial Netlib timing is 23.774 seconds native versus 25.980 Rust, an
aggregate ratio of 1.093 (previously 1.097), and the strict 96-model
trajectory fixture still passes. This does not establish that the remaining
kernel gap is entirely bounds checks: Rust still retains right-hand-side
checks and uses 64-bit row indices versus native GLOP's 32-bit indices.
The reproducible microbenchmark is `tools/benchmark_lower_transpose.py`.

A later bounds-check audit revisited that dense lower-transpose kernel after
triangular row bounds became construction invariants. `transpose_solve()` now
rejects incomplete matrices, so a full-length RHS and each validated stored
row index prove all hot RHS reads in range. One localized inlined unchecked
read eliminates the repeated per-entry checks, while debug builds assert the
invariant. In a five-trial matched microbenchmark at 3% lower-triangle
density, the dimension-1024 Rust/native ratio fell from 1.492 to 1.248
(7,684 to 6,364 ns per solve for Rust in separate contemporaneous runs);
dimension 256 fell from 1.479 to 1.383. A five-trial A/B of direct Netlib
solves against the prior committed build measured B/A median ratios 0.992
(`dfl001`), 1.002 (`stocfor3`), 0.994 (`truss`), 0.962 (`fit2p`), and 0.987
(`pilot87`). These are kernel and selected-solve results, not yet a full
96-model aggregate comparison. Compact-column dot-product checks remain:
its public slice API permits a vector shorter than matrix row count, so the
factor-row invariant alone does not make unchecked reads sound there.

The next Phase-8 pass tested the remaining row-width mismatch in that kernel.
`TriangularMatrix` now stores non-diagonal row indices in 32 bits, as pinned
GLOP does, while retaining the existing safe right-hand-side indexing and
exact arithmetic order. An isolated same-code layout A/B measured about an
8–9% gain at dimension 1024; the production lower-transpose benchmark moved
from 1.51× to 1.45× native at that size. This does **not** explain the main
simplex gap: a serial three-trial 96-model presolved-solve run is 11.941 versus
9.527 seconds, or 1.253×, essentially unchanged from the prior 1.254× result.
The 1,000-case native triangular and LU traces, workspace tests, strict
96-model pivot trajectories, and 96-model public postsolve/KKT gate pass. The full timing
artifact is `baselines/phase8-presolved-solve-u32-fast96.json`. Further work
should profile update-row and factorization costs on the largest absolute
outliers; storage width alone is not a performance-parity solution.

For the next numerical-kernel campaign, `glop/examples/netlib_simplex_probe.rs`
and `tools/generate_netlib_simplex_probes.py` provide opt-in Matrix Market
snapshots under `../datasets/netlib-simplex-probes/<instance>-<pivot>/`.
Each successful LU construction records its exact square input; subsequent
FTRAN/BTRAN right-hand sides are retained only before the next pivot and
before any rank-one update. A second square matrix records the solve basis
when post-LU column-permutation absorption changes its order. This diagnostic
is outside the pinned upstream algorithm and stays disabled in ordinary
solves. The format, naming, scope, and reproduction commands are in
`tools/netlib_simplex_probes.md`.
The complete 98-instance corpus was generated and validated: all 2,593 LU
construction counts agree with the pinned Phase 7 Rust report (one initial
construction plus each refactorization), and the 96 fast-model trajectory
gate still agrees with upstream.
`lp_data/src/matrix_market.rs` and the paired native/Rust LU harnesses now
read those exact inputs without reordering columns and time only factorization,
not file loading. `tools/benchmark_netlib_factorizations.py` runs the two
solvers serially and records per-matrix timings; build and measurement commands
are in `tools/netlib_factorization_benchmark.md`.
Two one-trial, one-factorization, no-warm-up full-corpus smoke tests succeeded
on all 2,593 probes with matching dimensions, nonzero counts, and determinant
finiteness (and close finite determinants). Timed totals varied: 44.608 s
native versus 43.740 s Rust in the first pass, and 37.550 s versus 41.582 s
in the second. These were coverage checks, not stable performance estimates;
the five-trial comparison below supersedes them. Both used
release builds on Apple M2 with Rust 1.95.0 and Apple Clang 17 against upstream
commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`.
A subsequent five-trial serial comparison (one untimed warm-up and one timed
factorization per matrix, alternating native/Rust order) produced paired
aggregate ratios 1.095, 1.093, 1.093, 1.088, and 1.114. The sum of
per-matrix medians was 36.720 s native versus 40.134 s Rust (1.093). The
`qap15` probes contributed 31.451 s versus 34.939 s (1.111), more than the
whole net difference; excluding `qap15` gave about 0.986, and excluding both
`qap15` and `qap12` gave about 0.86. Thus the next LU profile should focus
on `qap15`'s 522 large bases, while preserving the faster Rust paths on other
models. Raw per-probe and per-model CSVs are written under `target/` by the
runner; this is isolated factorization time, not full simplex time.

Paired sparse solve-probe adapters and `tools/benchmark_netlib_solve_probes.py`
now cover the same corpus's 161,608 FTRAN and 5,809 BTRAN right-hand sides.
They factor the recorded input before timing, verify the saved solve basis
against the LU column permutation, absorb that permutation exactly as GLOP's
sparse solve API requires, and time only paired sparse LU solve stages. A
complete one-trial, ten-repetition pass matched dimensions, input nonzeros,
and three solution summaries for every RHS. Summed per-RHS solve time was
0.4134 s native versus 0.5385 s Rust for FTRAN (1.303) and 0.2008 s versus
0.2649 s for BTRAN (1.319). The qap15 and qap12 probes dominate absolute
time. A three-trial alternating comparison of the four largest contributors (`qap15`, `qap12`,
`truss`, `dfl001`) confirmed a 1.300 FTRAN ratio on 37,653 RHS and a 1.310
BTRAN ratio on 2,435 RHS, again with no solution-summary mismatch. The probe
format does not encode the original dense/scattered/specialized solve choice; this
benchmark deliberately standardizes on the sparse two-stage path.

A subsequent profile-guided transpose-kernel pass removed repeated validated
RHS bounds checks from dense upper-transpose substitution and restored GLOP's
single reverse entry cursor in lower-transpose substitution. The latter uses
one documented unchecked factor-entry read helper; the safe-indexed cursor
version was slower and discarded. A one-trial, ten-repetition serial sweep of
`qap12`/`qap15` (23,817 FTRAN and 1,759 BTRAN RHS) now measures 0.3382 s
native versus 0.3707 s Rust for FTRAN (1.096) and 0.1770 s versus 0.2056 s
for BTRAN (1.161), with no solution-summary differences. All workspace tests,
1,000 differential triangular traces, and the strict 96-model trajectory
fixture pass. The remaining FTRAN/BTRAN gap is real; these isolated sparse
solve probes do not include simplex basis updates or the original call-site
choice among dense, sparse, and specialized solves. The complete 2,593-basis
corpus follow-up (one serial trial, ten repetitions per RHS) measured 161,608
FTRAN RHS at 0.4157 s native versus 0.4644 s Rust (1.117), and 5,809 BTRAN
RHS at 0.2016 s versus 0.2337 s (1.159), with zero solution-summary
mismatches. A final upper-transpose factor-entry change removed the matching
validated factor-storage checks. Its three-trial qap15 BTRAN microprobe fell
from 2.261 s to 1.904 s per 10,000 solves, versus 1.917 s native. Repeating
the complete serial corpus then measured 0.4137 s native versus 0.4658 s Rust
for FTRAN (1.126), and 0.2019 s versus 0.1996 s for BTRAN (0.989), again with
zero mismatches. Thus isolated sparse BTRAN is at parity on this machine;
FTRAN is much closer than the initial 1.303 but still about 13% slower.
`qap15` accounts for about half the absolute FTRAN gap (25.5 ms), followed by
`qap12` (6.8 ms), `truss` (3.4 ms), and `dfl001` (2.1 ms). These are single
full-corpus passes, not confidence intervals; repeat paired trials and profile
the residual FTRAN overhead before claiming full parity.

Small cross-crate accessors in `lp_data` and `glop` now carry ordinary
`#[inline]` hints, including typed-index conversion, bitset operations, sparse
iterators, and factor/view getters. No numerical algorithm or arithmetic
order changed. The workspace suite, strict linting, and the 96-model strict
trajectory fixture pass. On three paired qap15 trials, the 50-factorization
median moved from 3.084 to 3.065 s; 10,000 repetitions of a selected FTRAN
RHS moved from 1.853 to 1.830 s, and BTRAN was effectively unchanged. A
one-trial serial full-corpus solve pass measured 0.4157 s native versus
0.4590 s Rust for
FTRAN (1.104) and 0.2020 s versus 0.1999 s for BTRAN (0.989), with zero
checksum mismatches. The modest FTRAN gain is encouraging but these passes
do not establish a stable percentage improvement.

Profiling localized much of the `qap15` gap to the numerical update in
`TriangularMatrix::permuted_lower_sparse_solve`. Unlike upstream's unchecked
scratch-vector access, Rust emitted a bounds check for every successor row.
An isolated A/B/A/B build on `qap15-41066/factorization-0266.mtx` improved
the 200-factorization time from about 13.01 s to 12.29 s (roughly 5.5%) when
that check was removed. The production kernel now uses one documented unchecked
access, with row bounds enforced at triangular-matrix construction and mutation
boundaries; all other unsafe code remains denied. This addresses only part of
the measured gap, so repeat the full serial comparison and continue profiling
the remainder. All 522 `qap15` probes had matching native/Rust factor-entry
counts and column-permutation fingerprints before this change. A first
production-build pass over all 522 probes (one warm-up and one timed
factorization, alternating native/Rust order in 25-probe batches) measured
31.547 s native versus 33.156 s Rust (1.051), with all determinants agreeing.
This is an encouraging single trial, not yet a replacement for the earlier
five-trial baseline.

In a later single-basis qap15 profile, 200 LU factorizations of
`factorization-0266.mtx` took 12.750 s Rust versus 12.337 s native (1.033),
with matching determinants. The numerical candidate-column solve remains the
dominant hotspot; no further LU change was justified by that local sample.

The excluded `qap12` is now separately validated under matched settings.
Unscaled direct dual simplex terminates `OPTIMAL` in 115,513 iterations in
both pinned GLOP and Rust; sampled limits at 0, 1, 10, 100, 500, 1,000,
5,000, 10,000, 20,000, 40,000, and 80,000 have identical ordered bases,
reduced-cost bits, and dual-norm bits (apart from signed zeros in values).
One untraced release run took 44.49 seconds native versus 59.14 seconds Rust,
so this model has a large performance gap without an observed dual path gap.
The public LP-solver default is a different experiment: native GLOP uses its
preprocessing/scaling pipeline and took 22,308 iterations in the checked-in
baseline, whereas Rust's `LPSolver` intentionally omits that Phase-5 pipeline.
The former direct unscaled *primal* mismatch (23,367 native versus 23,276
Rust iterations) came from Rust's unconditional full reduced-cost
recomputation after an ordinary basis-update refactorization. GLOP retains
its incrementally updated reduced costs. On qap12, the extra Rust recomputation
erased an approximately `1.56e-8` accumulated error in entering column 3344;
native GLOP detected that error against its `1e-8` precision threshold and
refactorized after pivot 21,501. Rust skipped that branch, giving a different
basis permutation at pivot 21,502 and a different leaving column at pivot
21,530. Rust now preserves incremental reduced costs across ordinary
refactorization, as upstream does, and uses GLOP's sparse/dense scalar-product
order in the entering-cost check. Both direct primal solves now terminate
`OPTIMAL` after 23,367 iterations and match the native ordered-basis and
reduced-cost bit fingerprints at pivot 21,502 and termination. A full native
pivot hook also confirms that all 23,367 entering-column, leaving-column,
leaving-row, and iteration tuples agree. The opt-in `qap12_primal` regression
pins both complete pivot-sequence fingerprints and the terminal snapshots to
the native fixture.
`tools/compare_netlib_prefix.py` reproduces the localization without
modifying the pinned upstream checkout. Do not use the 22,308-versus-115,513
iteration counts as a direct native/Rust comparison.

The remaining outlier `qap15` has a separate direct, unscaled dual-prefix
fixture (`baselines/qap15-dual.json`). Native and Rust agree exactly on
status, iteration count, ordered basis, reduced-cost bits, and dual-norm bits
at 0, 10,000, 20,000, 30,000, and 40,000 pivots; their primal-value differences
through 20,000 are signed zeros. The 30,000-pivot cap ends `IMPRECISE` in
both implementations, with identical ordered basis, reduced-cost bits, and
dual-norm bits. The 40,000-pivot cap returns `DUAL_FEASIBLE` in both, with the
same basis and numerical fingerprints. The 20,000-pivot serial runs took
13.5 seconds native and 18.3 seconds Rust (1.35×). A serial 40,000-pivot
comparison initially took 141.7 seconds native and 240.8 seconds Rust
(1.70×). Sampling localized much of the Rust cost to the Markowitz
partially-permuted lower sparse solve. Iterating its DFS adjacency and numeric
column entries through borrowed slices removes repeated indexed-access work
without changing traversal or arithmetic order. The 40,000-pivot Rust solve
then took 177.5 seconds (1.25× native); all five pinned snapshots, including
the 40,000-pivot basis, reduced-cost bits, and norm bits, still agree. The
remaining runtime gap is open.
An October 2026 rerun on Apple arm64 with Rust 1.95.0 and Apple Clang 17,
after the lower-solve improvement, measured the direct 20,000-pivot prefix
serially at 16.09 and 16.15 seconds in Rust versus 14.32 and 14.00 seconds
in native GLOP (approximately 1.15× by medians). The Rust command used the
release `netlib_timing` example with `dual 20000 60`; the native command used
the pinned `dual_netlib_trace_reference_adapter` with a 20,000-iteration cap.
The Rust timer excludes model parsing; the native process wall includes parsing
and a larger trace written to discarded stdout, so its actual solve is slightly
faster still. A safe single-borrow rewrite of
the profiled triangular sparse-update loop did not materially change time and
was reverted. This narrows but does not explain the residual gap, and no
performance-parity claim is made from this prefix.
The opt-in `qap15_dual` regression checks these pinned snapshots. A native
direct solve with a one-million-iteration cap did not terminate within a
600-second wall bound, so this is not yet terminal validation. The public
GLOP solve time in `baselines/netlib-glop.json` includes scaling and
preprocessing and is not comparable to this direct Phase-4 experiment.
A deeper 100,000-pivot cap reached `DUAL_FEASIBLE` in native GLOP after 253
seconds, but the pre-optimization Rust port did not reach that cap within
300 seconds.
Those earlier runs did not establish 100,000-pivot equality; the new pinned
fixture and Rust regression now do, including status, ordered basis,
reduced-cost bits, dual-norm bits, and exact deterministic clock. Controlled
serial timing at this depth and terminal behavior remain Phase-4 work.
At 50,000 pivots, native took 171.8 seconds and Rust exceeded a 180-second
bound before the safe sparse-solve iteration change. A later direct,
unscaled 50,000-pivot run has now passed the pinned-native status, ordered
basis, reduced-cost bits, dual-norm bits, and exact deterministic clock.

The qap15 fixture now also pins native deterministic-operation clock bits.
Rust agrees bit-for-bit at 0, 10,000, 20,000, 30,000, 40,000, 50,000, and 100,000 pivots.
The former first split at pivot 28,149 was a cache-lifecycle discrepancy:
after a small-direction-pivot retry, `RefactorizeBasisIfNeeded()` leaves an
already refactorized LU and its cached `UpdateRow` untouched in GLOP. Rust
correctly skipped the LU rebuild but invalidated `UpdateRow` anyway, paying
for one unnecessary unit-row BTRAN. Conditional invalidation now matches
upstream. The `qap15-clock-transition.json` regression requires native clock,
status, ordered basis, reduced-cost bits, and dual-norm bits at 28,148 and
28,149; the seven-cap regression requires the same exact native clock through
100,000 pivots. Terminal behavior and deeper prefixes remain unvalidated.

Benchmark entry point: `tools/benchmark_netlib_solve.py` with release-built
`glop/examples/netlib_timing.rs` and the native
`tools/netlib_timing_reference_adapter.cc`. The timer excludes MPS parsing and
output, and the benchmark checks native fixture status and iteration count on
every trial. Local raw comparisons and profiles are in ignored `target/`.

The October 2026 Phase-8 campaign now has reproducible per-model artifacts in
`baselines/phase8-*.json` and a methods/results summary in
`baselines/phase8-performance.md`. Three serial timed trials and one warm-up
on the Apple M2 gave 1.086× Rust/native for the original 96-model direct-dual
baseline (23.80 versus 25.84 seconds) and 1.347× for the public default
pipeline (9.94 versus 13.38 seconds). Separate timers put parsing at 1.75×
(an absolute excess of 0.16 seconds), preprocessing at 1.47× (0.12 seconds),
and solving the already-presolved problems at 1.356× (3.38 seconds). Thus the
public gap is principally inside simplex, not a preprocessing copy or MPS
conversion. All 96 timed results retain native status and iteration count.

Profiling identified another representation mismatch: GLOP swaps a scattered
solution's storage with a reusable zero scratchpad when permuting and swaps
the cached pre-permutation left-solve values with its caller, whereas Rust
copied their dense arrays. Those swaps and `RightSolveForTau()`'s upstream
const-reference return now have safe Rust counterparts. Storage-identity unit
tests, six 300-case native basis-update suites, and the strict 96-model
trajectory test pass. A repeat of the already-presolved solves is 1.340×
(9.62 versus 12.89 seconds); the public pipeline remains 1.346× (9.68 versus
13.04 seconds). The direct-dual repeat is 1.104× (23.74 versus 26.22 seconds),
slightly worse than its prior 1.086× run, so no full-corpus speedup is claimed.
The swap helps `stocfor3`'s presolved solve in a focused comparison, but does
not account for the broader performance gap. A safe slice/chunk rewrite of
`CompactSparseMatrix::column_scalar_product_slice()` had no measurable benefit
and was reverted. Whole-process peak RSS, including parsing and runtime
overhead, is lower for Rust on every one of the 96 direct-dual models; its
median per-model ratio is 0.194. These memory numbers are not isolated solver
heap measurements.

A new matched profile of presolved `fit2d`, one of the largest absolute
remaining regressions (about 0.77 seconds per solve before this edit), put
about 73% of Rust samples in the compact-column scalar product used by update
rows and primal edge norms. Native spends the same stage in inlined
column-wise update code, but completes it faster. The new isolated benchmark
`tools/benchmark_compact_dot.py` measured 2.25–2.38× Rust/native at 4–64
entries per column. Disassembly found a redundant signedness check on each
Rust row-index conversion, in addition to the necessary safe vector bounds
check. Converting the 32-bit stored row directly as GLOP does preserves the
four-accumulator fused arithmetic and panics on invalid indices via checked
vector indexing. The isolated ratio fell to 1.81–2.11×, and a complete
matched presolved-solve rerun fell from 1.340× to 1.291×. The public default
pipeline fell from 1.346× to 1.301×, with all 96 statuses and iteration
counts retained and the strict trajectory fixture passing. A direct-dual
full-corpus rerun rose from 1.104× to 1.172×, driven largely by an anomalous
`dfl001` trial; alternating old/new Rust binaries on `dfl001`, `fit2d`, and
`stocfor3` instead measured the new code 4.5%, 9.5%, and 3.5% faster. Thus
the direct aggregate difference cannot be attributed to the row conversion
without better control of machine variability. The remaining scalar-product
gap likely involves safe gather bounds checks and generated code. GLOP also
declares this kernel inline in `sparse.h`, whereas Rust had retained an
out-of-line cross-crate function. An explicit inline annotation (supported by
the isolated benchmark) brings the kernel ratio to 1.73–1.83× and the full
presolved-solve ratio to 1.254× (9.81 versus 12.31 seconds). The public default
pipeline now measures 1.264× (10.32 versus 13.05 seconds). A direct-dual
three-model alternating A/B check found the inline binary faster on `dfl001`
and `fit2d` but slightly slower on `stocfor3`. The strict fast-96 trajectory
and public-postsolve KKT gates still pass. The separate lower-transpose kernel
gap also remains open.

A fresh serial fast-96 stage comparison on 12 October 2026 (one warm-up and
three alternating trials per model, release binaries) matched native status,
iteration count, and objective on every model. Summed per-model medians were
0.218/0.380 seconds for native/Rust parsing (1.741×), 0.260/0.382 seconds
for presolve (1.467×), and 9.616/11.192 seconds for the already-presolved
solve (1.164×). The solve excess is 1.576 seconds; `dfl001` (+0.521 seconds)
and `fit2d` (+0.354 seconds) account for over half of it. These timings are
a new machine-state sample, not evidence that the prior inline change alone
reduced the 1.254× ratio. The raw trials and dispersion are in ignored
`target/phase8-stages-inline-fast96.json`.

Matched eight-second samples of ten repeated presolved solves show different
dominant costs. On `dfl001`, Rust's triangular transpose solve has 1,212
top-of-stack samples; native's three transpose-lower/upper variants total
1,319. On `fit2d`, Rust spends most samples in primal edge-norm updates
(2,692), column-wise update rows (2,583), and primal price updates (1,005);
native has corresponding hotspots (2,573, 2,226, and 1,462). Sampling alone
does not isolate the excess time to one of these kernels, so no speculative
arithmetic or traversal rewrite was made. The next comparison should use
matched per-kernel timings/counters for these paths, including the already
known compact-dot gap, then rerun the strict trajectory and public KKT gates
after any change.

The next Phase-8 pass targeted stage overhead. `SparseVector`'s partial index
permutation now follows upstream's in-place stable compaction instead of
allocating and copying a temporary entry array during presolve row deletion.
The fixed-field MPS scanner borrows input slices, avoids an extra column-name
copy, and no longer tokenizes cards that it will parse by field position.
With the same fast-96 serial protocol, parsing now totals 0.218 seconds native
versus 0.199 seconds Rust (0.912×). `LinearProgram` row and column deletion
now also compacts metadata in place, matching upstream instead of rebuilding
bounds, types, and cloned-name vectors. A final presolve-only rerun totals
0.263 seconds native versus 0.267 seconds Rust (1.015×), down from 1.467×
before these compaction fixes. The stage benchmark checks matching parsed
dimensions and presolved status and dimensions. Raw stage trials are in
ignored `target/phase8-parse-presolve-final.json` (parser) and
`target/phase8-presolve-inplace-metadata.json` (final presolve). The remaining
presolve excess is only 0.004 seconds summed over 96 models; individual
outliers can still be profiled if they matter to a workload. The 1,000-case
free and 1,000-case fixed native MPS traces, 500 native `lp_data` traces,
strict fast-96 pivot trajectories, public fast-96 postsolve gate, workspace
tests, formatting, and Clippy all pass after these edits.

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

The follow-up ownership scan removed two transient sparse-row index copies in
the simplex price updates, one in `VariableValues`, and two retired-row
adjacency-list clones in initial-basis construction. Reduced-cost refresh now
reuses its retained left-inverse buffer instead of allocating a clone. These
changes preserve the stored traversal order. The serial fast-96 rerun on this
source measured 1.127× native for the matched presolved solve (previously
1.134×), 1.127× for the public pipeline (previously 1.121×), and 0.960× for
direct dual, whose aggregate is dominated by `pilot87`; see
`baselines/phase8-performance.md` for per-model and weighted results. These
small differences do not establish a measurable gain from the ownership edits,
so paired kernel measurements remain the next performance evidence to collect.

1. Continue Phase 8 with paired kernel-level measurements on `dfl001`'s
   FTRAN, refactorization, and basis-update shares, and `fit2d`'s remaining
   update-row/edge-norm work. Actual `dfl001` BTRAN probes are at parity,
   while compact-dot bounds-check work already improved `fit2d`. An isolated
   release disassembly and paired benchmark also found redundant positional
   bounds checks in the sparse-column update-row dot; four-entry slices retain
   exact arithmetic and accelerate 16-/64-entry columns by about 23%/34%,
   with no measurable change at four entries. The serial fast-96 rerun measured
   1.125× native for matched presolved solve (previously 1.127×) and 1.122×
   for the public pipeline (previously 1.127×); this does not establish a
   separate end-to-end gain from the edit. A subsequent bounded-access pass
   removed further repeated checks via safe paired views, chunks, and
   one-borrow read/modify/write updates. Its 1.131× fast-96 solve sample is
   slightly slower than 1.125×, but a paired five-trial Rust old/new check on
   four large contributors ranged from 0.995× to 1.002× with matching
   outcomes, so no material regression is established. Public APIs that allow
   out-of-range sparse indices still require checked gathers. Preserve
   upstream operation order and validate any edit with strict trajectories
   and the public-postsolve KKT gate.
   A subsequent paired probe run measured `dfl001` LU at 1.070×, FTRAN at
   1.101×, and BTRAN at 0.992×; `fit2d` LU is at parity. Their small absolute
   costs do not explain the solver gap. Serial native/Rust `fit2d` profiles
   localize most time to column-wise update-row, primal edge-norm, and
   primal-price updates. Column-wise update-row no longer clears its full
   coefficient vector each pivot, matching upstream's retaining `resize`.
   The five-trial Rust A/B was about 0.997× old on `fit2d` and `dfl001`;
   full fast-96 presolved solve and public pipeline remain 1.128× and
   1.124× native. Next isolate the per-pivot work in those three kernels
   with matched inputs and operation counts before changing their hot loops.
   Validating the immutable compact matrix's row indices once at binding
   allowed the primal edge-norm hot path to use upstream-equivalent unchecked
   gathers while retaining checked public/invalid-input behavior. Five-trial
   paired A/B gave `fit2d` new/old 0.947×; the full fast-96 matched solve and
   public pipeline now measure 1.113× and 1.109× native, respectively, with
   strict trajectories and postsolve/KKT unchanged. Threading the same token
   through column-wise update-row produced no reliable further gain (fast-96
   stayed 1.113×), so that extra wiring was removed. Next profile the
   remaining update-row kernel and pricing/edge-norm shares with matched
   inputs and operation counts; avoid broad unchecked access without an
   equally strong lifetime/dimension proof.
   The failed update-row extension had targeted the solver's later cached
   call, not the hot first call inside `PrimalEdgeNorms`. Passing its existing
   validated matrix token through those two first-call sites reduced paired
   `fit2d` solve time by 7.1% with bitwise-identical terminal outcomes.
   Fast-96 matched-solve and public-pipeline ratios are now 1.104× and
   1.090× native; strict trajectories and public KKT still pass. Continue
   with residual pricing and column-wise update-row profiling rather than
   broad bounds-check removal. A fresh `dfl001` sample after this change
   shows substantial time in both implementations' update-row, edge-norm,
   pricing, and transpose-solve kernels. Native's three transpose-solve
   instantiations collectively have more top-of-stack samples than Rust's
   single symbol, so this sample does not isolate transpose substitution
   as the source of the remaining gap. Use matched kernel inputs and paired
   timing before modifying that path.
   A literal branchless bitset-XOR translation of GLOP's primal-price
   candidate test was also measured and dropped: a serial paired A/B showed
   no gain (`dfl001` 1.004× and `fit2d` 1.009× old). Its short-circuit Rust
   form makes the same decisions, and the strict trajectory gate passed.
   A validated trailing-slack direct lookup in column-wise update-row was
   also exact but slower on paired `fit2d` solves (1.010× after hoisting
   validation), so it was dropped. The one-entry compact-column dot remains
   a semantic match; revisit only with a measured whole-loop specialization.
   The primal Harris first pass now matches upstream's sign-specialized
   template dispatch. Paired `dfl001` and `fit2d` solves were both 1.000×
   old; this is a fidelity improvement, not a measured performance gain.
   In primal edge norms, sparse direction positions now clear-and-extend a
   retained buffer instead of allocating a temporary vector each pivot, and
   exact norm recomputation resizes the retained norm vector as GLOP does.
   Paired timings were near neutral; the edits primarily correct allocation
   and storage-lifecycle fidelity. The serial fast-96 matched-presolve
   rerun measured 1.098× native (previous cross-run sample: 1.104×), with
   all statuses and iteration counts unchanged; strict trajectories and
   original-model public KKT checks pass.
   Binding the validated compact matrix and RHS once per update-row/edge-norm
   loop eliminates repeated identity and length checks without changing
   arithmetic. Seven-trial paired old/new solves improved `fit2d` by 2.4%
   and `dfl001` by 1.0%. The serial fast-96 matched-presolve ratio is now
   1.088× native (previous cross-run 1.098×), with strict trajectories and
   public KKT unchanged. The public pipeline remains 1.090× native in the
   serial fast-96 rerun, with all 96 path lengths matched. Next target
   residual short-column compact-dot
   overhead and the still-slower per-pivot pricing/update-row work using
   matched inputs.
   The middle-product update path now keeps GLOP's separate compact row and
   coefficient arrays, reuses the right scattered scratchpad, and appends its
   stored left column directly instead of building two temporary sparse
   vectors per pivot. The packed four-accumulator dot retains upstream's
   reduction order. All six 300-case native basis-update suites, the 1,000-case
   isolated rank-one trace, strict fast-96 trajectories, the public fast-96
   postsolve/KKT gate, and the release workspace suite pass. Paired direct
   append versus scratch-only timing improved `fit2d` and `dfl001` by 1.8%
   and 1.3%; the combined representation/lifecycle edit versus the pre-audit
   Rust binary measured 0.996 on `fit2d`, 0.972 on `fit2p` and `stocfor3`,
   and an inconclusive 0.995 on thermally variable `dfl001`. A serial
   fast-96 sample measured 1.063× native for matched solves and 1.060× for
   the public pipeline; cross-run differences are not assigned entirely to
   this edit. An isolated seven-trial rank-one right/left dense-solve probe
   with 32–96 updates and 4–64 entries per update places Rust at 0.827–0.996×
   native. Thus the packed update solve itself is not the next bottleneck on
   these patterns. Continue with matched update-row/edge-norm kernel timing
   and paired end-to-end evidence.
   A safe two-element slice for compact-column start pointers was tested as
   a way to collapse two bounds checks. The isolated bound-dot benchmark
   became slower at 16/64 entries per column (6.04→6.43 ns and 19.36→19.86
   ns); the change was reverted. It is not evidence that start-pointer checks
   explain the short-column gap.
   A matched native/Rust column-wise `UpdateRow` benchmark exposed a fidelity
   error in Rust's benchmark hook: it scattered its dense lhs while GLOP
   assigned the dense vector. After matching the hook, all 500 native traces
   still agree. With 2,048 structural plus 2,048 slack columns, validated
   Rust/native ratios are 1.201× for empty structural columns, 1.231× at
   four entries, 1.023× at 16, and 0.851× at 64. Subtracting the empty-case
   baseline leaves 3.03 µs Rust versus 2.30 µs native for the four-entry
   structural products. The unvalidated four-entry ratio is 1.344×. An
   exact-order short-column branch was slower and reverted. Next inspect
   generated short-column code and per-column overhead with this corrected
   hook; do not conflate input scattering with product cost.
   Hoisting the compact/sparse and validated/unvalidated dispatch out of the
   update-row column loop lowered the corrected validated four-entry hook
   ratio from 1.231× to 1.142× native, with 16/64-entry ratios now 0.962×
   and 0.829×. A serial paired fast-96 old/new presolved full-solve run (three
   trials and one warm-up) totaled 10.241 s versus 10.191 s (0.995×), with
   unchanged status and iteration count for every model. Focused seven-trial
   `fit2d` improved 2.4%, while `dfl001` worsened 0.9%, so this is a small
   aggregate gain, not universal. Strict fast-96 trajectories and public
   original-model KKT checks pass. Continue to profile `dfl001` separately.
   Opt-in production capture and native/Rust replay now isolate actual
   column-wise operands for `dfl001` and `fit2d` (captured under ignored
   `target/phase8-update-row-production/`). On a later `dfl001` call, a
   5,702-column replay showed a large `Vec::push`/output-path cost; GLOP
   instead pre-sizes its nonzero list and writes through a cursor. Mirroring
   that pattern improved the isolated Rust replay from about 27–28 to 21 us,
   but serial paired presolved solves did not improve: `dfl001` 1.004x,
   `fit2d` 0.992x, `fit2p` 0.997x, `stocfor3` 1.013x (new/old; seven timed
   trials). The production edit was reverted. Keep the replay to investigate
   the short-column dot and output interaction, but prioritize measured
   full-solve gains; the isolated kernel gap is not yet a demonstrated
   end-to-end bottleneck.
   The follow-up fidelity fix now stores the nonzero-position list at full
   column width with a separate occupied-prefix count, exactly as upstream
   does. All four population paths write into the retained buffer; same-size
   rebinds keep its allocation. A dedicated test checks the buffer pointer
   across repeated updates and a rebind. Five-trial serial paired presolved
   solves versus the append implementation gave new/old `dfl001` 1.003x,
   `fit2d` 0.988x, `fit2p` 0.998x, and `stocfor3` 1.009x, with aggregate
   1.000x. Retain this change for upstream fidelity and allocation stability,
   not a claimed whole-solver speedup.
   Fresh matched `fit2d`/`dfl001` native/Rust solve samples no longer show a
   clear pricing, edge-norm, or column-wise update-row excess after the above
   changes. In `dfl001`, the primal Harris ratio test was the clearer outlier:
   473 Rust top-of-stack samples versus 270 native first-pass samples in an
   eight-second sample. The Rust first pass indexed three parallel variable
   arrays independently. Asserting their equal lengths once before the scan
   lets the compiler share the index bound while retaining entirely safe
   indexing and GLOP's numerical order. A follow-up Rust sample fell to 243
   ratio-test samples. Five-trial paired presolved solves gave `dfl001`
   new/old 0.958x and `fit2d` 1.001x; the complete fast-96 three-trial A/B
   improved 10.131 to 9.914 s (0.979x) with all outcomes matched. The
   serial native/Rust fast-96 rerun measured 9.632 versus 9.980 s (1.036x),
   median model 1.070x and sqrt-nnz weighted geometric 1.066x. Strict
   fast-96 trajectories, public postsolve/KKT, release workspace tests, and
   Clippy pass. Next profile the residual absolute outliers `dfl001`,
   `fit2d`, `stocfor3`, `fit2p`, and `d6cube` rather than assume the prior
   update-row/pricing attribution still holds.
   A first matched `stocfor3` sample places Rust's largest costs in
   transpose triangular solve (1,069 samples), scattered permutation (324),
   scattered clear (298), and the primal loop; native's transpose-lower and
   transpose-upper variants contribute at least 1,153 samples, with another
   413 in a deduplicated symbol that needs attribution. Thus the 1.183x
   full-solve ratio should be localized with a matched `stocfor3` BTRAN/
   permutation probe before changing the transpose solve. The two samples
   are ignored `target/phase8-stocfor3-current-{rust,native}.sample.txt`.
   The 57 `stocfor3` saved basis groups contain 117 BTRAN and 6,924 FTRAN
   pre-pivot right-hand sides. Five-trial serial LU-only probe timings were
   1.144x and 1.287x Rust/native respectively, with equal checksums, but
   their tiny absolute times and absence of rank-one update chains cannot
   account for the full-solve excess. Inspecting the sampled RHS reset found
   a real upstream mismatch: Rust cleared each listed value and mask bit,
   whereas GLOP uses a 5%-density switch between sparse and dense clearing.
   Rust now uses that policy and clears its auxiliary mask before discarding
   positions. Seven-trial paired presolved solves measured new/old 0.963x
   `stocfor3`, 0.985x `dfl001`, 0.992x `fit2d`, 0.926x `fit2p`, and 0.989x
   `d6cube` (aggregate 0.982x). A separate serial three-trial fast-96 A/B
   measured 9.867 versus 10.001 seconds (0.987x, median model 0.990x),
   with matching statuses and iteration counts. Strict fast-96 trajectories,
   public postsolve/KKT, LU unit tests, formatting, and Clippy pass. The
   matched serial native/Rust fast-96 stage rerun totals 9.664 versus 9.860
   seconds (Rust/native 1.020x; median model 1.058x; sqrt-nnz weighted
   geometric 1.052x), improving on the preceding 1.036x aggregate run.
   The remaining `stocfor3` and LU solve gaps need updated-chain, matched-input
   profiling before changing transpose substitution or permutation kernels.
   In the updated BTRAN tau-cache path, upstream also uses the density-aware
   `ClearAndResizeVectorWithNonZeros()` for the retained pre-permutation
   intermediate. Rust still used `ScatteredVector::clear()` there; this is
   now aligned, with the Rust-only membership mask cleared first. A reuse
   regression covers a second RHS in the same cached workspace. Seven-trial
   paired presolved solves on the five outliers measured a modest 0.996x
   aggregate new/old (median model 0.998x); strict fast-96 trajectories and
   public postsolve/KKT pass. A three-trial full fast-96 A/B gave 0.984x
   aggregate but had an unusually high old `dfl001` trial (3.62 seconds),
   so that aggregate should not be interpreted as a stable 1.6% gain. The
   edit is retained for upstream lifecycle fidelity, not that noisy timing.
   A fresh serial direct-dual comparison on the two excluded QAP models uses
   the same pinned native and Rust parameters, a 20,000-pivot cap, three timed
   trials and one warm-up. `qap12` measured 6.938 s native versus 7.049 s Rust
   (1.016x); `qap15` measured 13.829 versus 13.910 s (1.006x), with matched
   status, iteration count, and objective in every trial. A single uncapped
   `qap12` terminal trial with a 120-second internal limit reached the same
   result in 44.461 versus 44.723 s (1.006x), versus the historical 1.33x
   ratio. A single 40,000-pivot `qap15` trial with a 240-second internal
   limit reached the same `DUAL_FEASIBLE` state in 141.020 versus 152.620 s
   (1.082x), versus the historical approximately 1.25x ratio after the
   lower-solve change. These longer single-trial ratios need replication.
   Serial 15-second late-run samples (after 60 seconds) on both sides put
   most sampled CPU in Markowitz refactorization and, within it, the
   partially permuted lower sparse solve. They may cover different basis
   rebuilds at the same wall offset, so use matched factorization probes to
   localize the remaining `qap15` gap before editing that kernel. Raw JSON
   timings and sample reports are ignored under `target/phase8-qap*.{json,txt}`.
   An isolated late-basis campaign restricted to the 124 saved `qap15`
   refactorization matrices at pivots 20,000–40,000 gives three paired
   Rust/native aggregate ratios of 1.036x, 1.042x, and 1.067x; summing
   per-matrix medians gives 9.147 s native versus 9.625 s Rust (1.052x),
   with identical determinants. Sampling 200 repeated factorizations of
   `qap15-32110/factorization-0208.mtx` puts both implementations in the
   permuted lower sparse solve and Rust somewhat more heavily there. The
   Rust DFS previously performed bounds-checked packed-bitset accesses for
   each graph edge, whereas GLOP's fixed-size bitset operations are unchecked.
   A kernel-local unsafe variant uses indices validated against the matrix
   dimension and has explicit safety comments. On the same 124 bases, three
   paired old/new Rust trials give 0.947x, 0.985x, and 0.986x, with median-
   summed 9.412 versus 9.274 s (0.985x); the first trial has a slow old
   outlier, so the credible gain is approximately 1.5%, not 5.3%.
   All determinants agree and the native-pinned `qap15` 40,000-pivot status,
   ordered basis, reduced-cost bits, dual-norm bits, and operation clock pass.
   All 57 `lp_data` release unit tests, LU unit tests, strict fast-96 native
   trajectories, public postsolve/KKT, formatting, and Clippy also pass.
   Important provenance correction: these saved matrices come from the
   *presolved and scaled* probe solve, while the 141/153-second 40,000-pivot
   comparison is *direct and unscaled*. They are matched inputs to the two
   isolated LU implementations, but not refactorizations of that direct
   solve; their roughly 0.48-second excess cannot be subtracted from its
   11.60-second gap. The 60-second-wall samples caught a refactorization-
   heavy window, not necessarily a representative whole-run profile.
   The next localization should measure stage totals across the same live
   solve (basis rebuild, updated-basis FTRAN/BTRAN, pricing, and other
   per-pivot work) before attributing the direct gap to any one kernel.
   To remove the probe-provenance mismatch, an opt-in direct/unscaled,
   factorization-only recorder now captures `qap15` with the exact dual
   40,000-iteration parameters used by the timing adapters. It yielded 1,826
   LU inputs (initial basis plus 1,825 refactorizations), reached the same
   `DUAL_FEASIBLE` status, and saved no RHS files. Three serial paired LU
   passes over **all** these matrices, excluding file loading, measured
   50.670/51.829, 50.522/51.687, and 50.716/51.874 seconds native/Rust:
   stable 1.023x ratios. Per-matrix-median totals were 50.181/51.375
   seconds (1.024x), with matched dimensions, nnz, and determinants on
   every matrix. Isolated LU on **materialized contiguous bases** differs by
   only about 1.2 seconds across the workload. This cannot be subtracted
   from the approximately 10–12-second direct-solve gap: the live LU factors
   a selected-column view of the original compact problem matrix, while the
   Matrix Market harness factors newly compacted, contiguous basis columns.
   The split at pivots
   20,000 and 30,000 gives native/Rust LU seconds of
   4.648/4.752, 25.250/25.818, and 20.283/20.805 respectively, so the LU
   ratio does not explode only in the late pivots. The next comparison must
   preserve the original compact matrix and selected-column indirection, or
   time the live refactorization stages directly; broad sampling alone cannot
   identify the source of the direct-solve gap.
   A one-trial iteration-cap sweep narrows the live gap: at 25,000 pivots
   native/Rust are 18.090/18.091 seconds (1.000x), but at 30,000 they are
   76.997/83.086 seconds (1.079x). Thus essentially the entire 6.09-second
   gap at 30,000 develops over pivots 25,000–30,000; the corresponding
   materialized-basis LU difference through pivot 30,000 is only ~0.67 seconds.
   Native `StatString()` reports 1,127 forced refactorizations by pivot
   30,000; Rust records the same 1,127 rebuilds plus the initial basis, so
   this is not a frequency discrepancy. Of the 822 LU inputs at pivots
   25,000–29,999, materialized-basis native/Rust LU time is
   23.750/24.281 seconds.
   Matched 40-second samples starting 21 seconds into each 30,000-pivot
   solve show both spending most sampled time in Markowitz's partially
   permuted lower solve, but the wall windows do not cover exactly the same
   pivots and do not establish which live stage creates the residual gap.
   A selected-view probe now also saves the immutable rectangular compact
   source matrix and each ordered basis-column selection. The paired LU
   adapters factor that view directly, with source loading outside the timer.
   A clean serial three-trial rerun over all 1,826 selections (after another
   CPU-intensive process ended) measured 51.372/52.479, 50.205/51.522,
   and 50.125/51.477 seconds native/Rust: 1.022x, 1.026x, and 1.027x.
   Per-selection median totals are 50.192/51.515 seconds (1.026x), with
   matched dimension, nnz, and determinant on every input. This is close to
   the 1.024x materialized-basis result, but both used the default 0.01 LU
   pivot threshold. The live direct solve raises this threshold to 0.9 during
   pivots 28,090–28,252. A retained-LU chronological selected-view replay
   of 823 late bases at threshold 0.9 reproduces a stable 1.112–1.113x
   native/Rust LU gap across three serial trials. Matched late-window stack
   samples put roughly 91–92% of both solvers' time under refactorization;
   pricing, ratio tests, and update solves are not the primary explanation.
   This was traced to a data-flow mismatch in first-visit Markowitz columns:
   native solves directly from the borrowed compact source column, whereas
   Rust copied it into the mutable residual buffer first. Rust now uses the
   borrowed source for both symbolic reach and numeric scatter. A three-trial
   old/new Rust replay on 200 high-threshold late bases is 0.871–0.872x;
   the two clean full 823-basis native/new Rust trials are 0.963–0.971x.
   Three live direct solves to pivot 30,000 now have a median 0.975x
   Rust/native ratio, with identical status, iterations, and objective,
   versus 1.072x before the fix. Strict fast-96 trajectories, public KKT,
   and the 30,000-pivot qap15 snapshot pass. The measured qap15 30,000-pivot
   high-threshold regression is closed. One clean paired 40,000-pivot live
   trial also measured 0.971x Rust/native with identical status, iterations,
   and objective; a second native timing was contaminated, so the longer-run
   distribution remains less certain than the 30,000-pivot result.
   The ~1.7 GB diagnostic corpus and raw results are ignored under
   `target/phase8-qap15-direct-*`.
2. The serial fast-96 stage and public-pipeline comparisons were repeated
   with per-instance trial medians and dispersion. Paired profiles localized
   part of the `dfl001`/`stocfor3` excess to transpose substitutions. A
   validated, upstream-ordered bounds-check reduction improved alternating
   old/new solves by 1.3%/1.9% on those models; all 96 strict trajectories
   and public KKT checks pass. The fresh full matched-presolve and public
   pipeline ratios are 1.013x and 1.009x. `fit2d` and `d6cube` did not
   materially improve (1.057x and 1.061x matched-presolve); `stocfor3`
   remains 1.111x despite a measurable isolated BTRAN/FTRAN gain. Continue
   paired kernel-level diagnosis of these residuals, without accepting
   desktop-load noise as a source-level speedup.
   Keep `qap15` outside deterministic wall-time gates until its performance
   is tractable. Details and raw artifact paths are in
   `baselines/phase8-performance.md`.
3. If a native auxiliary-LP solver error appears in production or a numerical
   stress case, retain it as a differential fixture. The injected-error fallback
   and whole-model native-equilibration output are already covered; a search of
   11,000 finite sparse cases found no natural failure.

The Phase-8 differential and timing harnesses now also select parser-verified
MIPLIB 2017 and Mittelmann LPopt MPS models with fewer than 150,000 constraint-
matrix nonzeros, excluding all Netlib names. The corrected 166-model corpus is distinct
from the pinned fast-96 gate; its MIPLIB members are LP relaxations. Parse and
presolve smoke comparisons pass across the full selection. The public-solver
status, iterations, objective, vectors, and basis statuses match native on
162 completed models in the initial 163-model pilot; `satellites2-60-fs` hit
the 20-second wall limit there. At a 60-second limit it completes identically
on both solvers. The matched-presolved-solve stage agreed on 161 pilot models;
`enlight_hard` is fully solved in presolve. A strict trajectory fixture and
reproducible multi-trial timing gate were the next steps; record any
native/Rust divergences as fixtures before treating this corpus as a
completion criterion.

A corrected parser-verified 166-model baseline now has one warmup and three
alternating native/Rust trials per model/stage at a 60-second solve limit.
All 166 parse/presolve comparisons match; 163 matched-presolved solves and
164 public solves have comparable outcomes. `enlight_hard` ends in
presolve, while `rmatr200-p5` and `nug08-3rd` remain wall-limited even in
separate 180-second public probes. Do not infer parity from their capped
elapsed times. The sum-of-medians Rust/native ratios are 1.034x parse,
1.038x presolve, 1.035x matched-presolved solve, and 1.030x public pipeline. The three
public trial-total ratios span only 1.0297–1.0338x. The largest credible
absolute gaps are `satellites2-60-fs`, `neos-827175`, both `brazil3` copies,
`unitcal_7`, and `comp07-2idx`; investigate these using paired profiles.
Pinned selection, trial data, build fingerprints, and caveats are in
`baselines/phase8-extra-performance.md`. A strict pivot-by-pivot fixture
for this corpus remains future validation work.

The first outlier pass repaired a concrete upstream-fidelity gap in the
doubleton-free-column sparse merge: direct structure-of-arrays construction
and storage swap cut `neos-1445765` presolve from 1.956x to 1.484x native
under matched 100-repetition trials. Dense LU permutation and triangular
symbolic traversal received smaller upstream-aligned improvements. Workspace
tests, strict fast-96 trajectories, and the complete 164-model extra-corpus
outcome gate pass. The leading nine public ratio outliers are improved in a
fresh three-pair run except `neos-950242` (roughly unchanged), but most still
exceed 1.2x. Profile their dominant BTRAN, primal-edge-norm, and refactorization
kernels with isolated paired timings before declaring the outliers resolved;
the measurements and model-by-model ratios are in
`baselines/phase8-extra-performance.md`.

The next outlier pass restored upstream's conditional-XOR bitset test in
primal pricing, with a focused boundary test and unchanged strict Netlib and
164-model extra-corpus outcomes. Six-model paired solve timings show only a
small `bnatt400/500` improvement; the other ratios remain effectively
unchanged. Unchecked sparse-merge reads did not improve `neos-1445765` and
were removed. A paired sample shows `fhnw-binpack4-48` refactorization is
costly on both sides, so its full 1.5x solve ratio still needs live-basis
kernel timing before a Markowitz change. See the follow-up section in the
extra-corpus performance report.
Captured `fhnw` basis matrices isolate a roughly 2.2x LU ratio, though their
tiny absolute timings and lack of a selected compact view limit attribution.
The residual-singleton copy and initial-singleton scratch setup now follow
upstream's one-pass/reusable-workspace implementation. This improves fidelity
but only slightly changes the isolated ratio. Next, profile the live selected
Markowitz path and reconcile the remaining LU gap without changing pivot ties.
The presolved selected-view benchmark reproduces a 1.97–2.06x LU ratio.
Using the existing libc++-compatible singleton sort and reusing row/column
permutation storage reduce it only to roughly 1.92x; the paired presolved
solve is 1.63x. Captured LU calls account for only about 2 ms of the roughly
11 ms full-solve gap. Continue with matched per-pivot stage timing to isolate
edge-norm, update-row, ratio-test, and solve work before more LU tuning.
The first per-pivot repair restores GLOP's Phase-I breakpoint heap (instead
of sorting all breakpoints) and traverses the known sparse direction support
for its infinity norm. The measured `fhnw` presolved-solve ratio drops from
about 1.63x to 1.575x; further per-pivot localization is still required.
After recycling LU pivot permutation buffers, a clean five-trial `fhnw`
selected-view LU run is 1.897–1.908x and the presolved-solve ratio is
1.572x. The major regression is not yet fixed; use matched per-pivot stage
timers or equivalent profiles to localize the remaining roughly 10 ms per
solve before changing further kernels.
An isolated LU profile then exposed a storage-lifecycle mismatch: native
`TriangularMatrix::Reset()` pre-sizes its diagonal, starts, and pruning arrays
and closes columns by indexed writes, whereas Rust grew all three arrays at
each pivot. The Rust factor now mirrors upstream, tracking completed columns
separately from capacity. Five paired selected-view LU trials improve from
about 1.90x to 1.70x, and the ordinary release presolved solve improves from
about 1.57x to 1.21x (about 17.7 ms native, 21.5 ms Rust per solve). This is
a substantial improvement but **not parity**; continue investigating the
remaining roughly 3.8 ms per-solve gap. A machine-target build is now only
about the same speed as the ordinary release build after this correction, so
the earlier apparent target-CPU win was code-generation sensitivity to the
old grow-on-close representation, not a reason to change portable build flags.
The next upstream-aligned Markowitz pass replaced the true-singleton generic
column copy with diagonal-only factor writes and kept the singleton coefficient
in a compact strong-index scratch entry, as native GLOP does. Residual-pattern
setup now counts and reserves before filling row adjacency, and the compact
view's empty test is constant-time. On 28 chronological `fhnw-binpack4-48`
selected-view matrices, paired release trials improve from about 1.69x to
1.53x native, but parity remains open. The finite-coefficient validation is
still present; removing it experimentally gave a further roughly 0.04x
improvement but would weaken the Rust API. The latest Rust sample places most
remaining CPU in inlined Markowitz work, with factor transposition and column
closing the largest separately named functions. Continue stage-localized
profiling and fidelity repairs; do not infer parity from the singleton gain.
The next row-only traversal alignment and one-read singleton scan bring the
28-view ratio to about 1.52x native in seven paired 500-repetition trials.
Temporary stage timing over the selected views puts Rust Markowitz setup at
12 ms, true singleton extraction at 243 ms, residual singleton extraction at
217 ms, residual-pattern construction at 68 ms, and numerical elimination at
1.4 ms across 7,000 refactorizations; the remaining factor-transpose work is
outside these stages. These are instrumented Rust timings, not matched native
stage clocks. A 32-bit residual-adjacency representation and compile-time
selected-view specialization were measured, regressed, and reverted. The
current paired matched-presolved whole-solve ratio is 1.194x (versus 1.215x
before this pass); continue investigating singleton extraction and factor
transposition, with strict trajectory and outcome gates after each edit.
The retained transpose no longer clears solve workspaces when populated;
upstream only replaces compact storage and pruning ends. This has a focused
dimension-changing regression test and no material effect on the selected-view
LU ratio (still about 1.52x). The remaining performance gap is not resolved.
Removing a second, non-upstream pivot-order representation from Markowitz
substantially improves the singleton-heavy case: direct permutations are now
written once, ownership moves to LU for inverse population, and their buffers
return to Markowitz for the next refactorization. Seven paired 500-repetition
selected-view trials give about 1.32x Rust/native (down from 1.52x); five
paired matched-presolved whole-solve trials give 1.167x (down from 1.194x).
The remaining gap is still open; profile the retained transpose and remaining
singleton column-write cost without reintroducing redundant permutations.
Line-level sampling of retained transpose population found its indexed
adjacent-start prefix sum dominating that routine. Expressing the identical
prefix calculation as one mutable-slice pass improves seven paired
selected-view LU trials to 1.128–1.135x native (sum-of-medians 1.131x), from
1.317x before that edit. The five-pair matched-presolved `fhnw` solve ratio is
now 1.146x. The residual LU gap was predominantly a Rust-only full scan of
every selected compact basis coefficient before each factorization. Compact
storage now maintains a nonfinite-entry count during construction, mutation,
reset, and transpose, and scans selected coefficients only if the source
contains any nonfinite value. This preserves the selected-column validation
contract (including nonfinite values in unselected columns). Seven paired
selected-view trials are now 1.021–1.031x native (sum-of-medians 1.025x);
matched-presolved solve is 1.124x. Isolated selected-view LU is effectively at
parity; investigate the remaining whole-solve gap at its other stages.
On `neos-1445765`, fresh seven-pair stage timings localized a 1.493x presolve
and 1.386x matched-solve ratio. The pinned upstream sparse-vector merge
reserves output storage and writes entries directly; Rust still pushed through
two checked `Vec`s. Indexed writes into reserved parallel buffers, followed
by single guarded reads of each source and accumulator entry, retain the same
merge order and drop rule. The unconditional sum of input lengths bounds all
raw output writes even if a public caller violates the common-index
precondition. Presolve is now 1.022x native in seven pairs, and the public
pipeline improved from 1.375x to 1.078x. In the simplex residual correction,
using GLOP's compact-column add instead of a hand-written entry loop improved
matched solve from 1.386x to 1.332x. DynamicMaximum now stores candidates in
the upstream-style packed bitset and iterates set bits once; that fidelity
change did not materially affect timing. The remaining public gap is about
8%, mostly in matched solve, and is not yet resolved.
Paired whole-solve samples of `fhnw-binpack4-48` place substantial time in
primal edge-norm BTRAN and the inlined primal-phase loop, not solely in LU.
The isolated dense rank-one solve microbenchmark is already faster in Rust at
small update columns, so the sample alone does not justify changing its
arithmetic. A concrete lifecycle discrepancy in the primal-residual check was
fixed: Rust now reuses GLOP's retained scratch vector rather than allocating
one on every check. Seven paired, 50-solve trials remain 1.123x native, so
this is a fidelity repair, not the explanation for the remaining gap. Localize
the primal-phase loop and BTRAN mix further with matched stage accounting.
