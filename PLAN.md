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
helper factor. The optional `LINEAR_PROGRAM`
matrix-scaling branch intentionally remains open because it calls the revised
simplex itself; a different optimizer would not be a faithful Phase-1
replacement.
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

Status: in progress. File-level Rust counterparts for `revised_simplex` and
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
is retained after refactorization and in the current phase-I driver, whose
feasibility objective changes during the iteration.
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
Imprecise pivots also increase the LU pivot threshold before rebuilding the
basis when fewer than ten updates have accumulated, matching `UpdateAndPivot()`.
The dual objective limit follows GLOP's shifted/scaled external
coordinates and is tested separately. One- and two-pivot regressions cover
optimal solves, a dedicated-phase-I regression starts from a dual-infeasible
basis, and a dual-ray regression covers primal infeasibility. The CLI and Netlib validator expose an
opt-in dual mode; all smallest-50 models pass status, objective, and independent
primal/dual feasibility validation with a 10-second per-model limit using the
dual driver throughout. The nondefault transformed-problem dual Phase-I
alternative is not yet connected and currently falls back to the primal driver.
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

This is not yet a validated Phase-4 port. The primal phase-I objective update is
not yet connected to GLOP's incremental `ReducedCosts` orchestration; the
nondefault transformed dual Phase-I loop and dual reoptimization after cleanup,
initial random cost perturbation, full
termination/reoptimization checks, and complete incremental warm-start cases
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
repeated LU/Markowitz refactorization. Rust currently constructs a new
`LuFactorization` and fresh Markowitz workspaces at each rebuild, whereas
upstream reuses its factorization object and its allocated workspaces. The next
performance-fidelity task is to make that lifecycle match upstream, then
reprofile and rerun the serial benchmark and strict trajectory fixture.

Benchmark entry point: `tools/benchmark_netlib_solve.py` with release-built
`glop/examples/netlib_timing.rs` and the native
`tools/netlib_timing_reference_adapter.cc`. The timer excludes MPS parsing and
output, and the benchmark checks native fixture status and iteration count on
every trial. Local raw comparisons and profiles are in ignored `target/`.

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

1. Begin Phase 3 with the remaining simplex-state components, preserving the
   same side-by-side source correspondence and native trace discipline.
2. Profile and reduce the measured roughly 1.8x--2.0x LU factorization gap, and
   add equivalent allocation/traversal/runtime benchmarks for basis updates.
3. Implement the revised-simplex-dependent `LINEAR_PROGRAM` scaling branch in
   Phase 5 rather than introducing a non-upstream placeholder.
