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

The optional upstream integrality-scale polishing path and
`MinimizeFromTransposedMatrixWithSlack()` entry point remain unported; the
ordinary continuous-LP solve does not invoke them without client opt-in.
These are explicit Phase-4 fidelity gaps, not validated branches.

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
The opt-in `qap15_dual` regression checks these pinned snapshots. A native
direct solve with a one-million-iteration cap did not terminate within a
600-second wall bound, so this is not yet terminal validation. The public
GLOP solve time in `baselines/netlib-glop.json` includes scaling and
preprocessing and is not comparable to this direct Phase-4 experiment.
A deeper 100,000-pivot cap reached `DUAL_FEASIBLE` in native GLOP after 253
seconds, but the pre-optimization Rust port did not reach that cap within
300 seconds.
No terminal-state or 100,000-pivot equality claim is made from those runs;
this large-model performance discrepancy remains part of Phase 4.
At 50,000 pivots, native took 171.8 seconds and Rust exceeded a 180-second
bound before the safe sparse-solve iteration change; that depth has not been
rerun since, so no trajectory comparison there is claimed.

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
