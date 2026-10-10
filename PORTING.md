# Porting inventory

Upstream commit: `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`

Status values are `not started`, `in progress`, `ported`, and `validated`.
“Validated” requires relevant unit tests plus differential evidence where the
component affects solver behavior.

## `ortools/glop`

| Upstream source | Rust destination | Status | Notes |
|---|---|---|---|
| `basis_representation.{h,cc}` | `glop/src/basis_representation.rs` | validated | Sparse LU plus both of GLOP's update modes are present: the default middle-product form built from cached scattered/hypersparse partial solves, and the nondefault product-form eta factorization selected by `use_middle_product_form_update`. As upstream does, the simplex-owned factorization now retains a view of the immutable problem matrix plus its basis-column mapping: a pivot replaces one column index in O(1), and LU refactorization traverses the selected original columns directly without cloning, cleaning, or materializing a basis matrix. An `Rc` provides Rust-safe shared ownership of the immutable matrix without copying its sparse storage; the older owning path remains only for standalone kernel tests. The port also includes the identity-basis shortcut, basis-column permutation absorption, refactorization, adaptive cached `RightSolveForTau()` intermediate, matrix/inverse norms, condition estimates, entry counts, and deterministic-time accounting. Repeated unit-row solves cache their pre-update `U^T` result in a persistent `CompactSparseMatrix` pool keyed by row. Their construction uses the direct counterpart of upstream `LeftSolveUForUnitRow()`, including its unit-RHS starting-column solve; this preserves numerically nonzero residual entries that a generic symbolic hypersparse solve can omit. Problem-column solves likewise store their post-`L`/post-update, pre-`U` intermediate in a second compact pool keyed by problem column; the middle-product update consumes both stored columns exactly as upstream does and falls back to refactorization if either is missing. Both pools survive rank-one updates and clear with LU rebuild/permutation absorption. Specialized temporary unit-row solves and factor-only primal/dual squared-norm solves use scattered LU paths and GLOP's exact input-density time charges. Six separate 300-case multi-pivot native suites cover both update modes under ordinary, fixed-period, and dynamically adjusted refactorization policies; they agree after every update and after a final forced refactorization on solves, exact update sparsity, specialized squared norms, temporary unit rows, condition estimates, and deterministic time. The audit found and corrected three representation-sensitive mismatches: rank-one `x += alpha u` updates use the fused operation emitted by the pinned optimized GLOP build, an empty eta sparse column selects the populated dense representation rather than representing a zero column, and scattered right solves finish with GLOP's conditional nonzero-position sort. `Clear()`, conditional `Refactorize()`, unconditional `ForceRefactorization()`, statistics accumulation/reset, and their state-retention semantics now agree in all six 300-case update suites |
| `dual_edge_norms.{h,cc}` | `glop/src/dual_edge_norms.rs` | validated | Exact recomputation, Koberstein incremental update with contracted arithmetic ordering, precision-triggered recomputation, lower bound, resize, and basis-permutation paths are ported; exact initialization calls GLOP's specialized row-indexed LU squared-norm kernel rather than allocating dense unit vectors and using the general transpose solve. Incremental updates consume the existing scattered BTRAN result directly, preserving its sparse/hypersparse support without a per-pivot allocation and dense scan. Five hundred native traces over independently row- and column-permuted sparse triangular bases agree on solves, precision decisions, adaptive cached tau solves, and incremental norms; the validated multi-pivot basis suites separately cover cached tau solves across updates and refactorizations. Full recomputation uses GLOP's same optional `TimeLimit` check only above 10,000 LU entries; a dedicated large trace asserts that threshold is crossed and agrees on the resulting partial norm vector after an immediate deterministic stop. The always-enabled accuracy statistic and pinned-release compiled-out statistics surface are included in the differential trace |
| `entering_variable.{h,cc}` | `glop/src/entering_variable.rs` | validated | Both dual phase-I and phase-II breakpoint heaps, Harris tolerances, stable-pivot preference, bound-flip accounting, minimum steps, deterministic-operation accounting, and exact-comparison tie detection are direct ports. Breakpoint ordering uses GLOP's ordinary comparisons, including equality of negative and positive zero, rather than Rust's sign-sensitive `f64::total_cmp()`. A 100-case native trace builds actual GLOP update rows and reduced costs, then agrees on phase-I/phase-II entering columns and ordered bound-flip candidates. An explicit-collaborator Phase-II entry point lets the revised-simplex owner reuse the same validated ratio test without a self-referential `ReducedCosts` borrow. All randomized simplex collaborators consume the shared port of GLOP's deterministic `std::mt19937_64` stream and libc++/Abseil distributions |
| `initial_basis.{h,cc}` | `glop/src/initial_basis.rs` | validated | Bixby, primal/dual triangular, and primal/dual Maros crashes retain the incremental `MatrixNonZeroPattern`, stability thresholds, source-order scans, and category/penalty rules. The triangular queue reproduces the pinned libc++ `make_heap`/`push_heap`/`pop_heap` behavior rather than Rust `BinaryHeap`, whose equivalent-key child choice changes bases. A 375-case native trace agrees exactly on all five returned bases and exposed two subtle requirements now preserved: `RestrictedInfinityNorm()` returns a magnitude, and Maros uses GLOP's literal row-to-column availability lookup |
| `lp_solver.{h,cc}` | `glop/src/lp_solver.rs` | in progress | Public no-preprocessing/no-scaling orchestration, basis import, solution extraction, compensated primal/dual objective calculation, original-coordinate activity/reduced-cost calculation, coordinate-scaled primal/dual infeasibility checks, expected-objective-error gap validation, complete solution/basis consistency checks, and both variable- and constraint-facet multiplicity detection are present. Certificate extraction now clears stale rays, gates them by terminal status, removes internal slack coordinates, and applies GLOP's dual-ray sign/row-combination convention. Focused regressions cover invalid status/value/basis combinations, constraint facets, maximization coordinates, an independently detectable objective gap, and primal-ray lifetime; the smallest-50 Netlib differential gate still passes. Phase-5 preprocessing/postsolve and the cleanup-coupled strong-guarantee perturbation checks remain; end-to-end validation is blocked on the incomplete revised-simplex driver |
| `lu_factorization.{h,cc}` | `glop/src/lu_factorization.rs` | validated | Contiguous sparse triangular L/U and explicit transposes, GLOP-direction permutations and inverses (including empty-as-identity after basis absorption), identity state, dense plus scattered/hypersparse direct/transpose and split-factor solves, initial-basis completion, norms, fill and diagnostics; 1,000 generated factorization traces and 500 rectangular initial-basis traces agree, and a large low-density trace directly covers hypersparse values and position sets. Sampling localized 75% of Rust factorization time to candidate-column solves; removing non-upstream candidate/pattern copies and duplicate scattered-workspace clearing reduced the large-case gap. In-process release benchmarks on Apple arm64 (Rust 1.95, Apple Clang 17; median of repeated trials) now measure Rust/native ratios of 1.72x, 1.83x, and 1.93x at n=500, 750, and 1000 with 2% generated density. Material optimization work remains, but there is no unexplained order-of-magnitude regression. `Clear()` retains parameters and last deterministic time, test-only `L*U` materialization is exposed, and exact Markowitz statistics strings are included in the 1,000-case trace |
| `markowitz.{h,cc}` | `glop/src/markowitz.rs` | validated | Square LU and rectangular permutation modes use two-stage singleton ordering, including GLOP's direct residual-singleton chain before the first general candidate solve, an incremental residual pattern, bucketed Zlatev queue, and cached sparse candidates backed by GLOP-style reusable physical-column pools. As upstream, the actual in-progress `TriangularMatrix` L owns the DFS-restricted partially permuted solve, persistent in-place dependency-graph pruning, swapped numerical traversal order, and operation counter; there is no parallel symbolic graph or duplicate unpruned L. Symbolic-zero behavior, pivot ties, contracted multiply-add ordering, and floating-operation accounting are preserved. Rust records that a cached candidate needs splitting while already traversing a singleton pivot row, retaining GLOP's constant-time cardinality fast path without letting an exact-cancellation support overestimate mask a deleted row. Residual degree removal and residual-singleton selection distinguish structural zeros from nonstructural exact-zero reachability overestimates rather than depending on incidental sparse-entry order; regressions cover both cases, `scagr25`, and the `scsd6` factorization path. Strict traces agree on deterministic time as well as factors for 1,000 standard, 300 dense 40–100-dimensional, and 500 nondefault-threshold/Zlatev cases; all four structural statistic distributions and formatting agree exactly in the native LU and repeated-refactorization traces |
| `parameters.proto` | `glop/src/parameters.rs` | ported | The complete pinned 59-field schema, five enum types, protobuf scalar types, and defaults are represented directly. Existing LU/Markowitz, basis-update, update-row, and edge-norm code consumes the shared bundle; 500 nondefault-threshold/Zlatev native LU traces agree. Rust does not reproduce protobuf field-presence/reflection machinery, which is serialization infrastructure rather than solver behavior |
| `parameters_validation.{h,cc}` | `glop/src/parameters.rs` | validated | Every pinned finite, nonnegative, not-NaN, integer, magnitude-limit, and Zlatev check is ported in source order with the exact diagnostic text. A dedicated native adapter exercises all validation branches and boundary categories; 124 cases agree byte-for-byte |
| `preprocessor.{h,cc}` | `glop/src/preprocessor.rs` | not started | Presolve and postsolve stack |
| `pricing.h` | `glop/src/pricing.rs` | validated | The lazily invalidated top-31 heap, monotone threshold, dense-update bypass, stale/duplicate filtering, exact-tie collection, and O(1) sparse updates are ported. A 500-case native operation trace agrees on maxima through sparse updates, removals, stale top entries, heap rebuilds, and dense updates. The heap is a direct vector implementation rather than Rust's `BinaryHeap`, preserving libc++ `make_heap` ordering, root-only equal-threshold replacement, and scan order. A focused tied-heap differential sequence also agrees exactly using the shared GLOP-compatible random stream |
| `primal_edge_norms.{h,cc}` | `glop/src/primal_edge_norms.rs` | validated | Dantzig column norms, exact steepest-edge recomputation, entering-edge precision checks, contracted Koberstein updates, precise-sum Devex updates/reset policy, lower bounds, recomputation watchers, deterministic-time accounting, and default parameters are ported. The Phase-4 primal driver selects the configured phase-specific rule, tests entering precision, and performs the upstream-ordered pre-pivot norm update through the shared `UpdateRow`; exact initialization now calls the LU-specific sparse squared-norm solve directly, removing the provisional dense allocation/general-solve path. Five hundred native traces using canonical structural-plus-trailing-basis layout with independently row- and column-permuted sparse triangular bases agree on column norms, edge norms, Devex weights, directions, precision decisions, watcher notifications, deterministic time, UpdateRow integration, and both incremental updates. Full recomputation uses the same optional time-limit check only above 10,000 LU entries; a large trace proves that branch and its partial result agree. Always-enabled accuracy/lower-bound statistics and pinned-release compiled-out measurements are included; their audit exposed and corrected the dedicated trailing-slack scalar-product path |
| `rank_one_update.h` | `glop/src/rank_one_update.rs` | validated | Upstream middle-product elementary `I + uv^T` representation, packed contiguous u/v storage, four-accumulator `CompactSparseMatrix::ColumnScalarProduct` numerical order, contracted dense updates, separately rounded scattered addends, chained dense and hypersparse/scattered right/left solves, 5% switching policy, entry accounting, singularity test, and deterministic timing are ported. The product-form `EtaMatrix` and `EtaFactorization` remain beside `BasisRepresentation`, matching their location in upstream `basis_representation.{h,cc}`. In addition to the six 300-case native basis-update suites covering both update forms and three refactorization policies, 1,000 isolated native traces agree on elementary multiply/inverse operations, packed dense and scattered factorization solves, sparsity state, entry accounting, clearing, and deterministic time |
| `reduced_costs.{h,cc}` | `glop/src/reduced_costs.rs` | validated | Basic objectives, dual values, full and relevant reduced-cost recomputation, residual-adjusted dual tolerance, incremental pivot updates, precision/refactorization flags, infeasibility summaries, pinned-native FMA-ordered perturbations, shifts, and `PrimalPrices` are ported. Rust owns the mutable objective and passes mutable norm/price collaborators explicitly instead of retaining C++ raw pointers; the numerical traversal and asymptotic update paths remain the same. `PrimalPrices` additionally accepts already-materialized reduced costs and norms so the Phase-4 driver can use the same heap without a self-referential owner; the optimization driver uses those entry points for GLOP's sparse incremental reduced-cost and price updates. A 100-case native state trace agrees on dual values and every reduced cost, a separate 100-case trace exercises both ratio tests through this object, 20 controlled three-pivot native primal traces agree on iterations, basic sets, final values, and reduced costs, and a direct unit test checks the extracted incremental value/heap path |
| `revised_simplex.{h,cc}` | `glop/src/revised_simplex.rs`, `glop/src/primal_ratio_test.rs` | in progress | The file-level driver now includes equation-form initialization, primal phase I and II, GLOP's default dedicated dual Phase I, and dual Phase II. The dual path uses sparse `DynamicMaximum` pricing, the incremental dual-infeasibility improvement direction, exact/incremental dual edge norms, BTRAN/update rows, phase-I and Harris bound-flipping entering tests, sparse reduced-cost/value updates, middle-product basis updates, boxed-variable flips, independent pivot checks, and final refactorization checks. Degenerate pivots use GLOP's minimum reduced-cost shifts and remove them with a refactorized reoptimization before accepting termination; early imprecise pivots likewise strengthen the LU pivot threshold as upstream does. Focused tests cover dedicated Phase I, one- and two-pivot Phase II, objective limits, and the public dual-unbounded certificate. The opt-in dual smallest-50 Netlib gate passes status, objective, and independent primal/dual feasibility checks with a 10-second per-model limit entirely through the dual driver. The full 98-model gate now has 96 validated solves and two timeouts (`qap12` and `qap15`) with no abnormal termination; this resolves the former `perold`, `pilot`, and `pilot87` failures. The primal path retains its validated pricing, edge-norm, reduced-cost, bound-flip, precision, and final-check behavior described in `PLAN.md`. A native/Rust iteration-prefix adapter proves that `perold`'s crash basis, initial LU permutation, post-permutation basis, and first 132 entering-column/leaving-basic-column/leaving-row choices agree exactly. The pivot-12 defect was dense-sentinel loss in the dual-edge tau solve; the pivot-46 defect was a linear `u dot v` fold where upstream constructs the middle-product denominator with its four-accumulator compact-column scalar product. A later defect was an unconditional post-Phase-I LU rebuild where upstream's conditional `Refactorize()` preserves an already refactorized basis; matching the no-op behavior removed basis-order and random-stream drift. Phase II now applies bound flips before the preceding direction's pending price updates, and dense bound-flip FTRAN results rebuild all prices and the lazy heap exactly as `VariableValues::UpdateGivenNonBasicVariables()` does upstream. Pending direction rows remain available across retry iterations, as upstream's `direction_.non_zeros` does, so repeated repricing preserves deliberate duplicate heap entries and shared-RNG draws; this restores `bore3d`'s native 128-iteration path and final ordered basis. Incremental reduced-cost updates and the four compact-column recomputation accumulators explicitly use fused multiply-add to match optimized native contraction; recomputation also follows GLOP's distinct structural-column and trailing-slack paths. Basic-objective left inverses use GLOP's scattered-row dense-sentinel left solve instead of the distinct packed dense transpose solve. These changes restore `blend`, make `capri` exact, move the first `scorpion` and `israel` divergences substantially later, and move `lotfi`'s first divergence from pivot 18 to 59. Rust propagates every incorporated basis permutation to the Phase-I pricing vector and dual norms and checks the iteration limit at GLOP's post-pricing/pre-pivot location. RNG, distribution, and isolated pricing-heap traces agree. Thirteen of the smallest 25 Netlib models have completely identical pivot sequences; pivot 133 is the next localized `perold` target and is an exact ratio-test tie exposed by remaining numerical drift in recomputed basic values and dual prices. The `vtp.base` pivot-16 tie is localized to three-ulp drift first appearing in the update-13 middle-product unit-row/tau solve. Degenerate Phase-II leaving values retain GLOP's bound shift; cleanup can switch to dual reoptimization when removing that shift exposes primal infeasibility. Initial dual cost perturbation now matches pinned GLOP's RNG draws and optimized fused scale: all 96 fast Netlib zero-iteration states agree bit-for-bit in basis, reduced costs, and dual edge norms. The nondefault transformed dual Phase I now matches pinned GLOP's auxiliary-bound sequence; all 96 fast Netlib terminal states agree exactly on status, iterations, ordered basis, reduced-cost bits, and dual-norm bits with a 20-second per-model limit. Dedicated dual Phase I now clears perturbations and retries when its basis is unrefactorized even without an explicit cost shift, restoring `afiro`'s exact 14-iteration native path. The primal Harris ratio test now consumes the shared RNG on exact leaving-row ties, restoring `gfrd-pnc`. The final `scsd6` discrepancy came from forcing a new LU factorization after a precise entering-cost check invalidated a candidate even though the current basis and reduced costs were already precise; unlike GLOP's no-op `MakeReducedCostsPrecise()`, that extra factorization permuted the tied leaving candidates. The primal final check now also uses conditional refactorization. A full perturbed 96-model audit agrees exactly on status, iterations, ordered basis, reduced-cost bits, and dual-norm bits, with focused `afiro` and `scsd6` regressions. Residual-aware cleanup, including shift-driven cross-algorithm switches, and final imprecise-status checks are ported, but remaining termination paths, complete warm starts, and large-model performance work remain, so this row is deliberately not marked ported or validated |
| `status.{h,cc}` | `glop/src/status.rs` | validated | All five unrecoverable status codes, exact names (including upstream's `INVALID_PROBLEM` spelling), success/message semantics, and Rust `Result`-oriented accessors are present and unit-tested. Problem, variable, and constraint statuses remain in the validated `lp_types` row, matching upstream ownership |
| `update_row.{h,cc}` | `glop/src/update_row.rs` | validated | Stateful cached scattered left inverse through GLOP's specialized unit-row solve, typed VariablesInfo relevance integration, drop filtering, upstream kernel-selection thresholds, column-wise, row-wise, hypersparse row-wise, single-row, full-update-row, nonzero tracking, operation counting, deterministic-time conversion, and benchmark selection are ported. A 500-case native trace agrees for all three product kernels and deterministic time, while the 500-case permuted-basis primal-edge trace covers the combined basis-solve and update-row path. The public statistics surface correctly remains empty for the pinned non-`OR_STATS` release build |
| `variable_values.{h,cc}` | `glop/src/variable_values.rs` | validated | Status-derived nonbasic values, sparse `-B^-1 A_N x_N` basic recomputation, residual and infeasibility summaries, pivot updates, sparse incremental nonbasic updates, primal phase-I costs, and dense/incremental dual infeasibility prices are ported. Mutable norm/price collaborators are explicit Rust method arguments rather than stored raw pointers. A 100-case native state trace agrees on every variable value and all three feasibility summaries; 20 controlled three-pivot native traces agree on final numerical state |
| `variables_info.{h,cc}` | `glop/src/variables_info.rs` | validated | Equation-form and allocation-free incremental structural/slack bound loading, type derivation, warm/default statuses, all movement/basic/relevance bitsets, relevant-entry accounting, boxed relevance, snapping, and dual phase-I transformations are ported with direct invariant tests. The advanced zero-copy mutable-bound API and `InitializeFromMutatedState()` are included. An expanded 500-case native trace agrees on types, statuses, every bitset, relevant entry counts, bound transformations, mutated-state reconstruction, and the structural/slack unchanged fast path. Rust owns its compact matrix to avoid a self-referential solver object; this changes lifetime organization but not behavior or traversal |

The `revised_simplex` row remains in progress. In particular, the opt-in
`SetIntegralityScale()`/`Polish()` path and
`MinimizeFromTransposedMatrixWithSlack()` entry point are not implemented or
differentially tested; ordinary continuous-LP solves do not invoke them.

Trajectory evidence in the `revised_simplex` row predating the October 2026
left-solve audit is superseded by `PLAN.md`. The audit found that the former
132-pivot `perold` prefix used a numerically equivalent but non-upstream solve:
GLOP obtains the hypersparse closure from the explicit transpose but performs
numerical transpose substitution on the original triangular factor. The port
now follows that split and contracts the scalar tail operations. This makes the
localized `vtp.base` unit-row/tau solve bit-identical. A subsequent audit found
that Rust cleared dual norms after a pivot-triggered refactorization even after
permuting them into the new basis order; GLOP retains those incrementally
updated norms. Removing that clear makes all 141 `vtp.base` pivots agree and
moves the common `scorpion` and `israel` prefixes to 162 and 106 pivots. Exact
dual-edge norm initialization separately retains GLOP's specialized
transpose-factor solve. The later `perold` audit found that the optimized tau
cache rebuilt an exact sparse support from GLOP's dense-vector sentinel. That
selected a sparse rank-one solve where upstream selected its dense kernel,
eventually changing a pricing-heap Bernoulli draw and the pivot-139 tie. The
cache now preserves populated values with an empty nonzero list exactly as
upstream does. All 1,049 `perold` pivots, the final ordered basis, reduced-cost
bits, and dual-norm bits now agree with native GLOP; remaining value-bit
differences are signed zeros.

The separate `qap12` audit uses identical direct-simplex settings rather
than comparing the native default LP-solver pipeline with Rust's currently
unpreprocessed `LPSolver`. Direct dual simplex reaches `OPTIMAL` in 115,513
iterations on both sides, with identical ordered basis, reduced costs, and
dual norms at the sampled prefixes and termination. The former direct primal
gap (23,367 native versus 23,276 Rust iterations) was a reduced-cost
lifecycle discrepancy: Rust recomputed all reduced costs after every ordinary
primal basis-update refactorization; GLOP retains the incremental values.
That erased the native `1.56e-8` accuracy warning on qap12 and suppressed its
reduced-cost-triggered refactorization after pivot 21,501. Removing the extra
recomputation and matching GLOP's entering-cost scalar-product order restores
the 23,367-iteration native primal path, including native ordered-basis and
reduced-cost fingerprints at the former pivot-21,502 split and termination.
All 23,367 native/Rust primal pivot tuples also agree. The opt-in
`qap12_primal` regression pins both snapshots and complete pivot sequences;
the reusable prefix
diagnostic is `tools/compare_netlib_prefix.py`.

The opt-in `qap15_dual` fixture separately pins direct, unscaled dual
snapshots at 0, 10,000, 20,000, 30,000, and 40,000 pivots. Status, ordered
basis, reduced-cost bits, and dual-norm bits agree with native GLOP; the
30,000-pivot cap returns `IMPRECISE` in both implementations and 40,000 returns
`DUAL_FEASIBLE`. Primal-value differences through 20,000 are only signed
zero. The native direct solve did not
terminate under a 600-second wall bound with a one-million-iteration cap,
so qap15 terminal validation remains open. The scaled/preprocessed
public-solver timing is not a direct comparison for this Phase-4 path.
At a 100,000-pivot cap, native returned `DUAL_FEASIBLE` in 253 seconds while
the pre-optimization Rust build exceeded a 300-second wall bound. This is
not a status or trajectory comparison at that depth; it is an unresolved
direct-solver performance gap.
The deepest completed serial trajectory comparison is now 40,000 pivots;
the 20,000-pivot timings were 13.5 seconds native and 18.3 seconds Rust.
At 40,000 pivots, native took 141.7 seconds and Rust initially took 240.8;
after safe slice iteration in the Markowitz partially-permuted lower sparse
solve, Rust took 177.5 seconds with the same pinned numerical snapshots.
Before that optimization, at 50,000 pivots native took 171.8 seconds and
Rust exceeded a 180-second bound; that depth has not been rerun since.

A fresh 10-second-per-model native audit has 96 exact terminal path
fingerprints—status, iteration count, and ordered basis—and two timeouts
(`qap12` and `qap15`). The correction also reconciles `fit2p`; the remaining
`maros-r7` discrepancy came from replacing the basis-factorization object after
rejecting its triangular crash basis. Upstream reinitializes the same object
with the all-slack basis and retains the rejected factorization's deterministic
cost for its dynamic refactorization clock. Rust now preserves that lifecycle,
avoiding its premature pivot-65 refactorization. All 4,954 `maros-r7` pivots,
the final ordered basis, reduced costs, and dual norms are bit-identical; its
four value-bit differences are signed zeros. The former `pilot`, `pilot.we`,
and `pilot87` mismatches came from treating GLOP's conditional
`RefactorizeBasisIfNeeded()` requests as unconditional rebuilds and from
recomputing reduced costs after routine and imprecise-pivot refactorizations.
Rust now preserves an already fresh LU and GLOP's incrementally updated reduced
costs in those paths. All three pilot models match status, iteration count,
ordered basis, reduced-cost bits, and dual-norm bits; remaining value-bit
differences are signed zeros.

The remaining five final dual-norm mismatches (`bandm`, `d2q06c`, `sc50a`,
`scfxm2`, and `stocfor1`) were already present in exact norm initialization.
Upstream finishes each inverse-row solve with
`SquaredNormAndResetToZero()`. Rust discarded the temporary afterward and had
used its non-clearing squared-norm kernel instead; the interleaved clearing
stores give the optimized native reduction a distinct rounding sequence.
Using the clearing kernel makes all 96 non-QAP Netlib models bit-identical in
status, iteration count, ordered basis, reduced costs, and dual norms under a
20-second differential audit. Their only remaining value-bit differences are
signed zeros.
The native results and every pivot event are now preserved in the checked-in
compressed fixture `baselines/netlib-dual-trajectories.json.gz`. The ignored
release-mode integration test `glop/tests/netlib_trajectories.rs` gives each
instance an independent 20-second wall-clock limit and compares all 96 statuses,
iteration and update counts, ordered bases, primal values modulo signed zero,
reduced costs, dual norms, and complete pivot trajectories exactly.
The corresponding perturbed-dual native terminal states are preserved in
`baselines/netlib-perturbed-dual.json.gz` with per-model input checksums. A
second ignored release integration test in `glop/tests/netlib_perturbed_dual.rs`
compares all 96 statuses, iteration and update counts, ordered bases, primal
values modulo signed zero, reduced costs, and dual norms under independent
20-second limits. Native pivot events
are not exported by this adapter, so this fixture does not validate every
intermediate perturbed pivot choice.
The revised-simplex cleanup and final status paths now check primal equation
and basic-column dual residuals and use residual-adjusted feasibility
tolerances before reoptimization or `IMPRECISE` classification, matching
`SolveInternal()`. Both 96-model release fixtures remain green; bound-shift
cleanup and the other termination branches remain in progress.
The 60-case pinned-native Phase-4 branch fixture covers the named primal,
dual, ray, limit, and repeated-solve paths in `baselines/phase4-coverage.md`.
It found and repaired final solution-snapshot mismatches on primal
infeasibility (Phase-I objective and reduced costs) and unbounded objectives
(signed infinity). The ledger names the unexercised branches; this is partial
coverage, not a completed driver validation.
The limit fixtures further aligned primal iteration-limit placement and the
Phase-I-to-Phase-II gate. The fixture also exercises dual cost-shift removal,
boxed bound flips, objective-changing warm starts, and supplied starting
values with upstream push-to-vertex disabled.
A separate 3×3 case now pins cost-shift removal followed by dual-to-primal
cleanup and primal reoptimization, with exact native state and clock.
Another 3×3 case removes a perturbation after dual Phase II and then switches
to primal reoptimization, explicitly without a cost shift; state and clock
match native GLOP. Immediate deterministic-limit cases additionally validate primal and dedicated
dual Phase-I exits. The latter repaired an entry-condition mismatch: dedicated
dual Phase I now runs even if the initial basis is dual feasible, matching
upstream's `INIT` outcome when its time limit is already exhausted.
A loaded-basis fixture with two simultaneous row-bound changes also matches
native status, iterations, basis, values, and reduced costs.
An externally supplied nondefault basis, two added-column warm starts, and
two added-row warm starts also agree. The added-column cases exposed
saved-slack-status misalignment, now corrected
using upstream's `num_new_cols` status remapping after confirming the old
structural columns are unchanged. The incremental path retains LU and rebinds
the basis view; the added-row path extends the saved basis and refactorizes it.
Explicit branch fixtures and native operation clocks validate both paths.
Primal Phase I now follows `UpdatePrimalPhaseICosts()` by refreshing all basic
costs after a refactorization and only touched direction rows otherwise. It
retains reduced costs across unchanged objectives and clears the leaving
nonbasic cost after a pivot. A coupled-row native fixture exposed that Dantzig
pricing can skip update-row construction in the edge-norm path; the driver
now requests the row before incremental reduced-cost scattering, matching
`ReducedCosts::UpdateBeforeBasisPivot()`.
The starting-value `PrimalPush()` path is now connected after optimal cleanup;
two native cases check zeroing an unconstrained nonbasic variable and a
one-row push pivot. A third case stops inside the push at a deterministic
limit, preserving the super-basic value and native `OPTIMAL` status without a
pivot. For dual warm starts with unchanged matrix and objective
but changed bounds, the solver now retains its ordered basis and factorization
and recomputes basic values from the saved state. Two native fixtures prove
that quick reuse branch is taken and its terminal state matches GLOP. The
added-row/column quick paths now agree too; the remaining push arms are tracked
in the coverage ledger.
Two strict-tolerance native fixtures validate the dual-to-primal cleanup
decision, including actual primal reoptimization. A no-pivot reoptimization
previously looped on a tiny reduced cost because the provisional primal final
check was invalidated when a precise entering candidate was rejected; the
driver now retains that check until a basis pivot, as GLOP does.
The converse primal-to-dual cleanup switch and actual dual reoptimization
also match a pinned three-row native case. Relaxed-tolerance counterparts
explicitly assert that neither cross-algorithm switch is taken, while matching
native final states. Shift-induced switches and both-infeasible cleanup still
need targeted evidence.
Positive Phase-II iteration limits now have one-pivot primal and dual cases;
both agree with native status, iteration count, ordered basis, values, and
reduced costs. The added-row cases likewise cover saved structural and slack
bases after the constraint count grows.
The final postsolve checks now calculate GLOP's primal blocking-distance and
cost-gain heuristic and the dual ray's full row combination and implied
infeasibility bound. Six additional native cases cover weak-ray rejection,
strong-ray acceptance, the dual tolerance boundary, and disabled imprecise
conversion on both sides.
Two additional cases bracket the optimal-cleanup residual test at nonzero
solution tolerances; they do not cover the subsequent residual-adjusted
primal/dual infeasibility comparisons.
An exact primal Harris leaving-row tie is now tagged and checked end-to-end
against native GLOP's selected basis and full final state.
For the `revised_simplex.{h,cc}` row, unchanged-matrix warm initialization now
retains the basis factorization on both upstream-supported quick paths: primal
simplex with unchanged bounds (including a changed objective), and dual
simplex with unchanged objective (including unchanged or changed bounds).
The 130-case pinned-native Phase-4 fixture exercises repeated primal and dual
solves, including a dedicated dual Phase-I stop after one pivot. It also checks
that triangular and Maros crashes select structural bases for equality rows
under both simplex orientations, without pivoting. That limit
case requires the upstream post-Phase-I cleanup and invalidation of reduced-
cost precision on the pivot; its final state and 220 ns operation clock agree.
Complete saved bases rejected by the initial condition threshold now undergo
GLOP's Markowitz candidate-basis retry before the all-slack fallback; the
corresponding native fixture agrees on terminal state and operation clock.
It also exercises a changed-objective primal solve and changed primal and dual objective
limits on quick starts, checking branch visits and final
state. Added-column warm starts retain LU; added-row warm starts refactorize
the extended basis. A lowered-condition-threshold fixture additionally
exercises recovery when that extended basis is rejected; its operation clock
now agrees after porting the Markowitz candidate-basis pass.
A degenerate primal pivot now retains its off-bound leaving value as upstream
does; a one-row native fixture reaches this shift and checks the resulting
final state. A 3×3 Harris-tolerance fixture now forces a shift-induced
primal-to-dual cleanup switch and dual reoptimization, with exact native state
and clock; a tight one-row control forbids that switch.
This supersedes the earlier `revised_simplex` row note saying the provisional
primal loop always snaps degenerate Phase-II leaving values to the bound.
A zero-threshold reduced-cost precision case reaches the retry and forced
refactorization, agreeing with native GLOP. Early imprecise-pivot escalation
is a distinct branch now covered by a 3×3 native fixture; its final LU pivot
threshold agrees bit-for-bit in addition to the terminal state.
The warm-start initialization now invokes the already ported
`VariablesInfo::SnapFreeVariablesToBound()` when unused BASIC candidates become
FREE. Native fixtures distinguish default snapping of a bounded candidate
from the bounded `PrimalPush()` path to either bound when snapping is disabled.
The same fixture also checks the bounded push in both directions and snapping
at an upper-bound distance equality. Rust now automatically retains its saved
state for a subsequent solve, like GLOP, rather than requiring an explicit
`LoadStateForNextSolve()` call; four no-load native cases and an explicit-clear
case validate this lifecycle;
a sixth case loads an external basis and then restores the previous statuses,
confirming that the external designation still bypasses quick reuse.
The revised-simplex driver now advances `TimeLimit` from six cumulative
operation counters at the native phase and loop boundaries. Positive primal
and dual limit fixtures agree with native status and iteration counts; each
fixture also checks that the elapsed deterministic time equals the driver's
clock delta. The native artifact records exact clock bits. The numerical
accounting is still **partially validated**. The lazy reduced-cost lifecycle,
fresh dual solves, reoptimization cleanup (including cleanup before ray
validation), PrimalPush's lazy reduced-cost invalidation, and skipping the
incremental update row when full reduced-cost recomputation is pending now give
exact native totals for common primal/dual, zero-limit, repeated-cleanup,
precision-refactorization, and ray fixtures. The differential test asserts
exact native clock totals for all 135 terminal small branch fixtures; two
additional cases pin upstream LU errors from cold and warm initialization.
The two singular-saved-basis recovery fixtures now match native clock after
absorbing LU's column permutation before the initial condition-number check,
as upstream does. The fallback all-slack basis now receives the same
condition-number check as upstream and returns an ill-conditioned LU error if
it also exceeds the threshold; a focused test covers this error path.
The final
Phase-I positive-limit gaps were an eager user-objective reduced-cost solve
after stopping at `INIT`; upstream only invalidates those costs and computes
them lazily for the final snapshot. Four adjacent-`f64` limit fixtures now
pin primal and dedicated-dual Phase-I loop boundaries. The dual pair exposed
an eager reduced-cost solve before the first limit check; moving the solve
inside the dedicated phase loop matches native zero-versus-one-pivot behavior
and exact clocks. Adjacent-limit pairs now also pin one-versus-two-pivot
boundaries in both Phase-II algorithms. Other limit boundaries remain untested.
When imprecise-status conversion is disabled, Rust now skips the terminal
residual check just as native does. The weak-ray check lazily refreshes stale
reduced costs before testing dual infeasibility, closing that fixture's
remaining 4 ns gap.
Strong primal and tolerance-boundary dual rays with imprecise conversion
disabled now have exact native clocks too. Primal ray validation computes the
dual residual even with conversion disabled; weak-ray reduced-cost refresh
reuses that left inverse. The compact matrix's slice scalar-product entry
point preserves the same four-accumulator fused kernel without copying the
dual vector.
An identical-LP no-limit dual control matches native exactly. Cold and warm
dual objective-limit cases now also match: the current dual call completes
cleanup before the limit blocks another call, and final reduced costs reuse
the dual-vector solve. A branch event enforces the cleanup path. The primal
objective-limit shortcut has now been removed. Native debugging confirmed
`RecomputeBasicVariableValues()` runs during that cleanup. The apparent 8 ns
gap was an earlier Rust BTRAN: it recomputed reduced costs before the
objective-limit check, while GLOP checks the limit first. Moving the check
ahead of lazy pricing and restoring cleanup aligns status, final state, and
clock; a matched no-limit primal control forbids both limit events.
Positive 1 µs wall-limit fixtures now also pin immediate primal and dedicated-
dual Phase-I interruption, with exact status and operation clocks. Wall limits
reached after substantial work remain uncovered.
Two-pivot primal steepest-edge and Devex driver cases now match native, and a
zero-threshold four-row case reaches exact edge-norm recomputation and the
following refactorization. Native permits the current pivot when its tested
norm is sufficiently precise; Rust no longer retries it prematurely.
`PrimalPrices` call sites also skip requesting norms while its full
recomputation is pending, matching the native edge-norm watcher. This removes
365.5 ns of premature work and makes the fixture's 830 ns clock exact.
The driver also tracks whether the first Phase-II call actually began before
the limit; it no longer performs native's post-optimization cleanup when Phase
I reached the limit first. The remaining Phase-I iteration-limit clock gap
was closed by deferring the new objective's reduced-cost computation until
its first actual use.
The transformed dual Phase-I path no longer repeats the basic-value solve
already performed by `EndDualPhaseI()`; its clock now matches native exactly.
In the `revised_simplex`, `basis_representation`, `primal_edge_norms`, and
`update_row` ledger entries, dimension-changing warm starts now follow
upstream's distinct shortcuts:
primal added columns rebind the saved factorization without refactorization,
and dual added rows extend the saved basis with slacks and refactorize it.
Both added-column and added-row fixtures match native clocks exactly. The
structural-basis added-row case exposed an omitted condition-number upper-bound
check after refactorization; adding it closed the 10 ns gap without adjusting
the clock.
A low-condition-threshold added-row fixture now exercises rejection of the
incremental basis and recovery with a fresh basis. Native/Rust final states,
iterations, and operation clock agree after porting
`BasisFactorization::ComputeInitialBasis(candidates)`, including use of the
returned basis and its Markowitz operation count.
The two-variable `PrimalPush` refactorization fixture and its no-refactor
control now match native trajectories, final states, and operation clocks.
The apparent 28 ns gaps came from an adapter error: Rust was configured to
run dual simplex for these fixtures while native ran primal simplex. A
matching no-push control and an identical-LP no-start control guard this
configuration. A duplicate final-snapshot BTRAN was also removed from the
Rust solver to match native's left-inverse reuse.

The October 2026 solve-path audit also reconciled
`LuFactorization::RightSolveUWithNonZeros()`: upstream computes reachability
from `U`, then performs numerical substitution as a transpose solve on the
explicitly stored `U^T`. Rust now preserves that symbolic/numerical split
instead of carrying out both stages directly on `U`. The fresh smallest-25
audit now has 20 completely identical trajectories. A stage trace localized
the next `scfxm1` difference to one-ulp code-generation drift in the optimized
four-product transpose-lower solve. ARM64 disassembly identifies the exact
native sequence as one rounded multiply for `i - 1`, then FMAs for `i`,
`i - 2`, and `i - 3`. Reproducing it makes the complete `scfxm1`, `sctap1`,
and `bandm` trajectories identical and restores `israel`'s 147-pivot prefix.
A follow-up `israel` trace localizes the pivot-148 tie to reduced costs from a
fresh `U^{-T}` solve. The forward four-product kernel now also matches the
pinned ARM64 contraction order (rounded `i + 1`, then fused `i`, `i + 2`, and
`i + 3`). Direct factor traces show that Rust's unsorted Markowitz columns
already match upstream entry-for-entry and bit-for-bit, so the provisional `U`
sort is removed. The resulting earlier `israel` pivot-143 drift was then
localized to rank-one update construction. Upstream preserves physical `U`
order for triangular solves, but `GetColumnOfU()` copies through a cleaned—and
therefore row-sorted—temporary `SparseColumn`; Rust had returned the physical
column directly. Sorting only this accessor copy makes the affected rank-one
vectors and middle-product multipliers bit-identical and restores native's
column-101 choice at pivot 143. A later end-to-end trace showed that the
remaining pivot-148 explanation was stale after those ordering repairs: the
fresh exact reduced-cost solve is bit-identical. The first subsequent drift
was instead the Phase-II iteration-97 column-wise update-row dot product,
where Rust used a linear sum rather than GLOP's four-accumulator compact-column
kernel. Matching that kernel, and preserving incrementally updated reduced
costs across routine update-count refactorizations as `DualMinimize()` does,
makes pivot 148 exact. The next difference is at pivot 149: native chooses
column 230 while Rust chooses column 236. A complete elementary-update trace
corrects the earlier rank-one attribution: all seven stored update matrices,
their scalar products, multipliers, intermediate vectors, and final output are
bit-identical. The subsequent `L^{-T}` result, relevant update-row
coefficients, and post-pivot-148 reduced-cost vector are bit-identical too.
An explicit cache trace shows that pivot 149 computes common leaving row 157
as a cache miss in both implementations and produces the same `U^{-T}` vector;
the complete BTRAN and ratio-test inputs agree. Columns 230 and 236 form an
exact entering tie. Rust selects 236 only because its shared RNG has advanced
two extra times in `DynamicMaximum::UpdateTopK()`: earlier updates for rows
128, 105, and 90 meet Rust's top-31 threshold, whereas native has only its
row-162 threshold tie in the corresponding interval. The remaining cause is
therefore the earlier dual-price/heap-state divergence, not rank-one
arithmetic. The differential rank-one suite still confirms that optimized GLOP contracts dense
updates while sparse scattered updates materialize `multiplier * coefficient`
before `Add()`; Rust preserves that distinction.
The pricing operation streams agree through the start of Phase II. Their first
value mismatch is a one-ulp basic-value difference in the initial dense
dual-price rebuild: bounds and dual norm agree, and the entire right-hand side
of the preceding refactorized-basis solve agrees bit-for-bit. The lower and
middle-product stages agree too; Rust's dense upper path was traversing `U`
directly instead of using the stored transpose and upstream's transpose-lower
reduction order. Aligning that path makes the complete solve exact, restores
the native pivot-149 choice, and reconciles all 287 `israel` pivots and final
optimal status. The exact-limit status additionally required moving the
iteration-limit check to GLOP's post-pricing, post-pivot-validation location. The
boxed-variable update now faithfully switches from scattered to dense column
accumulation when the scratch vector crosses GLOP's 80% threshold; this closes
a separate semantic and performance gap but is not exercised by that Israel
solve. The smallest-25 audit consequently has 24 exact iteration-count,
status, and basis matches; at that point `bore3d` was the sole remaining
smallest-25 trajectory mismatch.
`brandy` has the native 167-iteration count and an identical final basis. Its
preceding LU discrepancy was an exact numerical cancellation retained as a
structural zero in a Rust L column; upstream
`AddAndNormalizeTriangularColumn()` drops zero coefficients. Removing them
restores the sparse transpose-solve reachability and accumulation order.
`bore3d`'s former pivot-39 divergence was a missing control-flow branch. GLOP
checks the chosen update-row coefficient against
`dual_small_pivot_threshold` before FTRAN and requests a refactorized precise
retry; Rust had only the later direction-relative small-pivot check. Porting
the first check makes the nonidentity basis permutation after pivot 38 and the
first 65 pivots agree. Subsequent localization found that Rust did not mirror
GLOP's explicit
`non_zeros_are_sorted` cache transitions around hypersparse LU solves. This
changed the reduction order in an exact dual-edge norm recomputation. The port
now records all seven upstream sorted/unsorted transitions explicitly; the
formerly divergent row-24 and row-27 norm states and updates are bit-identical.
The formerly divergent pivot-66 choice is resolved: retaining the preceding
direction's price rows across retry iterations restores GLOP's duplicate heap
entries and shared-RNG draws. Both implementations now terminate in 128 pivots
with the same ordered basis, reduced costs, and dual norms.
The solver-level cleanup/reoptimization loop is now also ported: after shifts
are removed, it refactorizes, recomputes the primal and dual state, and switches
simplex algorithms when the precise solution violates an internal tolerance.
This resolves `scsd6`, whose common 234-pivot dual trajectory left a
`1.581e-8` dual infeasibility; both implementations now perform the same final
primal pivot and finish after 235 iterations with the same ordered basis. The
smallest-50 audit consequently has 50 exact complete trajectories/final bases.
The former `boeing1` split was at pivot 145: native entered column 134 and Rust
column 130 from the same eight-way ratio-test tie.
The ordered basis, reduced costs, and dual norms agree through pivot 144; the
only reported basic-value difference is a signed zero. The audit exposed and
fixed a real Phase-II orchestration error: after a routine basis
refactorization Rust still applied stale bound flips and pending price-row
updates, whereas GLOP takes the recomputation branch and skips both. That fix
removes the earlier numerical drift after pivot 114. The remaining missing
`DynamicMaximum` Bernoulli draw exposed a second omission in that branch:
GLOP moves every dual-infeasible nonbasic boxed variable to its opposite bound
before recomputing the basic values, whereas Rust had retained the old bound
statuses. Porting that full boxed-variable pass restores the row-42/row-234
price tie and shared-RNG draw. Every `boeing1` ordered-basis prefix now agrees;
both implementations terminate optimally after 507 iterations with identical
ordered bases, reduced costs, and dual norms. The only value-bit differences
are two immaterial negative-zero spellings in Rust.
A literal `RightSolveLForColumnView()` translation is now active. Its initial
`scagr25` failure exposed a Rust-only stale membership bit: a reached entry
that solved to zero could later become nonzero without reentering the position
list. Direct scattering into values and positions while leaving the temporary
mask cleared matches GLOP's protocol; the 482-pivot `scagr25` trajectory and
the smallest-25 status/objective/feasibility gate pass again.

Build metadata (`BUILD.bazel`, `CMakeLists.txt`) is deliberately replaced by
Cargo. The upstream README remains a behavioral/source guide rather than a file
to translate.

## `ortools/lp_data`

| Upstream source | Rust destination | Status | Notes |
|---|---|---|---|
| `lp_types.{h,cc}` | `lp_data/src/lp_types.rs` | validated | Pinned scalar constants, 32-bit row/column and 64-bit entry indices, conversions, invalid sentinels, `i8`-represented statuses/types and names, status conversion, all dense/mapping aliases, typed owned vectors, borrowed immutable/mutable spans, capacity/reserve/assign/zero/resize-down operations, and deterministic-time conversion are ported. `Bitset64` behavior includes borrowed views, resize/clear, buckets, paired bits, cross-size content copy, intersection/union, and bucket/trailing-zero set-bit iteration. A 1,000-case native trace agrees on status/type strings and conversions, scalar bit patterns, paired bits, cross-size copies and Boolean operations, shrink/grow behavior, logical contents, and set-bit iteration—including GLOP's non-obvious exposure of populated padding bits after cross-size operations |
| `sparse_vector.h` | `lp_data/src/sparse_vector.rs` | validated | The complete computational surface is ported, including exact-copy/value semantics, total-capacity `Reserve()`, logical resize-down, indexed and mutable entry access, dense population/copy/update, stable cleanup with last-duplicate precedence, weighted filtering, ordered sparse merge kernels, signed-sentinel `IndexPermutation` semantics for partial permutations and allocation-free index-preserving `MoveTaggedEntriesTo`, and reusable all-false duplicate-check scratch storage with O(1) successful-result caching. Storage follows upstream's structure-of-arrays layout with contiguous index and coefficient regions and value-returning entry iterators; safe Rust uses two typed allocations rather than manually splitting one untyped allocation, preserving traversal and asymptotic behavior. A 1,000-case native transformation trace agrees entry-for-entry, and the same representation passes matrix, triangular, Markowitz, and LU differential suites. C++-specific debug-string formatting is deliberately left to Rust's ordinary debugging facilities |
| `sparse_column.{h,cc}` | `lp_data/src/sparse_vector.rs` | validated | Row-named column entries and accessors, full and partial row permutations, borrowed parallel-slice `ColumnView` with last-duplicate lookup semantics, and touched-row `RandomAccessSparseColumn` are ported. Safe slices replace the raw-pointer view constructor without changing borrowing, traversal, or complexity. The 1,000-case sparse-vector native trace now covers view traversal plus random-access population, mutation, explicit-zero preservation, and sparse extraction alongside the inherited vector transformations |
| `sparse_row.h` | `lp_data/src/sparse_row.rs` | validated | Row-specialized entry accessors, first/last-column names, complete and partial column-permutation methods, inherited sparse-vector behavior, iteration, and typed row-major matrix storage are present. In addition to the underlying 1,000-case sparse-vector and permutation traces, a dedicated 1,000-case native trace agrees exactly on accessor and iterator column indices, coefficient bit patterns, storage order, both permutation paths, endpoint accessors, and `RowMajorSparseMatrix` traversal |
| `sparse.{h,cc}` | `lp_data/src/sparse.rs`, `lp_data/src/triangular_matrix.rs` | validated | General sparse matrices now include identity/unit construction, reserved stable-order transpose, copies/permuted copies, GLOP-ordered linear combinations and products, column/partial-row deletion, row append, in-place unsorted row permutation, duplicate-aware equality, and diagnostics; 1,000 native traces agree on transpose, products, combinations, storage order, and magnitudes. Borrowed matrix, matrix-pair, basis-subset, and compact basis views preserve GLOP's column order, dimensions, entry counts, and norms. Compact storage has explicit dimensions, and its four-accumulator column scalar product accepts a typed row or a borrowed slice without copying (bitwise regression tested), incremental and dense/nonzero builders, slack augmentation, stable-order transpose, GLOP's four-accumulator column scalar product with explicit fused accumulation matching the optimized native build, dense/scattered multiply-add, and copy operations; generated native traces agree on storage and view behavior. Triangular storage has GLOP's separate diagonal, incremental column builders (including normalized columns), identity-prefix tracking, row-permutation hook, copy paths, transpose, dense solves (including `LowerSolveStartingAt`), both sorted and DFS symbolic closures, all four ordered hypersparse solve variants, the factorization-time partially permuted sparse lower solve with persistent in-place dependency-graph pruning and operation counts, triangularity checks, and exact and estimated inverse infinity norms. Markowitz now uses this method on its actual in-progress L storage, matching the upstream ownership and pruning boundary. Dense transpose kernels preserve GLOP's optimized forward and reverse four-term contraction sequences, lower-triangular reverse storage traversal, and the trailing-zero skip (including its observable signed-zero behavior). A 1,000-case triangular native trace agrees on exact storage, builder and symbolic output, sparse position filtering, norms, and numerical solve results within cross-compiler ulps; the 1,000-case LU trace directly exercises and agrees on the specialized permuted solve and its pruning order |
| `scattered_vector.h` | `lp_data/src/scattered_vector.rs` | validated | Dense values, sparse-superset positions, typed packed-bit mask, empty-pattern-as-dense convention, sparse/dense switching, whole-bucket mask clear/repopulation (including stale-mask semantics), sorted-position tracking, and nonzero estimates are ported. Allocation-free entry iterators provide the specialized row/column names. A safe O(1) borrowed transpose view replaces upstream's layout-dependent `reinterpret_cast`, converting only the strong index wrapper during iteration. A 1,000-case native trace agrees on add order, values, sorting, density decisions, estimates, mask clearing/repopulation (including duplicate sparse positions), transpose traversal, and dense-representation switching |
| `permutation.h` | `lp_data/src/permutation.rs` | validated | Typed inverse, identity and random population, validity, signature, forward/inverse application across distinct strong index types, empty-as-identity, and the cross-index column-permutation specialization are ported. Rust's `rand` thread-local generator replaces `absl::BitGen`; both shuffle uniformly, but reproducible cross-language random sequences are not an upstream contract. A 1,000-case native trace covers valid and invalid permutations, inverse construction, signatures, cross-index application, inverse application, and identity construction |
| `lp_data.{h,cc}` | `lp_data/src/lp_data.rs` | validated | Netlib parse fingerprints agree. Core model behavior includes integer/binary classification, minimization costs, primal bound/row/integrality feasibility, objective scaling translation, atomic bound intersection, near-zero removal, integer-bound predicates, GLOP's structural-plus-rightmost-identity slack injection, slack value computation/removal, equation-form recognition, and row/column deletion with metadata/id remapping. The model follows GLOP's lazy transpose and three-list integer-variable caches, including the same mutation invalidation, mutable-transpose adoption, memory-release, and incremental row/column-deletion paths. Constraint-block append, full/variables-only copy, source-to-destination permuted population, and dual construction preserve upstream column grouping, permutation direction, metadata, cache state, and sparse entry order. Objective scaling implements all four GLOP algorithms with the same reduction/sort order; bound scaling and magnitude-limited validity checks also match. Generated lowercase fallback names, out-of-range name queries, dedicated slack construction, and whole-model swap semantics match upstream as well. A 500-case native trace agrees on classifications, names, feasibility decisions, objective transforms and scaling, slack structure/bounds/values, restored constraints, appended constraints, primal-to-dual transformations, permuted models, validity decisions, deletion mappings, and entry order. Dimension, objective, bound, structural, and nonzero-statistics strings, LP-format emission, and solution emission are also ported and compared as part of the generated MPS trace. C++ debug-only bound toggles and logging-oriented debug strings are infrastructure differences rather than model behavior |
| `lp_data_utils.{h,cc}` | `lp_data/src/lp_data_utils.rs`, `lp_data/src/lp_data.rs` | in progress | Every `LpScalingHelper` conversion is ported: configured row/column factors, all scalar scale/unscale directions, ordinary/slack variable factors, dense and sparse unit-row/column solve unscaling, average-cost and contain-one-bound scaling (including zero, subunit, superunit, and infinite cases), clearing/state retention, and factor access. A strengthened 1,000-case native trace agrees on all operations and both scattered-vector representations. Whole-model equilibration scaling now accepts all four cost-scaling algorithms; another 1,000 native traces agree on the objective, every bound, helper factors, and sparse matrix structure and values. The row remains in progress solely because the public scaling-method overload can select `LINEAR_PROGRAM`, whose matrix-scaling branch intrinsically invokes the not-yet-ported revised simplex; no substitute implementation is presented as faithful |
| `lp_utils.{h,cc}` | `lp_data/src/lp_utils.rs` | validated | Dense, sparse, and scattered scalar products and squared norms preserve GLOP's block/reduction order and dense-versus-sparse dispatch; precise variants use the upstream `AccurateSum` recurrence. Resetting norms, infinity and restricted norms, density, threshold removal, support clearing, domination, square, fractionality, dense support extraction, zero/false predicates, scratchpad and known-support permutations, sparse-aware clear/resize, sign change, and both one-sided-infinity `SumWithOneMissing` variants are ported. A 1,000-case native trace agrees on values and side effects, including support permutation and compensated sums. LU and primal/dual edge norms share these routines. Typed transpose views and generic iterator overloads are represented by safe Rust borrowing rather than C++ casts/templates |
| `matrix_utils.{h,cc}` | `lp_data/src/matrix_utils.rs` | validated | Both proportional-column algorithms (including GLOP's exact 64-bit pattern hash, fingerprint ordering, and representative normalization), leading-rectangle exact equality, and rightmost-identity recognition are ported; 1,000 generated native traces agree exactly |
| `matrix_scaler.{h,cc}` | `lp_data/src/matrix_scaler.rs` | in progress | The `EQUILIBRATION` path directly ports the four geometric row/column passes, variance stop, dynamic-range cutoff, final row/column equilibration, accumulated scaling factors, out-of-range factor semantics, partial-length row/column vector transforms, clear behavior, and the pinned `Init()` resize semantics (including its comment-defying retention of factors at unchanged dimensions). A strengthened 1,000-case native trace deliberately covers null/explicit-zero matrices, identity magnitudes, geometric and dynamic-range-cutoff branches, empty rows/columns, extreme ranges, every public factor getter, vector round trips, reinitialization, clearing, sparse entry order, and coefficients. The row remains in progress solely because `LINEAR_PROGRAM` calls the private `LPScale()`, which constructs and solves an auxiliary LP through the not-yet-ported `RevisedSimplex`; no different algorithm is labeled faithful |
| `mps_reader_template.{h,cc}` | `lp_data/src/mps_reader.rs` | validated | The LinearProgram reader preserves fixed/free field boundaries, case-sensitive section cards, whole-input fixed-then-free auto-detection, forced formats, section semantics, integer-marker defaults, ignored RHS/RANGES/BOUNDS vector names, range and bound corner cases, objective offsets, zero suppression, sparse cleanup, and accept/reject behavior. A direct native harness compares the complete parsed model—including names, integrality, sparse entry order, and every floating-point bit—and 1,000 generated free plus 1,000 generated fixed models agree in auto, forced-free, and forced-fixed modes. Focused malformed and numerical cases agree, and all 98 Netlib fingerprints agree. Indicator and semi-continuous models remain unsupported because upstream's LinearProgram wrapper itself rejects those protobuf-only features |
| `mps_reader.{h,cc}` | `lp_data/src/mps_reader.rs` | validated | File and string entry points plus explicit `AutoDetect`, `Free`, and `Fixed` selection mirror the LinearProgram-facing deprecated adapter. Protobuf-returning entry points are deliberately omitted with the rest of protobuf infrastructure. Validation is shared with the template row above |
| `lp_parser.{h,cc}` | `lp_data/src/lp_parser.rs` | validated | The semicolon-delimited LP parser directly preserves GLOP's ordered token recognizers, case-insensitive prefix-matched `int`/`bin`, legal-name patterns, sign compaction, optional multiplication/commas, scientific values and infinity, objective-offset placement, duplicate-term rules, named-constraint uniqueness, unnamed unit-coefficient bound intersection, binary bounds, unbounded defaults, partial-model-on-error behavior, and exact `ParseConstraint()` diagnostics. A 2,024-case native trace agrees byte-for-byte on parsed canonical dumps, partial failure state, constraint fields, coefficient bits, and diagnostics; additional focused overflow, underflow, malformed-token, and keyword-prefix cases agree. The protobuf conversion wrapper is omitted with protobuf infrastructure |
| `sol_reader.{h,cc}` | `lp_data/src/sol_reader.rs` | validated | LinearProgram SOL file/string entry points preserve generated fallback names, last-duplicate model and solution assignment, objective-line ignoring, token/comment rules, exact diagnostics, and zero defaults. The `strtod`-compatible leading-number parser covers decimal, infinity/NaN, trailing text, hexadecimal round-to-nearest-even, and the pinned platform's overflow/underflow behavior. A 1,265-case native trace includes malformed lines, unknown names, comments, duplicate assignments, randomized ordinary input, and randomized hexadecimal bit-pattern comparisons. The protobuf `MPModelProto`/`MPSolutionResponse` overload is omitted with protobuf infrastructure |
| `lp_print_utils.{h,cc}` | `lp_data/src/lp_print_utils.rs`, `lp_data/src/lp_data.rs` | validated | GLOP's `%.16g` decimal formatting, continued-fraction rational formatting, monomial sign/unit-coefficient rules, complete `LinearProgram::Dump()` LP emission, and default-formatted solution emission are ported. On the pinned Apple arm64 reference target, `long double` and `double` share the same 53-bit precision, allowing the continued-fraction operation order to be reproduced with `f64`. A 10,000-case native bit-pattern trace agrees exactly on decimal, rational, and monomial output, and generated MPS traces compare complete LP and solution emission byte-for-byte |
| `lp_decomposer.{h,cc}` | `lp_data/src/lp_decomposer.rs` | validated | Union-find decomposition preserves union-by-size/root tie-breaking, first-node component numbering, sorted global columns, and `SparseBitset` first-touch constraint order. Local extraction copies exactly the upstream names, types, bounds, costs, direction, coefficients, and sparse traversal order; global/local assignment projection and aggregation are included. A 1,000-case native trace agrees byte-for-byte on component order, complete extracted models, coefficient bits, and assignment reconstruction. A Rust shared borrow replaces the C++ raw pointer plus mutex and statically prevents source mutation |
| `proto_utils.{h,cc}` | focused model conversion, if needed | not started | Do not reproduce protobuf infrastructure |

## Direct supporting dependencies to assess

The initial include inventory found these non-GLOP/non-`lp_data` dependencies:

- `ortools/algorithms/dynamic_partition.h`
- `ortools/base/accurate_sum.h`
- `ortools/base/hash.h`
- `ortools/base/iterator_adaptors.h`
- `ortools/base/numbers.h`
- `ortools/base/strong_vector.h`
- `ortools/base/timer.h`
- `ortools/graph/iterators.h`
- `ortools/util/bitset.h`
- `ortools/util/fp_utils.h`
- `ortools/util/random_engine.h` — `glop/src/random.rs` ports the default
  shared `std::mt19937_64` stream, libc++ integer and real distribution
  semantics used by GLOP, and Abseil's 0.5 Bernoulli draw. Raw outputs, mixed
  distribution draws, and a tied `DynamicMaximum` sequence agree exactly with
  the pinned Apple/libc++ reference. The nondefault `use_absl_random` engine is
  not yet ported.
- `ortools/util/rational_approximation.h`
- `ortools/util/stats.h` — non-timing distributions, Welford accumulation,
  merging, reset, ordering, and byte-compatible formatting are ported in
  `glop/src/stats.rs`; scoped timing remains compiled out exactly as in the
  pinned non-`OR_STATS` reference build. A focused native trace validates the
  common machinery, and LU/basis/edge/update traces validate component wiring.
- `ortools/util/strong_integers.h`
- `ortools/util/time_limit.h` — the base `TimeLimit` behavior used by GLOP is
  ported in `glop/src/time_limit.rs`, including conservative wall checks,
  deterministic and two external limits, reset/history semantics, merging,
  and sticky wall expiration. A native trace covers those behaviors; large
  primal and dual norm traces cover its numerical-kernel integration. The
  thread-safe, nested, and periodic-check wrappers are outside current GLOP
  kernel usage and remain unported.

Logging, status macros, file helpers, protobuf helpers, and generated protobuf
headers are infrastructure replacements, not translation targets. Each source
module must still be checked for subtle semantics supplied by them.

## Performance-fidelity audit

The first 96-model serial Netlib timing pass found several implementation
differences that preserved pivot trajectories but cost time. `lu_factorization`
now reuses the dense zero scratchpad and nonzero-row list in dual-edge norm
solves; `update_row` uses a bitset for hypersparse column intersections and
the cached number of entries in relevant columns; `revised_simplex` reuses its
direction workspace; and `triangular_matrix` uses a persistent entry cursor in
the dense transpose solve and a counting-pass direct transpose. The strict
native trajectory fixture still passes. Aggregate Rust/native in-solver time
fell from 1.358 to 1.226; see `PLAN.md` for the benchmark conditions and
remaining outliers.

Profiles of `dfl001` and `stocfor3` identified a lifecycle discrepancy:
`basis_representation::force_refactorization()` constructed a fresh
`LuFactorization` and `markowitz::compute()` constructed fresh workspaces on
every rebuild, whereas upstream retains both objects. `ForceRefactorization`
retains its LU object; Markowitz resets residual-pattern, queue, candidate,
physical-column, and in-progress lower-factor storage in place; and the final
triangular factors and their transposes reuse their contiguous buffers.
Repeated A/B/A factorization and the strict 96-model trajectory fixture pass.
Those lifecycle changes alone had small timing gains. The separate
factor-representation divergence is now eliminated: Markowitz fills the
retained `L` and `U` triangular matrices directly, normalizes `L` in place,
and applies the final row permutation to their off-diagonal indices in place,
exactly as upstream `Markowitz::ComputeLU()` does. It no longer copies `L`
through sparse-column and tuple-column intermediates or reconstructs either
factor from those tuples. First-time candidate columns fill their recycled
physical-column storage directly. One thousand native LU differential traces
and all 96 strict native Netlib trajectories pass; full serial aggregate
Rust/native in-solver time fell from 1.226 to 1.103. Remaining performance
gaps require separate profiling and are not attributed to factor conversion.

The next profiling pass aligned the Markowitz DFS workspace with GLOP's
compact row bitset, including touched-bucket clearing, and specialized the
dense lower-transpose solve by unit-diagonal state with GLOP's fixed short
tail. Both preserve the native 96-model trajectories and 1,000 LU traces.
The serial aggregate ratio is now 1.097; `dfl001`, `stocfor3`, and `truss`
remain the largest absolute Rust excesses. Dense transpose substitution is
still the main sampled Rust self-time on `dfl001` and `fit2p`, but the precise
remaining machine-code cause has not yet been established. No unsafe code was
introduced; `lp_data` continues to forbid it.

The dense upper-transpose solve now follows upstream's one-time diagonal
dispatch and continuous forward entry cursor, preserving its four-term
contraction and fixed-tail operation order. A targeted unit test, 1,000 native
triangular traces, and the strict 96-model release trajectory fixture pass.
A three-model serial timing sample is essentially unchanged (aggregate ratio
1.279, versus 1.275 on the same models previously), so this is a structural
fidelity correction, not a measured performance improvement. The dense
lower-transpose kernel has since been isolated in a dedicated microbenchmark.

ARM64 disassembly of that benchmark identified repeated factor-storage bounds
checks in the original Rust lower-transpose loop. Per-column checked slices
with four-entry reverse chunks preserve GLOP's floating-point grouping and
traversal direction but avoid much of that cost without `unsafe`. At dimension
1024 and 3% density, its median microbenchmark ratio improved from 1.73 to
1.49; the full 96-model three-trial aggregate solve-time ratio moved from
1.097 to 1.093. The strict trajectory fixture passes. Rust's per-entry
right-hand-side checks and 64-bit row indices remain representation differences
from native GLOP; the current evidence does not apportion their separate costs.

## Test inventory caveat

The pinned public tree does not contain the historical file-local unit tests for
most `ortools/glop` and `ortools/lp_data` components. Available integration tests
and samples must be supplemented by differential tests against the native GLOP
binary and by Rust invariant/randomized tests. If corresponding upstream tests
are found in another public revision, record their exact provenance before
porting them.
