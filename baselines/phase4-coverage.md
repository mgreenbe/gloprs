# Phase-4 branch fixture coverage

The small-LP fixture currently has 159 pinned-native cases: 157 terminal
outcomes with exact clocks and two pinned LU errors. Each listed branch
tag is emitted by the Rust driver only when that path is taken; the test also
compares the case's final state to GLOP. This is **not** a claim that every
branch of `RevisedSimplex` is covered. The inventory below is the outstanding
work needed to make that claim.

| Path | Case(s) | State |
|---|---|---|
| Nondefault triangular and Maros initial-basis crashes, primal and dual | `initial_triangular_primal`, `initial_maros_primal`, `initial_triangular_dual`, `initial_maros_dual` | Exercised on equality rows; each selects the structural basis without pivots and matches native state and exact clock |
| Bixby initial-basis request while scaling is disabled | `initial_bixby_without_scaling` | Skips Bixby as upstream does; exact native initial state and clock |
| Triangular crash rejected by initial condition-number threshold | `initial_triangular_condition_fallback` | Reverts to all-slack basis, with native state and exact clock |
| All-slack basis rejected by initial condition-number threshold | `initial_all_slack_rejected_by_condition_threshold` | Native `ERROR_LU` message and Rust error agree; decision event required |
| Primal Phase I and infeasible exit | `primal_phase_one_infeasible` | Exercised |
| Primal Phase II, bounded pivot and optimal cleanup | `primal_optimization` | Exercised |
| Primal bound flip | `primal_boxed_bound_flip` | Exercised |
| Degenerate primal pivot retaining an off-bound leaving value | `primal_degenerate_pivot` | Exercised |
| Reduced-cost precision retry and forced refactorization | `primal_precision_refactorization` | Exercised |
| Primal steepest-edge and Devex pricing through two pivots | `primal_steepest_edge_two_pivots`, `primal_devex_two_pivots` | Exercised; exact native clocks |
| ETA-form basis updates in primal Phase II | `primal_eta_basis_updates` | Exercised through two pivots with middle-product updates disabled; exact native clock |
| ETA-form basis updates in primal Phase I | `primal_phase_one_eta_basis_updates` | Exercised through two pivots with middle-product updates disabled; exact native clock |
| ETA-form basis updates in dual Phase II | `dual_eta_basis_updates` | Exercised through two pivots with middle-product updates disabled; exact native clock |
| ETA-form basis updates in dedicated dual Phase I | `dual_phase_one_eta_basis_updates` | Exercised through two pivots with middle-product updates disabled; exact native clock |
| Steepest-edge precision request, exact norm recomputation, and next-iteration refactorization | `primal_steepest_zero_norm_threshold` | Exercised; native status, basis, values, and clock agree |
| Dual edge-norm precision requests recomputation | `dual_zero_norm_threshold` | Zero-threshold request is required; the same LP at the default threshold forbids it, and both match native state and exact clock. This case terminates without a subsequent norm-forced refactorization |
| Dual edge-norm precision request followed by forced LU refactorization | `dual_norm_precision_forces_refactorization`, `dual_norm_precision_default_control` | Feasible 4×4 LP requires both decision events at zero threshold; same LP at default threshold forbids both. Native terminal state and exact clock agree in both cases |
| Early imprecise pivot and adaptive LU threshold escalation | `adaptive_lu_pivot_threshold` | Exercised; native/Rust final threshold bits agree |
| Early imprecise dual pivot and adaptive LU threshold escalation | `dual_adaptive_lu_pivot_threshold`, `dual_adaptive_lu_default_control` | Four-row dual case requires escalation at zero refactorization threshold; same LP at the default threshold forbids it. Both match native status, state, clock, and final LU threshold bits |
| Exact primal Harris tie and shared-RNG choice | `primal_harris_exact_tie` | Exercised |
| Primal unbounded ray | `primal_unbounded`, `primal_unbounded_coupled_ray`, `primal_unbounded_negative_ray` | Exercised; coupled ray has two nonzero structural entries, and negative ray exercises the opposite movement direction |
| Primal ray postsolve validation (accepted) | `primal_unbounded` | Exercised |
| Primal weak-ray rejection and stronger-ray acceptance | `primal_weak_ray_rejected`, `primal_strong_ray_accepted` | Exercised |
| Primal weak-ray rejection with imprecise conversion disabled | `primal_weak_ray_rejected_without_imprecise_conversion` | Exercised; lazy reduced-cost refresh and native operation clock agree |
| Strong primal ray with imprecise conversion disabled | `primal_strong_ray_without_imprecise_conversion` | Exercised; unconditional dual-residual solve and native operation clock agree |
| Primal objective limit and post-call cleanup | `primal_objective_limit` | Exercised; `primal_objective_limit_cleanup` required, exact native clock |
| Same primal LP without an objective limit | `primal_objective_no_limit_control` | Exercised; exact native clock, objective-limit branch forbidden |
| Dedicated dual Phase I | `dual_dedicated_phase_one` | Exercised |
| Transformed dual Phase I | `dual_transformed_phase_one` | Exercised |
| Initial dual cost perturbation | `dual_perturbation` | Exercised |
| Perturbed-cost removal | `dual_perturbation` | Exercised |
| Dual Phase II, bounded pivot and optimal cleanup | `dual_phase_two` | Exercised |
| Nondefault dual norm-prioritized price (two pivots) | `dual_norm_prioritized_pricing` | Exercised; native status, ordered basis, reduced costs, and clock agree |
| Dual iteration limit | `dual_iteration_limit_zero` | Exercised |
| Dual Phase-I iteration limit | `dual_phase_one_iteration_limit_zero`, `dual_phase_one_positive_iteration_limit` | Exercised at zero and after one pivot; post-Phase-I cleanup is required, with exact native clock |
| Already optimal at zero dual iteration limit | `dual_already_optimal_at_zero_limit` | Exercised |
| Dual objective limit and cleanup before stopping | `dual_objective_limit`, `warm_dual_changed_objective_limit` | Exercised; `dual_objective_limit_cleanup` required and native operation clocks agree |
| Same dual LP without an objective limit | `dual_objective_no_limit_control` | Exercised; native trajectory and operation clock agree, limit branch forbidden |
| Dual-unbounded ray | `dual_unbounded_ray`, `dual_unbounded_coupled_ray`, `dual_unbounded_positive_ray` | Exercised; coupled certificate has two nonzero row entries, and the new upper-bound contradiction reverses the certificate sign |
| Dual ray postsolve validation (accepted) | `dual_unbounded_ray` | Exercised |
| Dual weak-ray rejection and tolerance-boundary acceptance | `dual_weak_ray_rejected`, `dual_ray_at_solution_tolerance` | Exercised |
| Dual weak-ray rejection with imprecise conversion disabled | `dual_weak_ray_without_imprecise_conversion` | Exercised |
| Dual ray at solution tolerance with imprecise conversion disabled | `dual_ray_at_tolerance_without_imprecise_conversion` | Exercised; native status, ray, and operation clock agree |
| Dual-infeasible exit | `dual_infeasible_no_rows` | Exercised |
| Loaded basis on repeated solve | `warm_primal_repeat` | Exercised |
| Loaded basis after a changed row bound | `warm_bound_change` | Exercised |
| Loaded basis after an objective change | `warm_objective_change` | Exercised |
| Loaded basis after two simultaneous row-bound changes | `warm_multiple_bound_changes` | Exercised |
| Supplied nondefault basis | `supplied_nondefault_basis` | Exercised |
| Incremental added-column warm start retaining LU, with structural or slack basis | `warm_added_column`, `warm_added_column_from_slack_basis` | Exercised; native operation clock agrees |
| Supplied starting value for a free variable, with push disabled | `starting_free_variable_value` | Exercised |
| Residual-driven `IMPRECISE` exit | `zero_tolerance_residual` | Exercised |
| Nonzero cleanup-residual tolerance, rejected and accepted | `nonzero_residual_tolerance_rejected`, `nonzero_residual_tolerance_accepted` | Exercised |
| Disabled `IMPRECISE` conversion | `zero_tolerance_no_imprecise` | Exercised |
| Feasible primal basis at zero iteration limit | `primal_feasible_at_zero_limit` | Exercised |
| Infeasible primal basis at zero iteration limit | `primal_infeasible_iteration_limit_zero` | Exercised |
| Phase I consumes the positive primal limit | `primal_phase_one_consumes_iteration_limit` | Exercised |
| Sparse Phase-I cost refresh, incremental reduced-cost reuse, and leaving-cost removal | `primal_phase_one_two_pivots_incremental`, `primal_phase_one_coupled_rows` | Exercised |
| Positive Phase-I limit with an infeasible row remaining | `primal_phase_one_positive_limit_with_remaining_infeasibility`, `primal_phase_one_coupled_rows_limit_one` | Exercised |
| Positive iteration limit after a primal Phase-II pivot | `primal_phase_two_positive_iteration_limit` | Exercised |
| Positive iteration limit after a dual Phase-II pivot | `dual_phase_two_positive_iteration_limit` | Exercised |
| Incremental added-row warm start refactorizing the extended basis, with structural or slack basis | `warm_added_row`, `warm_added_row_from_slack_basis` | Exercised; native operation clock agrees |
| Added row and structural column in the same warm start | `warm_added_row_and_column` | Full warm initialization and Markowitz candidates agree with native terminal state and clock |
| Removed row leaving too many saved BASIC candidates | `warm_removed_row` | Markowitz candidate pass agrees with native state and clock; candidate-count test no longer truncates before deciding |
| Removed first row with surviving second-row structural pivot | `warm_removed_first_row` | Native and Rust agree on reordered row/basis coordinates and exact clock |
| Removal of every row after a solved two-row basis | `warm_removed_all_rows` | Native and Rust agree on zero-dimensional reinitialization, terminal state, and exact clock |
| Removed BASIC structural column | `warm_removed_basic_column` | Native and Rust agree on `DUAL_UNBOUNDED`, basis, certificate, and exact clock |
| Removal of every structural column after a solved two-row basis | `warm_removed_all_columns` | Native and Rust agree on the slack-only infeasibility certificate, terminal state, and exact clock |
| Added-row warm basis rejected by condition threshold, then recovered with a fresh basis | `warm_added_row_condition_fallback` | Exercised; computed Markowitz candidate basis is used and native operation clock agrees |
| Added-row warm start with a changed existing coefficient | `warm_added_row_with_changed_old_coefficient` | Exercised; full warm-basis rebuild selects initial-basis candidates before factorization, with exact native clock |
| Saved basis made singular by an existing-coefficient edit | `warm_singular_saved_basis_after_matrix_change` | Recovery via Markowitz candidate basis matches native terminal state and exact clock |
| Singular saved basis with a lowered condition threshold | `warm_singular_saved_basis_with_lowered_condition_threshold` | Candidate recovery passes the condition check and matches native terminal state and exact clock |
| Singular saved basis whose fallback also fails the condition threshold | `warm_singular_saved_basis_fallback_condition_error` | Native `ERROR_LU` message agrees with Rust after candidate recovery and all-slack fallback |
| Full-rank saved basis after an existing-coefficient edit, without a dimension change | `warm_full_rank_saved_basis_after_matrix_change` | Direct saved-basis factorization matches native state and exact clock; candidate fallback is forbidden |
| Two-row full-rank saved basis after an existing-coefficient edit | `warm_full_rank_two_row_saved_basis_after_matrix_change` | Direct saved-basis factorization matches native state and exact clock; controls the singular-candidate recovery case |
| Complete saved basis rejected by a lowered condition threshold | `warm_saved_basis_rejected_by_condition_threshold` | Markowitz candidate-basis retry then all-slack fallback matches native state and exact clock |
| Dual quick warm start reusing the current factorization after changed bounds | `warm_dual_bound_change_reuses_factorization`, `warm_dual_multiple_bound_changes_reuse` | Exercised |
| Primal quick warm start with unchanged matrix and bounds, including changed objective | `warm_primal_repeat`, `warm_objective_change` | Exercised |
| Primal quick warm start with a changed objective limit | `warm_primal_changed_objective_limit` | Exercised |
| Dual quick warm start with a changed objective limit | `warm_dual_changed_objective_limit` | Exercised |
| Dual quick warm start with unchanged matrix, objective, and bounds | `warm_dual_repeat_reuses_factorization` | Exercised |
| Automatic reuse of saved state without `LoadStateForNextSolve` in primal and dual solves | `warm_auto_primal_repeat`, `warm_auto_primal_objective_change`, `warm_auto_dual_repeat`, `warm_auto_dual_bound_change` | Exercised |
| Explicit clear of saved state forces a fresh start | `warm_clear_state_restarts_from_scratch` | Exercised; reuse branches explicitly forbidden |
| External basis loaded and then replaced by the prior saved basis | `warm_external_then_restore_saved_state` | Exercised; external validation path required, quick reuse forbidden |
| Starting-value push of a free variable to zero, including a basis pivot | `starting_free_variable_pushed_to_zero`, `starting_free_variable_push_pivot` | Exercised |
| Starting-value push interrupted by a deterministic-time limit | `starting_free_variable_push_interrupted` | Exercised; preserves `OPTIMAL` with the super-basic value and matches native clock |
| Two-variable push with small-pivot refactorization request, no-refactor control, and matched no-push control | `starting_free_variable_push_refactorization`, `starting_free_variable_push_control`, `starting_free_variable_no_push_control` | Exercised; native trajectories and clocks agree |
| Same LP without supplied starting values | `two_free_variables_no_start_control` | Exercised; native trajectory and clock agree; push and starting-value branches forbidden |
| Unused bounded BASIC warm candidate snapped to a bound at default distance | `starting_boxed_superbasic_snapped_to_lower_bound` | Exercised; push explicitly forbidden |
| Bounded BASIC candidate snapped at the upper-distance equality boundary | `starting_boxed_superbasic_snapped_at_upper_distance_boundary` | Exercised; push explicitly forbidden |
| Bounded super-basic variable pushed to its nearest lower or upper bound when snapping is disabled | `starting_boxed_variable_pushed_to_lower_bound`, `starting_boxed_variable_pushed_to_upper_bound` | Exercised |
| Dual-to-primal cleanup decision at the iteration limit and full reoptimization | `cleanup_dual_to_primal_at_iteration_limit`, `cleanup_dual_to_primal_reoptimization` | Exercised |
| Primal-to-dual cleanup and dual reoptimization | `cleanup_primal_to_dual_reoptimization` | Exercised |
| Primal bound shift that forces primal-to-dual cleanup, with a tight negative control | `primal_bound_shift_forces_dual_cleanup`, `primal_bound_shift_tight_cleanup` | Shift-induced switch and dual reoptimization match native state and exact clock; one-row tight-tolerance control explicitly forbids the switch |
| Cleanup reoptimization count exhausted | `cleanup_reoptimization_limit_zero`, `cleanup_reoptimization_limit_zero_dual` | Exercised in both cross-algorithm directions with zero allowed reoptimizations; follow-up solve forbidden, exact native clocks |
| Relaxed internal tolerances avoid both cross-algorithm cleanup switches | `cleanup_primal_to_dual_relaxed_tolerance`, `cleanup_dual_to_primal_relaxed_tolerance` | Exercised and explicitly forbidden in the relaxed cases |
| Dual cost shift and its removal | `dual_degenerate_cost_shift` | Exercised |
| Dual cost-shift removal followed by dual-to-primal cleanup | `dual_cost_shift_forces_primal_cleanup`, `zero_internal_tolerance_dual_cleanup` | Native and Rust agree on switch, primal reoptimization, terminal state, and exact clock, including the zero-internal-tolerance boundary |
| Dual perturbation removal after Phase II followed by dual-to-primal cleanup | `dual_perturbation_removal_forces_primal_cleanup` | Native and Rust agree on switch, primal reoptimization, terminal state, and exact clock; cost shift forbidden |
| Dual boxed-variable bound flip | `dual_boxed_bound_flip` | Exercised |
| Immediate deterministic-time limit in primal Phase I | `primal_deterministic_limit_zero` | Exercised |
| Immediate deterministic-time limit in dedicated dual Phase I | `dual_deterministic_limit_zero` | Exercised |
| Positive wall-time limit reached before primal Phase I work | `primal_positive_wall_limit_before_phase_one` | Exercised with a 1 µs limit; native and Rust statuses and operation clocks agree |
| Positive wall-time limit reached before dedicated dual Phase I work | `dual_positive_wall_limit_before_phase_one` | Exercised with a 1 µs limit; native and Rust statuses and operation clocks agree |
| Positive deterministic-time limit after primal Phase-I work | `primal_deterministic_limit_positive` | Exercised; exact native clock |
| Positive deterministic-time limit after dual work | `dual_deterministic_limit_positive` | Exercised; exact native clock |
| Adjacent `f64` deterministic limits around a primal Phase-I loop boundary | `primal_phase_one_limit_at_clock_boundary`, `primal_phase_one_limit_after_clock_boundary` | Exercised; one versus two pivots, with exact native clocks |
| Adjacent `f64` deterministic limits around dedicated dual Phase I | `dual_limit_at_clock_boundary`, `dual_limit_after_clock_boundary` | Exercised; zero versus one pivot, with exact native clocks |
| Adjacent `f64` deterministic limits around primal Phase II | `primal_phase_two_limit_at_clock_boundary`, `primal_phase_two_limit_after_clock_boundary` | Exercised; one versus two pivots, with exact native clocks |
| Adjacent `f64` deterministic limits around dual Phase II | `dual_phase_two_limit_at_clock_boundary`, `dual_phase_two_limit_after_clock_boundary` | Exercised; one versus two pivots, with exact native clocks |
| Deterministic limit reached during cleanup after a primal bound shift or dual cost shift | `primal_cleanup_stopped_by_time_limit`, `dual_cleanup_stopped_by_time_limit` | Both hit the cleanup time-limit decision; native status, ordered basis, solution, and operation clock agree |
| Post-optimal integrality polishing, bound flip, basis pivot, and randomized candidate rejection | `post_optimal_integrality_polish`, `post_optimal_integrality_polish_pivot`, `post_optimal_integrality_polish_multiple_candidates`, `post_optimal_integrality_polish_control` | Optional scale-enabled path performs an improving bound flip or pivot; two-candidate case rejects a non-improving choice before pivoting; no-scale control retains the fractional solution. Native terminal state and operation clock agree |
| Deterministic time limit reached before polishing makes a move | `post_optimal_integrality_polish_time_limit` | Optimal solve enters polishing but stops before the first candidate; native status, unchanged fractional values, basis, and exact clock agree |
| Explicitly cleared integrality scales | `post_optimal_integrality_scales_cleared` | Setting then clearing the scale vector disables polishing; native fractional solution, basis, and clock agree |
| Polishing after cleanup reoptimization | `post_optimal_integrality_polish_after_cleanup_reoptimization` | Primal-to-dual cleanup causes a second optimization; both post-optimal polishing calls match native terminal state and operation clock |
| Polishing accepts a pivot, then rejects the next candidate or stops at the deterministic limit | `post_optimal_integrality_polish_accept_then_reject`, `post_optimal_integrality_polish_time_limit_after_pivot` | Candidate removal and the post-pivot time check match native terminal state and exact clocks; the limited case forbids the later rejection |
| Advanced transposed-matrix entry, changed bounds/matrix, and transformed slack objective | `transposed_matrix_with_slack_entry`, `transposed_matrix_with_slack_warm_bound_change`, `transposed_matrix_with_slack_changed_matrix`, `transposed_matrix_with_slack_nonzero_slack_cost_and_objective_transform` | First solve, quick dual warm solve after a bound mutation, full reinitialization after a matrix mutation, and nonzero slack cost with objective scale/offset match native state and exact operation clocks. Unchanged advanced calls skip the matrix scan; the equation-form adapter borrows the compact candidate without a sparse-column copy. Three large Netlib models have a serial advanced-entry timing comparison below |
| Zero-row advanced transposed solve | `transposed_matrix_with_slack_zero_rows` | Native/Rust status, variable values, reduced costs, and operation clocks agree without a basis or slack columns |
| Changed objective on the advanced transposed entry | `transposed_matrix_with_slack_warm_objective_change` | Native and Rust reject the unchanged-objective dual fast path, rebuild from the saved basis, and agree on status, basis, values, reduced costs, and exact clock |
| External starting basis on the advanced transposed entry | `transposed_matrix_with_slack_external_basis` | Native and Rust initialize from the caller-supplied structural basis with matching terminal state and exact clock |
| Direct, unscaled qap15 prefixes and former clock split (deferred; not a Phase-4 gate) | `qap15_dual`, `qap15-clock-transition.json` | Status, ordered basis, reduced-cost bits, dual-norm bits, and deterministic clock agree through 150,000 pivots. The former pivot-28,149 split was caused by invalidating `UpdateRow` after a no-op refactorization; the opt-in fixture now requires exact native clock equality |

Phase-4 exit evidence: the 96 fast Netlib models agree with native GLOP on
status, iteration count, ordered basis, full pivot path, reduced-cost bits, and
dual-norm bits. The same regression independently recomputes objective,
primal equations and bounds, dual stationarity, and complementary dual signs
from each input MPS. The 159 small native-pinned cases exercise the named
decision branches above, including zero-internal-tolerance cleanup.

Further coverage/performance work:

The default `exploit_singleton_column_in_initial_basis` path now selects
singleton structural columns before the advanced crash, matching native GLOP.
The Phase-5 partial-pipeline fixture checks enabled and disabled controls,
including the formerly divergent zero-pivot one-row solves.

- The advanced transposed-matrix entry has seven native-pinned functional
  cases and shares its compact matrix with the basis representation without a
  sparse-column copy. On `fit2p`, `maros-r7`, and `stocfor3`, serial cold-entry
  Rust/native solve-time ratios were 1.309, 0.958, and 1.546; ordinary-entry
  controls were 1.310, 0.957, and 1.547. Initialization-only ratios were
  1.587, 1.966, and 0.957 for advanced entry versus 1.662, 2.118, and 1.218
  for ordinary entry. Thus no advanced-specific penalty was measured, but the
  shared solver performance gap remains Phase-8 work. Reproduce with
  `tools/benchmark_netlib_solve.py --mode advanced` and `--mode dual`, adding
  `--iteration-limit 0` for initialization-only timing.
- A precision request inside `Polish()` is not reachable in the pinned upstream
  control flow: its initial `GetReducedCosts()` clears the request on the
  refactorized optimal basis, and no call inside `Polish()` sets it again. The
  branch is retained for source correspondence; polishing moves, rejection,
  cleanup reoptimization, and time limits have native fixtures.
- Additional both-infeasible cleanup, ray, warm-start mutation, Harris,
  pivot-precision, wall-clock, and error-exit variants can strengthen branch
  coverage. The corresponding algorithms are implemented; this ledger does
  not claim exhaustive branch coverage. Native matrix mutation requires
  `CleanUp()` before the next `Solve()`.

For each missing path, add an event at the decision site, a minimized LP (or a
multi-solve mutation sequence), and a pinned-native expected result. The test
must assert the event was hit; matching terminal status alone is insufficient.

Deferred until scaling and preprocessing are implemented: direct, unscaled
qap15 terminal behavior and prefixes beyond 150,000 pivots. The existing
qap15 fixtures remain opt-in regressions, not Phase-4 completion gates.
