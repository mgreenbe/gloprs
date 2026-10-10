# Phase-4 branch fixture coverage

The small-LP fixture currently has 80 pinned-native cases. Each listed branch
tag is emitted by the Rust driver only when that path is taken; the test also
compares the case's final state to GLOP. This is **not** a claim that every
branch of `RevisedSimplex` is covered. The inventory below is the outstanding
work needed to make that claim.

| Path | Case(s) | State |
|---|---|---|
| Primal Phase I and infeasible exit | `primal_phase_one_infeasible` | Exercised |
| Primal Phase II, bounded pivot and optimal cleanup | `primal_optimization` | Exercised |
| Primal bound flip | `primal_boxed_bound_flip` | Exercised |
| Degenerate primal pivot retaining an off-bound leaving value | `primal_degenerate_pivot` | Exercised |
| Reduced-cost precision retry and forced refactorization | `primal_precision_refactorization` | Exercised |
| Early imprecise pivot and adaptive LU threshold escalation | `adaptive_lu_pivot_threshold` | Exercised; native/Rust final threshold bits agree |
| Exact primal Harris tie and shared-RNG choice | `primal_harris_exact_tie` | Exercised |
| Primal unbounded ray | `primal_unbounded` | Exercised |
| Primal ray postsolve validation (accepted) | `primal_unbounded` | Exercised |
| Primal weak-ray rejection and stronger-ray acceptance | `primal_weak_ray_rejected`, `primal_strong_ray_accepted` | Exercised |
| Primal weak-ray rejection with imprecise conversion disabled | `primal_weak_ray_rejected_without_imprecise_conversion` | Exercised |
| Primal objective limit | `primal_objective_limit` | Exercised |
| Dedicated dual Phase I | `dual_dedicated_phase_one` | Exercised |
| Transformed dual Phase I | `dual_transformed_phase_one` | Exercised |
| Initial dual cost perturbation | `dual_perturbation` | Exercised |
| Perturbed-cost removal | `dual_perturbation` | Exercised |
| Dual Phase II, bounded pivot and optimal cleanup | `dual_phase_two` | Exercised |
| Dual iteration limit | `dual_iteration_limit_zero` | Exercised |
| Dual Phase-I iteration limit | `dual_phase_one_iteration_limit_zero` | Exercised |
| Already optimal at zero dual iteration limit | `dual_already_optimal_at_zero_limit` | Exercised |
| Dual objective limit | `dual_objective_limit` | Exercised |
| Dual-unbounded ray | `dual_unbounded_ray` | Exercised |
| Dual ray postsolve validation (accepted) | `dual_unbounded_ray` | Exercised |
| Dual weak-ray rejection and tolerance-boundary acceptance | `dual_weak_ray_rejected`, `dual_ray_at_solution_tolerance` | Exercised |
| Dual weak-ray rejection with imprecise conversion disabled | `dual_weak_ray_without_imprecise_conversion` | Exercised |
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
| Added-row warm basis rejected by condition threshold, then recovered with a fresh basis | `warm_added_row_condition_fallback` | Exercised; computed Markowitz candidate basis is used and native operation clock agrees |
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
| Relaxed internal tolerances avoid both cross-algorithm cleanup switches | `cleanup_primal_to_dual_relaxed_tolerance`, `cleanup_dual_to_primal_relaxed_tolerance` | Exercised and explicitly forbidden in the relaxed cases |
| Dual cost shift and its removal | `dual_degenerate_cost_shift` | Exercised |
| Dual boxed-variable bound flip | `dual_boxed_bound_flip` | Exercised |
| Immediate deterministic-time limit in primal Phase I | `primal_deterministic_limit_zero` | Exercised |
| Immediate deterministic-time limit in dedicated dual Phase I | `dual_deterministic_limit_zero` | Exercised |
| Positive deterministic-time limit after primal Phase-I work | `primal_deterministic_limit_positive` | Exercised; clock totals still differ from native |
| Positive deterministic-time limit after dual work | `dual_deterministic_limit_positive` | Exercised; clock totals still differ from native |

Still needed:

- Larger primal Phase-II bound shifts that specifically force the now-covered
  primal-to-dual cleanup switch. The small degenerate fixture retains an
  off-bound leaving value, but its shift clears without forcing reoptimization.
- A dual-to-primal cleanup switch specifically caused by cost-shift or
  perturbation removal. One-sided tolerance-sensitive switches are covered;
  the both-infeasible cleanup classification and sharper boundary cases remain.
- More ray configurations beyond the six targeted threshold cases above.
- Other dimension-changing warm-start failure modes remain untested.
- Edge-norm and other pivot-precision refactorizations, plus further Harris
  stability cases.
- Nonzero wall-clock limits and near-boundary deterministic limits (the
  cumulative clock is connected, but operation totals still differ),
  singular/abnormal and error exits, and
  iteration-limit placements beyond the Phase-II cases above.

For each missing path, add an event at the decision site, a minimized LP (or a
multi-solve mutation sequence), and a pinned-native expected result. The test
must assert the event was hit; matching terminal status alone is insufficient.
