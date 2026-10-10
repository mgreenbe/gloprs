//! Small, branch-targeted `RevisedSimplex` cases against pinned native GLOP.

use glop::lu_factorization::FactorizationError;
use glop::parameters::{GlopParameters, InitialBasisHeuristic, PricingRule};
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use glop::variables_info::BasisState;
use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{
    ColIndex, DenseBooleanColumn, DenseBooleanRow, RowIndex, VariableStatus, VariableStatusRow,
    VectorIndex,
};
use serde::Deserialize;
use sha2::Digest;
use std::collections::BTreeSet;

const CASES: &str = include_str!("../../baselines/phase4-cases.json");
const EXPECTED: &str = include_str!("../../baselines/phase4-native.json");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

#[derive(Deserialize)]
struct Specification {
    upstream_commit: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    mode: String,
    iterations: i64,
    #[serde(default)]
    deterministic_limit: Option<String>,
    matrix: Vec<(usize, usize, String)>,
    variables: Vec<(String, String, String)>,
    constraints: Vec<(String, String)>,
    covers: Vec<String>,
    #[serde(default)]
    forbids: Vec<String>,
}

#[derive(Deserialize)]
struct Fixture {
    upstream_commit: String,
    cases_sha256: String,
    results: Vec<NativeResult>,
}

#[derive(Deserialize)]
struct NativeResult {
    name: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    status: String,
    #[serde(default)]
    iterations: u64,
    #[serde(default)]
    objective_bits: u64,
    #[serde(default)]
    lu_pivot_threshold_bits: u64,
    #[serde(default)]
    deterministic_time_bits: u64,
    #[serde(default)]
    basis: Vec<usize>,
    #[serde(default)]
    value_bits: Vec<u64>,
    #[serde(default)]
    reduced_bits: Vec<u64>,
    #[serde(default)]
    primal_ray_bits: Vec<u64>,
    #[serde(default)]
    dual_ray_bits: Vec<u64>,
}

fn number(value: &str) -> f64 {
    match value {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => value.parse().expect("invalid LP coefficient or bound"),
    }
}

fn normalized_zero(bits: u64) -> u64 {
    if bits == (-0.0_f64).to_bits() {
        0
    } else {
        bits
    }
}

#[test]
fn all_slack_basis_rejected_by_condition_threshold() {
    // Pinned native GLOP returns ERROR_LU with upper bound 1 when this
    // threshold is below 1; the fallback basis must not silently pass.
    let mut lp = LinearProgram::default();
    let column = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_coefficient(row, column, 1.0);
    lp.set_variable_bounds(column, 0.0, 2.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 1.0);
    lp.set_objective_coefficient(column, -1.0);
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_scaling: false,
        initial_basis: InitialBasisHeuristic::None,
        initial_condition_number_threshold: 0.1,
        ..GlopParameters::default()
    });
    assert_eq!(
        simplex.solve(&lp, &mut TimeLimit::new(20.0, f64::INFINITY)),
        Err(FactorizationError::IllConditioned { upper_bound: 1.0 })
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn small_phase4_cases_match_pinned_native_glop() {
    let specification: Specification = serde_json::from_str(CASES).unwrap();
    let fixture: Fixture = serde_json::from_str(EXPECTED).unwrap();
    assert_eq!(specification.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(
        fixture.cases_sha256,
        format!("{:x}", sha2::Sha256::digest(CASES.as_bytes()))
    );
    assert_eq!(specification.cases.len(), fixture.results.len());
    let covered: BTreeSet<_> = specification
        .cases
        .iter()
        .flat_map(|case| case.covers.iter().map(String::as_str))
        .collect();
    for call in include_str!("../src/revised_simplex.rs")
        .split("self.phase4_event(\"")
        .skip(1)
    {
        let event = call.split('"').next().expect("unterminated Phase-4 event");
        assert!(
            covered.contains(event),
            "no native fixture covers Phase-4 event {event}"
        );
    }

    for (case, expected) in specification.cases.iter().zip(&fixture.results) {
        assert_eq!(case.name, expected.name);
        assert!(!case.covers.is_empty(), "{}: no branch tags", case.name);
        let mut lp = LinearProgram::default();
        for _ in &case.variables {
            lp.create_new_variable();
        }
        for _ in &case.constraints {
            lp.create_new_constraint();
        }
        for &(row, column, ref coefficient) in &case.matrix {
            lp.set_coefficient(
                RowIndex::from_usize(row),
                ColIndex::from_usize(column),
                number(coefficient),
            );
        }
        for (column, (objective, lower, upper)) in case.variables.iter().enumerate() {
            let column = ColIndex::from_usize(column);
            lp.set_objective_coefficient(column, number(objective));
            lp.set_variable_bounds(column, number(lower), number(upper));
        }
        for (row, (lower, upper)) in case.constraints.iter().enumerate() {
            lp.set_constraint_bounds(RowIndex::from_usize(row), number(lower), number(upper));
        }

        let mut simplex = RevisedSimplex::new();
        simplex.set_phase4_events_enabled(true);
        simplex.set_parameters(&GlopParameters {
            use_scaling: false,
            initial_basis: match case.mode.as_str() {
                "initial_triangular_primal"
                | "initial_triangular_dual"
                | "initial_triangular_condition_fallback" => InitialBasisHeuristic::Triangular,
                "initial_maros_primal" | "initial_maros_dual" => InitialBasisHeuristic::Maros,
                "initial_bixby_without_scaling" => InitialBasisHeuristic::Bixby,
                _ => InitialBasisHeuristic::None,
            },
            exploit_singleton_column_in_initial_basis: false,
            use_dual_simplex: !matches!(
                case.mode.as_str(),
                "primal"
                    | "initial_triangular_primal"
                    | "initial_triangular_condition_fallback"
                    | "initial_maros_primal"
                    | "initial_bixby_without_scaling"
                    | "primal_limit"
                    | "primal_no_imprecise"
                    | "primal_time_zero"
                    | "primal_wall_tiny"
                    | "primal_steepest"
                    | "primal_steepest_zero_norm_threshold"
                    | "primal_devex"
                    | "primal_harris_wide"
                    | "primal_eta"
                    | "primal_eta_phase_one"
                    | "no_reopt_primal"
                    | "tight_internal_primal"
                    | "relaxed_internal_primal"
                    | "warm_primal"
                    | "warm_auto_primal_repeat"
                    | "warm_clear_state"
                    | "warm_external_then_restore"
                    | "warm_auto_primal_objective_change"
                    | "warm_primal_limit_change"
                    | "warm_bound_change"
                    | "warm_objective_change"
                    | "warm_multiple_bound_changes"
                    | "external_basis"
                    | "warm_added_column"
                    | "starting_values"
                    | "starting_values_push"
                    | "starting_values_push_boxed"
                    | "starting_values_push_boxed_no_snap"
                    | "starting_values_push_boxed_upper_no_snap"
                    | "starting_values_push_boxed_snap_upper_boundary"
                    | "starting_values_push_with_row"
                    | "starting_values_push_refactorize"
                    | "starting_values_push_control"
                    | "starting_values_two_no_push"
                    | "zero_tolerance"
                    | "tight_tolerance"
                    | "loose_tolerance"
                    | "precision_refactorization"
                    | "adaptive_pivot"
                    | "zero_tolerance_no_imprecise"
            ),
            perturb_costs_in_dual_simplex: case.mode == "dual_perturbed",
            use_middle_product_form_update: !matches!(
                case.mode.as_str(),
                "primal_eta" | "primal_eta_phase_one" | "dual_eta" | "dual_eta_phase_one"
            ),
            use_dedicated_dual_feasibility_algorithm: case.mode != "dual_transformed",
            max_number_of_iterations: case.iterations,
            harris_tolerance_ratio: if case.mode == "primal_harris_wide" {
                10.0
            } else {
                GlopParameters::default().harris_tolerance_ratio
            },
            initial_condition_number_threshold: if case.mode == "initial_all_slack_condition_error"
            {
                0.1
            } else if case.mode == "initial_triangular_condition_fallback" {
                1.0
            } else {
                GlopParameters::default().initial_condition_number_threshold
            },
            max_number_of_reoptimizations: if matches!(
                case.mode.as_str(),
                "no_reopt_primal" | "no_reopt_dual"
            ) {
                0.0
            } else {
                GlopParameters::default().max_number_of_reoptimizations
            },
            push_to_vertex: !matches!(
                case.mode.as_str(),
                "starting_values" | "starting_values_two_no_push"
            ),
            solution_feasibility_tolerance: if case.mode == "tight_tolerance" {
                1e-18
            } else if case.mode == "loose_tolerance" {
                1e-16
            } else if case.mode.starts_with("zero_tolerance") {
                0.0
            } else {
                GlopParameters::default().solution_feasibility_tolerance
            },
            change_status_to_imprecise: !matches!(
                case.mode.as_str(),
                "zero_tolerance_no_imprecise" | "primal_no_imprecise" | "dual_no_imprecise"
            ),
            recompute_reduced_costs_threshold: if case.mode == "precision_refactorization" {
                0.0
            } else {
                GlopParameters::default().recompute_reduced_costs_threshold
            },
            recompute_edges_norm_threshold: if matches!(
                case.mode.as_str(),
                "primal_steepest_zero_norm_threshold" | "dual_zero_norm_threshold"
            ) {
                0.0
            } else {
                GlopParameters::default().recompute_edges_norm_threshold
            },
            refactorization_threshold: if matches!(
                case.mode.as_str(),
                "adaptive_pivot" | "dual_adaptive_pivot"
            ) {
                0.0
            } else {
                GlopParameters::default().refactorization_threshold
            },
            small_pivot_threshold: if case.mode == "starting_values_push_refactorize" {
                0.1
            } else {
                GlopParameters::default().small_pivot_threshold
            },
            crossover_bound_snapping_distance: if matches!(
                case.mode.as_str(),
                "starting_values_push_boxed_no_snap" | "starting_values_push_boxed_upper_no_snap"
            ) {
                0.0
            } else if case.mode == "starting_values_push_boxed_snap_upper_boundary" {
                0.25
            } else {
                GlopParameters::default().crossover_bound_snapping_distance
            },
            feasibility_rule: PricingRule::Dantzig,
            dual_price_prioritize_norm: case.mode == "dual_prioritize_norm",
            optimization_rule: match case.mode.as_str() {
                "primal_steepest" | "primal_steepest_zero_norm_threshold" => {
                    PricingRule::SteepestEdge
                }
                "primal_devex" => PricingRule::Devex,
                _ => PricingRule::Dantzig,
            },
            primal_feasibility_tolerance: if matches!(
                case.mode.as_str(),
                "relaxed_internal_primal" | "relaxed_internal_dual"
            ) {
                1e-15
            } else if matches!(
                case.mode.as_str(),
                "tight_internal_dual"
                    | "tight_internal_primal"
                    | "no_reopt_primal"
                    | "no_reopt_dual"
            ) {
                1e-16
            } else {
                GlopParameters::default().primal_feasibility_tolerance
            },
            dual_feasibility_tolerance: if matches!(
                case.mode.as_str(),
                "relaxed_internal_primal" | "relaxed_internal_dual"
            ) {
                1e-15
            } else if matches!(
                case.mode.as_str(),
                "tight_internal_dual"
                    | "tight_internal_primal"
                    | "no_reopt_primal"
                    | "no_reopt_dual"
            ) {
                1e-16
            } else {
                GlopParameters::default().dual_feasibility_tolerance
            },
            objective_lower_limit: if case.mode == "primal_limit" {
                -0.5
            } else {
                f64::NEG_INFINITY
            },
            objective_upper_limit: if case.mode == "dual_limit" {
                0.5
            } else {
                f64::INFINITY
            },
            ..GlopParameters::default()
        });
        if matches!(
            case.mode.as_str(),
            "starting_values"
                | "starting_values_push"
                | "starting_values_push_boxed"
                | "starting_values_push_boxed_no_snap"
                | "starting_values_push_boxed_upper_no_snap"
                | "starting_values_push_boxed_snap_upper_boundary"
                | "starting_values_push_with_row"
                | "starting_values_push_refactorize"
                | "starting_values_push_control"
                | "starting_values_two_no_push"
        ) {
            let start = if matches!(
                case.mode.as_str(),
                "starting_values_push_boxed_upper_no_snap"
                    | "starting_values_push_boxed_snap_upper_boundary"
            ) {
                0.75
            } else {
                0.5
            };
            simplex.set_starting_variable_values_for_next_solve(
                &lp_data::lp_types::DenseRow::from_vec(
                    if matches!(
                        case.mode.as_str(),
                        "starting_values_push_refactorize"
                            | "starting_values_push_control"
                            | "starting_values_two_no_push"
                    ) {
                        vec![start, 0.5]
                    } else {
                        vec![start]
                    },
                ),
            );
        }
        if case.mode == "external_basis" {
            simplex.load_state_for_next_solve(&BasisState {
                statuses: VariableStatusRow::from_vec(vec![
                    VariableStatus::Basic,
                    VariableStatus::AtUpperBound,
                ]),
            });
        }
        if matches!(
            case.mode.as_str(),
            "starting_values_push_boxed"
                | "starting_values_push_boxed_no_snap"
                | "starting_values_push_boxed_upper_no_snap"
                | "starting_values_push_boxed_snap_upper_boundary"
        ) {
            simplex.load_state_for_next_solve(&BasisState {
                statuses: VariableStatusRow::from_vec(vec![VariableStatus::Basic]),
            });
        }
        if matches!(
            case.mode.as_str(),
            "warm_primal"
                | "warm_auto_primal_repeat"
                | "warm_clear_state"
                | "warm_external_then_restore"
                | "warm_auto_primal_objective_change"
                | "warm_auto_dual_repeat"
                | "warm_auto_dual_bound_change"
                | "warm_primal_limit_change"
                | "warm_bound_change"
                | "warm_dual_bound_change"
                | "warm_dual_repeat"
                | "warm_dual_limit_change"
                | "warm_objective_change"
                | "warm_multiple_bound_changes"
                | "warm_dual_multiple_bound_changes"
                | "warm_added_column"
                | "warm_added_row"
                | "warm_added_row_and_column"
                | "warm_removed_row"
                | "warm_removed_column"
                | "warm_added_row_low_condition_threshold"
                | "warm_added_row_slack"
                | "warm_added_row_changed_coefficient"
                | "warm_singular_saved_basis"
                | "warm_singular_saved_basis_condition_reject"
                | "warm_singular_saved_basis_condition_error"
                | "warm_changed_coefficient_only"
                | "warm_changed_two_row_full_rank"
                | "warm_saved_basis_condition_reject"
        ) {
            simplex
                .solve(&lp, &mut TimeLimit::new(20.0, f64::INFINITY))
                .unwrap_or_else(|error| panic!("{}: first solve: {error}", case.name));
            let state = simplex.state().clone();
            if case.mode == "warm_clear_state" {
                simplex.clear_state_for_next_solve();
                assert!(simplex.state().is_empty());
            } else if case.mode == "warm_external_then_restore" {
                simplex.load_state_for_next_solve(&BasisState {
                    statuses: VariableStatusRow::from_vec(vec![
                        VariableStatus::AtLowerBound,
                        VariableStatus::Basic,
                    ]),
                });
                simplex.load_state_for_next_solve(&state);
            } else if !case.mode.starts_with("warm_auto_") {
                simplex.load_state_for_next_solve(&state);
            }
            if case.mode == "warm_primal_limit_change" {
                let mut parameters = simplex.parameters().clone();
                parameters.objective_lower_limit = -0.5;
                simplex.set_parameters(&parameters);
            }
            if case.mode == "warm_dual_limit_change" {
                let mut parameters = simplex.parameters().clone();
                parameters.objective_upper_limit = 0.5;
                simplex.set_parameters(&parameters);
            }
            if matches!(
                case.mode.as_str(),
                "warm_added_row_low_condition_threshold"
                    | "warm_saved_basis_condition_reject"
                    | "warm_singular_saved_basis_condition_reject"
            ) {
                let mut parameters = simplex.parameters().clone();
                parameters.initial_condition_number_threshold = 1.0;
                simplex.set_parameters(&parameters);
            }
            if case.mode == "warm_singular_saved_basis_condition_error" {
                let mut parameters = simplex.parameters().clone();
                parameters.initial_condition_number_threshold = 0.1;
                simplex.set_parameters(&parameters);
            }
            if matches!(
                case.mode.as_str(),
                "warm_bound_change" | "warm_dual_bound_change" | "warm_auto_dual_bound_change"
            ) {
                lp.set_constraint_bounds(RowIndex::from_usize(0), f64::NEG_INFINITY, 0.5);
            }
            if matches!(
                case.mode.as_str(),
                "warm_objective_change" | "warm_auto_primal_objective_change"
            ) {
                lp.set_objective_coefficient(ColIndex::from_usize(0), 1.0);
            }
            if matches!(
                case.mode.as_str(),
                "warm_multiple_bound_changes" | "warm_dual_multiple_bound_changes"
            ) {
                lp.set_constraint_bounds(RowIndex::from_usize(0), f64::NEG_INFINITY, 0.5);
                lp.set_constraint_bounds(RowIndex::from_usize(1), f64::NEG_INFINITY, 0.5);
            }
            if matches!(
                case.mode.as_str(),
                "warm_added_column" | "warm_added_row_and_column"
            ) {
                let added = lp.create_new_variable();
                lp.set_variable_bounds(added, 0.0, f64::INFINITY);
                lp.set_objective_coefficient(
                    added,
                    if case.mode == "warm_added_column" {
                        -2.0
                    } else {
                        0.0
                    },
                );
                lp.set_coefficient(RowIndex::from_usize(0), added, 1.0);
            }
            if matches!(
                case.mode.as_str(),
                "warm_singular_saved_basis"
                    | "warm_singular_saved_basis_condition_reject"
                    | "warm_singular_saved_basis_condition_error"
            ) {
                lp.set_coefficient(RowIndex::from_usize(0), ColIndex::from_usize(1), 1.0);
                lp.set_coefficient(RowIndex::from_usize(1), ColIndex::from_usize(1), 0.0);
            }
            if case.mode == "warm_changed_coefficient_only" {
                lp.set_coefficient(RowIndex::from_usize(0), ColIndex::from_usize(0), 2.0);
            }
            if case.mode == "warm_changed_two_row_full_rank" {
                lp.set_coefficient(RowIndex::from_usize(0), ColIndex::from_usize(1), 1.0);
            }
            if case.mode == "warm_saved_basis_condition_reject" {
                lp.set_coefficient(RowIndex::from_usize(0), ColIndex::from_usize(0), 2.0);
            }
            if matches!(
                case.mode.as_str(),
                "warm_added_row"
                    | "warm_added_row_and_column"
                    | "warm_added_row_slack"
                    | "warm_added_row_low_condition_threshold"
                    | "warm_added_row_changed_coefficient"
            ) {
                let added = lp.create_new_constraint();
                let lower = if case.mode == "warm_added_row_slack" {
                    1.0
                } else {
                    2.0
                };
                lp.set_constraint_bounds(added, lower, f64::INFINITY);
                lp.set_coefficient(added, ColIndex::from_usize(0), 1.0);
                if case.mode == "warm_added_row_and_column" {
                    lp.set_coefficient(added, ColIndex::from_usize(1), 1.0);
                }
                if case.mode == "warm_added_row_changed_coefficient" {
                    lp.set_coefficient(RowIndex::from_usize(0), ColIndex::from_usize(0), 2.0);
                }
            }
            if case.mode == "warm_removed_row" {
                let mut deleted = DenseBooleanColumn::filled(RowIndex::from_usize(2), false);
                deleted[RowIndex::from_usize(1)] = true;
                lp.delete_rows(&deleted);
            }
            if case.mode == "warm_removed_column" {
                let mut deleted = DenseBooleanRow::filled(ColIndex::from_usize(2), false);
                deleted[ColIndex::from_usize(1)] = true;
                lp.delete_columns(&deleted);
            }
        }
        let deterministic_limit =
            if matches!(case.mode.as_str(), "primal_time_zero" | "dual_time_zero") {
                0.0
            } else {
                case.deterministic_limit
                    .as_deref()
                    .map_or(f64::INFINITY, number)
            };
        let clock_before = simplex.deterministic_time();
        let wall_limit = if matches!(case.mode.as_str(), "primal_wall_tiny" | "dual_wall_tiny") {
            1e-6
        } else {
            20.0
        };
        let mut time_limit = TimeLimit::new(wall_limit, deterministic_limit);
        let result = simplex.solve(&lp, &mut time_limit);
        if let Some(expected_error) = &expected.error {
            assert_eq!(
                result.unwrap_err().to_string(),
                *expected_error,
                "{}: error",
                case.name
            );
            for event in &case.covers {
                assert!(
                    simplex.phase4_events().contains(&event.as_str()),
                    "{}: expected branch {event}",
                    case.name
                );
            }
            continue;
        }
        result.unwrap_or_else(|error| panic!("{}: {error}", case.name));
        let charged = simplex.deterministic_time() - clock_before;
        assert!(
            (time_limit.elapsed_deterministic_time() - charged).abs() <= 1e-18,
            "{}: deterministic clock was not fully charged",
            case.name
        );
        if std::env::var_os("GLOPRS_AUDIT_CLOCK").is_some() {
            eprintln!(
                "{}: native={} rust={}",
                case.name,
                f64::from_bits(expected.deterministic_time_bits),
                simplex.deterministic_time()
            );
        }
        let native_clock = f64::from_bits(expected.deterministic_time_bits);
        assert!(
            (simplex.deterministic_time() - native_clock).abs() <= 1e-18,
            "{}: deterministic clock diverges from native: rust={} native={}",
            case.name,
            simplex.deterministic_time(),
            native_clock
        );
        for event in &case.covers {
            assert!(
                simplex.phase4_events().contains(&event.as_str()),
                "{}: expected branch {event}, got {:?}",
                case.name,
                simplex.phase4_events()
            );
        }
        for event in &case.forbids {
            assert!(
                !simplex.phase4_events().contains(&event.as_str()),
                "{}: forbidden branch {event} was reached",
                case.name
            );
        }
        assert_eq!(
            simplex.problem_status().to_string(),
            expected.status,
            "{}: status",
            case.name
        );
        assert_eq!(
            simplex.number_of_iterations(),
            expected.iterations,
            "{}: iterations",
            case.name
        );
        assert_eq!(
            normalized_zero(simplex.objective_value().to_bits()),
            normalized_zero(expected.objective_bits),
            "{}: objective",
            case.name
        );
        assert_eq!(
            simplex
                .parameters()
                .lu_factorization_pivot_threshold
                .to_bits(),
            expected.lu_pivot_threshold_bits,
            "{}: adaptive LU pivot threshold",
            case.name
        );
        let basis: Vec<_> = (0..lp.num_constraints().to_usize())
            .map(|row| simplex.basis(RowIndex::from_usize(row)).to_usize())
            .collect();
        assert_eq!(basis, expected.basis, "{}: basis", case.name);
        let total_columns = lp.num_variables().to_usize() + lp.num_constraints().to_usize();
        let value_bits: Vec<_> = (0..total_columns)
            .map(|column| {
                normalized_zero(
                    simplex
                        .variable_value(ColIndex::from_usize(column))
                        .to_bits(),
                )
            })
            .collect();
        let expected_values: Vec<_> = expected
            .value_bits
            .iter()
            .copied()
            .map(normalized_zero)
            .collect();
        assert_eq!(value_bits, expected_values, "{}: values", case.name);
        let reduced_bits: Vec<_> = (0..total_columns)
            .map(|column| {
                normalized_zero(simplex.reduced_cost(ColIndex::from_usize(column)).to_bits())
            })
            .collect();
        let expected_reduced: Vec<_> = expected
            .reduced_bits
            .iter()
            .copied()
            .map(normalized_zero)
            .collect();
        assert_eq!(
            reduced_bits, expected_reduced,
            "{}: reduced costs",
            case.name
        );
        let primal_ray: Vec<_> = simplex
            .primal_ray()
            .as_slice()
            .iter()
            .map(|value| normalized_zero(value.to_bits()))
            .collect();
        let native_primal_ray: Vec<_> = expected
            .primal_ray_bits
            .iter()
            .copied()
            .map(normalized_zero)
            .collect();
        assert_eq!(primal_ray, native_primal_ray, "{}: primal ray", case.name);
        let dual_ray: Vec<_> = simplex
            .dual_ray()
            .as_slice()
            .iter()
            .map(|value| normalized_zero(value.to_bits()))
            .collect();
        let native_dual_ray: Vec<_> = expected
            .dual_ray_bits
            .iter()
            .copied()
            .map(normalized_zero)
            .collect();
        assert_eq!(dual_ray, native_dual_ray, "{}: dual ray", case.name);
        if case.name == "primal_unbounded_coupled_ray" {
            assert_eq!(
                primal_ray[..case.variables.len()]
                    .iter()
                    .filter(|&&bits| bits != 0)
                    .count(),
                2,
                "coupled primal certificate must involve both structural columns"
            );
        }
        if case.name == "dual_unbounded_coupled_ray" {
            assert_eq!(
                dual_ray.iter().filter(|&&bits| bits != 0).count(),
                2,
                "coupled dual certificate must involve both rows"
            );
        }
    }
}
