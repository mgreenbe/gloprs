//! Small, branch-targeted `RevisedSimplex` cases against pinned native GLOP.

use glop::parameters::{GlopParameters, InitialBasisHeuristic, PricingRule};
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use glop::variables_info::BasisState;
use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{ColIndex, RowIndex, VariableStatus, VariableStatusRow, VectorIndex};
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
    status: String,
    iterations: u64,
    objective_bits: u64,
    basis: Vec<usize>,
    value_bits: Vec<u64>,
    reduced_bits: Vec<u64>,
    primal_ray_bits: Vec<u64>,
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
            initial_basis: InitialBasisHeuristic::None,
            exploit_singleton_column_in_initial_basis: false,
            use_dual_simplex: !matches!(
                case.mode.as_str(),
                "primal"
                    | "primal_limit"
                    | "primal_no_imprecise"
                    | "primal_time_zero"
                    | "tight_internal_primal"
                    | "relaxed_internal_primal"
                    | "warm_primal"
                    | "warm_bound_change"
                    | "warm_objective_change"
                    | "warm_multiple_bound_changes"
                    | "external_basis"
                    | "warm_added_column"
                    | "starting_values"
                    | "starting_values_push"
                    | "starting_values_push_with_row"
                    | "zero_tolerance"
                    | "tight_tolerance"
                    | "loose_tolerance"
                    | "precision_refactorization"
                    | "zero_tolerance_no_imprecise"
            ),
            perturb_costs_in_dual_simplex: case.mode == "dual_perturbed",
            use_dedicated_dual_feasibility_algorithm: case.mode != "dual_transformed",
            max_number_of_iterations: case.iterations,
            push_to_vertex: case.mode != "starting_values",
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
            feasibility_rule: PricingRule::Dantzig,
            optimization_rule: PricingRule::Dantzig,
            primal_feasibility_tolerance: if matches!(
                case.mode.as_str(),
                "relaxed_internal_primal" | "relaxed_internal_dual"
            ) {
                1e-15
            } else if matches!(
                case.mode.as_str(),
                "tight_internal_dual" | "tight_internal_primal"
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
                "tight_internal_dual" | "tight_internal_primal"
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
            "starting_values" | "starting_values_push" | "starting_values_push_with_row"
        ) {
            simplex.set_starting_variable_values_for_next_solve(
                &lp_data::lp_types::DenseRow::from_vec(vec![0.5]),
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
            "warm_primal"
                | "warm_bound_change"
                | "warm_dual_bound_change"
                | "warm_objective_change"
                | "warm_multiple_bound_changes"
                | "warm_dual_multiple_bound_changes"
                | "warm_added_column"
                | "warm_added_row"
                | "warm_added_row_slack"
        ) {
            simplex
                .solve(&lp, &mut TimeLimit::new(20.0, f64::INFINITY))
                .unwrap_or_else(|error| panic!("{}: first solve: {error}", case.name));
            let state = simplex.state().clone();
            simplex.load_state_for_next_solve(&state);
            if matches!(
                case.mode.as_str(),
                "warm_bound_change" | "warm_dual_bound_change"
            ) {
                lp.set_constraint_bounds(RowIndex::from_usize(0), f64::NEG_INFINITY, 0.5);
            }
            if case.mode == "warm_objective_change" {
                lp.set_objective_coefficient(ColIndex::from_usize(0), 1.0);
            }
            if matches!(
                case.mode.as_str(),
                "warm_multiple_bound_changes" | "warm_dual_multiple_bound_changes"
            ) {
                lp.set_constraint_bounds(RowIndex::from_usize(0), f64::NEG_INFINITY, 0.5);
                lp.set_constraint_bounds(RowIndex::from_usize(1), f64::NEG_INFINITY, 0.5);
            }
            if case.mode == "warm_added_column" {
                let added = lp.create_new_variable();
                lp.set_variable_bounds(added, 0.0, f64::INFINITY);
                lp.set_objective_coefficient(added, -2.0);
                lp.set_coefficient(RowIndex::from_usize(0), added, 1.0);
            }
            if matches!(
                case.mode.as_str(),
                "warm_added_row" | "warm_added_row_slack"
            ) {
                let added = lp.create_new_constraint();
                let lower = if case.mode == "warm_added_row" {
                    2.0
                } else {
                    1.0
                };
                lp.set_constraint_bounds(added, lower, f64::INFINITY);
                lp.set_coefficient(added, ColIndex::from_usize(0), 1.0);
            }
        }
        let deterministic_limit =
            if matches!(case.mode.as_str(), "primal_time_zero" | "dual_time_zero") {
                0.0
            } else {
                f64::INFINITY
            };
        simplex
            .solve(&lp, &mut TimeLimit::new(20.0, deterministic_limit))
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
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
    }
}
