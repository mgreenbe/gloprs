use glop::parameters::{GlopParameters, PricingRule};
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{ProblemStatus, VectorIndex};

#[test]
fn solves_a_bounded_problem_through_phase_one_and_two() {
    // min -x - y; x + 2y <= 4; 0 <= x,y <= 10.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let y = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_variable_bounds(y, 0.0, 10.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 4.0);
    lp.set_coefficient(row, x, 1.0);
    lp.set_coefficient(row, y, 2.0);
    lp.set_objective_coefficient(x, -1.0);
    lp.set_objective_coefficient(y, -1.0);
    lp.clean_up();

    let mut simplex = RevisedSimplex::new();
    simplex.set_trace_enabled(true);
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();
    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
    assert!(
        (simplex.objective_value() + 4.0).abs() < 1e-9,
        "objective={}, x={}, y={}, trace={:?}",
        simplex.objective_value(),
        simplex.variable_value(x),
        simplex.variable_value(y),
        simplex.trace()
    );
    assert!((simplex.variable_value(x) - 4.0).abs() < 1e-9);
    assert!(simplex.variable_value(y).abs() < 1e-9);
    assert!(!simplex.trace().is_empty());
    assert_eq!(simplex.problem_num_cols().to_usize(), 2);
}

#[test]
fn all_primal_pricing_rules_drive_the_revised_simplex_loop() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let y = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_variable_bounds(y, 0.0, 10.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 4.0);
    lp.set_coefficient(row, x, 1.0);
    lp.set_coefficient(row, y, 2.0);
    lp.set_objective_coefficient(x, -1.0);
    lp.set_objective_coefficient(y, -1.0);
    lp.clean_up();

    for rule in [
        PricingRule::Dantzig,
        PricingRule::SteepestEdge,
        PricingRule::Devex,
    ] {
        let parameters = GlopParameters {
            feasibility_rule: rule,
            optimization_rule: rule,
            ..GlopParameters::default()
        };
        let mut simplex = RevisedSimplex::new();
        simplex.set_parameters(&parameters);
        simplex
            .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
            .unwrap();
        assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
        assert!((simplex.objective_value() + 4.0).abs() < 1e-9);
    }
}

#[test]
fn maximization_and_primal_objective_limit_use_external_objective_coordinates() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 4.0);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.set_objective_offset(3.0);
    lp.set_objective_scaling_factor(2.0);
    lp.set_maximization_problem(true);
    lp.clean_up();

    let parameters = GlopParameters {
        objective_upper_limit: 10.0,
        ..GlopParameters::default()
    };
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&parameters);
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::PrimalFeasible);
    assert!(simplex.objective_limit_reached());
    assert!((simplex.objective_value() - 14.0).abs() < 1e-9);
}

#[test]
fn dual_phase_two_repairs_a_dual_feasible_primal_infeasible_basis() {
    // min x; x >= 1; x >= 0.  The all-slack basis is dual feasible, but its
    // slack value violates the row bound, so dual Phase II performs one pivot.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, 1.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.clean_up();

    let parameters = GlopParameters {
        use_dual_simplex: true,
        ..GlopParameters::default()
    };
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&parameters);
    simplex.set_trace_enabled(true);
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
    assert!((simplex.variable_value(x) - 1.0).abs() < 1e-9);
    assert!((simplex.objective_value() - 1.0).abs() < 1e-9);
    assert_eq!(simplex.number_of_iterations(), 1);
    assert_eq!(simplex.trace().len(), 1);
}

#[test]
fn dual_cost_perturbation_is_removed_before_accepting_optimality() {
    // A nonbasic column's reduced cost must also return to its original value.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let y = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_variable_bounds(y, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, 1.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.set_objective_coefficient(y, 3.0);
    lp.clean_up();

    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_dual_simplex: true,
        perturb_costs_in_dual_simplex: true,
        ..GlopParameters::default()
    });
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
    assert!((simplex.variable_value(x) - 1.0).abs() < 1e-12);
    assert!((simplex.objective_value() - 1.0).abs() < 1e-12);
    assert!((simplex.reduced_cost(y) - 3.0).abs() < 1e-12);
    assert_eq!(simplex.number_of_iterations(), 1);
}

#[test]
fn dual_reports_optimality_at_the_exact_iteration_limit() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, 1.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.clean_up();

    let parameters = GlopParameters {
        use_dual_simplex: true,
        max_number_of_iterations: 1,
        ..GlopParameters::default()
    };
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&parameters);
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.number_of_iterations(), 1);
    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
}

#[test]
fn dual_phase_two_handles_successive_pivots() {
    // min x + y; x + y >= 2; x + 2y >= 3; x,y >= 0.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let y = lp.create_new_variable();
    let first = lp.create_new_constraint();
    let second = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_variable_bounds(y, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(first, 2.0, f64::INFINITY);
    lp.set_constraint_bounds(second, 3.0, f64::INFINITY);
    lp.set_coefficient(first, x, 1.0);
    lp.set_coefficient(first, y, 1.0);
    lp.set_coefficient(second, x, 1.0);
    lp.set_coefficient(second, y, 2.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.set_objective_coefficient(y, 1.0);
    lp.clean_up();

    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_dual_simplex: true,
        ..GlopParameters::default()
    });
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
    assert!((simplex.objective_value() - 2.0).abs() < 1e-9);
    assert!(simplex.maximum_equation_residual() < 1e-9);
    assert!(simplex.number_of_iterations() >= 2);
}

#[test]
fn dual_objective_limit_uses_external_objective_coordinates() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, 1.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.clean_up();

    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_dual_simplex: true,
        objective_upper_limit: 0.5,
        ..GlopParameters::default()
    });
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::DualFeasible);
    assert!(simplex.objective_limit_reached());
    assert!((simplex.objective_value() - 1.0).abs() < 1e-9);
}

#[test]
fn dedicated_dual_phase_one_establishes_dual_feasibility() {
    // min -x; x <= 1; x >= 0.  The all-slack basis has reduced cost -1 for x,
    // so the default dedicated dual Phase I must pivot before Phase II.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 1.0);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, -1.0);
    lp.clean_up();

    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_dual_simplex: true,
        ..GlopParameters::default()
    });
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
    assert!((simplex.variable_value(x) - 1.0).abs() < 1e-9);
    assert!((simplex.objective_value() + 1.0).abs() < 1e-9);
}

#[test]
fn transformed_dual_phase_one_establishes_dual_feasibility() {
    // The all-slack basis has reduced cost -1 for x, so transformed dual
    // Phase I must repair dual feasibility before optimizing the original LP.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 1.0);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, -1.0);
    lp.clean_up();

    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_dual_simplex: true,
        use_dedicated_dual_feasibility_algorithm: false,
        ..GlopParameters::default()
    });
    simplex
        .solve(&lp, &mut TimeLimit::new(f64::INFINITY, f64::INFINITY))
        .unwrap();

    assert_eq!(simplex.problem_status(), ProblemStatus::Optimal);
    assert!((simplex.variable_value(x) - 1.0).abs() < 1e-9);
    assert!((simplex.objective_value() + 1.0).abs() < 1e-9);
}
