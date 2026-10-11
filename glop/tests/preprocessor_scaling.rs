#![allow(clippy::float_cmp)] // Exact original-bound snapping and identity are intentional.

use glop::lp_solver::LPSolver;
use glop::parameters::{GlopParameters, ScalingAlgorithm};
use glop::preprocessor::ScalingPreprocessor;
use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{ProblemStatus, VariableStatus};

#[test]
fn scaling_postsolve_restores_values_duals_and_exact_bounds() {
    let mut original = LinearProgram::default();
    let x = original.create_new_variable();
    let y = original.create_new_variable();
    let row = original.create_new_constraint();
    original.set_variable_bounds(x, 0.0, 10.0);
    original.set_variable_bounds(y, 1.1, 7.3);
    original.set_constraint_bounds(row, 2_000.0, 2_000.0);
    original.set_coefficient(row, x, 1_000.0);
    original.set_coefficient(row, y, 0.01);
    original.set_objective_coefficient(x, 5.0);
    original.set_objective_coefficient(y, 0.2);
    original.clean_up();

    let mut scaled = original.clone();
    let mut transform = ScalingPreprocessor::new();
    assert!(transform.run(&mut scaled, &GlopParameters::default()));
    let mut solution = ProblemSolution::new(scaled.num_constraints(), scaled.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values[x] = 2.0 * scaled.variable_upper_bounds()[x] / 10.0;
    solution.primal_values[y] = scaled.variable_lower_bounds()[y];
    solution.variable_statuses[x] = VariableStatus::Basic;
    solution.variable_statuses[y] = VariableStatus::AtLowerBound;
    solution.dual_values[row] = 3.0;
    transform.recover_solution(&mut solution);

    assert!((solution.primal_values[x] - 2.0).abs() < 1e-12);
    assert_eq!(solution.primal_values[y], 1.1);
    assert!(solution.dual_values[row].is_finite());
    assert_ne!(solution.dual_values[row], 3.0);
}

#[test]
fn disabled_scaling_is_identity_and_auxiliary_lp_method_is_available() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    lp.set_variable_bounds(x, 0.0, 2.0);
    lp.clean_up();
    let original = lp.clone();
    let mut transform = ScalingPreprocessor::new();
    let mut parameters = GlopParameters {
        use_scaling: false,
        ..GlopParameters::default()
    };
    assert!(!transform.run(&mut lp, &parameters));
    assert_eq!(
        lp.variable_upper_bounds()[x],
        original.variable_upper_bounds()[x]
    );
    parameters.use_scaling = true;
    parameters.scaling_method = ScalingAlgorithm::LinearProgram;
    assert!(transform.run(&mut lp, &parameters));
    assert!(lp.variable_upper_bounds()[x].is_finite());
}

#[test]
fn scaled_solve_recovers_an_original_coordinate_optimum() {
    let mut original = LinearProgram::default();
    let x = original.create_new_variable();
    let row = original.create_new_constraint();
    original.set_variable_bounds(x, 0.0, 10.0);
    original.set_constraint_bounds(row, 2_000.0, f64::INFINITY);
    original.set_coefficient(row, x, 1_000.0);
    original.set_objective_coefficient(x, 5.0);
    original.clean_up();

    let mut scaled = original.clone();
    let mut transform = ScalingPreprocessor::new();
    assert!(transform.run(&mut scaled, &GlopParameters::default()));
    let mut solver = LPSolver::new();
    assert_eq!(solver.solve(&scaled), ProblemStatus::Optimal);
    let mut solution = ProblemSolution::new(scaled.num_constraints(), scaled.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values.clone_from(solver.variable_values());
    solution.dual_values.clone_from(solver.dual_values());
    solution
        .variable_statuses
        .clone_from(solver.variable_statuses());
    solution
        .constraint_statuses
        .clone_from(solver.constraint_statuses());
    transform.recover_solution(&mut solution);

    assert!((solution.primal_values[x] - 2.0).abs() < 1e-8);
    assert!((solution.dual_values[row] - 0.005).abs() < 1e-8);
    let mut verifier = LPSolver::new();
    assert_eq!(
        verifier.load_and_verify_solution(&original, &solution),
        ProblemStatus::Optimal
    );
    assert!((verifier.objective_value() - 10.0).abs() < 1e-8);
}
