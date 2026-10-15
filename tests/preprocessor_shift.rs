#![allow(clippy::float_cmp)] // Exact bound snapping and algebraic shifts are intentional.

use gloprs::lp_solver::LPSolver;
use gloprs::parameters::GlopParameters;
use gloprs::preprocessor::{ScalingPreprocessor, ShiftVariableBoundsPreprocessor};
use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{ProblemStatus, VariableStatus};

#[test]
fn shifts_both_signs_and_recovers_by_variable_status() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let y = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 2.0, 10.0);
    lp.set_variable_bounds(y, -10.0, -3.0);
    lp.set_constraint_bounds(row, 20.0, 20.0);
    lp.set_coefficient(row, x, 2.0);
    lp.set_coefficient(row, y, -3.0);
    lp.set_objective_coefficient(x, 5.0);
    lp.set_objective_coefficient(y, 7.0);
    lp.clean_up();

    let mut transform = ShiftVariableBoundsPreprocessor::new();
    assert_eq!(transform.run(&mut lp), Ok(true));
    assert_eq!(transform.offsets()[x], 2.0);
    assert_eq!(transform.offsets()[y], -3.0);
    assert_eq!(lp.variable_lower_bounds()[x], 0.0);
    assert_eq!(lp.variable_upper_bounds()[y], 0.0);
    assert_eq!(lp.constraint_lower_bounds()[row], 7.0);
    assert_eq!(lp.constraint_upper_bounds()[row], 7.0);
    assert_eq!(lp.objective_offset(), -11.0);

    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.primal_values[x] = 0.0;
    solution.variable_statuses[x] = VariableStatus::AtLowerBound;
    solution.primal_values[y] = -2.0;
    solution.variable_statuses[y] = VariableStatus::Basic;
    transform.recover_solution(&mut solution);
    assert_eq!(solution.primal_values[x], 2.0);
    assert_eq!(solution.primal_values[y], -5.0);

    solution.variable_statuses[y] = VariableStatus::AtUpperBound;
    transform.recover_solution(&mut solution);
    assert_eq!(solution.primal_values[y], -3.0);
}

#[test]
fn no_shift_when_every_domain_contains_zero() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    lp.set_variable_bounds(x, -2.0, 3.0);
    lp.clean_up();
    let mut transform = ShiftVariableBoundsPreprocessor::new();
    assert_eq!(transform.run(&mut lp), Ok(false));
    assert_eq!(lp.variable_lower_bounds()[x], -2.0);
}

#[test]
fn overflowing_row_shift_reports_invalid_problem() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 1e200, 2e200);
    lp.set_constraint_bounds(row, 0.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1e200);
    lp.clean_up();
    let mut transform = ShiftVariableBoundsPreprocessor::new();
    assert_eq!(transform.run(&mut lp), Err(ProblemStatus::InvalidProblem));
}

#[test]
fn shifted_solve_recovers_original_optimum_and_objective() {
    let mut original = LinearProgram::default();
    let x = original.create_new_variable();
    let row = original.create_new_constraint();
    original.set_variable_bounds(x, 2.0, 10.0);
    original.set_constraint_bounds(row, 5.0, f64::INFINITY);
    original.set_coefficient(row, x, 1.0);
    original.set_objective_coefficient(x, 1.0);
    original.clean_up();

    let mut shifted = original.clone();
    let mut transform = ShiftVariableBoundsPreprocessor::new();
    assert_eq!(transform.run(&mut shifted), Ok(true));
    let mut solver = LPSolver::new();
    assert_eq!(solver.solve_status(&shifted), ProblemStatus::Optimal);
    let mut solution = ProblemSolution::new(shifted.num_constraints(), shifted.num_variables());
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

    let mut verifier = LPSolver::new();
    assert_eq!(
        verifier.load_and_verify_solution(&original, &solution),
        ProblemStatus::Optimal
    );
    assert!((solution.primal_values[x] - 5.0).abs() < 1e-8);
    assert!((verifier.objective_value() - 5.0).abs() < 1e-8);
}

#[test]
fn shift_then_scale_recovers_in_reverse_order() {
    let mut original = LinearProgram::default();
    let x = original.create_new_variable();
    let row = original.create_new_constraint();
    original.set_variable_bounds(x, 2.0, 10.0);
    original.set_constraint_bounds(row, 5_000.0, f64::INFINITY);
    original.set_coefficient(row, x, 1_000.0);
    original.set_objective_coefficient(x, 5.0);
    original.clean_up();

    let mut transformed = original.clone();
    let mut shift = ShiftVariableBoundsPreprocessor::new();
    let mut scaling = ScalingPreprocessor::new();
    assert_eq!(shift.run(&mut transformed), Ok(true));
    assert!(scaling.run(&mut transformed, &GlopParameters::default()));
    let mut solver = LPSolver::new();
    assert_eq!(solver.solve_status(&transformed), ProblemStatus::Optimal);
    let mut solution =
        ProblemSolution::new(transformed.num_constraints(), transformed.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values.clone_from(solver.variable_values());
    solution.dual_values.clone_from(solver.dual_values());
    solution
        .variable_statuses
        .clone_from(solver.variable_statuses());
    solution
        .constraint_statuses
        .clone_from(solver.constraint_statuses());
    scaling.recover_solution(&mut solution);
    shift.recover_solution(&mut solution);

    let mut verifier = LPSolver::new();
    assert_eq!(
        verifier.load_and_verify_solution(&original, &solution),
        ProblemStatus::Optimal
    );
    assert!((solution.primal_values[x] - 5.0).abs() < 1e-8);
    assert!((verifier.objective_value() - 25.0).abs() < 1e-8);
}
