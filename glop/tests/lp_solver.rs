use glop::lp_solver::LPSolver;
use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{ProblemStatus, VectorIndex};

#[test]
fn maximization_solution_uses_original_objective_coordinates() {
    // max x; x <= 4; 0 <= x <= 10.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 4.0);
    lp.set_coefficient(row, x, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.set_maximization_problem(true);
    lp.clean_up();

    let mut solver = LPSolver::new();
    let status = solver.solve(&lp);
    assert_eq!(
        status,
        ProblemStatus::Optimal,
        "primal={}, dual={}, rc={:?}, duals={:?}",
        solver.maximum_primal_infeasibility(),
        solver.maximum_dual_infeasibility(),
        solver.reduced_costs().as_slice(),
        solver.dual_values().as_slice(),
    );
    assert!((solver.objective_value() - 4.0).abs() < 1e-9);
    assert!((solver.variable_values()[x] - 4.0).abs() < 1e-9);
    assert!(solver.reduced_costs()[x].abs() < 1e-9);
    assert_eq!(solver.variable_values().len().to_usize(), 1);
}
