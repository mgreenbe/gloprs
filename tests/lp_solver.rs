use gloprs::lp_solver::LPSolver;
use gloprs::parameters::{GlopParameters, ScalingAlgorithm};
use gloprs::time_limit::TimeLimit;
use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{ConstraintStatus, ProblemStatus, VariableStatus, VectorIndex};

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
    let status = solver.solve_status(&lp);
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

#[test]
fn rejects_solution_states_that_violate_basis_status_contracts() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 4.0);
    lp.set_coefficient(row, x, 1.0);
    lp.clean_up();

    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values[x] = 4.0;
    solution.variable_statuses[x] = VariableStatus::AtUpperBound;
    solution.constraint_statuses[row] = ConstraintStatus::AtUpperBound;

    let mut solver = LPSolver::new();
    assert_eq!(
        solver.load_and_verify_solution(&lp, &solution),
        ProblemStatus::Abnormal
    );

    // `AreWithinAbsoluteTolerance(infinity, infinity, ...)` is false in
    // upstream GLOP; do not accidentally accept it through a `NaN > tol`
    // comparison.
    let mut unbounded_lp = LinearProgram::default();
    let free = unbounded_lp.create_new_variable();
    unbounded_lp.set_variable_bounds(free, f64::NEG_INFINITY, f64::INFINITY);
    unbounded_lp.clean_up();
    let mut infinite_solution =
        ProblemSolution::new(unbounded_lp.num_constraints(), unbounded_lp.num_variables());
    infinite_solution.status = ProblemStatus::Optimal;
    infinite_solution.primal_values[free] = f64::INFINITY;
    infinite_solution.variable_statuses[free] = VariableStatus::AtUpperBound;
    assert_eq!(
        solver.load_and_verify_solution(&unbounded_lp, &infinite_solution),
        ProblemStatus::Abnormal
    );
}

#[test]
fn strong_optimal_guarantee_snaps_primal_and_one_sided_dual_values() {
    let mut lp = LinearProgram::new();
    let col = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(col, 0.0, f64::INFINITY);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 0.0);
    lp.set_coefficient(row, col, 1.0);
    lp.clean_up();

    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values[col] = -1e-12;
    solution.variable_statuses[col] = VariableStatus::Basic;
    solution.dual_values[row] = 1e-12;
    solution.constraint_statuses[row] = ConstraintStatus::AtUpperBound;

    let mut solver = LPSolver::new();
    assert_eq!(
        solver.load_and_verify_solution(&lp, &solution),
        ProblemStatus::Optimal
    );
    assert_eq!(solver.variable_values()[col].to_bits(), 0.0_f64.to_bits());
    assert_eq!(solver.dual_values()[row].to_bits(), 0.0_f64.to_bits());
}

#[test]
fn strong_optimal_guarantee_checks_cost_perturbation() {
    let mut lp = LinearProgram::new();
    let col = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(col, -1.0, 1.0);
    lp.set_constraint_bounds(row, 0.0, 0.0);
    lp.set_coefficient(row, col, 1.0);
    lp.set_objective_coefficient(col, 1.0);
    lp.clean_up();

    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.variable_statuses[col] = VariableStatus::Basic;
    solution.constraint_statuses[row] = ConstraintStatus::FixedValue;

    let mut solver = LPSolver::new();
    assert_eq!(
        solver.load_and_verify_solution(&lp, &solution),
        ProblemStatus::Imprecise
    );
    let parameters = GlopParameters {
        provide_strong_optimal_guarantee: false,
        ..GlopParameters::default()
    };
    solver.set_parameters(&parameters);
    assert_eq!(
        solver.load_and_verify_solution(&lp, &solution),
        ProblemStatus::Optimal
    );
}

#[test]
fn detects_an_optimal_constraint_facet() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, -10.0, 10.0);
    lp.set_constraint_bounds(row, f64::NEG_INFINITY, 0.0);
    lp.set_coefficient(row, x, 1.0);
    lp.clean_up();

    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values[x] = 0.0;
    solution.variable_statuses[x] = VariableStatus::Basic;
    solution.constraint_statuses[row] = ConstraintStatus::AtUpperBound;

    let mut solver = LPSolver::new();
    assert_eq!(
        solver.load_and_verify_solution(&lp, &solution),
        ProblemStatus::Optimal
    );
    assert!(solver.may_have_multiple_optimal_solutions());
}

#[test]
fn rejects_an_optimal_solution_with_a_large_objective_gap() {
    // The status/value pair is structurally consistent, but x = 10 cannot be
    // optimal for min x over [0, 10].  GLOP's primal/dual objective check is
    // what detects this when neither finite bound creates a dual residual.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.clean_up();

    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.status = ProblemStatus::Optimal;
    solution.primal_values[x] = 10.0;
    solution.variable_statuses[x] = VariableStatus::AtUpperBound;

    let mut solver = LPSolver::new();
    assert_eq!(
        solver.load_and_verify_solution(&lp, &solution),
        ProblemStatus::Imprecise
    );
}

#[test]
fn primal_unbounded_ray_is_exposed_only_for_the_current_solve() {
    // min -x over x >= 0 has the improving ray d = 1.
    let mut unbounded = LinearProgram::default();
    let x = unbounded.create_new_variable();
    unbounded.set_variable_bounds(x, 0.0, f64::INFINITY);
    unbounded.set_objective_coefficient(x, -1.0);
    unbounded.clean_up();

    let mut solver = LPSolver::new();
    solver.set_parameters(&GlopParameters {
        use_preprocessing: false,
        use_scaling: false,
        ..GlopParameters::default()
    });
    assert_eq!(
        solver.solve_status(&unbounded),
        ProblemStatus::PrimalUnbounded
    );
    assert_eq!(solver.primal_ray().len().to_usize(), 1);
    assert!((solver.primal_ray()[x] - 1.0).abs() < 1e-15);
    assert!(solver.constraints_dual_ray().is_empty());
    assert!(solver.variable_bounds_dual_ray().is_empty());

    let mut bounded = LinearProgram::default();
    let y = bounded.create_new_variable();
    bounded.set_variable_bounds(y, 0.0, 1.0);
    bounded.set_objective_coefficient(y, -1.0);
    bounded.clean_up();
    assert_eq!(solver.solve_status(&bounded), ProblemStatus::Optimal);
    assert!(solver.primal_ray().is_empty());
    assert!(solver.constraints_dual_ray().is_empty());
    assert!(solver.variable_bounds_dual_ray().is_empty());
}

#[test]
fn dual_unbounded_ray_has_public_certificate_signs() {
    // x is fixed at zero but must satisfy x >= 1.
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 0.0);
    lp.set_constraint_bounds(row, 1.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.clean_up();

    let mut solver = LPSolver::new();
    solver.set_parameters(&GlopParameters {
        use_dual_simplex: true,
        use_preprocessing: false,
        use_scaling: false,
        ..GlopParameters::default()
    });
    assert_eq!(solver.solve_status(&lp), ProblemStatus::DualUnbounded);
    assert_eq!(solver.constraints_dual_ray().len().to_usize(), 1);
    assert_eq!(solver.variable_bounds_dual_ray().len().to_usize(), 1);
    assert!((solver.constraints_dual_ray()[row] - 1.0).abs() < 1e-15);
    assert!((solver.variable_bounds_dual_ray()[x] + 1.0).abs() < 1e-15);
}

#[test]
fn auxiliary_lp_scaling_is_available_through_public_solver() {
    let mut lp = LinearProgram::default();
    let x = lp.create_new_variable();
    lp.set_variable_bounds(x, 0.0, 1.0);
    lp.clean_up();
    let mut solver = LPSolver::new();
    solver.set_parameters(&GlopParameters {
        scaling_method: ScalingAlgorithm::LinearProgram,
        ..GlopParameters::default()
    });
    assert_eq!(solver.solve_status(&lp), ProblemStatus::Optimal);
    assert_eq!(solver.last_error(), None);
}

#[test]
fn empty_and_expired_controls_match_native_public_solver() {
    // Pinned native `phase5_control_reference_adapter`: OPTIMAL, then INIT.
    let mut solver = LPSolver::new();
    solver.set_parameters(&GlopParameters {
        use_preprocessing: false,
        use_scaling: false,
        ..GlopParameters::default()
    });
    assert_eq!(
        solver.solve_status(&LinearProgram::new()),
        ProblemStatus::Optimal
    );

    let mut lp = LinearProgram::new();
    let x = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_constraint_bounds(row, 5.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.clean_up();
    let mut limit = TimeLimit::new(f64::INFINITY, 0.0);
    assert_eq!(
        solver.solve_with_time_limit(&lp, &mut limit),
        ProblemStatus::Init
    );
}
