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
