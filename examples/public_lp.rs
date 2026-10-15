//! Construct an LP, solve it, and reuse its basis through the public API.
//! Run: `cargo run -p gloprs --example public_lp`

use gloprs::{LPSolver, LinearProgram};

fn main() {
    // min x + 2y subject to x + y >= 5 and 0 <= x,y <= 10.
    let mut lp = LinearProgram::new();
    let x = lp.create_new_variable();
    let y = lp.create_new_variable();
    let row = lp.create_new_constraint();
    lp.set_variable_bounds(x, 0.0, 10.0);
    lp.set_variable_bounds(y, 0.0, 10.0);
    lp.set_constraint_bounds(row, 5.0, f64::INFINITY);
    lp.set_coefficient(row, x, 1.0);
    lp.set_coefficient(row, y, 1.0);
    lp.set_objective_coefficient(x, 1.0);
    lp.set_objective_coefficient(y, 2.0);
    lp.clean_up();

    let mut solver = LPSolver::new();
    solver.parameters_mut().use_preprocessing = false;
    solver.parameters_mut().use_scaling = false;
    let result = solver.solve(&lp);
    println!(
        "{} objective={} x={} y={}",
        result.status, result.objective_value, result.variable_values[x], result.variable_values[y]
    );
    let basis = solver.export_basis();
    let mut warm = LPSolver::new();
    warm.parameters_mut().use_preprocessing = false;
    warm.parameters_mut().use_scaling = false;
    warm.import_basis(&basis);
    let resumed = warm.solve(&lp);
    println!(
        "warm {} iterations={}",
        resumed.status, resumed.simplex_iterations
    );
}
