//! Load an MPS model and solve it with the default GLOP parameters.
//! Run: `cargo run -p gloprs-glop --example public_mps -- model.mps`

use std::env;

use glop::lp_solver::LPSolver;
use lp_data::mps_reader::parse_mps_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: public_mps MODEL.mps")?;
    let model = parse_mps_file(path)?;
    let mut solver = LPSolver::new();
    let result = solver.solve_result(&model);
    println!(
        "{} objective={} iterations={}",
        result.status, result.objective_value, result.simplex_iterations
    );
    Ok(())
}
