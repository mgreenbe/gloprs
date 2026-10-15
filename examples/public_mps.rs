//! Load an MPS model and solve it with the default GLOP parameters.
//! Run: `cargo run -p gloprs --example public_mps -- model.mps`

use std::env;

use gloprs::{LPSolver, parse_mps_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: public_mps MODEL.mps")?;
    let model = parse_mps_file(path)?;
    let mut solver = LPSolver::new();
    let result = solver.solve(&model);
    println!(
        "{} objective={} iterations={}",
        result.status, result.objective_value, result.simplex_iterations
    );
    Ok(())
}
