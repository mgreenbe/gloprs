//! Time the public LP-solver pipeline with its default parameters.

use std::env;
use std::time::Instant;

use glop::lp_solver::LPSolver;
use lp_data::mps_reader::parse_mps_file;

fn main() {
    let path = env::args_os().nth(1).expect("missing MPS path");
    let model = parse_mps_file(path).expect("failed to parse MPS");
    let mut solver = LPSolver::new();
    let start = Instant::now();
    let status = solver.solve(&model);
    println!(
        "{:.17} {} {} {:.17}",
        start.elapsed().as_secs_f64(),
        status,
        solver.number_of_simplex_iterations(),
        solver.objective_value()
    );
}
