//! Compact default-pipeline public-solver result trace for Netlib.

use std::env;
use std::time::Instant;

use glop::lp_solver::LPSolver;
use glop::parameters::GlopParameters;
use glop::time_limit::TimeLimit;
use lp_data::mps_reader::parse_mps_file;

fn hash_values(values: impl IntoIterator<Item = f64>) -> u64 {
    values
        .into_iter()
        .fold(14_695_981_039_346_656_037_u64, |hash, value| {
            (hash ^ value.to_bits()).wrapping_mul(1_099_511_628_211)
        })
}

fn hash_statuses(values: impl IntoIterator<Item = String>) -> u64 {
    let mut hash = 14_695_981_039_346_656_037_u64;
    for name in values {
        for byte in name.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
        }
        hash = (hash ^ 255).wrapping_mul(1_099_511_628_211);
    }
    hash
}

fn print_values(label: &str, values: impl IntoIterator<Item = f64>) {
    print!("{label}");
    for value in values {
        print!(" {}", value.to_bits());
    }
    println!();
}

fn print_statuses(label: &str, values: impl IntoIterator<Item = String>) {
    print!("{label}");
    for value in values {
        print!(" {value}");
    }
    println!();
}

fn main() {
    let path = env::args().nth(1).expect("MPS path required");
    let raw = matches!(env::args().nth(2).as_deref(), Some("raw"));
    let lp = parse_mps_file(path).expect("valid MPS");
    let mut solver = LPSolver::new();
    if raw {
        let parameters = GlopParameters {
            use_preprocessing: false,
            use_scaling: false,
            ..GlopParameters::default()
        };
        solver.set_parameters(&parameters);
    }
    let mut limit = TimeLimit::new(20.0, f64::INFINITY);
    let start = Instant::now();
    let status = solver.solve_with_time_limit(&lp, &mut limit);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{} {} {} {} {}",
        status,
        solver.number_of_simplex_iterations(),
        solver.objective_value().to_bits(),
        solver.maximum_primal_infeasibility().to_bits(),
        solver.maximum_dual_infeasibility().to_bits(),
    );
    println!(
        "{} {} {} {} {} {} {} {}",
        lp.num_variables().value(),
        lp.num_constraints().value(),
        hash_values(solver.variable_values().iter().copied()),
        hash_values(solver.reduced_costs().iter().copied()),
        hash_values(solver.dual_values().iter().copied()),
        hash_values(solver.constraint_activities().iter().copied()),
        hash_statuses(solver.variable_statuses().iter().map(ToString::to_string)),
        hash_statuses(solver.constraint_statuses().iter().map(ToString::to_string)),
    );
    print_values("primal", solver.variable_values().iter().copied());
    print_values("reduced", solver.reduced_costs().iter().copied());
    print_values("dual", solver.dual_values().iter().copied());
    print_values("activity", solver.constraint_activities().iter().copied());
    print_statuses(
        "variable_status",
        solver.variable_statuses().iter().map(ToString::to_string),
    );
    print_statuses(
        "constraint_status",
        solver.constraint_statuses().iter().map(ToString::to_string),
    );
    println!("elapsed {elapsed:.17}");
}
