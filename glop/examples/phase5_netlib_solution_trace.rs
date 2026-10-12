//! Compact default-pipeline public-solver result trace for Netlib.

use std::env;
use std::time::Instant;

use glop::lp_solver::LPSolver;
use glop::parameters::GlopParameters;
use glop::time_limit::TimeLimit;
use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;

fn independent_primal_violation(lp: &LinearProgram, solver: &LPSolver) -> f64 {
    let mut activities = vec![0.0; lp.num_constraints().to_usize()];
    let mut scales = vec![1.0; activities.len()];
    let mut worst: f64 = 0.0;
    for column in 0..lp.num_variables().to_usize() {
        let col = ColIndex::from_usize(column);
        let value = solver.variable_values()[col];
        if !value.is_finite() {
            return f64::INFINITY;
        }
        worst = worst.max(
            (lp.variable_lower_bounds()[col] - value)
                .max(value - lp.variable_upper_bounds()[col])
                .max(0.0)
                / (1.0 + value.abs()),
        );
        for entry in lp.sparse_column(col) {
            let row = entry.index().to_usize();
            let term = entry.coefficient() * value;
            activities[row] += term;
            scales[row] += term.abs();
        }
    }
    for row in 0..activities.len() {
        let index = RowIndex::from_usize(row);
        let value = activities[row];
        if !value.is_finite() {
            return f64::INFINITY;
        }
        worst = worst.max(
            (lp.constraint_lower_bounds()[index] - value)
                .max(value - lp.constraint_upper_bounds()[index])
                .max(0.0)
                / scales[row],
        );
    }
    worst
}

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
    let mode = env::args().nth(2);
    let raw = matches!(mode.as_deref(), Some("raw"));
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
    let wall_seconds = env::args().nth(3).map_or(20.0, |value| {
        value.parse::<f64>().expect("wall seconds must be a number")
    });
    assert!(wall_seconds > 0.0 && wall_seconds.is_finite());
    let mut limit = TimeLimit::new(wall_seconds, f64::INFINITY);
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
    if matches!(mode.as_deref(), Some("phase7")) {
        println!("deterministic {}", solver.deterministic_time().to_bits());
        println!(
            "independent_primal_violation {:.17e}",
            independent_primal_violation(&lp, &solver)
        );
        println!(
            "refactorizations {}",
            solver.revised_simplex().num_basis_refactorizations()
        );
    }
}
