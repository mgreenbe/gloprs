//! Diagnostic default-presolve comparison with the pinned GLOP public solver.

use std::env;

use glop::lp_solver::LPSolver;
use glop::preprocessor::MainLpPreprocessor;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;

#[allow(clippy::too_many_lines)]
fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: full_presolve_netlib MODEL.mps");
    let lp = parse_mps_file(path).expect("valid MPS model");
    let mut presolved = lp.clone();
    let parameters = glop::parameters::GlopParameters::default();
    presolved.remove_near_zero_entries(parameters.drop_magnitude);
    let mut preprocessor = MainLpPreprocessor::new();
    let mut presolve_limit = TimeLimit::new(20.0, f64::INFINITY);
    preprocessor.run(&mut presolved, &parameters, &mut presolve_limit);
    eprintln!(
        "presolved {} {} {} {}",
        preprocessor.status(),
        presolved.num_constraints().value(),
        presolved.num_variables().value(),
        presolved.num_entries().value()
    );
    for column in 0..presolved.num_variables().to_usize() {
        let col = ColIndex::from_usize(column);
        eprintln!(
            "col {column} {} {} {}",
            presolved.objective_coefficients()[col].to_bits(),
            presolved.variable_lower_bounds()[col].to_bits(),
            presolved.variable_upper_bounds()[col].to_bits()
        );
        for entry in presolved.sparse_column(col) {
            eprintln!(
                "entry {column} {} {}",
                entry.index().value(),
                entry.coefficient().to_bits()
            );
        }
    }
    for row in 0..presolved.num_constraints().to_usize() {
        let r = RowIndex::from_usize(row);
        eprintln!(
            "row {row} {} {}",
            presolved.constraint_lower_bounds()[r].to_bits(),
            presolved.constraint_upper_bounds()[r].to_bits()
        );
    }
    eprintln!("offset {}", presolved.objective_offset().to_bits());
    if let Some(cap) = env::args().nth(2) {
        let mut direct = RevisedSimplex::new();
        let mut direct_parameters = parameters.clone();
        direct_parameters.max_number_of_iterations = cap.parse().unwrap();
        direct.set_parameters(&direct_parameters);
        let mut direct_limit = TimeLimit::new(20.0, f64::INFINITY);
        direct.solve(&presolved, &mut direct_limit).unwrap();
        println!(
            "direct {} {}",
            direct.problem_status(),
            direct.number_of_iterations()
        );
        print!("basis");
        for row in 0..presolved.num_constraints().to_usize() {
            print!(" {}", direct.basis(RowIndex::from_usize(row)).value());
        }
        print!("\nreduced");
        for col in 0..presolved.num_variables().to_usize() + presolved.num_constraints().to_usize()
        {
            print!(
                " {}",
                direct.reduced_cost(ColIndex::from_usize(col)).to_bits()
            );
        }
        print!("\ncached_reduced");
        for value in direct.diagnostic_cached_reduced_costs() {
            print!(" {}", value.to_bits());
        }
        print!("\nvalues");
        for col in 0..presolved.num_variables().to_usize() + presolved.num_constraints().to_usize()
        {
            print!(
                " {}",
                direct.variable_value(ColIndex::from_usize(col)).to_bits()
            );
        }
        print!("\nstatuses");
        for col in 0..presolved.num_variables().to_usize() + presolved.num_constraints().to_usize()
        {
            print!(" {}", direct.variable_status(ColIndex::from_usize(col)));
        }
        print!("\nnorms");
        for norm in direct.diagnostic_primal_pricing_norms().unwrap() {
            print!(" {}", norm.to_bits());
        }
        let (inverse, inverse_order) = direct.diagnostic_primal_direction_left_inverse();
        print!("\ninverse");
        for value in inverse {
            print!(" {}", value.to_bits());
        }
        print!("\ninverse_order");
        for col in inverse_order {
            print!(" {col}");
        }
        print!("\nupdate_row");
        for (col, value) in direct.diagnostic_update_row() {
            print!(" {col}:{}", value.to_bits());
        }
        print!("\nsolve20");
        for value in direct
            .diagnostic_basis_solve_for_column(&presolved, ColIndex::from_usize(20))
            .unwrap()
        {
            print!(" {}", value.to_bits());
        }
        if presolved.num_variables().to_usize() > 63 {
            print!("\nsolve63");
            for value in direct
                .diagnostic_basis_solve_for_column(&presolved, ColIndex::from_usize(63))
                .unwrap()
            {
                print!(" {}", value.to_bits());
            }
            print!("\nsolve63_order");
            for row in direct
                .diagnostic_sparse_basis_solve_order(ColIndex::from_usize(63))
                .unwrap()
            {
                print!(" {row}");
            }
        }
        if presolved.num_variables().to_usize() + presolved.num_constraints().to_usize() > 603 {
            print!("\nsolve603");
            for value in direct
                .diagnostic_basis_solve_for_column(&presolved, ColIndex::from_usize(603))
                .unwrap()
            {
                print!(" {}", value.to_bits());
            }
            print!("\nsolve603_order");
            for row in direct
                .diagnostic_sparse_basis_solve_order(ColIndex::from_usize(603))
                .unwrap()
            {
                print!(" {row}");
            }
        }
        let (norm, nonzeros) = direct
            .diagnostic_primal_column_norm_and_order(&presolved, ColIndex::from_usize(20))
            .unwrap();
        print!("\nrenorm20 {}\nnonzeros20", norm.to_bits());
        for row in nonzeros {
            print!(" {row}");
        }
        print!("\nrenorms");
        for norm in direct.diagnostic_recompute_primal_pricing_norms().unwrap() {
            print!(" {}", norm.to_bits());
        }
        println!();
        return;
    }
    let mut solver = LPSolver::new();
    let mut limit = TimeLimit::new(20.0, f64::INFINITY);
    let status = solver.solve_with_time_limit(&lp, &mut limit);
    println!(
        "{status} {} {:.17} {:.17} {:.17}",
        solver.number_of_simplex_iterations(),
        solver.objective_value(),
        solver.maximum_primal_infeasibility(),
        solver.maximum_dual_infeasibility()
    );
}
