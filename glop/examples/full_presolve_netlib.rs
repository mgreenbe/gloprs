//! Diagnostic default-presolve comparison with the pinned GLOP public solver.

use std::env;
use std::time::Instant;

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
    let parse_start = Instant::now();
    let lp = parse_mps_file(path).expect("valid MPS model");
    let parse_seconds = parse_start.elapsed().as_secs_f64();
    if matches!(env::args().nth(2).as_deref(), Some("parse_timing")) {
        println!(
            "parse_seconds {parse_seconds:.17} rows {} columns {} nonzeros {}",
            lp.num_constraints().value(),
            lp.num_variables().value(),
            lp.num_entries().value()
        );
        return;
    }
    if matches!(env::args().nth(2).as_deref(), Some("copy_timing")) {
        let repetitions: u32 = env::args()
            .nth(3)
            .map_or(1, |value| value.parse().expect("positive repetitions"));
        assert!(repetitions > 0);
        let mut copy = lp_data::lp_data::LinearProgram::new();
        let start = Instant::now();
        let mut nonzeros = 0;
        for _ in 0..repetitions {
            copy.populate_from_linear_program(&lp);
            nonzeros = std::hint::black_box(copy.num_entries().value());
        }
        println!(
            "copy_seconds {:.17} nonzeros {nonzeros}",
            start.elapsed().as_secs_f64() / f64::from(repetitions)
        );
        return;
    }
    if matches!(env::args().nth(2).as_deref(), Some("matrix_copy_timing")) {
        let repetitions: u32 = env::args().nth(3).unwrap().parse().unwrap();
        let mut copy = lp_data::sparse::SparseMatrix::default();
        let start = Instant::now();
        for _ in 0..repetitions {
            copy.populate_from_sparse(lp.matrix());
            std::hint::black_box(copy.num_entries());
        }
        println!(
            "matrix_copy_seconds {:.17}",
            start.elapsed().as_secs_f64() / f64::from(repetitions)
        );
        return;
    }
    if matches!(env::args().nth(2).as_deref(), Some("presolve_timing")) {
        let repetitions: u32 = env::args()
            .nth(3)
            .map_or(1, |value| value.parse().expect("positive repetitions"));
        assert!(repetitions > 0);
        let parameters = glop::parameters::GlopParameters::default();
        let start = Instant::now();
        let mut result = (String::new(), 0, 0, 0);
        for _ in 0..repetitions {
            let mut presolved = lp_data::lp_data::LinearProgram::new();
            presolved.populate_from_linear_program(&lp);
            presolved.remove_near_zero_entries(parameters.drop_magnitude);
            let mut preprocessor = MainLpPreprocessor::new();
            let mut limit = TimeLimit::new(20.0, f64::INFINITY);
            preprocessor.run(&mut presolved, &parameters, &mut limit);
            result = (
                preprocessor.status().to_string(),
                presolved.num_constraints().value(),
                presolved.num_variables().value(),
                presolved.num_entries().value(),
            );
        }
        let seconds = start.elapsed().as_secs_f64() / f64::from(repetitions);
        println!(
            "presolve_seconds {seconds:.17} status {} rows {} columns {} nonzeros {}",
            result.0, result.1, result.2, result.3
        );
        return;
    }
    let mut presolved = lp_data::lp_data::LinearProgram::new();
    presolved.populate_from_linear_program(&lp);
    let parameters = glop::parameters::GlopParameters::default();
    presolved.remove_near_zero_entries(parameters.drop_magnitude);
    let mut preprocessor = MainLpPreprocessor::new();
    let mut presolve_limit = TimeLimit::new(20.0, f64::INFINITY);
    preprocessor.run(&mut presolved, &parameters, &mut presolve_limit);
    if matches!(
        env::args().nth(2).as_deref(),
        Some("presolved_solve_timing")
    ) {
        let repetitions: u32 = env::args()
            .nth(3)
            .map_or(1, |value| value.parse().expect("positive repetitions"));
        assert!(repetitions > 0);
        assert_eq!(preprocessor.status().to_string(), "INIT");
        let wall_seconds = env::args().nth(4).map_or(20.0, |value| {
            value.parse::<f64>().expect("numeric solve wall limit")
        });
        assert!(wall_seconds > 0.0 && wall_seconds.is_finite());
        let iteration_cap = env::args()
            .nth(5)
            .map(|value| value.parse::<i64>().expect("nonnegative iteration cap"));
        let mut solve_parameters = parameters.clone();
        if let Some(cap) = iteration_cap {
            assert!(cap >= 0);
            solve_parameters.max_number_of_iterations = cap;
        }
        let start = Instant::now();
        let mut result = (String::new(), 0, 0);
        for _ in 0..repetitions {
            let mut simplex = RevisedSimplex::new();
            simplex.set_parameters(&solve_parameters);
            let mut limit = TimeLimit::new(wall_seconds, f64::INFINITY);
            simplex.solve(&presolved, &mut limit).expect("direct solve");
            result = (
                simplex.problem_status().to_string(),
                simplex.number_of_iterations(),
                simplex.objective_value().to_bits(),
            );
        }
        let seconds = start.elapsed().as_secs_f64() / f64::from(repetitions);
        println!(
            "solve_seconds {seconds:.17} status {} iterations {} objective_bits {}",
            result.0, result.1, result.2
        );
        return;
    }
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
    if matches!(env::args().nth(2).as_deref(), Some("presolve_only")) {
        return;
    }
    if let Some(cap) = env::args().nth(2) {
        let mut direct = RevisedSimplex::new();
        direct.set_diagnostic_capture_direction(true);
        let mut direct_parameters = parameters.clone();
        direct_parameters.max_number_of_iterations = cap.parse().unwrap();
        direct.set_parameters(&direct_parameters);
        let direct_wall_seconds = env::args().nth(4).map_or(20.0, |value| {
            value.parse::<f64>().expect("numeric direct wall limit")
        });
        let mut direct_limit = TimeLimit::new(direct_wall_seconds, f64::INFINITY);
        direct.solve(&presolved, &mut direct_limit).unwrap();
        if matches!(env::args().nth(3).as_deref(), Some("rng")) {
            println!("random_next {}", direct.diagnostic_random_next());
            return;
        }
        if matches!(env::args().nth(3).as_deref(), Some("heap")) {
            print!(
                "random_next {}\nprice_tops",
                direct.diagnostic_random_next()
            );
            for (column, bits) in direct.diagnostic_primal_price_tops() {
                print!(" {column}:{bits}");
            }
            println!();
            return;
        }
        if matches!(env::args().nth(3).as_deref(), Some("clock")) {
            println!(
                "clock {} updates {} basis_time {} refactorizations {}",
                direct.deterministic_time().to_bits(),
                direct.num_basis_updates(),
                direct.basis_deterministic_time().to_bits(),
                direct.num_basis_refactorizations()
            );
            return;
        }
        if matches!(env::args().nth(3).as_deref(), Some("fingerprint")) {
            let mut basis = 14_695_981_039_346_656_037_u64;
            let mut values = basis;
            let mut reduced = basis;
            for row in 0..presolved.num_constraints().to_usize() {
                basis = (basis
                    ^ u64::try_from(direct.basis(RowIndex::from_usize(row)).value())
                        .expect("nonnegative basis index"))
                .wrapping_mul(1_099_511_628_211);
            }
            for col in
                0..presolved.num_variables().to_usize() + presolved.num_constraints().to_usize()
            {
                let index = ColIndex::from_usize(col);
                let value = direct.variable_value(index);
                let cost = direct.reduced_cost(index);
                values = (values ^ if value == 0.0 { 0 } else { value.to_bits() })
                    .wrapping_mul(1_099_511_628_211);
                reduced = (reduced ^ if cost == 0.0 { 0 } else { cost.to_bits() })
                    .wrapping_mul(1_099_511_628_211);
            }
            println!(
                "fingerprint {} {} {} {} {} {} {} {}",
                direct.problem_status(),
                direct.number_of_iterations(),
                basis,
                values,
                reduced,
                direct.deterministic_time().to_bits(),
                direct.basis_deterministic_time().to_bits(),
                direct.num_basis_refactorizations()
            );
            return;
        }
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
        print!("\nlu_row_perm");
        for row in direct.diagnostic_lu_row_permutation() {
            print!(" {row}");
        }
        print!(
            "\nrandom_next {}\nprice_tops",
            direct.diagnostic_random_next()
        );
        for (column, bits) in direct.diagnostic_primal_price_tops() {
            print!(" {column}:{bits}");
        }
        if matches!(env::args().nth(3).as_deref(), Some("brief")) {
            println!();
            return;
        }
        if let Some(direction) = direct.diagnostic_last_direction() {
            print!("\ndirection_values");
            for value in direction.values().as_slice() {
                print!(" {}", value.to_bits());
            }
            print!("\ndirection_order");
            for row in direction.non_zeros() {
                print!(" {}", row.to_usize());
            }
            if matches!(env::args().nth(3).as_deref(), Some("stages")) {
                for (label, (stage, order)) in ["stage_upper", "stage_updates", "stage_lower"]
                    .into_iter()
                    .zip(direct.diagnostic_left_solve_stages().unwrap())
                {
                    print!("\n{label}");
                    for value in stage {
                        print!(" {}", value.to_bits());
                    }
                    print!("\n{label}_order");
                    for index in order {
                        print!(" {index}");
                    }
                }
                println!();
                return;
            }
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
