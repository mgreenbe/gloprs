use std::env;

use glop::parameters::GlopParameters;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut arguments = env::args_os().skip(1);
    let path = arguments.next().expect("missing MPS path");
    let iterations: i64 = arguments
        .next()
        .expect("missing iteration limit")
        .to_string_lossy()
        .parse()
        .expect("invalid iteration limit");
    let mut perturb = false;
    let mut transformed = false;
    let mut primal = false;
    for flag in arguments {
        if flag == "perturb" {
            perturb = true;
        } else if flag == "transformed" {
            transformed = true;
        } else if flag == "primal" {
            primal = true;
        } else {
            panic!("unknown option");
        }
    }

    let model = parse_mps_file(path).expect("failed to parse MPS");
    let parameters = GlopParameters {
        use_scaling: false,
        use_dual_simplex: !primal,
        perturb_costs_in_dual_simplex: perturb,
        use_dedicated_dual_feasibility_algorithm: !transformed,
        max_number_of_iterations: iterations,
        ..GlopParameters::default()
    };
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&parameters);
    simplex.set_trace_enabled(true);
    let mut limit = TimeLimit::from_parameters(&parameters);
    if let Err(error) = simplex.solve(&model, &mut limit) {
        println!("error {error}");
        return;
    }

    println!("iterations {}", simplex.number_of_iterations());
    println!("status {}", simplex.problem_status());
    println!("updates {}", simplex.num_basis_updates());
    for event in simplex.trace() {
        if let (Some(entering), Some(leaving), Some(leaving_column)) = (
            event.entering_column,
            event.leaving_row,
            event.leaving_column,
        ) {
            println!(
                "pivot {} {} {} {}",
                entering.to_usize(),
                leaving_column.to_usize(),
                leaving.to_usize(),
                event.iteration
            );
        }
    }
    print!("initial_basis");
    for &column in simplex.initial_basis_before_permutation().as_slice() {
        print!(" {}", column.to_usize());
    }
    print!("\ninitial_permutation");
    for &column in simplex.initial_column_permutation() {
        print!(" {column}");
    }
    println!();
    if !primal {
        print!("phasevec");
        for &value in simplex.dual_phase_one_pricing_vector().as_slice() {
            print!(" {value:.17}");
        }
        println!();
        print!("norms");
        for &value in simplex
            .dual_edge_squared_norms()
            .expect("failed to obtain dual norms")
        {
            print!(" {value:.17}");
        }
        print!("\nnorm_bits");
        for &value in simplex
            .dual_edge_squared_norms()
            .expect("failed to obtain dual norms")
        {
            print!(" {}", value.to_bits());
        }
        println!();
    }
    print!("basis");
    for row in 0..model.num_constraints().to_usize() {
        print!(" {}", simplex.basis(RowIndex::from_usize(row)).to_usize());
    }
    print!("\nvalues");
    let total_columns = model.num_variables().to_usize() + model.num_constraints().to_usize();
    for column in 0..total_columns {
        print!(
            " {:.17}",
            simplex.variable_value(ColIndex::from_usize(column))
        );
    }
    print!("\nvalue_bits");
    for column in 0..total_columns {
        print!(
            " {}",
            simplex
                .variable_value(ColIndex::from_usize(column))
                .to_bits()
        );
    }
    print!("\nreduced");
    for column in 0..total_columns {
        print!(
            " {:.17}",
            simplex.reduced_cost(ColIndex::from_usize(column))
        );
    }
    print!("\nreduced_bits");
    for column in 0..total_columns {
        print!(
            " {}",
            simplex.reduced_cost(ColIndex::from_usize(column)).to_bits()
        );
    }
    println!();
}
