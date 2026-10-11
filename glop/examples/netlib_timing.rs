use std::env;
use std::time::Instant;

use glop::parameters::GlopParameters;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, DenseRow, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;
use lp_data::sparse::CompactSparseMatrix;

fn main() {
    let mut arguments = env::args_os().skip(1);
    let path = arguments.next().expect("missing MPS path");
    let repetitions: usize = arguments.next().map_or(1, |value| {
        value
            .to_string_lossy()
            .parse()
            .expect("invalid repetitions")
    });
    let mode = arguments.next();
    assert!(
        mode.as_ref()
            .is_none_or(|value| value == "primal" || value == "dual" || value == "advanced")
    );
    let primal = mode.as_ref().is_some_and(|value| value == "primal");
    let advanced = mode.as_ref().is_some_and(|value| value == "advanced");
    let iteration_limit: i64 = arguments.next().map_or(1_000_000, |value| {
        value
            .to_string_lossy()
            .parse()
            .expect("invalid iteration limit")
    });
    let wall_limit: f64 = arguments.next().map_or(20.0, |value| {
        value.to_string_lossy().parse().expect("invalid wall limit")
    });
    assert!(arguments.next().is_none());
    assert!(repetitions > 0);

    let model = parse_mps_file(path).expect("failed to parse MPS");
    let parameters = GlopParameters {
        use_scaling: false,
        use_dual_simplex: !primal,
        max_number_of_iterations: iteration_limit,
        ..GlopParameters::default()
    };
    let start = Instant::now();
    let mut simplex = RevisedSimplex::new();
    for _ in 0..repetitions {
        simplex = RevisedSimplex::new();
        simplex.set_parameters(&parameters);
        let mut limit = TimeLimit::new(wall_limit, f64::INFINITY);
        if advanced {
            let mut with_slack = CompactSparseMatrix::default();
            with_slack.populate_from_sparse_and_add_slacks(model.matrix());
            simplex
                .mutable_transposed_matrix_with_slack()
                .populate_from_transpose(&with_slack);
            let rows = model.num_constraints().to_usize();
            let columns = model.num_variables().to_usize();
            let mut lower = vec![0.0; columns + rows];
            let mut upper = vec![0.0; columns + rows];
            let mut objective = vec![0.0; columns + rows];
            for column in 0..columns {
                let index = ColIndex::from_usize(column);
                lower[column] = model.variable_lower_bounds()[index];
                upper[column] = model.variable_upper_bounds()[index];
                objective[column] = model.objective_coefficients()[index];
            }
            for row in 0..rows {
                let index = RowIndex::from_usize(row);
                lower[columns + row] = -model.constraint_upper_bounds()[index];
                upper[columns + row] = -model.constraint_lower_bounds()[index];
            }
            *simplex.mutable_lower_bounds() = lower;
            *simplex.mutable_upper_bounds() = upper;
            simplex
                .minimize_from_transposed_matrix_with_slack(
                    &DenseRow::from_vec(objective),
                    1.0,
                    0.0,
                    &mut limit,
                )
                .expect("advanced solve failed");
        } else {
            simplex.solve(&model, &mut limit).expect("solve failed");
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{elapsed:.17} {} {} {:.17}",
        simplex.problem_status(),
        simplex.number_of_iterations(),
        simplex.objective_value()
    );
}
