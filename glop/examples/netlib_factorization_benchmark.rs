//! Time only LU construction on preloaded Matrix Market probe matrices.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use glop::lu_factorization::LuFactorization;
use glop::parameters::GlopParameters;
use lp_data::lp_types::VectorIndex;
use lp_data::matrix_market::read_matrix_market_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let repetitions: usize = arguments
        .next()
        .ok_or("usage: netlib_factorization_benchmark REPETITIONS WARMUPS FILE.mtx...")?
        .parse()?;
    let warmups: usize = arguments.next().ok_or("missing WARMUPS")?.parse()?;
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    let files: Vec<_> = arguments.collect();
    if files.is_empty() {
        return Err("at least one Matrix Market file is required".into());
    }
    let parameters = GlopParameters::default();
    for path in files {
        let matrix = read_matrix_market_file(&path)?;
        let n = matrix.num_rows().to_usize();
        if n != matrix.num_cols().to_usize() {
            return Err(format!("{path}: matrix is not square").into());
        }
        let columns: Vec<_> = (0..n).collect();
        let mut factorization = LuFactorization::new();
        for _ in 0..warmups {
            factorization.compute_factorization_selected_compact_with_parameters(
                black_box(&matrix),
                black_box(&columns),
                black_box(&parameters),
            )?;
        }
        let start = Instant::now();
        for _ in 0..repetitions {
            factorization.compute_factorization_selected_compact_with_parameters(
                black_box(&matrix),
                black_box(&columns),
                black_box(&parameters),
            )?;
        }
        let seconds = start.elapsed().as_secs_f64();
        let determinant = black_box(factorization.determinant());
        println!(
            "{path}\t{n}\t{}\t{repetitions}\t{seconds:.9}\t{determinant:.17e}",
            matrix.num_entries().value()
        );
    }
    Ok(())
}
