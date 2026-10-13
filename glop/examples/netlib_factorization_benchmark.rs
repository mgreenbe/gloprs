//! Time only LU construction on preloaded Matrix Market probe matrices.

use std::env;
use std::fs;
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
    let next = arguments.next();
    let selected_source = if next.as_deref() == Some("--selected-source") {
        Some(read_matrix_market_file(
            arguments.next().ok_or("missing selected source matrix")?,
        )?)
    } else {
        None
    };
    let files: Vec<_> = next
        .filter(|value| value != "--selected-source")
        .into_iter()
        .chain(arguments)
        .collect();
    if files.is_empty() {
        return Err("at least one Matrix Market file is required".into());
    }
    let parameters = GlopParameters::default();
    for path in files {
        let owned_matrix;
        let (matrix, columns) = if let Some(source) = &selected_source {
            let contents = fs::read_to_string(&path)?;
            let mut words = contents.split_whitespace();
            let n: usize = words.next().ok_or("empty selection")?.parse()?;
            let columns: Vec<usize> = words.map(str::parse).collect::<Result<_, _>>()?;
            if columns.len() != n
                || n != source.num_rows().to_usize()
                || columns
                    .iter()
                    .any(|&column| column >= source.num_cols().to_usize())
            {
                return Err(format!("{path}: invalid selected columns").into());
            }
            (source, columns)
        } else {
            owned_matrix = read_matrix_market_file(&path)?;
            let n = owned_matrix.num_rows().to_usize();
            if n != owned_matrix.num_cols().to_usize() {
                return Err(format!("{path}: matrix is not square").into());
            }
            (&owned_matrix, (0..n).collect())
        };
        let n = columns.len();
        let entries: usize = columns
            .iter()
            .map(|&column| {
                matrix
                    .column(lp_data::lp_types::ColIndex::from_usize(column))
                    .len()
            })
            .sum();
        let mut factorization = LuFactorization::new();
        for _ in 0..warmups {
            factorization.compute_factorization_selected_compact_with_parameters(
                black_box(matrix),
                black_box(&columns),
                black_box(&parameters),
            )?;
        }
        let start = Instant::now();
        for _ in 0..repetitions {
            factorization.compute_factorization_selected_compact_with_parameters(
                black_box(matrix),
                black_box(&columns),
                black_box(&parameters),
            )?;
        }
        let seconds = start.elapsed().as_secs_f64();
        let determinant = black_box(factorization.determinant());
        println!("{path}\t{n}\t{entries}\t{repetitions}\t{seconds:.9}\t{determinant:.17e}");
    }
    Ok(())
}
