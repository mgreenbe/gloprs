//! Time only LU construction on preloaded Matrix Market probe matrices.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use gloprs::lu_factorization::LuFactorization;
use gloprs::parameters::GlopParameters;
use lp_data::lp_types::VectorIndex;
use lp_data::matrix_market::read_matrix_market_file;

// Keep option parsing and both matched timing modes in one diagnostic harness.
#[allow(clippy::too_many_lines)]
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
    let mut next = arguments.next();
    let mut evict_mib: usize = 0;
    let mut retain_factorization = false;
    let mut pivot_threshold = None;
    loop {
        match next.as_deref() {
            Some("--evict-mib") => {
                evict_mib = arguments
                    .next()
                    .ok_or("missing cache eviction size")?
                    .parse()?;
                next = arguments.next();
            }
            Some("--retain-factorization") => {
                retain_factorization = true;
                next = arguments.next();
            }
            Some("--pivot-threshold") => {
                pivot_threshold = Some(arguments.next().ok_or("missing pivot threshold")?.parse()?);
                next = arguments.next();
            }
            _ => break,
        }
    }
    if evict_mib > 1024 {
        return Err("cache eviction size exceeds 1024 MiB".into());
    }
    let eviction = vec![
        1_u8;
        evict_mib
            .checked_mul(1024 * 1024)
            .ok_or("eviction size overflow")?
    ];
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
    let mut parameters = GlopParameters::default();
    if let Some(threshold) = pivot_threshold {
        parameters.lu_factorization_pivot_threshold = threshold;
        parameters.validate()?;
    }
    let mut retained_factorization = LuFactorization::new();
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
        let mut local_factorization = LuFactorization::new();
        let factorization = if retain_factorization {
            &mut retained_factorization
        } else {
            &mut local_factorization
        };
        for _ in 0..warmups {
            factorization.compute_factorization_selected_compact_with_parameters(
                black_box(matrix),
                black_box(&columns),
                black_box(&parameters),
            )?;
        }
        let seconds = if eviction.is_empty() {
            let start = Instant::now();
            for _ in 0..repetitions {
                factorization.compute_factorization_selected_compact_with_parameters(
                    black_box(matrix),
                    black_box(&columns),
                    black_box(&parameters),
                )?;
            }
            start.elapsed().as_secs_f64()
        } else {
            let mut seconds = 0.0;
            for _ in 0..repetitions {
                // Evict the compact source and LU workspace from cache outside
                // the timer, approximating intervening simplex pivot work.
                let bytes = black_box(eviction.as_slice());
                let mut checksum = 0_u64;
                for offset in (0..bytes.len()).step_by(64) {
                    checksum = checksum.wrapping_add(u64::from(bytes[offset]));
                }
                black_box(checksum);
                let start = Instant::now();
                factorization.compute_factorization_selected_compact_with_parameters(
                    black_box(matrix),
                    black_box(&columns),
                    black_box(&parameters),
                )?;
                seconds += start.elapsed().as_secs_f64();
            }
            seconds
        };
        let determinant = black_box(factorization.determinant());
        println!("{path}\t{n}\t{entries}\t{repetitions}\t{seconds:.9}\t{determinant:.17e}");
    }
    Ok(())
}
