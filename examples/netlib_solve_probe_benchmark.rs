//! Time sparse FTRAN/BTRAN on preloaded Netlib basis/RHS Matrix Market probes.

use std::env;
use std::hint::black_box;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::Instant;

use gloprs::lu_factorization::LuFactorization;
use gloprs::parameters::GlopParameters;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::matrix_market::read_matrix_market_file;
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};

fn solve_once(
    factorization: &LuFactorization,
    entries: &[(usize, f64)],
    n: usize,
    ftran: bool,
) -> Result<(u128, Vec<f64>), Box<dyn std::error::Error>> {
    if ftran {
        let mut work = ScatteredColumn::new(RowIndex::from_usize(n));
        for &(row, value) in entries {
            work.scatter_entry_without_mask(RowIndex::from_usize(row), value);
        }
        let start = Instant::now();
        factorization.solve_with_nonzeros(black_box(&mut work))?;
        Ok((
            start.elapsed().as_nanos(),
            work.values().as_slice().to_vec(),
        ))
    } else {
        let mut work = ScatteredRow::new(ColIndex::from_usize(n));
        for &(row, value) in entries {
            work.scatter_entry_without_mask(ColIndex::from_usize(row), value);
        }
        let start = Instant::now();
        factorization.transpose_solve_with_nonzeros(black_box(&mut work))?;
        Ok((
            start.elapsed().as_nanos(),
            work.values().as_slice().to_vec(),
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let repetitions: usize = arguments
        .next()
        .ok_or("usage: netlib_solve_probe_benchmark REPETITIONS WARMUPS FACTORIZATION.mtx BASIS.mtx RHS.mtx...")?
        .parse()?;
    let warmups: usize = arguments.next().ok_or("missing WARMUPS")?.parse()?;
    let factorization_path = arguments.next().ok_or("missing FACTORIZATION.mtx")?;
    let basis_path = arguments.next().ok_or("missing BASIS.mtx")?;
    let mut rhs_paths: Vec<String> = arguments.collect();
    if rhs_paths == ["-"] {
        rhs_paths = io::stdin().lock().lines().collect::<Result<_, _>>()?;
    }
    if repetitions == 0 || rhs_paths.is_empty() {
        return Err("positive REPETITIONS and at least one RHS.mtx are required".into());
    }
    let matrix = read_matrix_market_file(&factorization_path)?;
    let basis = read_matrix_market_file(&basis_path)?;
    let n = matrix.num_rows().to_usize();
    if matrix.num_cols().to_usize() != n
        || basis.num_rows().to_usize() != n
        || basis.num_cols().to_usize() != n
    {
        return Err("basis must be square".into());
    }
    let mut factorization = LuFactorization::new();
    factorization.compute_factorization_selected_compact_with_parameters(
        &matrix,
        &(0..n).collect::<Vec<_>>(),
        &GlopParameters::default(),
    )?;
    let inverse_permutation = factorization.inverse_column_permutation();
    for column in 0..n {
        let source = if inverse_permutation.is_empty() {
            column
        } else {
            inverse_permutation[column]
        };
        let actual: Vec<_> = basis.column(ColIndex::from_usize(column)).iter().collect();
        let expected: Vec<_> = matrix.column(ColIndex::from_usize(source)).iter().collect();
        if actual != expected {
            return Err(format!("{basis_path}: not the factorization's permuted basis").into());
        }
    }
    factorization.set_column_permutation_to_identity();

    for rhs_path in rhs_paths {
        let filename = Path::new(&rhs_path)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("invalid RHS filename")?;
        let ftran = filename.starts_with("ftran-");
        if !ftran && !filename.starts_with("btran-") {
            return Err(format!("{rhs_path}: expected ftran- or btran- filename").into());
        }
        let rhs = read_matrix_market_file(&rhs_path)?;
        if rhs.num_rows().to_usize() != n || rhs.num_cols().to_usize() != 1 {
            return Err(format!("{rhs_path}: RHS dimensions do not match basis").into());
        }
        let entries: Vec<_> = rhs
            .column(ColIndex::new(0))
            .iter()
            .map(|(row, value)| (row.to_usize(), value))
            .collect();
        let mut total_nanos = 0_u128;
        let mut l1 = 0.0;
        let mut linf: f64 = 0.0;
        let mut weighted = 0.0;
        for repetition in 0..warmups + repetitions {
            let (elapsed, values) = solve_once(&factorization, &entries, n, ftran)?;
            if repetition >= warmups {
                total_nanos += elapsed;
            }
            if repetition + 1 == warmups + repetitions {
                for (index, &value) in values.iter().enumerate() {
                    l1 += value.abs();
                    linf = linf.max(value.abs());
                    weighted += value * f64::from(u32::try_from(index % 17 + 1)?);
                }
            }
        }
        println!(
            "{rhs_path}\t{}\t{n}\t{}\t{repetitions}\t{total_nanos}\t{l1:.17e}\t{linf:.17e}\t{weighted:.17e}",
            if ftran { "ftran" } else { "btran" },
            entries.len(),
        );
    }
    Ok(())
}
