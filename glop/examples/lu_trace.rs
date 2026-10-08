use std::io::{self, Read};

use glop::lu_factorization::LuFactorization;
use glop::parameters::GlopParameters;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};
use lp_data::sparse::SparseMatrix;

#[allow(clippy::cast_precision_loss)]
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    for _ in 0..entries {
        let row: usize = fields.next().unwrap().parse().unwrap();
        let column: usize = fields.next().unwrap().parse().unwrap();
        let value: f64 = fields.next().unwrap().parse().unwrap();
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .set_coefficient(RowIndex::from_usize(row), value);
    }
    matrix.clean_up();
    let mut parameters = GlopParameters::default();
    if let Some(value) = fields.next() {
        parameters.lu_factorization_pivot_threshold = value.parse().unwrap();
        parameters.markowitz_zlatev_parameter = fields.next().unwrap().parse().unwrap();
        parameters.markowitz_singularity_threshold = fields.next().unwrap().parse().unwrap();
    }
    let factorization = match LuFactorization::factorize_with_parameters(&matrix, &parameters) {
        Ok(factorization) => factorization,
        Err(error) => {
            println!("singular {error}");
            return;
        }
    };
    print!("row_perm");
    for value in factorization.row_permutation() {
        print!(" {value}");
    }
    print!("\ninverse_col_perm");
    for value in factorization.inverse_column_permutation() {
        print!(" {value}");
    }
    println!("\ndeterminant {:.17}", factorization.determinant());
    println!(
        "deterministic_time {:.17}",
        factorization.deterministic_time_of_last_factorization()
    );
    println!("entries {}", factorization.number_of_entries());
    let upper_columns: Vec<_> = (0..n)
        .map(|column| factorization.column_of_upper(column))
        .collect();
    print!("upper");
    for (column, entries) in upper_columns.iter().enumerate() {
        for &(row, value) in entries {
            print!(" {column}:{row}:{value:.17}");
        }
    }
    println!();
    let upper_entries: usize = upper_columns.iter().map(Vec::len).sum();
    println!("upper_entries {upper_entries}");
    let rhs: Vec<f64> = (1..=n).map(|value| value as f64).collect();
    print!("right");
    for value in factorization.solve(&rhs).unwrap() {
        print!(" {value:.17}");
    }
    print!("\nleft");
    for value in factorization.transpose_solve(&rhs).unwrap() {
        print!(" {value:.17}");
    }
    let mut sparse_right = ScatteredColumn::new(RowIndex::from_usize(n));
    sparse_right.set(RowIndex::from_usize(n / 2), 1.0);
    factorization
        .solve_with_nonzeros(&mut sparse_right)
        .unwrap();
    print!("\nsparse_right_positions");
    for row in sparse_right.non_zeros() {
        print!(" {}", row.value());
    }
    print!("\nsparse_right");
    for value in sparse_right.values() {
        print!(" {value:.17}");
    }
    let mut sparse_left = ScatteredRow::new(ColIndex::from_usize(n));
    sparse_left.set(ColIndex::from_usize(n / 2), 1.0);
    factorization
        .transpose_solve_with_nonzeros(&mut sparse_left)
        .unwrap();
    print!("\nsparse_left_positions");
    for column in sparse_left.non_zeros() {
        print!(" {}", column.value());
    }
    print!("\nsparse_left");
    for value in sparse_left.values() {
        print!(" {value:.17}");
    }
    print!("\nstats_hex ");
    for byte in factorization.stat_string().bytes() {
        print!("{byte:02x}");
    }
    println!();
}
