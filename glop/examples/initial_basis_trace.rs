use std::io::{self, Read};

use glop::lu_factorization::LuFactorization;
use glop::parameters::GlopParameters;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let num_rows: usize = fields.next().unwrap().parse().unwrap();
    let num_columns: usize = fields.next().unwrap().parse().unwrap();
    let num_entries: usize = fields.next().unwrap().parse().unwrap();
    let num_candidates: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(
        RowIndex::from_usize(num_rows),
        ColIndex::from_usize(num_columns),
    );
    for _ in 0..num_entries {
        let row: usize = fields.next().unwrap().parse().unwrap();
        let column: usize = fields.next().unwrap().parse().unwrap();
        let value: f64 = fields.next().unwrap().parse().unwrap();
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .set_coefficient(RowIndex::from_usize(row), value);
    }
    matrix.clean_up();
    let candidates: Vec<_> = (0..num_candidates)
        .map(|_| ColIndex::from_usize(fields.next().unwrap().parse().unwrap()))
        .collect();
    let basis =
        LuFactorization::compute_initial_basis(&matrix, &candidates, &GlopParameters::default())
            .unwrap();
    print!("basis");
    for column in &basis {
        print!(" {}", column.value());
    }
    print!("\npivots");
    for (row, column) in LuFactorization::initial_basis_pivot_sequence(
        &matrix,
        &candidates,
        &GlopParameters::default(),
    ) {
        print!(" {row}:{}", column.value());
    }
    println!();
}
