use std::hint::black_box;
use std::time::Instant;

use glop::basis_representation::BasisRepresentation;
use glop::lu_factorization::LuFactorization;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn banded_matrix(n: usize) -> SparseMatrix {
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    for column in 0..n {
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .add_entry(RowIndex::from_usize(column), 4.0);
        if column > 0 {
            matrix
                .mutable_column(ColIndex::from_usize(column))
                .add_entry(RowIndex::from_usize(column - 1), -1.0);
        }
        if column + 1 < n {
            matrix
                .mutable_column(ColIndex::from_usize(column))
                .add_entry(RowIndex::from_usize(column + 1), -1.0);
        }
    }
    matrix.clean_up();
    matrix
}

fn main() {
    let matrix = banded_matrix(100);
    let start = Instant::now();
    for _ in 0..20 {
        black_box(LuFactorization::factorize(black_box(&matrix), 0.1).unwrap());
    }
    println!(
        "factorize_100_banded: 20 iterations in {:?}",
        start.elapsed()
    );

    let basis = BasisRepresentation::new(matrix, 0.1, 50).unwrap();
    let rhs = vec![1.0; 100];
    let start = Instant::now();
    for _ in 0..10_000 {
        black_box(basis.solve(black_box(&rhs)).unwrap());
    }
    println!("basis_solve_100: 10000 iterations in {:?}", start.elapsed());
}
