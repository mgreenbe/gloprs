use std::hint::black_box;
use std::time::Instant;

use lp_data::lp_types::{ColIndex, RowIndex};
use lp_data::sparse::SparseMatrix;

fn main() {
    const ITERATIONS: usize = 2_000;
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::new(10_000), ColIndex::new(100));
    for column in 0..100 {
        let sparse_column = matrix.mutable_column(ColIndex::new(column));
        for entry in 0..100 {
            sparse_column.add_entry(RowIndex::new(column * 100 + entry), 1.0);
        }
    }
    matrix.clean_up();

    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERATIONS {
        checksum += black_box(&matrix).one_norm();
    }
    let elapsed = start.elapsed();
    println!("sparse_matrix_one_norm: {ITERATIONS} iterations in {elapsed:?}; checksum={checksum}");
}
