//! Isolated counterpart of GLOP's compact-column scalar-product benchmark.

use std::io::{self, Read};
use std::rc::Rc;
use std::time::Instant;

use lp_data::lp_types::{RowIndex, VectorIndex};
use lp_data::sparse::CompactSparseMatrix;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let columns: usize = fields.next().unwrap().parse().unwrap();
    let entries_per_column: usize = fields.next().unwrap().parse().unwrap();
    let repetitions: u32 = fields.next().unwrap().parse().unwrap();
    let mut matrix = CompactSparseMatrix::default();
    matrix.reset(RowIndex::from_usize(rows));
    for _ in 0..columns {
        for _ in 0..entries_per_column {
            let row: usize = fields.next().unwrap().parse().unwrap();
            let coefficient: f64 = fields.next().unwrap().parse().unwrap();
            matrix.add_entry_to_current_column(RowIndex::from_usize(row), coefficient);
        }
        matrix.close_current_column();
    }
    let matrix = Rc::new(matrix);
    let validated = CompactSparseMatrix::validated_rows(Rc::clone(&matrix), rows)
        .expect("benchmark rows fit the dense vector");
    let mode = std::env::args().nth(1);
    let vector: Vec<f64> = (0..rows)
        .map(|row| 1.0 + f64::from(u32::try_from(row % 17).unwrap()) * 0.125)
        .collect();
    let bound = validated.bind(&matrix, &vector);
    let mut checksum = 0.0;
    let start = Instant::now();
    for _ in 0..repetitions {
        for column in 0..columns {
            let column = lp_data::lp_types::ColIndex::from_usize(column);
            checksum += match mode.as_deref() {
                Some("bound") => bound.column_scalar_product(column),
                Some("validated") => {
                    matrix.column_scalar_product_validated(&validated, column, &vector)
                }
                _ => matrix.column_scalar_product_slice(column, &vector),
            };
        }
    }
    let nanos = start.elapsed().as_secs_f64() * 1.0e9;
    println!(
        "{:.17} {:.17}",
        nanos / (f64::from(repetitions) * f64::from(u32::try_from(columns).unwrap())),
        checksum
    );
}
