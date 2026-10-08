use std::io::{self, Read};

use glop::update_row::{UpdateRow, UpdateRowAlgorithm};
use lp_data::lp_types::{ColBitVec, ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let structural_columns: usize = fields.next().unwrap().parse().unwrap();
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let columns = structural_columns + rows;
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::from_usize(rows), ColIndex::from_usize(columns));
    for _ in 0..entries {
        let row: usize = fields.next().unwrap().parse().unwrap();
        let column: usize = fields.next().unwrap().parse().unwrap();
        let value: f64 = fields.next().unwrap().parse().unwrap();
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .add_entry(RowIndex::from_usize(row), value);
    }
    for row in 0..rows {
        matrix
            .mutable_column(ColIndex::from_usize(structural_columns + row))
            .add_entry(RowIndex::from_usize(row), 1.0);
    }
    matrix.clean_up();
    let lhs: Vec<f64> = (0..rows)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let mut relevant = ColBitVec::new(ColIndex::from_usize(columns));
    for column in 0..columns {
        relevant.set(ColIndex::from_usize(column));
    }
    let mut update = UpdateRow::new(&matrix);
    for (name, algorithm) in [
        ("column", UpdateRowAlgorithm::Column),
        ("row", UpdateRowAlgorithm::Row),
        ("row_hypersparse", UpdateRowAlgorithm::RowHypersparse),
    ] {
        update
            .compute_update_row_for_benchmark(&matrix, &relevant, &lhs, algorithm)
            .unwrap();
        print!("{name}_positions");
        for column in update.non_zero_positions() {
            print!(" {column}");
        }
        println!();
        print!("{name}_coefficients");
        for value in update.coefficients() {
            print!(" {value}");
        }
        println!();
    }
    println!("deterministic_time {:.17}", update.deterministic_time());
    println!("stats_size {}", update.stat_string().len());
}
