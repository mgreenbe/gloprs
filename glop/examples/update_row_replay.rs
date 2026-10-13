//! Replay production update-row operands outside the solver, timed without I/O.

use std::fs;
use std::hint::black_box;
use std::rc::Rc;
use std::time::Instant;

use lp_data::lp_types::{ColBitVec, ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::CompactSparseMatrix;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("fixture path");
    let mode = args.next().expect("dot, output, or full");
    let repetitions: usize = args.next().expect("repetitions").parse().unwrap();
    assert!(
        repetitions > 0
            && matches!(
                mode.as_str(),
                "dot" | "output" | "count" | "coeffs" | "positions" | "full"
            )
    );
    let contents = fs::read_to_string(path).expect("read production probe");
    let mut fields = contents.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let columns: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = CompactSparseMatrix::default();
    matrix.reset(RowIndex::from_usize(rows));
    for _ in 0..columns {
        let count: usize = fields.next().unwrap().parse().unwrap();
        for _ in 0..count {
            let row: usize = fields.next().unwrap().parse().unwrap();
            let coefficient: f64 = fields.next().unwrap().parse().unwrap();
            matrix.add_entry_to_current_column(RowIndex::from_usize(row), coefficient);
        }
        matrix.close_current_column();
    }
    let lhs: Vec<f64> = (0..rows)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let relevant_count: usize = fields.next().unwrap().parse().unwrap();
    let mut relevant = ColBitVec::new(ColIndex::from_usize(columns));
    let mut relevant_positions = Vec::with_capacity(relevant_count);
    for _ in 0..relevant_count {
        let column: usize = fields.next().unwrap().parse().unwrap();
        relevant.set(ColIndex::from_usize(column));
        relevant_positions.push(column);
    }
    assert!(fields.next().is_none());
    let matrix = Rc::new(matrix);
    let validated = CompactSparseMatrix::validated_rows(Rc::clone(&matrix), rows).unwrap();
    let dot = validated.bind(&matrix, &lhs);
    let mut precomputed = vec![0.0; columns];
    if mode == "output" {
        for column in relevant.iter_ones() {
            precomputed[column.to_usize()] = dot.column_scalar_product(column);
        }
    }
    let mut coefficients = vec![0.0; columns];
    let mut positions = Vec::with_capacity(columns);
    let mut checksum = 0.0;
    let start = Instant::now();
    if mode == "dot" {
        for repetition in 0..repetitions {
            for column in relevant.iter_ones() {
                coefficients[column.to_usize()] = dot.column_scalar_product(column);
            }
            black_box(&coefficients);
            if !relevant_positions.is_empty() {
                checksum += coefficients[relevant_positions[repetition % relevant_count]];
            }
        }
    } else if mode == "output" {
        for repetition in 0..repetitions {
            positions.clear();
            for column in relevant.iter_ones() {
                let coefficient = precomputed[column.to_usize()];
                if coefficient.abs() > 1e-14 {
                    coefficients[column.to_usize()] = coefficient;
                    positions.push(column);
                }
            }
            checksum += f64::from(u32::try_from(positions.len()).unwrap());
            if !positions.is_empty() {
                checksum += coefficients[positions[repetition % positions.len()].to_usize()];
            }
        }
    } else if mode == "count" {
        for _ in 0..repetitions {
            let mut count = 0usize;
            for column in relevant.iter_ones() {
                let coefficient = dot.column_scalar_product(column);
                count += usize::from(coefficient.abs() > 1e-14);
            }
            checksum += f64::from(u32::try_from(count).unwrap());
        }
    } else if mode == "coeffs" {
        for repetition in 0..repetitions {
            let mut count = 0usize;
            for column in relevant.iter_ones() {
                let coefficient = dot.column_scalar_product(column);
                if coefficient.abs() > 1e-14 {
                    coefficients[column.to_usize()] = coefficient;
                    count += 1;
                }
            }
            checksum += f64::from(u32::try_from(count).unwrap());
            if !relevant_positions.is_empty() {
                checksum += coefficients[relevant_positions[repetition % relevant_count]];
            }
        }
    } else if mode == "positions" {
        for _ in 0..repetitions {
            positions.clear();
            for column in relevant.iter_ones() {
                let coefficient = dot.column_scalar_product(column);
                if coefficient.abs() > 1e-14 {
                    positions.push(column);
                }
            }
            checksum += f64::from(u32::try_from(positions.len()).unwrap());
        }
    } else {
        for repetition in 0..repetitions {
            positions.resize(columns, ColIndex::from_usize(0));
            let mut nonzero_count = 0;
            for column in relevant.iter_ones() {
                let coefficient = dot.column_scalar_product(column);
                if coefficient.abs() > 1e-14 {
                    coefficients[column.to_usize()] = coefficient;
                    positions[nonzero_count] = column;
                    nonzero_count += 1;
                }
            }
            positions.truncate(nonzero_count);
            checksum += f64::from(u32::try_from(positions.len()).unwrap());
            if !positions.is_empty() {
                checksum += coefficients[positions[repetition % positions.len()].to_usize()];
            }
        }
    }
    let micros =
        start.elapsed().as_secs_f64() * 1e6 / f64::from(u32::try_from(repetitions).unwrap());
    println!(
        "{micros:.17} {checksum:.17} {relevant_count} {}",
        matrix.num_entries().value()
    );
}
