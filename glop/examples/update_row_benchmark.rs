//! Isolated column-wise update-row benchmark paired with pinned GLOP.

use std::hint::black_box;
use std::rc::Rc;
use std::time::Instant;

use glop::update_row::{UpdateRow, UpdateRowAlgorithm};
use lp_data::lp_types::{ColBitVec, ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("nonnegative integer"))
        .collect();
    assert_eq!(args.len(), 5);
    let [rows, structural_columns, entries, repetitions, validated] = args.try_into().unwrap();
    assert!(rows > 0 && structural_columns > 0);
    assert!(entries * 13 < rows && repetitions > 0);
    let columns = structural_columns + rows;
    let mut sparse = SparseMatrix::new();
    sparse.populate_from_zero(RowIndex::from_usize(rows), ColIndex::from_usize(columns));
    for column in 0..structural_columns {
        for entry in 0..entries {
            sparse
                .mutable_column(ColIndex::from_usize(column))
                .add_entry(
                    RowIndex::from_usize((column * 97 + entry * 13) % rows),
                    f64::from(u32::try_from(entry % 7 + 1).unwrap()) * 0.125,
                );
        }
    }
    for row in 0..rows {
        sparse
            .mutable_column(ColIndex::from_usize(structural_columns + row))
            .add_entry(RowIndex::from_usize(row), 1.0);
    }
    sparse.clean_up();
    let matrix = Rc::new(CompactSparseMatrix::from_sparse(&sparse));
    let token = (validated != 0)
        .then(|| CompactSparseMatrix::validated_rows(Rc::clone(&matrix), rows).unwrap());
    let mut relevant = ColBitVec::new(ColIndex::from_usize(columns));
    for column in 0..columns {
        relevant.set(ColIndex::from_usize(column));
    }
    let mut lhs = vec![0.0; rows];
    for (row, value) in lhs.iter_mut().enumerate() {
        if row % 4 == 0 {
            *value = f64::from(u32::try_from(row % 9 + 1).unwrap()) * 0.125;
        }
    }
    let mut update = UpdateRow::new(&sparse);
    let mut checksum = 0.0;
    let start = Instant::now();
    for repetition in 0..repetitions {
        update
            .compute_update_row_for_benchmark_compact(
                &matrix,
                token.as_ref(),
                &relevant,
                black_box(&lhs),
                UpdateRowAlgorithm::Column,
            )
            .unwrap();
        checksum += update.coefficient(repetition % columns);
    }
    let micros = start.elapsed().as_secs_f64() * 1e6
        / f64::from(u32::try_from(repetitions).expect("repetitions fit u32"));
    println!("{micros:.17} {checksum:.17}");
}
