//! Paired source-level indexing comparison for `UpdateRow`'s sparse dot product.

use std::hint::black_box;
use std::time::Instant;

use lp_data::lp_types::{RowIndex, VectorIndex};
use lp_data::sparse_vector::SparseColumn;

#[inline(never)]
fn indexed(column: &SparseColumn, values: &[f64]) -> f64 {
    let mut position = 0;
    let shifted_end = column.num_entries().saturating_sub(3);
    let (mut r1, mut r2, mut r3, mut r4) = (0.0, 0.0, 0.0, 0.0);
    while position < shifted_end {
        r1 = column
            .coefficient(position)
            .mul_add(values[column.index(position).to_usize()], r1);
        r2 = column
            .coefficient(position + 1)
            .mul_add(values[column.index(position + 1).to_usize()], r2);
        r3 = column
            .coefficient(position + 2)
            .mul_add(values[column.index(position + 2).to_usize()], r3);
        r4 = column
            .coefficient(position + 3)
            .mul_add(values[column.index(position + 3).to_usize()], r4);
        position += 4;
    }
    let mut result = r1 + r2 + r3 + r4;
    while position < column.num_entries() {
        result = column
            .coefficient(position)
            .mul_add(values[column.index(position).to_usize()], result);
        position += 1;
    }
    result
}

#[inline(never)]
fn paired(column: &SparseColumn, values: &[f64]) -> f64 {
    let (rows, coefficients) = column.entry_slices();
    let mut row_chunks = rows.chunks_exact(4);
    let mut coefficient_chunks = coefficients.chunks_exact(4);
    let (mut r1, mut r2, mut r3, mut r4) = (0.0, 0.0, 0.0, 0.0);
    for (r, c) in row_chunks.by_ref().zip(coefficient_chunks.by_ref()) {
        r1 = c[0].mul_add(values[r[0].to_usize()], r1);
        r2 = c[1].mul_add(values[r[1].to_usize()], r2);
        r3 = c[2].mul_add(values[r[2].to_usize()], r3);
        r4 = c[3].mul_add(values[r[3].to_usize()], r4);
    }
    let mut result = r1 + r2 + r3 + r4;
    for (&row, &coefficient) in row_chunks
        .remainder()
        .iter()
        .zip(coefficient_chunks.remainder())
    {
        result = coefficient.mul_add(values[row.to_usize()], result);
    }
    result
}

fn main() {
    let entries: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let repetitions: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let trial: usize = std::env::args().nth(3).unwrap().parse().unwrap();
    let rows = 4096;
    let columns = 1024;
    let values: Vec<f64> = (0..rows)
        .map(|row| 1.0 + f64::from(u32::try_from(row % 17).unwrap()) * 0.125)
        .collect();
    let mut matrix = Vec::with_capacity(columns);
    for column in 0..columns {
        let mut sparse = SparseColumn::new();
        for entry in 0..entries {
            let row = (column * 97 + entry * 193) % rows;
            sparse.add_entry(
                RowIndex::from_usize(row),
                f64::from(u32::try_from(entry % 13).unwrap()) * 0.125 - 0.5,
            );
        }
        matrix.push(sparse);
    }
    for mode in [trial % 2, (trial + 1) % 2] {
        let start = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..repetitions {
            for column in &matrix {
                checksum += if mode == 0 {
                    indexed(black_box(column), black_box(&values))
                } else {
                    paired(black_box(column), black_box(&values))
                };
            }
        }
        let result = start.elapsed().as_secs_f64() * 1e9
            / f64::from(u32::try_from(repetitions * columns).unwrap());
        black_box(checksum);
        println!("{mode} {result:.17} {checksum:.17}");
    }
}
