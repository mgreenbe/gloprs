use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, TypedVec, VectorIndex};
use lp_data::permutation::ColumnPermutation;
use lp_data::sparse_row::{RowMajorSparseMatrix, SparseRow};

fn emit(label: &str, row: &SparseRow) {
    print!("{label} {}", row.num_entries());
    for position in 0..row.num_entries() {
        print!(
            " {} {:x}",
            row.entry_col(position).value(),
            row.entry_coefficient(position).to_bits()
        );
    }
    print!(" iter");
    for entry in row {
        print!(
            " {} {:x}",
            entry.index().value(),
            entry.coefficient().to_bits()
        );
    }
    println!(
        " ends {} {}",
        row.first_col().value(),
        row.last_col().value()
    );
}

fn main() {
    let mut input_text = String::new();
    io::stdin().read_to_string(&mut input_text).unwrap();
    let mut fields = input_text.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let count: usize = fields.next().unwrap().parse().unwrap();
    let mut input = SparseRow::new();
    for _ in 0..count {
        let column = ColIndex::from_usize(fields.next().unwrap().parse().unwrap());
        let coefficient: f64 = fields.next().unwrap().parse().unwrap();
        input.set_coefficient(column, coefficient);
    }
    let complete = ColumnPermutation::from_vec(
        (0..n)
            .map(|_| ColIndex::new(fields.next().unwrap().parse().unwrap()))
            .collect(),
    );
    let partial = ColumnPermutation::from_vec(
        (0..n)
            .map(|_| ColIndex::new(fields.next().unwrap().parse().unwrap()))
            .collect(),
    );

    emit("input", &input);
    let mut permuted = input.clone();
    permuted.apply_col_permutation(&complete);
    emit("complete", &permuted);
    let mut retained = input.clone();
    retained.apply_partial_col_permutation(&partial);
    emit("partial", &retained);

    let mut matrix: RowMajorSparseMatrix = TypedVec::filled(RowIndex::new(2), SparseRow::new());
    matrix[RowIndex::new(0)] = input;
    matrix[RowIndex::new(1)] = permuted;
    print!("matrix {}", matrix.len().value());
    for row in &matrix {
        print!(" {}", row.num_entries());
        for entry in row {
            print!(
                " {} {:x}",
                entry.index().value(),
                entry.coefficient().to_bits()
            );
        }
    }
    println!();
}
