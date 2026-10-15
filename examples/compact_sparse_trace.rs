use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, DenseRow, RowIndex, VectorIndex};
use lp_data::sparse::{CompactSparseMatrix, CompactSparseMatrixView, MatrixView, SparseMatrix};

fn print_matrix(name: &str, matrix: &CompactSparseMatrix) {
    print!(
        "{name} {} {} {}",
        matrix.num_rows().value(),
        matrix.num_cols().value(),
        matrix.num_entries().value()
    );
    for position in 0..matrix.num_cols().to_usize() {
        let column = ColIndex::from_usize(position);
        for (row, coefficient) in matrix.column(column).iter() {
            print!(" {} {} {coefficient:.17e}", column.value(), row.value());
        }
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let columns = ColIndex::new(fields.next().unwrap().parse().unwrap());
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let mut sparse = SparseMatrix::new();
    sparse.populate_from_zero(rows, columns);
    for _ in 0..entries {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let column = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        sparse.mutable_column(column).set_coefficient(row, value);
    }
    sparse.clean_up();
    let compact = CompactSparseMatrix::from_sparse(&sparse);
    let mut transpose = CompactSparseMatrix::default();
    transpose.populate_from_transpose(&compact);
    let mut slacks = CompactSparseMatrix::default();
    slacks.populate_from_sparse_and_add_slacks(&sparse);
    print_matrix("compact", &compact);
    print_matrix("transpose", &transpose);
    print_matrix("slacks", &slacks);
    let selected: Vec<_> = (0..columns.to_usize())
        .rev()
        .step_by(2)
        .map(ColIndex::from_usize)
        .collect();
    let compact_view = CompactSparseMatrixView::new(&compact, &selected);
    print!(
        "compact_view {} {} {} {:.17e} {:.17e}",
        compact_view.num_rows().value(),
        compact_view.num_cols().value(),
        compact_view.num_entries().value(),
        compact_view.one_norm(),
        compact_view.infinity_norm()
    );
    for position in 0..compact_view.num_cols().to_usize() {
        let column = ColIndex::from_usize(position);
        for (row, coefficient) in compact_view.column(column).iter() {
            print!(" {} {} {coefficient:.17e}", column.value(), row.value());
        }
    }
    println!();
    let full_view = MatrixView::from_matrix(&sparse);
    let sparse_view = MatrixView::from_basis(&full_view, &selected);
    println!(
        "sparse_view {} {} {} {:.17e} {:.17e}",
        sparse_view.num_rows().value(),
        sparse_view.num_cols().value(),
        sparse_view.num_entries().value(),
        sparse_view.one_norm(),
        sparse_view.infinity_norm()
    );
    let vector = DenseRow::from_vec(
        (0..rows.to_usize())
            .map(|row| f64::from(u32::try_from(row).unwrap()) + 0.25)
            .collect(),
    );
    print!("products");
    for position in 0..columns.to_usize() {
        print!(
            " {:.17e}",
            compact.column_scalar_product(ColIndex::from_usize(position), &vector)
        );
    }
    println!();
}
