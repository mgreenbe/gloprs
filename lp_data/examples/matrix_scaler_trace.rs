use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::matrix_scaler::SparseMatrixScaler;
use lp_data::sparse::SparseMatrix;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let columns = ColIndex::new(fields.next().unwrap().parse().unwrap());
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(rows, columns);
    for _ in 0..entries {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let column = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        matrix.mutable_column(column).set_coefficient(row, value);
    }
    matrix.clean_up();
    let mut scaler = SparseMatrixScaler::new();
    scaler.init(&matrix);
    scaler.scale(&mut matrix);
    print!("rows");
    for value in scaler.row_scales() {
        print!(" {value:.17e}");
    }
    print!("\ncolumns");
    for value in scaler.col_scales() {
        print!(" {value:.17e}");
    }
    print!("\nmatrix");
    for index in 0..matrix.num_cols().to_usize() {
        let column = ColIndex::from_usize(index);
        for entry in matrix.column(column) {
            print!(
                " {} {} {:.17e}",
                column.value(),
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();
}
