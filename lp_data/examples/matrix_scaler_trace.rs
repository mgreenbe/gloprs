use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::matrix_scaler::SparseMatrixScaler;
use lp_data::sparse::SparseMatrix;

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
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
    print!("\nrow_factors");
    for row in 0..=rows.to_usize() + 1 {
        let row = RowIndex::from_usize(row);
        print!(
            " {:.17e} {:.17e}",
            scaler.row_scaling_factor(row),
            scaler.row_unscaling_factor(row)
        );
    }
    print!("\ncolumn_factors");
    for column in 0..=columns.to_usize() + 1 {
        let column = ColIndex::from_usize(column);
        print!(
            " {:.17e} {:.17e}",
            scaler.col_scaling_factor(column),
            scaler.col_unscaling_factor(column)
        );
    }
    let mut row_vector = lp_data::lp_types::DenseRow::from_vec(
        (0..columns.to_usize() + 2)
            .map(|column| column as f64 + 0.25)
            .collect(),
    );
    scaler.scale_row_vector(true, &mut row_vector);
    print!("\nrow_up");
    for value in &row_vector {
        print!(" {value:.17e}");
    }
    scaler.scale_row_vector(false, &mut row_vector);
    print!("\nrow_roundtrip");
    for value in &row_vector {
        print!(" {value:.17e}");
    }
    let mut column_vector = lp_data::lp_types::DenseColumn::from_vec(
        (0..rows.to_usize() + 2)
            .map(|row| row as f64 - 0.75)
            .collect(),
    );
    scaler.scale_column_vector(true, &mut column_vector);
    print!("\ncolumn_up");
    for value in &column_vector {
        print!(" {value:.17e}");
    }
    scaler.scale_column_vector(false, &mut column_vector);
    print!("\ncolumn_roundtrip");
    for value in &column_vector {
        print!(" {value:.17e}");
    }
    scaler.init(&matrix);
    print!("\nreinit_rows");
    for value in scaler.row_scales() {
        print!(" {value:.17e}");
    }
    print!("\nreinit_columns");
    for value in scaler.col_scales() {
        print!(" {value:.17e}");
    }
    scaler.clear();
    print!(
        "\nclear {} {} {:.17e} {:.17e}",
        scaler.row_scales().len().value(),
        scaler.col_scales().len().value(),
        scaler.row_unscaling_factor(RowIndex::from_usize(rows.to_usize() + 1)),
        scaler.col_unscaling_factor(ColIndex::from_usize(columns.to_usize() + 1))
    );
    println!();
}
