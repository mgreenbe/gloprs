use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn read_matrix(
    fields: &mut std::str::SplitWhitespace<'_>,
    rows: RowIndex,
    columns: ColIndex,
    entries: usize,
) -> SparseMatrix {
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(rows, columns);
    for _ in 0..entries {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let column = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        matrix.mutable_column(column).set_coefficient(row, value);
    }
    matrix.clean_up();
    matrix
}

fn print_matrix(name: &str, matrix: &SparseMatrix) {
    print!(
        "{name} {} {} {}",
        matrix.num_rows().value(),
        matrix.num_cols().value(),
        matrix.num_entries().value()
    );
    for position in 0..matrix.num_cols().to_usize() {
        let column = ColIndex::from_usize(position);
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

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let m = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let k_value: i32 = fields.next().unwrap().parse().unwrap();
    let n = ColIndex::new(fields.next().unwrap().parse().unwrap());
    let a_entries = fields.next().unwrap().parse().unwrap();
    let b_entries = fields.next().unwrap().parse().unwrap();
    let alpha = fields.next().unwrap().parse().unwrap();
    let beta = fields.next().unwrap().parse().unwrap();
    let a = read_matrix(&mut fields, m, ColIndex::new(k_value), a_entries);
    let b = read_matrix(&mut fields, RowIndex::new(k_value), n, b_entries);
    print_matrix("transpose", &a.transpose());
    let mut product = SparseMatrix::new();
    product.populate_from_product(&a, &b);
    print_matrix("product", &product);
    let mut combination = SparseMatrix::new();
    combination.populate_from_linear_combination(alpha, &a, beta, &a);
    print_matrix("combination", &combination);
    let (minimum, maximum) = a.min_and_max_magnitudes();
    println!("magnitudes {minimum:.17e} {maximum:.17e}");
}
