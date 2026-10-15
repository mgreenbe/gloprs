use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex};
use lp_data::matrix_utils::{
    are_first_columns_and_rows_exactly_equal, find_proportional_columns,
    find_proportional_columns_using_simple_algorithm, is_rightmost_square_matrix_identity,
};
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let columns = ColIndex::new(fields.next().unwrap().parse().unwrap());
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let tolerance = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(rows, columns);
    for _ in 0..entries {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let column = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        matrix.mutable_column(column).set_coefficient(row, value);
    }
    matrix.clean_up();
    for (name, mapping) in [
        ("fast", find_proportional_columns(&matrix, tolerance)),
        (
            "simple",
            find_proportional_columns_using_simple_algorithm(&matrix, tolerance),
        ),
    ] {
        print!("{name}");
        for value in mapping.as_slice() {
            print!(" {}", value.value());
        }
        println!();
    }
    println!(
        "identity {}",
        i32::from(is_rightmost_square_matrix_identity(&matrix))
    );
    let compact = CompactSparseMatrix::from_sparse(&matrix);
    println!(
        "equal {}",
        i32::from(are_first_columns_and_rows_exactly_equal(
            rows, columns, &matrix, &compact
        ))
    );
}
