use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;
use lp_data::triangular_matrix::{Triangle, TriangularMatrix};

#[allow(clippy::too_many_lines)]
fn main() {
    let mut input_text = String::new();
    io::stdin().read_to_string(&mut input_text).unwrap();
    let mut fields = input_text.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let count: usize = fields.next().unwrap().parse().unwrap();
    let root = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let mut input = SparseMatrix::new();
    input.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    for _ in 0..count {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let column = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        input.mutable_column(column).set_coefficient(row, value);
    }
    input.clean_up();

    let lower = TriangularMatrix::from_sparse(&input, Triangle::Lower, false).unwrap();
    print!(
        "metadata {} {} {} {}",
        lower.num_rows(),
        lower.num_cols(),
        lower.num_entries(),
        lower.first_non_identity_column()
    );
    for column in 0..n {
        print!(
            " {:.17e} {}",
            lower.diagonal(column),
            usize::from(lower.column_is_diagonal_only(column))
        );
    }
    print!(
        " {} {} {:.17e} {:.17e}",
        usize::from(lower.is_lower_triangular()),
        usize::from(lower.is_upper_triangular()),
        lower.inverse_infinity_norm_upper_bound(),
        lower.inverse_infinity_norm()
    );
    println!();

    let mut copied = SparseMatrix::new();
    lower.copy_to_sparse_matrix(&mut copied);
    print!("copied");
    for column in 0..n {
        for entry in copied.column(ColIndex::from_usize(column)) {
            print!(
                " {column} {} {:.17e}",
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();

    let mut right: Vec<_> = (0..n)
        .map(|row| f64::from(u32::try_from(row).unwrap()) + 0.375)
        .collect();
    let mut left = right.clone();
    lower.solve(&mut right).unwrap();
    lower.transpose_solve(&mut left).unwrap();
    let start = n / 3;
    let mut starting = vec![0.0; n];
    for (row, value) in starting.iter_mut().enumerate().skip(start) {
        *value = f64::from(u32::try_from(row).unwrap()) + 0.625;
    }
    lower.lower_solve_starting_at(start, &mut starting).unwrap();
    let mut signed_zero_right = vec![-0.0; n];
    let mut signed_zero_left = vec![-0.0; n];
    lower.solve(&mut signed_zero_right).unwrap();
    lower.transpose_solve(&mut signed_zero_left).unwrap();
    print!("right");
    for value in right {
        print!(" {value:.17e}");
    }
    print!("\nleft");
    for value in left {
        print!(" {value:.17e}");
    }
    print!("\nstarting");
    for value in starting {
        print!(" {value:.17e}");
    }
    print!("\nsigned_zero");
    for value in signed_zero_right.into_iter().chain(signed_zero_left) {
        print!(" x{:x}", value.to_bits());
    }
    println!();

    let mut sorted_rows = if n == 0 { Vec::new() } else { vec![root] };
    lower.compute_rows_to_consider_in_sorted_order(&mut sorted_rows);
    let mut dfs_rows = if n == 0 { Vec::new() } else { vec![root] };
    lower.compute_rows_to_consider_with_dfs(&mut dfs_rows);
    print!("sorted {}", sorted_rows.len());
    for row in &sorted_rows {
        print!(" {}", row.value());
    }
    print!("\ndfs {}", dfs_rows.len());
    for row in &dfs_rows {
        print!(" {}", row.value());
    }

    let mut hyper_sorted = vec![0.0; n];
    if n != 0 {
        hyper_sorted[root.to_usize()] = 1.25;
    }
    let mut hyper_sorted_rows = sorted_rows.clone();
    lower.hyper_sparse_solve(&mut hyper_sorted, &mut hyper_sorted_rows);
    print!("\nhyper_sorted {}", hyper_sorted_rows.len());
    for row in &hyper_sorted_rows {
        print!(" {}", row.value());
    }
    for value in hyper_sorted {
        print!(" {value:.17e}");
    }

    let mut hyper_dfs = vec![0.0; n];
    if n != 0 {
        hyper_dfs[root.to_usize()] = 1.25;
    }
    let mut hyper_dfs_rows = dfs_rows.clone();
    lower.hyper_sparse_solve_with_reversed_nonzeros(&mut hyper_dfs, &mut hyper_dfs_rows);
    print!("\nhyper_dfs {}", hyper_dfs_rows.len());
    for row in &hyper_dfs_rows {
        print!(" {}", row.value());
    }
    for value in hyper_dfs {
        print!(" {value:.17e}");
    }

    let upper = TriangularMatrix::from_sparse(&input.transpose(), Triangle::Upper, false).unwrap();
    let mut transpose_rows = if n == 0 {
        Vec::new()
    } else {
        vec![RowIndex::from_usize(n - 1)]
    };
    upper.compute_rows_to_consider_in_sorted_order(&mut transpose_rows);
    let mut transpose_hyper = vec![0.0; n];
    if n != 0 {
        transpose_hyper[n - 1] = -0.75;
    }
    lower.transpose_hyper_sparse_solve_with_reversed_nonzeros(
        &mut transpose_hyper,
        &mut transpose_rows,
    );
    print!("\ntranspose_hyper {}", transpose_rows.len());
    for row in &transpose_rows {
        print!(" {}", row.value());
    }
    for value in transpose_hyper {
        print!(" {value:.17e}");
    }
    println!();

    let mut normalized = TriangularMatrix::empty(Triangle::Lower, true);
    normalized.reset(n, n);
    for column in 0..n {
        let column_index = ColIndex::from_usize(column);
        let diagonal = input.look_up_value(RowIndex::from_usize(column), column_index);
        normalized.add_and_normalize_triangular_column(
            input.column(column_index),
            RowIndex::from_usize(column),
            diagonal,
        );
    }
    let mut normalized_copy = SparseMatrix::new();
    normalized.copy_to_sparse_matrix(&mut normalized_copy);
    print!("normalized {}", normalized.first_non_identity_column());
    for column in 0..n {
        for entry in normalized_copy.column(ColIndex::from_usize(column)) {
            print!(
                " {column} {} {:.17e}",
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();
}
