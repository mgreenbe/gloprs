use std::cell::{Cell, RefCell};
use std::io::{self, Read};
use std::rc::Rc;

use glop::basis_representation::BasisRepresentation;
use glop::primal_edge_norms::PrimalEdgeNorms;
use glop::time_limit::TimeLimit;
use glop::update_row::UpdateRow;
use lp_data::lp_types::{ColBitVec, ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn add_basis_entries(
    matrix: &mut SparseMatrix,
    column_offset: usize,
    diagonal: &[f64],
    off_diagonal: &[(usize, usize, f64)],
) {
    for (row, &value) in diagonal.iter().enumerate() {
        matrix
            .mutable_column(ColIndex::from_usize(column_offset + row))
            .add_entry(RowIndex::from_usize(row), value);
    }
    for &(row, column, value) in off_diagonal {
        matrix
            .mutable_column(ColIndex::from_usize(column_offset + column))
            .add_entry(RowIndex::from_usize(row), value);
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let nonbasic_columns: usize = fields.next().unwrap().parse().unwrap();
    let leaving: usize = fields.next().unwrap().parse().unwrap();
    let off_diagonal_entries: usize = fields.next().unwrap().parse().unwrap();
    let deterministic_limit: f64 = fields.next().unwrap().parse().unwrap();
    let columns = n + nonbasic_columns;

    let diagonal: Vec<f64> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let off_diagonal: Vec<(usize, usize, f64)> = (0..off_diagonal_entries)
        .map(|_| {
            (
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            )
        })
        .collect();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(columns));
    for column in 0..nonbasic_columns {
        for row in 0..n {
            let value: f64 = fields.next().unwrap().parse().unwrap();
            if value != 0.0 {
                matrix
                    .mutable_column(ColIndex::from_usize(column))
                    .add_entry(RowIndex::from_usize(row), value);
            }
        }
    }
    add_basis_entries(&mut matrix, nonbasic_columns, &diagonal, &off_diagonal);
    matrix.clean_up();

    let mut basis_matrix = SparseMatrix::new();
    basis_matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    add_basis_entries(&mut basis_matrix, 0, &diagonal, &off_diagonal);
    basis_matrix.clean_up();
    let mut basis = BasisRepresentation::new(basis_matrix, 0.01, 10_000).unwrap();
    let mut basis_variables: Vec<_> = (0..n).map(|row| nonbasic_columns + row).collect();
    if !basis.column_permutation().is_empty() {
        let mut permuted = vec![0; n];
        for (source, &destination) in basis.column_permutation().iter().enumerate() {
            permuted[destination] = basis_variables[source];
        }
        basis_variables = permuted;
    }
    basis.set_column_permutation_to_identity();
    let mut relevant = ColBitVec::new(ColIndex::from_usize(columns));
    for column in 0..nonbasic_columns {
        relevant.set(ColIndex::from_usize(column));
    }

    let mut norms = PrimalEdgeNorms::new(&matrix);
    if deterministic_limit >= 0.0 {
        norms.set_time_limit(Some(Rc::new(RefCell::new(TimeLimit::new(
            f64::INFINITY,
            deterministic_limit,
        )))));
    }
    let watcher = Rc::new(Cell::new(false));
    norms.add_recomputation_watcher(Rc::clone(&watcher));
    let matrix_norms = norms.matrix_column_norms().to_vec();
    let edges = norms
        .edge_squared_norms(&basis, &relevant)
        .unwrap()
        .to_vec();
    let devex = norms.devex_weights().to_vec();
    let entering = 0;
    let mut entering_dense = vec![0.0; n];
    for entry in matrix.column(ColIndex::from_usize(entering)) {
        entering_dense[entry.index().to_usize()] = entry.coefficient();
    }
    let direction = basis.solve(&entering_dense).unwrap();
    let precise = norms.test_entering_edge_norm_precision(entering, &direction);
    let mut update = UpdateRow::new(&matrix);
    norms
        .update_before_basis_pivot(
            &basis,
            &relevant,
            entering,
            basis_variables[leaving],
            leaving,
            &direction,
            &mut update,
        )
        .unwrap();
    let updated_edges = norms
        .edge_squared_norms(&basis, &relevant)
        .unwrap()
        .to_vec();
    let updated_devex = norms.devex_weights().to_vec();
    watcher.set(false);
    norms.set_parameters(-1.0, 150, true);
    let _ = norms.test_entering_edge_norm_precision(entering, &direction);
    let precision_watcher = watcher.get();
    watcher.set(false);
    norms.clear();
    let clear_watcher = watcher.get();

    println!("entries {}", basis.number_of_entries_in_lu());
    for (name, values) in [
        ("matrix", matrix_norms),
        ("edges", edges),
        ("devex", devex),
        ("direction", direction),
        ("updated_edges", updated_edges),
        ("updated_devex", updated_devex),
    ] {
        print!("{name}");
        for value in values {
            print!(" {value:.17}");
        }
        println!();
        if name == "direction" {
            println!("precise {}", usize::from(precise));
        }
    }
    println!("precision_watcher {}", usize::from(precision_watcher));
    println!("clear_watcher {}", usize::from(clear_watcher));
    println!("deterministic_time {:.17}", norms.deterministic_time());
    print!("stats_hex ");
    for byte in norms.stat_string().bytes() {
        print!("{byte:02x}");
    }
    println!();
}
