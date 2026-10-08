use std::io::{self, Read};
use std::{cell::RefCell, rc::Rc};

use glop::basis_representation::BasisRepresentation;
use glop::dual_edge_norms::DualEdgeNorms;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::scattered_vector::ScatteredRow;
use lp_data::sparse::SparseMatrix;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let leaving: usize = fields.next().unwrap().parse().unwrap();
    let off_diagonal_entries: usize = fields.next().unwrap().parse().unwrap();
    let deterministic_limit: f64 = fields.next().unwrap().parse().unwrap();

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
    let entering: Vec<f64> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let mut basis_matrix = SparseMatrix::new();
    basis_matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    for (row, &value) in diagonal.iter().enumerate() {
        basis_matrix
            .mutable_column(ColIndex::from_usize(row))
            .add_entry(RowIndex::from_usize(row), value);
    }
    for (row, column, value) in off_diagonal {
        basis_matrix
            .mutable_column(ColIndex::from_usize(column))
            .add_entry(RowIndex::from_usize(row), value);
    }
    basis_matrix.clean_up();
    let mut basis = BasisRepresentation::new(basis_matrix, 0.01, 10_000).unwrap();
    basis.set_column_permutation_to_identity();

    let mut norms = DualEdgeNorms::new();
    if deterministic_limit >= 0.0 {
        norms.set_time_limit(Some(Rc::new(RefCell::new(TimeLimit::new(
            f64::INFINITY,
            deterministic_limit,
        )))));
    }
    let initial = norms.edge_squared_norms(&basis).unwrap().to_vec();
    let mut warm_left_inverse = ScatteredRow::new(ColIndex::from_usize(n));
    basis
        .left_solve_for_unit_row((leaving + 1) % n, &mut warm_left_inverse)
        .unwrap();
    basis.right_solve_for_tau(&warm_left_inverse).unwrap();
    let direction = basis.solve(&entering).unwrap();
    let mut left_inverse_scattered = ScatteredRow::new(ColIndex::from_usize(n));
    basis
        .left_solve_for_unit_row(leaving, &mut left_inverse_scattered)
        .unwrap();
    let left_inverse = left_inverse_scattered.values().as_slice().to_vec();
    let precise = norms.test_precision(leaving, &left_inverse);
    norms
        .update_before_basis_pivot(&basis, leaving, &direction, &left_inverse)
        .unwrap();
    let updated = norms.edge_squared_norms(&basis).unwrap().to_vec();

    println!("entries {}", basis.number_of_entries_in_lu());
    print!("initial");
    for value in initial {
        print!(" {value:.17}");
    }
    print!("\ndirection");
    for value in direction {
        print!(" {value:.17}");
    }
    print!("\nleft");
    for value in left_inverse {
        print!(" {value:.17}");
    }
    println!("\nprecise {}", usize::from(precise));
    print!("updated");
    for value in updated {
        print!(" {value:.17}");
    }
    print!("\nstats_hex ");
    for byte in norms.stat_string().bytes() {
        print!("{byte:02x}");
    }
    println!();
}
