use std::io::{self, Read};

use glop::rank_one_update::{RankOneUpdateElementaryMatrix, RankOneUpdateFactorization};
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};

fn print_dense(name: &str, values: &[f64]) {
    print!("{name}");
    for value in values {
        print!(" {value:.17e}");
    }
    println!();
}

fn print_scattered<I: VectorIndex + Ord>(
    name: &str,
    values: &lp_data::scattered_vector::ScatteredVector<I>,
) {
    print!("{name} positions");
    for position in values.non_zeros() {
        print!(" {}", position.value_i64());
    }
    print!(" values");
    for value in values.values() {
        print!(" {value:.17e}");
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let dimension: usize = fields.next().unwrap().parse().unwrap();
    let num_updates: usize = fields.next().unwrap().parse().unwrap();
    let ratio: f64 = fields.next().unwrap().parse().unwrap();
    let mut updates = Vec::with_capacity(num_updates);
    for _ in 0..num_updates {
        let u_count: usize = fields.next().unwrap().parse().unwrap();
        let v_count: usize = fields.next().unwrap().parse().unwrap();
        let dot = fields.next().unwrap().parse().unwrap();
        let u = (0..u_count)
            .map(|_| {
                (
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                )
            })
            .collect();
        let v = (0..v_count)
            .map(|_| {
                (
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                )
            })
            .collect();
        updates.push(RankOneUpdateElementaryMatrix::new(u, v, dot));
    }
    let rhs: Vec<f64> = (0..dimension)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let pattern_size: usize = fields.next().unwrap().parse().unwrap();
    let pattern: Vec<usize> = (0..pattern_size)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();

    if let Some(elementary) = updates.first() {
        let mut right = rhs.clone();
        elementary.right_multiply(&mut right);
        print_dense("elementary_right_multiply", &right);
        elementary.right_solve(&mut right);
        print_dense("elementary_right_restore", &right);
        let mut left = rhs.clone();
        elementary.left_multiply(&mut left);
        print_dense("elementary_left_multiply", &left);
        elementary.left_solve(&mut left);
        print_dense("elementary_left_restore", &left);
        println!(
            "elementary {} {}",
            i32::from(elementary.is_singular()),
            elementary.num_entries()
        );
    }

    let mut factorization = RankOneUpdateFactorization::default();
    factorization.set_hypersparse_ratio(ratio);
    for update in updates {
        factorization.update(update);
    }
    let mut right = rhs.clone();
    factorization.right_solve(&mut right);
    print_dense("dense_right", &right);
    let mut left = rhs.clone();
    factorization.left_solve(&mut left);
    print_dense("dense_left", &left);

    let mut sparse_right = ScatteredColumn::new(RowIndex::from_usize(dimension));
    sparse_right
        .values_mut()
        .as_mut_slice()
        .copy_from_slice(&rhs);
    sparse_right
        .non_zeros_mut()
        .extend(pattern.iter().copied().map(RowIndex::from_usize));
    factorization.right_solve_with_nonzeros(&mut sparse_right);
    print_scattered("sparse_right", &sparse_right);

    let mut sparse_left = ScatteredRow::new(ColIndex::from_usize(dimension));
    sparse_left
        .values_mut()
        .as_mut_slice()
        .copy_from_slice(&rhs);
    sparse_left
        .non_zeros_mut()
        .extend(pattern.iter().copied().map(ColIndex::from_usize));
    factorization.left_solve_with_nonzeros(&mut sparse_left);
    print_scattered("sparse_left", &sparse_left);
    println!(
        "factor {} {}",
        factorization.num_entries(),
        factorization
            .deterministic_time_since_last_reset()
            .to_bits()
    );
    factorization.clear();
    println!("cleared {}", factorization.num_entries());
    factorization.reset_deterministic_time();
    println!(
        "reset {}",
        factorization
            .deterministic_time_since_last_reset()
            .to_bits()
    );
}
