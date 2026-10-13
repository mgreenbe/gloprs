//! Isolated packed rank-one dense solve benchmark paired with pinned GLOP.

use std::hint::black_box;
use std::time::Instant;

use glop::rank_one_update::{RankOneUpdateElementaryMatrix, RankOneUpdateFactorization};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("nonnegative integer"))
        .collect();
    assert_eq!(args.len(), 5);
    let [dimension, updates, entries, repetitions, left] = args.try_into().unwrap();
    assert!(dimension >= 4 && updates > 0 && entries > 0);
    assert!(entries * 13 < dimension / 2 && repetitions > 0);
    let mut factorization = RankOneUpdateFactorization::default();
    for k in 0..updates {
        let mut u = Vec::with_capacity(entries);
        let mut v = Vec::with_capacity(entries);
        for e in 0..entries {
            u.push((
                (k * 97 + e * 13) % (dimension / 2),
                f64::from(u32::try_from(e % 7 + 1).unwrap()) * 0.125,
            ));
            v.push((
                dimension / 2 + (k * 97 + e * 13) % (dimension / 2),
                f64::from(u32::try_from(e % 5 + 1).unwrap()) * 0.125,
            ));
        }
        factorization.update(RankOneUpdateElementaryMatrix::new(u, v, 0.0));
    }
    let mut values = vec![0.0; dimension];
    let mut checksum = 0.0;
    let start = Instant::now();
    for repetition in 0..repetitions {
        values.fill(1.0);
        if left != 0 {
            factorization.left_solve(black_box(&mut values));
        } else {
            factorization.right_solve(black_box(&mut values));
        }
        checksum += values[repetition % dimension];
    }
    let nanos = start.elapsed().as_secs_f64() * 1e9
        / f64::from(u32::try_from(repetitions).expect("repetitions fit u32"));
    println!("{nanos:.17} {checksum:.17}");
}
