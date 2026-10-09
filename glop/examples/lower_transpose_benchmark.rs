use std::io::{self, Read};
use std::time::Instant;

use lp_data::triangular_matrix::{Triangle, TriangularMatrix};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let count: usize = fields.next().unwrap().parse().unwrap();
    let repetitions: usize = fields.next().unwrap().parse().unwrap();
    let mut columns = vec![Vec::new(); n];
    for _ in 0..count {
        let row: usize = fields.next().unwrap().parse().unwrap();
        let column: usize = fields.next().unwrap().parse().unwrap();
        let value: f64 = fields.next().unwrap().parse().unwrap();
        if row != column {
            columns[column].push((row, value));
        }
    }
    let lower =
        TriangularMatrix::from_columns(&columns, &vec![1.0; n], Triangle::Lower, true).unwrap();
    let base: Vec<f64> = (0..n)
        .map(|row| 1.0 + f64::from(u32::try_from(row % 17).unwrap()) * 0.125)
        .collect();
    let mut rhs = vec![0.0; n];

    let mut checksum = 0.0;
    let start = Instant::now();
    for repetition in 0..repetitions {
        rhs.copy_from_slice(&base);
        lower.transpose_solve(&mut rhs).unwrap();
        checksum += rhs[repetition % n];
    }
    let nanos = start.elapsed().as_secs_f64() * 1.0e9;
    println!(
        "{:.17} {:.17}",
        nanos / f64::from(u32::try_from(repetitions).unwrap()),
        checksum
    );
}
