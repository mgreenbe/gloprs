use std::hint::black_box;
use std::io::{self, Read};
use std::time::Instant;

use glop::lu_factorization::LuFactorization;
use glop::parameters::GlopParameters;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let repetitions: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    for _ in 0..entries {
        let row = fields.next().unwrap().parse().unwrap();
        let column = fields.next().unwrap().parse().unwrap();
        let value = fields.next().unwrap().parse().unwrap();
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .set_coefficient(RowIndex::from_usize(row), value);
    }
    matrix.clean_up();
    let parameters = GlopParameters::default();
    let mut checksum = 0.0;
    let start = Instant::now();
    for _ in 0..repetitions {
        let factorization =
            LuFactorization::factorize_with_parameters(black_box(&matrix), &parameters).unwrap();
        checksum += black_box(factorization.determinant());
    }
    println!("{:.17} {:.17}", start.elapsed().as_secs_f64(), checksum);
}
