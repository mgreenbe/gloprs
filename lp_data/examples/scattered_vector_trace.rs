use std::io::{self, Read};

use lp_data::lp_types::{RowIndex, VectorIndex};
use lp_data::scattered_vector::{ScatteredColumn, transposed_row_view};

fn print(name: &str, vector: &ScatteredColumn) {
    print!("{name} positions");
    for row in vector.non_zeros() {
        print!(" {}", row.value());
    }
    print!(" values");
    for value in vector.values() {
        print!(" {value:.17e}");
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let size = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let count: usize = fields.next().unwrap().parse().unwrap();
    let ratio: f64 = fields.next().unwrap().parse().unwrap();
    let mut vector = ScatteredColumn::new(size);
    for _ in 0..count {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        vector.add(row, value);
    }
    print("added", &vector);
    println!(
        "metrics {} {} {}",
        i32::from(vector.should_use_dense_iteration(ratio)),
        i32::from(vector.should_use_dense_iteration(0.8)),
        vector.num_non_zeros_estimate()
    );
    vector.sort_non_zeros_if_needed();
    print("sorted", &vector);

    if size.to_usize() > 0 {
        vector.clear_sparse_mask();
        vector.add(RowIndex::new(0), 0.75);
        print("mask_cleared_add", &vector);
        vector.repopulate_sparse_mask();
        vector.add(RowIndex::new(0), 0.5);
        print("mask_repopulated_add", &vector);
    }

    print!("transpose");
    for entry in transposed_row_view(&vector).iter() {
        print!(" {} {:.17e}", entry.column().value(), entry.coefficient());
    }
    println!();

    vector.clear_non_zeros_if_too_dense(ratio);
    print("density_switch", &vector);
    println!("estimate {}", vector.num_non_zeros_estimate());
}
