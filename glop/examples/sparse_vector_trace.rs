use std::io::{self, Read};

use lp_data::lp_types::{DenseColumn, RowIndex, VectorIndex};
use lp_data::permutation::RowPermutation;
use lp_data::sparse_vector::{ColumnView, RandomAccessSparseColumn, SparseColumn};

fn print_vector(name: &str, vector: &SparseColumn) {
    print!("{name} {}", vector.num_entries());
    for entry in vector {
        print!(" {} {:.17e}", entry.index().value(), entry.coefficient());
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let count: usize = fields.next().unwrap().parse().unwrap();
    let mut vector = SparseColumn::new();
    for _ in 0..count {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        vector.set_coefficient(row, value);
    }
    let weights = DenseColumn::from_vec(
        (0..n)
            .map(|_| fields.next().unwrap().parse().unwrap())
            .collect(),
    );
    let threshold = fields.next().unwrap().parse().unwrap();
    let partial = RowPermutation::from_vec(
        (0..n)
            .map(|_| RowIndex::new(fields.next().unwrap().parse().unwrap()))
            .collect(),
    );
    let tags: Vec<i32> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();

    let mut clean = SparseColumn::new();
    clean.populate_from_sparse_vector(&vector);
    clean.clean_up();
    print_vector("clean", &clean);
    print!("view {}", clean.num_entries());
    for entry in ColumnView::from_column(&clean) {
        print!(" {} {:.17e}", entry.index().value(), entry.coefficient());
    }
    println!();

    let mut random_access = RandomAccessSparseColumn::new(RowIndex::from_usize(n));
    random_access.populate_from_sparse_column(&clean);
    random_access.add_to_coefficient(RowIndex::new(0), 1.25);
    random_access.set_coefficient(RowIndex::from_usize(n - 1), -0.5);
    let mut random_access_output = SparseColumn::new();
    random_access.populate_sparse_column(&mut random_access_output);
    print_vector("random", &random_access_output);

    let mut near = clean.clone();
    near.remove_near_zero_entries(threshold);
    print_vector("near", &near);

    let mut weighted = clean.clone();
    weighted.remove_near_zero_entries_with_weights(threshold, &weights);
    print_vector("weighted", &weighted);

    let mut permuted = clean.clone();
    permuted.apply_partial_index_permutation(&partial);
    print_vector("partial", &permuted);

    let tagged = RowPermutation::from_vec(tags.into_iter().map(RowIndex::new).collect());
    let mut retained = clean;
    let mut moved = SparseColumn::new();
    retained.move_tagged_entries_to(&tagged, &mut moved);
    print_vector("retained", &retained);
    print_vector("moved", &moved);
}
