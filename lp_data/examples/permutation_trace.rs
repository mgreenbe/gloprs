use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, TypedVec, VectorIndex};
use lp_data::permutation::{
    ColumnPermutation, RowPermutation, apply_inverse_permutation, apply_permutation,
};

fn print_values<I: VectorIndex>(name: &str, values: &TypedVec<I, i64>) {
    print!("{name}");
    for value in values {
        print!(" {value}");
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let size: usize = fields.next().unwrap().parse().unwrap();
    let permutation = ColumnPermutation::from_vec(
        (0..size)
            .map(|_| ColIndex::new(fields.next().unwrap().parse().unwrap()))
            .collect(),
    );
    let values = TypedVec::<RowIndex, i64>::from_vec(
        (0..size)
            .map(|_| fields.next().unwrap().parse().unwrap())
            .collect(),
    );
    println!("check {}", i32::from(permutation.check()));
    if !permutation.check() {
        return;
    }
    println!("signature {}", permutation.signature());

    let mut inverse = ColumnPermutation::default();
    inverse.populate_from_inverse(&permutation);
    print!("inverse");
    for value in inverse.as_slice() {
        print!(" {}", value.value());
    }
    println!();

    let mut result = TypedVec::new();
    apply_permutation(&permutation, &values, &mut result);
    print_values("apply", &result);
    let mut restored = TypedVec::new();
    apply_inverse_permutation(&permutation, &result, &mut restored);
    print_values("restore", &restored);

    let mut identity = RowPermutation::new(RowIndex::from_usize(size));
    identity.populate_identity();
    print!("identity");
    for value in identity.as_slice() {
        print!(" {}", value.value());
    }
    println!();
}
