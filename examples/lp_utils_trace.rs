use std::io::{self, Read};

use lp_data::lp_types::{RowIndex, TypedVec, VectorIndex};
use lp_data::lp_utils::{
    SumWithNegativeInfiniteAndOneMissing, SumWithPositiveInfiniteAndOneMissing, change_sign,
    clear_and_resize_vector_with_non_zeros, compute_non_zeros, infinity_norm,
    permute_with_known_non_zeros, permute_with_scratchpad, precise_scalar_product,
    precise_squared_norm, scalar_product, squared_norm, squared_norm_and_reset_to_zero,
};
use lp_data::permutation::RowPermutation;
use lp_data::scattered_vector::ScatteredColumn;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let left: Vec<f64> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let right: Vec<f64> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let mut reset = left.clone();
    let left_typed = TypedVec::<RowIndex, _>::from_vec(left.clone());
    let mut support = Vec::new();
    compute_non_zeros(&left_typed, &mut support);
    let permutation = RowPermutation::from_vec(
        (0..n)
            .map(|row| RowIndex::from_usize(n - row - 1))
            .collect(),
    );
    let mut scratch = TypedVec::filled(RowIndex::from_usize(n), 0.0);
    let mut permuted = left_typed.clone();
    permute_with_scratchpad(&permutation, &mut scratch, &mut permuted);
    let mut known_permuted = left_typed.clone();
    permute_with_known_non_zeros(
        &permutation,
        &mut scratch,
        &mut known_permuted,
        &mut support,
    );
    change_sign(&mut known_permuted);
    let mut positive_sum = SumWithPositiveInfiniteAndOneMissing::default();
    let mut negative_sum = SumWithNegativeInfiniteAndOneMissing::default();
    for &value in &left {
        positive_sum.add(value);
        negative_sum.add(value);
    }
    for _ in 0..n % 3 {
        positive_sum.add(f64::INFINITY);
        negative_sum.add(f64::NEG_INFINITY);
    }
    let mut cleared = ScatteredColumn::new(RowIndex::from_usize(n));
    for (row, &value) in left.iter().enumerate() {
        if value != 0.0 {
            cleared.add(RowIndex::from_usize(row), value);
        }
    }
    cleared.sort_non_zeros_if_needed();
    clear_and_resize_vector_with_non_zeros(RowIndex::from_usize(n), &mut cleared);
    if n != 0 {
        cleared.add(RowIndex::new(0), 1.0);
    }
    println!("scalar {:.17e}", scalar_product(&left, &right));
    println!(
        "precise_scalar {:.17e}",
        precise_scalar_product(&left, &right)
    );
    println!("squared {:.17e}", squared_norm(&left));
    println!("precise_squared {:.17e}", precise_squared_norm(&left));
    println!("infinity {:.17e}", infinity_norm(&left));
    println!("reset {:.17e}", squared_norm_and_reset_to_zero(&mut reset));
    print!("reset_values");
    for value in reset {
        print!(" {value:.17e}");
    }
    print!("\nsupport");
    for row in support {
        print!(" {}", row.to_usize());
    }
    print!("\npermuted");
    for value in permuted.as_slice() {
        print!(" {value:.17e}");
    }
    print!("\nknown_negated");
    for value in known_permuted.as_slice() {
        print!(" {value:.17e}");
    }
    let omitted = left.first().copied().unwrap_or(0.0);
    print!(
        "\npositive_sum {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
        positive_sum.sum(),
        positive_sum.sum_without(omitted),
        positive_sum.sum_without_lb(omitted),
        positive_sum.sum_without_ub(omitted),
        positive_sum.sum_without(f64::INFINITY)
    );
    print!(
        "\nnegative_sum {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
        negative_sum.sum(),
        negative_sum.sum_without(omitted),
        negative_sum.sum_without_lb(omitted),
        negative_sum.sum_without_ub(omitted),
        negative_sum.sum_without(f64::NEG_INFINITY)
    );
    print!("\nclear_protocol {}", cleared.non_zeros().len());
    if n != 0 {
        print!(" {:.17e}", cleared.value(RowIndex::new(0)));
    }
    println!();
}
