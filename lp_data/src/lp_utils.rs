//! Numerically ordered vector utilities from `ortools/lp_data/lp_utils`.

use crate::lp_types::{RowIndex, TypedVec, VectorIndex};
use crate::permutation::Permutation;
use crate::scattered_vector::{ScatteredColumn, ScatteredVector};
use crate::sparse_vector::SparseColumn;

#[must_use]
pub fn square(value: f64) -> f64 {
    value * value
}

/// Distance to the nearest integer, in `[0, 0.5]` for finite inputs.
#[must_use]
pub fn fractionality(value: f64) -> f64 {
    (value - value.round()).abs()
}

/// Dense scalar product using GLOP's four-products-per-accumulation order.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn scalar_product(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    let mut sum = 0.0;
    let blocks = left.len() / 4;
    for block in 0..blocks {
        let i = 4 * block;
        sum += left[i] * right[i]
            + left[i + 1] * right[i + 1]
            + left[i + 2] * right[i + 2]
            + left[i + 3] * right[i + 3];
    }
    for i in 4 * blocks..left.len() {
        sum += left[i] * right[i];
    }
    sum
}

fn accurate_sum(terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut error_sum = 0.0;
    for term in terms {
        error_sum += term;
        let new_sum = sum + error_sum;
        error_sum += sum - new_sum;
        sum = new_sum;
    }
    sum
}

/// Dense scalar product using GLOP's `AccurateSum` recurrence.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn precise_scalar_product(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    accurate_sum(left.iter().zip(right).map(|(&a, &b)| a * b))
}

#[must_use]
pub fn sparse_scalar_product(dense: &[f64], sparse: &SparseColumn) -> f64 {
    sparse.into_iter().fold(0.0, |sum, entry| {
        sum + dense[entry.index().to_usize()] * entry.coefficient()
    })
}

#[must_use]
/// # Panics
///
/// Panics if the dense and scattered vectors have different dimensions.
pub fn scattered_scalar_product(dense: &[f64], scattered: &ScatteredColumn) -> f64 {
    assert_eq!(dense.len(), scattered.len().to_usize());
    if scattered.should_use_dense_iteration(0.8) {
        scalar_product(dense, scattered.values().as_slice())
    } else {
        scattered.iter().fold(0.0, |sum, entry| {
            sum + dense[entry.index().to_usize()] * entry.coefficient()
        })
    }
}

#[must_use]
pub fn precise_sparse_scalar_product(dense: &[f64], sparse: &SparseColumn) -> f64 {
    accurate_sum(
        sparse
            .into_iter()
            .map(|entry| dense[entry.index().to_usize()] * entry.coefficient()),
    )
}

/// Scalar product over the sorted sparse prefix with row `< end`.
#[must_use]
pub fn partial_scalar_product(dense: &[f64], sparse: &SparseColumn, end: usize) -> f64 {
    let mut sum = 0.0;
    for entry in sparse {
        let row = entry.index().to_usize();
        if row >= end {
            return sum;
        }
        sum += dense[row] * entry.coefficient();
    }
    sum
}

/// Returns the squared Euclidean norm using GLOP's four-accumulator order.
#[must_use]
pub fn squared_norm(data: &[f64]) -> f64 {
    let mut sums = [0.0; 4];
    let chunks = data.len() / 4;
    for chunk in 0..chunks {
        let i = 4 * chunk;
        sums[0] += data[i] * data[i];
        sums[1] += data[i + 1] * data[i + 1];
        sums[2] += data[i + 2] * data[i + 2];
        sums[3] += data[i + 3] * data[i + 3];
    }
    let mut sum = sums[0] + sums[1] + sums[2] + sums[3];
    for &value in &data[4 * chunks..] {
        sum += value * value;
    }
    sum
}

/// Returns the squared norm and clears every dense entry in one traversal.
pub fn squared_norm_and_reset_to_zero(data: &mut [f64]) -> f64 {
    let mut sums = [0.0; 4];
    let chunks = data.len() / 4;
    for chunk in 0..chunks {
        let i = 4 * chunk;
        sums[0] += data[i] * data[i];
        sums[1] += data[i + 1] * data[i + 1];
        sums[2] += data[i + 2] * data[i + 2];
        sums[3] += data[i + 3] * data[i + 3];
        data[i..i + 4].fill(0.0);
    }
    let mut sum = sums[0] + sums[1] + sums[2] + sums[3];
    for value in &mut data[4 * chunks..] {
        sum += *value * *value;
        *value = 0.0;
    }
    sum
}

#[must_use]
pub fn sparse_squared_norm(sparse: &SparseColumn) -> f64 {
    sparse
        .into_iter()
        .fold(0.0, |sum, entry| sum + square(entry.coefficient()))
}

#[must_use]
pub fn scattered_squared_norm(scattered: &ScatteredColumn) -> f64 {
    if scattered.should_use_dense_iteration(0.8) {
        squared_norm(scattered.values().as_slice())
    } else {
        scattered
            .iter()
            .fold(0.0, |sum, entry| sum + square(entry.coefficient()))
    }
}

/// Returns the squared norm using GLOP's `AccurateSum` recurrence.
#[must_use]
pub fn precise_squared_norm(data: &[f64]) -> f64 {
    accurate_sum(data.iter().map(|&value| square(value)))
}

#[must_use]
pub fn precise_sparse_squared_norm(sparse: &SparseColumn) -> f64 {
    accurate_sum(sparse.into_iter().map(|entry| square(entry.coefficient())))
}

#[must_use]
pub fn precise_scattered_squared_norm(scattered: &ScatteredColumn) -> f64 {
    if scattered.should_use_dense_iteration(0.8) {
        precise_squared_norm(scattered.values().as_slice())
    } else {
        accurate_sum(scattered.iter().map(|entry| square(entry.coefficient())))
    }
}

#[must_use]
pub fn infinity_norm(data: &[f64]) -> f64 {
    data.iter().fold(0.0, |norm, value| norm.max(value.abs()))
}

#[must_use]
pub fn sparse_infinity_norm(sparse: &SparseColumn) -> f64 {
    sparse
        .into_iter()
        .fold(0.0, |norm, entry| norm.max(entry.coefficient().abs()))
}

#[must_use]
pub fn density(data: &[f64]) -> f64 {
    if data.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let result = data.iter().filter(|&&value| value != 0.0).count() as f64 / data.len() as f64;
        result
    }
}

pub fn remove_near_zero_entries(threshold: f64, data: &mut [f64]) {
    if threshold == 0.0 {
        return;
    }
    for value in data {
        if value.abs() < threshold {
            *value = 0.0;
        }
    }
}

/// Returns the restricted infinity norm and the first row attaining it.
#[must_use]
pub fn restricted_infinity_norm(
    sparse: &SparseColumn,
    rows_to_consider: &[bool],
) -> (f64, Option<RowIndex>) {
    let mut norm = 0.0;
    let mut row = None;
    for entry in sparse {
        let magnitude = entry.coefficient().abs();
        if rows_to_consider[entry.index().to_usize()] && magnitude > norm {
            norm = magnitude;
            row = Some(entry.index());
        }
    }
    (norm, row)
}

pub fn set_support_to_false(sparse: &SparseColumn, support: &mut [bool]) {
    for entry in sparse {
        if entry.coefficient() != 0.0 {
            support[entry.index().to_usize()] = false;
        }
    }
}

#[must_use]
pub fn is_dominated(sparse: &SparseColumn, radius: &[f64]) -> bool {
    sparse.into_iter().all(|entry| {
        debug_assert!(radius[entry.index().to_usize()] >= 0.0);
        entry.coefficient().abs() <= radius[entry.index().to_usize()]
    })
}

/// Computes the positions of the nonzeros of a dense typed vector.
pub fn compute_non_zeros<I: VectorIndex>(input: &TypedVec<I, f64>, non_zeros: &mut Vec<I>) {
    non_zeros.clear();
    for (position, &value) in input.as_slice().iter().enumerate() {
        if value != 0.0 {
            non_zeros.push(I::from_usize(position));
        }
    }
}

#[must_use]
pub fn is_all_zero(input: impl IntoIterator<Item = impl std::borrow::Borrow<f64>>) -> bool {
    input.into_iter().all(|value| *value.borrow() == 0.0)
}

#[must_use]
pub fn is_all_false(input: impl IntoIterator<Item = impl std::borrow::Borrow<bool>>) -> bool {
    input.into_iter().all(|value| !*value.borrow())
}

/// Applies a permutation using an all-zero reusable dense scratchpad.
///
/// # Panics
///
/// Panics if the permutation and vector dimensions differ.
pub fn permute_with_scratchpad<P: VectorIndex, I: VectorIndex>(
    permutation: &Permutation<P>,
    zero_scratchpad: &mut TypedVec<I, f64>,
    input_output: &mut TypedVec<I, f64>,
) {
    debug_assert!(is_all_zero(zero_scratchpad.as_slice()));
    let size = input_output.len();
    assert_eq!(permutation.as_slice().len(), size.to_usize());
    std::mem::swap(zero_scratchpad, input_output);
    input_output.resize(size, 0.0);
    for (position, &destination) in permutation.as_slice().iter().enumerate() {
        let value = zero_scratchpad[I::from_usize(position)];
        if value != 0.0 {
            input_output[I::from_usize(destination.to_usize())] = value;
        }
    }
    zero_scratchpad.assign(size, 0.0);
}

/// Applies a permutation when the input support is already known.
pub fn permute_with_known_non_zeros<I: VectorIndex>(
    permutation: &Permutation<I>,
    zero_scratchpad: &mut TypedVec<I, f64>,
    output: &mut TypedVec<I, f64>,
    non_zeros: &mut [I],
) {
    debug_assert!(is_all_zero(zero_scratchpad.as_slice()));
    std::mem::swap(zero_scratchpad, output);
    output.resize(zero_scratchpad.len(), 0.0);
    for index in non_zeros {
        let value = zero_scratchpad[*index];
        zero_scratchpad[*index] = 0.0;
        let permuted = permutation[*index];
        output[permuted] = value;
        *index = permuted;
    }
}

pub fn clear_and_resize_vector_with_non_zeros<I: VectorIndex + Ord>(
    size: I,
    vector: &mut ScatteredVector<I>,
) {
    vector.clear_and_resize_values_with_known_non_zeros(size);
}

pub fn change_sign<I: VectorIndex>(data: &mut TypedVec<I, f64>) {
    for value in data.as_mut_slice() {
        *value = -*value;
    }
}

/// Maintains a sum that can omit one term while handling one-sided infinity.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumWithOneMissing<const POSITIVE_INFINITY: bool> {
    num_infinities: usize,
    sum: f64,
    error_sum: f64,
}

#[allow(clippy::float_cmp)]
impl<const POSITIVE_INFINITY: bool> SumWithOneMissing<POSITIVE_INFINITY> {
    #[must_use]
    const fn infinity() -> f64 {
        if POSITIVE_INFINITY {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    }

    pub fn add(&mut self, value: f64) {
        debug_assert!(!value.is_nan());
        if !value.is_finite() {
            debug_assert_eq!(value, Self::infinity());
            self.num_infinities += 1;
            return;
        }
        if !self.sum.is_finite() {
            return;
        }
        self.error_sum += value;
        let new_sum = self.sum + self.error_sum;
        self.error_sum += self.sum - new_sum;
        self.sum = new_sum;
    }

    pub fn remove_one_infinity(&mut self) {
        debug_assert!(self.num_infinities >= 1);
        self.num_infinities -= 1;
    }

    #[must_use]
    pub fn sum(&self) -> f64 {
        if self.num_infinities > 0 {
            Self::infinity()
        } else {
            self.sum
        }
    }

    #[must_use]
    pub fn sum_without(&self, value: f64) -> f64 {
        if value.is_finite() {
            if self.num_infinities > 0 {
                Self::infinity()
            } else {
                self.sum - value
            }
        } else {
            debug_assert_eq!(value, Self::infinity());
            if self.num_infinities > 1 {
                Self::infinity()
            } else {
                self.sum
            }
        }
    }

    #[must_use]
    pub fn sum_without_lb(&self, value: f64) -> f64 {
        if value.is_finite() {
            self.sum_without(value) - value.abs() * 1e-12
        } else {
            self.sum_without(value)
        }
    }

    #[must_use]
    pub fn sum_without_ub(&self, value: f64) -> f64 {
        if value.is_finite() {
            self.sum_without(value) + value.abs() * 1e-12
        } else {
            self.sum_without(value)
        }
    }
}

pub type SumWithPositiveInfiniteAndOneMissing = SumWithOneMissing<true>;
pub type SumWithNegativeInfiniteAndOneMissing = SumWithOneMissing<false>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn reductions_preserve_upstream_operation_order() {
        let values = [1e100, 3.0, 4.0, 5.0, 1e-100, 6.0, 7.0];
        assert_eq!(squared_norm(&values), 1e200);
        assert_eq!(precise_squared_norm(&values), 1e200);

        let left = [1e11, 1.0, 1.0, 1.0, 1e11, 1.0, 1.0, 1.0];
        let right = [1.0, 2e-6, 2e-6, 2e-6, -1.0, 2e-6, 2e-6, 2e-6];
        assert_eq!(scalar_product(&left, &right), 0.0);
        assert_eq!(precise_scalar_product(&left, &right), 0.000_006);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn sparse_dense_and_reset_utilities_match_upstream_semantics() {
        let mut sparse = SparseColumn::new();
        sparse.add_entry(RowIndex::new(1), -3.0);
        sparse.add_entry(RowIndex::new(3), 4.0);
        let dense = [10.0, 2.0, 30.0, 5.0];
        assert_eq!(sparse_scalar_product(&dense, &sparse), 14.0);
        assert_eq!(partial_scalar_product(&dense, &sparse, 3), -6.0);
        assert_eq!(sparse_squared_norm(&sparse), 25.0);
        assert_eq!(sparse_infinity_norm(&sparse), 4.0);
        assert_eq!(
            restricted_infinity_norm(&sparse, &[true, true, true, false]),
            (3.0, Some(RowIndex::new(1)))
        );
        assert!(is_dominated(&sparse, &[0.0, 3.0, 0.0, 4.0]));

        let mut values = [3.0, 4.0, 0.0];
        assert_eq!(squared_norm_and_reset_to_zero(&mut values), 25.0);
        assert_eq!(values, [0.0; 3]);
        let mut near = [1e-4, -1e-5, 0.0];
        remove_near_zero_entries(1e-4, &mut near);
        assert_eq!(near, [1e-4, 0.0, 0.0]);
        assert_eq!(density(&near), 1.0 / 3.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn permutation_support_and_one_missing_sum_match_upstream_semantics() {
        use crate::lp_types::TypedVec;
        use crate::permutation::RowPermutation;

        let permutation =
            RowPermutation::from_vec(vec![RowIndex::new(2), RowIndex::new(0), RowIndex::new(1)]);
        let mut values = TypedVec::from_vec(vec![3.0, 0.0, -4.0]);
        let mut support = Vec::new();
        compute_non_zeros(&values, &mut support);
        assert_eq!(support, [RowIndex::new(0), RowIndex::new(2)]);
        let mut scratch = TypedVec::filled(RowIndex::new(3), 0.0);
        permute_with_known_non_zeros(&permutation, &mut scratch, &mut values, &mut support);
        assert_eq!(values.as_slice(), &[0.0, -4.0, 3.0]);
        assert_eq!(support, [RowIndex::new(2), RowIndex::new(1)]);
        assert!(is_all_zero(&scratch));

        let mut sum = SumWithPositiveInfiniteAndOneMissing::default();
        sum.add(1e100);
        sum.add(1.0);
        sum.add(f64::INFINITY);
        assert_eq!(sum.sum(), f64::INFINITY);
        assert_eq!(sum.sum_without(f64::INFINITY), 1e100);
        sum.remove_one_infinity();
        assert_eq!(sum.sum(), 1e100);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn sparse_clear_resize_preserves_upstream_temporary_mask_protocol() {
        let mut vector = ScatteredColumn::new(RowIndex::new(100));
        vector.add(RowIndex::new(7), 2.0);
        vector.sort_non_zeros_if_needed();
        clear_and_resize_vector_with_non_zeros(RowIndex::new(100), &mut vector);
        assert_eq!(vector.value(RowIndex::new(7)), 0.0);
        assert!(vector.non_zeros().is_empty());

        // Upstream deliberately leaves is_non_zero untouched. Until the
        // caller clears/repopulates that temporary cache, another Add() at the
        // old position changes the value without recording the position.
        vector.add(RowIndex::new(7), 1.0);
        assert_eq!(vector.value(RowIndex::new(7)), 1.0);
        assert!(vector.non_zeros().is_empty());
    }
}
