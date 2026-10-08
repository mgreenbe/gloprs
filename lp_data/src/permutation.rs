//! Typed row and column permutations.
//!
//! This directly follows upstream `ortools/lp_data/permutation.h`: `p[i]` is
//! the destination of index `i`, and an empty permutation denotes identity.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use rand::seq::SliceRandom;

use crate::lp_types::{ColIndex, RowIndex, TypedVec, VectorIndex};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Permutation<I> {
    values: Vec<I>,
    index: PhantomData<fn(I) -> I>,
}

impl<I: VectorIndex> Permutation<I> {
    #[must_use]
    pub fn new(size: I) -> Self {
        Self {
            values: vec![I::from_usize(0); size.to_usize()],
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn from_vec(values: Vec<I>) -> Self {
        Self {
            values,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn len(&self) -> I {
        I::from_usize(self.values.len())
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn resize(&mut self, size: I, value: I) {
        self.values.resize(size.to_usize(), value);
    }

    pub fn assign(&mut self, size: I, value: I) {
        self.values = vec![value; size.to_usize()];
    }

    #[must_use]
    pub fn as_slice(&self) -> &[I] {
        &self.values
    }

    pub fn populate_identity(&mut self) {
        for (position, value) in self.values.iter_mut().enumerate() {
            *value = I::from_usize(position);
        }
    }

    /// Populates this object with a uniformly shuffled permutation, replacing
    /// GLOP's `absl::BitGen()` with Rust's thread-local random generator.
    pub fn populate_randomly(&mut self) {
        self.populate_identity();
        self.values.shuffle(&mut rand::rng());
    }

    pub fn populate_from_inverse(&mut self, inverse: &Self) {
        self.values.resize(inverse.values.len(), I::from_usize(0));
        for (position, &destination) in inverse.values.iter().enumerate() {
            self.values[destination.to_usize()] = I::from_usize(position);
        }
    }

    #[must_use]
    pub fn check(&self) -> bool {
        let mut visited = vec![false; self.values.len()];
        for &destination in &self.values {
            let Ok(position) = usize::try_from(destination.value_i64()) else {
                return false;
            };
            if position >= visited.len() || visited[position] {
                return false;
            }
            visited[position] = true;
        }
        true
    }

    #[must_use]
    pub fn signature(&self) -> i32 {
        debug_assert!(self.check());
        let mut visited = vec![false; self.values.len()];
        let mut signature = 1;
        for start in 0..self.values.len() {
            if visited[start] {
                continue;
            }
            let mut size = 0;
            let mut current = start;
            loop {
                visited[current] = true;
                current = self.values[current].to_usize();
                size += 1;
                if current == start {
                    break;
                }
            }
            if size % 2 == 0 {
                signature = -signature;
            }
        }
        signature
    }

    #[must_use]
    pub fn apply<T: Clone>(&self, input: &TypedVec<I, T>) -> TypedVec<I, T> {
        let mut output = TypedVec::new();
        apply_permutation(self, input, &mut output);
        output
    }

    #[must_use]
    pub fn apply_inverse<T: Clone>(&self, input: &TypedVec<I, T>) -> TypedVec<I, T> {
        let mut output = TypedVec::new();
        apply_inverse_permutation(self, input, &mut output);
        output
    }
}

/// Applies `permutation` to a typed vector, overwriting `result` as upstream's
/// `ApplyPermutation()` does. The permutation and vector may use different
/// strong index types.
///
/// # Panics
///
/// Panics if a nonempty permutation and the input have different lengths.
pub fn apply_permutation<P: VectorIndex, V: VectorIndex, T: Clone>(
    permutation: &Permutation<P>,
    input: &TypedVec<V, T>,
    result: &mut TypedVec<V, T>,
) {
    if permutation.is_empty() {
        *result = input.clone();
        return;
    }
    assert_eq!(permutation.as_slice().len(), input.as_slice().len());
    let junk = input.as_slice().last().unwrap().clone();
    result.resize(input.len(), junk);
    for (source, &destination) in permutation.as_slice().iter().enumerate() {
        result[V::from_usize(destination.to_usize())] = input[V::from_usize(source)].clone();
    }
}

/// Applies the inverse of `permutation`, overwriting `result` and accepting a
/// vector with a different strong index type.
///
/// # Panics
///
/// Panics if a nonempty permutation and the input have different lengths.
pub fn apply_inverse_permutation<P: VectorIndex, V: VectorIndex, T: Clone>(
    permutation: &Permutation<P>,
    input: &TypedVec<V, T>,
    result: &mut TypedVec<V, T>,
) {
    if permutation.is_empty() {
        *result = input.clone();
        return;
    }
    assert_eq!(permutation.as_slice().len(), input.as_slice().len());
    let junk = input.as_slice().last().unwrap().clone();
    result.resize(input.len(), junk);
    for (source, &destination) in permutation.as_slice().iter().enumerate() {
        result[V::from_usize(source)] = input[V::from_usize(destination.to_usize())].clone();
    }
}

/// Applies a column permutation to storage whose logical indices are rows.
/// This is the cross-index specialization provided by pinned GLOP.
///
/// # Panics
///
/// Panics when a nonempty permutation and the vector have different sizes.
pub fn apply_column_permutation_to_row_indexed_vector<T: Clone>(
    permutation: &ColumnPermutation,
    values: &mut TypedVec<RowIndex, T>,
    temporary: &mut TypedVec<RowIndex, T>,
) {
    if permutation.is_empty() {
        return;
    }
    assert_eq!(permutation.as_slice().len(), values.as_slice().len());
    let Some(junk) = values.as_slice().last().cloned() else {
        return;
    };
    temporary.resize(values.len(), junk);
    for (source, &destination) in permutation.as_slice().iter().enumerate() {
        temporary[RowIndex::from_usize(destination.to_usize())] =
            values[RowIndex::from_usize(source)].clone();
    }
    std::mem::swap(temporary, values);
}

impl<I: VectorIndex> Index<I> for Permutation<I> {
    type Output = I;

    fn index(&self, index: I) -> &Self::Output {
        &self.values[index.to_usize()]
    }
}

impl<I: VectorIndex> IndexMut<I> for Permutation<I> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.values[index.to_usize()]
    }
}

pub type RowPermutation = Permutation<RowIndex>;
pub type ColumnPermutation = Permutation<ColIndex>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_application_and_signature_match_cycle_structure() {
        let permutation = ColumnPermutation::from_vec(vec![
            ColIndex::new(1),
            ColIndex::new(2),
            ColIndex::new(0),
            ColIndex::new(3),
        ]);
        assert_eq!(permutation.signature(), 1);
        let input = TypedVec::from_vec(vec![10, 20, 30, 40]);
        assert_eq!(permutation.apply(&input).as_slice(), &[30, 10, 20, 40]);
        assert_eq!(permutation.apply_inverse(&permutation.apply(&input)), input);
    }

    #[test]
    fn invalid_entries_return_false_and_cross_index_application_matches() {
        let invalid = RowPermutation::from_vec(vec![RowIndex::new(0), RowIndex::new(-1)]);
        assert!(!invalid.check());

        let permutation =
            ColumnPermutation::from_vec(vec![ColIndex::new(2), ColIndex::new(0), ColIndex::new(1)]);
        let mut values = TypedVec::from_vec(vec![10, 20, 30]);
        let mut temporary = TypedVec::new();
        apply_column_permutation_to_row_indexed_vector(&permutation, &mut values, &mut temporary);
        assert_eq!(values.as_slice(), &[20, 30, 10]);

        let input = TypedVec::<RowIndex, _>::from_vec(vec![10, 20, 30]);
        let mut output = TypedVec::new();
        apply_permutation(&permutation, &input, &mut output);
        assert_eq!(output.as_slice(), &[20, 30, 10]);
        apply_inverse_permutation(&permutation, &output, &mut values);
        assert_eq!(values.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn random_population_always_produces_a_permutation() {
        for size in 0..32 {
            let mut permutation = RowPermutation::new(RowIndex::from_usize(size));
            permutation.populate_randomly();
            assert!(permutation.check());
        }
    }
}
