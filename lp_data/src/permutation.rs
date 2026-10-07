//! Typed row and column permutations.
//!
//! This directly follows upstream `ortools/lp_data/permutation.h`: `p[i]` is
//! the destination of index `i`, and an empty permutation denotes identity.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

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

    pub fn populate_identity(&mut self) {
        for (position, value) in self.values.iter_mut().enumerate() {
            *value = I::from_usize(position);
        }
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
            let position = destination.to_usize();
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
        if self.is_empty() {
            return input.clone();
        }
        debug_assert_eq!(self.values.len(), input.as_slice().len());
        let mut output = input.clone();
        for (source, &destination) in self.values.iter().enumerate() {
            output[destination] = input[I::from_usize(source)].clone();
        }
        output
    }

    #[must_use]
    pub fn apply_inverse<T: Clone>(&self, input: &TypedVec<I, T>) -> TypedVec<I, T> {
        if self.is_empty() {
            return input.clone();
        }
        debug_assert_eq!(self.values.len(), input.as_slice().len());
        (0..self.values.len())
            .map(|position| input[self.values[position]].clone())
            .collect()
    }
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
}
