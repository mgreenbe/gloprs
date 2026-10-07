//! Dense values with a sparse superset of touched indices.
//!
//! This ports the reusable workspace representation in upstream
//! `ortools/lp_data/scattered_vector.h` while clearing only touched entries.

use crate::lp_types::{ColIndex, Fractional, RowIndex, TypedVec, VectorIndex};

#[derive(Clone, Debug)]
pub struct ScatteredVector<I: VectorIndex + Ord> {
    values: TypedVec<I, Fractional>,
    non_zeros: Vec<I>,
    touched: TypedVec<I, bool>,
    non_zeros_are_sorted: bool,
}

impl<I: VectorIndex + Ord> ScatteredVector<I> {
    #[must_use]
    pub fn new(size: I) -> Self {
        Self {
            values: TypedVec::filled(size, 0.0),
            non_zeros: Vec::new(),
            touched: TypedVec::filled(size, false),
            non_zeros_are_sorted: true,
        }
    }

    #[must_use]
    pub fn len(&self) -> I {
        self.values.len()
    }

    pub fn add(&mut self, index: I, value: Fractional) {
        self.values[index] += value;
        if value != 0.0 && !self.touched[index] {
            self.touched[index] = true;
            self.non_zeros.push(index);
            self.non_zeros_are_sorted = false;
        }
    }

    pub fn set(&mut self, index: I, value: Fractional) {
        if value != 0.0 && !self.touched[index] {
            self.touched[index] = true;
            self.non_zeros.push(index);
            self.non_zeros_are_sorted = false;
        }
        self.values[index] = value;
    }

    #[must_use]
    pub fn value(&self, index: I) -> Fractional {
        self.values[index]
    }

    #[must_use]
    pub fn non_zeros(&self) -> &[I] {
        &self.non_zeros
    }

    pub fn sort_non_zeros_if_needed(&mut self) {
        if !self.non_zeros_are_sorted {
            self.non_zeros.sort_unstable();
            self.non_zeros_are_sorted = true;
        }
    }

    pub fn clear(&mut self) {
        for &index in &self.non_zeros {
            self.values[index] = 0.0;
            self.touched[index] = false;
        }
        self.non_zeros.clear();
        self.non_zeros_are_sorted = true;
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn should_use_dense_iteration(&self, ratio: f64) -> bool {
        self.non_zeros.is_empty()
            || self.non_zeros.len() as f64 > ratio * self.values.len().to_usize() as f64
    }
}

pub type ScatteredColumn = ScatteredVector<RowIndex>;
pub type ScatteredRow = ScatteredVector<ColIndex>;

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn clear_resets_only_touched_positions() {
        let mut vector = ScatteredColumn::new(RowIndex::new(100));
        vector.add(RowIndex::new(7), 2.0);
        vector.add(RowIndex::new(7), -2.0);
        assert_eq!(vector.non_zeros(), &[RowIndex::new(7)]);
        vector.clear();
        assert_eq!(vector.value(RowIndex::new(7)), 0.0);
        assert!(vector.non_zeros().is_empty());
    }
}
