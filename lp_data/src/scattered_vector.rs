//! Dense values with a sparse superset of touched indices.
//!
//! This ports the reusable workspace representation in upstream
//! `ortools/lp_data/scattered_vector.h` while clearing only touched entries.

use std::marker::PhantomData;

use crate::lp_types::{BitVec, ColIndex, Fractional, RowIndex, TypedVec, VectorIndex};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatteredEntry<I> {
    index: I,
    coefficient: Fractional,
}

impl<I: Copy> ScatteredEntry<I> {
    #[must_use]
    pub const fn index(self) -> I {
        self.index
    }

    #[must_use]
    pub const fn coefficient(self) -> Fractional {
        self.coefficient
    }
}

impl ScatteredEntry<RowIndex> {
    #[must_use]
    pub const fn row(self) -> RowIndex {
        self.index
    }
}

impl ScatteredEntry<ColIndex> {
    #[must_use]
    pub const fn column(self) -> ColIndex {
        self.index
    }
}

#[derive(Clone, Debug)]
pub struct ScatteredIter<'a, I> {
    positions: std::slice::Iter<'a, I>,
    values: &'a TypedVec<I, Fractional>,
}

impl<I: VectorIndex> Iterator for ScatteredIter<'_, I> {
    type Item = ScatteredEntry<I>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = *self.positions.next()?;
        Some(ScatteredEntry {
            index,
            coefficient: self.values[index],
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.positions.size_hint()
    }
}

impl<I: VectorIndex> DoubleEndedIterator for ScatteredIter<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = *self.positions.next_back()?;
        Some(ScatteredEntry {
            index,
            coefficient: self.values[index],
        })
    }
}

impl<I: VectorIndex> ExactSizeIterator for ScatteredIter<'_, I> {}

#[derive(Clone, Debug)]
pub struct ScatteredVector<I: VectorIndex + Ord> {
    values: TypedVec<I, Fractional>,
    non_zeros: Vec<I>,
    is_non_zero: BitVec<I>,
    non_zeros_are_sorted: bool,
}

impl<I: VectorIndex + Ord> ScatteredVector<I> {
    #[must_use]
    pub fn new(size: I) -> Self {
        Self {
            values: TypedVec::filled(size, 0.0),
            non_zeros: Vec::new(),
            is_non_zero: BitVec::new(size),
            non_zeros_are_sorted: false,
        }
    }

    #[must_use]
    pub fn len(&self) -> I {
        self.values.len()
    }

    pub fn add(&mut self, index: I, value: Fractional) {
        self.values[index] += value;
        if value != 0.0 && !self.is_non_zero.contains(index) {
            self.is_non_zero.set(index);
            self.non_zeros.push(index);
            self.non_zeros_are_sorted = false;
        }
    }

    pub fn set(&mut self, index: I, value: Fractional) {
        if value != 0.0 && !self.is_non_zero.contains(index) {
            self.is_non_zero.set(index);
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

    #[must_use]
    pub fn values(&self) -> &TypedVec<I, Fractional> {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut TypedVec<I, Fractional> {
        &mut self.values
    }

    pub fn non_zeros_mut(&mut self) -> &mut Vec<I> {
        self.non_zeros_are_sorted = false;
        &mut self.non_zeros
    }

    pub fn mutable_parts(&mut self) -> (&mut [Fractional], &mut Vec<I>) {
        // These fields are public in GLOP's `ScatteredVector`; taking mutable
        // references does not itself invalidate its temporary sortedness flag.
        // The low-level solve routines preserve that behavior deliberately.
        (self.values.as_mut_slice(), &mut self.non_zeros)
    }

    /// Records that the position list is sorted.
    ///
    /// GLOP exposes this temporary cache flag to its sparse kernels.  Keep its
    /// updates explicit so that subsequent reductions use the same traversal
    /// order as upstream.
    pub fn mark_non_zeros_sorted(&mut self) {
        self.non_zeros_are_sorted = true;
    }

    /// Records that the position list may no longer be sorted.
    pub fn mark_non_zeros_unsorted(&mut self) {
        self.non_zeros_are_sorted = false;
    }

    #[must_use]
    pub fn iter(&self) -> ScatteredIter<'_, I> {
        ScatteredIter {
            positions: self.non_zeros.iter(),
            values: &self.values,
        }
    }

    pub fn sort_non_zeros_if_needed(&mut self) {
        if !self.non_zeros_are_sorted {
            self.non_zeros.sort_unstable();
            self.non_zeros_are_sorted = true;
        }
    }

    pub fn clear(&mut self) {
        if self.non_zeros.is_empty() {
            self.values.as_mut_slice().fill(0.0);
            self.is_non_zero.clear_and_resize(self.values.len());
        } else {
            for &index in &self.non_zeros {
                self.values[index] = 0.0;
                self.is_non_zero.clear_bit(index);
            }
        }
        self.non_zeros.clear();
        self.non_zeros_are_sorted = false;
    }

    /// Implements GLOP's `ClearAndResizeVectorWithNonZeros()` protocol.
    ///
    /// Like upstream, this clears `values` and `non_zeros` but deliberately
    /// leaves the temporary membership mask and sorted flag untouched. Its
    /// callers clear or repopulate that cache at the same points as GLOP.
    pub(crate) fn clear_and_resize_values_with_known_non_zeros(&mut self, size: I) {
        const SPARSE_THRESHOLD: f64 = 0.05;
        #[allow(clippy::cast_precision_loss)]
        let use_sparse_clear = !self.non_zeros.is_empty()
            && (self.non_zeros.len() as f64) < SPARSE_THRESHOLD * size.to_usize() as f64;
        if use_sparse_clear {
            for &index in &self.non_zeros {
                debug_assert!(index.to_usize() < self.values.len().to_usize());
                self.values[index] = 0.0;
            }
            self.values.resize(size, 0.0);
        } else {
            self.values.assign_to_zero(size);
        }
        self.non_zeros.clear();
    }

    /// Clears the sparse membership mask without changing values or positions.
    pub fn clear_sparse_mask(&mut self) {
        if self.should_use_dense_iteration(0.8) {
            self.is_non_zero.clear_and_resize(self.values.len());
        } else {
            self.is_non_zero.resize(self.values.len());
            for &index in &self.non_zeros {
                // GLOP clears complete 64-bit buckets here. The mask is an
                // explicitly temporary cache and may contain stale bits not
                // represented in `non_zeros`.
                self.is_non_zero.clear_bucket(index);
            }
        }
    }

    /// Makes the membership mask consistent with the recorded positions.
    pub fn repopulate_sparse_mask(&mut self) {
        self.clear_sparse_mask();
        for &index in &self.non_zeros {
            self.is_non_zero.set(index);
        }
    }

    /// Switches to upstream's dense convention (an empty position list) when
    /// sparse traversal would inspect too large a fraction of the vector.
    pub fn clear_non_zeros_if_too_dense(&mut self, ratio: f64) {
        if self.should_use_dense_iteration(ratio) {
            self.clear_sparse_mask();
            self.non_zeros.clear();
            self.non_zeros_are_sorted = false;
        }
    }

    #[must_use]
    pub fn num_non_zeros_estimate(&self) -> usize {
        if self.non_zeros.is_empty() {
            self.values.len().to_usize()
        } else {
            self.non_zeros.len()
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn should_use_dense_iteration(&self, ratio: f64) -> bool {
        self.non_zeros.is_empty()
            || self.non_zeros.len() as f64 > ratio * self.values.len().to_usize() as f64
    }
}

impl<'a, I: VectorIndex + Ord> IntoIterator for &'a ScatteredVector<I> {
    type Item = ScatteredEntry<I>;
    type IntoIter = ScatteredIter<'a, I>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Safe O(1) counterpart of GLOP's reinterpret-cast `TransposedView()`.
#[derive(Clone, Copy, Debug)]
pub struct TransposedScatteredView<'a, Target, Source> {
    values: &'a [Fractional],
    non_zeros: &'a [Source],
    target: PhantomData<Target>,
}

impl<'a, Target: VectorIndex, Source: VectorIndex> TransposedScatteredView<'a, Target, Source> {
    #[must_use]
    pub fn values(self) -> &'a [Fractional] {
        self.values
    }

    #[must_use]
    pub fn non_zeros(self) -> impl ExactSizeIterator<Item = Target> + 'a {
        self.non_zeros
            .iter()
            .map(|index| Target::from_usize(index.to_usize()))
    }

    #[must_use]
    pub fn iter(self) -> impl ExactSizeIterator<Item = ScatteredEntry<Target>> + 'a {
        self.non_zeros().map(move |index| ScatteredEntry {
            index,
            coefficient: self.values[index.to_usize()],
        })
    }
}

#[must_use]
pub fn transposed_row_view(
    column: &ScatteredColumn,
) -> TransposedScatteredView<'_, ColIndex, RowIndex> {
    TransposedScatteredView {
        values: column.values().as_slice(),
        non_zeros: column.non_zeros(),
        target: PhantomData,
    }
}

#[must_use]
pub fn transposed_column_view(
    row: &ScatteredRow,
) -> TransposedScatteredView<'_, RowIndex, ColIndex> {
    TransposedScatteredView {
        values: row.values().as_slice(),
        non_zeros: row.non_zeros(),
        target: PhantomData,
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

    #[test]
    fn sparse_mask_clear_resets_whole_touched_buckets() {
        let mut vector = ScatteredColumn::new(RowIndex::new(128));
        vector.add(RowIndex::new(1), 1.0);
        vector.add(RowIndex::new(2), 1.0);
        vector
            .non_zeros_mut()
            .retain(|&row| row == RowIndex::new(2));
        vector.repopulate_sparse_mask();
        vector.add(RowIndex::new(1), 1.0);
        assert_eq!(vector.non_zeros(), &[RowIndex::new(2), RowIndex::new(1)]);
    }
}
