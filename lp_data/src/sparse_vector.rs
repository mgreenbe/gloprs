//! Sparse vectors and columns.
//!
//! This ports the core behavior of upstream `sparse_vector.h` and
//! `sparse_column.{h,cc}`. Entries retain insertion order until cleanup;
//! cleanup sorts by index, removes zeros, and makes the last duplicate win.

use crate::lp_types::{ColIndex, DenseColumn, Fractional, RowIndex, TypedVec, VectorIndex};
use crate::permutation::Permutation;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseEntry<I> {
    index: I,
    coefficient: Fractional,
}

impl<I: Copy> SparseEntry<I> {
    #[must_use]
    pub const fn index(self) -> I {
        self.index
    }

    #[must_use]
    pub const fn coefficient(self) -> Fractional {
        self.coefficient
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseVector<I> {
    entries: Vec<SparseEntry<I>>,
    may_contain_duplicates: bool,
}

impl<I: VectorIndex + Ord> SparseVector<I> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            may_contain_duplicates: false,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.may_contain_duplicates = false;
    }

    pub fn clear_and_release(&mut self) {
        *self = Self::new();
    }

    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub const fn num_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn set_coefficient(&mut self, index: I, value: Fractional) {
        let _ = index.to_usize();
        self.entries.push(SparseEntry {
            index,
            coefficient: value,
        });
        self.may_contain_duplicates = true;
    }

    /// Appends an entry without searching for an existing index.
    pub fn add_entry(&mut self, index: I, value: Fractional) {
        let _ = index.to_usize();
        self.entries.push(SparseEntry {
            index,
            coefficient: value,
        });
        self.may_contain_duplicates = true;
    }

    /// Restores canonical order. A later duplicate overrides every earlier one.
    pub fn clean_up(&mut self) {
        self.entries.sort_by_key(|entry| entry.index);
        let mut write = 0;
        let mut read = 0;
        while read < self.entries.len() {
            let mut last = read;
            while last + 1 < self.entries.len()
                && self.entries[last + 1].index == self.entries[read].index
            {
                last += 1;
            }
            let entry = self.entries[last];
            if entry.coefficient != 0.0 {
                self.entries[write] = entry;
                write += 1;
            }
            read = last + 1;
        }
        self.entries.truncate(write);
        self.may_contain_duplicates = false;
    }

    #[must_use]
    pub fn is_cleaned_up(&self) -> bool {
        self.entries.iter().all(|entry| entry.coefficient != 0.0)
            && self
                .entries
                .windows(2)
                .all(|pair| pair[0].index < pair[1].index)
    }

    #[must_use]
    pub fn check_no_duplicates(&self) -> bool {
        if !self.may_contain_duplicates || self.entries.len() <= 1 {
            return true;
        }
        let mut seen = vec![false; self.max_index_plus_one()];
        for entry in &self.entries {
            let position = entry.index.to_usize();
            if seen[position] {
                return false;
            }
            seen[position] = true;
        }
        true
    }

    fn max_index_plus_one(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.index.to_usize())
            .max()
            .map_or(0, |maximum| maximum.saturating_add(1))
    }

    pub fn delete_entry(&mut self, index: I) {
        debug_assert!(self.check_no_duplicates());
        if let Some(position) = self.entries.iter().position(|entry| entry.index == index) {
            self.entries.remove(position);
        }
    }

    pub fn remove_near_zero_entries(&mut self, threshold: Fractional) {
        debug_assert!(self.check_no_duplicates());
        self.entries
            .retain(|entry| entry.coefficient.abs() > threshold);
    }

    pub fn remove_near_zero_entries_with_weights(
        &mut self,
        threshold: Fractional,
        weights: &TypedVec<I, Fractional>,
    ) {
        debug_assert!(self.check_no_duplicates());
        self.entries
            .retain(|entry| entry.coefficient.abs() * weights[entry.index] > threshold);
    }

    pub fn multiply_by_constant(&mut self, factor: Fractional) {
        for entry in &mut self.entries {
            entry.coefficient *= factor;
        }
    }

    pub fn component_wise_multiply(&mut self, factors: &TypedVec<I, Fractional>) {
        for entry in &mut self.entries {
            entry.coefficient *= factors[entry.index];
        }
    }

    pub fn divide_by_constant(&mut self, factor: Fractional) {
        for entry in &mut self.entries {
            entry.coefficient /= factor;
        }
    }

    pub fn component_wise_divide(&mut self, factors: &TypedVec<I, Fractional>) {
        for entry in &mut self.entries {
            entry.coefficient /= factors[entry.index];
        }
    }

    pub fn populate_from_dense(&mut self, dense: &TypedVec<I, Fractional>) {
        self.clear();
        for (position, &value) in dense.as_slice().iter().enumerate() {
            if value != 0.0 {
                self.add_entry(I::from_usize(position), value);
            }
        }
        self.may_contain_duplicates = false;
    }

    /// Appends entries after shifting their logical indices.
    ///
    /// # Panics
    ///
    /// Panics if a shifted index is negative or cannot be represented.
    pub fn append_entries_with_offset(&mut self, source: &Self, offset: i32) {
        for entry in &source.entries {
            let shifted =
                i64::from(i32::try_from(entry.index.to_usize()).expect("sparse index exceeds i32"))
                    + i64::from(offset);
            let shifted = usize::try_from(shifted).expect("shifted sparse index is negative");
            self.set_coefficient(I::from_usize(shifted), entry.coefficient);
        }
    }

    #[must_use]
    pub fn scalar_product(&self, dense: &TypedVec<I, Fractional>) -> Fractional {
        self.entries
            .iter()
            .map(|entry| entry.coefficient * dense[entry.index])
            .sum()
    }

    #[must_use]
    pub fn first(&self) -> Option<SparseEntry<I>> {
        self.entries.first().copied()
    }

    #[must_use]
    pub fn last(&self) -> Option<SparseEntry<I>> {
        self.entries.last().copied()
    }

    pub fn move_entry_to_first_position(&mut self, index: I) {
        if let Some(position) = self.entries.iter().position(|entry| entry.index == index) {
            self.entries.swap(0, position);
        }
    }

    pub fn move_entry_to_last_position(&mut self, index: I) {
        if let Some(position) = self.entries.iter().position(|entry| entry.index == index) {
            let last = self.entries.len() - 1;
            self.entries.swap(last, position);
        }
    }

    #[must_use]
    pub fn look_up_coefficient(&self, index: I) -> Fractional {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.index == index)
            .map_or(0.0, |entry| entry.coefficient)
    }

    pub fn copy_to_dense_vector(&self, size: I, dense: &mut TypedVec<I, Fractional>) {
        *dense = TypedVec::filled(size, 0.0);
        for entry in &self.entries {
            dense[entry.index] = entry.coefficient;
        }
    }

    pub fn add_multiple_to_dense_vector(
        &self,
        multiplier: Fractional,
        dense: &mut TypedVec<I, Fractional>,
    ) {
        if multiplier == 0.0 {
            return;
        }
        for entry in &self.entries {
            dense[entry.index] += multiplier * entry.coefficient;
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SparseEntry<I>> {
        self.entries.iter()
    }

    pub fn apply_index_permutation(&mut self, permutation: &Permutation<I>) {
        if permutation.is_empty() {
            return;
        }
        for entry in &mut self.entries {
            entry.index = permutation[entry.index];
        }
        self.may_contain_duplicates = true;
    }

    /// Adds a clean sparse vector into this clean vector in linear time.
    pub fn add_multiple_to_sparse_vector(
        &self,
        multiplier: Fractional,
        drop_tolerance: Fractional,
        accumulator: &mut Self,
    ) {
        debug_assert!(self.is_cleaned_up());
        debug_assert!(accumulator.is_cleaned_up());
        let mut merged = Vec::with_capacity(self.entries.len() + accumulator.entries.len());
        let (mut left, mut right) = (0, 0);
        while left < accumulator.entries.len() || right < self.entries.len() {
            let entry = if right == self.entries.len()
                || (left < accumulator.entries.len()
                    && accumulator.entries[left].index < self.entries[right].index)
            {
                let entry = accumulator.entries[left];
                left += 1;
                entry
            } else if left == accumulator.entries.len()
                || self.entries[right].index < accumulator.entries[left].index
            {
                let mut entry = self.entries[right];
                entry.coefficient *= multiplier;
                right += 1;
                entry
            } else {
                let entry = SparseEntry {
                    index: accumulator.entries[left].index,
                    coefficient: accumulator.entries[left].coefficient
                        + multiplier * self.entries[right].coefficient,
                };
                left += 1;
                right += 1;
                entry
            };
            if entry.coefficient.abs() > drop_tolerance {
                merged.push(entry);
            }
        }
        accumulator.entries = merged;
        accumulator.may_contain_duplicates = false;
    }

    pub fn apply_partial_index_permutation(&mut self, mapping: &TypedVec<I, Option<I>>) {
        self.entries = self
            .entries
            .iter()
            .filter_map(|entry| {
                mapping[entry.index].map(|index| SparseEntry {
                    index,
                    coefficient: entry.coefficient,
                })
            })
            .collect();
        self.may_contain_duplicates = true;
    }

    pub fn move_tagged_entries_to(&mut self, tagged: &TypedVec<I, bool>, output: &mut Self) {
        let mut retained = Vec::with_capacity(self.entries.len());
        for entry in self.entries.drain(..) {
            if tagged[entry.index] {
                output.set_coefficient(entry.index, entry.coefficient);
            } else {
                retained.push(entry);
            }
        }
        self.entries = retained;
    }

    #[must_use]
    pub fn entries(&self) -> &[SparseEntry<I>] {
        &self.entries
    }
}

impl<'a, I> IntoIterator for &'a SparseVector<I> {
    type Item = &'a SparseEntry<I>;
    type IntoIter = std::slice::Iter<'a, SparseEntry<I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

pub type SparseColumn = SparseVector<RowIndex>;
pub type ColumnView<'a> = &'a [SparseEntry<RowIndex>];
pub type RowView<'a> = &'a [SparseEntry<ColIndex>];

#[derive(Clone, Debug)]
pub struct RandomAccessSparseColumn {
    column: DenseColumn,
    changed: TypedVec<RowIndex, bool>,
    changed_rows: Vec<RowIndex>,
}

impl RandomAccessSparseColumn {
    #[must_use]
    pub fn new(num_rows: RowIndex) -> Self {
        Self {
            column: DenseColumn::filled(num_rows, 0.0),
            changed: TypedVec::filled(num_rows, false),
            changed_rows: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        for &row in &self.changed_rows {
            self.column[row] = 0.0;
            self.changed[row] = false;
        }
        self.changed_rows.clear();
    }

    pub fn resize(&mut self, num_rows: RowIndex) {
        if num_rows > self.column.len() {
            self.column.resize(num_rows, 0.0);
            self.changed.resize(num_rows, false);
        }
    }

    pub fn set_coefficient(&mut self, row: RowIndex, value: Fractional) {
        self.column[row] = value;
        self.mark_row_as_changed(row);
    }

    pub fn add_to_coefficient(&mut self, row: RowIndex, value: Fractional) {
        self.column[row] += value;
        self.mark_row_as_changed(row);
    }

    fn mark_row_as_changed(&mut self, row: RowIndex) {
        if !self.changed[row] {
            self.changed[row] = true;
            self.changed_rows.push(row);
        }
    }

    pub fn populate_from_sparse_column(&mut self, sparse: &SparseColumn) {
        self.clear();
        for entry in sparse {
            self.set_coefficient(entry.index(), entry.coefficient());
        }
    }

    pub fn populate_sparse_column(&self, sparse: &mut SparseColumn) {
        sparse.clear();
        for &row in &self.changed_rows {
            sparse.set_coefficient(row, self.column[row]);
        }
    }

    #[must_use]
    pub fn num_rows(&self) -> RowIndex {
        self.column.len()
    }

    #[must_use]
    pub fn coefficient(&self, row: RowIndex) -> Fractional {
        self.column[row]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_sorts_removes_zeros_and_keeps_last_duplicate() {
        let mut vector = SparseColumn::new();
        vector.set_coefficient(RowIndex::new(2), 3.0);
        vector.set_coefficient(RowIndex::new(0), 1.0);
        vector.set_coefficient(RowIndex::new(2), 4.0);
        vector.set_coefficient(RowIndex::new(1), 0.0);

        assert!(!vector.check_no_duplicates());
        vector.clean_up();

        assert!(vector.is_cleaned_up());
        assert_eq!(vector.num_entries(), 2);
        assert_eq!(
            vector.look_up_coefficient(RowIndex::new(0)).to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            vector.look_up_coefficient(RowIndex::new(2)).to_bits(),
            4.0_f64.to_bits()
        );
    }

    #[test]
    fn random_access_column_clears_only_touched_rows() {
        let mut column = RandomAccessSparseColumn::new(RowIndex::new(5));
        column.add_to_coefficient(RowIndex::new(3), 2.0);
        column.add_to_coefficient(RowIndex::new(3), 1.0);

        let mut sparse = SparseColumn::new();
        column.populate_sparse_column(&mut sparse);
        assert_eq!(sparse.num_entries(), 1);
        assert_eq!(
            sparse.look_up_coefficient(RowIndex::new(3)).to_bits(),
            3.0_f64.to_bits()
        );

        column.clear();
        assert_eq!(
            column.coefficient(RowIndex::new(3)).to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn sparse_merge_is_linear_ordered_and_drops_small_results() {
        let mut source = SparseColumn::new();
        source.set_coefficient(RowIndex::new(0), 2.0);
        source.set_coefficient(RowIndex::new(2), -1.0);
        source.clean_up();
        let mut accumulator = SparseColumn::new();
        accumulator.set_coefficient(RowIndex::new(1), 3.0);
        accumulator.set_coefficient(RowIndex::new(2), 1.0);
        accumulator.clean_up();
        source.add_multiple_to_sparse_vector(1.0, 0.0, &mut accumulator);
        assert_eq!(accumulator.num_entries(), 2);
        assert_eq!(accumulator.entries()[0].index(), RowIndex::new(0));
        assert_eq!(accumulator.entries()[1].index(), RowIndex::new(1));
    }
}
