//! Sparse vectors and columns.
//!
//! This ports the core behavior of upstream `sparse_vector.h` and
//! `sparse_column.{h,cc}`. Entries retain insertion order until cleanup;
//! cleanup sorts by index, removes zeros, and makes the last duplicate win.

use std::cell::Cell;

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

impl SparseEntry<RowIndex> {
    #[must_use]
    pub const fn row(self) -> RowIndex {
        self.index
    }
}

impl SparseEntry<ColIndex> {
    #[must_use]
    pub const fn col(self) -> ColIndex {
        self.index
    }
}

#[derive(Clone, Debug, Default)]
pub struct SparseVector<I> {
    // GLOP keeps indices and coefficients in two contiguous regions of one
    // allocation. Rust cannot safely split one untyped allocation this way,
    // so use two parallel allocations while preserving the important SoA
    // traversal and cache behavior.
    indices: Vec<I>,
    coefficients: Vec<Fractional>,
    may_contain_duplicates: Cell<bool>,
}

#[derive(Clone, Debug)]
pub struct SparseVectorIter<'a, I> {
    indices: std::slice::Iter<'a, I>,
    coefficients: std::slice::Iter<'a, Fractional>,
}

impl<I: Copy> Iterator for SparseVectorIter<'_, I> {
    type Item = SparseEntry<I>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(SparseEntry {
            index: *self.indices.next()?,
            coefficient: *self.coefficients.next()?,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<I: Copy> DoubleEndedIterator for SparseVectorIter<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(SparseEntry {
            index: *self.indices.next_back()?,
            coefficient: *self.coefficients.next_back()?,
        })
    }
}

impl<I: Copy> ExactSizeIterator for SparseVectorIter<'_, I> {}

impl<I: VectorIndex + Ord> SparseVector<I> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            indices: Vec::new(),
            coefficients: Vec::new(),
            may_contain_duplicates: Cell::new(false),
        }
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.coefficients.clear();
        self.may_contain_duplicates.set(false);
    }

    pub fn clear_and_release(&mut self) {
        *self = Self::new();
    }

    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    /// Ensures capacity for at least `new_capacity` total entries.
    ///
    /// This deliberately follows GLOP's `Reserve()` contract rather than
    /// `Vec::reserve()`'s additional-capacity contract.
    pub fn reserve(&mut self, new_capacity: usize) {
        if self.indices.capacity() < new_capacity {
            self.indices.reserve(new_capacity - self.indices.len());
        }
        if self.coefficients.capacity() < new_capacity {
            self.coefficients
                .reserve(new_capacity - self.coefficients.len());
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[must_use]
    pub const fn num_entries(&self) -> usize {
        self.indices.len()
    }

    /// Shrinks the logical entry range without releasing capacity.
    ///
    /// # Panics
    ///
    /// Panics if `new_size` exceeds the current number of entries.
    pub fn resize_down(&mut self, new_size: usize) {
        assert!(new_size <= self.num_entries());
        self.indices.truncate(new_size);
        self.coefficients.truncate(new_size);
    }

    #[must_use]
    pub fn index(&self, position: usize) -> I {
        self.indices[position]
    }

    #[must_use]
    pub fn coefficient(&self, position: usize) -> Fractional {
        self.coefficients[position]
    }

    pub fn mutable_index(&mut self, position: usize) -> &mut I {
        &mut self.indices[position]
    }

    pub fn mutable_coefficient(&mut self, position: usize) -> &mut Fractional {
        &mut self.coefficients[position]
    }

    #[must_use]
    pub fn first_index(&self) -> I {
        self.index(0)
    }

    #[must_use]
    pub fn first_coefficient(&self) -> Fractional {
        self.coefficient(0)
    }

    #[must_use]
    pub fn last_index(&self) -> I {
        self.index(self.num_entries() - 1)
    }

    #[must_use]
    pub fn last_coefficient(&self) -> Fractional {
        self.coefficient(self.num_entries() - 1)
    }

    pub fn set_coefficient(&mut self, index: I, value: Fractional) {
        let _ = index.to_usize();
        self.indices.push(index);
        self.coefficients.push(value);
        self.may_contain_duplicates.set(true);
    }

    /// Appends an entry without searching for an existing index.
    pub fn add_entry(&mut self, index: I, value: Fractional) {
        let _ = index.to_usize();
        self.indices.push(index);
        self.coefficients.push(value);
        self.may_contain_duplicates.set(true);
    }

    /// Restores canonical order. A later duplicate overrides every earlier one.
    pub fn clean_up(&mut self) {
        // This staging representation and stable sort deliberately mirror
        // upstream SparseVector::CleanUp().
        let mut entries: Vec<_> = self.iter().collect();
        entries.sort_by_key(|entry| entry.index);
        let mut write = 0;
        let mut read = 0;
        while read < entries.len() {
            let mut last = read;
            while last + 1 < entries.len() && entries[last + 1].index == entries[read].index {
                last += 1;
            }
            let entry = entries[last];
            if entry.coefficient != 0.0 {
                self.indices[write] = entry.index;
                self.coefficients[write] = entry.coefficient;
                write += 1;
            }
            read = last + 1;
        }
        self.indices.truncate(write);
        self.coefficients.truncate(write);
        self.may_contain_duplicates.set(false);
    }

    #[must_use]
    pub fn is_cleaned_up(&self) -> bool {
        let cleaned = self
            .coefficients
            .iter()
            .all(|&coefficient| coefficient != 0.0)
            && self.indices.windows(2).all(|pair| pair[0] < pair[1]);
        if cleaned {
            self.may_contain_duplicates.set(false);
        }
        cleaned
    }

    #[must_use]
    pub fn check_no_duplicates(&self) -> bool {
        let mut seen = TypedVec::new();
        self.check_no_duplicates_with_workspace(&mut seen)
    }

    /// Checks duplicate indices using reusable all-false scratch storage and
    /// restores that storage to all false before returning.
    #[must_use]
    pub fn check_no_duplicates_with_workspace(&self, seen: &mut TypedVec<I, bool>) -> bool {
        if !self.may_contain_duplicates.get() || self.indices.len() <= 1 {
            return true;
        }
        let required = self.max_index_plus_one();
        if seen.as_slice().len() < required {
            seen.resize(I::from_usize(required), false);
        }
        self.may_contain_duplicates.set(false);
        for &index in &self.indices {
            if seen[index] {
                self.may_contain_duplicates.set(true);
                break;
            }
            seen[index] = true;
        }
        for &index in &self.indices {
            seen[index] = false;
        }
        !self.may_contain_duplicates.get()
    }

    fn max_index_plus_one(&self) -> usize {
        self.indices
            .iter()
            .map(|index| index.to_usize())
            .max()
            .map_or(0, |maximum| maximum.saturating_add(1))
    }

    pub fn delete_entry(&mut self, index: I) {
        debug_assert!(self.check_no_duplicates());
        if let Some(position) = self
            .indices
            .iter()
            .position(|&entry_index| entry_index == index)
        {
            self.indices.remove(position);
            self.coefficients.remove(position);
        }
    }

    /// Moves entries whose row has a tag to `tagged` without changing their
    /// index. Untagged entries remain in place without allocating.
    pub fn move_entries_with_index_tags_to(
        &mut self,
        tags: &[usize],
        invalid_tag: usize,
        tagged: &mut Self,
    ) {
        let mut write = 0;
        for read in 0..self.indices.len() {
            let entry = self.entry(read);
            let tag = tags[entry.index.to_usize()];
            if tag == invalid_tag {
                self.indices[write] = entry.index;
                self.coefficients[write] = entry.coefficient;
                write += 1;
            } else {
                // The tag selects the destination; it does not permute the
                // entry. This matches SparseVector::MoveTaggedEntriesTo().
                tagged.add_entry(entry.index, entry.coefficient);
            }
        }
        self.indices.truncate(write);
        self.coefficients.truncate(write);
    }

    pub fn remove_near_zero_entries(&mut self, threshold: Fractional) {
        debug_assert!(self.check_no_duplicates());
        self.retain(|entry| entry.coefficient.abs() > threshold);
    }

    pub fn remove_near_zero_entries_with_weights(
        &mut self,
        threshold: Fractional,
        weights: &TypedVec<I, Fractional>,
    ) {
        debug_assert!(self.check_no_duplicates());
        self.retain(|entry| entry.coefficient.abs() * weights[entry.index] > threshold);
    }

    pub fn multiply_by_constant(&mut self, factor: Fractional) {
        for coefficient in &mut self.coefficients {
            *coefficient *= factor;
        }
    }

    pub fn component_wise_multiply(&mut self, factors: &TypedVec<I, Fractional>) {
        for (&index, coefficient) in self.indices.iter().zip(&mut self.coefficients) {
            *coefficient *= factors[index];
        }
    }

    pub fn divide_by_constant(&mut self, factor: Fractional) {
        for coefficient in &mut self.coefficients {
            *coefficient /= factor;
        }
    }

    pub fn component_wise_divide(&mut self, factors: &TypedVec<I, Fractional>) {
        for (&index, coefficient) in self.indices.iter().zip(&mut self.coefficients) {
            *coefficient /= factors[index];
        }
    }

    pub fn populate_from_dense(&mut self, dense: &TypedVec<I, Fractional>) {
        self.clear();
        for (position, &value) in dense.as_slice().iter().enumerate() {
            if value != 0.0 {
                self.add_entry(I::from_usize(position), value);
            }
        }
        self.may_contain_duplicates.set(false);
    }

    /// Replaces this vector with an exact copy of `source`, including its
    /// internal entry order and duplicate-state cache.
    pub fn populate_from_sparse_vector(&mut self, source: &Self) {
        self.indices.clone_from(&source.indices);
        self.coefficients.clone_from(&source.coefficients);
        self.may_contain_duplicates
            .set(source.may_contain_duplicates.get());
    }

    /// Appends entries after shifting their logical indices.
    ///
    /// # Panics
    ///
    /// Panics if a shifted index is negative or cannot be represented.
    pub fn append_entries_with_offset(&mut self, source: &Self, offset: i32) {
        for entry in source {
            let shifted =
                i64::from(i32::try_from(entry.index.to_usize()).expect("sparse index exceeds i32"))
                    + i64::from(offset);
            let shifted = usize::try_from(shifted).expect("shifted sparse index is negative");
            self.set_coefficient(I::from_usize(shifted), entry.coefficient);
        }
    }

    #[must_use]
    pub fn scalar_product(&self, dense: &TypedVec<I, Fractional>) -> Fractional {
        self.iter()
            .map(|entry| entry.coefficient * dense[entry.index])
            .sum()
    }

    #[must_use]
    pub fn first(&self) -> Option<SparseEntry<I>> {
        (!self.is_empty()).then(|| self.entry(0))
    }

    #[must_use]
    pub fn last(&self) -> Option<SparseEntry<I>> {
        (!self.is_empty()).then(|| self.entry(self.num_entries() - 1))
    }

    pub fn move_entry_to_first_position(&mut self, index: I) {
        if let Some(position) = self
            .indices
            .iter()
            .position(|&entry_index| entry_index == index)
        {
            self.indices.swap(0, position);
            self.coefficients.swap(0, position);
        }
    }

    pub fn move_entry_to_last_position(&mut self, index: I) {
        if let Some(position) = self
            .indices
            .iter()
            .position(|&entry_index| entry_index == index)
        {
            let last = self.indices.len() - 1;
            self.indices.swap(last, position);
            self.coefficients.swap(last, position);
        }
    }

    #[must_use]
    pub fn look_up_coefficient(&self, index: I) -> Fractional {
        self.iter()
            .rev()
            .find(|entry| entry.index == index)
            .map_or(0.0, |entry| entry.coefficient)
    }

    pub fn copy_to_dense_vector(&self, size: I, dense: &mut TypedVec<I, Fractional>) {
        *dense = TypedVec::filled(size, 0.0);
        for entry in self {
            dense[entry.index] = entry.coefficient;
        }
    }

    pub fn permuted_copy_to_dense_vector(
        &self,
        permutation: &Permutation<I>,
        size: I,
        dense: &mut TypedVec<I, Fractional>,
    ) {
        *dense = TypedVec::filled(size, 0.0);
        if permutation.is_empty() {
            for entry in self {
                dense[entry.index] = entry.coefficient;
            }
        } else {
            for entry in self {
                dense[permutation[entry.index]] = entry.coefficient;
            }
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
        for entry in self {
            dense[entry.index] += multiplier * entry.coefficient;
        }
    }

    pub fn iter(&self) -> SparseVectorIter<'_, I> {
        SparseVectorIter {
            indices: self.indices.iter(),
            coefficients: self.coefficients.iter(),
        }
    }

    pub fn apply_index_permutation(&mut self, permutation: &Permutation<I>) {
        if permutation.is_empty() {
            return;
        }
        for index in &mut self.indices {
            *index = permutation[*index];
        }
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
        let mut merged = Vec::with_capacity(self.num_entries() + accumulator.num_entries());
        let (mut left, mut right) = (0, 0);
        while left < accumulator.num_entries() || right < self.num_entries() {
            let entry = if right == self.num_entries()
                || (left < accumulator.num_entries()
                    && accumulator.indices[left] < self.indices[right])
            {
                let entry = accumulator.entry(left);
                left += 1;
                entry
            } else if left == accumulator.num_entries()
                || self.indices[right] < accumulator.indices[left]
            {
                let mut entry = self.entry(right);
                entry.coefficient *= multiplier;
                right += 1;
                entry
            } else {
                let entry = SparseEntry {
                    index: accumulator.indices[left],
                    coefficient: accumulator.coefficients[left]
                        + multiplier * self.coefficients[right],
                };
                left += 1;
                right += 1;
                entry
            };
            if entry.coefficient.abs() > drop_tolerance {
                merged.push(entry);
            }
        }
        accumulator.replace_entries(merged);
        accumulator.may_contain_duplicates.set(false);
    }

    /// Upstream's factorization merge: add a clean vector and remove a common
    /// pivot index while pruning newly computed small coefficients.
    pub fn add_multiple_to_sparse_vector_and_delete_common_index(
        &self,
        multiplier: Fractional,
        common_index: I,
        drop_tolerance: Fractional,
        accumulator: &mut Self,
    ) {
        self.add_multiple_to_sparse_vector_with_common_index(
            multiplier,
            common_index,
            drop_tolerance,
            false,
            accumulator,
        );
    }

    /// As above, but retain the accumulator's existing common-index value.
    pub fn add_multiple_to_sparse_vector_and_ignore_common_index(
        &self,
        multiplier: Fractional,
        common_index: I,
        drop_tolerance: Fractional,
        accumulator: &mut Self,
    ) {
        self.add_multiple_to_sparse_vector_with_common_index(
            multiplier,
            common_index,
            drop_tolerance,
            true,
            accumulator,
        );
    }

    fn add_multiple_to_sparse_vector_with_common_index(
        &self,
        multiplier: Fractional,
        common_index: I,
        drop_tolerance: Fractional,
        retain_common: bool,
        accumulator: &mut Self,
    ) {
        debug_assert!(self.is_cleaned_up());
        debug_assert!(accumulator.is_cleaned_up());
        let mut merged = Vec::with_capacity(self.num_entries() + accumulator.num_entries());
        let (mut source, mut destination) = (0, 0);
        while source < self.num_entries() || destination < accumulator.num_entries() {
            let source_index = self.indices.get(source).copied();
            let destination_index = accumulator.indices.get(destination).copied();
            match (source_index, destination_index) {
                (Some(left), Some(right)) if left == right => {
                    if left == common_index {
                        if retain_common {
                            merged.push(accumulator.entry(destination));
                        }
                    } else {
                        let value = accumulator.coefficients[destination]
                            + multiplier * self.coefficients[source];
                        if value.abs() > drop_tolerance {
                            merged.push(SparseEntry {
                                index: left,
                                coefficient: value,
                            });
                        }
                    }
                    source += 1;
                    destination += 1;
                }
                (Some(left), Some(right)) if left < right => {
                    let value = multiplier * self.coefficients[source];
                    if left != common_index && value.abs() > drop_tolerance {
                        merged.push(SparseEntry {
                            index: left,
                            coefficient: value,
                        });
                    }
                    source += 1;
                }
                (Some(_) | None, Some(_)) => {
                    let entry = accumulator.entry(destination);
                    if entry.index != common_index || retain_common {
                        merged.push(entry);
                    }
                    destination += 1;
                }
                (Some(left), None) => {
                    let value = multiplier * self.coefficients[source];
                    if left != common_index && value.abs() > drop_tolerance {
                        merged.push(SparseEntry {
                            index: left,
                            coefficient: value,
                        });
                    }
                    source += 1;
                }
                (None, None) => break,
            }
        }
        accumulator.replace_entries(merged);
        accumulator.may_contain_duplicates.set(false);
    }

    pub fn apply_partial_index_permutation(&mut self, permutation: &Permutation<I>) {
        let entries = self
            .iter()
            .filter_map(|entry| {
                let index = permutation[entry.index];
                (index.value_i64() >= 0).then_some(SparseEntry {
                    index,
                    coefficient: entry.coefficient,
                })
            })
            .collect();
        self.replace_entries(entries);
    }

    pub fn move_tagged_entries_to(&mut self, permutation: &Permutation<I>, output: &mut Self) {
        let mut write = 0;
        for read in 0..self.num_entries() {
            let entry = self.entry(read);
            if permutation[entry.index].value_i64() >= 0 {
                output.set_coefficient(entry.index, entry.coefficient);
            } else {
                self.indices[write] = entry.index;
                self.coefficients[write] = entry.coefficient;
                write += 1;
            }
        }
        self.indices.truncate(write);
        self.coefficients.truncate(write);
    }

    #[must_use]
    pub fn entry(&self, position: usize) -> SparseEntry<I> {
        SparseEntry {
            index: self.indices[position],
            coefficient: self.coefficients[position],
        }
    }

    /// Exact entry-by-entry equality in internal order. As in GLOP, the
    /// duplicate-state cache is not part of a vector's value.
    #[must_use]
    pub fn is_equal_to(&self, other: &Self) -> bool {
        self.indices == other.indices && self.coefficients == other.coefficients
    }

    fn replace_entries(&mut self, entries: Vec<SparseEntry<I>>) {
        self.indices.clear();
        self.coefficients.clear();
        self.reserve(entries.len());
        for entry in entries {
            self.indices.push(entry.index);
            self.coefficients.push(entry.coefficient);
        }
    }

    fn retain(&mut self, mut keep: impl FnMut(SparseEntry<I>) -> bool) {
        let mut write = 0;
        for read in 0..self.num_entries() {
            let entry = self.entry(read);
            if keep(entry) {
                self.indices[write] = entry.index;
                self.coefficients[write] = entry.coefficient;
                write += 1;
            }
        }
        self.indices.truncate(write);
        self.coefficients.truncate(write);
    }
}

impl<I: PartialEq> PartialEq for SparseVector<I> {
    fn eq(&self, other: &Self) -> bool {
        self.indices == other.indices && self.coefficients == other.coefficients
    }
}

impl<'a, I: VectorIndex + Ord> IntoIterator for &'a SparseVector<I> {
    type Item = SparseEntry<I>;
    type IntoIter = SparseVectorIter<'a, I>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type SparseColumn = SparseVector<RowIndex>;

impl SparseVector<RowIndex> {
    #[must_use]
    pub fn entry_row(&self, position: usize) -> RowIndex {
        self.entry(position).index()
    }

    #[must_use]
    pub fn entry_coefficient(&self, position: usize) -> Fractional {
        self.entry(position).coefficient()
    }

    /// # Panics
    ///
    /// Panics if the column is empty, matching upstream's unchecked accessor.
    #[must_use]
    pub fn first_row(&self) -> RowIndex {
        self.first().unwrap().index()
    }

    /// # Panics
    ///
    /// Panics if the column is empty, matching upstream's unchecked accessor.
    #[must_use]
    pub fn last_row(&self) -> RowIndex {
        self.last().unwrap().index()
    }

    pub fn apply_row_permutation(&mut self, permutation: &Permutation<RowIndex>) {
        self.apply_index_permutation(permutation);
    }

    pub fn apply_partial_row_permutation(&mut self, permutation: &Permutation<RowIndex>) {
        self.apply_partial_index_permutation(permutation);
    }
}

/// Borrowed sparse-column view corresponding to upstream `ColumnView`.
#[derive(Clone, Copy, Debug)]
pub struct ColumnView<'a> {
    rows: &'a [RowIndex],
    coefficients: &'a [Fractional],
}

impl<'a> ColumnView<'a> {
    /// Creates a view over parallel row and coefficient slices.
    ///
    /// # Panics
    ///
    /// Panics when the slices have different lengths.
    #[must_use]
    pub fn new(rows: &'a [RowIndex], coefficients: &'a [Fractional]) -> Self {
        assert_eq!(rows.len(), coefficients.len());
        Self { rows, coefficients }
    }

    #[must_use]
    pub fn from_column(column: &'a SparseColumn) -> Self {
        Self::new(&column.indices, &column.coefficients)
    }

    #[must_use]
    pub const fn num_entries(self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub fn entry_coefficient(self, position: usize) -> Fractional {
        self.coefficients[position]
    }

    #[must_use]
    pub fn entry_row(self, position: usize) -> RowIndex {
        self.rows[position]
    }

    /// # Panics
    ///
    /// Panics if the view is empty, matching upstream's unchecked accessor.
    #[must_use]
    pub fn first_coefficient(self) -> Fractional {
        self.entry_coefficient(0)
    }

    /// # Panics
    ///
    /// Panics if the view is empty, matching upstream's unchecked accessor.
    #[must_use]
    pub fn first_row(self) -> RowIndex {
        self.entry_row(0)
    }

    #[must_use]
    pub fn look_up_coefficient(self, row: RowIndex) -> Fractional {
        let mut value = 0.0;
        for entry in self {
            if entry.index() == row {
                value = entry.coefficient();
            }
        }
        value
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.rows.is_empty()
    }

    #[must_use]
    pub fn iter(self) -> SparseVectorIter<'a, RowIndex> {
        SparseVectorIter {
            indices: self.rows.iter(),
            coefficients: self.coefficients.iter(),
        }
    }
}

impl<'a> From<&'a SparseColumn> for ColumnView<'a> {
    fn from(column: &'a SparseColumn) -> Self {
        Self::from_column(column)
    }
}

impl<'a> IntoIterator for ColumnView<'a> {
    type Item = SparseEntry<RowIndex>;
    type IntoIter = SparseVectorIter<'a, RowIndex>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type RowView<'a> = SparseVectorIter<'a, ColIndex>;

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
    fn reusable_duplicate_workspace_is_restored_and_result_is_cached() {
        let mut vector = SparseColumn::new();
        vector.set_coefficient(RowIndex::new(3), 1.0);
        vector.set_coefficient(RowIndex::new(7), 2.0);
        let mut workspace = crate::lp_types::DenseBooleanColumn::new();
        assert!(vector.check_no_duplicates_with_workspace(&mut workspace));
        assert!(workspace.as_slice().iter().all(|&value| !value));
        assert!(vector.check_no_duplicates_with_workspace(&mut workspace));

        vector.set_coefficient(RowIndex::new(3), 4.0);
        assert!(!vector.check_no_duplicates_with_workspace(&mut workspace));
        assert!(workspace.as_slice().iter().all(|&value| !value));
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
    fn column_specialization_and_view_preserve_entry_order_and_last_lookup() {
        let mut column = SparseColumn::new();
        column.add_entry(RowIndex::new(3), 2.0);
        column.add_entry(RowIndex::new(1), -1.0);
        column.add_entry(RowIndex::new(3), 4.0);
        assert_eq!(column.entry_row(0), RowIndex::new(3));
        assert_eq!(column.entry_coefficient(1).to_bits(), (-1.0_f64).to_bits());
        assert_eq!(column.first_row(), RowIndex::new(3));
        assert_eq!(column.last_row(), RowIndex::new(3));

        let view = ColumnView::from_column(&column);
        assert_eq!(view.num_entries(), 3);
        assert_eq!(view.first_row(), RowIndex::new(3));
        assert_eq!(view.first_coefficient().to_bits(), 2.0_f64.to_bits());
        assert_eq!(
            view.look_up_coefficient(RowIndex::new(3)).to_bits(),
            4.0_f64.to_bits()
        );
        assert_eq!(
            view.into_iter().map(SparseEntry::index).collect::<Vec<_>>(),
            vec![RowIndex::new(3), RowIndex::new(1), RowIndex::new(3)]
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
        assert_eq!(accumulator.entry(0).index(), RowIndex::new(0));
        assert_eq!(accumulator.entry(1).index(), RowIndex::new(1));
    }

    #[test]
    fn moving_tagged_entries_preserves_their_indices() {
        let mut source = SparseColumn::new();
        source.add_entry(RowIndex::new(1), 2.0);
        source.add_entry(RowIndex::new(3), 4.0);
        let tags = [usize::MAX, 7, usize::MAX, usize::MAX];
        let mut tagged = SparseColumn::new();
        source.move_entries_with_index_tags_to(&tags, usize::MAX, &mut tagged);
        assert_eq!(source.entry(0).index(), RowIndex::new(3));
        assert_eq!(tagged.entry(0).index(), RowIndex::new(1));
    }

    #[test]
    fn factorization_merges_delete_or_ignore_the_common_index() {
        let mut source = SparseColumn::new();
        source.set_coefficient(RowIndex::new(0), 2.0);
        source.set_coefficient(RowIndex::new(1), 3.0);
        source.set_coefficient(RowIndex::new(3), -1.0);
        source.clean_up();
        let mut deleted = SparseColumn::new();
        deleted.set_coefficient(RowIndex::new(1), 7.0);
        deleted.set_coefficient(RowIndex::new(2), 4.0);
        deleted.clean_up();
        let mut ignored = deleted.clone();

        source.add_multiple_to_sparse_vector_and_delete_common_index(
            -2.0,
            RowIndex::new(1),
            0.0,
            &mut deleted,
        );
        source.add_multiple_to_sparse_vector_and_ignore_common_index(
            -2.0,
            RowIndex::new(1),
            0.0,
            &mut ignored,
        );

        assert_eq!(
            deleted
                .iter()
                .map(|entry| (entry.index().value(), entry.coefficient()))
                .collect::<Vec<_>>(),
            [(0, -4.0), (2, 4.0), (3, 2.0)]
        );
        assert_eq!(
            ignored.look_up_coefficient(RowIndex::new(1)).to_bits(),
            7.0_f64.to_bits()
        );
    }
}
