//! Column-oriented sparse matrices.
//!
//! This is the Phase-1 subset of upstream `ortools/lp_data/sparse.{h,cc}`.

use crate::lp_types::{
    ColIndex, DenseBooleanRow, DenseColumn, DenseRow, EntryIndex, Fractional, RowIndex, VectorIndex,
};
use crate::permutation::{ColumnPermutation, RowPermutation};
use crate::scattered_vector::ScatteredColumn;
use crate::sparse_vector::SparseColumn;
use std::rc::Rc;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseMatrix {
    columns: Vec<SparseColumn>,
    num_rows: RowIndex,
}

#[derive(Clone, Debug, Default)]
pub struct CompactSparseMatrix {
    starts: Vec<usize>,
    rows: Vec<RowIndex>,
    coefficients: Vec<Fractional>,
    num_rows: RowIndex,
    num_cols: ColIndex,
}

/// Proof that this immutable compact matrix's row indices fit a dense vector.
/// Owning an `Rc` prevents mutation while the token exists; the identity check
/// prevents reuse after a different matrix is rebound.
#[derive(Clone, Debug)]
pub struct ValidatedCompactRows {
    matrix: Rc<CompactSparseMatrix>,
    dimension: usize,
}

impl CompactSparseMatrix {
    /// Validates sparse gathers once for repeated products with this matrix.
    #[must_use]
    pub fn validated_rows(matrix: Rc<Self>, dimension: usize) -> Option<ValidatedCompactRows> {
        if matrix.rows.iter().all(|row| {
            row.value() >= 0 && usize::try_from(row.value()).is_ok_and(|index| index < dimension)
        }) {
            Some(ValidatedCompactRows { matrix, dimension })
        } else {
            None
        }
    }
    #[must_use]
    pub fn from_sparse(matrix: &SparseMatrix) -> Self {
        Self::from_matrix_view(&MatrixView::from_matrix(matrix))
    }

    #[must_use]
    pub fn from_matrix_view(matrix: &MatrixView<'_>) -> Self {
        let mut starts = Vec::with_capacity(matrix.num_cols().to_usize() + 1);
        let mut rows = Vec::with_capacity(matrix.num_entries().value().try_into().unwrap_or(0));
        let mut coefficients = Vec::with_capacity(rows.capacity());
        starts.push(0);
        for column in 0..matrix.num_cols().to_usize() {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                rows.push(entry.index());
                coefficients.push(entry.coefficient());
            }
            starts.push(rows.len());
        }
        Self {
            starts,
            rows,
            coefficients,
            num_rows: matrix.num_rows(),
            num_cols: matrix.num_cols(),
        }
    }

    pub fn populate_from_sparse(&mut self, matrix: &SparseMatrix) {
        *self = Self::from_sparse(matrix);
    }

    pub fn populate_from_matrix_view(&mut self, matrix: &MatrixView<'_>) {
        *self = Self::from_matrix_view(matrix);
    }

    pub fn populate_from_sparse_and_add_slacks(&mut self, matrix: &SparseMatrix) {
        self.reset(matrix.num_rows());
        for position in 0..matrix.num_cols().to_usize() {
            for entry in matrix.column(ColIndex::from_usize(position)) {
                self.add_entry_to_current_column(entry.index(), entry.coefficient());
            }
            self.close_current_column();
        }
        for position in 0..matrix.num_rows().to_usize() {
            self.add_entry_to_current_column(RowIndex::from_usize(position), 1.0);
            self.close_current_column();
        }
    }

    pub fn populate_from_transpose(&mut self, input: &Self) {
        self.num_rows = RowIndex::new(input.num_cols().value());
        self.num_cols = ColIndex::new(input.num_rows().value());
        let output_columns = self.num_cols.to_usize();
        self.starts = vec![0; output_columns + 2];
        for &row in &input.rows {
            self.starts[row.to_usize() + 2] += 1;
        }
        for column in 2..self.starts.len() {
            self.starts[column] += self.starts[column - 1];
        }
        let entries = *self.starts.last().unwrap_or(&0);
        self.coefficients.resize(entries, 0.0);
        self.rows.resize(entries, RowIndex::new(-1));
        self.starts.pop();
        for column in 0..input.num_cols.to_usize() {
            for entry in input.starts[column]..input.starts[column + 1] {
                let transposed_column = input.rows[entry].to_usize();
                let destination = self.starts[transposed_column + 1];
                self.starts[transposed_column + 1] += 1;
                self.coefficients[destination] = input.coefficients[entry];
                self.rows[destination] = RowIndex::from_usize(column);
            }
        }
    }

    pub fn reset(&mut self, num_rows: RowIndex) {
        self.num_rows = num_rows;
        self.num_cols = ColIndex::new(0);
        self.rows.clear();
        self.coefficients.clear();
        self.starts.clear();
        self.starts.push(0);
    }

    pub fn add_entry_to_current_column(&mut self, row: RowIndex, coefficient: Fractional) {
        self.rows.push(row);
        self.coefficients.push(coefficient);
    }

    pub fn close_current_column(&mut self) {
        self.starts.push(self.rows.len());
        self.num_cols = ColIndex::new(self.num_cols.value() + 1);
    }

    pub fn add_dense_column(&mut self, column: &DenseColumn) -> ColIndex {
        self.add_dense_column_prefix(column, RowIndex::new(0))
    }

    pub fn add_dense_column_prefix(&mut self, column: &DenseColumn, start: RowIndex) -> ColIndex {
        for position in start.to_usize()..column.len().to_usize() {
            let row = RowIndex::from_usize(position);
            if column[row] != 0.0 {
                self.add_entry_to_current_column(row, column[row]);
            }
        }
        self.close_current_column();
        ColIndex::new(self.num_cols.value() - 1)
    }

    pub fn add_dense_column_with_nonzeros(
        &mut self,
        column: &DenseColumn,
        nonzeros: &[RowIndex],
    ) -> ColIndex {
        if nonzeros.is_empty() {
            return self.add_dense_column(column);
        }
        for &row in nonzeros {
            if column[row] != 0.0 {
                self.add_entry_to_current_column(row, column[row]);
            }
        }
        self.close_current_column();
        ColIndex::new(self.num_cols.value() - 1)
    }

    pub fn add_and_clear_column_with_nonzeros(
        &mut self,
        column: &mut DenseColumn,
        nonzeros: &mut Vec<RowIndex>,
    ) -> ColIndex {
        for &row in nonzeros.iter() {
            if column[row] != 0.0 {
                self.add_entry_to_current_column(row, column[row]);
                column[row] = 0.0;
            }
        }
        nonzeros.clear();
        self.close_current_column();
        ColIndex::new(self.num_cols.value() - 1)
    }

    #[must_use]
    #[inline]
    pub const fn num_rows(&self) -> RowIndex {
        self.num_rows
    }

    #[must_use]
    #[inline]
    pub fn num_cols(&self) -> ColIndex {
        self.num_cols
    }

    #[must_use]
    #[inline]
    pub fn num_entries(&self) -> EntryIndex {
        EntryIndex::new(i64::try_from(self.coefficients.len()).unwrap_or(i64::MAX))
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn column_num_entries(&self, column: ColIndex) -> EntryIndex {
        EntryIndex::new(
            i64::try_from(self.starts[column.to_usize() + 1] - self.starts[column.to_usize()])
                .unwrap_or(i64::MAX),
        )
    }

    #[must_use]
    #[inline]
    pub fn column_is_empty(&self, column: ColIndex) -> bool {
        self.starts[column.to_usize()] == self.starts[column.to_usize() + 1]
    }

    #[must_use]
    #[inline]
    pub fn column(&self, column: ColIndex) -> CompactColumn<'_> {
        let start = self.starts[column.to_usize()];
        let end = self.starts[column.to_usize() + 1];
        CompactColumn {
            rows: &self.rows[start..end],
            coefficients: &self.coefficients[start..end],
        }
    }

    #[must_use]
    pub fn column_scalar_product(&self, column: ColIndex, vector: &DenseRow) -> Fractional {
        self.column_scalar_product_slice(column, vector.as_slice())
    }

    /// Same four-accumulator kernel for a row-indexed untyped slice.
    #[must_use]
    // Casting the 32-bit row directly mirrors GLOP's RowToColIndex() and
    // avoids a redundant signedness branch per sparse entry. A negative or
    // out-of-range row still panics at the checked vector indexing below.
    #[allow(clippy::cast_sign_loss)]
    // The isolated native comparison shows a substantial gap in this hot
    // kernel; only the two matrix-owned arrays are unchecked below.
    #[allow(unsafe_code)]
    // GLOP defines this kernel inline in sparse.h. The isolated benchmark
    // measures a further 0–18% reduction from inlining the Rust counterpart.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn column_scalar_product_slice(
        &self,
        column: ColIndex,
        vector: &[Fractional],
    ) -> Fractional {
        self.column_scalar_product_slice_impl::<false>(column, vector)
    }

    /// Same product after all sparse row indices have been validated once.
    ///
    /// # Panics
    ///
    /// Panics if the token belongs to another matrix or the vector is too short.
    #[must_use]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn column_scalar_product_validated(
        &self,
        rows: &ValidatedCompactRows,
        column: ColIndex,
        vector: &[Fractional],
    ) -> Fractional {
        assert!(std::ptr::eq(Rc::as_ptr(&rows.matrix), self));
        assert!(vector.len() >= rows.dimension);
        self.column_scalar_product_slice_impl::<true>(column, vector)
    }

    #[allow(unsafe_code, clippy::inline_always, clippy::cast_sign_loss)]
    #[inline(always)]
    fn column_scalar_product_slice_impl<const VALIDATED: bool>(
        &self,
        column: ColIndex,
        vector: &[Fractional],
    ) -> Fractional {
        let start = self.starts[column.to_usize()];
        let end = self.starts[column.to_usize() + 1];
        debug_assert_eq!(self.rows.len(), self.coefficients.len());
        debug_assert!(end <= self.rows.len());
        let entry_data = |index: usize| {
            // SAFETY: every CompactSparseMatrix builder appends one row and one
            // coefficient per entry and closes a column at their common length.
            // `index` is in [start, end), so both backing arrays contain it.
            // The caller's vector is still indexed safely because malformed
            // row coordinates must not become unchecked memory accesses.
            unsafe {
                (
                    *self.coefficients.get_unchecked(index),
                    self.rows.get_unchecked(index).value() as usize,
                )
            }
        };
        let gathered = |row: usize| {
            if VALIDATED {
                // SAFETY: the identity-bound token was created by scanning
                // every stored row and the vector length was checked above.
                unsafe { *vector.get_unchecked(row) }
            } else {
                vector[row]
            }
        };
        let shifted_end = end.saturating_sub(3);
        let mut entry = start;
        let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
        while entry < shifted_end {
            let (coefficient, row) = entry_data(entry);
            result1 = coefficient.mul_add(gathered(row), result1);
            let (coefficient, row) = entry_data(entry + 1);
            result2 = coefficient.mul_add(gathered(row), result2);
            let (coefficient, row) = entry_data(entry + 2);
            result3 = coefficient.mul_add(gathered(row), result3);
            let (coefficient, row) = entry_data(entry + 3);
            result4 = coefficient.mul_add(gathered(row), result4);
            entry += 4;
        }
        let mut result = result1 + result2 + result3 + result4;
        if entry < end {
            let (coefficient, row) = entry_data(entry);
            result = coefficient.mul_add(gathered(row), result);
            if entry + 1 < end {
                let (coefficient, row) = entry_data(entry + 1);
                result = coefficient.mul_add(gathered(row), result);
                if entry + 2 < end {
                    let (coefficient, row) = entry_data(entry + 2);
                    result = coefficient.mul_add(gathered(row), result);
                }
            }
        }
        result
    }

    pub fn column_add_multiple_to_dense_column(
        &self,
        column: ColIndex,
        multiplier: Fractional,
        output: &mut DenseColumn,
    ) {
        if multiplier == 0.0 {
            return;
        }
        // The column view bounds the paired matrix arrays once. Each output
        // slot is borrowed once, retaining a checked gather for public callers.
        for (row, coefficient) in self.column(column).iter() {
            let slot = &mut output.as_mut_slice()[row.to_usize()];
            *slot = multiplier.mul_add(coefficient, *slot);
        }
    }

    pub fn column_add_multiple_to_scattered_column(
        &self,
        column: ColIndex,
        multiplier: Fractional,
        output: &mut ScatteredColumn,
    ) {
        if multiplier == 0.0 {
            return;
        }
        for (row, coefficient) in self.column(column).iter() {
            output.add(row, multiplier * coefficient);
        }
    }

    pub fn column_copy_to_dense_column(&self, column: ColIndex, output: &mut DenseColumn) {
        *output = DenseColumn::filled(self.num_rows, 0.0);
        self.column_copy_to_cleared_dense_column(column, output);
    }

    pub fn column_copy_to_cleared_dense_column(&self, column: ColIndex, output: &mut DenseColumn) {
        output.resize(self.num_rows, 0.0);
        for (row, coefficient) in self.column(column).iter() {
            output[row] = coefficient;
        }
    }

    pub fn column_copy_to_cleared_dense_column_with_nonzeros(
        &self,
        column: ColIndex,
        output: &mut DenseColumn,
        nonzeros: &mut Vec<RowIndex>,
    ) {
        output.resize(self.num_rows, 0.0);
        nonzeros.clear();
        for (row, coefficient) in self.column(column).iter() {
            output[row] = coefficient;
            nonzeros.push(row);
        }
    }

    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompactColumn<'a> {
    rows: &'a [RowIndex],
    coefficients: &'a [Fractional],
}

impl<'a> CompactColumn<'a> {
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = (RowIndex, Fractional)> + 'a {
        self.rows
            .iter()
            .copied()
            .zip(self.coefficients.iter().copied())
    }

    #[must_use]
    #[inline]
    pub const fn len(self) -> usize {
        self.rows.len()
    }

    #[must_use]
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.rows.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn entry_row(self, entry: usize) -> RowIndex {
        self.rows[entry]
    }

    #[must_use]
    #[inline]
    pub fn entry_coefficient(self, entry: usize) -> Fractional {
        self.coefficients[entry]
    }
}

#[derive(Clone, Debug)]
pub struct MatrixView<'a> {
    columns: Vec<&'a SparseColumn>,
    num_rows: RowIndex,
}

impl<'a> MatrixView<'a> {
    #[must_use]
    pub fn from_matrix(matrix: &'a SparseMatrix) -> Self {
        let columns = (0..matrix.num_cols().to_usize())
            .map(|column| matrix.column(ColIndex::from_usize(column)))
            .collect();
        Self {
            columns,
            num_rows: matrix.num_rows(),
        }
    }

    #[must_use]
    pub fn from_matrix_pair(left: &'a SparseMatrix, right: &'a SparseMatrix) -> Self {
        let mut columns =
            Vec::with_capacity(left.num_cols().to_usize() + right.num_cols().to_usize());
        columns.extend(
            (0..left.num_cols().to_usize()).map(|column| left.column(ColIndex::from_usize(column))),
        );
        columns.extend(
            (0..right.num_cols().to_usize())
                .map(|column| right.column(ColIndex::from_usize(column))),
        );
        Self {
            columns,
            num_rows: left.num_rows().max(right.num_rows()),
        }
    }

    #[must_use]
    pub fn from_basis(matrix: &Self, basis: &[ColIndex]) -> Self {
        Self {
            columns: basis.iter().map(|&column| matrix.column(column)).collect(),
            num_rows: matrix.num_rows(),
        }
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn num_rows(&self) -> RowIndex {
        self.num_rows
    }

    #[must_use]
    #[inline]
    pub fn num_cols(&self) -> ColIndex {
        ColIndex::from_usize(self.columns.len())
    }

    #[must_use]
    #[inline]
    pub fn column(&self, column: ColIndex) -> &'a SparseColumn {
        self.columns[column.to_usize()]
    }

    #[must_use]
    pub fn num_entries(&self) -> EntryIndex {
        EntryIndex::new(
            i64::try_from(
                self.columns
                    .iter()
                    .map(|column| column.num_entries())
                    .sum::<usize>(),
            )
            .unwrap_or(i64::MAX),
        )
    }

    #[must_use]
    pub fn one_norm(&self) -> Fractional {
        self.columns
            .iter()
            .map(|column| column.iter().map(|entry| entry.coefficient().abs()).sum())
            .fold(0.0, Fractional::max)
    }

    #[must_use]
    pub fn infinity_norm(&self) -> Fractional {
        let mut sums = vec![0.0; self.num_rows.to_usize()];
        for column in &self.columns {
            for entry in *column {
                sums[entry.index().to_usize()] += entry.coefficient().abs();
            }
        }
        sums.into_iter().fold(0.0, Fractional::max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompactSparseMatrixView<'a> {
    matrix: &'a CompactSparseMatrix,
    columns: &'a [ColIndex],
}

impl<'a> CompactSparseMatrixView<'a> {
    #[must_use]
    pub const fn new(matrix: &'a CompactSparseMatrix, columns: &'a [ColIndex]) -> Self {
        Self { matrix, columns }
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn num_rows(&self) -> RowIndex {
        self.matrix.num_rows()
    }

    #[must_use]
    #[inline]
    pub fn num_cols(&self) -> ColIndex {
        ColIndex::from_usize(self.columns.len())
    }

    #[must_use]
    #[inline]
    pub fn column(&self, column: ColIndex) -> CompactColumn<'a> {
        self.matrix.column(self.columns[column.to_usize()])
    }

    #[must_use]
    pub fn num_entries(&self) -> EntryIndex {
        EntryIndex::new(
            self.columns
                .iter()
                .map(|&column| self.matrix.column_num_entries(column).value())
                .sum(),
        )
    }

    #[must_use]
    pub fn one_norm(&self) -> Fractional {
        self.columns
            .iter()
            .map(|&column| {
                self.matrix
                    .column(column)
                    .iter()
                    .map(|(_, value)| value.abs())
                    .sum()
            })
            .fold(0.0, Fractional::max)
    }

    #[must_use]
    pub fn infinity_norm(&self) -> Fractional {
        let mut sums = vec![0.0; self.num_rows().to_usize()];
        for &column in self.columns {
            for (row, value) in self.matrix.column(column).iter() {
                sums[row.to_usize()] += value.abs();
            }
        }
        sums.into_iter().fold(0.0, Fractional::max)
    }
}

impl SparseMatrix {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            columns: Vec::new(),
            num_rows: RowIndex::new(0),
        }
    }

    /// Adds a multiple of one live column into another while removing the
    /// common row, using disjoint borrows rather than copying either column.
    ///
    /// # Panics
    ///
    /// Panics if source and destination are the same column.
    pub fn add_multiple_between_columns_and_delete_common_index(
        &mut self,
        source: ColIndex,
        destination: ColIndex,
        multiplier: Fractional,
        common_row: RowIndex,
        drop_tolerance: Fractional,
    ) {
        assert_ne!(source, destination);
        let source_index = source.to_usize();
        let destination_index = destination.to_usize();
        if source_index < destination_index {
            let (left, right) = self.columns.split_at_mut(destination_index);
            left[source_index].add_multiple_to_sparse_vector_and_delete_common_index(
                multiplier,
                common_row,
                drop_tolerance,
                &mut right[0],
            );
        } else {
            let (left, right) = self.columns.split_at_mut(source_index);
            right[0].add_multiple_to_sparse_vector_and_delete_common_index(
                multiplier,
                common_row,
                drop_tolerance,
                &mut left[destination_index],
            );
        }
    }

    pub fn clear(&mut self) {
        self.columns.clear();
        self.num_rows = RowIndex::new(0);
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty() || self.num_rows.value() == 0
    }

    pub fn clean_up(&mut self) {
        for column in &mut self.columns {
            column.clean_up();
        }
    }

    #[must_use]
    pub fn is_cleaned_up(&self) -> bool {
        self.columns.iter().all(SparseColumn::is_cleaned_up)
    }

    #[must_use]
    pub fn check_no_duplicates(&self) -> bool {
        self.columns.iter().all(SparseColumn::check_no_duplicates)
    }

    pub fn set_num_rows(&mut self, num_rows: RowIndex) {
        debug_assert!(num_rows.value() >= 0);
        self.num_rows = num_rows;
    }

    pub fn append_empty_column(&mut self) -> ColIndex {
        let index = ColIndex::from_usize(self.columns.len());
        self.columns.push(SparseColumn::new());
        index
    }

    pub fn append_unit_vector(&mut self, row: RowIndex, value: Fractional) {
        debug_assert!(row < self.num_rows);
        let mut column = SparseColumn::new();
        column.set_coefficient(row, value);
        self.columns.push(column);
    }

    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    pub fn populate_from_zero(&mut self, num_rows: RowIndex, num_cols: ColIndex) {
        self.num_rows = num_rows;
        self.columns = vec![SparseColumn::new(); num_cols.to_usize()];
    }

    pub fn populate_from_identity(&mut self, size: ColIndex) {
        self.populate_from_zero(RowIndex::new(size.value()), size);
        for position in 0..size.to_usize() {
            self.columns[position].set_coefficient(RowIndex::from_usize(position), 1.0);
        }
    }

    pub fn populate_from_transpose(&mut self, input: &Self) {
        self.populate_from_zero(
            RowIndex::new(input.num_cols().value()),
            ColIndex::new(input.num_rows().value()),
        );
        let mut degrees = vec![0_usize; input.num_rows().to_usize()];
        for column in &input.columns {
            for entry in column {
                degrees[entry.index().to_usize()] += 1;
            }
        }
        for (position, &degree) in degrees.iter().enumerate() {
            self.columns[position].reserve(degree);
        }
        for (position, column) in input.columns.iter().enumerate() {
            let row = RowIndex::from_usize(position);
            for entry in column {
                self.columns[entry.index().to_usize()].set_coefficient(row, entry.coefficient());
            }
        }
        debug_assert!(self.is_cleaned_up());
    }

    pub fn populate_from_sparse(&mut self, input: &Self) {
        self.num_rows = input.num_rows;
        self.columns.clone_from(&input.columns);
    }

    pub fn populate_from_permuted_matrix(
        &mut self,
        input: &Self,
        row_permutation: &RowPermutation,
        inverse_column_permutation: &ColumnPermutation,
    ) {
        self.populate_from_zero(input.num_rows(), input.num_cols());
        for position in 0..input.num_cols().to_usize() {
            let output_column = ColIndex::from_usize(position);
            for entry in input.column(inverse_column_permutation[output_column]) {
                self.columns[position]
                    .set_coefficient(row_permutation[entry.index()], entry.coefficient());
            }
        }
        debug_assert!(self.check_no_duplicates());
    }

    pub fn populate_from_linear_combination(
        &mut self,
        alpha: Fractional,
        left: &Self,
        beta: Fractional,
        right: &Self,
    ) {
        debug_assert_eq!(left.num_cols(), right.num_cols());
        debug_assert_eq!(left.num_rows(), right.num_rows());
        self.populate_from_zero(left.num_rows(), left.num_cols());
        let mut workspace = crate::sparse_vector::RandomAccessSparseColumn::new(left.num_rows());
        for position in 0..left.num_cols().to_usize() {
            for entry in &left.columns[position] {
                workspace.add_to_coefficient(entry.index(), alpha * entry.coefficient());
            }
            for entry in &right.columns[position] {
                workspace.add_to_coefficient(entry.index(), beta * entry.coefficient());
            }
            workspace.populate_sparse_column(&mut self.columns[position]);
            self.columns[position].clean_up();
            workspace.clear();
        }
    }

    pub fn populate_from_product(&mut self, left: &Self, right: &Self) {
        self.populate_from_zero(left.num_rows(), right.num_cols());
        let mut workspace = crate::sparse_vector::RandomAccessSparseColumn::new(left.num_rows());
        for right_position in 0..right.num_cols().to_usize() {
            for right_entry in &right.columns[right_position] {
                if right_entry.coefficient() == 0.0 {
                    continue;
                }
                let left_column = ColIndex::new(right_entry.index().value());
                for left_entry in left.column(left_column) {
                    workspace.add_to_coefficient(
                        left_entry.index(),
                        left_entry.coefficient() * right_entry.coefficient(),
                    );
                }
            }
            workspace.populate_sparse_column(&mut self.columns[right_position]);
            self.columns[right_position].clean_up();
            workspace.clear();
        }
    }

    #[must_use]
    #[inline]
    pub const fn num_rows(&self) -> RowIndex {
        self.num_rows
    }

    #[must_use]
    #[inline]
    pub fn num_cols(&self) -> ColIndex {
        ColIndex::from_usize(self.columns.len())
    }

    #[must_use]
    pub fn num_entries(&self) -> EntryIndex {
        let entries = self
            .columns
            .iter()
            .map(SparseColumn::num_entries)
            .sum::<usize>();
        EntryIndex::new(i64::try_from(entries).unwrap_or(i64::MAX))
    }

    #[must_use]
    #[inline]
    pub fn column(&self, column: ColIndex) -> &SparseColumn {
        &self.columns[column.to_usize()]
    }

    pub fn mutable_column(&mut self, column: ColIndex) -> &mut SparseColumn {
        &mut self.columns[column.to_usize()]
    }

    pub fn replace_column(&mut self, column: ColIndex, replacement: SparseColumn) {
        self.columns[column.to_usize()] = replacement;
    }

    /// Applies a source-to-destination permutation to the matrix columns.
    ///
    /// # Panics
    ///
    /// Panics if the permutation size differs from the number of columns or a
    /// destination lies outside that range.
    pub fn apply_column_permutation(&mut self, destination_by_source: &[usize]) {
        if destination_by_source.is_empty() {
            return;
        }
        assert_eq!(destination_by_source.len(), self.columns.len());
        let mut permuted = vec![SparseColumn::new(); self.columns.len()];
        for (source, &destination) in destination_by_source.iter().enumerate() {
            permuted[destination] = self.columns[source].clone();
        }
        self.columns = permuted;
    }

    #[must_use]
    pub fn look_up_value(&self, row: RowIndex, column: ColIndex) -> Fractional {
        self.column(column).look_up_coefficient(row)
    }

    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut result = Self::new();
        result.populate_from_transpose(self);
        result
    }

    pub fn apply_row_permutation(&mut self, permutation: &RowPermutation) {
        for column in &mut self.columns {
            column.apply_index_permutation(permutation);
        }
    }

    pub fn delete_columns(&mut self, deleted: &DenseBooleanRow) {
        let mut position = 0;
        self.columns.retain(|_| {
            let keep =
                position >= deleted.as_slice().len() || !deleted[ColIndex::from_usize(position)];
            position += 1;
            keep
        });
    }

    pub fn delete_rows(&mut self, new_num_rows: RowIndex, permutation: &RowPermutation) {
        debug_assert_eq!(self.num_rows.value(), permutation.len().value());
        for column in &mut self.columns {
            column.apply_partial_index_permutation(permutation);
        }
        self.num_rows = new_num_rows;
    }

    pub fn append_rows_from_sparse(&mut self, input: &Self) -> bool {
        if self.num_cols() != input.num_cols() {
            return false;
        }
        let offset = self.num_rows.value();
        for position in 0..self.num_cols().to_usize() {
            self.columns[position].append_entries_with_offset(&input.columns[position], offset);
        }
        self.num_rows = RowIndex::new(offset + input.num_rows.value());
        true
    }

    #[must_use]
    pub fn equals_with_tolerance(&self, other: &Self, tolerance: Fractional) -> bool {
        if self.num_rows != other.num_rows || self.num_cols() != other.num_cols() {
            return false;
        }
        let mut left = crate::sparse_vector::RandomAccessSparseColumn::new(self.num_rows);
        let mut right = crate::sparse_vector::RandomAccessSparseColumn::new(self.num_rows);
        for position in 0..self.num_cols().to_usize() {
            for entry in &self.columns[position] {
                left.add_to_coefficient(entry.index(), entry.coefficient());
            }
            for entry in &other.columns[position] {
                if (entry.coefficient() - left.coefficient(entry.index())).abs() > tolerance {
                    return false;
                }
            }
            for entry in &other.columns[position] {
                right.add_to_coefficient(entry.index(), entry.coefficient());
            }
            for entry in &self.columns[position] {
                if (entry.coefficient() - right.coefficient(entry.index())).abs() > tolerance {
                    return false;
                }
            }
            left.clear();
            right.clear();
        }
        true
    }

    #[must_use]
    pub fn min_and_max_magnitudes(&self) -> (Fractional, Fractional) {
        let mut minimum = Fractional::INFINITY;
        let mut maximum: Fractional = 0.0;
        for column in &self.columns {
            for entry in column {
                let magnitude = entry.coefficient().abs();
                if magnitude != 0.0 {
                    minimum = minimum.min(magnitude);
                    maximum = maximum.max(magnitude);
                }
            }
        }
        if maximum == 0.0 {
            minimum = 0.0;
        }
        (minimum, maximum)
    }

    #[must_use]
    pub fn one_norm(&self) -> Fractional {
        self.columns
            .iter()
            .map(|column| {
                column
                    .into_iter()
                    .map(|entry| entry.coefficient().abs())
                    .sum()
            })
            .fold(0.0, Fractional::max)
    }

    #[must_use]
    pub fn infinity_norm(&self) -> Fractional {
        let mut sums = vec![0.0; self.num_rows.to_usize()];
        for column in &self.columns {
            for entry in column {
                sums[entry.index().to_usize()] += entry.coefficient().abs();
            }
        }
        sums.into_iter().fold(0.0, Fractional::max)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_coefficients_and_dimensions() {
        let mut matrix = SparseMatrix::new();
        matrix.populate_from_zero(RowIndex::new(3), ColIndex::new(2));
        matrix
            .mutable_column(ColIndex::new(0))
            .add_entry(RowIndex::new(2), 4.0);
        matrix
            .mutable_column(ColIndex::new(1))
            .add_entry(RowIndex::new(0), -3.0);
        let transpose = matrix.transpose();
        assert_eq!(transpose.num_rows(), RowIndex::new(2));
        assert_eq!(transpose.num_cols(), ColIndex::new(3));
        assert_eq!(
            transpose.look_up_value(RowIndex::new(0), ColIndex::new(2)),
            4.0
        );
        assert_eq!(matrix.one_norm(), 4.0);
        assert_eq!(matrix.infinity_norm(), 4.0);

        let compact = CompactSparseMatrix::from_sparse(&matrix);
        assert_eq!(compact.num_rows(), RowIndex::new(3));
        assert_eq!(compact.num_cols(), ColIndex::new(2));
        assert_eq!(
            compact.column(ColIndex::new(0)).iter().collect::<Vec<_>>(),
            vec![(RowIndex::new(2), 4.0)]
        );
        let full_view = MatrixView::from_matrix(&matrix);
        let view = MatrixView::from_basis(&full_view, &[ColIndex::new(1)]);
        assert_eq!(view.num_cols(), ColIndex::new(1));
        assert_eq!(
            view.column(ColIndex::new(0))
                .look_up_coefficient(RowIndex::new(0)),
            -3.0
        );
        assert_eq!(view.num_entries(), EntryIndex::new(1));
        assert_eq!(view.one_norm(), 3.0);
        assert_eq!(view.infinity_norm(), 3.0);

        let compact_columns = [ColIndex::new(1)];
        let compact_view = CompactSparseMatrixView::new(&compact, &compact_columns);
        assert_eq!(compact_view.num_rows(), RowIndex::new(3));
        assert_eq!(compact_view.num_cols(), ColIndex::new(1));
        assert_eq!(compact_view.num_entries(), EntryIndex::new(1));
        assert_eq!(compact_view.one_norm(), 3.0);
        assert_eq!(compact_view.infinity_norm(), 3.0);
    }

    #[test]
    fn compact_builders_transpose_and_dense_operations_match_sparse_matrix() {
        let mut dense = DenseColumn::from_vec(vec![2.0, 0.0, -3.0]);
        let mut compact = CompactSparseMatrix::default();
        compact.reset(RowIndex::new(3));
        assert_eq!(compact.add_dense_column(&dense), ColIndex::new(0));
        let mut positions = vec![RowIndex::new(0), RowIndex::new(2), RowIndex::new(0)];
        assert_eq!(
            compact.add_and_clear_column_with_nonzeros(&mut dense, &mut positions),
            ColIndex::new(1)
        );
        assert!(positions.is_empty());
        assert_eq!(dense.as_slice(), &[0.0, 0.0, 0.0]);
        assert_eq!(compact.num_cols(), ColIndex::new(2));

        let row = DenseRow::from_vec(vec![1.0, 2.0, 4.0]);
        assert_eq!(compact.column_scalar_product(ColIndex::new(0), &row), -10.0);
        let mut output = DenseColumn::new();
        compact.column_copy_to_dense_column(ColIndex::new(0), &mut output);
        assert_eq!(output.as_slice(), &[2.0, 0.0, -3.0]);

        let mut transpose = CompactSparseMatrix::default();
        transpose.populate_from_transpose(&compact);
        assert_eq!(transpose.num_rows(), RowIndex::new(2));
        assert_eq!(transpose.num_cols(), ColIndex::new(3));
        assert_eq!(
            transpose
                .column(ColIndex::new(0))
                .iter()
                .collect::<Vec<_>>(),
            vec![(RowIndex::new(0), 2.0), (RowIndex::new(1), 2.0)]
        );
    }

    #[test]
    fn compact_column_scalar_product_preserves_fused_native_rounding() {
        let mut compact = CompactSparseMatrix::default();
        compact.reset(RowIndex::new(3));
        compact.add_dense_column(&DenseColumn::from_vec(vec![1.0, 1.0, -48.0]));

        // These are operands from lotfi's reduced-cost recomputation. The
        // optimized upstream kernel contracts the final multiply-add, leaving
        // a small nonzero residual instead of rounding the two terms to exact
        // cancellation.
        let first = f64::from_bits(0xbf3f_7510_4d55_1d6a);
        let third = f64::from_bits(0xbee4_f8b5_88e3_68f1);
        let vector = DenseRow::from_vec(vec![first, 0.0, third]);
        let expected = (-48.0_f64).mul_add(third, first);

        assert_ne!(expected, 0.0);
        assert_eq!(
            compact.column_scalar_product(ColIndex::new(0), &vector),
            expected
        );
        assert_eq!(
            compact
                .column_scalar_product_slice(ColIndex::new(0), vector.as_slice())
                .to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn compact_column_scalar_product_rejects_negative_rows() {
        let mut compact = CompactSparseMatrix::default();
        compact.reset(RowIndex::new(1));
        compact.add_entry_to_current_column(RowIndex::new(-1), 2.0);
        compact.close_current_column();
        let _ = compact.column_scalar_product_slice(ColIndex::new(0), &[1.0]);
    }

    #[test]
    fn validated_compact_rows_match_checked_dot_and_reject_bad_rows() {
        let mut matrix = CompactSparseMatrix::default();
        matrix.reset(RowIndex::new(3));
        matrix.add_entry_to_current_column(RowIndex::new(0), 2.0);
        matrix.add_entry_to_current_column(RowIndex::new(2), -3.0);
        matrix.close_current_column();
        let matrix = Rc::new(matrix);
        let validated = CompactSparseMatrix::validated_rows(Rc::clone(&matrix), 3).unwrap();
        let rhs = [1.0, 4.0, 5.0];
        assert_eq!(
            matrix
                .column_scalar_product_validated(&validated, ColIndex::new(0), &rhs)
                .to_bits(),
            matrix
                .column_scalar_product_slice(ColIndex::new(0), &rhs)
                .to_bits()
        );
        assert!(CompactSparseMatrix::validated_rows(Rc::clone(&matrix), 2).is_none());
        assert!(
            std::panic::catch_unwind(|| {
                matrix.column_scalar_product_validated(&validated, ColIndex::new(0), &rhs[..2])
            })
            .is_err()
        );
        let other = CompactSparseMatrix::from_sparse(&SparseMatrix::new());
        assert!(
            std::panic::catch_unwind(|| {
                other.column_scalar_product_validated(&validated, ColIndex::new(0), &rhs)
            })
            .is_err()
        );
    }
}
