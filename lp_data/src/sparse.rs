//! Column-oriented sparse matrices.
//!
//! This is the Phase-1 subset of upstream `ortools/lp_data/sparse.{h,cc}`.

use crate::lp_types::{ColIndex, DenseBooleanRow, EntryIndex, Fractional, RowIndex, VectorIndex};
use crate::permutation::RowPermutation;
use crate::sparse_vector::SparseColumn;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseMatrix {
    columns: Vec<SparseColumn>,
    num_rows: RowIndex,
}

#[derive(Clone, Debug)]
pub struct CompactSparseMatrix {
    starts: Vec<usize>,
    rows: Vec<RowIndex>,
    coefficients: Vec<Fractional>,
    num_rows: RowIndex,
}

impl CompactSparseMatrix {
    #[must_use]
    pub fn from_sparse(matrix: &SparseMatrix) -> Self {
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
        }
    }

    #[must_use]
    pub const fn num_rows(&self) -> RowIndex {
        self.num_rows
    }

    #[must_use]
    pub fn num_cols(&self) -> ColIndex {
        ColIndex::from_usize(self.starts.len() - 1)
    }

    #[must_use]
    pub fn column(&self, column: ColIndex) -> CompactColumn<'_> {
        let start = self.starts[column.to_usize()];
        let end = self.starts[column.to_usize() + 1];
        CompactColumn {
            rows: &self.rows[start..end],
            coefficients: &self.coefficients[start..end],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompactColumn<'a> {
    rows: &'a [RowIndex],
    coefficients: &'a [Fractional],
}

impl<'a> CompactColumn<'a> {
    pub fn iter(self) -> impl Iterator<Item = (RowIndex, Fractional)> + 'a {
        self.rows
            .iter()
            .copied()
            .zip(self.coefficients.iter().copied())
    }

    #[must_use]
    pub const fn len(self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.rows.is_empty()
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
    pub fn from_basis(matrix: &'a SparseMatrix, basis: &[ColIndex]) -> Self {
        Self {
            columns: basis.iter().map(|&column| matrix.column(column)).collect(),
            num_rows: matrix.num_rows(),
        }
    }

    #[must_use]
    pub const fn num_rows(&self) -> RowIndex {
        self.num_rows
    }

    #[must_use]
    pub fn num_cols(&self) -> ColIndex {
        ColIndex::from_usize(self.columns.len())
    }

    #[must_use]
    pub fn column(&self, column: ColIndex) -> &'a SparseColumn {
        self.columns[column.to_usize()]
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

    pub fn clear(&mut self) {
        self.columns.clear();
        self.num_rows = RowIndex::new(0);
    }

    #[must_use]
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

    pub fn set_num_rows(&mut self, num_rows: RowIndex) {
        debug_assert!(num_rows.value() >= 0);
        self.num_rows = num_rows;
    }

    pub fn append_empty_column(&mut self) -> ColIndex {
        let index = ColIndex::from_usize(self.columns.len());
        self.columns.push(SparseColumn::new());
        index
    }

    pub fn populate_from_zero(&mut self, num_rows: RowIndex, num_cols: ColIndex) {
        self.num_rows = num_rows;
        self.columns = vec![SparseColumn::new(); num_cols.to_usize()];
    }

    #[must_use]
    pub const fn num_rows(&self) -> RowIndex {
        self.num_rows
    }

    #[must_use]
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
    pub fn column(&self, column: ColIndex) -> &SparseColumn {
        &self.columns[column.to_usize()]
    }

    pub fn mutable_column(&mut self, column: ColIndex) -> &mut SparseColumn {
        &mut self.columns[column.to_usize()]
    }

    pub fn replace_column(&mut self, column: ColIndex, replacement: SparseColumn) {
        self.columns[column.to_usize()] = replacement;
    }

    #[must_use]
    pub fn look_up_value(&self, row: RowIndex, column: ColIndex) -> Fractional {
        self.column(column).look_up_coefficient(row)
    }

    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut result = Self::new();
        result.populate_from_zero(
            RowIndex::new(self.num_cols().value()),
            ColIndex::new(self.num_rows.value()),
        );
        for (column_position, column) in self.columns.iter().enumerate() {
            let transposed_row = RowIndex::new(ColIndex::from_usize(column_position).value());
            for entry in column {
                result
                    .mutable_column(ColIndex::new(entry.index().value()))
                    .add_entry(transposed_row, entry.coefficient());
            }
        }
        result.clean_up();
        result
    }

    pub fn apply_row_permutation(&mut self, permutation: &RowPermutation) {
        for column in &mut self.columns {
            column.apply_index_permutation(permutation);
            column.clean_up();
        }
    }

    pub fn delete_columns(&mut self, deleted: &DenseBooleanRow) {
        debug_assert_eq!(deleted.as_slice().len(), self.columns.len());
        let mut position = 0;
        self.columns.retain(|_| {
            let keep = !deleted[ColIndex::from_usize(position)];
            position += 1;
            keep
        });
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
        let view = MatrixView::from_basis(&matrix, &[ColIndex::new(1)]);
        assert_eq!(view.num_cols(), ColIndex::new(1));
        assert_eq!(
            view.column(ColIndex::new(0))
                .look_up_coefficient(RowIndex::new(0)),
            -3.0
        );
    }
}
