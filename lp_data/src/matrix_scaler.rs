//! Sparse matrix scaling from `ortools/lp_data/matrix_scaler.{h,cc}`.

use crate::lp_types::{
    ColIndex, DenseColumn, DenseRow, Fractional, INFINITY, RowIndex, VectorIndex,
};
use crate::sparse::SparseMatrix;

/// GLOP's geometric scaling followed by row and column equilibration.
#[derive(Clone, Debug, Default)]
pub struct SparseMatrixScaler {
    row_scales: DenseColumn,
    col_scales: DenseRow,
}

#[allow(clippy::float_cmp)] // The pinned algorithm skips exact identity factors.
impl SparseMatrixScaler {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            row_scales: DenseColumn::new(),
            col_scales: DenseRow::new(),
        }
    }

    pub fn init(&mut self, matrix: &SparseMatrix) {
        self.row_scales = DenseColumn::filled(matrix.num_rows(), 1.0);
        self.col_scales = DenseRow::filled(matrix.num_cols(), 1.0);
    }

    pub fn clear(&mut self) {
        self.row_scales.clear();
        self.col_scales.clear();
    }

    #[must_use]
    pub fn row_unscaling_factor(&self, row: RowIndex) -> Fractional {
        self.row_scales
            .as_slice()
            .get(row.to_usize())
            .copied()
            .unwrap_or(1.0)
    }

    #[must_use]
    pub fn col_unscaling_factor(&self, col: ColIndex) -> Fractional {
        self.col_scales
            .as_slice()
            .get(col.to_usize())
            .copied()
            .unwrap_or(1.0)
    }

    #[must_use]
    pub fn row_scaling_factor(&self, row: RowIndex) -> Fractional {
        1.0 / self.row_unscaling_factor(row)
    }

    #[must_use]
    pub fn col_scaling_factor(&self, col: ColIndex) -> Fractional {
        1.0 / self.col_unscaling_factor(col)
    }

    #[must_use]
    pub const fn row_scales(&self) -> &DenseColumn {
        &self.row_scales
    }

    #[must_use]
    pub const fn col_scales(&self) -> &DenseRow {
        &self.col_scales
    }

    /// Applies pinned GLOP's default equilibration algorithm.
    pub fn scale(&mut self, matrix: &mut SparseMatrix) {
        debug_assert_eq!(self.row_scales.len(), matrix.num_rows());
        debug_assert_eq!(self.col_scales.len(), matrix.num_cols());
        let (minimum, maximum) = matrix.min_and_max_magnitudes();
        if minimum == 0.0 {
            return;
        }
        if maximum / minimum < 1e20 {
            for _ in 0..4 {
                let rows = self.scale_rows_geometrically(matrix);
                let columns = self.scale_columns_geometrically(matrix);
                if Self::variance_of_absolute_nonzeros(matrix) < 10.0 || (rows == 0 && columns == 0)
                {
                    break;
                }
            }
        }
        self.equilibrate_rows(matrix);
        self.equilibrate_columns(matrix);
    }

    pub fn scale_row_vector(&self, up: bool, values: &mut DenseRow) {
        let size = self
            .col_scales
            .len()
            .to_usize()
            .min(values.len().to_usize());
        for index in 0..size {
            let col = ColIndex::from_usize(index);
            if up {
                values[col] *= self.col_scales[col];
            } else {
                values[col] /= self.col_scales[col];
            }
        }
    }

    pub fn scale_column_vector(&self, up: bool, values: &mut DenseColumn) {
        let size = self
            .row_scales
            .len()
            .to_usize()
            .min(values.len().to_usize());
        for index in 0..size {
            let row = RowIndex::from_usize(index);
            if up {
                values[row] *= self.row_scales[row];
            } else {
                values[row] /= self.row_scales[row];
            }
        }
    }

    fn variance_of_absolute_nonzeros(matrix: &SparseMatrix) -> Fractional {
        let mut square_sum = 0.0;
        let mut absolute_sum = 0.0;
        let mut count = 0.0;
        for column in 0..matrix.num_cols().to_usize() {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                let magnitude = entry.coefficient().abs();
                square_sum += magnitude * magnitude;
                absolute_sum += magnitude;
                count += 1.0;
            }
        }
        if count == 0.0 {
            0.0
        } else {
            (square_sum - absolute_sum * absolute_sum / count) / count
        }
    }

    fn scale_rows_geometrically(&mut self, matrix: &mut SparseMatrix) -> usize {
        let mut maxima = DenseColumn::filled(matrix.num_rows(), 0.0);
        let mut minima = DenseColumn::filled(matrix.num_rows(), INFINITY);
        for column in 0..matrix.num_cols().to_usize() {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                let magnitude = entry.coefficient().abs();
                if magnitude != 0.0 {
                    let row = entry.index();
                    maxima[row] = maxima[row].max(magnitude);
                    minima[row] = minima[row].min(magnitude);
                }
            }
        }
        let mut factors = DenseColumn::filled(matrix.num_rows(), 1.0);
        for index in 0..matrix.num_rows().to_usize() {
            let row = RowIndex::from_usize(index);
            if maxima[row] != 0.0 {
                factors[row] = (maxima[row] * minima[row]).sqrt();
            }
        }
        self.scale_matrix_rows(matrix, &factors)
    }

    fn scale_columns_geometrically(&mut self, matrix: &mut SparseMatrix) -> usize {
        let mut scaled = 0;
        for index in 0..matrix.num_cols().to_usize() {
            let col = ColIndex::from_usize(index);
            let mut maximum: Fractional = 0.0;
            let mut minimum = INFINITY;
            for entry in matrix.column(col) {
                let magnitude = entry.coefficient().abs();
                if magnitude != 0.0 {
                    maximum = maximum.max(magnitude);
                    minimum = minimum.min(magnitude);
                }
            }
            if maximum != 0.0 {
                self.scale_matrix_column(matrix, col, (maximum * minimum).sqrt());
                scaled += 1;
            }
        }
        scaled
    }

    fn equilibrate_rows(&mut self, matrix: &mut SparseMatrix) -> usize {
        let mut maxima = DenseColumn::filled(matrix.num_rows(), 0.0);
        for column in 0..matrix.num_cols().to_usize() {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                let row = entry.index();
                maxima[row] = maxima[row].max(entry.coefficient().abs());
            }
        }
        for index in 0..matrix.num_rows().to_usize() {
            let row = RowIndex::from_usize(index);
            if maxima[row] == 0.0 {
                maxima[row] = 1.0;
            }
        }
        self.scale_matrix_rows(matrix, &maxima)
    }

    fn equilibrate_columns(&mut self, matrix: &mut SparseMatrix) -> usize {
        let mut scaled = 0;
        for index in 0..matrix.num_cols().to_usize() {
            let col = ColIndex::from_usize(index);
            let maximum = matrix
                .column(col)
                .iter()
                .map(|entry| entry.coefficient().abs())
                .fold(0.0_f64, Fractional::max);
            if maximum != 0.0 && maximum != 1.0 {
                self.scale_matrix_column(matrix, col, maximum);
                scaled += 1;
            }
        }
        scaled
    }

    fn scale_matrix_rows(&mut self, matrix: &mut SparseMatrix, factors: &DenseColumn) -> usize {
        let mut scaled = 0;
        for index in 0..matrix.num_rows().to_usize() {
            let row = RowIndex::from_usize(index);
            if factors[row] != 1.0 {
                self.row_scales[row] *= factors[row];
                scaled += 1;
            }
        }
        for index in 0..matrix.num_cols().to_usize() {
            matrix
                .mutable_column(ColIndex::from_usize(index))
                .component_wise_divide(factors);
        }
        scaled
    }

    fn scale_matrix_column(
        &mut self,
        matrix: &mut SparseMatrix,
        col: ColIndex,
        factor: Fractional,
    ) {
        self.col_scales[col] *= factor;
        matrix.mutable_column(col).divide_by_constant(factor);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn scaling_equilibrates_nonempty_rows_and_columns() {
        let mut matrix = SparseMatrix::new();
        matrix.set_num_rows(RowIndex::new(2));
        matrix.append_empty_column();
        matrix
            .mutable_column(ColIndex::new(0))
            .add_entry(RowIndex::new(0), 4.0);
        matrix
            .mutable_column(ColIndex::new(0))
            .add_entry(RowIndex::new(1), 0.25);
        let mut scaler = SparseMatrixScaler::new();
        scaler.init(&matrix);
        scaler.scale(&mut matrix);
        assert_eq!(matrix.infinity_norm(), 1.0);
        assert_eq!(matrix.one_norm(), 2.0);
    }
}
