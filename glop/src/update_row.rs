//! Revised-simplex update-row computation.
//!
//! The stateful [`UpdateRow`] follows `ortools/glop/update_row.{h,cc}`: it
//! caches `B^-T e_p`, filters it with the configured drop tolerance, and
//! chooses among column-wise, row-wise, and hypersparse row-wise products.

use lp_data::lp_types::{ColBitVec, ColIndex, VectorIndex, deterministic_time_for_fp_operations};
use lp_data::scattered_vector::ScatteredRow;
use lp_data::sparse::SparseMatrix;
use lp_data::sparse_vector::SparseColumn;

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;
use crate::parameters::GlopParameters;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateRowAlgorithm {
    Column,
    Row,
    RowHypersparse,
}

fn column_scalar_product(column: &SparseColumn, values: &[f64]) -> f64 {
    let mut position = 0;
    let shifted_end = column.num_entries().saturating_sub(3);
    let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
    while position < shifted_end {
        result1 = column
            .coefficient(position)
            .mul_add(values[column.index(position).to_usize()], result1);
        result2 = column
            .coefficient(position + 1)
            .mul_add(values[column.index(position + 1).to_usize()], result2);
        result3 = column
            .coefficient(position + 2)
            .mul_add(values[column.index(position + 2).to_usize()], result3);
        result4 = column
            .coefficient(position + 3)
            .mul_add(values[column.index(position + 3).to_usize()], result4);
        position += 4;
    }
    let mut result = result1 + result2 + result3 + result4;
    while position < column.num_entries() {
        result = column
            .coefficient(position)
            .mul_add(values[column.index(position).to_usize()], result);
        position += 1;
    }
    result
}

#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)]
pub struct UpdateRow {
    transposed_matrix: SparseMatrix,
    unit_row_left_inverse: ScatteredRow,
    filtered_non_zeros: Vec<usize>,
    non_zero_positions: Vec<usize>,
    non_zero_position_set: ColBitVec,
    coefficients: Vec<f64>,
    left_inverse_computed_for: Option<usize>,
    update_row_computed_for: Option<usize>,
    use_transposed_matrix: bool,
    drop_tolerance: f64,
    num_operations: i64,
}

impl UpdateRow {
    #[must_use]
    pub fn new(matrix: &SparseMatrix) -> Self {
        Self {
            transposed_matrix: matrix.transpose(),
            unit_row_left_inverse: ScatteredRow::new(ColIndex::new(matrix.num_rows().value())),
            filtered_non_zeros: Vec::new(),
            non_zero_positions: Vec::new(),
            non_zero_position_set: ColBitVec::new(matrix.num_cols()),
            coefficients: vec![0.0; matrix.num_cols().to_usize()],
            left_inverse_computed_for: None,
            update_row_computed_for: None,
            use_transposed_matrix: true,
            drop_tolerance: 1e-14,
            num_operations: 0,
        }
    }

    pub fn invalidate(&mut self) {
        self.left_inverse_computed_for = None;
        self.update_row_computed_for = None;
    }

    pub fn set_parameters(&mut self, use_transposed_matrix: bool, drop_tolerance: f64) {
        self.use_transposed_matrix = use_transposed_matrix;
        self.drop_tolerance = drop_tolerance;
        self.invalidate();
    }

    pub fn set_glop_parameters(&mut self, parameters: &GlopParameters) {
        self.set_parameters(parameters.use_transposed_matrix, parameters.drop_tolerance);
    }

    /// Computes and caches `B^-T e_p`.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates a basis solve failure.
    pub fn compute_unit_row_left_inverse(
        &mut self,
        basis: &BasisRepresentation,
        leaving_row: usize,
    ) -> Result<(), FactorizationError> {
        if self.left_inverse_computed_for == Some(leaving_row) {
            return Ok(());
        }
        if leaving_row >= basis.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.unit_row_left_inverse.clear();
        basis.left_solve_for_unit_row(leaving_row, &mut self.unit_row_left_inverse)?;
        self.left_inverse_computed_for = Some(leaving_row);
        Ok(())
    }

    /// Invalidates the cache, computes `B^-T e_p`, and returns it.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates a basis solve failure.
    pub fn compute_and_get_unit_row_left_inverse(
        &mut self,
        basis: &BasisRepresentation,
        leaving_row: usize,
    ) -> Result<&[f64], FactorizationError> {
        self.invalidate();
        self.compute_unit_row_left_inverse(basis, leaving_row)?;
        Ok(self.unit_row_left_inverse.values().as_slice())
    }

    /// Computes relevant coefficients of `(e_p^T B^-1) A`.
    ///
    /// `relevant[col]` corresponds to GLOP's `GetIsRelevantBitRow()`.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates a basis solve failure.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_update_row(
        &mut self,
        basis: &BasisRepresentation,
        matrix: &SparseMatrix,
        relevant: &ColBitVec,
        num_entries_in_relevant_columns: usize,
        leaving_row: usize,
    ) -> Result<(), FactorizationError> {
        if self.update_row_computed_for == Some(leaving_row) {
            return Ok(());
        }
        let num_cols = matrix.num_cols().to_usize();
        if relevant.len().to_usize() != num_cols
            || matrix.num_rows().to_usize() != basis.dimension()
            || self.transposed_matrix.num_rows().to_usize() != num_cols
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.compute_unit_row_left_inverse(basis, leaving_row)?;
        self.update_row_computed_for = Some(leaving_row);

        if !self.use_transposed_matrix {
            self.compute_column_wise(matrix, relevant);
            return Ok(());
        }

        self.filtered_non_zeros.clear();
        let mut row_wise_entries = 0_usize;
        if self.unit_row_left_inverse.non_zeros().is_empty() {
            for (row, &value) in self
                .unit_row_left_inverse
                .values()
                .as_slice()
                .iter()
                .enumerate()
            {
                if value.abs() <= self.drop_tolerance {
                    continue;
                }
                self.filtered_non_zeros.push(row);
                row_wise_entries += self
                    .transposed_matrix
                    .column(ColIndex::from_usize(row))
                    .num_entries();
            }
        } else {
            for entry in &self.unit_row_left_inverse {
                if entry.coefficient().abs() <= self.drop_tolerance {
                    continue;
                }
                self.filtered_non_zeros.push(entry.index().to_usize());
                row_wise_entries += self.transposed_matrix.column(entry.index()).num_entries();
            }
        }

        if self.filtered_non_zeros.len() == 1 {
            self.compute_single_row(relevant);
            self.num_operations += i64::try_from(row_wise_entries).unwrap_or(i64::MAX);
            return Ok(());
        }

        let column_wise_entries = num_entries_in_relevant_columns;
        let row_wise = row_wise_entries as f64;
        if row_wise < 0.5 * column_wise_entries as f64 {
            if row_wise < 1.1 * num_cols as f64 {
                self.compute_row_wise_hypersparse(relevant);
                self.num_operations +=
                    i64::try_from(5 * row_wise_entries + num_cols / 64).unwrap_or(i64::MAX);
            } else {
                self.compute_row_wise(relevant);
                self.num_operations +=
                    i64::try_from(row_wise_entries + matrix.num_rows().to_usize())
                        .unwrap_or(i64::MAX);
            }
        } else {
            self.compute_column_wise(matrix, relevant);
            self.num_operations +=
                i64::try_from(column_wise_entries + num_cols).unwrap_or(i64::MAX);
        }
        Ok(())
    }

    #[must_use]
    pub fn is_computed_for(&self, leaving_row: usize) -> bool {
        self.update_row_computed_for == Some(leaving_row)
    }

    #[must_use]
    pub fn unit_row_left_inverse(&self) -> &[f64] {
        self.unit_row_left_inverse.values().as_slice()
    }

    #[must_use]
    pub const fn unit_row_left_inverse_scattered(&self) -> &ScatteredRow {
        &self.unit_row_left_inverse
    }

    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    #[must_use]
    pub fn non_zero_positions(&self) -> &[usize] {
        &self.non_zero_positions
    }

    #[must_use]
    pub fn coefficient(&self, column: usize) -> f64 {
        self.coefficients[column]
    }

    #[must_use]
    pub const fn num_operations(&self) -> i64 {
        self.num_operations
    }

    #[must_use]
    pub fn deterministic_time(&self) -> f64 {
        deterministic_time_for_fp_operations(self.num_operations)
    }

    /// All `UpdateRow` statistics are compiled out in the pinned release
    /// reference because it is built without `OR_STATS`.
    #[must_use]
    pub fn stat_string(&self) -> String {
        String::new()
    }

    /// Computes all nonbasic update coefficients after the left inverse exists.
    ///
    /// The basic leaving column is explicitly set to one, exactly as in
    /// GLOP's `ComputeFullUpdateRow()`.
    ///
    /// # Errors
    ///
    /// Returns a dimension error if the cached row or input masks disagree.
    pub fn compute_full_update_row(
        &self,
        matrix: &SparseMatrix,
        basis_columns: &[usize],
        not_basic: &ColBitVec,
        leaving_row: usize,
    ) -> Result<Vec<f64>, FactorizationError> {
        let num_cols = matrix.num_cols().to_usize();
        if self.left_inverse_computed_for != Some(leaving_row)
            || leaving_row >= basis_columns.len()
            || not_basic.len().to_usize() != num_cols
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut output = vec![0.0; num_cols];
        output[basis_columns[leaving_row]] = 1.0;
        for typed_column in not_basic.iter_ones() {
            let column = typed_column.to_usize();
            let coefficient = column_scalar_product(
                matrix.column(ColIndex::from_usize(column)),
                self.unit_row_left_inverse.values().as_slice(),
            );
            if coefficient.abs() > self.drop_tolerance {
                output[column] = coefficient;
            }
        }
        Ok(output)
    }

    /// Runs one selected product kernel, matching GLOP's benchmark hook.
    ///
    /// # Errors
    ///
    /// Returns a dimension error when either input slice has the wrong size.
    pub fn compute_update_row_for_benchmark(
        &mut self,
        matrix: &SparseMatrix,
        relevant: &ColBitVec,
        left_inverse: &[f64],
        algorithm: UpdateRowAlgorithm,
    ) -> Result<(), FactorizationError> {
        if left_inverse.len() != matrix.num_rows().to_usize()
            || relevant.len().to_usize() != matrix.num_cols().to_usize()
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.unit_row_left_inverse.clear();
        for (index, &value) in left_inverse.iter().enumerate() {
            if value != 0.0 {
                self.unit_row_left_inverse
                    .set(ColIndex::from_usize(index), value);
            }
        }
        self.filtered_non_zeros.clear();
        self.filtered_non_zeros.extend(
            left_inverse
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (*value != 0.0).then_some(index)),
        );
        match algorithm {
            UpdateRowAlgorithm::Column => self.compute_column_wise(matrix, relevant),
            UpdateRowAlgorithm::Row => self.compute_row_wise(relevant),
            UpdateRowAlgorithm::RowHypersparse => {
                self.compute_row_wise_hypersparse(relevant);
            }
        }
        Ok(())
    }

    fn compute_row_wise(&mut self, relevant: &ColBitVec) {
        self.coefficients.fill(0.0);
        for &row in &self.filtered_non_zeros {
            let multiplier = self.left_inverse_value(row);
            for entry in self.transposed_matrix.column(ColIndex::from_usize(row)) {
                let column = entry.index().to_usize();
                self.coefficients[column] =
                    multiplier.mul_add(entry.coefficient(), self.coefficients[column]);
            }
        }
        self.rebuild_non_zeros(relevant);
    }

    fn compute_row_wise_hypersparse(&mut self, relevant: &ColBitVec) {
        self.non_zero_position_set
            .clear_and_resize(ColIndex::from_usize(self.coefficients.len()));
        for &row in &self.filtered_non_zeros {
            let multiplier = self.left_inverse_value(row);
            for entry in self.transposed_matrix.column(ColIndex::from_usize(row)) {
                let column = entry.index().to_usize();
                let value = multiplier * entry.coefficient();
                let position = ColIndex::from_usize(column);
                if self.non_zero_position_set.contains(position) {
                    self.coefficients[column] += value;
                } else {
                    self.coefficients[column] = value;
                    self.non_zero_position_set.set(position);
                }
            }
        }
        self.non_zero_position_set.intersection(relevant);
        self.non_zero_positions.clear();
        for position in self.non_zero_position_set.iter_ones() {
            let column = position.to_usize();
            if self.coefficients[column].abs() > self.drop_tolerance {
                self.non_zero_positions.push(column);
            }
        }
    }

    fn compute_single_row(&mut self, relevant: &ColBitVec) {
        self.coefficients.fill(0.0);
        self.non_zero_positions.clear();
        let row = self.filtered_non_zeros[0];
        let multiplier = self.left_inverse_value(row);
        for entry in self.transposed_matrix.column(ColIndex::from_usize(row)) {
            let column = entry.index().to_usize();
            if !relevant.contains(ColIndex::from_usize(column)) {
                continue;
            }
            let value = multiplier * entry.coefficient();
            if value.abs() > self.drop_tolerance {
                self.coefficients[column] = value;
                self.non_zero_positions.push(column);
            }
        }
    }

    fn compute_column_wise(&mut self, matrix: &SparseMatrix, relevant: &ColBitVec) {
        self.coefficients.fill(0.0);
        self.non_zero_positions.clear();
        for typed_column in relevant.iter_ones() {
            let column = typed_column.to_usize();
            let coefficient = column_scalar_product(
                matrix.column(ColIndex::from_usize(column)),
                self.unit_row_left_inverse.values().as_slice(),
            );
            if coefficient.abs() > self.drop_tolerance {
                self.coefficients[column] = coefficient;
                self.non_zero_positions.push(column);
            }
        }
    }

    fn rebuild_non_zeros(&mut self, relevant: &ColBitVec) {
        self.non_zero_positions.clear();
        for (column, &value) in self.coefficients.iter().enumerate() {
            if relevant.contains(ColIndex::from_usize(column)) && value.abs() > self.drop_tolerance
            {
                self.non_zero_positions.push(column);
            }
        }
    }

    fn left_inverse_value(&self, row: usize) -> f64 {
        self.unit_row_left_inverse.value(ColIndex::from_usize(row))
    }
}

/// Computes row `leaving_row` of `B^-1 A` as `(B^-T e_r)^T A`.
///
/// # Errors
///
/// Returns an error for an invalid row or failed basis solve.
pub fn compute_update_row(
    basis: &BasisRepresentation,
    matrix: &SparseMatrix,
    leaving_row: usize,
) -> Result<Vec<f64>, FactorizationError> {
    if leaving_row >= basis.dimension() || matrix.num_rows().to_usize() != basis.dimension() {
        return Err(FactorizationError::DimensionMismatch);
    }
    let mut unit = vec![0.0; basis.dimension()];
    unit[leaving_row] = 1.0;
    let left_inverse = basis.transpose_solve(&unit)?;
    let mut result = vec![0.0; matrix.num_cols().to_usize()];
    for (column, value) in result.iter_mut().enumerate() {
        *value = column_scalar_product(matrix.column(ColIndex::from_usize(column)), &left_inverse);
    }
    Ok(result)
}
