//! Revised-simplex update-row computation.
//!
//! The stateful [`UpdateRow`] follows `ortools/glop/update_row.{h,cc}`: it
//! caches `B^-T e_p`, filters it with the configured drop tolerance, and
//! chooses among column-wise, row-wise, and hypersparse row-wise products.

use std::rc::Rc;

use lp_data::lp_types::{
    ColBitVec, ColIndex, RowIndex, VectorIndex, deterministic_time_for_fp_operations,
};
use lp_data::scattered_vector::ScatteredRow;
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix, ValidatedCompactRows};
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
    // Both entry arrays are built together. Four-entry slices prove their
    // positional accesses in range, while sparse gathers from the caller's
    // `values` remain checked. Keep GLOP's four accumulators and tail order.
    let (rows, coefficients) = column.entry_slices();
    let mut row_chunks = rows.chunks_exact(4);
    let mut coefficient_chunks = coefficients.chunks_exact(4);
    let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
    for (r, c) in row_chunks.by_ref().zip(coefficient_chunks.by_ref()) {
        result1 = c[0].mul_add(values[r[0].to_usize()], result1);
        result2 = c[1].mul_add(values[r[1].to_usize()], result2);
        result3 = c[2].mul_add(values[r[2].to_usize()], result3);
        result4 = c[3].mul_add(values[r[3].to_usize()], result4);
    }
    let mut result = result1 + result2 + result3 + result4;
    for (&row, &coefficient) in row_chunks
        .remainder()
        .iter()
        .zip(coefficient_chunks.remainder())
    {
        result = coefficient.mul_add(values[row.to_usize()], result);
    }
    result
}

#[derive(Clone, Copy)]
enum MatrixSource<'a> {
    Sparse(&'a SparseMatrix),
    Compact(&'a CompactSparseMatrix, Option<&'a ValidatedCompactRows>),
}

impl MatrixSource<'_> {
    fn num_rows(self) -> RowIndex {
        match self {
            Self::Sparse(matrix) => matrix.num_rows(),
            Self::Compact(matrix, _) => matrix.num_rows(),
        }
    }

    fn num_cols(self) -> ColIndex {
        match self {
            Self::Sparse(matrix) => matrix.num_cols(),
            Self::Compact(matrix, _) => matrix.num_cols(),
        }
    }

    #[cfg(test)]
    fn column_scalar_product(self, column: ColIndex, values: &[f64]) -> f64 {
        match self {
            Self::Sparse(matrix) => column_scalar_product(matrix.column(column), values),
            Self::Compact(matrix, Some(rows)) => {
                matrix.column_scalar_product_validated(rows, column, values)
            }
            Self::Compact(matrix, None) => matrix.column_scalar_product_slice(column, values),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)]
pub struct UpdateRow {
    transposed_matrix: Rc<CompactSparseMatrix>,
    unit_row_left_inverse: ScatteredRow,
    filtered_non_zeros: Vec<usize>,
    non_zero_positions: Vec<usize>,
    num_non_zeros: usize,
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
        Self::new_from_compact(&CompactSparseMatrix::from_sparse(matrix))
    }

    #[must_use]
    pub(crate) fn new_from_compact(matrix: &CompactSparseMatrix) -> Self {
        let mut transposed = CompactSparseMatrix::default();
        transposed.populate_from_transpose(matrix);
        Self::new_from_transposed_compact(Rc::new(transposed), matrix.num_rows(), matrix.num_cols())
    }

    #[must_use]
    pub fn new_from_transposed_compact(
        transposed_matrix: Rc<CompactSparseMatrix>,
        num_rows: RowIndex,
        num_cols: ColIndex,
    ) -> Self {
        Self {
            transposed_matrix,
            unit_row_left_inverse: ScatteredRow::new(ColIndex::new(num_rows.value())),
            filtered_non_zeros: Vec::new(),
            // GLOP keeps one slot per column and tracks the occupied prefix.
            non_zero_positions: vec![0; num_cols.to_usize()],
            num_non_zeros: 0,
            non_zero_position_set: ColBitVec::new(num_cols),
            coefficients: vec![0.0; num_cols.to_usize()],
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

    pub(crate) fn rebind_compact(&mut self, matrix: &CompactSparseMatrix) {
        let mut transposed = CompactSparseMatrix::default();
        transposed.populate_from_transpose(matrix);
        self.transposed_matrix = Rc::new(transposed);
        self.unit_row_left_inverse = ScatteredRow::new(ColIndex::new(matrix.num_rows().value()));
        self.non_zero_positions
            .resize(matrix.num_cols().to_usize(), 0);
        self.num_non_zeros = 0;
        self.non_zero_position_set = ColBitVec::new(matrix.num_cols());
        self.coefficients.resize(matrix.num_cols().to_usize(), 0.0);
        self.invalidate();
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
        self.compute_update_row_from_source(
            basis,
            MatrixSource::Sparse(matrix),
            relevant,
            num_entries_in_relevant_columns,
            leaving_row,
        )
    }

    /// Compact-matrix counterpart used by the solver's numerical source.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates a basis solve failure.
    pub fn compute_update_row_compact(
        &mut self,
        basis: &BasisRepresentation,
        matrix: &CompactSparseMatrix,
        relevant: &ColBitVec,
        num_entries_in_relevant_columns: usize,
        leaving_row: usize,
    ) -> Result<(), FactorizationError> {
        self.compute_update_row_from_source(
            basis,
            MatrixSource::Compact(matrix, None),
            relevant,
            num_entries_in_relevant_columns,
            leaving_row,
        )
    }

    /// Reuses a once-validated immutable compact matrix for the hot product.
    pub(crate) fn compute_update_row_compact_validated(
        &mut self,
        basis: &BasisRepresentation,
        matrix: &CompactSparseMatrix,
        rows: Option<&ValidatedCompactRows>,
        relevant: &ColBitVec,
        num_entries_in_relevant_columns: usize,
        leaving_row: usize,
    ) -> Result<(), FactorizationError> {
        self.compute_update_row_from_source(
            basis,
            MatrixSource::Compact(matrix, rows),
            relevant,
            num_entries_in_relevant_columns,
            leaving_row,
        )
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_update_row_from_source(
        &mut self,
        basis: &BasisRepresentation,
        matrix: MatrixSource<'_>,
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
                    .len();
            }
        } else {
            for entry in &self.unit_row_left_inverse {
                if entry.coefficient().abs() <= self.drop_tolerance {
                    continue;
                }
                self.filtered_non_zeros.push(entry.index().to_usize());
                row_wise_entries += self.transposed_matrix.column(entry.index()).len();
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
    #[inline]
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
        &self.non_zero_positions[..self.num_non_zeros]
    }

    #[must_use]
    #[inline]
    pub fn coefficient(&self, column: usize) -> f64 {
        self.coefficients[column]
    }

    #[must_use]
    #[inline]
    pub const fn num_operations(&self) -> i64 {
        self.num_operations
    }

    #[must_use]
    #[inline]
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
        matrix: &CompactSparseMatrix,
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
            let coefficient = matrix.column_scalar_product_slice(
                ColIndex::from_usize(column),
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
        self.compute_update_row_for_benchmark_source(
            MatrixSource::Sparse(matrix),
            relevant,
            left_inverse,
            algorithm,
        )
    }

    /// Runs GLOP's benchmark hook against the solver's compact matrix layout.
    ///
    /// # Errors
    ///
    /// Returns a dimension error when either input slice has the wrong size.
    #[doc(hidden)]
    pub fn compute_update_row_for_benchmark_compact(
        &mut self,
        matrix: &CompactSparseMatrix,
        validated_rows: Option<&ValidatedCompactRows>,
        relevant: &ColBitVec,
        left_inverse: &[f64],
        algorithm: UpdateRowAlgorithm,
    ) -> Result<(), FactorizationError> {
        self.compute_update_row_for_benchmark_source(
            MatrixSource::Compact(matrix, validated_rows),
            relevant,
            left_inverse,
            algorithm,
        )
    }

    fn compute_update_row_for_benchmark_source(
        &mut self,
        matrix: MatrixSource<'_>,
        relevant: &ColBitVec,
        left_inverse: &[f64],
        algorithm: UpdateRowAlgorithm,
    ) -> Result<(), FactorizationError> {
        if left_inverse.len() != matrix.num_rows().to_usize()
            || relevant.len().to_usize() != matrix.num_cols().to_usize()
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        // GLOP's benchmark hook assigns the dense lhs once, then computes
        // its nonzero positions separately; it does not scatter each entry.
        self.unit_row_left_inverse
            .values_mut()
            .as_mut_slice()
            .copy_from_slice(left_inverse);
        self.unit_row_left_inverse.non_zeros_mut().clear();
        self.filtered_non_zeros.clear();
        self.filtered_non_zeros.extend(
            left_inverse
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (*value != 0.0).then_some(index)),
        );
        match algorithm {
            UpdateRowAlgorithm::Column => {
                self.compute_column_wise(matrix, relevant);
            }
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
            for (column, value) in self
                .transposed_matrix
                .column(ColIndex::from_usize(row))
                .iter()
            {
                let column = column.to_usize();
                let slot = &mut self.coefficients[column];
                *slot = multiplier.mul_add(value, *slot);
            }
        }
        self.rebuild_non_zeros(relevant);
    }

    fn compute_row_wise_hypersparse(&mut self, relevant: &ColBitVec) {
        self.non_zero_position_set
            .clear_and_resize(ColIndex::from_usize(self.coefficients.len()));
        for &row in &self.filtered_non_zeros {
            let multiplier = self.left_inverse_value(row);
            for (column, entry_value) in self
                .transposed_matrix
                .column(ColIndex::from_usize(row))
                .iter()
            {
                let column = column.to_usize();
                let value = multiplier * entry_value;
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
        self.num_non_zeros = 0;
        for position in self.non_zero_position_set.iter_ones() {
            let column = position.to_usize();
            if self.coefficients[column].abs() > self.drop_tolerance {
                self.non_zero_positions[self.num_non_zeros] = column;
                self.num_non_zeros += 1;
            }
        }
    }

    fn compute_single_row(&mut self, relevant: &ColBitVec) {
        self.coefficients.fill(0.0);
        self.num_non_zeros = 0;
        let row = self.filtered_non_zeros[0];
        let multiplier = self.left_inverse_value(row);
        for (column, entry_value) in self
            .transposed_matrix
            .column(ColIndex::from_usize(row))
            .iter()
        {
            let column = column.to_usize();
            if !relevant.contains(ColIndex::from_usize(column)) {
                continue;
            }
            let value = multiplier * entry_value;
            if value.abs() > self.drop_tolerance {
                self.coefficients[column] = value;
                self.non_zero_positions[self.num_non_zeros] = column;
                self.num_non_zeros += 1;
            }
        }
    }

    fn compute_column_wise(&mut self, matrix: MatrixSource<'_>, relevant: &ColBitVec) {
        let left_inverse_values = self.unit_row_left_inverse.values().as_slice();
        #[cfg(feature = "phase8_update_row_probe")]
        phase8_probe::record(matrix, relevant, left_inverse_values);
        match matrix {
            MatrixSource::Compact(compact, Some(rows)) => {
                let dot = rows.bind(compact, left_inverse_values);
                Self::compute_column_wise_loop(
                    &mut self.coefficients,
                    &mut self.non_zero_positions,
                    &mut self.num_non_zeros,
                    self.drop_tolerance,
                    matrix.num_cols(),
                    relevant,
                    |column| dot.column_scalar_product(column),
                );
            }
            MatrixSource::Compact(compact, None) => {
                Self::compute_column_wise_loop(
                    &mut self.coefficients,
                    &mut self.non_zero_positions,
                    &mut self.num_non_zeros,
                    self.drop_tolerance,
                    matrix.num_cols(),
                    relevant,
                    |column| compact.column_scalar_product_slice(column, left_inverse_values),
                );
            }
            MatrixSource::Sparse(sparse) => {
                Self::compute_column_wise_loop(
                    &mut self.coefficients,
                    &mut self.non_zero_positions,
                    &mut self.num_non_zeros,
                    self.drop_tolerance,
                    matrix.num_cols(),
                    relevant,
                    |column| column_scalar_product(sparse.column(column), left_inverse_values),
                );
            }
        }
    }

    #[inline]
    fn compute_column_wise_loop(
        coefficients: &mut Vec<f64>,
        non_zero_positions: &mut [usize],
        num_non_zeros: &mut usize,
        drop_tolerance: f64,
        num_cols: ColIndex,
        relevant: &ColBitVec,
        dot: impl Fn(ColIndex) -> f64,
    ) {
        // GLOP's coefficient_.resize(num_cols, 0.0) retains existing entries.
        // Only columns reported in non_zero_positions are read by consumers;
        // clearing this full vector each pivot is both unnecessary and unlike
        // the upstream column-wise kernel.
        coefficients.resize(num_cols.to_usize(), 0.0);
        // The full-width buffer is allocated at construction (or expanded on
        // rebind) and is never shrunk after an update. Match GLOP's cursor
        // writes into its retained non_zero_position_list_.
        *num_non_zeros = 0;
        for typed_column in relevant.iter_ones() {
            let column = typed_column.to_usize();
            let coefficient = dot(typed_column);
            if coefficient.abs() > drop_tolerance {
                coefficients[column] = coefficient;
                non_zero_positions[*num_non_zeros] = column;
                *num_non_zeros += 1;
            }
        }
    }

    fn rebuild_non_zeros(&mut self, relevant: &ColBitVec) {
        self.num_non_zeros = 0;
        for (column, &value) in self.coefficients.iter().enumerate() {
            if relevant.contains(ColIndex::from_usize(column)) && value.abs() > self.drop_tolerance
            {
                self.non_zero_positions[self.num_non_zeros] = column;
                self.num_non_zeros += 1;
            }
        }
    }

    fn left_inverse_value(&self, row: usize) -> f64 {
        self.unit_row_left_inverse.value(ColIndex::from_usize(row))
    }
}

#[cfg(feature = "phase8_update_row_probe")]
mod phase8_probe {
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{ColBitVec, ColIndex, MatrixSource, VectorIndex};

    struct Config {
        directory: PathBuf,
        calls: Vec<usize>,
    }

    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    static CALL: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn record(matrix: MatrixSource<'_>, relevant: &ColBitVec, lhs: &[f64]) {
        let Some(config) = CONFIG.get_or_init(|| {
            let directory = std::env::var_os("GLOPRS_UPDATE_ROW_CAPTURE_DIR")?;
            let calls = std::env::var("GLOPRS_UPDATE_ROW_CAPTURE_CALLS")
                .unwrap_or_else(|_| "0".to_owned())
                .split(',')
                .filter_map(|value| value.parse().ok())
                .collect();
            Some(Config {
                directory: PathBuf::from(directory),
                calls,
            })
        }) else {
            return;
        };
        let call = CALL.fetch_add(1, Ordering::Relaxed);
        if !config.calls.contains(&call) {
            return;
        }
        fs::create_dir_all(&config.directory).expect("create update-row probe directory");
        let mut file = File::create(config.directory.join(format!("call-{call:06}.txt")))
            .expect("create update-row probe");
        writeln!(file, "{} {}", lhs.len(), matrix.num_cols().to_usize()).unwrap();
        for column in 0..matrix.num_cols().to_usize() {
            match matrix {
                MatrixSource::Compact(compact, _) => {
                    let entries = compact.column(ColIndex::from_usize(column));
                    writeln!(file, "{}", entries.len()).unwrap();
                    for (row, coefficient) in entries.iter() {
                        writeln!(file, "{} {coefficient:.17e}", row.to_usize()).unwrap();
                    }
                }
                MatrixSource::Sparse(sparse) => {
                    let entries = sparse.column(ColIndex::from_usize(column));
                    writeln!(file, "{}", entries.num_entries()).unwrap();
                    for entry in entries {
                        writeln!(
                            file,
                            "{} {:.17e}",
                            entry.index().to_usize(),
                            entry.coefficient()
                        )
                        .unwrap();
                    }
                }
            }
        }
        for value in lhs {
            writeln!(file, "{value:.17e}").unwrap();
        }
        let positions: Vec<_> = relevant.iter_ones().collect();
        writeln!(file, "{}", positions.len()).unwrap();
        for column in positions {
            writeln!(file, "{}", column.to_usize()).unwrap();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_dot_keeps_four_accumulator_order_and_checks_rhs_indices() {
        let values = [1.0e16, -1.0e16, 3.0, 0.25];
        for count in 0..12 {
            let mut column = SparseColumn::new();
            for position in 0..count {
                column.add_entry(
                    RowIndex::from_usize(position % values.len()),
                    (f64::from(u32::try_from(position).unwrap()) - 4.0) * 0.125,
                );
            }
            let mut sums = [0.0_f64; 4];
            for position in 0..count / 4 * 4 {
                sums[position % 4] = column.coefficient(position).mul_add(
                    values[column.index(position).to_usize()],
                    sums[position % 4],
                );
            }
            let mut expected = sums[0] + sums[1] + sums[2] + sums[3];
            for position in count / 4 * 4..count {
                expected = column
                    .coefficient(position)
                    .mul_add(values[column.index(position).to_usize()], expected);
            }
            assert_eq!(
                column_scalar_product(&column, &values).to_bits(),
                expected.to_bits()
            );
        }
        let mut invalid = SparseColumn::new();
        invalid.add_entry(RowIndex::from_usize(values.len()), 1.0);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                column_scalar_product(&invalid, &values)
            }))
            .is_err()
        );
    }

    #[test]
    fn advanced_update_row_shares_the_transposed_compact_matrix() {
        let mut transpose = CompactSparseMatrix::default();
        transpose.reset(RowIndex::new(2));
        transpose.add_entry_to_current_column(RowIndex::new(0), 3.0);
        transpose.close_current_column();
        let transpose = Rc::new(transpose);
        let update_row = UpdateRow::new_from_transposed_compact(
            Rc::clone(&transpose),
            RowIndex::new(1),
            ColIndex::new(2),
        );
        assert!(Rc::ptr_eq(&update_row.transposed_matrix, &transpose));
    }

    #[test]
    fn nonzero_position_storage_is_full_width_and_reused() {
        let mut matrix = CompactSparseMatrix::default();
        matrix.reset(RowIndex::new(2));
        for _ in 0..4 {
            matrix.close_current_column();
        }
        let mut update_row = UpdateRow::new_from_compact(&matrix);
        let storage = update_row.non_zero_positions.as_ptr();
        assert_eq!(update_row.non_zero_positions.len(), 4);
        let mut relevant = ColBitVec::new(ColIndex::new(4));
        for column in 0..4 {
            relevant.set(ColIndex::new(column));
        }
        for _ in 0..10 {
            UpdateRow::compute_column_wise_loop(
                &mut update_row.coefficients,
                &mut update_row.non_zero_positions,
                &mut update_row.num_non_zeros,
                1e-14,
                ColIndex::new(4),
                &relevant,
                |column| if column.value() % 2 == 0 { 1.0 } else { 0.0 },
            );
            assert_eq!(update_row.non_zero_positions(), &[0, 2]);
            assert_eq!(update_row.non_zero_positions.len(), 4);
            assert_eq!(update_row.non_zero_positions.as_ptr(), storage);
        }
        update_row.rebind_compact(&matrix);
        assert_eq!(update_row.non_zero_positions.as_ptr(), storage);
        assert!(update_row.non_zero_positions().is_empty());
        update_row
            .coefficients
            .copy_from_slice(&[0.0, 2.0, 0.0, 3.0]);
        update_row.rebuild_non_zeros(&relevant);
        assert_eq!(update_row.non_zero_positions(), &[1, 3]);
        assert_eq!(update_row.non_zero_positions.as_ptr(), storage);
    }

    #[test]
    fn validated_compact_update_row_dot_matches_checked_bits() {
        let mut matrix = CompactSparseMatrix::default();
        matrix.reset(RowIndex::new(4));
        for (row, value) in [(0, 1.25), (2, -0.5), (3, 3.0), (1, 0.75)] {
            matrix.add_entry_to_current_column(RowIndex::from_usize(row), value);
        }
        matrix.close_current_column();
        let matrix = Rc::new(matrix);
        let rows = CompactSparseMatrix::validated_rows(Rc::clone(&matrix), 4).unwrap();
        let values = [1.0e16, 0.5, -2.0, -1.0e16];
        let column = ColIndex::new(0);
        assert_eq!(
            MatrixSource::Compact(&matrix, Some(&rows))
                .column_scalar_product(column, &values)
                .to_bits(),
            MatrixSource::Compact(&matrix, None)
                .column_scalar_product(column, &values)
                .to_bits()
        );
    }
}
