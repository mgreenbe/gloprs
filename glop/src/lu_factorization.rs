//! LU factorization with threshold Markowitz pivoting.
//!
//! The numerical elimination follows upstream `lu_factorization` and
//! `markowitz`: permutations satisfy `A[row_permutation, column_permutation] =
//! L U`, `L` has an implicit unit diagonal, and pivot ties are deterministic.

use std::cell::RefCell;
use std::fmt;

use lp_data::lp_types::{
    ColIndex, RowIndex, RowToColMapping, VectorIndex, deterministic_time_for_fp_operations,
};
use lp_data::lp_utils::squared_norm_and_reset_to_zero;
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow, ScatteredVector};
use lp_data::sparse::SparseMatrix;
use lp_data::sparse_vector::SparseColumn;
use lp_data::triangular_matrix::{Triangle, TriangularMatrix};

use crate::markowitz;
use crate::parameters::GlopParameters;
use crate::stats::{DistributionKind, StatsGroup};

#[derive(Clone, Debug, PartialEq)]
pub enum FactorizationError {
    NonSquare { rows: usize, columns: usize },
    NonFinite,
    Singular { step: usize },
    IllConditioned { upper_bound: f64 },
    DimensionMismatch,
    InvalidParameters(String),
}

impl fmt::Display for FactorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquare { rows, columns } => {
                write!(formatter, "matrix is {rows} by {columns}, not square")
            }
            Self::NonFinite => formatter.write_str("matrix contains a nonfinite coefficient"),
            Self::Singular { step } => write!(formatter, "matrix is singular at step {step}"),
            Self::IllConditioned { upper_bound } => write!(
                formatter,
                "The matrix condition number upper bound is too high: {upper_bound}"
            ),
            Self::DimensionMismatch => formatter.write_str("right-hand side dimension mismatch"),
            Self::InvalidParameters(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for FactorizationError {}

#[derive(Clone, Debug)]
pub struct LuFactorization {
    is_identity_factorization: bool,
    // Both factors use compact column-oriented storage. L's unit diagonal is
    // implicit, as it is in upstream TriangularMatrix.
    lower: TriangularMatrix,
    upper: TriangularMatrix,
    transpose_lower: TriangularMatrix,
    transpose_upper: TriangularMatrix,
    // Input index -> factor position, matching GLOP's row_perm_/col_perm_.
    row_permutation: Vec<usize>,
    column_permutation: Vec<usize>,
    // Factor position -> input index.
    inverse_row_permutation: Vec<usize>,
    inverse_column_permutation: Vec<usize>,
    parameters: GlopParameters,
    deterministic_time_of_last_factorization: f64,
    dense_zero_scratchpad: RefCell<Vec<f64>>,
    non_zero_rows: RefCell<Vec<RowIndex>>,
    markowitz_stats: StatsGroup,
    markowitz_workspace: markowitz::MarkowitzWorkspace,
}

impl Default for LuFactorization {
    fn default() -> Self {
        Self::new()
    }
}

impl LuFactorization {
    /// Creates GLOP's cleared state: an identity factorization that can solve
    /// vectors of any dimension without storing factors.
    #[must_use]
    pub fn new() -> Self {
        Self {
            is_identity_factorization: true,
            lower: TriangularMatrix::empty(Triangle::Lower, true),
            upper: TriangularMatrix::empty(Triangle::Upper, false),
            transpose_lower: TriangularMatrix::empty(Triangle::Upper, true),
            transpose_upper: TriangularMatrix::empty(Triangle::Lower, false),
            row_permutation: Vec::new(),
            column_permutation: Vec::new(),
            inverse_row_permutation: Vec::new(),
            inverse_column_permutation: Vec::new(),
            parameters: GlopParameters::default(),
            deterministic_time_of_last_factorization: 0.0,
            dense_zero_scratchpad: RefCell::new(Vec::new()),
            non_zero_rows: RefCell::new(Vec::new()),
            markowitz_stats: StatsGroup::new("Markowitz"),
            markowitz_workspace: markowitz::MarkowitzWorkspace::default(),
        }
    }

    /// Resets to the dimension-independent identity factorization.
    pub fn clear(&mut self) {
        self.is_identity_factorization = true;
        self.lower.reset(0, 0);
        self.upper.reset(0, 0);
        self.transpose_lower.reset(0, 0);
        self.transpose_upper.reset(0, 0);
        self.row_permutation.clear();
        self.column_permutation.clear();
        self.inverse_row_permutation.clear();
        self.inverse_column_permutation.clear();
        self.dense_zero_scratchpad.get_mut().clear();
        self.non_zero_rows.get_mut().clear();
        // GLOP's Clear() deliberately retains both SetParameters() state and
        // Markowitz's deterministic time for the last factorization.
    }

    #[must_use]
    pub const fn is_identity_factorization(&self) -> bool {
        self.is_identity_factorization
    }

    /// Factorizes a square sparse matrix using threshold Markowitz pivots.
    ///
    /// # Errors
    ///
    /// Returns an error for nonsquare, nonfinite, or singular matrices.
    pub fn factorize(
        matrix: &SparseMatrix,
        pivot_threshold: f64,
    ) -> Result<Self, FactorizationError> {
        let parameters = GlopParameters {
            lu_factorization_pivot_threshold: pivot_threshold,
            ..GlopParameters::default()
        };
        Self::factorize_with_parameters(matrix, &parameters)
    }

    /// Factorizes using the same parameter bundle passed through GLOP's
    /// `LuFactorization::SetParameters()`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-parameter, shape, coefficient, or singularity error.
    pub fn factorize_with_parameters(
        matrix: &SparseMatrix,
        parameters: &GlopParameters,
    ) -> Result<Self, FactorizationError> {
        let mut result = Self::new();
        result.compute_factorization_with_parameters(matrix, parameters)?;
        Ok(result)
    }

    pub(crate) fn compute_factorization_with_parameters(
        &mut self,
        matrix: &SparseMatrix,
        parameters: &GlopParameters,
    ) -> Result<(), FactorizationError> {
        parameters
            .validate()
            .map_err(FactorizationError::InvalidParameters)?;
        let rows = matrix.num_rows().to_usize();
        let columns = matrix.num_cols().to_usize();
        if rows != columns {
            return Err(FactorizationError::NonSquare { rows, columns });
        }
        for column in 0..columns {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                if !entry.coefficient().is_finite() {
                    return Err(FactorizationError::NonFinite);
                }
            }
        }
        let factors = markowitz::factorize(
            matrix,
            parameters,
            &mut self.markowitz_workspace,
            &mut self.lower,
            &mut self.upper,
        )
        .map_err(|step| FactorizationError::Singular { step })?;
        self.install_sparse_lu(factors, rows, columns, parameters);
        Ok(())
    }

    pub(crate) fn factorize_selected_with_parameters(
        matrix: &SparseMatrix,
        selected_columns: &[usize],
        parameters: &GlopParameters,
    ) -> Result<Self, FactorizationError> {
        let mut result = Self::new();
        result.compute_factorization_selected_with_parameters(
            matrix,
            selected_columns,
            parameters,
        )?;
        Ok(result)
    }

    pub(crate) fn compute_factorization_selected_with_parameters(
        &mut self,
        matrix: &SparseMatrix,
        selected_columns: &[usize],
        parameters: &GlopParameters,
    ) -> Result<(), FactorizationError> {
        parameters
            .validate()
            .map_err(FactorizationError::InvalidParameters)?;
        let rows = matrix.num_rows().to_usize();
        let columns = selected_columns.len();
        if rows != columns {
            return Err(FactorizationError::NonSquare { rows, columns });
        }
        for &column in selected_columns {
            if column >= matrix.num_cols().to_usize() {
                return Err(FactorizationError::DimensionMismatch);
            }
            for entry in matrix.column(ColIndex::from_usize(column)) {
                if !entry.coefficient().is_finite() {
                    return Err(FactorizationError::NonFinite);
                }
            }
        }
        let factors = markowitz::factorize_selected(
            matrix,
            selected_columns,
            parameters,
            &mut self.markowitz_workspace,
            &mut self.lower,
            &mut self.upper,
        )
        .map_err(|step| FactorizationError::Singular { step })?;
        self.install_sparse_lu(factors, rows, columns, parameters);
        Ok(())
    }

    fn install_sparse_lu(
        &mut self,
        factors: markowitz::SparseLu,
        rows: usize,
        columns: usize,
        parameters: &GlopParameters,
    ) {
        let deterministic_time_of_last_factorization =
            deterministic_time_for_fp_operations(factors.num_fp_operations);
        self.lower.transpose_into(&mut self.transpose_lower);
        self.upper.transpose_into(&mut self.transpose_upper);
        let inverse_row_permutation = factors.row_permutation;
        let inverse_column_permutation = factors.column_permutation;
        let mut row_permutation = vec![0; rows];
        let mut column_permutation = vec![0; columns];
        for (position, &row) in inverse_row_permutation.iter().enumerate() {
            row_permutation[row] = position;
        }
        for (position, &column) in inverse_column_permutation.iter().enumerate() {
            column_permutation[column] = position;
        }
        if let Some(stats) = factors.stats {
            self.markowitz_stats.add(
                "basis_singleton_column_ratio",
                DistributionKind::Ratio,
                stats.basis_singleton_column_ratio,
            );
            self.markowitz_stats.add(
                "basis_residual_singleton_column_ratio",
                DistributionKind::Ratio,
                stats.basis_residual_singleton_column_ratio,
            );
            self.markowitz_stats.add(
                "pivots_without_fill_in_ratio",
                DistributionKind::Ratio,
                stats.pivots_without_fill_in_ratio,
            );
            self.markowitz_stats.add(
                "degree_two_pivot_columns",
                DistributionKind::Ratio,
                stats.degree_two_pivot_columns,
            );
        }
        self.is_identity_factorization = false;
        self.row_permutation = row_permutation;
        self.column_permutation = column_permutation;
        self.inverse_row_permutation = inverse_row_permutation;
        self.inverse_column_permutation = inverse_column_permutation;
        self.parameters = parameters.clone();
        self.deterministic_time_of_last_factorization = deterministic_time_of_last_factorization;
        let scratchpad = self.dense_zero_scratchpad.get_mut();
        scratchpad.resize(rows, 0.0);
        scratchpad.fill(0.0);
        self.non_zero_rows.get_mut().clear();
    }

    /// Finds a stable independent subset of `candidates` and completes it
    /// with the matrix's trailing identity/slack columns, in GLOP's order.
    ///
    /// Candidate indices refer to columns of `matrix`. As in GLOP, the matrix
    /// is expected to end with one slack column per row.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch for an invalid candidate or absent slack
    /// block, or an invalid-parameter error.
    pub fn compute_initial_basis(
        matrix: &SparseMatrix,
        candidates: &[ColIndex],
        parameters: &GlopParameters,
    ) -> Result<RowToColMapping, FactorizationError> {
        Self::compute_initial_basis_with_operations(matrix, candidates, parameters)
            .map(|(basis, _)| basis)
    }

    pub(crate) fn compute_initial_basis_with_operations(
        matrix: &SparseMatrix,
        candidates: &[ColIndex],
        parameters: &GlopParameters,
    ) -> Result<(RowToColMapping, i64), FactorizationError> {
        parameters
            .validate()
            .map_err(FactorizationError::InvalidParameters)?;
        let num_rows = matrix.num_rows().to_usize();
        let num_columns = matrix.num_cols().to_usize();
        if num_columns < num_rows
            || candidates
                .iter()
                .any(|column| column.to_usize() >= num_columns)
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        let candidate_indices: Vec<_> = candidates.iter().map(|column| column.to_usize()).collect();
        let (pivot_rows, pivot_columns, num_fp_operations) =
            markowitz::compute_pivot_sequence_with_operations(
                matrix,
                &candidate_indices,
                parameters,
            );
        let mut pivoted_rows = vec![false; num_rows];
        for row in pivot_rows {
            pivoted_rows[row] = true;
        }
        let mut selected_candidates = vec![false; candidates.len()];
        for column in pivot_columns {
            selected_candidates[column] = true;
        }

        let first_slack = num_columns - num_rows;
        let mut basis = RowToColMapping::new();
        for (row, &pivoted) in pivoted_rows.iter().enumerate() {
            if !pivoted {
                basis.push(ColIndex::from_usize(first_slack + row));
            }
        }
        for (position, &selected) in selected_candidates.iter().enumerate() {
            if selected {
                basis.push(candidates[position]);
            }
        }
        Ok((basis, num_fp_operations))
    }

    #[doc(hidden)]
    #[must_use]
    pub fn initial_basis_pivot_sequence(
        matrix: &SparseMatrix,
        candidates: &[ColIndex],
        parameters: &GlopParameters,
    ) -> Vec<(usize, ColIndex)> {
        let candidate_indices: Vec<_> = candidates.iter().map(|column| column.to_usize()).collect();
        let (rows, columns) =
            markowitz::compute_pivot_sequence(matrix, &candidate_indices, parameters);
        rows.into_iter()
            .zip(columns)
            .map(|(row, column)| (row, candidates[column]))
            .collect()
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.lower.dimension()
    }

    #[must_use]
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_permutation
    }

    #[must_use]
    pub fn column_permutation(&self) -> &[usize] {
        &self.column_permutation
    }

    #[must_use]
    pub fn inverse_row_permutation(&self) -> &[usize] {
        &self.inverse_row_permutation
    }

    #[must_use]
    pub fn inverse_column_permutation(&self) -> &[usize] {
        &self.inverse_column_permutation
    }

    /// Clears `Q` after the caller has incorporated it into its basis mapping.
    pub fn set_column_permutation_to_identity(&mut self) {
        self.column_permutation.clear();
        self.inverse_column_permutation.clear();
    }

    #[must_use]
    pub const fn pivot_threshold(&self) -> f64 {
        self.parameters.lu_factorization_pivot_threshold
    }

    #[must_use]
    pub const fn parameters(&self) -> &GlopParameters {
        &self.parameters
    }

    #[must_use]
    pub const fn deterministic_time_of_last_factorization(&self) -> f64 {
        self.deterministic_time_of_last_factorization
    }

    #[must_use]
    pub fn stat_string(&self) -> String {
        // LuFactorization's own two distributions are guarded by OR_STATS in
        // the pinned release build; Markowitz's structural ratios are not.
        self.markowitz_stats.stat_string()
    }

    /// Solves `A x = rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error when the right-hand side has the wrong dimension.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(rhs.to_vec());
        }
        let lower_solution = self.right_solve_lower(rhs)?;
        self.right_solve_upper(&lower_solution)
    }

    /// Solves the permuted lower system `L z = P rhs`.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn right_solve_lower(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(rhs.to_vec());
        }
        let n = self.dimension();
        if rhs.len() != n {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut work: Vec<f64> = self
            .inverse_row_permutation
            .iter()
            .map(|&row| rhs[row])
            .collect();
        self.lower
            .solve(&mut work)
            .map_err(|_| FactorizationError::DimensionMismatch)?;
        Ok(work)
    }

    /// Solves `U t = rhs` and maps `t` back through the column permutation.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn right_solve_upper(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(rhs.to_vec());
        }
        if rhs.len() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut work = rhs.to_vec();
        // GLOP performs this solve through the explicitly stored transpose of
        // U. Solving U directly is algebraically equivalent, but traverses the
        // entries in a different order and therefore changes rounding.
        self.transpose_upper
            .transpose_solve(&mut work)
            .map_err(|_| FactorizationError::DimensionMismatch)?;
        if self.inverse_column_permutation.is_empty() {
            return Ok(work);
        }
        let mut solution = vec![0.0; self.dimension()];
        for (position, &column) in self.inverse_column_permutation.iter().enumerate() {
            solution[column] = work[position];
        }
        Ok(solution)
    }

    /// Solves `A^T x = rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error when the right-hand side has the wrong dimension.
    pub fn transpose_solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(rhs.to_vec());
        }
        let upper_solution = self.left_solve_upper(rhs)?;
        self.left_solve_lower(&upper_solution)
    }

    /// Solves `A x = rhs` while maintaining a sparse superset of result rows.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn solve_with_nonzeros(&self, rhs: &mut ScatteredColumn) -> Result<(), FactorizationError> {
        self.right_solve_lower_with_nonzeros(rhs)?;
        self.right_solve_upper_with_nonzeros(rhs)
    }

    /// Applies `L^-1 P` while preserving sparse positions.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn right_solve_lower_with_nonzeros(
        &self,
        rhs: &mut ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        if self.is_identity_factorization {
            return Ok(());
        }
        if rhs.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.permute_scattered(rhs, &self.row_permutation);
        {
            let (_values, non_zeros) = rhs.mutable_parts();
            self.lower
                .compute_rows_to_consider_in_sorted_order(non_zeros);
        }
        rhs.mark_non_zeros_sorted();
        {
            let (values, non_zeros) = rhs.mutable_parts();
            if non_zeros.is_empty() {
                self.lower
                    .solve(values)
                    .map_err(|_| FactorizationError::DimensionMismatch)?;
            } else {
                self.lower.hyper_sparse_solve(values, non_zeros);
            }
        }
        Ok(())
    }

    /// Applies `L^-1 P` directly to a sparse problem column.
    ///
    /// This is GLOP's `RightSolveLForColumnView()`, including its specialized
    /// dense fallback beginning at the first relevant nonidentity column.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn right_solve_lower_for_column(
        &self,
        column: &SparseColumn,
        result: &mut ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        result.clear();
        result.clear_sparse_mask();
        if self.is_identity_factorization {
            let (values, non_zeros) = result.mutable_parts();
            for entry in column {
                if entry.index().to_usize() >= values.len() {
                    return Err(FactorizationError::DimensionMismatch);
                }
                values[entry.index().to_usize()] = entry.coefficient();
                non_zeros.push(entry.index());
            }
            return Ok(());
        }
        if result.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut first_column_to_consider = self.dimension();
        let limit = self.lower.first_non_identity_column();
        {
            let (values, non_zeros) = result.mutable_parts();
            for entry in column {
                if entry.index().to_usize() >= self.dimension() {
                    return Err(FactorizationError::DimensionMismatch);
                }
                let permuted_row = self.row_permutation[entry.index().to_usize()];
                values[permuted_row] = entry.coefficient();
                non_zeros.push(RowIndex::from_usize(permuted_row));
                if permuted_row >= limit && !self.lower.column_is_diagonal_only(permuted_row) {
                    first_column_to_consider = first_column_to_consider.min(permuted_row);
                }
            }
            self.lower
                .compute_rows_to_consider_in_sorted_order(non_zeros);
        }
        result.mark_non_zeros_sorted();
        let (values, non_zeros) = result.mutable_parts();
        if non_zeros.is_empty() {
            self.lower
                .lower_solve_starting_at(first_column_to_consider, values)
                .map_err(|_| FactorizationError::DimensionMismatch)?;
        } else {
            self.lower.hyper_sparse_solve(values, non_zeros);
        }
        // Like GLOP, the position list is authoritative here. Leave Rust's
        // auxiliary membership cache cleared so a later rank-one update can
        // record a structurally reached zero that becomes numerically nonzero.
        result.clear_sparse_mask();
        Ok(())
    }

    /// Applies `Q U^-1` while preserving sparse positions.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn right_solve_upper_with_nonzeros(
        &self,
        rhs: &mut ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        if self.is_identity_factorization {
            return Ok(());
        }
        if rhs.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        {
            let (_values, non_zeros) = rhs.mutable_parts();
            // GLOP uses U to compute the structural closure, but performs the
            // numerical solve as a transpose solve on its explicitly stored
            // transpose. Preserve that split: the two formulations are
            // algebraically equivalent but do not have the same floating-
            // point traversal order.
            self.upper
                .compute_rows_to_consider_in_sorted_order(non_zeros);
        }
        rhs.mark_non_zeros_sorted();
        {
            let (values, non_zeros) = rhs.mutable_parts();
            if non_zeros.is_empty() {
                self.transpose_upper
                    .transpose_solve(values)
                    .map_err(|_| FactorizationError::DimensionMismatch)?;
            } else {
                self.transpose_upper
                    .transpose_hyper_sparse_solve_with_reversed_nonzeros(values, non_zeros);
            }
        }
        self.permute_scattered(rhs, &self.inverse_column_permutation);
        Ok(())
    }

    /// Solves `A^T x = rhs` while maintaining sparse result positions.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn transpose_solve_with_nonzeros(
        &self,
        rhs: &mut ScatteredRow,
    ) -> Result<(), FactorizationError> {
        self.left_solve_upper_with_nonzeros(rhs)?;
        self.left_solve_lower_with_nonzeros(rhs)
    }

    /// Applies `U^-T Q^T` while preserving sparse positions.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn left_solve_upper_with_nonzeros(
        &self,
        rhs: &mut ScatteredRow,
    ) -> Result<(), FactorizationError> {
        if self.is_identity_factorization {
            return Ok(());
        }
        if rhs.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.permute_scattered(rhs, &self.column_permutation);
        {
            let (_values, non_zeros) = rhs.mutable_parts();
            self.transpose_upper
                .compute_rows_to_consider_in_sorted_order(non_zeros);
        }
        rhs.mark_non_zeros_sorted();
        {
            let (values, non_zeros) = rhs.mutable_parts();
            if non_zeros.is_empty() {
                self.upper
                    .transpose_solve(values)
                    .map_err(|_| FactorizationError::DimensionMismatch)?;
            } else {
                self.upper.transpose_hyper_sparse_solve(values, non_zeros);
            }
        }
        Ok(())
    }

    /// Applies `U^-T Q^T` to a unit row, using the same specialized path as
    /// GLOP's `LeftSolveUForUnitRow()`.
    ///
    /// # Errors
    ///
    /// Returns a dimension error for an invalid row or result vector.
    pub fn left_solve_upper_for_unit_row(
        &self,
        row: usize,
        result: &mut ScatteredRow,
    ) -> Result<usize, FactorizationError> {
        if self.is_identity_factorization {
            if row >= result.len().to_usize() {
                return Err(FactorizationError::DimensionMismatch);
            }
            debug_assert!(result.values().as_slice().iter().all(|&value| value == 0.0));
            debug_assert!(result.non_zeros().is_empty());
            let column = ColIndex::from_usize(row);
            let (values, non_zeros) = result.mutable_parts();
            values[row] = 1.0;
            non_zeros.push(column);
            return Ok(row);
        }
        if row >= self.dimension() || result.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        debug_assert!(result.values().as_slice().iter().all(|&value| value == 0.0));
        debug_assert!(result.non_zeros().is_empty());
        let permuted_row = if self.column_permutation.is_empty() {
            row
        } else {
            self.column_permutation[row]
        };
        {
            let (values, non_zeros) = result.mutable_parts();
            values[permuted_row] = 1.0;
            non_zeros.push(ColIndex::from_usize(permuted_row));
        }
        if self.transpose_upper.column_is_diagonal_only(permuted_row) {
            let diagonal = self.transpose_upper.diagonal(permuted_row);
            result.values_mut()[ColIndex::from_usize(permuted_row)] /= diagonal;
        } else {
            {
                let (_values, non_zeros) = result.mutable_parts();
                self.transpose_upper
                    .compute_rows_to_consider_in_sorted_order(non_zeros);
            }
            result.mark_non_zeros_sorted();
            let (values, non_zeros) = result.mutable_parts();
            if non_zeros.is_empty() {
                self.transpose_upper
                    .lower_solve_starting_at(permuted_row, values)
                    .map_err(|_| FactorizationError::DimensionMismatch)?;
            } else {
                self.transpose_upper.hyper_sparse_solve(values, non_zeros);
            }
        }
        Ok(permuted_row)
    }

    /// Applies `P^T L^-T` while preserving sparse positions.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn left_solve_lower_with_nonzeros(
        &self,
        rhs: &mut ScatteredRow,
    ) -> Result<(), FactorizationError> {
        if self.is_identity_factorization {
            return Ok(());
        }
        if rhs.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        {
            let (_values, non_zeros) = rhs.mutable_parts();
            self.transpose_lower
                .compute_rows_to_consider_in_sorted_order(non_zeros);
        }
        rhs.mark_non_zeros_sorted();
        {
            let (values, non_zeros) = rhs.mutable_parts();
            if non_zeros.is_empty() {
                self.lower
                    .transpose_solve(values)
                    .map_err(|_| FactorizationError::DimensionMismatch)?;
            } else {
                self.lower
                    .transpose_hyper_sparse_solve_with_reversed_nonzeros(values, non_zeros);
            }
        }
        self.permute_scattered(rhs, &self.inverse_row_permutation);
        Ok(())
    }

    /// Applies `P^T L^-T` and optionally retains the result before `P^T`.
    ///
    /// This is the Rust counterpart of GLOP's two-output
    /// `LeftSolveLWithNonZeros()`, used to cache the intermediate needed by
    /// `BasisFactorization::RightSolveForTau()`.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn left_solve_lower_with_nonzeros_and_cache(
        &self,
        rhs: &mut ScatteredRow,
        result_before_permutation: &mut ScatteredColumn,
    ) -> Result<bool, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(false);
        }
        if rhs.len().to_usize() != self.dimension()
            || result_before_permutation.len().to_usize() != self.dimension()
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        {
            let (_values, non_zeros) = rhs.mutable_parts();
            self.transpose_lower
                .compute_rows_to_consider_in_sorted_order(non_zeros);
        }
        rhs.mark_non_zeros_sorted();
        {
            let (values, non_zeros) = rhs.mutable_parts();
            if non_zeros.is_empty() {
                self.lower
                    .transpose_solve(values)
                    .map_err(|_| FactorizationError::DimensionMismatch)?;
            } else {
                self.lower
                    .transpose_hyper_sparse_solve_with_reversed_nonzeros(values, non_zeros);
            }
        }
        result_before_permutation.clear();
        if rhs.non_zeros().is_empty() {
            // An empty position list is GLOP's dense-vector sentinel. Preserve
            // it in the cached, pre-permutation result: rebuilding the exact
            // support here would make RightSolveForTau() select a different
            // numerical kernel from upstream.
            result_before_permutation
                .values_mut()
                .as_mut_slice()
                .copy_from_slice(rhs.values().as_slice());
        } else {
            for entry in rhs.iter() {
                result_before_permutation.set(
                    RowIndex::from_usize(entry.index().to_usize()),
                    entry.coefficient(),
                );
            }
        }
        self.permute_scattered(rhs, &self.inverse_row_permutation);
        if !rhs.non_zeros().is_empty() {
            rhs.mark_non_zeros_unsorted();
        }
        Ok(true)
    }

    /// Applies `L^-1` when the input is already in factor-row coordinates.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn right_solve_lower_with_permuted_input(
        &self,
        rhs: &mut ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        if self.is_identity_factorization {
            return Ok(());
        }
        if rhs.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let (values, non_zeros) = rhs.mutable_parts();
        self.lower
            .solve_with_nonzeros(values, non_zeros)
            .map_err(|_| FactorizationError::DimensionMismatch)
    }

    fn permute_scattered<I: VectorIndex + Ord>(
        &self,
        vector: &mut ScatteredVector<I>,
        destination_by_source: &[usize],
    ) {
        if destination_by_source.is_empty() {
            return;
        }
        // The sparse membership mask is a temporary cache keyed by the
        // current coordinates. Clear its source-coordinate buckets before
        // permuting the recorded positions; otherwise stale source bits can
        // suppress positions introduced by a later rank-one update.
        vector.clear_sparse_mask();
        let n = destination_by_source.len();
        let mut scratch = self.dense_zero_scratchpad.borrow_mut();
        scratch.resize(n, 0.0);
        let (values, non_zeros) = vector.mutable_parts();
        if non_zeros.is_empty() {
            for source in 0..n {
                scratch[destination_by_source[source]] = values[source];
                values[source] = 0.0;
            }
            for destination in 0..n {
                values[destination] = scratch[destination];
                scratch[destination] = 0.0;
            }
        } else {
            for source in non_zeros.iter_mut() {
                let source_position = source.to_usize();
                let destination = destination_by_source[source_position];
                scratch[destination] = values[source_position];
                values[source_position] = 0.0;
                *source = I::from_usize(destination);
            }
            for &destination in non_zeros.iter() {
                let position = destination.to_usize();
                values[position] = scratch[position];
                scratch[position] = 0.0;
            }
        }
    }

    /// Solves `U^T y = Q^T rhs`.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn left_solve_upper(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(rhs.to_vec());
        }
        if rhs.len() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut work: Vec<f64> = if self.inverse_column_permutation.is_empty() {
            rhs.to_vec()
        } else {
            self.inverse_column_permutation
                .iter()
                .map(|&column| rhs[column])
                .collect()
        };
        self.upper
            .transpose_solve(&mut work)
            .map_err(|_| FactorizationError::DimensionMismatch)?;
        Ok(work)
    }

    /// Solves `L^T z = rhs` and maps through the inverse row permutation.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch.
    pub fn left_solve_lower(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(rhs.to_vec());
        }
        if rhs.len() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut work = rhs.to_vec();
        self.lower
            .transpose_solve(&mut work)
            .map_err(|_| FactorizationError::DimensionMismatch)?;
        let mut solution = vec![0.0; self.dimension()];
        for (position, &row) in self.inverse_row_permutation.iter().enumerate() {
            solution[row] = work[position];
        }
        Ok(solution)
    }

    #[must_use]
    pub fn lower_and_upper(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.dimension();
        let mut lower = vec![vec![0.0; n]; n];
        let mut upper = vec![vec![0.0; n]; n];
        for column in 0..n {
            lower[column][column] = 1.0;
            for (row, coefficient) in self.lower.column(column) {
                lower[row][column] = coefficient;
            }
            for (row, coefficient) in self.upper.column(column) {
                upper[row][column] = coefficient;
            }
            upper[column][column] = self.upper.diagonal(column);
        }
        (lower, upper)
    }

    /// Materializes `L * U` in factor coordinates for diagnostics and tests.
    #[must_use]
    pub fn lower_times_upper(&self) -> SparseMatrix {
        let mut lower = SparseMatrix::new();
        let mut upper = SparseMatrix::new();
        self.lower.copy_to_sparse_matrix(&mut lower);
        self.upper.copy_to_sparse_matrix(&mut upper);
        let mut product = SparseMatrix::new();
        product.populate_from_product(&lower, &upper);
        product
    }

    /// Returns a column of U using the input matrix's column numbering.
    #[must_use]
    pub fn column_of_upper(&self, input_column: usize) -> Vec<(usize, f64)> {
        if self.is_identity_factorization {
            return vec![(input_column, 1.0)];
        }
        let column = if self.column_permutation.is_empty() {
            input_column
        } else {
            let Some(&column) = self.column_permutation.get(input_column) else {
                return Vec::new();
            };
            column
        };
        let mut result: Vec<_> = self.upper.column(column).collect();
        result.push((column, self.upper.diagonal(column)));
        // GLOP's GetColumnOfU() materializes through
        // TriangularMatrix::CopyColumnToSparseColumn(), whose CleanUp() sorts
        // the copied entries. This order is distinct from U's physical column
        // order, which triangular solves deliberately preserve.
        result.sort_unstable_by_key(|entry| entry.0);
        result
    }

    /// Computes `||A^-1 a||_2^2` for a sparse input column.
    ///
    /// # Errors
    ///
    /// Returns a dimension error for an out-of-range sparse row.
    pub fn right_solve_squared_norm(
        &self,
        entries: &[(usize, f64)],
    ) -> Result<f64, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(entries
                .iter()
                .fold(0.0, |sum, entry| sum + entry.1 * entry.1));
        }
        let mut values = self.dense_zero_scratchpad.borrow_mut();
        let mut non_zeros = self.non_zero_rows.borrow_mut();
        values.resize(self.dimension(), 0.0);
        non_zeros.clear();
        debug_assert!(values.iter().all(|&value| value == 0.0));
        for &(row, value) in entries {
            if row >= self.dimension() {
                return Err(FactorizationError::DimensionMismatch);
            }
            let permuted_row = self.row_permutation[row];
            values[permuted_row] = value;
            non_zeros.push(RowIndex::from_usize(permuted_row));
        }
        self.lower
            .compute_rows_to_consider_in_sorted_order(&mut non_zeros);
        if non_zeros.is_empty() {
            self.lower
                .solve(&mut values)
                .map_err(|_| FactorizationError::DimensionMismatch)?;
        } else {
            self.lower.hyper_sparse_solve(&mut values, &mut non_zeros);
            self.upper
                .compute_rows_to_consider_in_sorted_order(&mut non_zeros);
        }
        if non_zeros.is_empty() {
            self.upper
                .solve(&mut values)
                .map_err(|_| FactorizationError::DimensionMismatch)?;
        } else {
            self.upper
                .hyper_sparse_solve_with_reversed_nonzeros(&mut values, &mut non_zeros);
        }
        Ok(squared_norm_and_reset_rows(&mut values, &non_zeros))
    }

    /// Computes `||(A^T)^-1 e_row||_2^2`.
    ///
    /// # Errors
    ///
    /// Returns a dimension error for an invalid row.
    pub fn dual_edge_squared_norm(&self, row: usize) -> Result<f64, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(1.0);
        }
        if row >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let permuted_row = if self.column_permutation.is_empty() {
            row
        } else {
            self.column_permutation[row]
        };
        let mut values = self.dense_zero_scratchpad.borrow_mut();
        let mut non_zeros = self.non_zero_rows.borrow_mut();
        values.resize(self.dimension(), 0.0);
        non_zeros.clear();
        debug_assert!(values.iter().all(|&value| value == 0.0));
        values[permuted_row] = 1.0;
        non_zeros.push(RowIndex::from_usize(permuted_row));
        self.transpose_upper
            .compute_rows_to_consider_in_sorted_order(&mut non_zeros);
        if non_zeros.is_empty() {
            self.transpose_upper
                .lower_solve_starting_at(permuted_row, &mut values)
                .map_err(|_| FactorizationError::DimensionMismatch)?;
        } else {
            self.transpose_upper
                .hyper_sparse_solve(&mut values, &mut non_zeros);
            self.transpose_lower
                .compute_rows_to_consider_in_sorted_order(&mut non_zeros);
        }
        if non_zeros.is_empty() {
            self.transpose_lower
                .solve(&mut values)
                .map_err(|_| FactorizationError::DimensionMismatch)?;
        } else {
            self.transpose_lower
                .hyper_sparse_solve_with_reversed_nonzeros(&mut values, &mut non_zeros);
        }
        Ok(squared_norm_and_reset_rows(&mut values, &non_zeros))
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fill_in_ratio(&self, matrix: &SparseMatrix) -> f64 {
        if self.is_identity_factorization || matrix.num_entries().value() == 0 {
            return 1.0;
        }
        self.number_of_entries() as f64 / matrix.num_entries().value() as f64
    }

    #[must_use]
    pub fn number_of_entries(&self) -> usize {
        if self.is_identity_factorization {
            return 0;
        }
        self.lower.num_entries() + self.upper.num_entries()
    }

    #[must_use]
    pub fn determinant(&self) -> f64 {
        if self.is_identity_factorization {
            return 1.0;
        }
        let diagonal_product: f64 = (0..self.dimension())
            .map(|column| self.upper.diagonal(column))
            .product();
        diagonal_product
            * f64::from(permutation_signature(&self.row_permutation))
            * f64::from(permutation_signature(&self.column_permutation))
    }

    /// Computes `||A^-1||_1` using one solve per column.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn inverse_one_norm(&self) -> Result<f64, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(1.0);
        }
        let n = self.dimension();
        let mut norm = 0.0_f64;
        for column in 0..n {
            let mut unit = vec![0.0; n];
            unit[column] = 1.0;
            norm = norm.max(self.solve(&unit)?.iter().map(|value| value.abs()).sum());
        }
        Ok(norm)
    }

    /// Computes `||A^-1||_infinity` using transpose solves.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn inverse_infinity_norm(&self) -> Result<f64, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(1.0);
        }
        let n = self.dimension();
        let mut norm = 0.0_f64;
        for row in 0..n {
            let mut unit = vec![0.0; n];
            unit[row] = 1.0;
            norm = norm.max(
                self.transpose_solve(&unit)?
                    .iter()
                    .map(|value| value.abs())
                    .sum(),
            );
        }
        Ok(norm)
    }

    /// Computes `||A||_1 ||A^-1||_1`.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn one_norm_condition_number(
        &self,
        matrix: &SparseMatrix,
    ) -> Result<f64, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(1.0);
        }
        Ok(matrix.one_norm() * self.inverse_one_norm()?)
    }

    /// Computes `||A||_infinity ||A^-1||_infinity`.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn infinity_norm_condition_number(
        &self,
        matrix: &SparseMatrix,
    ) -> Result<f64, FactorizationError> {
        if self.is_identity_factorization {
            return Ok(1.0);
        }
        Ok(matrix.infinity_norm() * self.inverse_infinity_norm()?)
    }

    #[must_use]
    pub fn inverse_infinity_norm_upper_bound(&self) -> f64 {
        if self.is_identity_factorization {
            return 1.0;
        }
        self.lower.inverse_infinity_norm_upper_bound()
            * self.upper.inverse_infinity_norm_upper_bound()
    }
}

fn squared_norm_and_reset_rows(values: &mut [f64], non_zeros: &[RowIndex]) -> f64 {
    if non_zeros.is_empty() {
        squared_norm_and_reset_to_zero(values)
    } else {
        let mut sum = 0.0;
        for &index in non_zeros {
            let position = index.to_usize();
            let value = values[position];
            sum += value * value;
            values[position] = 0.0;
        }
        sum
    }
}

fn permutation_signature(permutation: &[usize]) -> i32 {
    let mut visited = vec![false; permutation.len()];
    let mut signature = 1;
    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut size = 0;
        let mut current = start;
        loop {
            visited[current] = true;
            current = permutation[current];
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

#[cfg(test)]
mod tests {
    use lp_data::lp_types::RowIndex;

    use super::*;

    fn matrix(values: &[&[f64]]) -> SparseMatrix {
        let n = values.len();
        let mut result = SparseMatrix::new();
        result.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
        for (row, values) in values.iter().enumerate() {
            for (column, &value) in values.iter().enumerate() {
                if value != 0.0 {
                    result
                        .mutable_column(ColIndex::from_usize(column))
                        .add_entry(RowIndex::from_usize(row), value);
                }
            }
        }
        result.clean_up();
        result
    }

    fn multiply(matrix: &[&[f64]], vector: &[f64], transpose: bool) -> Vec<f64> {
        (0..matrix.len())
            .map(|row| {
                (0..matrix.len())
                    .map(|column| {
                        if transpose {
                            matrix[column][row] * vector[column]
                        } else {
                            matrix[row][column] * vector[column]
                        }
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn solve_and_transpose_solve_have_small_residuals() {
        let values: &[&[f64]] = &[&[0.0, 2.0, 1.0], &[1.0, -2.0, 0.0], &[3.0, 1.0, 4.0]];
        let factorization = LuFactorization::factorize(&matrix(values), 0.1).unwrap();
        assert!((factorization.determinant() + 1.0).abs() < 1e-12);
        for transpose in [false, true] {
            let expected = [1.0, -2.0, 3.0];
            let rhs = multiply(values, &expected, transpose);
            let actual = if transpose {
                factorization.transpose_solve(&rhs).unwrap()
            } else {
                factorization.solve(&rhs).unwrap()
            };
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(left, right)| (left - right).abs() < 1e-12)
            );
        }
    }

    #[test]
    fn dense_upper_solve_uses_the_same_transposed_factor_path_as_glop() {
        let values: &[&[f64]] = &[
            &[4.0, 1.0e16, 3.0, -7.0, 11.0],
            &[0.0, -3.0, -1.0e16, 5.0, -13.0],
            &[0.0, 0.0, 2.0, 1.0e-16, 17.0],
            &[0.0, 0.0, 0.0, 5.0, -19.0],
            &[0.0, 0.0, 0.0, 0.0, 7.0],
        ];
        let factorization = LuFactorization::factorize(&matrix(values), 0.1).unwrap();
        let rhs = [1.0, -2.0, 3.0, -4.0, 5.0];

        let dense = factorization.right_solve_upper(&rhs).unwrap();
        let mut scattered = ScatteredColumn::new(RowIndex::from_usize(rhs.len()));
        scattered.values_mut().as_mut_slice().copy_from_slice(&rhs);
        factorization
            .right_solve_upper_with_nonzeros(&mut scattered)
            .unwrap();

        assert_eq!(
            dense
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scattered
                .values()
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn scattered_permutation_invalidates_source_coordinate_membership_bits() {
        let factorization = LuFactorization::new();
        let mut vector = ScatteredColumn::new(RowIndex::new(3));
        vector.set(RowIndex::new(0), 1.0);

        factorization.permute_scattered(&mut vector, &[1, 0, 2]);
        vector.add(RowIndex::new(0), 2.0);

        assert_eq!(vector.value(RowIndex::new(0)).to_bits(), 2.0_f64.to_bits());
        assert_eq!(vector.value(RowIndex::new(1)).to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            vector
                .non_zeros()
                .iter()
                .map(|index| index.to_usize())
                .collect::<Vec<_>>(),
            [1, 0]
        );
    }

    #[test]
    fn problem_column_lower_solve_leaves_membership_cache_temporary() {
        let n = 100;
        let mut columns = vec![Vec::new(); n];
        columns[0].push((1, 1.0));
        let mut factorization = LuFactorization::new();
        factorization.is_identity_factorization = false;
        factorization.lower =
            TriangularMatrix::from_columns(&columns, &vec![1.0; n], Triangle::Lower, true).unwrap();
        factorization.row_permutation = (0..n).collect();

        let mut column = SparseColumn::new();
        column.add_entry(RowIndex::new(0), 1.0);
        column.add_entry(RowIndex::new(1), 1.0);
        let mut result = ScatteredColumn::new(RowIndex::from_usize(n));
        factorization
            .right_solve_lower_for_column(&column, &mut result)
            .unwrap();

        assert_eq!(result.value(RowIndex::new(1)).to_bits(), 0.0_f64.to_bits());
        assert!(!result.non_zeros().contains(&RowIndex::new(1)));
        result.add(RowIndex::new(1), 2.0);
        assert!(result.non_zeros().contains(&RowIndex::new(1)));
    }

    #[test]
    fn column_of_upper_cleans_the_copy_without_reordering_the_factor() {
        let mut columns = vec![Vec::new(); 4];
        columns[3] = vec![(2, 2.0), (0, 3.0), (1, 4.0)];
        let mut factorization = LuFactorization::new();
        factorization.is_identity_factorization = false;
        factorization.upper =
            TriangularMatrix::from_columns(&columns, &[1.0; 4], Triangle::Upper, false).unwrap();

        assert_eq!(
            factorization.upper.column(3).collect::<Vec<_>>(),
            columns[3]
        );
        assert_eq!(
            factorization.column_of_upper(3),
            vec![(0, 3.0), (1, 4.0), (2, 2.0), (3, 1.0)]
        );
        assert_eq!(
            factorization.upper.column(3).collect::<Vec<_>>(),
            columns[3]
        );
    }

    #[test]
    fn rejects_singular_and_nonfinite_matrices() {
        let singular = matrix(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(matches!(
            LuFactorization::factorize(&singular, 0.1),
            Err(FactorizationError::Singular { .. })
        ));
        let nonfinite = matrix(&[&[f64::NAN]]);
        assert_eq!(
            LuFactorization::factorize(&nonfinite, 0.1).unwrap_err(),
            FactorizationError::NonFinite
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn singleton_fast_path_does_not_apply_markowitz_singularity_threshold() {
        // ExtractSingletonColumns() accepts structural singleton pivots
        // directly in GLOP; the threshold is applied only by FindPivot().
        let tiny_singleton = matrix(&[&[1e-16]]);
        let factorization =
            LuFactorization::factorize_with_parameters(&tiny_singleton, &GlopParameters::default())
                .unwrap();
        assert_eq!(factorization.determinant(), 1e-16);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn cleared_factorization_is_dimension_independent_identity() {
        let mut factorization = LuFactorization::factorize(&matrix(&[&[2.0]]), 0.125).unwrap();
        let deterministic_time = factorization.deterministic_time_of_last_factorization();
        factorization.clear();
        assert!(factorization.is_identity_factorization());
        assert_eq!(factorization.pivot_threshold(), 0.125);
        assert_eq!(
            factorization.deterministic_time_of_last_factorization(),
            deterministic_time
        );
        assert_eq!(
            factorization.solve(&[1.0, -2.0, 3.0]).unwrap(),
            [1.0, -2.0, 3.0]
        );
        assert_eq!(
            factorization.transpose_solve(&[4.0, 5.0]).unwrap(),
            [4.0, 5.0]
        );
        assert_eq!(factorization.number_of_entries(), 0);
        assert_eq!(factorization.determinant().to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn hypersparse_solve_matches_dense_solve_through_permutations() {
        let n = 83;
        let mut input = SparseMatrix::new();
        input.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
        for column in 0..n {
            let row = (7 * column) % n;
            input
                .mutable_column(ColIndex::from_usize(column))
                .add_entry(RowIndex::from_usize(row), 1.0 + column as f64 / 100.0);
        }
        input.clean_up();
        let factorization = LuFactorization::factorize(&input, 0.01).unwrap();

        let mut rhs = vec![0.0; n];
        rhs[37] = -2.5;
        let expected = factorization.solve(&rhs).unwrap();
        let mut scattered = ScatteredColumn::new(RowIndex::from_usize(n));
        scattered.set(RowIndex::new(37), -2.5);
        factorization.solve_with_nonzeros(&mut scattered).unwrap();
        assert!((0..n).all(|index| {
            (expected[index] - scattered.value(RowIndex::from_usize(index))).abs() < 1e-14
        }));
        assert_eq!(scattered.non_zeros().len(), 1);

        let expected = factorization.transpose_solve(&rhs).unwrap();
        let mut scattered = ScatteredRow::new(ColIndex::from_usize(n));
        scattered.set(ColIndex::new(37), -2.5);
        factorization
            .transpose_solve_with_nonzeros(&mut scattered)
            .unwrap();
        assert!((0..n).all(|index| {
            (expected[index] - scattered.value(ColIndex::from_usize(index))).abs() < 1e-14
        }));
        assert_eq!(scattered.non_zeros().len(), 1);
    }
}
