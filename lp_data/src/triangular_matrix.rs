//! Sparse triangular solves and transpose solves.
//!
//! This is the focused Phase-2 counterpart of the triangular portion of
//! upstream `ortools/lp_data/sparse.{h,cc}`.

use std::cell::RefCell;
use std::fmt;

use crate::lp_types::{ColIndex, RowIndex, VectorIndex};
use crate::permutation::RowPermutation;
use crate::sparse::SparseMatrix;
use crate::sparse_vector::SparseColumn;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Triangle {
    Lower,
    Upper,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TriangularError {
    NonSquare,
    WrongTriangle,
    Singular { column: usize },
    DimensionMismatch,
}

impl fmt::Display for TriangularError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquare => formatter.write_str("triangular matrix must be square"),
            Self::WrongTriangle => formatter.write_str("entry lies outside the selected triangle"),
            Self::Singular { column } => write!(formatter, "zero diagonal at column {column}"),
            Self::DimensionMismatch => formatter.write_str("right-hand side dimension mismatch"),
        }
    }
}

impl std::error::Error for TriangularError {}

#[derive(Clone, Debug)]
pub struct TriangularMatrix {
    num_rows: usize,
    starts: Vec<usize>,
    rows: Vec<usize>,
    coefficients: Vec<f64>,
    diagonal: Vec<f64>,
    triangle: Triangle,
    unit_diagonal: bool,
    first_non_identity_column: usize,
    symbolic_workspace: RefCell<SymbolicWorkspace>,
    permuted_workspace: PermutedLowerWorkspace,
}

#[derive(Clone, Debug, Default)]
struct SymbolicWorkspace {
    stored: Vec<bool>,
    touched: Vec<usize>,
    nodes_to_explore: Vec<SymbolicNode>,
}

#[derive(Clone, Copy, Debug)]
enum SymbolicNode {
    Enter(usize),
    Exit(usize),
}

/// Mutable storage used by GLOP's factorization-time permuted lower solve.
/// The pruned column ends persist between solves; all other vectors are reused.
#[derive(Clone, Debug, Default)]
struct PermutedLowerWorkspace {
    stored: Vec<bool>,
    marked: Vec<bool>,
    touched: Vec<usize>,
    lower_rows: Vec<usize>,
    upper_rows: Vec<usize>,
    nodes_to_explore: Vec<usize>,
    scratchpad: Vec<f64>,
    pruned_ends: Vec<usize>,
    num_fp_operations: i64,
}

// GLOP deliberately uses exact zero and unit-diagonal predicates because they
// control storage, fast paths, and symbolic behavior.
#[allow(clippy::float_cmp)]
impl TriangularMatrix {
    #[must_use]
    pub fn empty(triangle: Triangle, unit_diagonal: bool) -> Self {
        Self {
            num_rows: 0,
            starts: vec![0],
            rows: Vec::new(),
            coefficients: Vec::new(),
            diagonal: Vec::new(),
            triangle,
            unit_diagonal,
            first_non_identity_column: 0,
            symbolic_workspace: RefCell::new(SymbolicWorkspace::default()),
            permuted_workspace: PermutedLowerWorkspace::default(),
        }
    }

    /// Validates and copies a sparse triangular matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for nonsquare, structurally nontriangular, or singular
    /// input.
    pub fn from_sparse(
        matrix: &SparseMatrix,
        triangle: Triangle,
        unit_diagonal: bool,
    ) -> Result<Self, TriangularError> {
        let n = matrix.num_cols().to_usize();
        if matrix.num_rows().to_usize() != n {
            return Err(TriangularError::NonSquare);
        }
        let mut starts = Vec::with_capacity(n + 1);
        let mut rows = Vec::new();
        let mut coefficients = Vec::new();
        let mut diagonal = vec![if unit_diagonal { 1.0 } else { 0.0 }; n];
        starts.push(0);
        for (column, diagonal_value) in diagonal.iter_mut().enumerate() {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                let row = entry.index().to_usize();
                if row == column {
                    if !unit_diagonal {
                        *diagonal_value = entry.coefficient();
                    }
                } else {
                    let valid = match triangle {
                        Triangle::Lower => row > column,
                        Triangle::Upper => row < column,
                    };
                    if !valid {
                        return Err(TriangularError::WrongTriangle);
                    }
                    rows.push(row);
                    coefficients.push(entry.coefficient());
                }
            }
            if *diagonal_value == 0.0 || !diagonal_value.is_finite() {
                return Err(TriangularError::Singular { column });
            }
            starts.push(rows.len());
        }
        let first_non_identity_column = Self::first_non_identity(&diagonal, &starts);
        let pruned_ends = starts.iter().copied().skip(1).collect();
        Ok(Self {
            num_rows: n,
            starts,
            rows,
            coefficients,
            diagonal,
            triangle,
            unit_diagonal,
            first_non_identity_column,
            symbolic_workspace: RefCell::new(SymbolicWorkspace::default()),
            permuted_workspace: PermutedLowerWorkspace {
                pruned_ends,
                ..PermutedLowerWorkspace::default()
            },
        })
    }

    /// Builds a triangular matrix from already triangular sparse columns.
    /// Diagonal entries are supplied separately, matching GLOP's storage.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-triangle entry or invalid diagonal.
    pub fn from_columns(
        columns: &[Vec<(usize, f64)>],
        diagonal: Vec<f64>,
        triangle: Triangle,
        unit_diagonal: bool,
    ) -> Result<Self, TriangularError> {
        if diagonal.len() != columns.len() {
            return Err(TriangularError::NonSquare);
        }
        let mut starts = Vec::with_capacity(columns.len() + 1);
        let mut rows = Vec::new();
        let mut coefficients = Vec::new();
        starts.push(0);
        for (column, entries) in columns.iter().enumerate() {
            if diagonal[column] == 0.0 || !diagonal[column].is_finite() {
                return Err(TriangularError::Singular { column });
            }
            for &(row, coefficient) in entries {
                let valid = match triangle {
                    Triangle::Lower => row > column,
                    Triangle::Upper => row < column,
                };
                if !valid {
                    return Err(TriangularError::WrongTriangle);
                }
                rows.push(row);
                coefficients.push(coefficient);
            }
            starts.push(rows.len());
        }
        let first_non_identity_column = Self::first_non_identity(&diagonal, &starts);
        let pruned_ends = starts.iter().copied().skip(1).collect();
        Ok(Self {
            num_rows: columns.len(),
            starts,
            rows,
            coefficients,
            diagonal,
            triangle,
            unit_diagonal,
            first_non_identity_column,
            symbolic_workspace: RefCell::new(SymbolicWorkspace::default()),
            permuted_workspace: PermutedLowerWorkspace {
                pruned_ends,
                ..PermutedLowerWorkspace::default()
            },
        })
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[must_use]
    pub fn num_cols(&self) -> usize {
        self.diagonal.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diagonal.is_empty()
    }

    #[must_use]
    pub const fn first_non_identity_column(&self) -> usize {
        self.first_non_identity_column
    }

    #[must_use]
    pub fn column_is_diagonal_only(&self, column: usize) -> bool {
        self.starts[column] == self.starts[column + 1]
    }

    pub fn reset(&mut self, num_rows: usize, column_capacity: usize) {
        self.num_rows = num_rows;
        self.starts.clear();
        self.starts.reserve(column_capacity.saturating_add(1));
        self.starts.push(0);
        self.rows.clear();
        self.coefficients.clear();
        self.diagonal.clear();
        self.diagonal.reserve(column_capacity);
        self.unit_diagonal = true;
        self.first_non_identity_column = 0;
        *self.symbolic_workspace.get_mut() = SymbolicWorkspace::default();
        self.permuted_workspace = PermutedLowerWorkspace::default();
    }

    pub fn add_diagonal_only_column(&mut self, diagonal_value: f64) {
        self.close_current_column(diagonal_value);
    }

    pub fn add_triangular_column(&mut self, column: &SparseColumn, diagonal_row: RowIndex) {
        let mut diagonal_value = 0.0;
        for entry in column {
            if entry.index() == diagonal_row {
                diagonal_value = entry.coefficient();
            } else {
                debug_assert_ne!(entry.coefficient(), 0.0);
                self.rows.push(entry.index().to_usize());
                self.coefficients.push(entry.coefficient());
            }
        }
        self.close_current_column(diagonal_value);
    }

    pub fn add_triangular_column_with_given_diagonal(
        &mut self,
        column: &SparseColumn,
        diagonal_row: RowIndex,
        diagonal_value: f64,
    ) {
        for entry in column {
            debug_assert_ne!(entry.index(), diagonal_row);
            self.rows.push(entry.index().to_usize());
            self.coefficients.push(entry.coefficient());
        }
        self.close_current_column(diagonal_value);
    }

    pub fn add_and_normalize_triangular_column(
        &mut self,
        column: &SparseColumn,
        diagonal_row: RowIndex,
        diagonal_coefficient: f64,
    ) {
        for entry in column {
            if entry.index() == diagonal_row {
                debug_assert_eq!(entry.coefficient(), diagonal_coefficient);
            } else if entry.coefficient() != 0.0 {
                self.rows.push(entry.index().to_usize());
                self.coefficients
                    .push(entry.coefficient() / diagonal_coefficient);
            }
        }
        self.close_current_column(1.0);
    }

    pub fn apply_row_permutation_to_non_diagonal_entries(&mut self, permutation: &RowPermutation) {
        for row in &mut self.rows {
            *row = permutation[RowIndex::from_usize(*row)].to_usize();
        }
    }

    pub fn copy_column_to_sparse_column(&self, column: usize, output: &mut SparseColumn) {
        output.clear();
        for (row, coefficient) in self.column(column) {
            output.set_coefficient(RowIndex::from_usize(row), coefficient);
        }
        output.set_coefficient(RowIndex::from_usize(column), self.diagonal[column]);
        output.clean_up();
    }

    pub fn copy_to_sparse_matrix(&self, output: &mut SparseMatrix) {
        output.populate_from_zero(
            RowIndex::from_usize(self.num_rows),
            ColIndex::from_usize(self.num_cols()),
        );
        for column in 0..self.num_cols() {
            self.copy_column_to_sparse_column(
                column,
                output.mutable_column(ColIndex::from_usize(column)),
            );
        }
    }

    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    fn close_current_column(&mut self, diagonal_value: f64) {
        debug_assert_ne!(diagonal_value, 0.0);
        let column = self.diagonal.len();
        self.diagonal.push(diagonal_value);
        self.starts.push(self.rows.len());
        self.permuted_workspace.pruned_ends.push(self.rows.len());
        if self.first_non_identity_column == column
            && diagonal_value == 1.0
            && self.starts[column] == self.starts[column + 1]
        {
            self.first_non_identity_column += 1;
        }
        self.unit_diagonal &= diagonal_value == 1.0;
    }

    fn first_non_identity(diagonal: &[f64], starts: &[usize]) -> usize {
        let mut column = 0;
        while column < diagonal.len()
            && diagonal[column] == 1.0
            && starts[column] == starts[column + 1]
        {
            column += 1;
        }
        column
    }

    pub fn column(&self, column: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.starts[column]..self.starts[column + 1];
        self.rows[range.clone()]
            .iter()
            .copied()
            .zip(self.coefficients[range].iter().copied())
    }

    /// Solves GLOP's partially permuted unit-lower system during Markowitz
    /// factorization. `row_permutation[row] == usize::MAX` denotes a row that
    /// has not yet been selected as a pivot. Such rows replace `lower`; solved
    /// pivot rows are appended to `upper`.
    ///
    /// The DFS also persistently prunes the factor's dependency graph by
    /// swapping redundant entries to the end of each stored column, exactly as
    /// `TriangularMatrix::PermutedLowerSparseSolve()` does upstream.
    ///
    /// # Panics
    ///
    /// Panics if the permutation or an input row is outside the matrix.
    #[allow(clippy::needless_range_loop, clippy::too_many_lines)]
    pub fn permuted_lower_sparse_solve(
        &mut self,
        lower: &mut SparseColumn,
        row_permutation: &[usize],
        upper: &mut SparseColumn,
    ) {
        assert_eq!(self.triangle, Triangle::Lower);
        assert!(self.unit_diagonal);
        assert_eq!(row_permutation.len(), self.num_rows);
        let invalid = usize::MAX;
        let Self {
            num_rows,
            starts,
            rows,
            coefficients,
            permuted_workspace: workspace,
            ..
        } = self;
        workspace.stored.resize(*num_rows, false);
        workspace.marked.resize(*num_rows, false);
        workspace.scratchpad.resize(*num_rows, 0.0);
        workspace.lower_rows.clear();
        workspace.upper_rows.clear();
        workspace.nodes_to_explore.clear();
        workspace.touched.clear();

        // Scatter the right-hand side before clearing `lower`, allowing the
        // input and lower output to share the same SparseColumn allocation.
        for entry in &*lower {
            let row = entry.index().to_usize();
            workspace.scratchpad[row] = entry.coefficient();
            let pivot_column = row_permutation[row];
            if pivot_column == invalid {
                if !workspace.stored[row] {
                    workspace.stored[row] = true;
                    workspace.touched.push(row);
                    workspace.lower_rows.push(row);
                }
            } else {
                workspace.nodes_to_explore.push(row);
            }
        }

        // DFS postorder with GLOP's sentinel stack. Marking every outgoing
        // edge on descent lets the return step prune edges implied by a path
        // through an already explored successor.
        while let Some(&row) = workspace.nodes_to_explore.last() {
            if row == invalid {
                workspace.nodes_to_explore.pop();
                let explored_row = workspace
                    .nodes_to_explore
                    .pop()
                    .expect("DFS sentinel follows its node");
                debug_assert!(!workspace.stored[explored_row]);
                workspace.stored[explored_row] = true;
                workspace.touched.push(explored_row);
                workspace.upper_rows.push(explored_row);

                let column = row_permutation[explored_row];
                let mut position = starts[column];
                let mut end = workspace.pruned_ends[column];
                while position < end {
                    let successor = rows[position];
                    if workspace.marked[successor] {
                        workspace.marked[successor] = false;
                        position += 1;
                    } else {
                        end -= 1;
                        rows.swap(position, end);
                        coefficients.swap(position, end);
                    }
                }
                workspace.pruned_ends[column] = end;
                continue;
            }
            if workspace.stored[row] {
                workspace.nodes_to_explore.pop();
                continue;
            }
            let column = row_permutation[row];
            if column == invalid {
                workspace.stored[row] = true;
                workspace.touched.push(row);
                workspace.lower_rows.push(row);
                workspace.nodes_to_explore.pop();
                continue;
            }
            workspace.nodes_to_explore.push(invalid);
            for position in starts[column]..workspace.pruned_ends[column] {
                let successor = rows[position];
                if !workspace.stored[successor] {
                    workspace.nodes_to_explore.push(successor);
                }
                workspace.marked[successor] = true;
            }
            debug_assert!(workspace.nodes_to_explore.len() <= 2 * *num_rows + rows.len());
        }

        workspace.num_fp_operations = 0;
        lower.clear();
        upper.reserve(
            upper
                .num_entries()
                .saturating_add(workspace.upper_rows.len()),
        );
        for &pivot_row in workspace.upper_rows.iter().rev() {
            let pivot = workspace.scratchpad[pivot_row];
            if pivot == 0.0 {
                continue;
            }
            workspace.scratchpad[pivot_row] = 0.0;
            let column = row_permutation[pivot_row];
            upper.add_entry(RowIndex::from_usize(pivot_row), pivot);
            workspace.num_fp_operations +=
                1 + i64::try_from(starts[column + 1] - starts[column]).unwrap_or(i64::MAX);
            for position in starts[column]..starts[column + 1] {
                let successor = rows[position];
                workspace.scratchpad[successor] =
                    (-coefficients[position]).mul_add(pivot, workspace.scratchpad[successor]);
            }
        }
        lower.reserve(workspace.lower_rows.len());
        for &row in &workspace.lower_rows {
            lower.add_entry(RowIndex::from_usize(row), workspace.scratchpad[row]);
            workspace.scratchpad[row] = 0.0;
        }
        for &row in &workspace.touched {
            workspace.stored[row] = false;
        }
    }

    /// Dense reference form of the partially permuted lower solve. This is
    /// primarily useful for checking the sparse reachability implementation,
    /// matching GLOP's testing-visible `PermutedLowerSolve()`.
    ///
    /// # Panics
    ///
    /// Panics if a permutation, inverse-permutation entry, or input row is out
    /// of range.
    pub fn permuted_lower_solve(
        &mut self,
        rhs: &SparseColumn,
        row_permutation: &[usize],
        partial_inverse_row_permutation: &[usize],
        lower: &mut SparseColumn,
        upper: &mut SparseColumn,
    ) {
        assert_eq!(self.triangle, Triangle::Lower);
        assert!(self.unit_diagonal);
        assert_eq!(row_permutation.len(), self.num_rows);
        let invalid = usize::MAX;
        let scratchpad = &mut self.permuted_workspace.scratchpad;
        scratchpad.resize(self.num_rows, 0.0);
        for entry in rhs {
            scratchpad[entry.index().to_usize()] = entry.coefficient();
        }
        for &permuted_row in partial_inverse_row_permutation
            .iter()
            .skip(self.first_non_identity_column)
        {
            let pivot = scratchpad[permuted_row];
            if pivot == 0.0 {
                continue;
            }
            let column = row_permutation[permuted_row];
            debug_assert_ne!(column, invalid);
            for position in self.starts[column]..self.starts[column + 1] {
                let row = self.rows[position];
                scratchpad[row] = (-self.coefficients[position]).mul_add(pivot, scratchpad[row]);
            }
        }
        lower.clear();
        for (row, value) in scratchpad.iter_mut().enumerate() {
            if *value == 0.0 {
                continue;
            }
            if row_permutation[row] == invalid {
                lower.add_entry(RowIndex::from_usize(row), *value);
            } else {
                upper.add_entry(RowIndex::from_usize(row), *value);
            }
            *value = 0.0;
        }
    }

    #[must_use]
    pub const fn num_fp_operations_in_last_permuted_lower_sparse_solve(&self) -> i64 {
        self.permuted_workspace.num_fp_operations
    }

    #[must_use]
    pub fn diagonal(&self, column: usize) -> f64 {
        self.diagonal[column]
    }

    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.rows.len() + self.diagonal.len()
    }

    /// Returns whether the stored entries form a nonsingular lower triangle.
    #[must_use]
    pub fn is_lower_triangular(&self) -> bool {
        self.diagonal.iter().all(|&value| value != 0.0)
            && (0..self.dimension())
                .all(|column| self.column(column).all(|(row, _coefficient)| row > column))
    }

    /// Returns whether the stored entries form a nonsingular upper triangle.
    #[must_use]
    pub fn is_upper_triangular(&self) -> bool {
        self.diagonal.iter().all(|&value| value != 0.0)
            && (0..self.dimension())
                .all(|column| self.column(column).all(|(row, _coefficient)| row < column))
    }

    /// GLOP's inexpensive upper bound for `||T^-1||_infinity`.
    #[must_use]
    pub fn inverse_infinity_norm_upper_bound(&self) -> f64 {
        if self.first_non_identity_column == self.dimension() {
            return 1.0;
        }
        let mut estimate = vec![1.0; self.dimension()];
        for offset in 0..self.dimension() {
            let column = match self.triangle {
                Triangle::Upper => self.dimension() - 1 - offset,
                Triangle::Lower => offset,
            };
            let coefficient = estimate[column] / self.diagonal[column].abs();
            estimate[column] = coefficient;
            for (row, value) in self.column(column) {
                estimate[row] += coefficient * value.abs();
            }
        }
        estimate.into_iter().fold(0.0, f64::max)
    }

    /// Computes `||T^-1||_infinity` by solving for every inverse column.
    #[must_use]
    pub fn inverse_infinity_norm(&self) -> f64 {
        let mut row_sums = vec![0.0; self.dimension()];
        let mut right_hand_side = vec![0.0; self.dimension()];
        for column in 0..self.dimension() {
            right_hand_side.fill(0.0);
            right_hand_side[column] = 1.0;
            // The dimensions and nonzero diagonal are construction invariants.
            let result = self.solve(&mut right_hand_side);
            debug_assert!(result.is_ok());
            for (sum, value) in row_sums.iter_mut().zip(&right_hand_side) {
                *sum += value.abs();
            }
        }
        row_sums.into_iter().fold(0.0, f64::max)
    }

    /// Solves `T x = rhs` in place.
    ///
    /// # Errors
    ///
    /// Returns an error for a dimension mismatch.
    pub fn solve(&self, rhs: &mut [f64]) -> Result<(), TriangularError> {
        if rhs.len() != self.dimension() {
            return Err(TriangularError::DimensionMismatch);
        }
        match self.triangle {
            Triangle::Lower => {
                for column in self.first_non_identity_column..self.dimension() {
                    let value = rhs[column];
                    if value == 0.0 {
                        continue;
                    }
                    let coefficient = if self.unit_diagonal {
                        value
                    } else {
                        let coefficient = value / self.diagonal[column];
                        rhs[column] = coefficient;
                        coefficient
                    };
                    for (row, entry_coefficient) in self.column(column) {
                        rhs[row] = (-coefficient).mul_add(entry_coefficient, rhs[row]);
                    }
                }
            }
            Triangle::Upper => {
                for column in (self.first_non_identity_column..self.dimension()).rev() {
                    let value = rhs[column];
                    if value == 0.0 {
                        continue;
                    }
                    let pivot = if self.unit_diagonal {
                        value
                    } else {
                        let pivot = value / self.diagonal[column];
                        rhs[column] = pivot;
                        pivot
                    };
                    for position in (self.starts[column]..self.starts[column + 1]).rev() {
                        let row = self.rows[position];
                        rhs[row] = (-self.coefficients[position]).mul_add(pivot, rhs[row]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Solves a lower-triangular system when entries before `start` are zero.
    ///
    /// # Errors
    ///
    /// Returns an error for a dimension mismatch.
    pub fn lower_solve_starting_at(
        &self,
        start: usize,
        rhs: &mut [f64],
    ) -> Result<(), TriangularError> {
        if rhs.len() != self.dimension() {
            return Err(TriangularError::DimensionMismatch);
        }
        debug_assert_eq!(self.triangle, Triangle::Lower);
        for column in start.max(self.first_non_identity_column)..self.dimension() {
            let value = rhs[column];
            if value == 0.0 {
                continue;
            }
            let pivot = if self.unit_diagonal {
                value
            } else {
                let pivot = value / self.diagonal[column];
                rhs[column] = pivot;
                pivot
            };
            for (row, coefficient) in self.column(column) {
                rhs[row] = (-pivot).mul_add(coefficient, rhs[row]);
            }
        }
        Ok(())
    }

    /// Solves `T^T x = rhs` in place.
    ///
    /// # Errors
    ///
    /// Returns an error for a dimension mismatch.
    pub fn transpose_solve(&self, rhs: &mut [f64]) -> Result<(), TriangularError> {
        if rhs.len() != self.dimension() {
            return Err(TriangularError::DimensionMismatch);
        }
        match self.triangle {
            Triangle::Lower => {
                // GLOP skips the trailing exact-zero positions before starting
                // the backward substitution. Besides avoiding work, this is
                // observable for signed zero and non-unit diagonals.
                let Some(last_nonzero) = (self.first_non_identity_column..self.dimension())
                    .rev()
                    .find(|&column| rhs[column] != 0.0)
                else {
                    return Ok(());
                };
                for column in (self.first_non_identity_column..=last_nonzero).rev() {
                    let sum = self.transpose_column_sum_reverse(column, rhs);
                    rhs[column] = if self.unit_diagonal {
                        sum
                    } else {
                        sum / self.diagonal[column]
                    };
                }
            }
            Triangle::Upper => {
                for column in self.first_non_identity_column..self.dimension() {
                    let sum = self.transpose_column_sum_forward(column, rhs);
                    rhs[column] = if self.unit_diagonal {
                        sum
                    } else {
                        sum / self.diagonal[column]
                    };
                }
            }
        }
        Ok(())
    }

    /// Computes the structural solve closure and solves only those positions.
    /// An empty position list denotes the dense representation, as upstream.
    ///
    /// # Errors
    ///
    /// Returns an error for a dimension mismatch.
    pub fn solve_with_nonzeros<I: VectorIndex + Ord>(
        &self,
        rhs: &mut [f64],
        non_zeros: &mut Vec<I>,
    ) -> Result<(), TriangularError> {
        if rhs.len() != self.dimension() {
            return Err(TriangularError::DimensionMismatch);
        }
        self.compute_rows_to_consider_in_sorted_order(non_zeros);
        if non_zeros.is_empty() {
            return self.solve(rhs);
        }
        if self.triangle == Triangle::Lower {
            self.hyper_sparse_solve(rhs, non_zeros);
        } else {
            self.hyper_sparse_solve_with_reversed_nonzeros(rhs, non_zeros);
        }
        Ok(())
    }

    /// Computes a sparse transpose solve using GLOP's sorted symbolic closure.
    ///
    /// # Errors
    ///
    /// Returns an error for a dimension mismatch.
    pub fn transpose_solve_with_nonzeros<I: VectorIndex + Ord>(
        &self,
        rhs: &mut [f64],
        non_zeros: &mut Vec<I>,
    ) -> Result<(), TriangularError> {
        if rhs.len() != self.dimension() {
            return Err(TriangularError::DimensionMismatch);
        }
        self.compute_rows_to_consider_in_sorted_order(non_zeros);
        if non_zeros.is_empty() {
            return self.transpose_solve(rhs);
        }
        if self.triangle == Triangle::Upper {
            self.transpose_hyper_sparse_solve(rhs, non_zeros);
        } else {
            self.transpose_hyper_sparse_solve_with_reversed_nonzeros(rhs, non_zeros);
        }
        Ok(())
    }

    pub fn hyper_sparse_solve<I: VectorIndex>(&self, rhs: &mut [f64], non_zeros: &mut Vec<I>) {
        let mut write = 0;
        for read in 0..non_zeros.len() {
            let row = non_zeros[read].to_usize();
            let value = rhs[row];
            if value == 0.0 {
                continue;
            }
            let pivot = if self.unit_diagonal {
                value
            } else {
                value / self.diagonal[row]
            };
            rhs[row] = pivot;
            for (entry_row, coefficient) in self.column(row) {
                rhs[entry_row] = (-pivot).mul_add(coefficient, rhs[entry_row]);
            }
            non_zeros[write] = non_zeros[read];
            write += 1;
        }
        non_zeros.truncate(write);
    }

    pub fn hyper_sparse_solve_with_reversed_nonzeros<I: VectorIndex>(
        &self,
        rhs: &mut [f64],
        non_zeros: &mut Vec<I>,
    ) {
        let mut new_start = non_zeros.len();
        for read in (0..non_zeros.len()).rev() {
            let typed_row = non_zeros[read];
            let row = typed_row.to_usize();
            let value = rhs[row];
            if value == 0.0 {
                continue;
            }
            let pivot = if self.unit_diagonal {
                value
            } else {
                value / self.diagonal[row]
            };
            rhs[row] = pivot;
            for (entry_row, coefficient) in self.column(row) {
                rhs[entry_row] = (-pivot).mul_add(coefficient, rhs[entry_row]);
            }
            new_start -= 1;
            non_zeros[new_start] = typed_row;
        }
        non_zeros.drain(0..new_start);
    }

    pub fn transpose_hyper_sparse_solve<I: VectorIndex>(
        &self,
        rhs: &mut [f64],
        non_zeros: &mut Vec<I>,
    ) {
        let mut write = 0;
        for read in 0..non_zeros.len() {
            let row = non_zeros[read].to_usize();
            let sum = self.transpose_column_sum_forward(row, rhs);
            rhs[row] = if self.unit_diagonal {
                sum
            } else {
                sum / self.diagonal[row]
            };
            if sum != 0.0 {
                non_zeros[write] = non_zeros[read];
                write += 1;
            }
        }
        non_zeros.truncate(write);
    }

    pub fn transpose_hyper_sparse_solve_with_reversed_nonzeros<I: VectorIndex>(
        &self,
        rhs: &mut [f64],
        non_zeros: &mut Vec<I>,
    ) {
        let mut new_start = non_zeros.len();
        for read in (0..non_zeros.len()).rev() {
            let typed_row = non_zeros[read];
            let row = typed_row.to_usize();
            let sum = self.transpose_column_sum_reverse(row, rhs);
            rhs[row] = if self.unit_diagonal {
                sum
            } else {
                sum / self.diagonal[row]
            };
            if sum != 0.0 {
                new_start -= 1;
                non_zeros[new_start] = typed_row;
            }
        }
        non_zeros.drain(0..new_start);
    }

    fn transpose_column_sum_forward(&self, column: usize, rhs: &[f64]) -> f64 {
        let mut sum = rhs[column];
        let mut position = self.starts[column];
        let end = self.starts[column + 1];
        while position + 3 < end {
            // This is the contraction sequence emitted for GLOP's four-term
            // expression by the pinned optimized Apple Clang build: one
            // rounded multiply for i+1, then FMAs for i, i+2, and i+3.
            let mut four_term_sum = self.coefficients[position + 1] * rhs[self.rows[position + 1]];
            four_term_sum =
                self.coefficients[position].mul_add(rhs[self.rows[position]], four_term_sum);
            four_term_sum = self.coefficients[position + 2]
                .mul_add(rhs[self.rows[position + 2]], four_term_sum);
            four_term_sum = self.coefficients[position + 3]
                .mul_add(rhs[self.rows[position + 3]], four_term_sum);
            sum -= four_term_sum;
            position += 4;
        }
        while position < end {
            sum = (-self.coefficients[position]).mul_add(rhs[self.rows[position]], sum);
            position += 1;
        }
        sum
    }

    fn transpose_column_sum_reverse(&self, column: usize, rhs: &[f64]) -> f64 {
        let mut sum = rhs[column];
        let start = self.starts[column];
        let mut end = self.starts[column + 1];
        while end >= start + 4 {
            // This is the contraction sequence emitted for GLOP's four-term
            // expression by the pinned optimized Apple Clang build: one
            // rounded multiply for i-1, then FMAs for i, i-2, and i-3.
            let mut four_term_sum = self.coefficients[end - 2] * rhs[self.rows[end - 2]];
            four_term_sum =
                self.coefficients[end - 1].mul_add(rhs[self.rows[end - 1]], four_term_sum);
            four_term_sum =
                self.coefficients[end - 3].mul_add(rhs[self.rows[end - 3]], four_term_sum);
            four_term_sum =
                self.coefficients[end - 4].mul_add(rhs[self.rows[end - 4]], four_term_sum);
            sum -= four_term_sum;
            end -= 4;
        }
        while end > start {
            end -= 1;
            sum = (-self.coefficients[end]).mul_add(rhs[self.rows[end]], sum);
        }
        sum
    }

    /// Returns an explicitly stored transpose, as used by GLOP's LU solves.
    ///
    /// # Panics
    ///
    /// Panics only if this already-validated matrix violates its invariants.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let n = self.dimension();
        let mut columns = vec![Vec::new(); n];
        for column in 0..n {
            for (row, coefficient) in self.column(column) {
                columns[row].push((column, coefficient));
            }
        }
        Self::from_columns(
            &columns,
            self.diagonal.clone(),
            match self.triangle {
                Triangle::Lower => Triangle::Upper,
                Triangle::Upper => Triangle::Lower,
            },
            self.unit_diagonal,
        )
        .map(|mut transpose| {
            transpose.num_rows = self.num_rows;
            transpose
        })
        .expect("the transpose of a triangular matrix is triangular")
    }

    pub fn compute_rows_to_consider_with_dfs<I: VectorIndex>(&self, non_zeros: &mut Vec<I>) {
        if non_zeros.is_empty() {
            return;
        }
        let sparsity_threshold = self.num_rows / 40;
        let operations_threshold = self.num_rows / 20;
        let mut operations = non_zeros.len();
        if operations > sparsity_threshold {
            non_zeros.clear();
            return;
        }

        let mut workspace = self.symbolic_workspace.borrow_mut();
        workspace.stored.resize(self.num_rows, false);
        workspace.nodes_to_explore.clear();
        workspace.nodes_to_explore.extend(
            non_zeros
                .drain(..)
                .map(|row| SymbolicNode::Enter(row.to_usize())),
        );
        while let Some(&node) = workspace.nodes_to_explore.last() {
            let row = match node {
                SymbolicNode::Exit(row) => {
                    workspace.nodes_to_explore.pop();
                    workspace.stored[row] = true;
                    non_zeros.push(I::from_usize(row));
                    continue;
                }
                SymbolicNode::Enter(row) => row,
            };
            if workspace.stored[row] {
                workspace.nodes_to_explore.pop();
                continue;
            }
            workspace.nodes_to_explore.pop();
            workspace.nodes_to_explore.push(SymbolicNode::Exit(row));
            for (successor, _) in self.column(row) {
                operations += 1;
                if !workspace.stored[successor] {
                    workspace
                        .nodes_to_explore
                        .push(SymbolicNode::Enter(successor));
                }
            }
            if operations > operations_threshold {
                break;
            }
        }
        for row in non_zeros.iter().copied() {
            workspace.stored[row.to_usize()] = false;
        }
        if operations > operations_threshold {
            non_zeros.clear();
        }
    }

    pub fn compute_rows_to_consider_in_sorted_order<I: VectorIndex + Ord>(
        &self,
        non_zeros: &mut Vec<I>,
    ) {
        if non_zeros.is_empty() {
            return;
        }
        let sparsity_threshold = self.num_rows / 40;
        let operations_threshold = self.num_rows / 20;
        let mut operations = non_zeros.len();
        if operations > sparsity_threshold {
            non_zeros.clear();
            return;
        }
        let mut workspace = self.symbolic_workspace.borrow_mut();
        workspace.stored.resize(self.dimension(), false);
        workspace.touched.clear();
        for &typed_row in non_zeros.iter() {
            let row = typed_row.to_usize();
            if !workspace.stored[row] {
                workspace.stored[row] = true;
                workspace.touched.push(row);
            }
        }
        let mut position = 0;
        while position < non_zeros.len() {
            let column = non_zeros[position].to_usize();
            for (row, _) in self.column(column) {
                operations += 1;
                if !workspace.stored[row] {
                    workspace.stored[row] = true;
                    workspace.touched.push(row);
                    non_zeros.push(I::from_usize(row));
                }
            }
            if operations > operations_threshold {
                break;
            }
            position += 1;
        }
        while let Some(row) = workspace.touched.pop() {
            workspace.stored[row] = false;
        }
        if operations > operations_threshold {
            non_zeros.clear();
        } else {
            non_zeros.sort_unstable();
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::lp_types::{RowIndex, VectorIndex};

    use super::*;

    fn matrix(values: &[&[f64]]) -> SparseMatrix {
        let mut result = SparseMatrix::new();
        result.populate_from_zero(
            RowIndex::from_usize(values.len()),
            ColIndex::from_usize(values.len()),
        );
        for (row, entries) in values.iter().enumerate() {
            for (column, &value) in entries.iter().enumerate() {
                if value != 0.0 {
                    result
                        .mutable_column(ColIndex::from_usize(column))
                        .add_entry(RowIndex::from_usize(row), value);
                }
            }
        }
        result
    }

    #[test]
    fn lower_and_transpose_solves_recover_known_solution() {
        let values: &[&[f64]] = &[&[2.0, 0.0, 0.0], &[1.0, 3.0, 0.0], &[-1.0, 2.0, 4.0]];
        let triangular =
            TriangularMatrix::from_sparse(&matrix(values), Triangle::Lower, false).unwrap();
        let expected = [1.0, 2.0, -1.0];
        let mut rhs = vec![2.0, 7.0, -1.0];
        triangular.solve(&mut rhs).unwrap();
        assert!(
            rhs.iter()
                .zip(expected)
                .all(|(left, right)| (left - right).abs() < 1e-12)
        );
        let mut transpose_rhs = vec![5.0, 4.0, -4.0];
        triangular.transpose_solve(&mut transpose_rhs).unwrap();
        assert!(
            transpose_rhs
                .iter()
                .zip(expected)
                .all(|(left, right)| (left - right).abs() < 1e-12)
        );
    }

    #[test]
    fn hypersparse_solve_tracks_the_exact_structural_closure() {
        let n = 100;
        let mut columns = vec![Vec::new(); n];
        columns[2].push((90, 3.0));
        let lower =
            TriangularMatrix::from_columns(&columns, vec![1.0; n], Triangle::Lower, true).unwrap();
        let mut dense_rhs = vec![0.0; n];
        dense_rhs[2] = 4.0;
        let mut sparse_rhs = dense_rhs.clone();
        lower.solve(&mut dense_rhs).unwrap();
        let mut non_zeros = vec![RowIndex::new(2)];
        lower
            .solve_with_nonzeros(&mut sparse_rhs, &mut non_zeros)
            .unwrap();
        assert_eq!(dense_rhs, sparse_rhs);
        assert_eq!(non_zeros, vec![RowIndex::new(2), RowIndex::new(90)]);

        let upper = lower.transpose();
        let mut dense_rhs = vec![0.0; n];
        dense_rhs[90] = 4.0;
        let mut sparse_rhs = dense_rhs.clone();
        upper.solve(&mut dense_rhs).unwrap();
        let mut non_zeros = vec![RowIndex::new(90)];
        upper
            .solve_with_nonzeros(&mut sparse_rhs, &mut non_zeros)
            .unwrap();
        assert_eq!(dense_rhs, sparse_rhs);
        assert_eq!(non_zeros, vec![RowIndex::new(2), RowIndex::new(90)]);
    }

    #[test]
    fn incremental_build_copy_and_identity_prefix_match_separate_diagonal_storage() {
        let mut triangular = TriangularMatrix::empty(Triangle::Lower, true);
        triangular.reset(3, 3);
        triangular.add_diagonal_only_column(1.0);
        let mut column = SparseColumn::new();
        column.add_entry(RowIndex::new(1), 2.0);
        column.add_entry(RowIndex::new(2), 6.0);
        triangular.add_and_normalize_triangular_column(&column, RowIndex::new(1), 2.0);
        let mut off_diagonal = SparseColumn::new();
        off_diagonal.add_entry(RowIndex::new(1), -4.0);
        triangular.add_triangular_column_with_given_diagonal(&off_diagonal, RowIndex::new(2), 5.0);

        assert_eq!(triangular.num_rows(), 3);
        assert_eq!(triangular.num_cols(), 3);
        assert_eq!(triangular.num_entries(), 5);
        assert_eq!(triangular.first_non_identity_column(), 1);
        assert!(triangular.column_is_diagonal_only(0));
        assert!(!triangular.column_is_diagonal_only(1));

        let mut sparse = SparseMatrix::new();
        triangular.copy_to_sparse_matrix(&mut sparse);
        assert_eq!(
            sparse.look_up_value(RowIndex::new(0), ColIndex::new(0)),
            1.0
        );
        assert_eq!(
            sparse.look_up_value(RowIndex::new(2), ColIndex::new(1)),
            3.0
        );
        assert_eq!(
            sparse.look_up_value(RowIndex::new(1), ColIndex::new(2)),
            -4.0
        );
        assert_eq!(
            sparse.look_up_value(RowIndex::new(2), ColIndex::new(2)),
            5.0
        );
    }

    #[test]
    fn sparse_permuted_lower_solve_matches_dense_reference_after_pruning() {
        let columns = [
            vec![(0, 0.5), (1, -0.25), (4, 0.75)],
            vec![(1, 2.0), (4, -1.0)],
            vec![(3, 0.125)],
        ];
        let mut factor = TriangularMatrix::empty(Triangle::Lower, true);
        factor.reset(5, 3);
        for column in columns {
            let mut sparse_column = SparseColumn::new();
            for (row, value) in column {
                sparse_column.add_entry(RowIndex::from_usize(row), value);
            }
            factor.add_triangular_column_with_given_diagonal(&sparse_column, RowIndex::new(2), 1.0);
        }
        let row_permutation = [1, usize::MAX, 0, usize::MAX, 2];
        let inverse = [2, 0, 4];

        for entries in [vec![(2, 3.0), (1, -2.0)], vec![(2, -1.0), (0, 4.0)]] {
            let mut rhs = SparseColumn::new();
            for (row, value) in entries {
                rhs.add_entry(RowIndex::from_usize(row), value);
            }
            let mut expected_lower = SparseColumn::new();
            let mut expected_upper = SparseColumn::new();
            factor.permuted_lower_solve(
                &rhs,
                &row_permutation,
                &inverse,
                &mut expected_lower,
                &mut expected_upper,
            );
            let mut actual_lower = rhs;
            let mut actual_upper = SparseColumn::new();
            factor.permuted_lower_sparse_solve(
                &mut actual_lower,
                &row_permutation,
                &mut actual_upper,
            );
            for row in 0..5 {
                let row = RowIndex::from_usize(row);
                assert_eq!(
                    actual_lower.look_up_coefficient(row),
                    expected_lower.look_up_coefficient(row)
                );
                assert_eq!(
                    actual_upper.look_up_coefficient(row),
                    expected_upper.look_up_coefficient(row)
                );
            }
        }
    }
}
