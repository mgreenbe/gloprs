//! Sparse triangular solves and transpose solves.
//!
//! This is the focused Phase-2 counterpart of the triangular portion of
//! upstream `ortools/lp_data/sparse.{h,cc}`.

use std::fmt;

use crate::lp_types::{ColIndex, VectorIndex};
use crate::sparse::SparseMatrix;

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
    columns: Vec<Vec<(usize, f64)>>,
    diagonal: Vec<f64>,
    triangle: Triangle,
    unit_diagonal: bool,
}

impl TriangularMatrix {
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
        let mut columns = vec![Vec::new(); n];
        let mut diagonal = vec![if unit_diagonal { 1.0 } else { 0.0 }; n];
        for column in 0..n {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                let row = entry.index().to_usize();
                if row == column {
                    if !unit_diagonal {
                        diagonal[column] = entry.coefficient();
                    }
                } else {
                    let valid = match triangle {
                        Triangle::Lower => row > column,
                        Triangle::Upper => row < column,
                    };
                    if !valid {
                        return Err(TriangularError::WrongTriangle);
                    }
                    columns[column].push((row, entry.coefficient()));
                }
            }
            if diagonal[column] == 0.0 || !diagonal[column].is_finite() {
                return Err(TriangularError::Singular { column });
            }
        }
        Ok(Self {
            columns,
            diagonal,
            triangle,
            unit_diagonal,
        })
    }

    /// Solves `T x = rhs` in place.
    ///
    /// # Errors
    ///
    /// Returns an error for a dimension mismatch.
    pub fn solve(&self, rhs: &mut [f64]) -> Result<(), TriangularError> {
        if rhs.len() != self.columns.len() {
            return Err(TriangularError::DimensionMismatch);
        }
        match self.triangle {
            Triangle::Lower => {
                for column in 0..self.columns.len() {
                    if !self.unit_diagonal {
                        rhs[column] /= self.diagonal[column];
                    }
                    let value = rhs[column];
                    for &(row, coefficient) in &self.columns[column] {
                        rhs[row] -= coefficient * value;
                    }
                }
            }
            Triangle::Upper => {
                for column in (0..self.columns.len()).rev() {
                    if !self.unit_diagonal {
                        rhs[column] /= self.diagonal[column];
                    }
                    let value = rhs[column];
                    for &(row, coefficient) in &self.columns[column] {
                        rhs[row] -= coefficient * value;
                    }
                }
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
        if rhs.len() != self.columns.len() {
            return Err(TriangularError::DimensionMismatch);
        }
        match self.triangle {
            Triangle::Lower => {
                for column in (0..self.columns.len()).rev() {
                    for &(row, coefficient) in &self.columns[column] {
                        rhs[column] -= coefficient * rhs[row];
                    }
                    if !self.unit_diagonal {
                        rhs[column] /= self.diagonal[column];
                    }
                }
            }
            Triangle::Upper => {
                for column in 0..self.columns.len() {
                    for &(row, coefficient) in &self.columns[column] {
                        rhs[column] -= coefficient * rhs[row];
                    }
                    if !self.unit_diagonal {
                        rhs[column] /= self.diagonal[column];
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
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
}
