//! LU factorization with threshold Markowitz pivoting.
//!
//! The numerical elimination follows upstream `lu_factorization` and
//! `markowitz`: permutations satisfy `A[row_permutation, column_permutation] =
//! L U`, `L` has an implicit unit diagonal, and pivot ties are deterministic.

use std::fmt;

use lp_data::lp_types::{ColIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

use crate::markowitz::choose_pivot;

#[derive(Clone, Debug, PartialEq)]
pub enum FactorizationError {
    NonSquare { rows: usize, columns: usize },
    NonFinite,
    Singular { step: usize },
    DimensionMismatch,
}

impl fmt::Display for FactorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquare { rows, columns } => {
                write!(formatter, "matrix is {rows} by {columns}, not square")
            }
            Self::NonFinite => formatter.write_str("matrix contains a nonfinite coefficient"),
            Self::Singular { step } => write!(formatter, "matrix is singular at step {step}"),
            Self::DimensionMismatch => formatter.write_str("right-hand side dimension mismatch"),
        }
    }
}

impl std::error::Error for FactorizationError {}

#[derive(Clone, Debug)]
pub struct LuFactorization {
    packed: Vec<Vec<f64>>,
    row_permutation: Vec<usize>,
    column_permutation: Vec<usize>,
    pivot_threshold: f64,
}

impl LuFactorization {
    /// Factorizes a square sparse matrix using threshold Markowitz pivots.
    ///
    /// # Errors
    ///
    /// Returns an error for nonsquare, nonfinite, or singular matrices.
    #[allow(clippy::needless_range_loop)]
    pub fn factorize(
        matrix: &SparseMatrix,
        pivot_threshold: f64,
    ) -> Result<Self, FactorizationError> {
        let rows = matrix.num_rows().to_usize();
        let columns = matrix.num_cols().to_usize();
        if rows != columns {
            return Err(FactorizationError::NonSquare { rows, columns });
        }
        let mut packed = vec![vec![0.0; columns]; rows];
        for column in 0..columns {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                if !entry.coefficient().is_finite() {
                    return Err(FactorizationError::NonFinite);
                }
                packed[entry.index().to_usize()][column] = entry.coefficient();
            }
        }
        let mut row_permutation: Vec<usize> = (0..rows).collect();
        let mut column_permutation: Vec<usize> = (0..columns).collect();
        for step in 0..rows {
            let pivot = choose_pivot(&packed, step, pivot_threshold)
                .ok_or(FactorizationError::Singular { step })?;
            packed.swap(step, pivot.row);
            row_permutation.swap(step, pivot.row);
            for row in &mut packed {
                row.swap(step, pivot.column);
            }
            column_permutation.swap(step, pivot.column);

            let diagonal = packed[step][step];
            if diagonal == 0.0 || !diagonal.is_finite() {
                return Err(FactorizationError::Singular { step });
            }
            for row in (step + 1)..rows {
                let multiplier = packed[row][step] / diagonal;
                packed[row][step] = multiplier;
                if multiplier == 0.0 {
                    continue;
                }
                for column in (step + 1)..columns {
                    packed[row][column] -= multiplier * packed[step][column];
                }
            }
        }
        Ok(Self {
            packed,
            row_permutation,
            column_permutation,
            pivot_threshold,
        })
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.packed.len()
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
    pub const fn pivot_threshold(&self) -> f64 {
        self.pivot_threshold
    }

    /// Solves `A x = rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error when the right-hand side has the wrong dimension.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        let n = self.dimension();
        if rhs.len() != n {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut work: Vec<f64> = self.row_permutation.iter().map(|&row| rhs[row]).collect();
        for row in 0..n {
            for column in 0..row {
                work[row] -= self.packed[row][column] * work[column];
            }
        }
        for row in (0..n).rev() {
            for column in (row + 1)..n {
                work[row] -= self.packed[row][column] * work[column];
            }
            work[row] /= self.packed[row][row];
        }
        let mut solution = vec![0.0; n];
        for (position, &column) in self.column_permutation.iter().enumerate() {
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
        let n = self.dimension();
        if rhs.len() != n {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut work: Vec<f64> = self
            .column_permutation
            .iter()
            .map(|&column| rhs[column])
            .collect();
        // U^T y = Q^T rhs.
        for row in 0..n {
            for column in 0..row {
                work[row] -= self.packed[column][row] * work[column];
            }
            work[row] /= self.packed[row][row];
        }
        // L^T z = y.
        for row in (0..n).rev() {
            for column in (row + 1)..n {
                work[row] -= self.packed[column][row] * work[column];
            }
        }
        let mut solution = vec![0.0; n];
        for (position, &row) in self.row_permutation.iter().enumerate() {
            solution[row] = work[position];
        }
        Ok(solution)
    }

    #[must_use]
    pub fn lower_and_upper(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.dimension();
        let mut lower = vec![vec![0.0; n]; n];
        let mut upper = vec![vec![0.0; n]; n];
        for row in 0..n {
            lower[row][row] = 1.0;
            for column in 0..n {
                if row > column {
                    lower[row][column] = self.packed[row][column];
                } else {
                    upper[row][column] = self.packed[row][column];
                }
            }
        }
        (lower, upper)
    }

    #[must_use]
    pub fn number_of_entries(&self) -> usize {
        let n = self.dimension();
        let lower = (0..n)
            .flat_map(|row| (0..row).map(move |column| (row, column)))
            .filter(|&(row, column)| self.packed[row][column] != 0.0)
            .count();
        let upper = (0..n)
            .flat_map(|row| (row..n).map(move |column| (row, column)))
            .filter(|&(row, column)| self.packed[row][column] != 0.0)
            .count();
        lower + upper
    }

    #[must_use]
    pub fn determinant(&self) -> f64 {
        let diagonal_product: f64 = (0..self.dimension())
            .map(|index| self.packed[index][index])
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
}
