//! Numerical residual and error estimates for basis solves.

use lp_data::lp_types::{ColIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

#[must_use]
pub fn residual(matrix: &SparseMatrix, solution: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = rhs.iter().map(|value| -*value).collect();
    for (column, &value) in solution.iter().enumerate() {
        for entry in matrix.column(ColIndex::from_usize(column)) {
            let row = entry.index().to_usize();
            result[row] = entry.coefficient().mul_add(value, result[row]);
        }
    }
    result
}

#[must_use]
pub fn relative_residual(matrix: &SparseMatrix, solution: &[f64], rhs: &[f64]) -> f64 {
    let residual_norm = residual(matrix, solution, rhs)
        .into_iter()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let solution_norm = solution
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let rhs_norm = rhs.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    residual_norm / (matrix.infinity_norm() * solution_norm + rhs_norm).max(f64::MIN_POSITIVE)
}
