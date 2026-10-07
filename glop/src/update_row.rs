//! Revised-simplex update-row computation.

use lp_data::lp_types::{ColIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;

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
        *value = matrix
            .column(ColIndex::from_usize(column))
            .into_iter()
            .map(|entry| left_inverse[entry.index().to_usize()] * entry.coefficient())
            .sum();
    }
    Ok(result)
}
