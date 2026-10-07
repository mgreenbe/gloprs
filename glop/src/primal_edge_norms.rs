//! Primal steepest-edge norms.

use lp_data::lp_types::{ColIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;

/// Returns `1 + ||B^-1 a_j||_2^2` for every structural column.
///
/// # Errors
///
/// Returns a basis solve or dimension error.
pub fn compute_primal_edge_squared_norms(
    basis: &BasisRepresentation,
    matrix: &SparseMatrix,
) -> Result<Vec<f64>, FactorizationError> {
    if matrix.num_rows().to_usize() != basis.dimension() {
        return Err(FactorizationError::DimensionMismatch);
    }
    let mut norms = Vec::with_capacity(matrix.num_cols().to_usize());
    for column in 0..matrix.num_cols().to_usize() {
        let mut dense = vec![0.0; basis.dimension()];
        for entry in matrix.column(ColIndex::from_usize(column)) {
            dense[entry.index().to_usize()] = entry.coefficient();
        }
        let direction = basis.solve(&dense)?;
        norms.push(1.0 + direction.iter().map(|value| value * value).sum::<f64>());
    }
    Ok(norms)
}
