//! Dual steepest-edge norms.

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;

/// Returns the squared Euclidean norm of every row of `B^-1`.
///
/// # Errors
///
/// Propagates transpose-solve failures.
pub fn compute_dual_edge_squared_norms(
    basis: &BasisRepresentation,
) -> Result<Vec<f64>, FactorizationError> {
    let n = basis.dimension();
    let mut norms = vec![0.0; n];
    for row in 0..n {
        let mut unit = vec![0.0; n];
        unit[row] = 1.0;
        let inverse_row = basis.transpose_solve(&unit)?;
        norms[row] = inverse_row.iter().map(|value| value * value).sum();
    }
    Ok(norms)
}
