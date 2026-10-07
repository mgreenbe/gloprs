//! Basis factorization and product-form updates.
//!
//! Like upstream `basis_representation`, this keeps a fresh LU plus eta updates
//! and can refactorize from the current basis when the update chain grows.

use lp_data::lp_types::{ColIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;
use lp_data::sparse_vector::SparseColumn;

use crate::lu_factorization::{FactorizationError, LuFactorization};
use crate::rank_one_update::RankOneUpdate;

#[derive(Clone, Debug)]
pub struct BasisRepresentation {
    basis: SparseMatrix,
    factorization: LuFactorization,
    updates: Vec<RankOneUpdate>,
    pivot_threshold: f64,
    max_updates: usize,
}

impl BasisRepresentation {
    /// Factorizes an initial square basis.
    ///
    /// # Errors
    ///
    /// Propagates matrix factorization errors.
    pub fn new(
        basis: SparseMatrix,
        pivot_threshold: f64,
        max_updates: usize,
    ) -> Result<Self, FactorizationError> {
        let factorization = LuFactorization::factorize(&basis, pivot_threshold)?;
        Ok(Self {
            basis,
            factorization,
            updates: Vec::new(),
            pivot_threshold,
            max_updates,
        })
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.basis.num_rows().to_usize()
    }

    #[must_use]
    pub fn basis(&self) -> &SparseMatrix {
        &self.basis
    }

    /// Solves `B x = rhs` through the LU and subsequent eta updates.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the factorization.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        let mut result = self.factorization.solve(rhs)?;
        for update in &self.updates {
            update.solve(&mut result);
        }
        Ok(result)
    }

    /// Solves `B^T x = rhs` through the eta updates and LU.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the factorization.
    pub fn transpose_solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if rhs.len() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut transformed = rhs.to_vec();
        for update in self.updates.iter().rev() {
            update.transpose_solve(&mut transformed);
        }
        self.factorization.transpose_solve(&transformed)
    }

    /// Replaces a basis column and records its product-form inverse update.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid pivot or failed refactorization.
    pub fn replace_column(
        &mut self,
        leaving_column: usize,
        mut entering_column: SparseColumn,
    ) -> Result<(), FactorizationError> {
        if leaving_column >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        entering_column.clean_up();
        let mut dense = vec![0.0; self.dimension()];
        for entry in &entering_column {
            dense[entry.index().to_usize()] = entry.coefficient();
        }
        let direction = self.solve(&dense)?;
        let update = RankOneUpdate::new(leaving_column, direction).map_err(|_| {
            FactorizationError::Singular {
                step: leaving_column,
            }
        })?;
        self.basis
            .replace_column(ColIndex::from_usize(leaving_column), entering_column);
        self.updates.push(update);
        if self.updates.len() >= self.max_updates {
            self.refactorize()?;
        }
        Ok(())
    }

    /// Rebuilds LU from the current basis and discards eta updates.
    ///
    /// # Errors
    ///
    /// Propagates factorization failures.
    pub fn refactorize(&mut self) -> Result<(), FactorizationError> {
        self.factorization = LuFactorization::factorize(&self.basis, self.pivot_threshold)?;
        self.updates.clear();
        Ok(())
    }

    /// Computes `||B||_1 ||B^-1||_1` exactly via basis solves.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn one_norm_condition_number(&self) -> Result<f64, FactorizationError> {
        let n = self.dimension();
        let mut inverse_column_sums = vec![0.0; n];
        for column in 0..n {
            let mut unit = vec![0.0; n];
            unit[column] = 1.0;
            let inverse_column = self.solve(&unit)?;
            inverse_column_sums[column] = inverse_column.iter().map(|value| value.abs()).sum();
        }
        Ok(self.basis.one_norm() * inverse_column_sums.into_iter().fold(0.0_f64, f64::max))
    }
}

#[cfg(test)]
mod tests {
    use lp_data::lp_types::RowIndex;

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
        result.clean_up();
        result
    }

    #[test]
    fn update_solve_agrees_with_fresh_refactorization() {
        let initial = matrix(&[&[2.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
        let mut updated = BasisRepresentation::new(initial, 0.1, 10).unwrap();
        let mut entering = SparseColumn::new();
        entering.add_entry(RowIndex::new(0), 1.0);
        entering.add_entry(RowIndex::new(1), -1.0);
        entering.add_entry(RowIndex::new(2), 3.0);
        updated.replace_column(1, entering).unwrap();

        let fresh = BasisRepresentation::new(updated.basis().clone(), 0.1, 10).unwrap();
        let rhs = [1.0, 2.0, -1.0];
        let left = updated.solve(&rhs).unwrap();
        let right = fresh.solve(&rhs).unwrap();
        assert!(
            left.iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() < 1e-12)
        );
        let left = updated.transpose_solve(&rhs).unwrap();
        let right = fresh.transpose_solve(&rhs).unwrap();
        assert!(
            left.iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() < 1e-12)
        );
    }
}
