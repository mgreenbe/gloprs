//! Dual steepest-edge norm maintenance.
//!
//! This is a direct Rust translation of the material algorithms in
//! `ortools/glop/dual_edge_norms.{h,cc}`. Rust's borrowing rules make the
//! current basis an explicit method argument instead of a stored reference.

use lp_data::lp_types::{ColIndex, VectorIndex};
use lp_data::lp_utils::squared_norm;
use lp_data::permutation::ColumnPermutation;
use lp_data::scattered_vector::ScatteredRow;

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;
use crate::parameters::GlopParameters;
use crate::stats::{DistributionKind, StatsGroup};
use crate::time_limit::TimeLimit;

/// Incrementally maintained squared Euclidean norms of the rows of `B^-1`.
#[derive(Clone, Debug)]
pub struct DualEdgeNorms {
    edge_squared_norms: Vec<f64>,
    temporary_edge_squared_norms: Vec<f64>,
    recompute_edge_squared_norms: bool,
    recompute_edges_norm_threshold: f64,
    time_limit: Option<Rc<RefCell<TimeLimit>>>,
    stats: StatsGroup,
}

impl Default for DualEdgeNorms {
    fn default() -> Self {
        Self {
            edge_squared_norms: Vec::new(),
            temporary_edge_squared_norms: Vec::new(),
            recompute_edge_squared_norms: true,
            // GLOP's protobuf default.
            recompute_edges_norm_threshold: 100.0,
            time_limit: None,
            stats: StatsGroup::new("DualEdgeNorms"),
        }
    }
}

impl DualEdgeNorms {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.recompute_edge_squared_norms = true;
    }

    #[must_use]
    pub const fn needs_basis_refactorization(&self) -> bool {
        self.recompute_edge_squared_norms
    }

    pub fn set_recompute_edges_norm_threshold(&mut self, threshold: f64) {
        self.recompute_edges_norm_threshold = threshold;
    }

    pub fn set_glop_parameters(&mut self, parameters: &GlopParameters) {
        self.set_recompute_edges_norm_threshold(parameters.recompute_edges_norm_threshold);
    }

    pub fn set_time_limit(&mut self, time_limit: Option<Rc<RefCell<TimeLimit>>>) {
        self.time_limit = time_limit;
    }

    pub fn resize_on_new_rows(&mut self, new_size: usize) {
        self.edge_squared_norms.resize(new_size, 1.0);
    }

    /// Returns the norms, recomputing them from a refactorized basis if needed.
    ///
    /// # Errors
    ///
    /// Propagates a basis-solve failure.
    pub fn edge_squared_norms<'a>(
        &'a mut self,
        basis: &BasisRepresentation,
    ) -> Result<&'a [f64], FactorizationError> {
        if self.recompute_edge_squared_norms {
            self.compute_edge_squared_norms(basis)?;
        }
        Ok(&self.edge_squared_norms)
    }

    /// Applies GLOP's row-indexed-vector permutation after a basis permutation.
    pub fn update_data_on_basis_permutation(&mut self, permutation: &ColumnPermutation) {
        if self.recompute_edge_squared_norms {
            return;
        }
        let n = self.edge_squared_norms.len();
        self.temporary_edge_squared_norms.resize(n, 0.0);
        for old_index in 0..n {
            let new_index = permutation[ColIndex::from_usize(old_index)].to_usize();
            self.temporary_edge_squared_norms[new_index] = self.edge_squared_norms[old_index];
        }
        self.edge_squared_norms
            .copy_from_slice(&self.temporary_edge_squared_norms);
    }

    /// Checks the maintained norm against the precise leaving-row inverse.
    #[must_use]
    pub fn test_precision(&mut self, leaving_row: usize, unit_row_left_inverse: &[f64]) -> bool {
        if self.recompute_edge_squared_norms {
            return true;
        }
        let leaving_squared_norm = squared_norm(unit_row_left_inverse);
        let old_squared_norm = self.edge_squared_norms[leaving_row];
        let precise_norm = leaving_squared_norm.sqrt();
        let estimated_accuracy = (precise_norm - old_squared_norm.sqrt()) / precise_norm;
        self.stats.add(
            "edge_norms_accuracy",
            DistributionKind::Double,
            estimated_accuracy,
        );
        if estimated_accuracy.abs() > self.recompute_edges_norm_threshold {
            self.recompute_edge_squared_norms = true;
        }
        self.edge_squared_norms[leaving_row] = leaving_squared_norm;
        old_squared_norm > 0.25 * leaving_squared_norm
    }

    #[must_use]
    pub fn stat_string(&self) -> String {
        self.stats.stat_string()
    }

    /// Applies GLOP's incremental norm update immediately before a basis pivot.
    ///
    /// `direction` is `B^-1 a_q`; `unit_row_left_inverse` is `B^-T e_p`.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates the solve used to compute tau.
    pub fn update_before_basis_pivot(
        &mut self,
        basis: &BasisRepresentation,
        leaving_row: usize,
        direction: &[f64],
        unit_row_left_inverse: &[f64],
    ) -> Result<(), FactorizationError> {
        if self.recompute_edge_squared_norms {
            return Ok(());
        }
        let n = basis.dimension();
        if leaving_row >= n || direction.len() != n || unit_row_left_inverse.len() != n {
            return Err(FactorizationError::DimensionMismatch);
        }

        // tau = B^-1 B^-T e_p. RightSolveForTau() upstream keeps the exact
        // nonzero positions produced by the preceding unit-row left solve and
        // therefore takes the sparse/hyper-sparse solve path. Preserve that
        // numerical path instead of silently switching to the dense solve.
        let mut left_inverse = ScatteredRow::new(ColIndex::from_usize(n));
        for (row, &value) in unit_row_left_inverse.iter().enumerate() {
            if value != 0.0 {
                left_inverse.set(ColIndex::from_usize(row), value);
            }
        }
        let tau = basis.right_solve_for_tau(&left_inverse)?;
        let pivot = direction[leaving_row];
        let new_leaving_squared_norm = self.edge_squared_norms[leaving_row] / (pivot * pivot);
        let factor = 2.0 / pivot;
        for (row, &coefficient) in direction.iter().enumerate() {
            if coefficient == 0.0 {
                continue;
            }
            // Koberstein 8.2.2.1: this operation ordering is intentional.
            let inner = coefficient.mul_add(new_leaving_squared_norm, -(factor * tau[row]));
            self.edge_squared_norms[row] = coefficient.mul_add(inner, self.edge_squared_norms[row]);
            if row != leaving_row && self.edge_squared_norms[row] < 1e-4 {
                self.edge_squared_norms[row] = 1e-4;
            }
        }
        self.edge_squared_norms[leaving_row] = new_leaving_squared_norm;
        Ok(())
    }

    fn compute_edge_squared_norms(
        &mut self,
        basis: &BasisRepresentation,
    ) -> Result<(), FactorizationError> {
        let n = basis.dimension();
        self.edge_squared_norms.resize(n, 1.0);
        let test_limit = self.time_limit.is_some() && basis.number_of_entries_in_lu() > 10_000;
        for row in 0..n {
            let mut unit = vec![0.0; n];
            unit[row] = 1.0;
            self.edge_squared_norms[row] = squared_norm(&basis.transpose_solve(&unit)?);
            if test_limit
                && self
                    .time_limit
                    .as_ref()
                    .is_some_and(|limit| limit.borrow_mut().limit_reached())
            {
                break;
            }
        }
        self.recompute_edge_squared_norms = false;
        Ok(())
    }
}

/// Returns the squared Euclidean norm of every row of `B^-1`.
///
/// # Errors
///
/// Propagates transpose-solve failures.
pub fn compute_dual_edge_squared_norms(
    basis: &BasisRepresentation,
) -> Result<Vec<f64>, FactorizationError> {
    let mut norms = DualEdgeNorms::new();
    Ok(norms.edge_squared_norms(basis)?.to_vec())
}
use std::cell::RefCell;
use std::rc::Rc;
